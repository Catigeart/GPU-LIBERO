"""Nonzero-xfrc acceleration combinator: three-RN smooth -> seeded D02 -> N06.

Source order (MuJoCo 2.3.7, Apache-2.0):

* engine_forward.c:379-381 -- ``mju_sub``(passive, bias) then two ``mju_addTo``
  (applied, actuator): three distinct rounding nodes, no fast-math fusion.
* engine_support.c:1174-1221 -- per body in index order 1..nbody-1, if any of
  the six xfrc components is nonzero, ``mj_applyFT`` at xipos: the force part
  adds to qfrc first, then the torque part adds to the same qfrc.  Summing all
  external wrenches first and adding once is a different rounding path and is
  rejected by construction (the three-RN seed enters the accepted D02 kernel,
  which accumulates body by body).
* engine_forward.c:422-428 -- ``solveM`` on the accumulated qfrc: accepted N06
  factor/solve.

The accepted D09 kernel (smooth_acceleration_chain.kernel) is retained
unchanged and still rejects nonzero xfrc (its status 6).  This workspace is
the new combinator required for nonzero external forces; it does not modify
any accepted kernel.
"""
from __future__ import annotations
import libero.gpu.precision as scalar
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys

import numpy as np
import warp as wp

wp.set_module_options({'enable_backward': False, 'fuse_fp': True, 'fast_math': False})


def _accepted(name, path):
    module_name = f'p06d_accepted_{name}'
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_GPU_ROOT = (GPU_ROOT / '.')
_d02 = _accepted('d02_external_force', (GPU_ROOT / 'reference/dynamics/external_force_projection/kernel.py'))
_n06 = _accepted('n06_mass_solve', (GPU_ROOT / 'reference/position/mass_solve/kernel.py'))

ExternalForceWorkspace = _d02.ExternalForceWorkspace
StaticTopology = _d02.StaticTopology
MassSolveWorkspace = _n06.MassSolveWorkspace

# Entry status codes (first error wins within each accepted kernel).
STATUS_OK = 0
STATUS_ACTIVE = 1
STATUS_INPUT = 3
STATUS_RN = 4

# Composite status space written by _route:
#   0        success (unmodified arithmetic path)
#   1        invalid active mask entry
#   3        nonfinite force inputs
#   4        nonfinite three-RN arithmetic
#   10..17   D02 projection status (10 + d02 code)
#   20..25   N06 factor/solve status (20 + n06 code); 25 is pivot repair


@dataclass(frozen=True)
class RnNodes:
    """The three rounding nodes of the original force chain, for certificates."""
    sub: int = 1            # mju_sub: passive - bias
    add_applied: int = 1    # mju_addTo: + applied
    add_actuator: int = 1   # mju_addTo: + actuator


@wp.kernel(enable_backward=False)
def _three_rn(passive: wp.array(dtype=scalar.wp_real, ndim=2),
              bias: wp.array(dtype=scalar.wp_real, ndim=2),
              applied: wp.array(dtype=scalar.wp_real, ndim=2),
              actuator: wp.array(dtype=scalar.wp_real, ndim=2),
              active: wp.array(dtype=wp.int32),
              entry_status: wp.array(dtype=wp.int32),
              smooth: wp.array(dtype=scalar.wp_real, ndim=2),
              seed: wp.array(dtype=scalar.wp_real, ndim=2),
              nv: int):
    w = wp.tid()
    entry_status[w] = 0
    for i in range(nv):
        smooth[w, i] = scalar.wp_real(0.0)
        seed[w, i] = scalar.wp_real(0.0)
    if active[w] == 0:
        return
    if active[w] != 1:
        entry_status[w] = STATUS_ACTIVE
        return
    for i in range(nv):
        if (not wp.isfinite(passive[w, i]) or not wp.isfinite(bias[w, i])
                or not wp.isfinite(applied[w, i]) or not wp.isfinite(actuator[w, i])):
            entry_status[w] = STATUS_INPUT
            return
    for i in range(nv):
        # mju_sub then two mju_addTo: three separate binary64 roundings.
        t0 = passive[w, i] - bias[w, i]
        t1 = t0 + applied[w, i]
        t2 = t1 + actuator[w, i]
        if not wp.isfinite(t2):
            entry_status[w] = STATUS_RN
            return
        smooth[w, i] = t2
        seed[w, i] = t2


@wp.kernel(enable_backward=False)
def _seed_rhs(qfrc: wp.array(dtype=scalar.wp_real, ndim=2),
              rhs: wp.array(dtype=scalar.wp_real, ndim=3),
              active: wp.array(dtype=wp.int32),
              entry_status: wp.array(dtype=wp.int32),
              d02_status: wp.array(dtype=wp.int32),
              nv: int):
    """Stage the accumulated force as RHS row 0; rows 1-3 stay zero."""
    w = wp.tid()
    for r in range(4):
        for i in range(nv):
            rhs[w, r, i] = scalar.wp_real(0.0)
    if active[w] != 1 or entry_status[w] != 0 or d02_status[w] != 0:
        return
    for i in range(nv):
        rhs[w, 0, i] = qfrc[w, i]


@wp.kernel(enable_backward=False)
def _route(entry_status: wp.array(dtype=wp.int32),
           d02_status: wp.array(dtype=wp.int32),
           solve_status: wp.array(dtype=wp.int32),
           active: wp.array(dtype=wp.int32),
           solution: wp.array(dtype=scalar.wp_real, ndim=3),
           smooth_force: wp.array(dtype=scalar.wp_real, ndim=2),
           smooth_rn: wp.array(dtype=scalar.wp_real, ndim=2),
           qacc_smooth: wp.array(dtype=scalar.wp_real, ndim=2),
           status: wp.array(dtype=wp.int32),
           nv: int):
    """Publish only trustworthy fields: every failed world's declared output
    (including the three-RN smooth) is explicitly zeroed (R02-F07)."""
    w = wp.tid()
    if active[w] == 0:
        status[w] = 0
        for i in range(nv):
            smooth_force[w, i] = scalar.wp_real(0.0)
            smooth_rn[w, i] = scalar.wp_real(0.0)
            qacc_smooth[w, i] = scalar.wp_real(0.0)
        return
    s = 0
    if entry_status[w] != 0:
        s = entry_status[w]
    elif d02_status[w] != 0:
        s = 10 + d02_status[w]
    elif solve_status[w] == 5:
        s = 25
    elif solve_status[w] != 0:
        s = 20 + solve_status[w]
    status[w] = s
    ok = (s == 0 or s == 25)
    for i in range(nv):
        if ok:
            qacc_smooth[w, i] = solution[w, 0, i]
        else:
            qacc_smooth[w, i] = scalar.wp_real(0.0)
            smooth_force[w, i] = scalar.wp_real(0.0)
            smooth_rn[w, i] = scalar.wp_real(0.0)


class SmoothXfrcSolveWorkspace:
    """Resident nonzero-xfrc acceleration chain for N worlds of one model.

    Prepare binds the accepted D02 topology and N06 model; launch enqueues
    only device kernels (three-RN, D02 project, N06 factor/solve, status
    route).  Host upload/download are explicit boundaries outside launch.

    status: 0 success; 1 invalid active; 3 nonfinite input forces; 4 nonfinite
    three-RN arithmetic; 10+d02 for projection failures (11 active, 12 count,
    13 body, 14 seed, 15 xfrc, 16 kinematic, 17 arithmetic); 20+n06 for
    factor/solve failures (21 active, 22 count, 23 nonfinite, 24 arithmetic);
    25 completed after native pivot repair.  Any failure zeroes force and
    acceleration outputs for that world; the first failing kernel's code stays
    visible in entry_status/d02_status/solve_status.
    """

    def __init__(self, topology: StaticTopology, model, capacity: int,
                 device='cuda:0'):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('positive int capacity required')
        if not isinstance(topology, StaticTopology):
            raise TypeError('accepted D02 StaticTopology required')
        self.capacity = capacity
        self.device = device
        self.nv = int(topology.nv)
        self.nbody = int(topology.nbody)
        self.d02 = ExternalForceWorkspace(topology, capacity, device=device)
        self.solver = MassSolveWorkspace(model, capacity, device=device)
        # solveM semantics: exactly one RHS per world (engine_forward.c:422-428).
        self.solver.count.fill_(1)
        # R02-F07: deterministic start state for the accepted D02 output
        # (its own allocation is wp.empty; early-return worlds would otherwise
        # publish uninitialized bytes before _route zeroes them).
        self.d02.qfrc.fill_(0.0)
        self.d02.scratch.fill_(0.0)
        F, I = scalar.wp_real, wp.int32
        self.passive = wp.zeros((capacity, self.nv), dtype=F, device=device)
        self.bias = wp.zeros((capacity, self.nv), dtype=F, device=device)
        self.applied = wp.zeros((capacity, self.nv), dtype=F, device=device)
        self.actuator = wp.zeros((capacity, self.nv), dtype=F, device=device)
        self.active = wp.zeros(capacity, dtype=I, device=device)
        self.entry_status = wp.zeros(capacity, dtype=I, device=device)
        self.smooth = wp.zeros((capacity, self.nv), dtype=F, device=device)
        self.status = wp.zeros(capacity, dtype=I, device=device)
        # The D02 seed buffer is the three-RN output; no production copy.
        self.seed = self.d02.seed_qfrc
        # Accumulated smooth force lives in the accepted D02 output.
        self.qfrc = self.d02.qfrc
        self.qacc_smooth = wp.zeros((capacity, self.nv), dtype=F, device=device)
        self.xfrc = self.d02.xfrc_applied
        self.xipos = self.d02.xipos
        self.subtree_com = self.d02.subtree_com
        self.cdof = self.d02.cdof
        self.qm = self.solver.qm

    def upload(self, **arrays):
        """Host staging boundary; never called by launch or Graph replay.

        `active` upload also arms the fixed D02 topology arrays (body ids
        1..nbody-1 in source order, count = nbody-1) and the N06 active mask.
        """
        allowed = {'passive', 'bias', 'applied', 'actuator', 'active', 'xfrc',
                   'xipos', 'subtree_com', 'cdof', 'qm'}
        for name, value in arrays.items():
            if name not in allowed:
                raise KeyError(name)
            raw = np.asarray(value)
            if name == 'active':
                expect, dtype = (self.capacity,), np.int32
            elif name == 'qm':
                expect, dtype = (self.capacity, self.solver.nM), np.float64
            elif name == 'xfrc':
                expect, dtype = (self.capacity, self.nbody, 6), np.float64
            elif name in ('xipos', 'subtree_com'):
                expect, dtype = (self.capacity, self.nbody, 3), np.float64
            elif name == 'cdof':
                expect, dtype = (self.capacity, self.nv, 6), np.float64
            else:
                expect, dtype = (self.capacity, self.nv), np.float64
            if raw.shape != expect or raw.dtype != dtype:
                raise TypeError(f'{name}: expected {expect} {dtype}, '
                                f'got {raw.shape} {raw.dtype}')
            payload = np.ascontiguousarray(raw)
            if name in ('passive', 'bias', 'applied', 'actuator'):
                getattr(self, name).assign(payload)
            elif name == 'qm':
                self.solver.qm.assign(payload)
            elif name == 'xfrc':
                self.d02.xfrc_applied.assign(payload)
            elif name == 'xipos':
                self.d02.xipos.assign(payload)
            elif name == 'subtree_com':
                self.d02.subtree_com.assign(payload)
            elif name == 'cdof':
                self.d02.cdof.assign(payload)
            else:  # active
                self.active.assign(payload)
                self.d02.active_world.assign(payload)
                self.solver.active.assign(payload)
                ids = np.tile(np.arange(1, self.nbody, dtype=np.int32),
                              (self.capacity, 1))
                self.d02.body_ids.assign(ids)
                self.d02.body_count.assign(
                    np.full(self.capacity, self.nbody - 1, dtype=np.int32))

    def launch(self, stream=None):
        """Five resident kernels; no host read, copy decision, or solve."""
        wp.launch(_three_rn, dim=self.capacity,
                  inputs=[self.passive, self.bias, self.applied, self.actuator,
                          self.active, self.entry_status, self.smooth,
                          self.seed, self.nv],
                  device=self.device, stream=stream)
        self.d02.launch(stream=stream)
        wp.launch(_seed_rhs, dim=self.capacity,
                  inputs=[self.d02.qfrc, self.solver.rhs, self.active,
                          self.entry_status, self.d02.status, self.nv],
                  device=self.device, stream=stream)
        self.solver.launch(stream=stream)
        wp.launch(_route, dim=self.capacity,
                  inputs=[self.entry_status, self.d02.status, self.solver.status,
                          self.active, self.solver.solution, self.qfrc,
                          self.smooth, self.qacc_smooth, self.status, self.nv],
                  device=self.device, stream=stream)

    def download(self):
        return {
            'smooth': np.array(self.smooth.numpy(), copy=True),
            'qfrc_smooth': np.array(self.qfrc.numpy(), copy=True),
            'qacc_smooth': np.array(self.qacc_smooth.numpy(), copy=True),
            'status': np.array(self.status.numpy(), copy=True),
            'entry_status': np.array(self.entry_status.numpy(), copy=True),
            'd02_status': np.array(self.d02.status.numpy(), copy=True),
            'solve_status': np.array(self.solver.status.numpy(), copy=True),
            'warning_count': np.array(self.solver.warning_count.numpy(), copy=True),
            'warning_last': np.array(self.solver.warning_last.numpy(), copy=True),
        }
