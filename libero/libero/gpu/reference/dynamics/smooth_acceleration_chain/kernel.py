"""D09 CPU-boundary smooth force to acceleration, with accepted N06 factor/solve.

Source order: MuJoCo 2.3.7 engine_forward.c:364-379. The native factor and
single-RHS solve are the accepted mass_solve._factor_solve GPU transcription.
"""
import libero.gpu.precision as scalar
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT

import importlib.util
import numpy as np
from pathlib import Path
import sys
import warp as wp

_n06_path = (GPU_ROOT / 'reference/position/mass_solve/kernel.py')
_n06_spec = importlib.util.spec_from_file_location('d09_accepted_n06_kernel',_n06_path)
_n06_mod = importlib.util.module_from_spec(_n06_spec)
sys.modules[_n06_spec.name] = _n06_mod
_n06_spec.loader.exec_module(_n06_mod)
MassSolveWorkspace = _n06_mod.MassSolveWorkspace
_factor_solve = _n06_mod._factor_solve

wp.set_module_options({'enable_backward': False, 'fuse_fp': True, 'fast_math': False})


@wp.kernel(enable_backward=False)
def _make_force(passive: wp.array(dtype=scalar.wp_real, ndim=2),
                bias: wp.array(dtype=scalar.wp_real, ndim=2),
                applied: wp.array(dtype=scalar.wp_real, ndim=2),
                actuator: wp.array(dtype=scalar.wp_real, ndim=2),
                xfrc: wp.array(dtype=scalar.wp_real, ndim=3),
                active: wp.array(dtype=wp.int32),
                work_active: wp.array(dtype=wp.int32),
                entry_status: wp.array(dtype=wp.int32),
                smooth: wp.array(dtype=scalar.wp_real, ndim=2),
                rhs: wp.array(dtype=scalar.wp_real, ndim=3),
                nv: int, nbody: int):
    w = wp.tid()
    entry_status[w] = 0
    work_active[w] = 0
    for i in range(nv):
        smooth[w, i] = scalar.wp_real(0.0)
        for r in range(4):
            rhs[w, r, i] = scalar.wp_real(0.0)
    if active[w] == 0:
        return
    if active[w] != 1:
        entry_status[w] = 1
        return
    for b in range(nbody):
        for k in range(6):
            if xfrc[w, b, k] != scalar.wp_real(0.0):
                entry_status[w] = 6
                return
    for i in range(nv):
        if not wp.isfinite(passive[w, i]) or not wp.isfinite(bias[w, i]) or not wp.isfinite(applied[w, i]) or not wp.isfinite(actuator[w, i]):
            entry_status[w] = 3
            return
    for i in range(nv):
        # Three distinct rounding nodes reproduce mju_sub, then two mju_addTo.
        t0 = passive[w, i] - bias[w, i]
        t1 = t0 + applied[w, i]
        t2 = t1 + actuator[w, i]
        if not wp.isfinite(t2):
            entry_status[w] = 4
            return
        smooth[w, i] = t2
        rhs[w, 0, i] = t2
    work_active[w] = 1


@wp.kernel(enable_backward=False)
def _finish(entry_status: wp.array(dtype=wp.int32),
            factor_status: wp.array(dtype=wp.int32),
            solution: wp.array(dtype=scalar.wp_real, ndim=3),
            smooth: wp.array(dtype=scalar.wp_real, ndim=2),
            qacc: wp.array(dtype=scalar.wp_real, ndim=2),
            status: wp.array(dtype=wp.int32), nv: int):
    w = wp.tid()
    s = factor_status[w]
    if entry_status[w] != 0:
        s = entry_status[w]
    status[w] = s
    for i in range(nv):
        if s == 0 or s == 5:
            qacc[w, i] = solution[w, 0, i]
        else:
            qacc[w, i] = scalar.wp_real(0.0)
        if s != 0 and s != 5:
            smooth[w, i] = scalar.wp_real(0.0)


class SmoothAccelerationWorkspace:
    """Prepared fixed-topology CUDA workspace. Inputs are device arrays.

    Host preparation validates the accepted N06 model topology and capacity.
    status: 0 normal, 5 pivot repair; 1 invalid active, 3 nonfinite input,
    4 nonfinite arithmetic, 6 unsupported nonzero xfrc. All failures have
    zero force and acceleration outputs. Factor details follow N06 semantics.
    """

    def __init__(self, model, capacity, device='cuda:0'):
        self.mass = MassSolveWorkspace(model, capacity, device=device)
        self.nv, self.nM, self.nbody = int(model.nv), int(model.nM), int(model.nbody)
        self.capacity, self.device = capacity, device
        self.passive = wp.zeros((capacity,self.nv), dtype=scalar.wp_real, device=device)
        self.bias = wp.zeros((capacity,self.nv), dtype=scalar.wp_real, device=device)
        self.applied = wp.zeros((capacity,self.nv), dtype=scalar.wp_real, device=device)
        self.actuator = wp.zeros((capacity,self.nv), dtype=scalar.wp_real, device=device)
        self.xfrc = wp.zeros((capacity,self.nbody,6), dtype=scalar.wp_real, device=device)
        self.active = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.work_active = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.entry_status = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.smooth = wp.zeros((capacity,self.nv), dtype=scalar.wp_real, device=device)
        self.qacc = wp.zeros((capacity,self.nv), dtype=scalar.wp_real, device=device)
        self.acc = self.qacc
        self.status = wp.zeros(capacity, dtype=wp.int32, device=device)
        # Aliases established in prepare; no production data copies.
        self.qm = self.mass.qm
        self.rhs = self.mass.rhs
        self.count = self.mass.count
        self.ld = self.mass.ld
        self.inv = self.mass.inv
        self.sqrt_inv = self.mass.sqrt_inv
        self.pivot = self.mass.pivot
        self.warning_count = self.mass.warning_count
        self.warning_last = self.mass.warning_last

    def launch(self, stream=None):
        wp.launch(_make_force, dim=self.capacity,
                  inputs=[self.passive,self.bias,self.applied,self.actuator,
                          self.xfrc,self.active,self.work_active,self.entry_status,
                          self.smooth,self.rhs,self.nv,self.nbody],
                  device=self.device,stream=stream)
        wp.launch(_factor_solve, dim=self.capacity,
                  inputs=[self.qm,self.rhs,self.count,self.work_active,
                          self.mass.dof_Madr,self.mass.dof_parentid,
                          self.ld,self.inv,self.sqrt_inv,self.mass.solution,
                          self.pivot,self.mass.status,self.warning_count,
                          self.warning_last,self.nv,self.nM],
                  device=self.device,stream=stream)
        wp.launch(_finish, dim=self.capacity,
                  inputs=[self.entry_status,self.mass.status,self.mass.solution,
                          self.smooth,self.qacc,self.status,self.nv],
                  device=self.device,stream=stream)


def validate_host_payload(payload, capacity, nv=43, nm=153, nbody=38):
    """Schema check before any cast or upload; active slots are still device checked."""
    shape = {'passive':(capacity,nv),'bias':(capacity,nv),
             'applied':(capacity,nv),'actuator':(capacity,nv),
             'qm':(capacity,nm),'xfrc':(capacity,nbody,6),
             'active':(capacity,)}
    for name, required in shape.items():
        a = np.asarray(payload[name])
        dtype = np.dtype(np.int32 if name == 'active' else np.float64)
        if a.shape != required or a.dtype != dtype:
            raise TypeError(f'{name}: expected {required} {dtype}, got {a.shape} {a.dtype}')
    return True
