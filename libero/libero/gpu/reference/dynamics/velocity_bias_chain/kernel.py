"""Compose accepted N01/N04/N05 kernels without host intermediate payloads."""
import libero.gpu.precision as scalar
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT

from dataclasses import dataclass
import importlib.util
from math import prod
from pathlib import Path
import sys

import warp as wp


def _accepted_kernel(name):
    # The repository's top-level libero initializer prompts for a dataset path.
    # Load the accepted modules directly without any parent-package side effect.
    path = GPU_ROOT / 'reference' / ('position' if name == 'compos' else 'dynamics') / name / 'kernel.py'
    module_name = f'd05_accepted_{name}_kernel'
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


compos = _accepted_kernel('compos')
comvel = _accepted_kernel('comvel')
rne_bias = _accepted_kernel('rne_bias')

wp.set_module_options({"enable_backward": False, "fuse_fp": True, "fast_math": False})


@dataclass(frozen=True)
class ChainModel:
    pos: compos.ComposModel
    vel: comvel.ComVelModel
    bias: rne_bias.RNEBiasModel

    @classmethod
    def prepare(cls, model, device="cuda"):
        return cls(compos.ComposModel.prepare(model, device),
                   comvel.ComVelModel.prepare(model, device),
                   rne_bias.RNEBiasModel.prepare(model, device))


class Workspace:
    """One topology and one fixed batch capacity; no allocation during launch."""

    def __init__(self, model: ChainModel, capacity: int, device="cuda"):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.model = model
        nb, nv = model.pos.nbody, model.pos.nv
        if nb != model.vel.nbody or nb != model.bias.nbody or nv != model.vel.nv or nv != model.bias.nv:
            raise ValueError("model dimension mismatch")
        def f(width):
            return wp.empty((capacity, width), dtype=scalar.wp_real, device=device)
        def i():
            return wp.empty(capacity, dtype=wp.int32, device=device)
        self.subtree_com = f(nb * 3)
        self.cinert = f(nb * 10)
        self.cdof = f(nv * 6)
        self.cvel = f(nb * 6)
        self.cdof_dot = f(nv * 6)
        self.cacc = f(nb * 6)
        self.frc = f(nb * 6)
        self.qfrc_bias = f(nv)
        self.status_pos, self.status_vel, self.status_bias = i(), i(), i()
        self.active_vel, self.active_bias, self.status = i(), i(), i()


@wp.kernel
def _gate(active: wp.array(dtype=wp.int32),
          upstream: wp.array(dtype=wp.int32),
          downstream: wp.array(dtype=wp.int32)):
    w = wp.tid()
    downstream[w] = wp.int32(1) if active[w] == 1 and upstream[w] == 0 else wp.int32(0)


@wp.kernel
def _finish(active: wp.array(dtype=wp.int32),
            pos: wp.array(dtype=wp.int32),
            vel: wp.array(dtype=wp.int32),
            bias: wp.array(dtype=wp.int32),
            result: wp.array(dtype=wp.int32)):
    w = wp.tid()
    if active[w] == 0:
        result[w] = 0
    elif pos[w] != 0:
        result[w] = 100 + pos[w]
    elif vel[w] != 0:
        result[w] = 200 + vel[w]
    elif bias[w] != 0:
        result[w] = 300 + bias[w]
    else:
        result[w] = 0


def launch(model: ChainModel, work: Workspace, active, xipos, ximat,
           xanchor, xmat, xaxis, qvel, stream=None):
    """Six true device kernels; caller owns stream and all input lifetimes."""
    n = work.capacity
    if work.model is not model:
        raise ValueError("workspace belongs to a different prepared model")
    nb, nj, nv = model.pos.nbody, model.pos.njnt, model.pos.nv
    schema = {'active': (active, (n,), wp.int32),
              'xipos': (xipos, (n, nb * 3), scalar.wp_real),
              'ximat': (ximat, (n, nb * 9), scalar.wp_real),
              'xanchor': (xanchor, (n, nj * 3), scalar.wp_real),
              'xmat': (xmat, (n, nb * 9), scalar.wp_real),
              'xaxis': (xaxis, (n, nj * 3), scalar.wp_real),
              'qvel': (qvel, (n, nv), scalar.wp_real),
              'subtree_com': (work.subtree_com, (n, nb * 3), scalar.wp_real),
              'cinert': (work.cinert, (n, nb * 10), scalar.wp_real),
              'cdof': (work.cdof, (n, nv * 6), scalar.wp_real),
              'cvel': (work.cvel, (n, nb * 6), scalar.wp_real),
              'cdof_dot': (work.cdof_dot, (n, nv * 6), scalar.wp_real),
              'cacc': (work.cacc, (n, nb * 6), scalar.wp_real),
              'frc': (work.frc, (n, nb * 6), scalar.wp_real),
              'qfrc_bias': (work.qfrc_bias, (n, nv), scalar.wp_real)}
    schema.update({name: (getattr(work, name), (n,), wp.int32)
                   for name in ('status_pos', 'status_vel', 'status_bias',
                                'active_vel', 'active_bias', 'status')})
    device = work.subtree_com.device
    intervals = []
    for name, (arr, shape, dtype) in schema.items():
        if (arr.shape != shape or arr.dtype != dtype or arr.device != device
                or not arr.device.is_cuda or not arr.is_contiguous or not int(arr.ptr)):
            raise ValueError(f"device array schema/layout mismatch: {name}")
        start = int(arr.ptr)
        intervals.append((start, start + prod(shape) * (scalar.itemsize if dtype == scalar.wp_real else 4), name))
    intervals.sort()
    for left, right in zip(intervals, intervals[1:]):
        if left[1] > right[0]:
            raise ValueError(f"overlapping device buffers: {left[2]} and {right[2]}")
    static_arrays = (model.pos.mass, model.pos.mass_subtree, model.pos.inertia,
                     model.pos.parent, model.pos.root, model.pos.jtype,
                     model.pos.jbody, model.pos.jdof, model.vel.body_parent,
                     model.vel.body_dofadr, model.vel.body_dofnum,
                     model.vel.dof_jntid, model.vel.jnt_type,
                     model.bias.body_parent, model.bias.body_dofadr,
                     model.bias.body_dofnum, model.bias.dof_body,
                     model.bias.gravity)
    for arr in static_arrays:
        if arr.device != device or not arr.is_contiguous or not int(arr.ptr):
            raise ValueError("static model device/layout mismatch")
        start = int(arr.ptr)
        end = start + arr.size * (scalar.itemsize if arr.dtype == scalar.wp_real else 4)
        if any(start < dynamic_end and end > dynamic_start
               for dynamic_start, dynamic_end, _ in intervals):
            raise ValueError("static model aliases dynamic buffer")
    if stream is not None and stream.device != device:
        raise ValueError("stream device mismatch")
    compos.launch(model.pos, active, xipos, ximat, xanchor, xmat, xaxis,
                  work.subtree_com, work.cinert, work.cdof, work.status_pos, stream)
    wp.launch(_gate, dim=n, inputs=[active, work.status_pos], outputs=[work.active_vel], stream=stream)
    comvel.launch(model.vel, work.active_vel, work.cdof, qvel,
                  work.cvel, work.cdof_dot, work.status_vel, stream)
    wp.launch(_gate, dim=n, inputs=[work.active_vel, work.status_vel],
              outputs=[work.active_bias], stream=stream)
    rne_bias.launch(model.bias, work.active_bias, work.cdof_dot, qvel,
                    work.cinert, work.cvel, work.cdof, work.cacc,
                    work.frc, work.qfrc_bias, work.status_bias, stream)
    wp.launch(_finish, dim=n,
              inputs=[active, work.status_pos, work.status_vel, work.status_bias],
              outputs=[work.status], stream=stream)
