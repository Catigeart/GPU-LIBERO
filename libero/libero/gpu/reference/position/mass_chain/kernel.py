"""Fixed-model, fixed-capacity mass chain. All intermediate edges are shared buffers.

The only boundary inputs are Cartesian arrays, active, count, and RHS. No
numeric intermediate is uploaded or read on the launch path.
"""
import libero.gpu.precision as scalar

import numpy as np
import warp as wp

from libero.gpu.reference.position.compos.kernel import ComposModel, launch as compos_launch
from libero.gpu.reference.position.crb.kernel import CRBModel, launch as crb_launch
from libero.gpu.reference.position.mass_solve.kernel import MassSolveWorkspace
from .static_signature import digest as static_digest

wp.set_module_options({"enable_backward": False, "fuse_fp": True, "fast_math": False})

_MODEL_SIGNATURES = {
    (38, 48, 43, 153, 0): "70ade775ea86948757b2dce8d67231777c7c32aba49d0f99f8bc4489b674038c",
    (6, 13, 11, 36, 1): "48d709614dd89b54ab47bd62a90ab6d690a578b977919220382586892b472fd7",
}


@wp.kernel
def _gate1(input_active: wp.array(dtype=wp.int32),
           compos_status: wp.array(dtype=wp.int32),
           crb_active: wp.array(dtype=wp.int32),
           first_stage: wp.array(dtype=wp.int32),
           first_code: wp.array(dtype=wp.int32)):
    w = wp.tid()
    crb_active[w] = 0
    first_stage[w] = 0
    first_code[w] = 0
    if input_active[w] == 0:
        return
    if compos_status[w] != 0:
        first_stage[w] = 1
        first_code[w] = compos_status[w]
    else:
        crb_active[w] = 1


@wp.kernel
def _gate2(crb_active: wp.array(dtype=wp.int32),
           crb_status: wp.array(dtype=wp.int32),
           solve_active: wp.array(dtype=wp.int32),
           first_stage: wp.array(dtype=wp.int32),
           first_code: wp.array(dtype=wp.int32)):
    w = wp.tid()
    solve_active[w] = 0
    if crb_active[w] == 0:
        return
    if crb_status[w] != 0:
        first_stage[w] = 2
        first_code[w] = crb_status[w]
    else:
        solve_active[w] = 1


@wp.kernel
def _finish(input_active: wp.array(dtype=wp.int32),
            solve_active: wp.array(dtype=wp.int32),
            solve_status: wp.array(dtype=wp.int32),
            first_stage: wp.array(dtype=wp.int32),
            first_code: wp.array(dtype=wp.int32),
            result_kind: wp.array(dtype=wp.int32)):
    w = wp.tid()
    # 0 inactive, 1 unmodified, 2 repaired, 3 upstream or local error.
    result_kind[w] = 0
    if input_active[w] == 0:
        return
    if solve_active[w] == 0:
        result_kind[w] = 3
        return
    code = solve_status[w]
    if code == 0:
        result_kind[w] = 1
    elif code == 5:
        result_kind[w] = 2
    else:
        first_stage[w] = 3
        first_code[w] = code
        result_kind[w] = 3


def _check_array(name, array, shape, dtype, device):
    if not isinstance(array, wp.array) or array.shape != shape or array.dtype != dtype:
        raise ValueError(f"{name}: device schema mismatch {shape}/{dtype}")
    if not array.device.is_cuda or array.device != wp.get_device(device):
        raise ValueError(f"{name}: device mismatch")
    itemsize = scalar.itemsize if dtype == scalar.wp_real else 4
    expected = []
    step = itemsize
    for size in reversed(shape):
        expected.insert(0, step)
        step *= size
    if tuple(array.strides) != tuple(expected):
        raise ValueError(f"{name}: noncontiguous array")
    if int(array.ptr) == 0:
        raise ValueError(f"{name}: null pointer")
    return int(array.ptr), int(array.ptr) + step


class MassChainWorkspace:
    """One static MJB per workspace; boundary buffers are preallocated.

    `mass.qm` is the CRB output itself. `cinert` and `cdof` are simultaneously
    compos outputs and CRB inputs. Gate kernels route world masks on device.
    """

    def __init__(self, model, capacity, device="cuda:0"):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive int")
        dims = (int(model.nbody), int(model.nq), int(model.nv),
                int(model.nM), int(model.nmocap))
        if dims not in _MODEL_SIGNATURES:
            raise ValueError("unsupported model dimensions")
        self.model_signature = static_digest(model)
        if self.model_signature != _MODEL_SIGNATURES[dims]:
            raise ValueError("unsupported model static signature")
        self.capacity, self.device, self.dims = capacity, device, dims
        self.compos = ComposModel.prepare(model, device)
        self.crb_model = CRBModel.prepare(model, device)
        self.mass = MassSolveWorkspace(model, capacity, device)
        nb, _, nv, _, _ = dims
        nj = int(model.njnt)
        self.input_active = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.xipos = wp.zeros((capacity, nb * 3), dtype=scalar.wp_real, device=device)
        self.ximat = wp.zeros((capacity, nb * 9), dtype=scalar.wp_real, device=device)
        self.xanchor = wp.zeros((capacity, nj * 3), dtype=scalar.wp_real, device=device)
        self.xmat = wp.zeros((capacity, nb * 9), dtype=scalar.wp_real, device=device)
        self.xaxis = wp.zeros((capacity, nj * 3), dtype=scalar.wp_real, device=device)
        self.subtree_com = wp.zeros((capacity, nb * 3), dtype=scalar.wp_real, device=device)
        self.cinert = wp.zeros((capacity, nb * 10), dtype=scalar.wp_real, device=device)
        self.cdof = wp.zeros((capacity, nv * 6), dtype=scalar.wp_real, device=device)
        self.compos_status = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.crb_active = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.crb = wp.zeros((capacity, nb * 10), dtype=scalar.wp_real, device=device)
        self.crb_status = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.first_stage = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.first_code = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.result_kind = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.validate_layout()

    def validate_layout(self):
        n = self.capacity
        nb, _, nv, nm, _ = self.dims
        nj = self.compos.njnt
        schema = {
            "input_active": ((n,), wp.int32),
            "xipos": ((n, nb * 3), scalar.wp_real),
            "ximat": ((n, nb * 9), scalar.wp_real),
            "xanchor": ((n, nj * 3), scalar.wp_real),
            "xmat": ((n, nb * 9), scalar.wp_real),
            "xaxis": ((n, nj * 3), scalar.wp_real),
            "subtree_com": ((n, nb * 3), scalar.wp_real),
            "cinert": ((n, nb * 10), scalar.wp_real),
            "cdof": ((n, nv * 6), scalar.wp_real),
            "compos_status": ((n,), wp.int32),
            "crb_active": ((n,), wp.int32),
            "crb": ((n, nb * 10), scalar.wp_real),
            "crb_status": ((n,), wp.int32),
            "first_stage": ((n,), wp.int32),
            "first_code": ((n,), wp.int32),
            "result_kind": ((n,), wp.int32),
            "mass.qm": ((n, nm), scalar.wp_real),
            "mass.rhs": ((n, 4, nv), scalar.wp_real),
            "mass.count": ((n,), wp.int32),
            "mass.active": ((n,), wp.int32),
            "mass.ld": ((n, nm), scalar.wp_real),
            "mass.inv": ((n, nv), scalar.wp_real),
            "mass.sqrt_inv": ((n, nv), scalar.wp_real),
            "mass.solution": ((n, 4, nv), scalar.wp_real),
            "mass.pivot": ((n, nv), wp.int32),
            "mass.status": ((n,), wp.int32),
            "mass.warning_count": ((n,), wp.int32),
            "mass.warning_last": ((n,), wp.int32),
        }
        ranges = {}
        for name, (shape, dtype) in schema.items():
            owner = self.mass if name.startswith("mass.") else self
            attr = name.split(".")[-1]
            ranges[name] = _check_array(name, getattr(owner, attr), shape, dtype, self.device)
        ordered = list(ranges)
        for i, a in enumerate(ordered):
            lo, hi = ranges[a]
            for b in ordered[i + 1:]:
                other_lo, other_hi = ranges[b]
                if lo < other_hi and other_lo < hi:
                    raise ValueError(f"illegal storage overlap: {a}/{b}")
        return {k: {"ptr": v[0], "bytes": v[1] - v[0]} for k, v in ranges.items()}

    def launch(self, stream=None):
        # Fixed buffers, one stream, six launches, no host reads or transfers.
        compos_launch(self.compos, self.input_active, self.xipos, self.ximat,
                      self.xanchor, self.xmat, self.xaxis, self.subtree_com,
                      self.cinert, self.cdof, self.compos_status, stream=stream)
        wp.launch(_gate1, dim=self.capacity,
                  inputs=[self.input_active, self.compos_status, self.crb_active,
                          self.first_stage, self.first_code], stream=stream)
        crb_launch(self.crb_model, self.crb_active, self.cinert, self.cdof,
                   self.crb, self.mass.qm, self.crb_status, stream=stream)
        wp.launch(_gate2, dim=self.capacity,
                  inputs=[self.crb_active, self.crb_status, self.mass.active,
                          self.first_stage, self.first_code], stream=stream)
        self.mass.launch(stream=stream)
        wp.launch(_finish, dim=self.capacity,
                  inputs=[self.input_active, self.mass.active, self.mass.status,
                          self.first_stage, self.first_code, self.result_kind],
                  stream=stream)

    def snapshot(self):
        """Download all boundaries after launch completion, for diagnostics only."""
        names = ("input_active", "xipos", "ximat", "xanchor", "xmat", "xaxis",
                 "subtree_com", "cinert", "cdof", "compos_status",
                 "crb_active", "crb", "crb_status", "first_stage", "first_code",
                 "result_kind")
        out = {name: np.array(getattr(self, name).numpy(), copy=True) for name in names}
        for name in ("qm", "rhs", "count", "active", "ld", "inv", "sqrt_inv",
                     "solution", "pivot", "status", "warning_count", "warning_last"):
            out["mass_" + name] = np.array(getattr(self.mass, name).numpy(), copy=True)
        return out
