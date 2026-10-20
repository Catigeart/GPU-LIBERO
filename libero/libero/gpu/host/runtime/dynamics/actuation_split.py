"""Force stage of the frozen nine-actuator transmission, consuming staged outputs.

Source: MuJoCo 2.3.7 engine_forward.c:158-217 (mj_fwdActuation): clear force
and qfrc_actuator, local clamped ctrl copy, fixed gain, affine or zero bias,
force limits, then mju_mulMatTVec in row order skipping exactly-zero force
rows.  These kernels consume the stage-split length/moment/velocity produced
by TransmissionWorkspace so the consumer witness is a real device edge.  The
borrowed fused implementation is a source/control reference; this module
still requires the complete original native and device scientific gate.
"""
from __future__ import annotations
import libero.gpu.precision as scalar
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT

import numpy as np
import warp as wp

wp.set_module_options({'enable_backward': False, 'fuse_fp': True, 'fast_math': False})

STATUS_OK = 0
STATUS_ACTIVE = 1
STATUS_INPUT = 2
STATUS_ARITHMETIC = 3


@wp.kernel(enable_backward=False)
def _force_stage(length: wp.array(dtype=scalar.wp_real, ndim=2),
                 moment: wp.array(dtype=scalar.wp_real, ndim=3),
                 velocity: wp.array(dtype=scalar.wp_real, ndim=2),
                 ctrl: wp.array(dtype=scalar.wp_real, ndim=2),
                 active: wp.array(dtype=wp.int32),
                 vel_status: wp.array(dtype=wp.int32),
                 gain: wp.array(dtype=scalar.wp_real),
                 bias0: wp.array(dtype=scalar.wp_real),
                 bias1: wp.array(dtype=scalar.wp_real),
                 bias2: wp.array(dtype=scalar.wp_real),
                 bias_type: wp.array(dtype=wp.int32),
                 ctrl_lo: wp.array(dtype=scalar.wp_real),
                 ctrl_hi: wp.array(dtype=scalar.wp_real),
                 force_lo: wp.array(dtype=scalar.wp_real),
                 force_hi: wp.array(dtype=scalar.wp_real),
                 force_limited: wp.array(dtype=wp.int32),
                 disable_flags: wp.int32,
                 force: wp.array(dtype=scalar.wp_real, ndim=2),
                 force_pre: wp.array(dtype=scalar.wp_real, ndim=2),
                 ctrl_used: wp.array(dtype=scalar.wp_real, ndim=2),
                 ctrl_route: wp.array(dtype=wp.int32, ndim=2),
                 force_route: wp.array(dtype=wp.int32, ndim=2),
                 qfrc_actuator: wp.array(dtype=scalar.wp_real, ndim=2),
                 status: wp.array(dtype=wp.int32),
                 warn_badctrl_count: wp.array(dtype=wp.int32),
                 warn_badctrl_lastinfo: wp.array(dtype=wp.int32),
                 warning_number: wp.array(dtype=wp.int32),
                 warning_lastinfo: wp.array(dtype=wp.int32)):
    w = wp.tid()
    status[w] = STATUS_OK
    warn_badctrl_count[w] = 0
    warn_badctrl_lastinfo[w] = -1
    for j in range(43):
        qfrc_actuator[w, j] = scalar.wp_real(0.0)
    for i in range(9):
        force[w, i] = scalar.wp_real(0.0)
        force_pre[w, i] = scalar.wp_real(0.0)
        ctrl_used[w, i] = scalar.wp_real(0.0)
        ctrl_route[w, i] = 0
        force_route[w, i] = 0
    if active[w] == 0:
        return
    if active[w] != 1:
        status[w] = STATUS_ACTIVE
        return
    if vel_status[w] != 0:
        status[w] = STATUS_INPUT
        return
    # mj_fwdActuation returns after clearing outputs when actuation is
    # disabled; length/moment/velocity above still run (already staged).
    if (disable_flags & 1024) != 0:
        return

    # Local clamped ctrl copy, then first-bad scan (mju_isBad: NaN or |u|>1e10).
    bad_index = -1
    for i in range(9):
        u = ctrl[w, i]
        if (disable_flags & 128) == 0:
            if u < ctrl_lo[i]:
                u = ctrl_lo[i]
                ctrl_route[w, i] = -1
            elif u > ctrl_hi[i]:
                u = ctrl_hi[i]
                ctrl_route[w, i] = 1
        if bad_index < 0 and (u != u or u > scalar.wp_real(1.0e10) or u < scalar.wp_real(-1.0e10)):
            bad_index = i
    if bad_index >= 0:
        warn_badctrl_count[w] = 1
        warn_badctrl_lastinfo[w] = bad_index
        # Original mj_warning changes this persistent MjData state only on
        # BADCTRL; a healthy, inactive, disabled or wrapper-error call retains it.
        # Per-call diagnostic delta above remains a distinct interface.
        warning_number[w] = warning_number[w] + 1
        warning_lastinfo[w] = bad_index

    for i in range(9):
        u = scalar.wp_real(0.0)
        if bad_index < 0:
            u = ctrl[w, i]
            if ctrl_route[w, i] == -1:
                u = ctrl_lo[i]
            elif ctrl_route[w, i] == 1:
                u = ctrl_hi[i]
        ctrl_used[w, i] = u
        f = gain[i] * u
        if bias_type[i] == 1:
            f = f + (bias0[i] + bias1[i] * length[w, i] + bias2[i] * velocity[w, i])
        force_pre[w, i] = f
        if force_limited[i] != 0:
            if f < force_lo[i]:
                f = force_lo[i]
                force_route[w, i] = -1
            elif f > force_hi[i]:
                f = force_hi[i]
                force_route[w, i] = 1
        force[w, i] = f

    # mju_mulMatTVec: row order, skip exactly-zero force rows.
    for i in range(9):
        f = force[w, i]
        if f != scalar.wp_real(0.0):
            for j in range(43):
                qfrc_actuator[w, j] = qfrc_actuator[w, j] + moment[w, i, j] * f
    for j in range(43):
        if not wp.isfinite(qfrc_actuator[w, j]):
            status[w] = STATUS_ARITHMETIC
            return


@wp.kernel(enable_backward=False)
def _reset_warning(mask: wp.array(dtype=wp.int32),
                   number: wp.array(dtype=wp.int32),
                   lastinfo: wp.array(dtype=wp.int32),
                   status: wp.array(dtype=wp.int32)):
    w = wp.tid()
    status[w] = 0
    if mask[w] == 0:
        return
    if mask[w] != 1:
        status[w] = 1
        return
    number[w] = 0
    lastinfo[w] = 0


class ActuationForceWorkspace:
    """Force/qfrc_actuator stage consuming staged length/moment/velocity."""

    def __init__(self, model, capacity: int, device='cuda:0'):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('positive int capacity required')
        self.capacity = capacity
        self.device = device
        self.disable_flags = _validate_fixed_model(model)
        self.qadr = wp.array(
            np.ascontiguousarray(
                np.asarray(model.jnt_qposadr)[np.asarray(model.actuator_trnid)[:, 0]],
                dtype=np.int32), dtype=wp.int32, device=device)
        arrays = {
            'gain': np.asarray(model.actuator_gainprm[:, 0], np.float64),
            'bias0': np.asarray(model.actuator_biasprm[:, 0], np.float64),
            'bias1': np.asarray(model.actuator_biasprm[:, 1], np.float64),
            'bias2': np.asarray(model.actuator_biasprm[:, 2], np.float64),
            'ctrl_lo': np.asarray(model.actuator_ctrlrange[:, 0], np.float64),
            'ctrl_hi': np.asarray(model.actuator_ctrlrange[:, 1], np.float64),
            'force_lo': np.asarray(model.actuator_forcerange[:, 0], np.float64),
            'force_hi': np.asarray(model.actuator_forcerange[:, 1], np.float64),
            'force_limited': np.asarray(model.actuator_forcelimited, np.int32),
            'bias_type': np.asarray(model.actuator_biastype, np.int32),
        }
        for name, arr in arrays.items():
            arr = np.ascontiguousarray(arr)
            dtype = wp.int32 if arr.dtype == np.int32 else scalar.wp_real
            setattr(self, name, wp.array(arr, dtype=dtype, device=device))
        F, I = scalar.wp_real, wp.int32
        self.ctrl = wp.zeros((capacity, 9), dtype=F, device=device)
        self.active = wp.zeros(capacity, dtype=I, device=device)
        self.force = wp.zeros((capacity, 9), dtype=F, device=device)
        self.force_pre = wp.zeros((capacity, 9), dtype=F, device=device)
        self.ctrl_used = wp.zeros((capacity, 9), dtype=F, device=device)
        self.ctrl_route = wp.zeros((capacity, 9), dtype=I, device=device)
        self.force_route = wp.zeros((capacity, 9), dtype=I, device=device)
        self.qfrc_actuator = wp.zeros((capacity, 43), dtype=F, device=device)
        self.status = wp.zeros(capacity, dtype=I, device=device)
        self.warn_badctrl_count = wp.zeros(capacity, dtype=I, device=device)
        self.warn_badctrl_lastinfo = wp.zeros(capacity, dtype=I, device=device)
        self.warning_number = wp.zeros(capacity, dtype=I, device=device)
        self.warning_lastinfo = wp.zeros(capacity, dtype=I, device=device)
        self.warning_reset_mask = wp.zeros(capacity, dtype=I, device=device)
        self.warning_reset_status = wp.zeros(capacity, dtype=I, device=device)

    def reset_warning_at_boundary(self, mask, stream=None):
        """Only the warning sub-state of an explicit original data-reset boundary.

        Full DynamicsStages/native reset integration remains separate. The
        live device mask controls reset/hold; invalid device values retain both
        persistent words and return local reset status 1 without a host read.
        """
        if (not isinstance(mask, wp.array) or mask.dtype != wp.int32 or
                tuple(mask.shape) != (self.capacity,) or
                str(mask.device) != str(wp.get_device(self.device))):
            raise ValueError('actual int32 CUDA warning reset mask (N,) required')
        wp.copy(self.warning_reset_mask, mask, stream=stream)
        wp.launch(_reset_warning, dim=self.capacity,
                  inputs=[self.warning_reset_mask, self.warning_number,
                          self.warning_lastinfo, self.warning_reset_status],
                  device=self.device, stream=stream)

    def launch(self, length, moment, velocity, vel_status, stream=None):
        wp.launch(_force_stage, dim=self.capacity,
                  inputs=[length, moment, velocity, self.ctrl, self.active,
                          vel_status, self.gain, self.bias0, self.bias1,
                          self.bias2, self.bias_type, self.ctrl_lo, self.ctrl_hi,
                          self.force_lo, self.force_hi, self.force_limited,
                          self.disable_flags, self.force, self.force_pre,
                          self.ctrl_used, self.ctrl_route, self.force_route,
                          self.qfrc_actuator, self.status,
                          self.warn_badctrl_count, self.warn_badctrl_lastinfo,
                          self.warning_number, self.warning_lastinfo],
                  device=self.device, stream=stream)


def _validate_fixed_model(model):
    """Same fixed topology as the borrowed fused actuation source reference."""
    import importlib.util
    import sys
    from pathlib import Path
    name = 'p06d_accepted_actuation_validate'
    if name in sys.modules:
        mod = sys.modules[name]
    else:
        path = (GPU_ROOT / 'reference/dynamics/actuation/kernel.py')
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod._validate_fixed_model(model)
