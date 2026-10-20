"""Current LIBERO model's MuJoCo 2.3.7 transmission and actuation.

Source: Apache-2.0 MuJoCo 2.3.7, engine_core_smooth.c:614,
engine_forward.c:135,158, engine_util_blas.c:633,698,707. The scalar
joint/constant-gain subset is expressed in Warp; no MuJoCo C is copied.
"""

from __future__ import annotations
import libero.gpu.precision as scalar

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False, "fuse_fp": True})


def _validate_fixed_model(model):
    """Check original MJB static schema before any device allocation/cast.

    The only configurable option in this fixed topology is the disable mask.
    NONE bias parameters are intentionally ignored, as in the original C.
    """
    scalar_expected = {'nq': 48, 'nv': 43, 'nu': 9, 'na': 0, 'nplugin': 0}
    for name, expected in scalar_expected.items():
        value = np.asarray(getattr(model, name))
        if value.shape != () or value.dtype.kind not in 'iu' or int(value) != expected:
            raise ValueError(f'{name} fixed-model mismatch')
    flags = np.asarray(model.opt_disableflags)
    if (flags.shape != () or flags.dtype.kind not in 'iu'
            or not 0 <= int(flags) <= np.iinfo(np.int32).max):
        raise ValueError('opt_disableflags scalar/range')
    arrays = {}
    for name, shape, dtype in (
        ('jnt_type', (18,), np.int32),
        ('jnt_qposadr', (18,), np.int32),
        ('jnt_dofadr', (18,), np.int32),
        ('jnt_actfrclimited', (18,), np.uint8),
        ('actuator_trntype', (9,), np.int32),
        ('actuator_trnid', (9, 2), np.int32),
        ('actuator_gaintype', (9,), np.int32),
        ('actuator_dyntype', (9,), np.int32),
        ('actuator_biastype', (9,), np.int32),
        ('actuator_actadr', (9,), np.int32),
        ('actuator_plugin', (9,), np.int32),
        ('actuator_gear', (9, 6), np.float64),
        ('actuator_gainprm', (9, 10), np.float64),
        ('actuator_biasprm', (9, 10), np.float64),
        ('actuator_ctrllimited', (9,), np.uint8),
        ('actuator_ctrlrange', (9, 2), np.float64),
        ('actuator_forcelimited', (9,), np.uint8),
        ('actuator_forcerange', (9, 2), np.float64),
    ):
        value = np.asarray(getattr(model, name))
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f'{name} original dtype/shape')
        arrays[name] = value
    m = arrays
    if (np.any(m['jnt_actfrclimited'])
            or not np.array_equal(m['actuator_trnid'][:, 0], np.arange(9, dtype=np.int32))
            or np.any(m['actuator_trnid'][:, 1] != -1)
            or not np.array_equal(m['jnt_type'][:9], [3]*7 + [2]*2)
            or not np.array_equal(m['jnt_qposadr'][:9], np.arange(9))
            or not np.array_equal(m['jnt_dofadr'][:9], np.arange(9))
            or np.any(m['actuator_trntype'] != 0)
            or np.any(m['actuator_gaintype'] != 0)
            or np.any(m['actuator_dyntype'] != 0)
            or not np.array_equal(m['actuator_biastype'], [0]*7 + [1]*2)
            or np.any(m['actuator_actadr'] != -1)
            or np.any(m['actuator_plugin'] != -1)
            or np.any(m['actuator_ctrllimited'] != 1)
            or not np.array_equal(m['actuator_forcelimited'], [0]*7 + [1]*2)):
        raise ValueError('unsupported fixed actuator topology')
    expected_ctrl = np.array([[-80., 80.]]*5 + [[-12., 12.]]*2
                             + [[0., .04], [-.04, 0.]], np.float64)
    if (not np.all(np.isfinite(m['actuator_gear']))
            or not np.all(np.isfinite(m['actuator_gainprm']))
            or not np.all(np.isfinite(m['actuator_biasprm']))
            or not np.all(np.isfinite(m['actuator_ctrlrange']))
            or not np.all(np.isfinite(m['actuator_forcerange']))
            or not np.array_equal(m['actuator_gear'][:, 0], np.ones(9))
            or not np.array_equal(m['actuator_gainprm'][:, 0], [1.]*7 + [1000.]*2)
            or not np.array_equal(m['actuator_biasprm'][7:9, :3], [[0., -1000., 0.]]*2)
            or not np.array_equal(m['actuator_ctrlrange'], expected_ctrl)
            or not np.array_equal(m['actuator_forcerange'][7:9], [[-20., 20.]]*2)):
        raise ValueError('unsupported fixed actuator parameters')
    return int(flags)


@wp.kernel(enable_backward=False)
def _actuate(
    qpos: wp.array(dtype=scalar.wp_real, ndim=2),
    qvel: wp.array(dtype=scalar.wp_real, ndim=2),
    ctrl: wp.array(dtype=scalar.wp_real, ndim=2),
    active: wp.array(dtype=wp.int32),
    qadr: wp.array(dtype=wp.int32),
    dadr: wp.array(dtype=wp.int32),
    gear: wp.array(dtype=scalar.wp_real),
    gain: wp.array(dtype=scalar.wp_real),
    bias0: wp.array(dtype=scalar.wp_real),
    bias1: wp.array(dtype=scalar.wp_real),
    bias2: wp.array(dtype=scalar.wp_real),
    ctrl_lo: wp.array(dtype=scalar.wp_real),
    ctrl_hi: wp.array(dtype=scalar.wp_real),
    force_lo: wp.array(dtype=scalar.wp_real),
    force_hi: wp.array(dtype=scalar.wp_real),
    force_limited: wp.array(dtype=wp.int32),
    bias_type: wp.array(dtype=wp.int32),
    disable_flags: wp.int32,
    length: wp.array(dtype=scalar.wp_real, ndim=2),
    moment: wp.array(dtype=scalar.wp_real, ndim=3),
    velocity: wp.array(dtype=scalar.wp_real, ndim=2),
    force: wp.array(dtype=scalar.wp_real, ndim=2),
    force_pre: wp.array(dtype=scalar.wp_real, ndim=2),
    ctrl_used: wp.array(dtype=scalar.wp_real, ndim=2),
    ctrl_route: wp.array(dtype=wp.int32, ndim=2),
    force_route: wp.array(dtype=wp.int32, ndim=2),
    qfrc: wp.array(dtype=scalar.wp_real, ndim=2),
    status: wp.array(dtype=wp.int32),
    warn_badctrl_count: wp.array(dtype=wp.int32),
    warn_badctrl_lastinfo: wp.array(dtype=wp.int32),
):
    w = wp.tid()
    status[w] = 0
    warn_badctrl_count[w] = 0
    warn_badctrl_lastinfo[w] = -1
    for j in range(43):
        qfrc[w, j] = scalar.wp_real(0.0)
    for i in range(9):
        length[w, i] = scalar.wp_real(0.0)
        velocity[w, i] = scalar.wp_real(0.0)
        force[w, i] = scalar.wp_real(0.0)
        force_pre[w, i] = scalar.wp_real(0.0)
        ctrl_used[w, i] = scalar.wp_real(0.0)
        ctrl_route[w, i] = 0
        force_route[w, i] = 0
        for j in range(43):
            moment[w, i, j] = scalar.wp_real(0.0)
    if active[w] == 0:
        return
    if active[w] != 1:
        status[w] = 1
        return
    bad = int(0)
    for j in range(48):
        if not wp.isfinite(qpos[w, j]):
            bad = 1
    for j in range(43):
        if not wp.isfinite(qvel[w, j]):
            bad = 1
    if bad != 0:
        status[w] = 2
        return

    # Transmission: all nine actuators use one scalar joint DOF.
    for i in range(9):
        a = qadr[i]
        b = dadr[i]
        g = gear[i]
        length[w, i] = qpos[w, a] * g
        moment[w, i, b] = g

        # mju_dot's four independent lanes and final three-element tail.
        s0 = scalar.wp_real(0.0)
        s1 = scalar.wp_real(0.0)
        s2 = scalar.wp_real(0.0)
        s3 = scalar.wp_real(0.0)
        for k in range(10):
            j = 4 * k
            s0 = s0 + moment[w, i, j] * qvel[w, j]
            s1 = s1 + moment[w, i, j + 1] * qvel[w, j + 1]
            s2 = s2 + moment[w, i, j + 2] * qvel[w, j + 2]
            s3 = s3 + moment[w, i, j + 3] * qvel[w, j + 3]
        dot = (s0 + s2) + (s1 + s3)
        dot = dot + (moment[w, i, 40] * qvel[w, 40] + moment[w, i, 41] * qvel[w, 41] + moment[w, i, 42] * qvel[w, 42])
        velocity[w, i] = dot

    # mj_fwdActuation returns here after clearing its own force and qfrc.
    # Transmission and velocity above still run when actuation is disabled.
    if (disable_flags & 1024) != 0:
        return

    # The source clamps every local ctrl before scanning for the first bad
    # value.  mju_isBad accepts finite values in [-1e10, 1e10], inclusive.
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

    # fwdActuation: local ctrl clamp, fixed gain, affine or zero bias.
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
                qfrc[w, j] = qfrc[w, j] + moment[w, i, j] * f


class ActuationWorkspace:
    """Fixed-model device workspace; upload and download are caller boundaries."""

    def __init__(self, model, capacity: int, device: str = "cuda:0"):
        # This module reads only ACTUATION and CLAMPCTRL.  Other MuJoCo
        # disable bits, including GRAVITY, do not change these functions.
        self.disable_flags = _validate_fixed_model(model)
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.device = device
        self.qpos = wp.zeros((capacity, 48), dtype=scalar.wp_real, device=device)
        self.qvel = wp.zeros((capacity, 43), dtype=scalar.wp_real, device=device)
        self.ctrl = wp.zeros((capacity, 9), dtype=scalar.wp_real, device=device)
        self.active = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.length = wp.zeros((capacity, 9), dtype=scalar.wp_real, device=device)
        self.moment = wp.zeros((capacity, 9, 43), dtype=scalar.wp_real, device=device)
        self.velocity = wp.zeros((capacity, 9), dtype=scalar.wp_real, device=device)
        self.force = wp.zeros((capacity, 9), dtype=scalar.wp_real, device=device)
        self.force_pre = wp.zeros((capacity, 9), dtype=scalar.wp_real, device=device)
        self.ctrl_used = wp.zeros((capacity, 9), dtype=scalar.wp_real, device=device)
        self.ctrl_route = wp.zeros((capacity, 9), dtype=wp.int32, device=device)
        self.force_route = wp.zeros((capacity, 9), dtype=wp.int32, device=device)
        self.qfrc = wp.zeros((capacity, 43), dtype=scalar.wp_real, device=device)
        self.status = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.warn_badctrl_count = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.warn_badctrl_lastinfo = wp.zeros(capacity, dtype=wp.int32, device=device)
        arrays = {
            "qadr": np.asarray(model.jnt_qposadr[model.actuator_trnid[:, 0]], np.int32),
            "dadr": np.asarray(model.jnt_dofadr[model.actuator_trnid[:, 0]], np.int32),
            "gear": np.asarray(model.actuator_gear[:, 0], np.float64),
            "gain": np.asarray(model.actuator_gainprm[:, 0], np.float64),
            "bias0": np.asarray(model.actuator_biasprm[:, 0], np.float64),
            "bias1": np.asarray(model.actuator_biasprm[:, 1], np.float64),
            "bias2": np.asarray(model.actuator_biasprm[:, 2], np.float64),
            "ctrl_lo": np.asarray(model.actuator_ctrlrange[:, 0], np.float64),
            "ctrl_hi": np.asarray(model.actuator_ctrlrange[:, 1], np.float64),
            "force_lo": np.asarray(model.actuator_forcerange[:, 0], np.float64),
            "force_hi": np.asarray(model.actuator_forcerange[:, 1], np.float64),
            "force_limited": np.asarray(model.actuator_forcelimited, np.int32),
            "bias_type": np.asarray(model.actuator_biastype, np.int32),
        }
        for name, arr in arrays.items():
            setattr(self, name, wp.array(np.ascontiguousarray(arr), dtype=wp.int32 if arr.dtype == np.int32 else scalar.wp_real, device=device))

    def launch(self, stream=None):
        wp.launch(_actuate, dim=self.capacity,
                  inputs=[self.qpos, self.qvel, self.ctrl, self.active,
                          self.qadr, self.dadr, self.gear, self.gain,
                          self.bias0, self.bias1, self.bias2,
                          self.ctrl_lo, self.ctrl_hi, self.force_lo,
                          self.force_hi, self.force_limited, self.bias_type,
                          self.disable_flags,
                          self.length, self.moment, self.velocity,
                          self.force, self.force_pre, self.ctrl_used, self.ctrl_route,
                          self.force_route, self.qfrc, self.status,
                          self.warn_badctrl_count, self.warn_badctrl_lastinfo],
                  device=self.device, stream=stream)
