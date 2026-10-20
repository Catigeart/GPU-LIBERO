"""Stage-split transmission for the frozen nine-actuator Panda model.

Source: MuJoCo 2.3.7 engine_forward.c:122 (mj_transmission in the position
stage writes actuator_length/actuator_moment) and :143 (actuator_velocity =
actuator_moment @ qvel via mju_mulMatVec in the velocity stage).  The accepted
fused actuation kernel (libero/gpu/reference/dynamics/actuation/kernel.py) computes these three
arrays together with the force stage; these split kernels keep the original
per-array arithmetic so each stage entry has its own write set and epoch.
The fixed topology is nine scalar-joint transmissions (trntype=0), constant
gear: length[i] = qpos[qadr[i]] * gear[i]; moment[i, dof[i]] = gear[i]; the
velocity reduction keeps the accepted mju_dot four-lane order.
"""
from __future__ import annotations
import libero.gpu.precision as scalar

import numpy as np
import warp as wp

wp.set_module_options({'enable_backward': False, 'fuse_fp': True, 'fast_math': False})

STATUS_OK = 0
STATUS_ACTIVE = 1
STATUS_INPUT = 2
STATUS_TOPOLOGY = 3
STATUS_ARITHMETIC = 4


@wp.kernel(enable_backward=False)
def _transmission_position(qpos: wp.array(dtype=scalar.wp_real, ndim=2),
                           active: wp.array(dtype=wp.int32),
                           qadr: wp.array(dtype=wp.int32),
                           dadr: wp.array(dtype=wp.int32),
                           gear: wp.array(dtype=scalar.wp_real),
                           length: wp.array(dtype=scalar.wp_real, ndim=2),
                           moment: wp.array(dtype=scalar.wp_real, ndim=3),
                           status: wp.array(dtype=wp.int32)):
    w = wp.tid()
    status[w] = STATUS_OK
    for i in range(9):
        length[w, i] = scalar.wp_real(0.0)
        for j in range(43):
            moment[w, i, j] = scalar.wp_real(0.0)
    if active[w] == 0:
        return
    if active[w] != 1:
        status[w] = STATUS_ACTIVE
        return
    for j in range(48):
        if not wp.isfinite(qpos[w, j]):
            status[w] = STATUS_INPUT
            return
    for i in range(9):
        a = qadr[i]
        b = dadr[i]
        if a < 0 or a > 47 or b < 0 or b > 42:
            status[w] = STATUS_TOPOLOGY
            return
        v = qpos[w, a] * gear[i]
        if not wp.isfinite(v):
            status[w] = STATUS_ARITHMETIC
            return
        length[w, i] = v
        moment[w, i, b] = gear[i]


@wp.kernel(enable_backward=False)
def _transmission_velocity(moment: wp.array(dtype=scalar.wp_real, ndim=3),
                           qvel: wp.array(dtype=scalar.wp_real, ndim=2),
                           active: wp.array(dtype=wp.int32),
                           pos_status: wp.array(dtype=wp.int32),
                           velocity: wp.array(dtype=scalar.wp_real, ndim=2),
                           status: wp.array(dtype=wp.int32)):
    w = wp.tid()
    status[w] = STATUS_OK
    for i in range(9):
        velocity[w, i] = scalar.wp_real(0.0)
    if active[w] == 0:
        return
    if active[w] != 1:
        status[w] = STATUS_ACTIVE
        return
    if pos_status[w] != 0:
        status[w] = STATUS_TOPOLOGY
        return
    for j in range(43):
        if not wp.isfinite(qvel[w, j]):
            status[w] = STATUS_INPUT
            return
    for i in range(9):
        # mju_dot's four independent lanes and final three-element tail,
        # matching the accepted fused actuation kernel exactly.
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
        dot = dot + (moment[w, i, 40] * qvel[w, 40]
                     + moment[w, i, 41] * qvel[w, 41]
                     + moment[w, i, 42] * qvel[w, 42])
        if not wp.isfinite(dot):
            status[w] = STATUS_ARITHMETIC
            return
        velocity[w, i] = dot


class TransmissionWorkspace:
    """Stage-split length/moment (position) and velocity (velocity prefix).

    The fixed-model rule (nine scalar-joint actuators, one DOF each) is
    validated on the host from the model arrays before allocation.  Launch
    methods are per stage; each writes only its own output group.
    """

    def __init__(self, model, capacity: int, device='cuda:0'):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('positive int capacity required')
        self.capacity = capacity
        self.device = device
        trnid = np.asarray(model.actuator_trnid)
        jnt_type = np.asarray(model.jnt_type)
        qadr = np.asarray(model.jnt_qposadr)[trnid[:, 0]]
        dadr = np.asarray(model.jnt_dofadr)[trnid[:, 0]]
        gear = np.asarray(model.actuator_gear[:, 0])
        if (int(model.nu), int(model.nv), int(model.nq)) != (9, 43, 48):
            raise ValueError('fixed model transmission topology')
        if not np.array_equal(trnid[:, 0], np.arange(9)) or np.any(trnid[:, 1] != -1):
            raise ValueError('fixed model transmission ids')
        if not np.array_equal(jnt_type[trnid[:, 0]], [3] * 7 + [2] * 2):
            raise ValueError('fixed model transmission joint types')
        if gear.shape != (9,) or gear.dtype != np.float64 or not np.isfinite(gear).all():
            raise ValueError('fixed model gear array')
        self.qadr = wp.array(np.ascontiguousarray(qadr, dtype=np.int32),
                             dtype=wp.int32, device=device)
        self.dadr = wp.array(np.ascontiguousarray(dadr, dtype=np.int32),
                             dtype=wp.int32, device=device)
        self.gear = wp.array(np.ascontiguousarray(gear, dtype=np.float64),
                             dtype=scalar.wp_real, device=device)
        self.qpos = wp.zeros((capacity, 48), dtype=scalar.wp_real, device=device)
        self.qvel = wp.zeros((capacity, 43), dtype=scalar.wp_real, device=device)
        self.position_active = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.velocity_active = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.length = wp.zeros((capacity, 9), dtype=scalar.wp_real, device=device)
        self.moment = wp.zeros((capacity, 9, 43), dtype=scalar.wp_real, device=device)
        self.velocity = wp.zeros((capacity, 9), dtype=scalar.wp_real, device=device)
        self.position_status = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.velocity_status = wp.zeros(capacity, dtype=wp.int32, device=device)

    def launch_position(self, stream=None):
        wp.launch(_transmission_position, dim=self.capacity,
                  inputs=[self.qpos, self.position_active, self.qadr, self.dadr,
                          self.gear, self.length, self.moment,
                          self.position_status],
                  device=self.device, stream=stream)

    def launch_velocity(self, stream=None):
        wp.launch(_transmission_velocity, dim=self.capacity,
                  inputs=[self.moment, self.qvel, self.velocity_active,
                          self.position_status, self.velocity,
                          self.velocity_status],
                  device=self.device, stream=stream)

    def download(self):
        return {
            'length': np.array(self.length.numpy(), copy=True),
            'moment': np.array(self.moment.numpy(), copy=True),
            'velocity': np.array(self.velocity.numpy(), copy=True),
            'position_status': np.array(self.position_status.numpy(), copy=True),
            'velocity_status': np.array(self.velocity_status.numpy(), copy=True),
        }
