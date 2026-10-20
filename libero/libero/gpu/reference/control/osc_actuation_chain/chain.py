"""Connect accepted N18, Phase 1 OSC and N02 kernels on one CUDA stream.

Inputs are uploaded by a caller before launch.  Launch only enqueues resident
device kernels.  The two small adapters combine the original N18 velocity
components and route the first stage error without a host read or device copy.
"""
from __future__ import annotations
import libero.gpu.precision as scalar

import numpy as np
import warp as wp

from libero.gpu.reference.control.osc_inputs.kernel import OSCInputsWorkspace
from libero.gpu.reference.control.osc.controller import PandaOSCWorkspace
from libero.gpu.reference.dynamics.actuation.kernel import ActuationWorkspace

wp.set_module_options({"enable_backward": False, "fuse_fp": True})


@wp.kernel(enable_backward=False)
def _before_control(source_active: wp.array(dtype=wp.int32),
                    input_status: wp.array(dtype=wp.int32),
                    force_refresh: wp.array(dtype=wp.int32),
                    reset_mask: wp.array(dtype=wp.int32),
                    velp: wp.array(dtype=scalar.wp_real, ndim=2),
                    velr: wp.array(dtype=scalar.wp_real, ndim=2),
                    velocity: wp.array(dtype=scalar.wp_real, ndim=2),
                    control_active: wp.array(dtype=wp.int32),
                    reset_active: wp.array(dtype=wp.int32),
                    chain_status: wp.array(dtype=wp.int32),
                    needs_update: wp.array(dtype=wp.int32)):
    w = wp.tid()
    control_active[w] = 0
    reset_active[w] = 0
    if source_active[w] == 0:
        chain_status[w] = 0
        return
    if source_active[w] != 1:
        chain_status[w] = 1
        return
    if input_status[w] != 0:
        chain_status[w] = 100 + input_status[w]
        return
    if force_refresh[w] != 0 and force_refresh[w] != 1:
        chain_status[w] = 2
        return
    if reset_mask[w] != 0 and reset_mask[w] != 1:
        chain_status[w] = 3
        return
    for i in range(3):
        velocity[w, i] = velp[w, i]
        velocity[w, i + 3] = velr[w, i]
    if force_refresh[w] == 1:
        needs_update[w] = 1
    reset_active[w] = reset_mask[w]
    control_active[w] = 1
    chain_status[w] = 0


@wp.kernel(enable_backward=False)
def _before_actuation(source_active: wp.array(dtype=wp.int32),
                      control_active: wp.array(dtype=wp.int32),
                      control_error: wp.array(dtype=wp.int32),
                      actuation_active: wp.array(dtype=wp.int32),
                      chain_status: wp.array(dtype=wp.int32)):
    w = wp.tid()
    actuation_active[w] = 0
    if source_active[w] != 1 or chain_status[w] != 0:
        return
    if control_error[w] != 0:
        chain_status[w] = 200 + control_error[w]
        return
    if control_active[w] == 1:
        actuation_active[w] = 1


@wp.kernel(enable_backward=False)
def _after_actuation(source_active: wp.array(dtype=wp.int32),
                     actuation_status: wp.array(dtype=wp.int32),
                     chain_status: wp.array(dtype=wp.int32)):
    w = wp.tid()
    if source_active[w] == 1 and chain_status[w] == 0 and actuation_status[w] != 0:
        chain_status[w] = 300 + actuation_status[w]


class OSCActuationChain:
    """Fixed-model capacity N; physical arrays are exposed as ``physical``.

    Persistent controller state belongs to ``control``.  The local N02 ctrl
    buffer is a view of the *actual* Phase 1 output.  No host-produced ctrl or
    solved matrix is accepted by production launch.
    """

    def __init__(self, model, capacity: int, device="cuda:0"):
        if type(capacity) is not int or capacity not in (1, 4, 20, 128):
            raise ValueError("capacity must be one of 1, 4, 20, 128")
        self.capacity = capacity
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError("actual CUDA device required")
        self.physical = OSCInputsWorkspace(model, capacity, device=device)
        self.control = PandaOSCWorkspace(capacity, device=device,
                                         uncouple=True, diagnostics=True)
        self.actuation = ActuationWorkspace(model, capacity, device=device)
        p, c, a = self.physical, self.control, self.actuation
        # reshape returns a view of one allocation and performs no D2D copy.
        c.phys_mass = p.M.reshape((capacity, 49))
        c.phys_jacobian = p.J.reshape((capacity, 42))
        c.phys_q = p.joint_pos
        c.phys_qvel = p.joint_vel
        c.phys_bias = p.bias
        c.phys_ee_pos = p.ee_pos
        c.phys_ee_rot = p.ee_mat
        a.qpos = p.qpos
        a.qvel = p.qvel
        a.ctrl = c.ctrl
        self.control_active = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.actuation_active = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.chain_status = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.force_refresh = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.reset_mask = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.reset_active = wp.zeros(capacity, dtype=wp.int32, device=device)
        c.active = self.control_active
        a.active = self.actuation_active

    def launch(self, *, stream=None):
        """Production: N18, cache/OSC, N02, with no host state decisions."""
        p, c, a = self.physical, self.control, self.actuation
        p.launch(stream=stream)
        wp.launch(_before_control, dim=self.capacity,
                  inputs=[p.active, p.status, self.force_refresh, self.reset_mask, p.xvelp,
                          p.xvelr, c.phys_ee_velocity, self.control_active,
                          self.reset_active, self.chain_status, c.needs_update],
                  device=self.device, stream=stream)
        # The zero mask is a no-op. A valid device-updated mask resets goals
        # from the newly produced physical observation before control.
        c.reset_selected(self.reset_active, clear_gripper=True,
                         update_initial_joints=False, stream=stream)
        c.launch(stream=stream)
        wp.launch(_before_actuation, dim=self.capacity,
                  inputs=[p.active, self.control_active, c.error,
                          self.actuation_active, self.chain_status],
                  device=self.device, stream=stream)
        a.launch(stream=stream)
        wp.launch(_after_actuation, dim=self.capacity,
                  inputs=[p.active, a.status, self.chain_status],
                  device=self.device, stream=stream)

    def reset_selected(self, mask, *, clear_gripper=True,
                       update_initial_joints=False, stream=None):
        if mask.dtype != wp.int32 or mask.shape != (self.capacity,) or mask.device != self.device:
            raise ValueError("reset mask must be resident int32[N]")
        self.control.reset_selected(mask, clear_gripper=clear_gripper,
                                    update_initial_joints=update_initial_joints,
                                    stream=stream)
