"""Actual OSC launch arguments at both source read boundaries.

D07's physical outputs are observed before refresh_cache. Cache and all other
OSC inputs are observed after refresh_cache, immediately before fused_osc_control.
Snapshots are D2D on the prepared stream; masks describe source reads/writes,
including attempted writes subsequently rolled back by the dynamics wrapper.
"""
from __future__ import annotations
import libero.gpu.precision as scalar
import warp as wp

wp.set_module_options({"enable_backward": False, "fuse_fp": True, "fast_math": False})

# These are the exact unchanged PandaOSCWorkspace.launch argument orders.
REFRESH_ARGS = (
    "active", "needs_update", "cache_valid", "error", "phys_mass",
    "phys_jacobian", "phys_q", "phys_qvel", "phys_ee_pos", "phys_ee_rot",
    "phys_ee_velocity", "mass", "jacobian", "q", "qvel", "ee_pos",
    "ee_rot", "ee_velocity",
)
OSC_ARGS = (
    "mass", "jacobian", "q", "qvel", "phys_bias", "q0", "ee_pos", "ee_rot",
    "ee_velocity", "action", "goal_pos", "goal_rot", "gripper", "active",
    "gains", "ctrl_range", "ctrl", "torques_raw", "error", "ranks",
    "policy_step", "needs_update", "mass_inv", "A_full", "A_pos", "A_ori",
    "lambda_full", "lambda_pos", "lambda_ori", "nullspace_matrix",
    "position_error", "orientation_error", "quat2mat_input", "quat_cast_wxyz",
    "quat2mat_result", "quat_valid", "nullspace_torque", "torques_clipped",
)
PHYS_FIELDS = (
    "phys_mass", "phys_jacobian", "phys_q", "phys_qvel", "phys_ee_pos",
    "phys_ee_rot", "phys_ee_velocity", "phys_bias",
)
CACHE_FIELDS = ("mass", "jacobian", "q", "qvel", "ee_pos", "ee_rot", "ee_velocity")
CONTEXT_COLUMNS = (
    "state_epoch", "forward_seq", "position_writer", "velocity_writer",
    "bias_writer", "action_input_writer", "q0_input_writer", "gripper_input_writer",
    "cache_writer_before", "cache_forward_before", "gripper_writer_before",
    "adapter_writer", "adapter_forward", "cache_consumed_writer",
    "cache_consumed_forward", "live_bias_consumed_writer",
)
REFRESH_READS = ("active", "error", "needs_update", "cache_valid") + PHYS_FIELDS[:-1]
OSC_READS = (
    "active", "error", "mass", "jacobian", "phys_bias", "q0", "q", "qvel",
    "ee_pos", "ee_rot", "ee_velocity", "action", "goal_pos", "goal_rot",
    "gripper", "gains", "ctrl_range", "policy_step",
)

@wp.kernel(enable_backward=False)
def adapter_finish(gate: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32),
                   forward: wp.array(dtype=wp.int64), version: wp.array(dtype=wp.int64),
                   stamp: wp.array(dtype=wp.int64), valid: wp.array(dtype=wp.int32)):
    w = wp.tid()
    valid[w] = 0
    if gate[w] == 1 and status[w] == 0:
        version[w] = version[w] + wp.int64(1)
        stamp[w] = forward[w]
        valid[w] = 1

@wp.kernel(enable_backward=False)
def context_before(
    state: wp.array(dtype=wp.int64), forward: wp.array(dtype=wp.int64),
    position: wp.array(dtype=wp.int64), velocity: wp.array(dtype=wp.int64),
    bias: wp.array(dtype=wp.int64), input_versions: wp.array(dtype=wp.int64, ndim=2),
    clock: wp.array(dtype=wp.int64, ndim=2), cache_forward: wp.array(dtype=wp.int64),
    adapter_version: wp.array(dtype=wp.int64), adapter_stamp: wp.array(dtype=wp.int64),
    active: wp.array(dtype=wp.int32), error: wp.array(dtype=wp.int32),
    needs_update: wp.array(dtype=wp.int32), context: wp.array(dtype=wp.int64, ndim=2)):
    w = wp.tid()
    context[w,0] = state[w]
    context[w,1] = forward[w]
    context[w,2] = position[w]
    context[w,3] = velocity[w]
    context[w,4] = bias[w]
    for i in range(3):
        context[w,5+i] = input_versions[w,i]
    context[w,8] = clock[w,5]
    context[w,9] = cache_forward[w]
    context[w,10] = clock[w,6]
    context[w,11] = adapter_version[w]
    context[w,12] = adapter_stamp[w]
    context[w,13] = clock[w,5]
    context[w,14] = cache_forward[w]
    if active[w] != 0 and error[w] == 0 and needs_update[w] != 0:
        context[w,13] = clock[w,5] + wp.int64(1)
        context[w,14] = forward[w]
    context[w,15] = bias[w]

@wp.kernel(enable_backward=False)
def refresh_read_mask(active: wp.array(dtype=wp.int32), error: wp.array(dtype=wp.int32),
                      needs: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32), kind: int):
    w = wp.tid()
    out[w] = 0
    if kind == 0:
        out[w] = 1  # active is tested in every slot
    elif kind == 1:
        if active[w] != 0:
            out[w] = 1
    elif active[w] != 0 and error[w] == 0:
        if kind == 2 or (kind == 3 and needs[w] == 0) or (kind == 4 and needs[w] != 0):
            out[w] = 1

@wp.kernel(enable_backward=False)
def osc_read_mask(active: wp.array(dtype=wp.int32), error_before: wp.array(dtype=wp.int32),
                  error_after: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32), kind: int):
    w = wp.tid()
    out[w] = 0
    if kind == 0:
        out[w] = 1
    elif kind == 1:
        if active[w] != 0:
            out[w] = 1
    elif active[w] != 0 and error_before[w] == 0:
        if kind == 2 or (kind == 3 and error_after[w] == 0):
            out[w] = 1

@wp.kernel(enable_backward=False)
def action_read_mask(active: wp.array(dtype=wp.int32), error_before: wp.array(dtype=wp.int32),
                     error_after: wp.array(dtype=wp.int32), policy: wp.array(dtype=wp.int32),
                     out: wp.array(dtype=wp.int32, ndim=2)):
    w = wp.tid()
    for i in range(7):
        out[w,i] = 0
        if active[w] != 0 and error_before[w] == 0 and error_after[w] == 0:
            if i == 6 or policy[w] != 0:
                out[w,i] = 1

@wp.kernel(enable_backward=False)
def refresh_write_mask(active: wp.array(dtype=wp.int32), error: wp.array(dtype=wp.int32),
                       needs: wp.array(dtype=wp.int32), valid: wp.array(dtype=wp.int32),
                       out: wp.array(dtype=wp.int32), kind: int):
    w = wp.tid()
    out[w] = 0
    if active[w] != 0 and error[w] == 0:
        if kind == 0 and needs[w] != 0:
            out[w] = 1
        elif kind == 1 and needs[w] == 0 and valid[w] == 0:
            out[w] = 1

@wp.kernel(enable_backward=False)
def osc_write_mask(active: wp.array(dtype=wp.int32), before: wp.array(dtype=wp.int32),
                   after: wp.array(dtype=wp.int32), policy: wp.array(dtype=wp.int32),
                   action: wp.array(dtype=scalar.wp_real, ndim=2),
                   out: wp.array(dtype=wp.int32), kind: int, diagnostics: int):
    w = wp.tid()
    out[w] = 0
    if active[w] == 0 or before[w] != 0:
        return
    if kind == 0:
        if after[w] != 1 and after[w] != 2:
            out[w] = 1  # mass_inv and ranks, even if later pinv fails
    elif kind == 1:
        if after[w] != 0:
            out[w] = 1  # error first written by this kernel
    elif after[w] == 0:
        if kind == 2 or (kind == 3 and diagnostics != 0):
            out[w] = 1
        elif kind == 4 and policy[w] != 0:
            # clipping keeps nonzero/NaN nonzero; only signed zero suppresses.
            if action[w,3] != scalar.wp_real(0.0) or action[w,4] != scalar.wp_real(0.0) or action[w,5] != scalar.wp_real(0.0):
                out[w] = 1

@wp.kernel(enable_backward=False)
def finalize_context(mass_read: wp.array(dtype=wp.int32), bias_read: wp.array(dtype=wp.int32),
                     context: wp.array(dtype=wp.int64, ndim=2)):
    w = wp.tid()
    if mass_read[w] == 0:
        context[w,13] = wp.int64(-1)
        context[w,14] = wp.int64(-1)
    if bias_read[w] == 0:
        context[w,15] = wp.int64(-1)

@wp.kernel(enable_backward=False)
def commit_masks(active: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32),
                 committed: wp.array(dtype=wp.int32), rollback: wp.array(dtype=wp.int32)):
    w = wp.tid()
    committed[w] = 0
    rollback[w] = 1
    if active[w] == 1 and status[w] == 0:
        committed[w] = 1
        rollback[w] = 0


class ControlConsumption:
    """Prepared typed views, owner bindings and source-accurate D2D captures."""
    def __init__(self, controller, capacity, device):
        self.refresh_sources = {name:getattr(controller,name) for name in REFRESH_ARGS}
        self.osc_sources = {name:getattr(controller,name) for name in OSC_ARGS}
        self.refresh_before = {name:wp.empty_like(a) for name,a in self.refresh_sources.items()}
        self.osc_before = {name:wp.empty_like(a) for name,a in self.osc_sources.items()}
        self.refresh_read = {name:wp.zeros(capacity,dtype=wp.int32,device=device) for name in REFRESH_READS}
        self.osc_read = {name:wp.zeros(capacity,dtype=wp.int32,device=device) for name in OSC_READS}
        self.refresh_write = {name:wp.zeros(capacity,dtype=wp.int32,device=device)
                              for name in CACHE_FIELDS+("cache_valid","needs_update","error")}
        self.osc_write = {name:wp.zeros(capacity,dtype=wp.int32,device=device)
                          for name in OSC_ARGS if name not in OSC_READS or name in ("error","goal_pos","goal_rot","gripper")}
        self.action_read = wp.zeros((capacity,7),dtype=wp.int32,device=device)
        self.context = wp.zeros((capacity,len(CONTEXT_COLUMNS)),dtype=wp.int64,device=device)
        self.adapter_version = wp.zeros(capacity,dtype=wp.int64,device=device)
        self.adapter_stamp = wp.zeros(capacity,dtype=wp.int64,device=device)
        self.adapter_valid = wp.zeros(capacity,dtype=wp.int32,device=device)
        self.osc_error_after = wp.zeros(capacity,dtype=wp.int32,device=device)
        self.committed = wp.zeros(capacity,dtype=wp.int32,device=device)
        self.rollback = wp.zeros(capacity,dtype=wp.int32,device=device)
        self._capacity, self._device = capacity, device

    def launch(self, dynamics, module, stream):
        c, w = dynamics.control, dynamics.witness
        wp.launch(adapter_finish,dim=self._capacity,
                  inputs=[dynamics.control_active,dynamics.status_control,w.scratch_epoch,
                          self.adapter_version,self.adapter_stamp,self.adapter_valid],
                  device=self._device,stream=stream)
        for name,source in self.refresh_sources.items():
            wp.copy(self.refresh_before[name],source,stream=stream)
        for name in PHYS_FIELDS:
            wp.copy(dynamics.consumer_witness["control"][name],getattr(c,name),stream=stream)
        wp.launch(context_before,dim=self._capacity,
                  inputs=[w.state_epoch,w.scratch_epoch,w.versions["position"],w.versions["velocity_prefix"],
                          w.versions["bias"],dynamics.input_versions,dynamics.device_clock,
                          dynamics.cache_forward,self.adapter_version,self.adapter_stamp,
                          self.refresh_before["active"],self.refresh_before["error"],
                          self.refresh_before["needs_update"],self.context],
                  device=self._device,stream=stream)
        for name,out in self.refresh_read.items():
            kind = 0 if name=="active" else 1 if name=="error" else 2 if name=="needs_update" else 3 if name=="cache_valid" else 4
            wp.launch(refresh_read_mask,dim=self._capacity,
                      inputs=[self.refresh_before["active"],self.refresh_before["error"],
                              self.refresh_before["needs_update"],out,kind],device=self._device,stream=stream)
        for name,out in self.refresh_write.items():
            wp.launch(refresh_write_mask,dim=self._capacity,
                      inputs=[self.refresh_before["active"],self.refresh_before["error"],
                              self.refresh_before["needs_update"],self.refresh_before["cache_valid"],
                              out,1 if name=="error" else 0],device=self._device,stream=stream)
        wp.launch(module.refresh_cache,dim=self._capacity,
                  inputs=[self.refresh_sources[name] for name in REFRESH_ARGS],device=self._device,stream=stream)
        for name,source in self.osc_sources.items():
            wp.copy(self.osc_before[name],source,stream=stream)
        wp.launch(module.fused_osc_control,dim=self._capacity,
                  inputs=[self.osc_sources[name] for name in OSC_ARGS]+[c.diagnostics,c.uncouple],
                  device=self._device,stream=stream)
        wp.copy(self.osc_error_after,c.error,stream=stream)
        for name,out in self.osc_read.items():
            kind = 0 if name=="active" else 1 if name=="error" else 2 if name in ("mass","jacobian","phys_bias","q0") else 3
            wp.launch(osc_read_mask,dim=self._capacity,
                      inputs=[self.osc_before["active"],self.osc_before["error"],self.osc_error_after,out,kind],
                      device=self._device,stream=stream)
        wp.launch(finalize_context,dim=self._capacity,
                  inputs=[self.osc_read["mass"],self.osc_read["phys_bias"],self.context],
                  device=self._device,stream=stream)
        wp.launch(action_read_mask,dim=self._capacity,
                  inputs=[self.osc_before["active"],self.osc_before["error"],self.osc_error_after,
                          self.osc_before["policy_step"],self.action_read],device=self._device,stream=stream)
        for name,out in self.osc_write.items():
            kind = 0 if name in ("mass_inv","ranks") else 1 if name=="error" else 3 if name=="nullspace_matrix" else 4 if name in ("quat2mat_input","quat_cast_wxyz","quat2mat_result") else 2
            wp.launch(osc_write_mask,dim=self._capacity,
                      inputs=[self.osc_before["active"],self.osc_before["error"],self.osc_error_after,
                              self.osc_before["policy_step"],self.osc_before["action"],out,kind,c.diagnostics],
                      device=self._device,stream=stream)

    def finish(self, dynamics, stream):
        wp.launch(commit_masks,dim=self._capacity,
                  inputs=[dynamics.gate_control,dynamics.status_control,self.committed,self.rollback],
                  device=self._device,stream=stream)
