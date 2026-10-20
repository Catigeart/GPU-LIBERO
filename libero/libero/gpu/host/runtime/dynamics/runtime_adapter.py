"""Resident D and OSC scheduling over prepared, model-qualified workspaces.

This module owns scheduling/publication, not physics arithmetic.  It calls the
existing accepted workspaces and keeps failed primitive work separate from
public results.  Preparation is outside capture; enqueue paths perform no host
array reads, uploads, allocation, or reconstruction of a device view.
"""
from __future__ import annotations
import libero.gpu.precision as scalar

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import warp as wp

wp.set_module_options({"enable_backward": False, "fuse_fp": True,
                       "fast_math": False})


@wp.kernel(enable_backward=False)
def _begin(mask: wp.array(dtype=wp.int32), error: wp.array(dtype=wp.int32),
           state_epoch: wp.array(dtype=wp.int64),
           source_epoch: wp.array(dtype=wp.int64),
           epoch: wp.array(dtype=wp.int64), active: wp.array(dtype=wp.int32),
           position: wp.array(dtype=wp.int32), velocity: wp.array(dtype=wp.int32),
           bias: wp.array(dtype=wp.int32), smooth: wp.array(dtype=wp.int32)):
    w = wp.tid()
    active[w] = 0
    if mask[w] == 0:
        return
    position[w] = 0
    velocity[w] = 0
    bias[w] = 0
    smooth[w] = 0
    if error[w] != 0:
        return
    if mask[w] != 1 or state_epoch[w] < wp.int64(0):
        error[w] = 201
        return
    if epoch[w] == wp.int64(9223372036854775807):
        error[w] = 9
        return
    epoch[w] = epoch[w] + wp.int64(1)
    source_epoch[w] = state_epoch[w]
    active[w] = 1


@wp.kernel(enable_backward=False)
def _gate(active: wp.array(dtype=wp.int32), error: wp.array(dtype=wp.int32),
          state_epoch: wp.array(dtype=wp.int64),
          source_epoch: wp.array(dtype=wp.int64),
          epoch: wp.array(dtype=wp.int64), parent_valid: wp.array(dtype=wp.int32),
          parent_written: wp.array(dtype=wp.int64), gate: wp.array(dtype=wp.int32),
          require_parent: int):
    w = wp.tid()
    gate[w] = 0
    if active[w] != 1 or error[w] != 0:
        return
    if state_epoch[w] != source_epoch[w]:
        error[w] = 203
        return
    if require_parent == 1:
        if parent_valid[w] != 1 or parent_written[w] != epoch[w]:
            return
    gate[w] = 1


@wp.kernel(enable_backward=False)
def _finish(gate: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32),
            epoch: wp.array(dtype=wp.int64), valid: wp.array(dtype=wp.int32),
            written: wp.array(dtype=wp.int64), error: wp.array(dtype=wp.int32),
            allow_repair: int, code: int):
    w = wp.tid()
    valid[w] = 0
    if gate[w] != 1 or error[w] != 0:
        return
    if status[w] != 0 and not (allow_repair == 1 and status[w] == 25):
        error[w] = code
        return
    written[w] = epoch[w]
    valid[w] = 1


@wp.kernel(enable_backward=False)
def _reference_gate(done: wp.array(dtype=wp.int32), err: wp.array(dtype=wp.int32),
                    written: wp.array(dtype=wp.int64),
                    expected: wp.array(dtype=wp.int64),
                    epoch: wp.array(dtype=wp.int64), gate: wp.array(dtype=wp.int32),
                    first: wp.array(dtype=wp.int32)):
    w = wp.tid()
    if gate[w] == 1:
        if done[w] != 1 or err[w] != 0 or written[w] != expected[w] or expected[w] != epoch[w]:
            gate[w] = 0
            if first[w] == 0:
                first[w] = 208


@wp.kernel(enable_backward=False)
def _position_status(kind: wp.array(dtype=wp.int32), code: wp.array(dtype=wp.int32),
                     transmission: wp.array(dtype=wp.int32),
                     status: wp.array(dtype=wp.int32)):
    w = wp.tid()
    status[w] = transmission[w]
    if kind[w] == 2:
        status[w] = 25
    elif kind[w] == 3:
        status[w] = 30 + code[w]


@wp.kernel(enable_backward=False)
def _velocity_status(a: wp.array(dtype=wp.int32), b: wp.array(dtype=wp.int32),
                     c: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
    w = wp.tid()
    out[w] = 0
    if a[w] != 0:
        out[w] = 100 + a[w]
    elif b[w] != 0:
        out[w] = 200 + b[w]
    elif c[w] != 0:
        out[w] = 300 + c[w]


@wp.kernel(enable_backward=False)
def _force_gate(ctrl_valid: wp.array(dtype=wp.int32), gate: wp.array(dtype=wp.int32)):
    w = wp.tid()
    if ctrl_valid[w] != 1:
        gate[w] = 0


@wp.kernel(enable_backward=False)
def _actuation_finish(gate: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32),
                      first: wp.array(dtype=wp.int32)):
    w = wp.tid()
    if gate[w] == 1 and status[w] != 0:
        gate[w] = 0
        if first[w] == 0:
            first[w] = 214


@wp.kernel(enable_backward=False)
def _selected_velocity(gate: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32),
                       vp: wp.array(dtype=scalar.wp_real, ndim=2),
                       vo: wp.array(dtype=scalar.wp_real, ndim=2),
                       target: wp.array(dtype=scalar.wp_real, ndim=2),
                       first: wp.array(dtype=wp.int32)):
    w = wp.tid()
    if gate[w] == 1:
        if status[w] != 0:
            if first[w] == 0:
                first[w] = 202
        else:
            for k in range(3):
                target[w, k] = vp[w, k]
                target[w, k + 3] = vo[w, k]


@wp.kernel(enable_backward=False)
def _live_bias_input(mask: wp.array(dtype=wp.int32), first: wp.array(dtype=wp.int32),
                     state: wp.array(dtype=wp.int64), source: wp.array(dtype=wp.int64),
                     epoch: wp.array(dtype=wp.int64), valid: wp.array(dtype=wp.int32),
                     written: wp.array(dtype=wp.int64),
                     bias: wp.array(dtype=scalar.wp_real, ndim=2),
                     selected: wp.array(dtype=scalar.wp_real, ndim=2),
                     indices: wp.array(dtype=wp.int32),
                     parent_valid: wp.array(dtype=wp.int32),
                     parent_written: wp.array(dtype=wp.int64)):
    w = wp.tid()
    parent_valid[w] = 0
    if mask[w] == 1 and first[w] == 0 and state[w] == source[w]:
        if valid[w] == 1 and written[w] == epoch[w]:
            for k in range(7):
                selected[w, k] = bias[w, indices[k]]
            parent_written[w] = epoch[w]
            parent_valid[w] = 1


@wp.kernel(enable_backward=False)
def _copy_admitted(gate: wp.array(dtype=wp.int32),
                   source: wp.array(dtype=scalar.wp_real, ndim=2),
                   target: wp.array(dtype=scalar.wp_real, ndim=2), width: int):
    w = wp.tid()
    if gate[w] == 1:
        for i in range(width):
            target[w, i] = source[w, i]


@wp.kernel(enable_backward=False)
def _hold_unrequested(mask: wp.array(dtype=wp.int32),
                      before: wp.array(dtype=Any, ndim=2),
                      after: wp.array(dtype=Any, ndim=2), width: int):
    w = wp.tid()
    if mask[w] == 0:
        for i in range(width):
            after[w, i] = before[w, i]


@wp.kernel(enable_backward=False)
def _bootstrap_finish(mask: wp.array(dtype=wp.int32), error: wp.array(dtype=wp.int32),
                      cache_valid: wp.array(dtype=wp.int32),
                      needs_update: wp.array(dtype=wp.int32),
                      first: wp.array(dtype=wp.int32)):
    w = wp.tid()
    if mask[w] == 1:
        if error[w] != 0 or cache_valid[w] != 1:
            mask[w] = 0
            if first[w] == 0:
                first[w] = 202
        else:
            needs_update[w] = 0


class _NoForwardHold:
    """Prepared byte-preserving history for explicitly unrequested lanes.

    Atomic kernels retain their accepted inactive-zero behavior. These copies
    implement the enclosing runtime's *no invocation* branch: only mask==0
    lanes are restored. Selected failures keep their diagnostic work and never
    acquire a valid publication. Every owner/view is declared and frozen.
    """
    def __init__(self, fields, capacity, device):
        self.fields, self.views, self.before = dict(fields), {}, {}
        spans = set()
        for name, a in fields.items():
            if (not isinstance(a, wp.array) or not a.is_contiguous or
                    a.device != device or a.size % capacity != 0):
                raise ValueError("invalid declared per-world hold: " + name)
            span = (int(a.ptr), int(a.size), a.dtype)
            if span in spans:
                continue
            spans.add(span)
            view = a.reshape((capacity, a.size // capacity))
            self.views[name] = view
            self.before[name] = wp.empty_like(view)
        self.tokens = {k: _token(a) for k, a in self.fields.items()}
        self.view_tokens = {k: _token(a) for k, a in self.views.items()}
        self.before_tokens = {k: _token(a) for k, a in self.before.items()}

    def validate(self, fields):
        if fields.keys() != self.fields.keys():
            raise ValueError("no-forward hold field set changed")
        for k, a in fields.items():
            if _token(a) != self.tokens[k]:
                raise ValueError("no-forward hold owner changed: " + k)
        for k, a in self.views.items():
            if _token(a) != self.view_tokens[k] or _token(self.before[k]) != self.before_tokens[k]:
                raise ValueError("no-forward hold view/history changed: " + k)

    def snapshot(self, stream):
        for k, a in self.views.items():
            wp.copy(self.before[k], a, stream=stream)

    def restore_unrequested(self, mask, stream):
        for k, a in self.views.items():
            wp.launch(_hold_unrequested, dim=mask.size,
                inputs=[mask, self.before[k], a, a.shape[1]],
                device=a.device, stream=stream)


def _token(array):
    from .stages import DynamicsStages
    return DynamicsStages._array_binding(array)


def _array(array, shape, dtype, device):
    if (not isinstance(array, wp.array) or array.shape != shape or
            array.dtype != dtype or array.device != device or not array.is_contiguous):
        raise ValueError(f"expected resident contiguous {dtype} {shape} on {device}")
    _token(array)
    return array


class LiveDynamicsAdapter:
    """Live device roots replace the old PRE-record whitelist.

    ``stages`` is already prepared by a model-qualified factory.  Integrators
    retain ``qpos`` / ``qvel`` owners; only their real device state epoch may
    advance.  A runtime must call begin_forward for each distinct forward,
    including a conditional controller refresh, and insert genuine C/S work
    between the exposed position/velocity/bias entries.
    """

    def __init__(self, stages, *, state_epoch, first_error, stream):
        from .publication import commit_bias, commit_smooth
        self._commit_bias, self._commit_smooth = commit_bias, commit_smooth
        self.stages, self.stream = stages, stream
        self.device, self.capacity = wp.get_device(stages.device), stages.capacity
        if not self.device.is_cuda or stream.device != self.device:
            raise ValueError("live dynamics requires the prepared CUDA stream")
        n = self.capacity
        self.state_epoch = _array(state_epoch, (n,), wp.int64, self.device)
        self.first_error = _array(first_error, (n,), wp.int32, self.device)
        self.qpos, self.qvel = stages.qpos, stages.qvel
        self.forward_epoch = wp.zeros(n, dtype=wp.int64, device=self.device)
        self.source_state_epoch = wp.zeros(n, dtype=wp.int64, device=self.device)
        self.active = wp.zeros(n, dtype=wp.int32, device=self.device)
        self.requested_mask = wp.zeros(n, dtype=wp.int32, device=self.device)
        self.valid = {k: wp.zeros(n, dtype=wp.int32, device=self.device)
                      for k in ("position", "velocity", "bias", "smooth")}
        self.written = {k: wp.zeros(n, dtype=wp.int64, device=self.device) for k in self.valid}
        nv = int(stages.model_pack.model.nv)
        self.public = {k: wp.zeros((n, nv), dtype=scalar.wp_real, device=self.device)
                       for k in ("qfrc_bias", "qfrc_smooth", "qacc_smooth")}
        self.smooth_repaired = wp.zeros(n, dtype=wp.int32, device=self.device)
        self.controller_parent_valid = wp.zeros(n, dtype=wp.int32, device=self.device)
        self.controller_parent_written = wp.zeros(n, dtype=wp.int64, device=self.device)
        self._arm_indices = wp.array(stages.model_pack.osc_fields()["qvel_index"],
                                     dtype=wp.int32, device=self.device)
        # One declared preparation delta; primitive RNE continues to write its
        # original scratch.  Both actual consumers receive the public bias.
        self.original_stage_catalog = stages.binding_catalog()
        old_bias = stages.qfrc_bias
        if (stages.acceleration.bias is not old_bias or
                stages.d07.osc.qfrc_bias is not old_bias or
                stages.consumer_inputs["acceleration"]["bias"] is not old_bias or
                stages.consumer_inputs["control"]["qfrc_bias"] is not old_bias):
            raise ValueError("undeclared original bias consumer")
        old_arrays, old_owners, _, old_scalars = stages._binding_state()
        stages.acceleration.bias = self.public["qfrc_bias"]
        stages.d07.osc.qfrc_bias = self.public["qfrc_bias"]
        stages.consumer_inputs["acceleration"]["bias"] = self.public["qfrc_bias"]
        stages.consumer_inputs["control"]["qfrc_bias"] = self.public["qfrc_bias"]
        new_arrays, new_owners, _, new_scalars = stages._binding_state()
        changed = {p for p in set(old_arrays) | set(new_arrays)
                   if old_arrays.get(p) != new_arrays.get(p)}
        if (changed != {"dynamics.acceleration.bias", "dynamics.d07.osc.qfrc_bias",
                        "dynamics.consumer_inputs.acceleration.bias",
                        "dynamics.consumer_inputs.control.qfrc_bias"} or
                old_owners != new_owners or old_scalars != new_scalars):
            raise ValueError("undeclared live dynamics preparation delta")
        stages._freeze_bindings()
        self._hold_specs = {
            "position": {
                "position.body": "qpos xpos xquat xmat xanchor xaxis world_xipos world_ximat status",
                "position.frames": "xipos ximat geom_xpos geom_xmat site_xpos site_xmat status",
                "position.mass": "subtree_com cinert cdof crb compos_status crb_status first_stage first_code result_kind",
                "position.mass.mass": "qm ld inv sqrt_inv solution pivot status warning_count warning_last",
                "position": "first_stage first_code result_kind",
                "transmission": "length moment position_status",
                "": "status_position"},
            "velocity": {"transmission": "velocity velocity_status",
                "": "cvel cdof_dot status_comvel status_velocity",
                "passive": "qfrc status"},
            "bias": {"": "cacc frc qfrc_bias status_rne"},
            "smooth": {"actuation": "force force_pre ctrl_used ctrl_route force_route qfrc_actuator status warn_badctrl_count warn_badctrl_lastinfo warning_number warning_lastinfo",
                "acceleration": "smooth qfrc qacc_smooth status entry_status"}}
        self._holds = {phase: _NoForwardHold(self._hold_fields(phase), n, self.device)
                       for phase in self._hold_specs}
        self._prepared_stages = stages
        self._arrays = self._binding_arrays()
        self._frozen = {k: _token(a) for k, a in self._arrays.items()}
        self._nv = nv
        self._prepared_stream = (id(stream), int(stream.cuda_stream), int(self.device.context))
        # This positional factor workspace also exposes a diagnostic RHS
        # solve.  A single genuine zero RHS is its lawful prepared count;
        # dynamics acceleration is solved separately from the actual RHS.
        with wp.ScopedStream(stream):
            stages.position.mass.mass.count.fill_(1)
            stages.position.mass.mass.rhs.fill_(0.0)

    @classmethod
    def prepare(cls, model_pack, capacity, *, state_epoch, first_error,
                device="cuda:0", stream=None):
        from .stages import DynamicsStages
        stream = wp.get_stream(device) if stream is None else stream
        with wp.ScopedStream(stream):
            stages = DynamicsStages(model_pack, capacity, device=device)
            return cls(stages, state_epoch=state_epoch, first_error=first_error,
                       stream=stream)

    @property
    def position_valid(self):
        return self.valid["position"]

    @property
    def velocity_valid(self):
        return self.valid["velocity"]

    @property
    def bias_valid(self):
        return self.valid["bias"]

    @property
    def smooth_valid(self):
        return self.valid["smooth"]

    def validate_bindings(self):
        self.validate()

    def _binding_arrays(self):
        return {"state_epoch": self.state_epoch, "first_error": self.first_error,
                "qpos": self.qpos, "qvel": self.qvel, "epoch": self.forward_epoch,
                "active": self.active, "requested_mask": self.requested_mask,
                "source_epoch": self.source_state_epoch,
                "smooth_repaired": self.smooth_repaired, **self.public,
                "arm_indices": self._arm_indices,
                "controller_parent_valid": self.controller_parent_valid,
                "controller_parent_written": self.controller_parent_written,
                **{"valid." + k: a for k, a in self.valid.items()},
                **{"written." + k: a for k, a in self.written.items()}}

    def _hold_fields(self, phase):
        out = {}
        for path, names in self._hold_specs[phase].items():
            owner = self.stages
            for part in path.split(".") if path else ():
                owner = getattr(owner, part)
            for name in names.split():
                out[path + "." + name] = getattr(owner, name)
        out["valid"] = self.valid[phase]
        out["written"] = self.written[phase]
        if phase == "smooth":
            out["repaired"] = self.smooth_repaired
        return out

    def validate(self, stream=None):
        if stream is not None and stream is not self.stream:
            raise ValueError("foreign dynamics stream")
        if self.stages is not self._prepared_stages:
            raise ValueError("prepared dynamics owner replaced")
        if (id(self.stream), int(self.stream.cuda_stream), int(self.device.context)) != self._prepared_stream:
            raise ValueError("retained dynamics stream/context changed")
        self.stages.validate_bindings()
        if self.stages.qpos is not self.qpos or self.stages.qvel is not self.qvel:
            raise ValueError("live state owner replaced")
        arrays = self._binding_arrays()
        if arrays.keys() != self._frozen.keys():
            raise ValueError("runtime field set changed")
        for name, a in arrays.items():
            if _token(a) != self._frozen[name]:
                raise ValueError("runtime binding changed: " + name)
        for phase, hold in self._holds.items():
            hold.validate(self._hold_fields(phase))

    def _launch(self, kernel, inputs):
        wp.launch(kernel, dim=self.capacity, inputs=inputs, device=self.device, stream=self.stream)

    def begin_forward(self, mask, stream=None):
        self.validate(stream)
        _array(mask, (self.capacity,), wp.int32, self.device)
        wp.copy(self.requested_mask, mask, stream=self.stream)
        self._launch(_begin, [mask, self.first_error, self.state_epoch,
            self.source_state_epoch, self.forward_epoch, self.active,
            *[self.valid[k] for k in ("position", "velocity", "bias", "smooth")]])

    def _gate(self, target, parent):
        self._launch(_gate, [self.active, self.first_error, self.state_epoch,
            self.source_state_epoch, self.forward_epoch, self.valid[parent],
            self.written[parent], target, 0 if parent == "position_start" else 1])

    def launch_position(self, stream=None):
        self.validate(stream)
        s = self.stages
        self._holds["position"].snapshot(self.stream)
        self._launch(_gate, [self.active, self.first_error, self.state_epoch,
            self.source_state_epoch, self.forward_epoch, self.valid["position"],
            self.written["position"], s.gate_position, 0])
        s.position.launch(stream=self.stream)
        s.transmission.launch_position(stream=self.stream)
        self._launch(_position_status, [s.position.result_kind, s.position.first_code,
            s.transmission.position_status, s.status_position])
        self._launch(_finish, [s.gate_position, s.status_position, self.forward_epoch,
            self.valid["position"], self.written["position"], self.first_error, 1, 210])
        self._holds["position"].restore_unrequested(self.requested_mask, self.stream)

    def launch_velocity(self, stream=None):
        self.validate(stream)
        s = self.stages
        self._holds["velocity"].snapshot(self.stream)
        self._gate(s.gate_velocity, "position")
        s.transmission.launch_velocity(stream=self.stream)
        s._comvel_launch(stream=self.stream)
        s.passive.launch(stream=self.stream)
        self._launch(_velocity_status, [s.transmission.velocity_status, s.status_comvel,
            s.status_passive, s.status_velocity])
        self._launch(_finish, [s.gate_velocity, s.status_velocity, self.forward_epoch,
            self.valid["velocity"], self.written["velocity"], self.first_error, 0, 212])
        self._holds["velocity"].restore_unrequested(self.requested_mask, self.stream)

    def launch_bias(self, *, reference_done, reference_error, reference_written,
                    reference_expected, stream=None):
        self.validate(stream)
        s = self.stages
        self._holds["bias"].snapshot(self.stream)
        for a, dtype in ((reference_done, wp.int32), (reference_error, wp.int32),
                         (reference_written, wp.int64), (reference_expected, wp.int64)):
            _array(a, (self.capacity,), dtype, self.device)
        self._gate(s.gate_bias, "velocity")
        self._launch(_reference_gate, [reference_done, reference_error,
            reference_written, reference_expected, self.forward_epoch,
            s.gate_bias, self.first_error])
        s._rne_launch(stream=self.stream)
        self._launch(self._commit_bias, [self.active, s.gate_bias, s.status_rne,
            self.forward_epoch, self.first_error, s.qfrc_bias, self.public["qfrc_bias"],
            self.valid["bias"], self.written["bias"], self._nv])
        self._holds["bias"].restore_unrequested(self.requested_mask, self.stream)

    def launch_force(self, *, ctrl, ctrl_valid, stream=None):
        self.validate(stream)
        s = self.stages
        self._holds["smooth"].snapshot(self.stream)
        _array(ctrl, (self.capacity, 9), scalar.wp_real, self.device)
        _array(ctrl_valid, (self.capacity,), wp.int32, self.device)
        self._gate(s.actuation_active, "bias")
        self._launch(_force_gate, [ctrl_valid, s.actuation_active])
        # The actuation workspace consumes the actual admitted public ctrl,
        # copied device-to-device to its immutable original input owner.
        self._launch(_copy_admitted, [s.actuation_active, ctrl, s.actuation.ctrl, 9])
        s.actuation.launch(s.transmission.length, s.transmission.moment,
            s.transmission.velocity, s.transmission.velocity_status, stream=self.stream)
        self._launch(_actuation_finish, [s.actuation_active, s.actuation.status,
                                        self.first_error])
        # A failed actuation first cause suppresses the whole downstream route.
        self._gate(s.gate_acceleration, "bias")
        self._launch(_force_gate, [s.actuation_active, s.gate_acceleration])
        s.acceleration.launch(stream=self.stream)
        self._launch(self._commit_smooth, [self.active, s.gate_acceleration,
            s.acceleration.status, self.forward_epoch, self.first_error,
            s.acceleration.qfrc, s.acceleration.qacc_smooth,
            self.public["qfrc_smooth"], self.public["qacc_smooth"],
            self.valid["smooth"], self.written["smooth"], self.smooth_repaired, self._nv])
        self._holds["smooth"].restore_unrequested(self.requested_mask, self.stream)

    def public_arrays(self):
        return dict(self.public)

    def gather_controller_inputs(self, mask, stream=None):
        """Refresh-only original full GPU Jacobian and selected OSC operands."""
        from .stages import _gather_eef_site, EEF_SITE_ID
        self.validate(stream)
        _array(mask, (self.capacity,), wp.int32, self.device)
        s = self.stages
        wp.copy(s.gate_control, mask, stream=self.stream)
        self._launch(_gate, [mask, self.first_error, self.state_epoch,
            self.source_state_epoch, self.forward_epoch, self.valid["bias"],
            self.written["bias"], s.gate_control, 1])
        self._launch(_gather_eef_site, [s.site_xpos_flat, s.site_xmat_flat,
            s.gate_control, s.d07.osc.site_pos, s.d07.osc.site_mat,
            int(s.model_pack.model.nsite), EEF_SITE_ID])
        s.d07.launch(stream=self.stream)
        self._launch(_selected_velocity, [s.gate_control, s.d07.status,
            s.d07.osc.xvelp, s.d07.osc.xvelr, s.control.phys_ee_velocity,
            self.first_error])

    def gather_live_bias(self, mask, stream=None):
        """Every-call bias source; does not refresh the persistent M/J cache."""
        self.validate(stream)
        _array(mask, (self.capacity,), wp.int32, self.device)
        self._launch(_live_bias_input, [mask, self.first_error, self.state_epoch,
            self.source_state_epoch, self.forward_epoch, self.valid["bias"],
            self.written["bias"], self.public["qfrc_bias"],
            self.stages.control.phys_bias, self._arm_indices,
            self.controller_parent_valid, self.controller_parent_written])

    def controller_inputs(self):
        return ControllerInputs(self.controller_parent_valid,
                                self.controller_parent_written, self.forward_epoch)

    def catalog(self):
        self.validate()
        return {"inputs": self._frozen, "original_stage": self.original_stage_catalog,
                "prepared_stage": self.stages.binding_catalog(),
                "public": {k: _token(a) for k, a in self.public.items()},
                "no_forward_hold": {p: h.tokens for p, h in self._holds.items()},
                "clock_owner": "D begin_forward; state_epoch owned by integrator"}


# Assembly-facing name; retains the same preparation and launch API.
RuntimeDynamicsAdapter = LiveDynamicsAdapter


@dataclass(frozen=True)
class ControllerInputs:
    """Selected GPU operands plus actual parent success/currentness tokens."""
    valid: object
    written: object
    expected: object


@wp.kernel(enable_backward=False)
def _plan_control(active: wp.array(dtype=wp.int32), first: wp.array(dtype=wp.int32),
                  needs_update: wp.array(dtype=wp.int32), refresh: wp.array(dtype=wp.int32),
                  admitted: wp.array(dtype=wp.int32), force: int):
    w = wp.tid()
    admitted[w] = 0
    refresh[w] = 0
    if active[w] == 1 and first[w] == 0:
        admitted[w] = 1
        if needs_update[w] == 1 or force == 1:
            refresh[w] = 1


@wp.kernel(enable_backward=False)
def _control_gate(active: wp.array(dtype=wp.int32), parent_valid: wp.array(dtype=wp.int32),
                  written: wp.array(dtype=wp.int64), expected: wp.array(dtype=wp.int64),
                  first: wp.array(dtype=wp.int32), work_error: wp.array(dtype=wp.int32),
                  policy: wp.array(dtype=wp.int32), policy_step: int):
    w = wp.tid()
    work_error[w] = 0
    policy[w] = policy_step
    if parent_valid[w] != 1 or written[w] != expected[w] or first[w] != 0:
        active[w] = 0


@wp.kernel(enable_backward=False)
def _control_commit_mask(active: wp.array(dtype=wp.int32), error: wp.array(dtype=wp.int32),
                         first: wp.array(dtype=wp.int32), commit: wp.array(dtype=wp.int32),
                         event: wp.array(dtype=wp.int64), valid: wp.array(dtype=wp.int32)):
    w = wp.tid()
    commit[w] = 0
    valid[w] = 0
    if active[w] != 1 or first[w] != 0:
        return
    if error[w] != 0:
        first[w] = 202
        return
    if event[w] == wp.int64(9223372036854775807):
        first[w] = 9
        return
    event[w] = event[w] + wp.int64(1)
    commit[w] = 1


@wp.kernel(enable_backward=False)
def _publish_control(commit: wp.array(dtype=wp.int32), event: wp.array(dtype=wp.int64),
                     parent: wp.array(dtype=wp.int64), written: wp.array(dtype=wp.int64),
                     consumed: wp.array(dtype=wp.int64), valid: wp.array(dtype=wp.int32)):
    w = wp.tid()
    if commit[w] == 1:
        consumed[w] = parent[w]
        written[w] = event[w]
        valid[w] = 1


@wp.kernel(enable_backward=False)
def _copy_i32(mask: wp.array(dtype=wp.int32), source: wp.array(dtype=wp.int32),
              target: wp.array(dtype=wp.int32)):
    w = wp.tid()
    if mask[w] == 1:
        target[w] = source[w]


@wp.kernel(enable_backward=False)
def _reset_ctrl_public(mask: wp.array(dtype=wp.int32), first: wp.array(dtype=wp.int32),
                       source: wp.array(dtype=scalar.wp_real, ndim=2),
                       ctrl: wp.array(dtype=scalar.wp_real, ndim=2),
                       valid: wp.array(dtype=wp.int32), written: wp.array(dtype=wp.int64),
                       event: wp.array(dtype=wp.int64)):
    w = wp.tid()
    if mask[w] != 1 or first[w] != 0:
        return
    good = int(1)
    for k in range(9):
        if not wp.isfinite(source[w, k]):
            good = 0
    if good == 0:
        first[w] = 202
        valid[w] = 0
        return
    for k in range(9):
        ctrl[w, k] = source[w, k]
    event[w] = wp.int64(0)
    written[w] = wp.int64(0)
    valid[w] = 1


class RuntimeControllerAdapter:
    """Conditional full-forward refresh, GPU OSC every substep, durable ctrl.

    The callbacks are true enqueue operations on the supplied device mask:
    ``forward_refresh(mask, stream)`` runs the entire model forward with old
    public ctrl; ``gather(mask, stream)`` produces full GPU J and selected
    operands.  Neither callback may upload CPU physics or manufacture ready.
    The physical ``phys_bias`` operand is the current public bias on *every*
    call, independently of the cache refresh decision.
    """

    action_dim = 7

    def __init__(self, controller, *, active, first_error, parents: ControllerInputs,
                 forward_refresh: Callable, gather: Callable, live_bias: Callable, stream):
        self.controller, self.stream = controller, stream
        self.device, self.capacity = controller.device, controller.nworld
        # Exact admitted default OSC action range; feature adapters override cold.
        self.control_limits = (tuple([-1.0] * self.action_dim), tuple([1.0] * self.action_dim))
        if not self.device.is_cuda or stream.device != self.device:
            raise ValueError("controller runtime requires CUDA")
        n = self.capacity
        self.active = _array(active, (n,), wp.int32, self.device)
        self.first_error = _array(first_error, (n,), wp.int32, self.device)
        for a, t in ((parents.valid, wp.int32), (parents.written, wp.int64),
                     (parents.expected, wp.int64)):
            _array(a, (n,), t, self.device)
        if parents.written is parents.expected:
            raise ValueError("parent expected must not mirror its result stamp")
        self.parents = parents
        self.forward_refresh, self.gather, self.live_bias = forward_refresh, gather, live_bias
        self.refresh_mask = wp.zeros(n, dtype=wp.int32, device=self.device)
        self.call_mask = wp.zeros(n, dtype=wp.int32, device=self.device)
        self.commit_mask = wp.zeros(n, dtype=wp.int32, device=self.device)
        self.event = wp.zeros(n, dtype=wp.int64, device=self.device)
        self.valid = wp.zeros(n, dtype=wp.int32, device=self.device)
        self.written = wp.zeros(n, dtype=wp.int64, device=self.device)
        self.consumed_parent = wp.zeros(n, dtype=wp.int64, device=self.device)
        # Public cache/history is the only persistent state.  Each call starts
        # with D2D copies into work; failed primitive changes stay in work and
        # can be diagnosed without corrupting the next call's history.
        names = tuple(k for k in controller.WIDTHS if not k.startswith("phys_") and k not in
                      ("action", "q0", "gains", "ctrl_range"))
        self.public = {k: wp.zeros_like(getattr(controller, k)) for k in names}
        self.public_int = {k: wp.zeros_like(getattr(controller, k))
                           for k in ("cache_valid", "needs_update")}
        self.public_int["needs_update"].fill_(1)
        self.public["ctrl"].fill_(0.0)
        self._physical_names = ("phys_mass", "phys_jacobian", "phys_q", "phys_qvel",
            "phys_bias", "phys_ee_pos", "phys_ee_rot", "phys_ee_velocity")
        self._fixed = {k: _token(getattr(controller, k))
                       for k in (*controller.WIDTHS, *self._physical_names,
                                 "active", "policy_step", "error", "cache_valid",
                                 "needs_update", "ranks", "quat_cast_wxyz", "quat_valid")}
        self._owner = {k: _token(a) for k, a in {**self.public, **self.public_int}.items()}
        self._runtime_arrays = self._binding_arrays()
        self._runtime_tokens = {k: _token(a) for k, a in self._runtime_arrays.items()}
        self._callbacks = (forward_refresh, gather, live_bias)
        self._prepared_controller = controller
        self._config = (controller.nworld, str(controller.device), controller.uncouple,
                        controller.diagnostics, id(stream), int(stream.cuda_stream),
                        int(self.device.context))

    def _binding_arrays(self):
        return {"active": self.active, "first_error": self.first_error,
                "parent_valid": self.parents.valid, "parent_written": self.parents.written,
                "parent_expected": self.parents.expected, "refresh_mask": self.refresh_mask,
                "call_mask": self.call_mask, "commit_mask": self.commit_mask,
                "event": self.event, "valid": self.valid, "written": self.written,
                "consumed_parent": self.consumed_parent}

    def stage_action(self, action):
        """The only host action upload; action boundary, outside CUDA capture."""
        import numpy as np
        self.validate()
        action = np.asarray(action)
        if action.dtype != np.float64 or action.shape != (self.capacity, 7):
            raise ValueError("action must be float64 (physical_worlds,7)")
        if not np.isfinite(action).all():
            raise ValueError("nonfinite policy action")
        with wp.ScopedStream(self.stream):
            self.controller.action.assign(np.ascontiguousarray(action))

    def initialize_ctrl_from_reset(self, mask, initial_ctrl, *, stream=None):
        """Explicit native-data-reset root, not a computed controller result.

        Enables the original first full forward with reset ctrl before the
        first OSC call. Do not invoke on set_init_state: that entry preserves
        the preceding ctrl/cache history. The caller records reset provenance.
        """
        self.validate(stream)
        _array(mask, (self.capacity,), wp.int32, self.device)
        _array(initial_ctrl, (self.capacity, 9), scalar.wp_real, self.device)
        wp.launch(_reset_ctrl_public, dim=self.capacity,
            inputs=[mask, self.first_error, initial_ctrl, self.public["ctrl"],
                    self.valid, self.written, self.event],
            device=self.device, stream=self.stream)

    def reset_selected(self, mask, *, clear_gripper=True,
                       update_initial_joints=True, stream=None):
        """Explicit controller reset only; set_init_state must not call this.

        Requires fresh current physical inputs. Shared first errors are reset
        only by the outer explicit data/episode reset protocol, never here.
        """
        self.validate(stream)
        _array(mask, (self.capacity,), wp.int32, self.device)
        c, st, n = self.controller, self.stream, self.capacity
        for k, a in self.public.items():
            wp.copy(getattr(c, k), a, stream=st)
        for k, a in self.public_int.items():
            wp.copy(getattr(c, k), a, stream=st)
        wp.copy(self.call_mask, mask, stream=st)
        wp.launch(_control_gate, dim=n,
            inputs=[self.call_mask, self.parents.valid, self.parents.written,
                    self.parents.expected, self.first_error, c.error, c.policy_step, 0],
            device=self.device, stream=st)
        c.reset_selected(self.call_mask, clear_gripper=clear_gripper,
                         update_initial_joints=update_initial_joints, stream=st)
        for k, a in self.public.items():
            wp.launch(_copy_admitted, dim=n, inputs=[self.call_mask, getattr(c, k), a,
                a.shape[1]], device=self.device, stream=st)
        for k, a in self.public_int.items():
            wp.launch(_copy_i32, dim=n, inputs=[self.call_mask, getattr(c, k), a],
                      device=self.device, stream=st)

    def bootstrap_from_current_physics(self, mask, *, stream=None):
        """Original constructor boundary, after arm init and before gripper init.

        Both native constructor forwards are explicit true resident forwards.
        No OSC control call occurs here. The second forward supplies the cache,
        initial q and goals; new_update is false after successful construction.
        set_init_state must retain this history and must not call this method.
        """
        self.validate(stream)
        _array(mask, (self.capacity,), wp.int32, self.device)
        c, st, n = self.controller, self.stream, self.capacity
        self.forward_refresh(mask, st)
        self.forward_refresh(mask, st)
        self.gather(mask, st)
        self.live_bias(mask, st)
        wp.copy(self.call_mask, mask, stream=st)
        wp.launch(_control_gate, dim=n, inputs=[self.call_mask, self.parents.valid,
            self.parents.written, self.parents.expected, self.first_error,
            c.error, c.policy_step, 0], device=self.device, stream=st)
        c.reset_selected(self.call_mask, clear_gripper=True,
                         update_initial_joints=True, stream=st)
        # Only the original refresh kernel is called, without fused OSC/grip.
        import sys
        module = sys.modules[type(c).__module__]
        fields = ("active", "needs_update", "cache_valid", "error",
            "phys_mass", "phys_jacobian", "phys_q", "phys_qvel", "phys_ee_pos",
            "phys_ee_rot", "phys_ee_velocity", "mass", "jacobian", "q", "qvel",
            "ee_pos", "ee_rot", "ee_velocity")
        wp.copy(c.active, self.call_mask, stream=st)
        wp.launch(module.refresh_cache, dim=n, inputs=[getattr(c, k) for k in fields],
                  device=self.device, stream=st)
        wp.launch(_bootstrap_finish, dim=n, inputs=[self.call_mask, c.error,
            c.cache_valid, c.needs_update, self.first_error], device=self.device, stream=st)
        for k, a in self.public.items():
            wp.launch(_copy_admitted, dim=n, inputs=[self.call_mask, getattr(c, k), a,
                a.shape[1]], device=self.device, stream=st)
        for k, a in self.public_int.items():
            wp.launch(_copy_i32, dim=n, inputs=[self.call_mask, getattr(c, k), a],
                      device=self.device, stream=st)

    def validate(self, stream=None):
        if stream is not None and stream is not self.stream:
            raise ValueError("foreign controller stream")
        if self.controller is not self._prepared_controller:
            raise ValueError("prepared controller owner replaced")
        c = self.controller
        if (c.nworld, str(c.device), c.uncouple, c.diagnostics,
                id(self.stream), int(self.stream.cuda_stream), int(self.device.context)) != self._config:
            raise ValueError("controller prepared mode/stream changed")
        if (self.forward_refresh, self.gather, self.live_bias) != self._callbacks:
            raise ValueError("prepared controller callbacks replaced")
        arrays = self._binding_arrays()
        if arrays.keys() != self._runtime_tokens.keys():
            raise ValueError("controller runtime field set changed")
        for k, a in arrays.items():
            if _token(a) != self._runtime_tokens[k]:
                raise ValueError("controller epoch/input owner replaced: " + k)
        for k, tok in self._fixed.items():
            if _token(getattr(self.controller, k)) != tok:
                raise ValueError("controller actual input/owner replaced: " + k)
        for k, tok in self._owner.items():
            if _token((self.public if k in self.public else self.public_int)[k]) != tok:
                raise ValueError("controller public owner replaced: " + k)

    def enqueue(self, *, policy_step=False, force_refresh=False, stream=None):
        self.validate(stream)
        c, n, st = self.controller, self.capacity, self.stream
        launch = lambda kernel, inputs: wp.launch(kernel, dim=n, inputs=inputs,
                                                  device=self.device, stream=st)
        launch(_plan_control, [self.active, self.first_error,
            self.public_int["needs_update"], self.refresh_mask, self.call_mask,
            int(bool(force_refresh))])
        self.forward_refresh(self.refresh_mask, st)
        self.gather(self.refresh_mask, st)
        self.live_bias(self.call_mask, st)
        for k, a in self.public.items():
            wp.copy(getattr(c, k), a, stream=st)
        for k, a in self.public_int.items():
            wp.copy(getattr(c, k), a, stream=st)
        wp.copy(c.active, self.call_mask, stream=st)
        # The original refresh_cache reads needs_update, not a host bool.
        wp.copy(c.needs_update, self.refresh_mask, stream=st)
        launch(_control_gate, [c.active, self.parents.valid, self.parents.written,
            self.parents.expected, self.first_error, c.error, c.policy_step,
            int(bool(policy_step))])
        c.launch(stream=st)
        launch(_control_commit_mask, [c.active, c.error, self.first_error,
            self.commit_mask, self.event, self.valid])
        for k, a in self.public.items():
            launch(_copy_admitted, [self.commit_mask, getattr(c, k), a, a.shape[1]])
        for k, a in self.public_int.items():
            launch(_copy_i32, [self.commit_mask, getattr(c, k), a])
        # All persistent cache/goal/grip/ctrl data commits finish before the
        # publication token.  The actuation consumer uses public['ctrl'].
        launch(_publish_control, [self.commit_mask, self.event, self.parents.expected,
            self.written, self.consumed_parent, self.valid])

    def public_arrays(self):
        return {**self.public, **self.public_int, "valid": self.valid,
                "written": self.written, "event": self.event,
                "consumed_parent": self.consumed_parent}

    def catalog(self):
        self.validate()
        return {"work_inputs": self._fixed, "public_owners": self._owner,
                "full_forward_callback": repr(self.forward_refresh),
                "gather_callback": repr(self.gather),
                "ctrl_owner": "controller", "cache_owner": "controller"}


@wp.kernel(enable_backward=False)
def _post_gate(active: wp.array(dtype=wp.int32), first: wp.array(dtype=wp.int32),
               dvalid: wp.array(dtype=wp.int32), nvalid: wp.array(dtype=wp.int32),
               nerror: wp.array(dtype=wp.int32), nwritten: wp.array(dtype=wp.int64),
               nattempt: wp.array(dtype=wp.int64), expected: wp.array(dtype=wp.int64),
               state: wp.array(dtype=wp.int64), consumed_state: wp.array(dtype=wp.int64),
               forward: wp.array(dtype=wp.int64), consumed_forward: wp.array(dtype=wp.int64),
               gate: wp.array(dtype=wp.int32)):
    w = wp.tid()
    gate[w] = 0
    if active[w] != 1 or first[w] != 0 or dvalid[w] != 1:
        return
    if (nvalid[w] != 1 or nerror[w] != 0 or expected[w] <= wp.int64(0) or
            nwritten[w] != expected[w] or nattempt[w] != expected[w] or
            state[w] != consumed_state[w] or forward[w] != consumed_forward[w]):
        return
    gate[w] = 1


@wp.kernel(enable_backward=False)
def _post_commit(gate: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32),
                 sensor_status: wp.array(dtype=wp.int32), first: wp.array(dtype=wp.int32),
                 epoch: wp.array(dtype=wp.int64), ext: wp.array(dtype=scalar.wp_real, ndim=2),
                 intr: wp.array(dtype=scalar.wp_real, ndim=2), sensor: wp.array(dtype=scalar.wp_real, ndim=2),
                 pub_ext: wp.array(dtype=scalar.wp_real, ndim=2), pub_int: wp.array(dtype=scalar.wp_real, ndim=2),
                 pub_sensor: wp.array(dtype=scalar.wp_real, ndim=2), valid: wp.array(dtype=wp.int32),
                 written: wp.array(dtype=wp.int64), body_width: int, sensor_width: int):
    w = wp.tid()
    valid[w] = 0
    if gate[w] != 1 or first[w] != 0:
        return
    if status[w] != 0 or sensor_status[w] != 0:
        # Composite aggregate status; does not claim exact kernel-first cause.
        first[w] = 215
        return
    for k in range(body_width):
        pub_ext[w, k] = ext[w, k]
        pub_int[w, k] = intr[w, k]
    for k in range(sensor_width):
        pub_sensor[w, k] = sensor[w, k]
    written[w] = epoch[w]
    valid[w] = 1


class RuntimePostAdapter:
    """Post/sensor source aliases and strict actual-N publication lineage.

    Caller supplies the prepared original post workspace and all 21 device
    inputs. ``capture_before_constraint`` runs immediately before each actual
    N enqueue, including in a Graph. It snapshots the independent *planned*
    N expected value and current D/state generations. Merely preparing this
    object does not grant post admission.
    """
    INPUTS = ("qvel", "qacc", "cinert", "cvel", "cdof", "cdof_dot", "xipos",
              "subtree_com", "site_xpos", "site_xmat", "xfrc_applied", "efc_force",
              "ncon", "nefc", "ne", "contact_dim", "contact_efc_address",
              "contact_geom1", "contact_geom2", "contact_frame", "contact_pos")

    def __init__(self, post, *, sources: Mapping, active, first_error, d_valid,
                 state_epoch, forward_epoch, n_valid, n_error, n_written,
                 n_attempt, n_expected, stream):
        self.post, self.stream = post, stream
        self.device, self.capacity = wp.get_device(post.device), post.capacity
        if not self.device.is_cuda or stream.device != self.device:
            raise ValueError("post requires actual CUDA prepared owner")
        if set(sources) != set(self.INPUTS):
            raise ValueError("post requires exactly its declared 21 device source inputs")
        self._sources, self._views = dict(sources), {}
        for name, source in self._sources.items():
            target = getattr(post, name)
            if (source.device != self.device or source.dtype != target.dtype or
                    source.size != target.size or not source.is_contiguous):
                raise ValueError("post source capacity/type mismatch: " + name)
            _token(source)
            self._views[name] = source.reshape(target.shape)
            setattr(post, name, self._views[name])
        post.sensor.com = post.subtree_com
        post.sensor.site_pos = post.site_xpos
        post.sensor.site_mat = post.site_xmat
        n = self.capacity
        for name, value, dtype in (
            ("active", active, wp.int32), ("first_error", first_error, wp.int32),
            ("d_valid", d_valid, wp.int32), ("state_epoch", state_epoch, wp.int64),
            ("forward_epoch", forward_epoch, wp.int64), ("n_valid", n_valid, wp.int32),
            ("n_error", n_error, wp.int32), ("n_written", n_written, wp.int64),
            ("n_attempt", n_attempt, wp.int64), ("n_expected", n_expected, wp.int64)):
            setattr(self, name, _array(value, (n,), dtype, self.device))
        if n_expected is n_written:
            raise ValueError("N expected cannot be its public written stamp")
        self.expected_consumed = wp.zeros(n, dtype=wp.int64, device=self.device)
        self.state_consumed = wp.zeros(n, dtype=wp.int64, device=self.device)
        self.forward_consumed = wp.zeros(n, dtype=wp.int64, device=self.device)
        self.valid = wp.zeros(n, dtype=wp.int32, device=self.device)
        self.written = wp.zeros(n, dtype=wp.int64, device=self.device)
        self._int_view = post.cfrc_int.reshape((n, int(post.cfrc_int.size // n)))
        self.public = {"cfrc_ext": wp.zeros_like(post.cfrc_ext),
                       "cfrc_int": wp.zeros_like(self._int_view),
                       "sensor": wp.zeros_like(post.sensor.result)}
        self._prepared_post = post
        self._source_tokens = {k: _token(a) for k, a in self._sources.items()}
        self._view_tokens = {k: _token(a) for k, a in self._views.items()}
        self._output_tokens = {k: _token(a) for k, a in self.public.items()}
        self._int_token = _token(self._int_view)
        self._work_tokens = {k: _token(v) for k, v in vars(post).items()
                             if isinstance(v, wp.array) and k not in self.INPUTS}
        self._sensor_tokens = {k: _token(v) for k, v in vars(post.sensor).items()
                               if isinstance(v, wp.array)}
        self._control_names = ("active", "first_error", "d_valid", "state_epoch",
            "forward_epoch", "n_valid", "n_error", "n_written", "n_attempt", "n_expected",
            "expected_consumed", "state_consumed", "forward_consumed", "valid", "written")
        self._control_tokens = {k: _token(getattr(self, k)) for k in self._control_names}

    def validate(self, stream=None):
        if stream is not None and stream is not self.stream:
            raise ValueError("foreign post stream")
        if self.post is not self._prepared_post:
            raise ValueError("prepared post owner replaced")
        for k, token in self._source_tokens.items():
            if (_token(self._sources[k]) != token or
                    _token(getattr(self.post, k)) != self._view_tokens[k]):
                raise ValueError("post actual input owner replaced: " + k)
        if (self.post.sensor.cfrc is not self.post.cfrc_int or
                self.post.sensor.com is not self.post.subtree_com or
                self.post.sensor.site_pos is not self.post.site_xpos or
                self.post.sensor.site_mat is not self.post.site_xmat or
                self.post.sensor.active is not self.post.active):
            raise ValueError("post actual sensor consumer replaced")
        if _token(self._int_view) != self._int_token:
            raise ValueError("post int3D commit view replaced")
        for k, tok in self._work_tokens.items():
            if _token(getattr(self.post, k)) != tok:
                raise ValueError("post actual work/static owner replaced: " + k)
        for k, tok in self._sensor_tokens.items():
            if _token(getattr(self.post.sensor, k)) != tok:
                raise ValueError("post actual sensor owner replaced: " + k)
        for k, tok in self._output_tokens.items():
            if _token(self.public[k]) != tok:
                raise ValueError("post public owner replaced")
        for k, tok in self._control_tokens.items():
            if _token(getattr(self, k)) != tok:
                raise ValueError("post source/expected owner replaced: " + k)

    def capture_before_constraint(self, stream=None):
        self.validate(stream)
        wp.copy(self.expected_consumed, self.n_expected, stream=self.stream)
        wp.copy(self.state_consumed, self.state_epoch, stream=self.stream)
        wp.copy(self.forward_consumed, self.forward_epoch, stream=self.stream)

    def enqueue(self, stream=None):
        self.validate(stream)
        p, n = self.post, self.capacity
        wp.launch(_post_gate, dim=n, inputs=[self.active, self.first_error,
            self.d_valid, self.n_valid, self.n_error, self.n_written, self.n_attempt,
            self.expected_consumed, self.state_epoch, self.state_consumed,
            self.forward_epoch, self.forward_consumed, p.active],
            device=self.device, stream=self.stream)
        p.launch(stream=self.stream)
        wp.launch(_post_commit, dim=n, inputs=[p.active, p.status, p.sensor.status,
            self.first_error, self.expected_consumed, p.cfrc_ext, self._int_view,
            p.sensor.result, self.public["cfrc_ext"], self.public["cfrc_int"],
            self.public["sensor"], self.valid, self.written,
            p.cfrc_ext.shape[1], p.sensor.result.shape[1]],
            device=self.device, stream=self.stream)

    def catalog(self):
        self.validate()
        return {"sources": self._source_tokens, "views": self._view_tokens,
                "public": self._output_tokens, "clocks": self._control_tokens,
                "int_commit_view": self._int_token,
                "cause_label": "original RNEPost composite aggregate (includes sensor)"}
