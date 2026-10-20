"""R1/R2 world-owned recovery composition; native reset grid stays separate.

Protocol arrays are bound once after runtime construction. The workspace's
root reset launches remain unchanged; only its final warning stage shares a
launch with the following runtime epoch protocol. Conditional forward and
event snapshots retain their original producer/consumer boundaries.
"""
import libero.gpu.precision as scalar
import warp as wp
from libero.gpu.host.runtime.dynamics import recovery as data
from libero.gpu.host.runtime.step import recovery_runtime_r02 as runtime
from libero.gpu.host.runtime.step import recovery_event_runtime as event_runtime
from libero.gpu.host.assembly.compose.backup import FusedEventRecoveringRuntime
from libero.gpu.host.assembly.compose.compose import compose, identity_mapping

admit_and_select = compose("recovery_admit_select", (
    (runtime._current_acceleration_admission, identity_mapping(runtime._current_acceleration_admission)),
    (event_runtime._select, identity_mapping(event_runtime._select,
        expected="event_expected", state="event_state")),
))
warning_and_protocol = compose("recovery_warning_protocol", (
    (data._warning_after_reset, identity_mapping(data._warning_after_reset)),
    (runtime._reset_protocol, identity_mapping(runtime._reset_protocol, mask="reset")),
))


class FusedDataRecoveryWorkspace(data.DataRecoveryWorkspace):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fusion_protocol = None

    def bind_protocol(self, owner):
        if self._fusion_protocol is not None:
            raise ValueError("recovery protocol already cold-bound")
        if owner.recovery is not self or self.N != owner.euler.capacity:
            raise ValueError("matching native recovery runtime world capacity required")
        arrays = (owner.first_error, owner.state_epoch, owner.expected_state_epoch,
                  owner.forward_valid, owner.valid, owner.reset_events)
        self._fusion_protocol = arrays
        self._fusion_protocol_tokens = tuple(data._token(a) for a in arrays)
        self._fusion_protocol_owner = owner

    def validate_protocol(self, owner):
        arrays = (owner.first_error, owner.state_epoch, owner.expected_state_epoch,
                  owner.forward_valid, owner.valid, owner.reset_events)
        if owner is not self._fusion_protocol_owner or any(a is not b for a,b in zip(arrays,self._fusion_protocol)):
            raise ValueError("prepared recovery protocol owner changed")
        if tuple(data._token(a) for a in arrays) != self._fusion_protocol_tokens:
            raise ValueError("prepared recovery protocol metadata changed")

    def _launch(self, field, warning, stream):
        self.validate_bindings()
        if self._fusion_protocol is not None:
            self.validate_protocol(self._fusion_protocol_owner)
        stream = self.stream if stream is None else stream
        wp.launch(data._check, self.N, inputs=[self.views[field],self.active,self.reset_mask,
            self.reason,self.bad_index,self.source_error,self.catalog[field]["width"],warning],
            device=self.device,stream=stream)
        kernels={scalar.wp_real:data._reset_f64,wp.int32:data._reset_i32,wp.int64:data._reset_i64}
        for name,view in self.views.items():
            width=self.catalog[name]["width"]
            wp.launch(kernels[view.dtype],(self.N,width),inputs=[view,self.defaults[name],self.reset_mask,width],
                      device=self.device,stream=stream)
        if self._fusion_protocol is None:
            wp.launch(data._warning_after_reset,self.N,inputs=[self.warning_number,self.warning_lastinfo,
                self.reset_mask,self.reason,self.bad_index],device=self.device,stream=stream)
        else:
            wp.launch(warning_and_protocol,self.N,inputs=[self.warning_number,self.warning_lastinfo,
                self.reset_mask,self.reason,self.bad_index,*self._fusion_protocol],device=self.device,stream=stream)
        return self.reset_mask


class RecoveryRuntimeMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(self.recovery,FusedDataRecoveryWorkspace):
            raise TypeError("R2 requires cold fused recovery workspace")
        self.recovery.bind_protocol(self)

    def _recovery_validate(self):
        super()._recovery_validate()
        self.recovery.validate_protocol(self)

    def _protocol(self, mask):
        # The immediately preceding check has already executed the composed
        # warning and epoch protocol after all parallel native roots reset.
        if mask is not self.recovery.reset_mask:
            raise ValueError("foreign recovery protocol mask")
        self.recovery.validate_protocol(self)

    def _admit_current_acceleration(self):
        if not self.capture_events:
            return runtime.CurrentWrittenRecoveringRuntime._admit_current_acceleration(self)
        if self._event_owner_tokens()!=self._event_tokens or self.forward.n.final_event is not self._event_solver:
            raise ValueError("event retained owner drift")
        n=self.forward.n; event=n.final_event
        values=dict(intent=self.intent,first=self.first_error,forward_valid=self.forward_valid,
            forward_written=self.forward_written,state_epoch=self.state_epoch,n_valid=n.valid,
            n_written=n.n_written,n_expected=self.forward.expected_newton,active=self.recovery.active,
            qa=self.euler.qacc,candidate=event.candidate,candidate_attempt=event.candidate_attempt,
            completed=event.candidate_completed,origin=event.candidate_origin,stage=event.source_stage,
            attempt=n.n_attempt,written=n.n_written,event_expected=self.forward.expected_newton,
            event_state=self.state_epoch,forward=self.forward.d.forward_epoch,count=self.event_count,
            slot=self.event_slot,overflow=self.event_overflow,metadata=self.event_metadata,
            capacity=8,nv=self.runtime_shape.nv)
        wp.launch(admit_and_select,self.euler.capacity,inputs=[values[name] for name in admit_and_select.fusion_argument_names],
                  device=self.stream.device,stream=self.stream)
        self._copy_event(self.event_before)


def runtime_type(base=FusedEventRecoveringRuntime):
    """Cold composition point for independent Euler/event-copy mixins."""
    return type("FusedRecoveryRuntime",(RecoveryRuntimeMixin,base),{"__module__":__name__})


FusedRecoveryRuntime=runtime_type()
