"""Separate shape-generic diagnostic scheduler; original numeric operations retained.

No native/CPU forward outputs enter this module. Component adapters own
their source/phase gates. The integration gate additionally requires the
actual current forward publication before advancing the persistent state.
"""
import libero.gpu.precision as scalar
import warp as wp

from .schedule import enqueue_substep, substeps_per_action
from .shape_schema import RuntimeShape

wp.set_module_options({'enable_backward': False, 'fuse_fp': True, 'fast_math': False})


@wp.kernel(enable_backward=False)
def _integration_gate(intent: wp.array(dtype=wp.int32),
                      epoch: wp.array(dtype=wp.int64),
                      forward_valid: wp.array(dtype=wp.int32),
                      forward_written: wp.array(dtype=wp.int64),
                      forward_error: wp.array(dtype=wp.int32),
                      first_error: wp.array(dtype=wp.int32),
                      qpos: wp.array(dtype=scalar.wp_real, ndim=2),
                      qvel: wp.array(dtype=scalar.wp_real, ndim=2),
                      qacc: wp.array(dtype=scalar.wp_real, ndim=2),
                      active: wp.array(dtype=wp.int32), nq: int, nv: int):
    w = wp.tid()
    active[w] = 0
    if intent[w] == 0 or first_error[w] != 0:
        return
    if intent[w] != 1:
        first_error[w] = 1
        return
    if epoch[w] == wp.int64(9223372036854775807):
        first_error[w] = 9
        return
    if forward_error[w] != 0:
        first_error[w] = forward_error[w]
        return
    if forward_valid[w] != 1 or forward_written[w] != epoch[w]:
        return
    # CPU_FLOW mj_checkPos/Vel/Acc domain. Warning-recovery full-forward
    # paths are not replaced by an apparently successful finite result.
    for i in range(nq):
        if not wp.isfinite(qpos[w, i]) or wp.abs(qpos[w, i]) > scalar.wp_real(1.0e10):
            first_error[w] = 2
            return
    for i in range(nv):
        if not wp.isfinite(qvel[w, i]) or wp.abs(qvel[w, i]) > scalar.wp_real(1.0e10):
            first_error[w] = 3
            return
        if not wp.isfinite(qacc[w, i]) or wp.abs(qacc[w, i]) > scalar.wp_real(1.0e10):
            first_error[w] = 4
            return
    active[w] = 1


@wp.kernel(enable_backward=False)
def _publish_step(active: wp.array(dtype=wp.int32),
                  status: wp.array(dtype=wp.int32),
                  epoch: wp.array(dtype=wp.int64),
                  written: wp.array(dtype=wp.int64),
                  valid: wp.array(dtype=wp.int32),
                  first_error: wp.array(dtype=wp.int32),
                  forward_valid: wp.array(dtype=wp.int32)):
    w = wp.tid()
    valid[w] = 0
    if active[w] != 1:
        return
    if status[w] != 0:
        if first_error[w] == 0:
            first_error[w] = 100 + status[w]
        return
    epoch[w] = epoch[w] + wp.int64(1)
    written[w] = epoch[w]
    valid[w] = 1
    # The arrays remain resident and retain their old scratch epoch, while
    # the integrated state has advanced. Observables may read this explicit
    # mixed-age state; future consumers may not claim a current forward.
    forward_valid[w] = 0


@wp.kernel(enable_backward=False)
def _advance_expected_state(intent: wp.array(dtype=wp.int32), expected: wp.array(dtype=wp.int64)):
    w = wp.tid()
    # An independent planned target, not a read of successful output stamps.
    if intent[w] == 1 and expected[w] < wp.int64(9223372036854775807):
        expected[w] = expected[w] + wp.int64(1)


def _token(a):
    return (id(a), int(a.ptr), str(a.dtype), tuple(a.shape), tuple(a.strides),
            str(a.device), int(a.capacity))


class ShapeResidentStepRuntime:
    """One exact cold native shape/stream; stable Euler roots are the live state.

    Initial host state transfer is the reset boundary. Euler.upload is never
    called by a substep. Forward/controller bind these same persistent roots
    and public force/acceleration owners during preparation.
    """
    def __init__(self, *, runtime_shape, euler, forward, controller, intent, state_epoch,
                 forward_valid, forward_written, forward_error, stream,
                 observe, validate_components, control_timestep=0.05,
                 outer_model_timestep=0.002, first_error=None, expected_state_epoch=None,
                 integration_tape=None):
        if not isinstance(runtime_shape, RuntimeShape):
            raise TypeError('cold actual native dimensions required')
        if (euler.nq,euler.nv,euler.nM,euler.njnt)!=(runtime_shape.nq,runtime_shape.nv,runtime_shape.nM,runtime_shape.njnt):
            raise ValueError('actual integrator dimensions differ')
        self.runtime_shape=runtime_shape
        self._runtime_shape=runtime_shape
        if not callable(observe) or not callable(validate_components):
            raise TypeError('explicit observable and component hooks required')
        self.euler, self.forward, self.controller = euler, forward, controller
        self.stream, self.intent, self.state_epoch = stream, intent, state_epoch
        self.forward_valid, self.forward_written = forward_valid, forward_written
        self.forward_error = forward_error
        self.expected_state_epoch = expected_state_epoch
        self.integration_tape = integration_tape
        self._integration_tape = integration_tape
        self.observe, self._validate_components = observe, validate_components
        self.control_timestep = float(control_timestep)
        self.outer_model_timestep = float(outer_model_timestep)
        self.substeps = substeps_per_action(self.control_timestep, self.outer_model_timestep)
        n = euler.capacity
        for a, dtype in ((intent, wp.int32), (state_epoch, wp.int64),
                         (forward_valid, wp.int32), (forward_written, wp.int64),
                         (forward_error, wp.int32)):
            if a.dtype != dtype or tuple(a.shape) != (n,) or a.device != euler.active.device:
                raise TypeError('runtime publication dtype/extent/device mismatch')
        if state_epoch.ptr == forward_written.ptr:
            raise ValueError('state epoch and forward publisher must be independent owners')
        self.first_error = (wp.zeros(n, dtype=wp.int32, device=euler.device)
                            if first_error is None else first_error)
        if self.first_error.dtype != wp.int32 or self.first_error.shape != (n,) or self.first_error.device != euler.active.device:
            raise TypeError('runtime sticky error schema mismatch')
        self.valid = wp.zeros(n, dtype=wp.int32, device=euler.device)
        self.written = wp.zeros(n, dtype=wp.int64, device=euler.device)
        self._step_owners = {name:getattr(self,name) for name in ('first_error','valid','written')}
        self._step_tokens = {name:_token(a) for name,a in self._step_owners.items()}
        self._arrays = {name: getattr(euler, name) for name in (
            'qpos', 'qvel', 'time', 'qM', 'qacc', 'qfrc_smooth',
            'qfrc_constraint', 'active', 'qH', 'qHDiagInv',
            'ctrl', 'qacc_warmstart', 'sensordata')}
        self._tokens = {name: _token(a) for name, a in self._arrays.items()}
        self._publications = (intent, state_epoch, forward_valid, forward_written, forward_error)
        self._publication_tokens = tuple(_token(a) for a in self._publications)
        self._expected_token = None if expected_state_epoch is None else _token(expected_state_epoch)
        self.validate()

    def validate(self):
        if self.runtime_shape is not self._runtime_shape or (self.euler.nq,self.euler.nv,self.euler.nM,self.euler.njnt)!=(self.runtime_shape.nq,self.runtime_shape.nv,self.runtime_shape.nM,self.runtime_shape.njnt):
            raise ValueError('cold native dimensions changed')
        for name,a in self._step_owners.items():
            if getattr(self,name) is not a or _token(a) != self._step_tokens[name]:
                raise ValueError('step publication allocation changed: '+name)
        for name, a in self._arrays.items():
            if getattr(self.euler, name) is not a or _token(a) != self._tokens[name]:
                raise ValueError('integrator binding changed: ' + name)
        current = (self.intent, self.state_epoch, self.forward_valid,
                   self.forward_written, self.forward_error)
        if tuple(_token(a) for a in current) != self._publication_tokens:
            raise ValueError('runtime publication owner changed')
        if self.expected_state_epoch is not None and _token(self.expected_state_epoch) != self._expected_token:
            raise ValueError('independent state plan owner changed')
        self._validate_components()
        if self.integration_tape is not self._integration_tape:
            raise ValueError('prepared input tape replaced')

    def _integrate(self):
        e = self.euler
        wp.launch(_integration_gate, dim=e.capacity, inputs=[self.intent,
            self.state_epoch, self.forward_valid, self.forward_written,
            self.forward_error, self.first_error, e.qpos, e.qvel, e.qacc,
            e.active,e.nq,e.nv], device=e.device, stream=self.stream)
        if self.integration_tape is not None:
            self.integration_tape.capture()
        e.launch(stream=self.stream)
        wp.launch(_publish_step, dim=e.capacity, inputs=[e.active, e.status,
            self.state_epoch, self.written, self.valid, self.first_error,
            self.forward_valid], device=e.device, stream=self.stream)
        if self.expected_state_epoch is not None:
            wp.launch(_advance_expected_state, dim=e.capacity,
                      inputs=[self.intent,self.expected_state_epoch],device=e.device,stream=self.stream)

    def enqueue_substep(self, *, policy_step=False, force_refresh=False):
        self.validate()
        enqueue_substep(self.forward, self.controller, self._integrate,
            self.observe, self.intent, self.stream, policy_step=policy_step,
            force_refresh=force_refresh)

    def enqueue_physics_step(self):
        """Public mj_step normal path with current ctrl, without robot control.

        This is B07's first executable target. B08 additionally uses the
        environment forward and the real controller schedule above.
        """
        self.validate()
        self.forward.enqueue(self.intent, self.stream)
        self._integrate()

    def enqueue_action(self):
        for i in range(self.substeps):
            self.enqueue_substep(policy_step=(i == 0))

    def public_arrays(self):
        return {**self._arrays, 'state_epoch': self.state_epoch,
                'step_valid': self.valid, 'step_written': self.written,
                'first_error': self.first_error}
