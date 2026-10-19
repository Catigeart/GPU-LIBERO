"""E1: world-local full Euler and publication, preserving tape/event boundaries."""
import ast
import inspect
import textwrap
import warp as wp
from libero.gpu.host.assembly.compose.compose import compose, identity_mapping
from libero.gpu.host.assembly.compose.euler import finish_euler
from libero.gpu.reference.integration.euler_integrator.kernel import _prepare, _source_solve_labels, _commit
from libero.gpu.reference.integration.euler_integrator.wide_angle import _integrate_wide, WideAngleEulerWorkspace
from libero.gpu.reference.position.mass_solve.kernel import _factor_solve
from libero.gpu.host.runtime.step.shape_runtime import _integration_gate, _publish_step, _advance_expected_state, _token
from libero.gpu.host.runtime.step.shape_counter import _count_completed
from libero.gpu.host.runtime.step.recovery_event_runtime import EventRecoveringRuntime

EULER_STAGES = (
    (_prepare, identity_mapping(_prepare, qm="qM", smooth="qfrc_smooth",
        constraint="qfrc_constraint", damping="dof_damping", madr="dof_Madr")),
    (_factor_solve, identity_mapping(_factor_solve, qm="mass_qm", rhs="mass_rhs",
        count="mass_count", active="mass_active", madr="dof_Madr", parent="dof_parentid",
        ld="mass_ld", inv="mass_inv", sqrt_inv="mass_sqrt_inv", solution="mass_solution",
        pivot="mass_pivot", status="mass_status", warning_count="mass_warning_count",
        warning_last="mass_warning_last")),
    (_source_solve_labels, identity_mapping(_source_solve_labels, rhs="mass_rhs",
        ld="mass_ld", madr="dof_Madr", parent="dof_parentid", simple="dof_simplenum", work="zero_work")),
    (_integrate_wide, identity_mapping(_integrate_wide, solution="mass_solution",
        jtype="jnt_type", padr="jnt_qposadr", vadr="jnt_dofadr")),
    (_commit, identity_mapping(_commit)),
)
full_euler = compose("full_euler_r03", EULER_STAGES)


def _values(e):
    m = e.mass
    values = dict(vars(e))
    values.update(dof_Madr=m.dof_Madr, dof_parentid=m.dof_parentid,
        dof_simplenum=m.dof_simplenum, scratch_pos=e.qpos_after,
        scratch_vel=e.qvel_after, used_acc=e.qacc_used, scratch_time=e.time_after, nm=e.nM)
    for name in ("qm", "rhs", "count", "active", "ld", "inv", "sqrt_inv", "solution",
                 "pivot", "status", "warning_count", "warning_last"):
        values["mass_" + name] = getattr(m, name)
    return values


class FusedEulerWorkspace(WideAngleEulerWorkspace):
    def launch(self, stream=None):
        values = _values(self)
        wp.launch(full_euler, self.capacity,
            inputs=[values[name] for name in full_euler.fusion_argument_names],
            device=self.device, stream=stream)


def _runtime_kernel(gate, expected, counted):
    stages = []
    if gate:
        stages.append((_integration_gate, identity_mapping(_integration_gate, epoch="state_epoch")))
    stages.extend(EULER_STAGES)
    stages.append((_publish_step, identity_mapping(_publish_step, epoch="state_epoch")))
    if expected:
        stages.append((_advance_expected_state, identity_mapping(_advance_expected_state)))
    if counted:
        stages.append((_count_completed, identity_mapping(_count_completed, count="integration_count")))
    return compose("runtime_euler_r03_" + "_".join(str(int(v)) for v in (gate, expected, counted)), stages)


def launch_runtime_integrate(self):
    """Bottom integration only; event snapshots must surround this call."""
    e = self.euler
    counted = hasattr(self, "integration_count")
    if counted and (self.integration_count is not self._count_ref or
                    _token(self.integration_count) != self._count_token):
        raise ValueError("actual integration count owner changed")
    # A tape observes gated inputs before prepare mutates scratch or Euler state.
    if self.integration_tape is not None:
        wp.launch(_integration_gate, e.capacity, inputs=[self.intent, self.state_epoch,
            self.forward_valid, self.forward_written, self.forward_error, self.first_error,
            e.qpos, e.qvel, e.qacc, e.active, e.nq, e.nv], device=e.device, stream=self.stream)
        self.integration_tape.capture()
    kernel = self._fusion_euler_kernel
    values = _values(e)
    values.update(intent=self.intent, state_epoch=self.state_epoch,
        forward_valid=self.forward_valid, forward_written=self.forward_written,
        forward_error=self.forward_error, first_error=self.first_error,
        written=self.written, valid=self.valid, expected=self.expected_state_epoch)
    if counted:
        values["integration_count"] = self.integration_count
    wp.launch(kernel, e.capacity, inputs=[values[name] for name in kernel.fusion_argument_names],
              device=e.device, stream=self.stream)


def runtime_type(base):
    """Cold class derivation preserving original event sampling and callbacks."""
    if issubclass(base, EventRecoveringRuntime):
        from .copies import event_warning_plan, launch_event_warnings
        tree = ast.parse(textwrap.dedent(inspect.getsource(EventRecoveringRuntime._integrate)))
        class Replace(ast.NodeTransformer):
            count = 0
            warnings = 0
            def visit_For(self, node):
                if isinstance(node.target, ast.Tuple) and ast.unparse(node.target) == '(source, target)':
                    if 'self.recovery.warning_number' in ast.unparse(node.iter):
                        self.warnings += 1
                        return ast.parse('launch_event_warnings(self)').body[0]
                return self.generic_visit(node)
            def visit_Call(self, node):
                if ast.unparse(node) == "super()._integrate()":
                    self.count += 1
                    return ast.parse("launch_runtime_integrate(self)", mode="eval").body
                return self.generic_visit(node)
        replace = Replace()
        tree = replace.visit(tree)
        if replace.count != 1 or replace.warnings != 1:
            raise ValueError("one bottom event integration and warning copy loop required")
        ast.fix_missing_locations(tree)
        namespace = dict(EventRecoveringRuntime._integrate.__globals__)
        namespace["launch_runtime_integrate"] = launch_runtime_integrate
        namespace["launch_event_warnings"] = launch_event_warnings
        exec(compile(tree, __file__ + ":event-integrate", "exec"), namespace)
        method = namespace["_integrate"]
    else:
        method = launch_runtime_integrate
    def initialize(self, **kwargs):
        base.__init__(self, **kwargs)
        self._fusion_euler_kernel = _runtime_kernel(self.integration_tape is None,
            self.expected_state_epoch is not None, hasattr(self, "integration_count"))
        if issubclass(base, EventRecoveringRuntime) and self.capture_events:
            self._fusion_warning_copy = event_warning_plan(self)
    return type("FusedEuler" + base.__name__, (base,), {
        "__init__": initialize, "_integrate": method, "__module__": __name__})
