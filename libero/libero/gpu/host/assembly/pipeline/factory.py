"""Cold integration of the eight large-fusion work packages.

All substitutions happen during construction. Original factories, source
modules and the preceding fusion baseline are retained.
"""
from functools import partial
from types import SimpleNamespace

from libero.gpu.host.assembly.compose.backup import FusedEventRecoveringRuntime
from libero.gpu.host.assembly.compose.s_rows import fused_workspace_type
from libero.gpu.host.runtime.dynamics import wide_factory, wide_runtime
from libero.gpu.host.runtime.newton import live_copy_api
from libero.gpu.host.runtime.step import recovery_event_factory, s04_jar_specialization_r01

from . import dynamics, euler, recovery, newton
from .collision import make_fused_adapter
from .copies import FusedNoForwardHold, FusedRuntimePostAdapter as CopyPost
from .forward import FusedShapePhysicsForward, FusedFoundationForwardAdapter
from .osc import FusedRuntimeControllerAdapter
from libero.gpu.host.runtime.step.shape_assembly import ShapeAssembly


def _initialize_dynamics(self, *args, **kwargs):
    wide_runtime.GenericWideRuntimeDynamicsAdapter.__init__(
        self, *args, hold_type=FusedNoForwardHold, **kwargs)
    dynamics.prepare_dynamics(self)


FusedWideDynamicsAdapter = type("FusedWideDynamicsAdapterR03",
    (wide_runtime.GenericWideRuntimeDynamicsAdapter,),
    dict(dynamics.runtime_methods(), __init__=_initialize_dynamics, __module__=__name__))


class FusedRuntimePostAdapter(dynamics.FusedRuntimePostAdapter, CopyPost):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def prepare_from_built(*args, **kwargs):
    return wide_factory.prepare_from_built(*args, adapter_type=FusedWideDynamicsAdapter, **kwargs)


def prepare_post(*args, **kwargs):
    adapter = wide_factory.prepare_post(*args, adapter_type=FusedRuntimePostAdapter, **kwargs)
    # The original factory binds and freezes the post workspace before the
    # source launcher's validation can be used for cold planning.
    dynamics.prepare_post(adapter)
    return adapter


def _load_candidate(nv, nm, build_root):
    package, manifest = s04_jar_specialization_r01.load_candidate(nv, nm, build_root)
    view = SimpleNamespace(workspace=package.workspace,
        solver=SimpleNamespace(FwdConstraintSolver=newton.fused_solver_class(package)))
    return view, dict(manifest, large_fusion_batch=3)


def _newton_factory(**kwargs):
    adapter = live_copy_api.prepare_live_copy_runtime(
        candidate_loader=_load_candidate, adapter_type=newton.FusedFinalEventAdapter, **kwargs)
    adapter.final_event = adapter.solver
    return adapter


def _cold_factories():
    return make_fused_adapter, _newton_factory, "cn"


FusedRuntime = recovery.runtime_type(euler.runtime_type(FusedEventRecoveringRuntime))
_recovery_loop = partial(recovery_event_factory._recovery_loop,
    runtime_type=FusedRuntime, recovery_workspace_type=recovery.FusedDataRecoveryWorkspace)


def _validate_controller_selection(dynamics_owner, controller):
    """Match the actual controller family and its frozen workspace owner."""
    if getattr(dynamics_owner, "joint_controller", False):
        from libero.gpu.host.runtime.control.joint import RuntimeJointControllerAdapter
        assert isinstance(controller, RuntimeJointControllerAdapter)
    else:
        from .osc import FusedRuntimeControllerAdapter
        assert isinstance(controller, FusedRuntimeControllerAdapter)
    assert controller.controller is dynamics_owner.control
    return True


def _validate_selection(bundle):
    from .collision import FusedCollisionChainWorkspace
    from .forward import FusedShapePhysicsForward, FusedFoundationForwardAdapter
    from .osc import FusedRuntimeControllerAdapter
    loop = bundle.runtime
    assert isinstance(bundle.dynamics, FusedWideDynamicsAdapter)
    assert isinstance(loop.forward.c.workspace, FusedCollisionChainWorkspace)
    assert isinstance(loop.forward, FusedShapePhysicsForward)
    assert isinstance(loop.forward.f, FusedFoundationForwardAdapter)
    assert isinstance(bundle.newton, newton.FusedFinalEventAdapter)
    assert isinstance(loop.euler, euler.FusedEulerWorkspace)
    assert isinstance(loop.recovery, recovery.FusedDataRecoveryWorkspace)
    assert isinstance(bundle.post, FusedRuntimePostAdapter)
    if bundle.controller is not None:
        _validate_controller_selection(bundle.dynamics, bundle.controller)
    assert all(isinstance(hold, FusedNoForwardHold) for hold in bundle.dynamics._holds.values())
    assert hasattr(loop.forward.s.workspace, "_fusion_s_before")
    assert loop.recovery._fusion_protocol_owner is loop
    assert hasattr(loop, "_fusion_euler_kernel")
    assert not loop.capture_events or hasattr(loop, "_fusion_warning_copy")
    return True


def prepare_fused_physics(*args, newton_diagnostic_profile='compatible', **kwargs):
    if kwargs.get("mode", "diagnostic") not in ("production", "diagnostic"):
        raise ValueError("unsupported execution mode")
    def cold_factories():
        return (make_fused_adapter,
                partial(_newton_factory, diagnostic_profile=newton_diagnostic_profile), 'cn')
    assembly = ShapeAssembly(dynamics_factory=prepare_from_built,
        post_factory=prepare_post, cold_factories=cold_factories,
        foundation_type=FusedFoundationForwardAdapter, forward_type=FusedShapePhysicsForward,
        controller_type=FusedRuntimeControllerAdapter, euler_type=euler.FusedEulerWorkspace,
        runtime_factory=_recovery_loop, constraint_workspace_type=fused_workspace_type)
    bundle = recovery_event_factory.prepare_recovery_event_physics(
        *args, _assembly=assembly, **kwargs)
    try:
        bundle.fusion_batch = 3
        bundle.fusion_dynamics_catalog = dynamics.plan_catalog(bundle.dynamics)
        bundle.fusion_post_catalog = dynamics.plan_catalog(bundle.post)
        bundle.validate_fusion_selection = lambda: _validate_selection(bundle)
        bundle.validate_fusion_selection()
    except BaseException:
        bundle.close()
        raise
    return bundle


def create(built):
    from libero.gpu.host.runtime.step.recovery_cleanup_factory_r02 import create as assemble
    return assemble(built, physics_factory=prepare_fused_physics,
                    factory_entry="libero.gpu.host.assembly.pipeline.factory:create")
