"""Cold second-batch fusion wiring; original and first-batch factories retained."""
from functools import partial
from types import SimpleNamespace

from . import factory as first
from .newton import FusedFinalEventAdapter
from .euler import FusedEulerWorkspace
from libero.gpu.host.runtime.step.shape_assembly import ShapeAssembly
from libero.gpu.host.runtime.newton import live_copy_api
from libero.gpu.host.runtime.step import recovery_event_factory, s04_jar_specialization_r01


def _load_candidate(nv, nm, build_root):
    from .newton_iteration import fused_solver_class
    package, manifest = s04_jar_specialization_r01.load_candidate(nv, nm, build_root)
    solver_type = fused_solver_class(package)
    # Private cold view: never mutate the shared generated candidate module.
    view = SimpleNamespace(workspace=package.workspace,
                           solver=SimpleNamespace(FwdConstraintSolver=solver_type))
    return view, dict(manifest, second_batch_fusion=True)


def _newton_factory(**kwargs):
    adapter = live_copy_api.prepare_live_copy_runtime(
        candidate_loader=_load_candidate, adapter_type=FusedFinalEventAdapter, **kwargs)
    adapter.final_event = adapter.solver
    return adapter


def _cold_factories():
    collision, unused, choice = first._cold_factories()
    return collision, _newton_factory, choice


def prepare_fused_physics(*args, **kwargs):
    if kwargs.get("mode", "diagnostic") not in ("production", "diagnostic"):
        raise ValueError("unsupported execution mode")
    from .backup import FusedEventRecoveringRuntime, FusedShapePhysicsForward
    from .s_rows import fused_workspace_type
    assembly = ShapeAssembly(dynamics_factory=first.prepare_from_built,
        euler_type=FusedEulerWorkspace, cold_factories=_cold_factories,
        forward_type=FusedShapePhysicsForward, constraint_workspace_type=fused_workspace_type,
        runtime_factory=partial(recovery_event_factory._recovery_loop,
                                runtime_type=FusedEventRecoveringRuntime))
    bundle = recovery_event_factory.prepare_recovery_event_physics(
        *args, _assembly=assembly, **kwargs)
    bundle.fusion_batch = 2
    return bundle


def create(built):
    from libero.gpu.host.runtime.step.recovery_cleanup_factory_r02 import create as assemble
    return assemble(built, physics_factory=prepare_fused_physics,
        factory_entry="libero.gpu.host.assembly.compose.factory_r02:create")
