"""Cold opt-in wiring for D/Euler/N fusion; no original factory mutation."""

from libero.gpu.host.runtime.dynamics import wide_runtime, wide_factory
from libero.gpu.host.runtime.step import recovery_event_factory, newton_jar_candidate_r01
from .dynamics import FusedSmoothWorkspace
from .euler import FusedEulerWorkspace
from libero.gpu.host.runtime.step.shape_assembly import ShapeAssembly


class FusedWideDynamicsAdapter(wide_runtime.GenericWideRuntimeDynamicsAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, smooth_type=FusedSmoothWorkspace, **kwargs)


def prepare_from_built(*args, **kwargs):
    return wide_factory.prepare_from_built(*args, adapter_type=FusedWideDynamicsAdapter, **kwargs)


prepare_post = wide_factory.prepare_post


def _newton_factory(**kwargs):
    from .newton import FusedFinalEventAdapter
    return newton_jar_candidate_r01.prepare_jar_final_event_runtime(
        adapter_type=FusedFinalEventAdapter, **kwargs)


def _cold_factories():
    from libero.gpu.host.runtime.step.jar_recovery_factory_r01 import _jar_cold_factories
    collision, unused, choice = _jar_cold_factories()
    return collision, _newton_factory, choice


def prepare_fused_physics(*args, **kwargs):
    if kwargs.get("mode", "diagnostic") not in ("production", "diagnostic"):
        raise ValueError("unsupported execution mode")
    assembly = ShapeAssembly(dynamics_factory=prepare_from_built,
        euler_type=FusedEulerWorkspace, cold_factories=_cold_factories)
    bundle = recovery_event_factory.prepare_recovery_event_physics(
        *args, _assembly=assembly, **kwargs)
    return bundle


def create(built):
    from libero.gpu.host.runtime.step.recovery_cleanup_factory_r02 import create as assemble
    return assemble(built, physics_factory=prepare_fused_physics,
                    factory_entry="libero.gpu.host.assembly.compose.factory:create")
