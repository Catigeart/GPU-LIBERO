"""Unselected cold jar-only candidate for the same wide recovery route."""
import hashlib
from pathlib import Path
from .shape_assembly import ShapeAssembly


def _jar_cold_factories():
    from .generic_prepare import _cold_factories
    from .newton_jar_candidate_r01 import prepare_jar_final_event_runtime
    collision, unused_newton, choice = _cold_factories()
    if choice != "cn":
        raise ValueError("jar-only candidate requires explicit CN cold scope")
    return collision, prepare_jar_final_event_runtime, choice


def prepare_jar_recovery_physics(*args, **kwargs):
    if kwargs.get("mode", "diagnostic") != "diagnostic":
        raise ValueError("jar-only candidate is not production qualified")
    from . import recovery_event_factory as original
    bundle = original.prepare_recovery_event_physics(*args,
        _assembly=ShapeAssembly(cold_factories=_jar_cold_factories), **kwargs)
    bundle.recovery_source_manifest["jar_candidate_cold_factory_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    bundle.recovery_source_manifest["unselected_jar_only_candidate"] = True
    return bundle
