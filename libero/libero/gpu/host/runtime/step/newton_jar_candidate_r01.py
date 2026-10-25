"""Explicit jar solver selection, preserving the final-event adapter."""
from libero.gpu.host.runtime.newton import live_copy_api
from .newton_final_event import FinalEventAdapter
from .s04_jar_specialization_r01 import load_candidate


def prepare_jar_final_event_runtime(*, adapter_type=FinalEventAdapter, **kwargs):
    if kwargs.get("mode", "diagnostic") not in ("production", "diagnostic"):
        raise ValueError("unsupported execution mode")
    adapter = live_copy_api.prepare_live_copy_runtime(
        candidate_loader=load_candidate, adapter_type=adapter_type, **kwargs)
    adapter.source_manifest["jar_candidate_selected_by_actual_predicate"] = False
    adapter.final_event = adapter.solver
    return adapter
