"""Explicit cold opt-in C factory; original adapter and chain remain unchanged.

Caller must retain the exact model/source/actual scoped qualifications. Selecting
this factory does not grant arbitrary models or trajectories. Construction must
precede device input binding and graph capture; no live workspace class swap.
"""
from .runtime_adapter import CollisionCapacityPlan, CollisionRuntimeAdapter, plan_capacity
from .parallel_finalize_copy import source_derived_workspace_class


def make_parallel_copy_adapter(static, plan, *, source_owner, source_paths,
                               stream, device='cuda:0', borrow_detector_raw=True,
                               capsule_source_root=None):
    if not isinstance(plan, CollisionCapacityPlan):
        raise TypeError('CollisionCapacityPlan required')
    actual = plan_capacity(static, plan.nworld,
        total_device_budget_bytes=plan.total_device_budget_bytes,
        reserved_other_bytes=plan.reserved_other_bytes,
        **{name: getattr(plan, name) for name in (
            'slots', 'call_capacity', 'pair_capacity', 'stage_slots',
            'req_slots', 'contact_capacity')})
    if actual != plan:
        raise ValueError('capacity plan does not match the actual model')
    candidate_class = source_derived_workspace_class()
    workspace = candidate_class(static, plan.nworld, device=device,
        borrow_detector_raw=borrow_detector_raw, **plan.workspace_kwargs())
    if capsule_source_root is not None:
        from libero.gpu.host.runtime.collision.capsule_extension import prepare_workspace_extension
        prepare_workspace_extension(workspace, capsule_source_root)
    adapter = CollisionRuntimeAdapter.from_workspace(workspace, stream=stream,
        source_owner=source_owner, source_paths=source_paths)
    adapter.plan = plan
    # Additional provenance is explicit; preserve the original native extension
    # candidate_binding schema and the existing retained-input manifest.
    adapter.parallel_copy_candidate_manifest = candidate_class.candidate_source_manifest
    return adapter