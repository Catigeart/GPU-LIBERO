"""Explicit independent complete-C cold ABI; default installation policy is unchanged."""
from pathlib import Path


def pending_handwritten_collision(static, plan, *, source_owner, source_paths, stream,
        device='cuda:0', borrow_detector_raw=True, capsule_source_root=None,
        admission_context=None,collision_options=None):
    """Cold C construction only. Every physical launch rejects pending bindings."""
    requested_options=dict(collision_options or {})
    width=requested_options.get('query_subgroup_width',32)
    box=requested_options.get('box_private_capacity',50)
    if type(width) is not int or width not in (0,32):
        raise ValueError('unqualified collision subgroup route; only 0/32 available')
    if type(box) is not int or box!=50:
        raise ValueError('unqualified collision box capacity; only 50 available')
    from libero.gpu.host.runtime.collision.runtime_adapter import CollisionCapacityPlan, CollisionRuntimeAdapter, plan_capacity
    from libero.gpu.host.assembly.pipeline.collision import FusedCollisionChainWorkspace
    if not isinstance(plan,CollisionCapacityPlan):raise TypeError('CollisionCapacityPlan required')
    actual=plan_capacity(static,plan.nworld,total_device_budget_bytes=plan.total_device_budget_bytes,
        reserved_other_bytes=plan.reserved_other_bytes,**{name:getattr(plan,name) for name in
        ('slots','call_capacity','pair_capacity','stage_slots','req_slots','contact_capacity')})
    if actual!=plan:raise ValueError('capacity plan does not match actual model')
    from libero.gpu.host.runtime.collision.handwritten_factory import prepare_collision_admission
    options=dict(requested_options,capacity_plan=plan.manifest(),borrow_detector_raw=borrow_detector_raw)
    admission=prepare_collision_admission(static,context=admission_context,options=options)
    try:
        ws=FusedCollisionChainWorkspace(admission.static,plan.nworld,device=device,
            borrow_detector_raw=borrow_detector_raw,detector_backend='handwritten_only',**plan.workspace_kwargs())
        admission.attach(ws)
        result=CollisionRuntimeAdapter.from_workspace(ws,stream=stream,source_owner=source_owner,source_paths=source_paths)
        result.plan=plan
        return result
    except BaseException:
        admission.revoke('cold collision construction failed')
        raise


def finalize_handwritten_collision(collision, dynamics, transport, *, build_root,
        dataflow_profile, collision_policy,complete_only=False):
    """Install actual C exports/direct transport owners, then seal exactly once."""
    from libero.gpu.host.bindings.collision import configure_temporal_front, install_workspace, install_combined_workspace
    from libero.gpu.host.bindings.collision_detect import install_detector
    ws=collision.workspace;root=Path(build_root)
    if ws._detector_backend!='handwritten_only':raise ValueError('handwritten_only cold ABI required')
    if type(complete_only) is not bool:raise TypeError('complete_only must be bool')
    if complete_only:
        options=ws._collision_cold_admission.options
        if (dataflow_profile!='fast' or collision_policy.get('temporal',False) or
                options.get('query_schedule','fused')!='fused' or
                type(options.get('query_subgroup_width',32)) is not int or options.get('query_subgroup_width',32)!=32 or
                type(options.get('box_private_capacity',50)) is not int or options.get('box_private_capacity',50)!=50):
            raise ValueError('complete_only requires fast/fused/non-temporal/full32/Box50 cold ABI')
        if hasattr(ws,'_cuda_collision_complete'):
            raise ValueError('complete_only finalizer requires a fresh pending backend')
    from libero.gpu.host.runtime.collision.shared import _handwritten_plans
    admission=ws._collision_cold_admission;options=admission.options
    options.update(dataflow_profile=dataflow_profile,collision_policy=dict(collision_policy),
        query_subgroup_width=options.get('query_subgroup_width',32),
        box_private_capacity=options.get('box_private_capacity',50),
        query_schedule=options.get('query_schedule','fused'),producer_blocks=options.get('producer_blocks',32))
    if complete_only:options['cold_installation_abi']='complete-only-v1'
    try:
        configure_temporal_front(ws,dynamics,dataflow_profile=dataflow_profile,**collision_policy)
        install_workspace(ws,root/'partial',dataflow_profile=dataflow_profile,complete_only=complete_only)
        install_detector(ws.detect,root/'detect',diagnostic_profile=dataflow_profile,
            query_subgroup_width=options.get('query_subgroup_width',32),box_private_capacity=options.get('box_private_capacity',50))
        install_combined_workspace(ws,root/'complete',transport=transport,
            query_schedule=options.get('query_schedule','fused'),producer_blocks=options.get('producer_blocks',32))
        admission.seal(_handwritten_plans(ws))
        ws.seal_handwritten_backend()
        ws.validate_device_bindings(collision.stream)
        return ws._cuda_collision_manifest
    except BaseException:
        admission.revoke('cold collision installation failed')
        raise
