"""Cold collision construction for the qualified copy path."""


def prepare_copy_collision(static,plan,*,source_owner,source_paths,stream,device):
    from libero.gpu.host.runtime.collision.parallel_finalize_copy import source_derived_workspace_class
    from libero.gpu.host.runtime.collision.runtime_adapter import CollisionRuntimeAdapter
    candidate=source_derived_workspace_class()
    workspace=candidate(static,plan.nworld,device=device,borrow_detector_raw=True,
                        **plan.workspace_kwargs())
    result=CollisionRuntimeAdapter.from_workspace(workspace,stream=stream,
        source_owner=source_owner,source_paths=source_paths)
    result.plan=plan
    result.candidate_manifest=dict(candidate.candidate_source_manifest)
    result.candidate_manifest['independent_equivalence_report_sha256']='37399458bb21827efef7681997841e65dee334453cef49d0a5641aaf0bc36fb1'
    return result
