"""One shape-aware cold route: wide recovery, final events and typed cleanup.

Sampler history and domain admission wrap this owner once at the public entry.
No registered numerical receipt is inferred from diagnostic execution.
"""
from . import recovery_final_event_environment_factory as source
from libero.gpu.host.adapters.environment import cleanup_factory_r02 as cleanup

def assemble_prepared(built,prepared,*,observation_continuation=None):
    from .recovery_cleanup_environment_r02 import make_boundary_callbacks
    result=cleanup.assemble_prepared(
        built,prepared,observation_continuation=observation_continuation,
        callback_factory=make_boundary_callbacks)
    hook=getattr(prepared,'post_assembly_hook',None)
    if hook is not None:
        try:hook(result)
        except BaseException:
            result.release()
            raise
    return result

def _publication(bundle,lease,expected):
    import warp as wp
    wp.synchronize_stream(bundle.stream)
    first=int(bundle.runtime.first_error.numpy()[0])
    # There is no separate uploader in this cold path: the actual commit
    # writes ready only after all owners validate and transfers complete.
    bundle.domain_admission.assert_domain_publication(lease,dict(
        model_sha256=bundle.model_spec.model_sha256,
        device_model_epoch=int(bundle.device_model_epoch.numpy()[0]),
        expected_model_epoch=int(bundle.expected_model_epoch.numpy()[0]),
        model_ready=int(bundle.model_ready.numpy()[0]),first_error=first,
        upload_completed=True),expected_model_epoch=expected)

def create(built,*,admission=None,physics_factory=None,factory_entry="libero.gpu.host.adapters.environment.production_factory_r02:create"):
    entry=factory_entry
    if not isinstance(entry,str) or not entry:raise TypeError("explicit combined factory ENTRY required")
    if physics_factory is None:
        from .recovery_final_event_factory import prepare_final_event_recovery_physics
        physics_factory=prepare_final_event_recovery_physics
    if not callable(physics_factory):raise TypeError("cold physical factory must be callable")
    lease=None if admission is None else admission.admission_domain_for_built(built,factory_entry=entry)
    def prepare(*args,**kwargs):
        bundle=physics_factory(*args,**kwargs)
        bundle.cold_recovery_prepare=physics_factory
        bundle.domain_admission=admission
        bundle.domain_lease=lease
        bundle.domain_factory_entry=entry
        if lease is not None:
            try:_publication(bundle,lease,0)
            except BaseException as primary:
                try:bundle.close()
                except BaseException:
                    from libero.gpu.host.adapters.environment.cleanup_ownership_r02 import attach_cleanup, ReleaseOwner
                    attach_cleanup(primary,ReleaseOwner(bundle,"close"))
                raise
        return bundle
    return source.create(built,physics_factory=prepare,assembler=assemble_prepared)

def rebuild_components(built,previous,*,admission=None,physics_factory=None,factory_entry="libero.gpu.host.adapters.environment.production_factory_r02:create"):
    # Actual metadata shape gate; never the old fixed 48/43 shortcut.
    from .shape_schema import RuntimeShape
    RuntimeShape.from_native_metadata(built.metadata)
    source.default_integration_state(built.native_model)
    source.actual_decoder_record(built)
    if admission is not None:
        admission.admission_domain_for_built(built,factory_entry=factory_entry)
    previous.release()
    return create(built,admission=admission,physics_factory=physics_factory,factory_entry=factory_entry)
create.rebuild_components=rebuild_components
