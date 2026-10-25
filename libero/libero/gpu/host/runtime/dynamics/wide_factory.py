"""Cold BuiltTaskModel -> exact new-model dynamics intake.

No MjData, CPU forward, controller reset or dynamic intermediate is produced.
The decoded native model and its prepared metadata are passed directly. Preparation and diagnostics do not grant production numerics.
"""
from copy import deepcopy
import numpy as np
from libero.gpu.host.runtime.core.contracts import ModelSpec
from libero.gpu.host.runtime.core.registry import CapabilityRegistry, KernelCandidate
from .wide_runtime import GenericWideRuntimeDynamicsAdapter as GenericRuntimeDynamicsAdapter
from .generic_post import GenericPostWorkspace
from .runtime_adapter import RuntimePostAdapter


def prepare_from_built(built,capacity,*,directory,decoder_record,state_epoch,first_error,stream,
                       registry=None,mode='diagnostic',device='cuda:0',model_record=None,spec=None,features=(),
                       adapter_type=None):
    """Return ``(dynamics, intake)`` for the actual supplied BuiltTaskModel.

    ``decoder_record`` and ``directory`` are legacy call-compatible keywords;
    model and metadata intake remain in memory. Existing models are never reloaded to infer counts.
    """
    if mode not in ('production', 'diagnostic'):raise ValueError('unsupported execution mode')
    md=deepcopy(built.metadata)
    # BuiltTaskModel owns the decoder output and live native model. Do not
    # serialize that owner or reread its decoder to authenticate the handoff.
    if model_record is None:
        model_record=dict(model=built.native_model, sha256=md['model_sha256'])
    if ('model' in model_record and model_record['model'] is not built.native_model):
        raise ValueError('model record refers to a different native owner')
    if model_record['sha256'] != md['model_sha256']:
        raise ValueError('BuiltTaskModel descriptor identity differs')
    for name, count in md['dimensions'].items():
        if hasattr(built.native_model, name) and int(getattr(built.native_model, name)) != count:
            raise ValueError('BuiltTaskModel dimensions differ: '+name)
    metadata_record=None  # Legacy component keyword; metadata is passed directly.
    if spec is None:
        spec=ModelSpec.from_factory_metadata(md,features=features)
    else:
        if features:
            raise ValueError('explicit spec owns its feature identity; do not also supply features')
        actual=ModelSpec.from_factory_metadata(md,features=spec.features)
        if actual.identity_key!=spec.identity_key:
            raise ValueError('supplied shared ModelSpec differs from actual BuiltTaskModel metadata')
    from ..core.static_readset import PreparedModelArrays
    arrays=PreparedModelArrays({name:getattr(built.native_model,name) for name in md['array_sha256']},md,spec)
    registry=CapabilityRegistry() if registry is None else registry
    ids={}
    # Capabilities are keyed by the complete model/layout/features and explicit
    # implementation family, not by runtime reads of development source files.
    for component in ('kinematics','frames','mass','dynamics'):
        cid=f'D-runtime-wide-v1-{component}-{spec.identity_key}'
        registry.register_candidate(KernelCandidate(candidate_id=cid,component='D',source_sha256=None,
            algorithm='MuJoCo2.3.7-native-count-sourceorder',layout_version=spec.layout_version,architecture='CUDA',
            math_profile='RN64-unfused-pipeline-CUDA-double-trig',author='D-runtime-author',model_sha256s=(spec.model_sha256,)))
        ids[component]=cid
    kwargs=dict(arrays=arrays,metadata=md,spec=spec,registry=registry,candidate_ids=ids,
        source_records=(),metadata_record=metadata_record,mode=mode,device=device)
    adapter_type = GenericRuntimeDynamicsAdapter if adapter_type is None else adapter_type
    d=adapter_type(capacity=capacity,robot=md['robot_metadata'][0],state_epoch=state_epoch,
        first_error=first_error,stream=stream,**kwargs)
    intake=dict(kwargs=kwargs,model_record=model_record,metadata_record=metadata_record,
        candidate_ids=ids,mode=mode,qualification='pending independent actual model numeric/writes/provenance acceptance')
    return d,intake


def prepare_post(intake,capacity,contact_capacity,efc_capacity,*,sources,active,first_error,d_valid,
                 state_epoch,forward_epoch,n_valid,n_error,n_written,n_attempt,n_expected,stream,adapter_type=None):
    """Prepare real resident post input owners and freeze them once.

    Caller must snapshot planned N before every N invocation using
    adapter.capture_before_constraint(), then actual N, then adapter.enqueue().
    No native/host qacc, force, contact or ready arrays are inserted here.
    """
    k=intake['kwargs'];post=GenericPostWorkspace(k['arrays'],k['metadata'],k['spec'],capacity,
        contact_capacity,efc_capacity,registry=k['registry'],candidate_id=k['candidate_ids']['dynamics'],
        source_records=k['source_records'],metadata_record=k['metadata_record'],mode=k['mode'],device=k['device'])
    adapter_type = RuntimePostAdapter if adapter_type is None else adapter_type
    adapter=adapter_type(post,sources=sources,active=active,first_error=first_error,d_valid=d_valid,
        state_epoch=state_epoch,forward_epoch=forward_epoch,n_valid=n_valid,n_error=n_error,
        n_written=n_written,n_attempt=n_attempt,n_expected=n_expected,stream=stream)
    post.freeze_bindings();adapter.validate(stream)
    return adapter
