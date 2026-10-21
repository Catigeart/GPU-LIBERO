"""Cold BuiltTaskModel -> exact new-model dynamics intake.

No MjData, CPU forward, controller reset or dynamic intermediate is produced.
The decoded native model and independently pinned decoder/source metadata are
the authority. Preparation and diagnostics do not grant production numerics.
"""
from libero.gpu.host.paths import GPU_ROOT
from pathlib import Path
import hashlib,json
import numpy as np
from libero.gpu.host.runtime.core.contracts import ModelSpec
from libero.gpu.host.runtime.core.registry import CapabilityRegistry, KernelCandidate
from ..core.model_bytes import memory_record, checked_records
from .generic_runtime import GenericRuntimeDynamicsAdapter
from .generic_post import GenericPostWorkspace
from .runtime_adapter import RuntimePostAdapter


def _pin(path):
    p=Path(path).resolve();return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def prepare_from_built(built,capacity,*,directory,decoder_record,state_epoch,first_error,stream,
                       registry=None,mode='diagnostic',device='cuda:0',model_record=None,spec=None,features=()):
    """Return ``(dynamics, intake)`` for the actual supplied BuiltTaskModel.

    ``decoder_record`` is an exact archived/live factory source record matching
    built.metadata.decoder_pin.source_sha256. ``directory`` is retained for call compatibility;
    model and metadata intake remain in memory. Existing models are never reloaded to infer counts.
    """
    md=json.loads(json.dumps(built.metadata))
    decoder=_pin(decoder_record['path'])
    if decoder!=decoder_record or decoder['sha256']!=md['decoder_pin']['source_sha256']:
        raise ValueError('actual BuiltTaskModel decoder source record mismatch')
    if model_record is None:
        model_record = dict(model=built.native_model, sha256=md['model_sha256'])
    checked_records([model_record])
    if model_record['sha256'] != md['model_sha256']:
        raise ValueError('BuiltTaskModel native model identity differs')
    metadata_record = memory_record((json.dumps(md, indent=2)+'\n').encode())
    arrays={name:np.array(getattr(built.native_model,name),copy=True,order='C') for name in md['array_sha256']}
    if spec is None:
        spec=ModelSpec.from_factory_metadata(md,features=features)
    else:
        if features:
            raise ValueError('explicit spec owns its feature identity; do not also supply features')
        actual=ModelSpec.from_factory_metadata(md,features=spec.features)
        if actual.identity_key!=spec.identity_key:
            raise ValueError('supplied shared ModelSpec differs from actual BuiltTaskModel metadata')
    gpu=(GPU_ROOT / '.')
    names=('host/runtime/dynamics/runtime_factory.py','host/runtime/dynamics/generic_runtime.py',
        'host/runtime/dynamics/generic_stages.py','host/runtime/dynamics/generic_atomic.py','host/runtime/dynamics/generic_post.py',
        'host/runtime/dynamics/specialization.py','host/runtime/dynamics/runtime_adapter.py','host/runtime/dynamics/publication.py',
        'host/runtime/dynamics/acceleration.py','host/runtime/foundation/runtime_adapter.py','reference/position/body_kinematics/workspace.py',
        'reference/position/body_kinematics/kernel.py','reference/position/kinematics_frames/frames.py','reference/position/compos/kernel.py','reference/position/crb/kernel.py','reference/position/mass_chain/kernel.py',
        'reference/position/mass_solve/kernel.py','reference/integration/euler_integrator/generic_workspace.py','reference/position/body_mass_chain/chain.py','reference/dynamics/comvel/kernel.py',
        'reference/dynamics/rne_bias/kernel.py','reference/dynamics/external_force_projection/kernel.py','reference/postprocess/rne_postconstraint/kernel.py',
        'reference/postprocess/force_sensor/kernel.py','reference/control/osc/controller.py')
    records=[_pin(gpu/name) for name in names]+[decoder,model_record,metadata_record]
    registry=CapabilityRegistry() if registry is None else registry
    bundle={r['path']:r['sha256'] for r in records[:len(names)]}
    bundle_sha=hashlib.sha256(json.dumps(bundle,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    ids={}
    for component,source in (('kinematics','reference/position/body_kinematics/kernel.py'),('frames','reference/position/kinematics_frames/frames.py'),('mass',None),('dynamics',None)):
        digest=_pin(gpu/source)['sha256'] if source else bundle_sha
        cid=f'D-generic-{component}-{spec.model_sha256[:16]}-{digest[:16]}'
        if cid not in registry.candidates:
            registry.register_candidate(KernelCandidate(candidate_id=cid,component='D',source_sha256=digest,
                algorithm='MuJoCo2.3.7-native-count-sourceorder',layout_version=spec.layout_version,architecture='CUDA',
                math_profile='RN64-noFMA-noFTZ',author='D-runtime-author',model_sha256s=(spec.model_sha256,),
                source_files={} if source else bundle))
        ids[component]=cid
    kwargs=dict(arrays=arrays,metadata=md,spec=spec,registry=registry,candidate_ids=ids,
        source_records=records,metadata_record=metadata_record,mode=mode,device=device)
    d=GenericRuntimeDynamicsAdapter(capacity=capacity,robot=md['robot_metadata'][0],state_epoch=state_epoch,
        first_error=first_error,stream=stream,**kwargs)
    intake=dict(kwargs=kwargs,model_record=model_record,metadata_record=metadata_record,
        candidate_ids=ids,mode=mode,qualification='pending independent actual model numeric/writes/provenance acceptance')
    return d,intake


def prepare_post(intake,capacity,contact_capacity,efc_capacity,*,sources,active,first_error,d_valid,
                 state_epoch,forward_epoch,n_valid,n_error,n_written,n_attempt,n_expected,stream):
    """Prepare real resident post input owners and freeze them once.

    Caller must snapshot planned N before every N invocation using
    adapter.capture_before_constraint(), then actual N, then adapter.enqueue().
    No native/host qacc, force, contact or ready arrays are inserted here.
    """
    k=intake['kwargs'];post=GenericPostWorkspace(k['arrays'],k['metadata'],k['spec'],capacity,
        contact_capacity,efc_capacity,registry=k['registry'],candidate_id=k['candidate_ids']['dynamics'],
        source_records=k['source_records'],metadata_record=k['metadata_record'],mode=k['mode'],device=k['device'])
    adapter=RuntimePostAdapter(post,sources=sources,active=active,first_error=first_error,d_valid=d_valid,
        state_epoch=state_epoch,forward_epoch=forward_epoch,n_valid=n_valid,n_error=n_error,
        n_written=n_written,n_attempt=n_attempt,n_expected=n_expected,stream=stream)
    post.freeze_bindings();adapter.validate(stream)
    return adapter
