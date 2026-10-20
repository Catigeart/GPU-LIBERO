"""Explicit measured local variants. No default or whole-application selection.

Cold ModelSpec identity and owner remain validated by the existing installer.
Live finite/status/error/held guards remain in the original or tested CUDA leaf.
The known table records measured domains, not a new mathematical admission.
"""
import libero.gpu.precision as scalar
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np


DOMAIN_TABLE = {'D02_NV21_W1': {'feature': 'scalar_support', 'decision': 'positive_local_domain_pending_root_adoption', 'nv': 21, 'worlds': 1, 'nu': 9, 'nM': 86, 'nbody': 26, 'model_sha256': 'a155e8063e3c69f23fec011b7d84d081df28b479401a160916b80d8c7cdc2be7'}, 'D02_NV21_W2': {'feature': 'scalar_support', 'decision': 'positive_local_domain_pending_root_adoption', 'nv': 21, 'worlds': 2, 'nu': 9, 'nM': 86, 'nbody': 26, 'model_sha256': 'a155e8063e3c69f23fec011b7d84d081df28b479401a160916b80d8c7cdc2be7'}, 'D02_NV43_W1': {'feature': 'scalar_support', 'decision': 'positive_local_domain_pending_root_adoption', 'nv': 43, 'worlds': 1, 'nu': 9, 'nM': 153, 'nbody': 38, 'model_sha256': '40b3c1f71b8f8c8ff5682ab175fe4f3f789a8f76f40fdbb9ea156f98328658c4'}, 'D02_NV43_W2': {'feature': 'scalar_support', 'decision': 'positive_local_domain_pending_root_adoption', 'nv': 43, 'worlds': 2, 'nu': 9, 'nM': 153, 'nbody': 38, 'model_sha256': '40b3c1f71b8f8c8ff5682ab175fe4f3f789a8f76f40fdbb9ea156f98328658c4'}, 'D02_NV57_W1': {'feature': 'scalar_support', 'decision': 'positive_local_domain_pending_root_adoption', 'nv': 57, 'worlds': 1, 'nu': 9, 'nM': 212, 'nbody': 29, 'model_sha256': '5392e0692d29e1ad0fa0f9f8000f7c8626ee2610985da80f33cc5b70fdc8706c'}, 'D02_NV57_W2': {'feature': 'scalar_support', 'decision': 'positive_local_domain_pending_root_adoption', 'nv': 57, 'worlds': 2, 'nu': 9, 'nM': 212, 'nbody': 29, 'model_sha256': '5392e0692d29e1ad0fa0f9f8000f7c8626ee2610985da80f33cc5b70fdc8706c'}, 'D04_NV21_W1': {'feature': None, 'decision': 'negative_keep_baseline', 'nv': 21, 'worlds': 1, 'nu': 9, 'nM': 86, 'nbody': 26, 'model_sha256': 'a155e8063e3c69f23fec011b7d84d081df28b479401a160916b80d8c7cdc2be7'}, 'D04_NV21_W2': {'feature': None, 'decision': 'negative_keep_baseline', 'nv': 21, 'worlds': 2, 'nu': 9, 'nM': 86, 'nbody': 26, 'model_sha256': 'a155e8063e3c69f23fec011b7d84d081df28b479401a160916b80d8c7cdc2be7'}, 'D04_NV43_W1': {'feature': None, 'decision': 'negative_keep_baseline', 'nv': 43, 'worlds': 1, 'nu': 9, 'nM': 153, 'nbody': 38, 'model_sha256': '40b3c1f71b8f8c8ff5682ab175fe4f3f789a8f76f40fdbb9ea156f98328658c4'}, 'D04_NV43_W2': {'feature': None, 'decision': 'negative_keep_baseline', 'nv': 43, 'worlds': 2, 'nu': 9, 'nM': 153, 'nbody': 38, 'model_sha256': '40b3c1f71b8f8c8ff5682ab175fe4f3f789a8f76f40fdbb9ea156f98328658c4'}, 'D04_NV57_W1': {'feature': None, 'decision': 'negative_keep_baseline', 'nv': 57, 'worlds': 1, 'nu': 9, 'nM': 212, 'nbody': 29, 'model_sha256': '5392e0692d29e1ad0fa0f9f8000f7c8626ee2610985da80f33cc5b70fdc8706c'}, 'D04_NV57_W2': {'feature': None, 'decision': 'negative_keep_baseline', 'nv': 57, 'worlds': 2, 'nu': 9, 'nM': 212, 'nbody': 29, 'model_sha256': '5392e0692d29e1ad0fa0f9f8000f7c8626ee2610985da80f33cc5b70fdc8706c'}, 'D05_NV21_W1': {'feature': 'cold_bounded_stage', 'decision': 'positive_cold_allocation_domain_pending_root_adoption', 'nv': 21, 'worlds': 1, 'nu': 9, 'nM': 86, 'nbody': 26, 'model_sha256': 'a155e8063e3c69f23fec011b7d84d081df28b479401a160916b80d8c7cdc2be7'}, 'D05_NV21_W2': {'feature': 'cold_bounded_stage', 'decision': 'positive_cold_allocation_domain_pending_root_adoption', 'nv': 21, 'worlds': 2, 'nu': 9, 'nM': 86, 'nbody': 26, 'model_sha256': 'a155e8063e3c69f23fec011b7d84d081df28b479401a160916b80d8c7cdc2be7'}, 'D05_NV43_W1': {'feature': 'cold_bounded_stage', 'decision': 'positive_cold_allocation_domain_pending_root_adoption', 'nv': 43, 'worlds': 1, 'nu': 9, 'nM': 153, 'nbody': 38, 'model_sha256': '40b3c1f71b8f8c8ff5682ab175fe4f3f789a8f76f40fdbb9ea156f98328658c4'}, 'D05_NV43_W2': {'feature': None, 'decision': 'mixed_cold_pairs_keep_baseline', 'nv': 43, 'worlds': 2, 'nu': 9, 'nM': 153, 'nbody': 38, 'model_sha256': '40b3c1f71b8f8c8ff5682ab175fe4f3f789a8f76f40fdbb9ea156f98328658c4'}, 'D05_NV57_W1': {'feature': None, 'decision': 'mixed_cold_pairs_keep_baseline', 'nv': 57, 'worlds': 1, 'nu': 9, 'nM': 212, 'nbody': 29, 'model_sha256': '5392e0692d29e1ad0fa0f9f8000f7c8626ee2610985da80f33cc5b70fdc8706c'}, 'D05_NV57_W2': {'feature': None, 'decision': 'negative_cold_keep_baseline', 'nv': 57, 'worlds': 2, 'nu': 9, 'nM': 212, 'nbody': 29, 'model_sha256': '5392e0692d29e1ad0fa0f9f8000f7c8626ee2610985da80f33cc5b70fdc8706c'}, 'D06_NV21_W1': {'feature': None, 'decision': 'negative_keep_baseline', 'nv': 21, 'worlds': 1, 'nu': 9, 'nM': 86, 'nbody': 26, 'model_sha256': 'a155e8063e3c69f23fec011b7d84d081df28b479401a160916b80d8c7cdc2be7'}, 'D06_NV21_W2': {'feature': None, 'decision': 'negative_keep_baseline', 'nv': 21, 'worlds': 2, 'nu': 9, 'nM': 86, 'nbody': 26, 'model_sha256': 'a155e8063e3c69f23fec011b7d84d081df28b479401a160916b80d8c7cdc2be7'}, 'D06_NV43_W1': {'feature': None, 'decision': 'negative_keep_baseline', 'nv': 43, 'worlds': 1, 'nu': 9, 'nM': 153, 'nbody': 38, 'model_sha256': '40b3c1f71b8f8c8ff5682ab175fe4f3f789a8f76f40fdbb9ea156f98328658c4'}, 'D06_NV43_W2': {'feature': None, 'decision': 'negative_keep_baseline', 'nv': 43, 'worlds': 2, 'nu': 9, 'nM': 153, 'nbody': 38, 'model_sha256': '40b3c1f71b8f8c8ff5682ab175fe4f3f789a8f76f40fdbb9ea156f98328658c4'}, 'D06_NV57_W1': {'feature': None, 'decision': 'negative_keep_baseline', 'nv': 57, 'worlds': 1, 'nu': 9, 'nM': 212, 'nbody': 29, 'model_sha256': '5392e0692d29e1ad0fa0f9f8000f7c8626ee2610985da80f33cc5b70fdc8706c'}, 'D06_NV57_W2': {'feature': None, 'decision': 'negative_keep_baseline', 'nv': 57, 'worlds': 2, 'nu': 9, 'nM': 212, 'nbody': 29, 'model_sha256': '5392e0692d29e1ad0fa0f9f8000f7c8626ee2610985da80f33cc5b70fdc8706c'}, 'D07_NV21_W1': {'feature': 'fluid_parallel_payload', 'decision': 'positive_local_domain_pending_root_adoption', 'nv': 21, 'worlds': 1, 'nu': 9, 'nM': 86, 'nbody': 26, 'model_sha256': 'a155e8063e3c69f23fec011b7d84d081df28b479401a160916b80d8c7cdc2be7'}, 'D07_NV21_W2': {'feature': 'fluid_parallel_payload', 'decision': 'positive_local_domain_pending_root_adoption', 'nv': 21, 'worlds': 2, 'nu': 9, 'nM': 86, 'nbody': 26, 'model_sha256': 'a155e8063e3c69f23fec011b7d84d081df28b479401a160916b80d8c7cdc2be7'}, 'D07_NV43_W1': {'feature': 'fluid_parallel_payload', 'decision': 'positive_local_domain_pending_root_adoption', 'nv': 43, 'worlds': 1, 'nu': 9, 'nM': 153, 'nbody': 38, 'model_sha256': '40b3c1f71b8f8c8ff5682ab175fe4f3f789a8f76f40fdbb9ea156f98328658c4'}, 'D07_NV43_W2': {'feature': 'fluid_parallel_payload', 'decision': 'positive_local_domain_pending_root_adoption', 'nv': 43, 'worlds': 2, 'nu': 9, 'nM': 153, 'nbody': 38, 'model_sha256': '40b3c1f71b8f8c8ff5682ab175fe4f3f789a8f76f40fdbb9ea156f98328658c4'}, 'D07_NV57_W1': {'feature': 'fluid_parallel_payload', 'decision': 'positive_local_domain_pending_root_adoption', 'nv': 57, 'worlds': 1, 'nu': 9, 'nM': 212, 'nbody': 29, 'model_sha256': '5392e0692d29e1ad0fa0f9f8000f7c8626ee2610985da80f33cc5b70fdc8706c'}, 'D07_NV57_W2': {'feature': 'fluid_parallel_payload', 'decision': 'positive_local_domain_pending_root_adoption', 'nv': 57, 'worlds': 2, 'nu': 9, 'nM': 212, 'nbody': 29, 'model_sha256': '5392e0692d29e1ad0fa0f9f8000f7c8626ee2610985da80f33cc5b70fdc8706c'}, 'X04_NV21_W1': {'feature': None, 'decision': 'negative_keep_baseline', 'nv': 21, 'worlds': 1, 'nu': 9, 'nM': 86, 'nbody': 26, 'model_sha256': 'a155e8063e3c69f23fec011b7d84d081df28b479401a160916b80d8c7cdc2be7'}, 'X04_NV21_W2': {'feature': None, 'decision': 'negative_keep_baseline', 'nv': 21, 'worlds': 2, 'nu': 9, 'nM': 86, 'nbody': 26, 'model_sha256': 'a155e8063e3c69f23fec011b7d84d081df28b479401a160916b80d8c7cdc2be7'}, 'X04_NV43_W1': {'feature': None, 'decision': 'negative_keep_baseline', 'nv': 43, 'worlds': 1, 'nu': 9, 'nM': 153, 'nbody': 38, 'model_sha256': '40b3c1f71b8f8c8ff5682ab175fe4f3f789a8f76f40fdbb9ea156f98328658c4'}, 'X04_NV43_W2': {'feature': None, 'decision': 'negative_keep_baseline', 'nv': 43, 'worlds': 2, 'nu': 9, 'nM': 153, 'nbody': 38, 'model_sha256': '40b3c1f71b8f8c8ff5682ab175fe4f3f789a8f76f40fdbb9ea156f98328658c4'}, 'X04_NV57_W1': {'feature': None, 'decision': 'negative_keep_baseline', 'nv': 57, 'worlds': 1, 'nu': 9, 'nM': 212, 'nbody': 29, 'model_sha256': '5392e0692d29e1ad0fa0f9f8000f7c8626ee2610985da80f33cc5b70fdc8706c'}, 'X04_NV57_W2': {'feature': None, 'decision': 'negative_keep_baseline', 'nv': 57, 'worlds': 2, 'nu': 9, 'nM': 212, 'nbody': 29, 'model_sha256': '5392e0692d29e1ad0fa0f9f8000f7c8626ee2610985da80f33cc5b70fdc8706c'}, 'X08_NV21_W1': {'feature': None, 'decision': 'mixed_pairs_keep_baseline', 'nv': 21, 'worlds': 1, 'nu': 9, 'nM': 86, 'nbody': 26, 'model_sha256': 'a155e8063e3c69f23fec011b7d84d081df28b479401a160916b80d8c7cdc2be7'}, 'X08_NV21_W2': {'feature': 'fused_rne_finite', 'decision': 'positive_local_domain_pending_root_adoption', 'nv': 21, 'worlds': 2, 'nu': 9, 'nM': 86, 'nbody': 26, 'model_sha256': 'a155e8063e3c69f23fec011b7d84d081df28b479401a160916b80d8c7cdc2be7'}, 'X08_NV43_W1': {'feature': 'fused_rne_finite', 'decision': 'positive_local_domain_pending_root_adoption', 'nv': 43, 'worlds': 1, 'nu': 9, 'nM': 153, 'nbody': 38, 'model_sha256': '40b3c1f71b8f8c8ff5682ab175fe4f3f789a8f76f40fdbb9ea156f98328658c4'}, 'X08_NV43_W2': {'feature': 'fused_rne_finite', 'decision': 'positive_local_domain_pending_root_adoption', 'nv': 43, 'worlds': 2, 'nu': 9, 'nM': 153, 'nbody': 38, 'model_sha256': '40b3c1f71b8f8c8ff5682ab175fe4f3f789a8f76f40fdbb9ea156f98328658c4'}, 'X08_NV57_W1': {'feature': 'fused_rne_finite', 'decision': 'positive_local_domain_pending_root_adoption', 'nv': 57, 'worlds': 1, 'nu': 9, 'nM': 212, 'nbody': 29, 'model_sha256': '5392e0692d29e1ad0fa0f9f8000f7c8626ee2610985da80f33cc5b70fdc8706c'}, 'X08_NV57_W2': {'feature': 'fused_rne_finite', 'decision': 'positive_local_domain_pending_root_adoption', 'nv': 57, 'worlds': 2, 'nu': 9, 'nM': 212, 'nbody': 29, 'model_sha256': '5392e0692d29e1ad0fa0f9f8000f7c8626ee2610985da80f33cc5b70fdc8706c'}}


def scalar_support_plan(static, nv):
    """Compile-time immutable support map; current moment is still guarded."""
    trnid = np.asarray(static['actuator_trnid'])
    joint_types = np.asarray(static['jnt_type'])
    addresses = np.asarray(static['jnt_dofadr'])
    if (trnid.ndim != 2 or trnid.shape[1] < 1 or trnid.dtype.kind not in 'iu' or
            joint_types.ndim != 1 or addresses.shape != joint_types.shape or
            addresses.dtype.kind not in 'iu'):
        raise ValueError('scalar support source schema')
    joint = trnid[:, 0]
    if np.any(joint < 0) or np.any(joint >= len(joint_types)):
        raise ValueError('scalar support joint domain')
    types = joint_types[joint]
    if (np.any(np.asarray(static['actuator_trntype']) != 0) or
            np.any(~np.isin(types, [2, 3])) or
            np.any(np.asarray(static['actuator_gear'])[:, 1:] != 0)):
        raise ValueError('scalar transmission support family')
    support_raw = addresses[joint]
    if np.any(support_raw < 0) or np.any(support_raw >= nv):
        raise ValueError('scalar support address domain')
    support = support_raw.astype(np.int32)
    if np.any(support < 0) or np.any(support >= nv):
        raise ValueError('scalar support address domain')
    return tuple(map(int, support))


@dataclass(frozen=True)
class LocalSelection:
    domains: tuple
    features: frozenset
    support: tuple
    model_token: tuple

    @property
    def scalar_support(self):return 'scalar_support' in self.features
    @property
    def fluid_parallel_payload(self):return 'fluid_parallel_payload' in self.features
    @property
    def fused_rne_finite(self):return 'fused_rne_finite' in self.features


def model_token(adapter):
    return (adapter.spec.identity_key,adapter.spec.model_sha256,adapter.nv,
            adapter.actuation.nu,adapter.nbody,adapter.acceleration.solver.nM,adapter.capacity)


def choose_domains(domains,token):
    if isinstance(domains,(str,bytes)) or not isinstance(domains,(tuple,list)):
        raise ValueError('explicit local domain sequence required')
    if len(domains)!=len(set(domains)) or any(type(tag) is not str for tag in domains):
        raise ValueError('unique explicit local domains required')
    _,model,nv,nu,nb,nm,W=token;features=set()
    for tag in domains:
        row=DOMAIN_TABLE.get(tag)
        if row is None:raise ValueError('unknown measured local domain')
        if (row['model_sha256'],row['nv'],row['nu'],row['nbody'],row['nM'],row['worlds'])!=(model,nv,nu,nb,nm,W):
            raise ValueError('local domain cold identity or extent mismatch')
        feature=row['feature']
        if feature and feature!='cold_bounded_stage':features.add(feature)
    return frozenset(features)


def scalar_prefix(support):
    # Source prefixes affect every TU without passing comma lists through -D.
    if not isinstance(support,tuple) or not support or any(type(x) is not int or x<0 for x in support):
        raise ValueError('ordered integer scalar support required')
    return '#define D_SCALAR_SUPPORT {'+','.join(map(str,support))+'}\n'


def prepare_selection(adapter,domains,wp,dataflow_profile):
    adapter.cuda_selected_local=None
    if not domains:return
    token=model_token(adapter);features=choose_domains(domains,token)
    if not features:return
    if getattr(adapter.device,'arch',None)!=120:
        raise ValueError('local measured variants require the tested SM120 device')
    # X08 changes only the measured standalone helper. Existing fast cache-hit
    # code stays exact; use of its fallback helper does not grant a cache-hit or
    # whole-phase performance qualification.
    support=scalar_support_plan(adapter.transmission.static,adapter.nv) if 'scalar_support' in features else ()
    if support and support!=tuple(range(adapter.actuation.nu)):
        raise ValueError('current scalar support differs from the measured cold map')
    state=SimpleNamespace(selection=LocalSelection(tuple(domains),features,support,token),
        domains=tuple(domains),features=features,support=support,model_token=token,
        scalar_support='scalar_support' in features,fluid_parallel_payload='fluid_parallel_payload' in features,
        fused_rne_finite='fused_rne_finite' in features)
    if state.fluid_parallel_payload:
        state.payload=wp.zeros((adapter.capacity,adapter.nbody,9),dtype=scalar.wp_real,device=adapter.device)
        state.payload_ok=wp.zeros((adapter.capacity,adapter.nbody),dtype=wp.int32,device=adapter.device)
    state.tokens={key:(value.ptr,value.shape,value.dtype,value.device) for key,value in vars(state).items() if isinstance(value,wp.array)}
    adapter.cuda_selected_local=state


def validate_selection(adapter):
    state=getattr(adapter,'cuda_selected_local',None)
    if state is None:return
    token=model_token(adapter)
    expected=LocalSelection(state.domains,state.features,state.support,token)
    flags=(state.scalar_support,state.fluid_parallel_payload,state.fused_rne_finite)
    expected_flags=tuple(key in state.features for key in ('scalar_support','fluid_parallel_payload','fused_rne_finite'))
    if (state.selection!=expected or choose_domains(state.domains,token)!=state.features or
        flags!=expected_flags or any(type(flag) is not bool for flag in flags)):
        raise ValueError('local selection model/options owner drift')
    actual={key:(getattr(state,key).ptr,getattr(state,key).shape,getattr(state,key).dtype,getattr(state,key).device) for key in state.tokens}
    if actual!=state.tokens:raise ValueError('local private payload physical owner drift')


def selected_stages(context,stages):
    state=context.adapter.cuda_selected_local;out=[]
    for fn,args,mode in stages:
        if fn=='d_scalar_transmission_velocity' and state.scalar_support:
            fn='d_scalar_transmission_velocity_support';mode='block'
        elif fn=='d_fluid_count' and state.fluid_parallel_payload and args[13]==0:
            # Append private owners before the original optional CSR arguments.
            args=list(args);args=args[:19]+[state.payload,state.payload_ok]+args[19:]
            fn='d_fluid_parallel_selected';mode='block'
        out.append((fn,args,mode))
    return out


def allocate_selected_cold_stage(wp,nv,nbody,worlds,device,*,domain=None,model_sha256=None):
    """Exact zeros allocation only. This does not install a hot workspace."""
    if any(type(x) is not int or x<=0 for x in (nv,nbody,worlds)):
        raise ValueError('positive cold allocation extents required')
    bounded=False
    if domain is not None:
        row=DOMAIN_TABLE.get(domain)
        if row is None or not domain.startswith('D05_'):
            raise ValueError('known D05 cold allocation domain required')
        if (row['nv'],row['nbody'],row['worlds'],row['model_sha256'])!=(nv,nbody,worlds,model_sha256):
            raise ValueError('cold allocation identity mismatch')
        bounded=row['feature']=='cold_bounded_stage'
        if bounded and getattr(device,'arch',None)!=120:
            raise ValueError('positive cold allocation measurement requires the tested SM120 device')
    shape=(worlds,1,1) if bounded else (worlds,nbody,33+7*nv)
    stage=wp.zeros(shape,dtype=scalar.wp_real,device=device)
    return SimpleNamespace(stages=stage,profile='bounded' if bounded else 'full',
        owns_bytes=int(np.prod(shape))*8,cold_allocation_only=True,hot_binding_qualified=False)
