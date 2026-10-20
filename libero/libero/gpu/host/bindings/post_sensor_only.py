"""Explicit opt-in sensor-only leaf binding, independent of legacy Post validity.

The caller supplies raw input views and new leaf output/work owners. It also
supplies the complete legacy mutable-owner inventory, which remains held. This
entry does not install adapter.enqueue or the full-force forward callback.
Only sensor/valid/written returned here belong to post_sensor_only_v1.
"""
import libero.gpu.precision as scalar
from pathlib import Path
from types import SimpleNamespace
import hashlib
import numpy as np
import warp as wp
from libero.gpu.host.paths import GPU_ROOT
from libero.gpu.host.build.native import NativePlan
from libero.gpu.host.bindings.post_sensor_only_abi import RAW_NAMES,RAW_DECLARATIONS,COLD_STATIC_SHAS,STATIC_NAMES
from libero.gpu.host.bindings.post_topology import sensor_closure
from libero.gpu.host.bindings.sensor_only_contract import FAMILIES,WRITE_FIELDS,validate_write_ranges

PROFILE='post_sensor_only_20261012_r01'


def bind(raw,*,force_mask,acc_mask,worlds,root,legacy_mutable_owners,
         profile=PROFILE,device='cuda:0'):
    if profile!=PROFILE:raise ValueError('unknown sensor-only qualification profile')
    if set(raw)!=set(RAW_NAMES):raise ValueError('sensor-only raw ABI fields')
    family=tuple(int(raw[k]) for k in ('r_b','r_v','r_ns','r_ng','r_cc','r_ec'))
    if family not in FAMILIES or worlds not in (1,2) or isinstance(worlds,bool):
        raise ValueError('sensor-only cold family unqualified')
    if raw['diagnostics']!=0:raise ValueError('sensor-only cannot publish legacy diagnostic validity')
    b,v,ns,ng,cc,ec=family
    if raw['c_bw']!=6*b or raw['c_sw']!=6 or raw['s_b']!=b or raw['s_ns']!=ns:
        raise ValueError('sensor-only geometry/layout contract')
    if raw['contact_index_mode']!=2 or raw['radix_min_contacts']!=256 or raw['topology_cache']!=1:
        raise ValueError('sensor-only measured contact-index configuration required')
    arrays={};scalars={}
    for name,decl in zip(RAW_NAMES,RAW_DECLARATIONS):
        a=raw[name]
        if '*' not in decl:
            if isinstance(a,bool) or not isinstance(a,(int,np.integer)):raise ValueError('sensor-only exact scalar: '+name)
            scalars[name]=int(a);continue
        dtype=wp.int64 if 'long long' in decl or 'int64_t' in decl else scalar.wp_real if 'real_t' in decl else wp.int32
        if not isinstance(a,wp.array) or a.dtype!=dtype or int(a.ptr)<=0 or int(a.ptr)%wp.types.type_size_in_bytes(dtype):
            raise ValueError('sensor-only typed owner: '+name)
        arrays[name]=a
    static=[np.ascontiguousarray(raw[n].numpy()) for n in STATIC_NAMES]
    if hashlib.sha256(b''.join(a.tobytes() for a in static)).hexdigest()!=COLD_STATIC_SHAS[family]:
        raise ValueError('sensor-only cold topology/sensor demand family unqualified')
    parent=raw['r_parent'].numpy().reshape(-1);depth=np.zeros(b,np.int32)
    for j in range(1,b):depth[j]=depth[int(parent[j])]+1
    rows=[[j for j in range(1,b) if depth[j]==d] for d in range(1,int(depth.max())+1)]
    child=[[j for j in range(b-1,0,-1) if parent[j]==p] for p in range(b)]
    def pack(rows):return np.array([0]+list(np.cumsum([len(r) for r in rows])),np.int32),np.array([j for r in rows for j in r],np.int32)
    lo,lb=pack(rows);co,ci=pack(child)
    if raw['nlevels']!=len(rows):raise ValueError('sensor-only cold level count')
    for name,expected in (('level_offsets',lo),('level_bodies',lb),('children_offsets',co),('children_indices',ci)):
        if raw[name].numpy().tobytes()!=expected.tobytes():raise ValueError('sensor-only exact topology table: '+name)
    # Current full-input CUDA validation remains inside every invocation.
    # These cold checks prove mask closure and typed owning allocation extents.
    closure=sensor_closure(raw['r_parent'].numpy(),raw['s_sb'].numpy(),raw['s_site'].numpy())
    for name,a,expected in (('force_mask',force_mask,closure['force_mask']),('acc_mask',acc_mask,closure['acceleration_mask'])):
        if not isinstance(a,wp.array) or a.dtype!=wp.int32 or a.shape!=(b,) or a.numpy().tobytes()!=expected.tobytes():
            raise ValueError('sensor-only exact closed mask: '+name)
        arrays[name]=a
    def span(a):return int(a.ptr),int(a.size)*wp.types.type_size_in_bytes(a.dtype)
    ranges={name:span(a) for name,a in arrays.items()}
    legacy=tuple(legacy_mutable_owners)
    if any(not isinstance(a,wp.array) for a in legacy):raise ValueError('legacy inventory requires owning arrays')
    validate_write_ranges(ranges,[span(a) for a in legacy])
    minimum={'g_first':worlds,'g_g':worlds,'r_active':worlds,'r_status':worlds,'s_status':worlds,
        'c_first':worlds,'c_valid':worlds,'c_written':worlds,'c_ps':6*worlds,
        'r_acc':6*b*worlds,'r_ext':6*b*worlds,'r_intr':6*b*worlds,'r_decoded':6*cc*worlds,
        'c_pe':6*b*worlds,'c_pi':6*b*worlds,'s_result':6*worlds,
        'body_index':2*cc*worlds,'body_offsets':(b+1)*worlds,'body_counts':b*worlds,
        'event_ping':2*cc*worlds,'event_pong':2*cc*worlds,'cached_event_body':2*cc*worlds,
        'cached_contact_meta':2*worlds}
    def extent(names,n):
        for name in names.split():minimum[name]=n
    extent('g_active g_dv g_nv g_ne g_nw g_na g_expected g_state g_consumed g_forward g_cf c_gate c_status c_ss c_epoch cache_status cache_valid cache_state cache_forward',worlds)
    extent('r_qv r_qa',worlds*v);extent('r_cd r_cdd',6*worlds*v)
    extent('r_ci',10*worlds*b);extent('r_cv r_xf cache_acc cache_force s_fr c_ext c_intr',6*worlds*b)
    extent('r_xi r_com s_com',3*worlds*b);extent('r_sp s_pos',3*worlds*ns)
    extent('r_sm s_mat',9*worlds*ns);extent('r_ef',worlds*ec)
    extent('r_nc r_ne r_neq s_active',worlds);extent('r_dim r_adr r_g1 r_g2',worlds*cc)
    extent('r_frame',9*worlds*cc);extent('r_pos',3*worlds*cc)
    extent('r_parent r_da r_dn r_root s_br',b);extent('r_gb',ng);extent('r_grav',3)
    extent('s_type s_site s_adr',2);extent('s_sb',ns);extent('c_sensor',6*worlds)
    extent('level_offsets',raw['nlevels']+1);extent('level_bodies children_indices',b-1)
    extent('children_offsets',b+1);extent('chain_bodies',b-1)
    extent('chain_level_offsets',raw['nchain_levels']+1)
    for name,n in minimum.items():
        if arrays[name].size<n:raise ValueError('sensor-only short owning view: '+name)
    values=[raw[n] for n in RAW_NAMES]+[0,0,worlds,force_mask,acc_mask]
    plan=NativePlan('post_sensor_only_v1',[GPU_ROOT/'cuda/postprocess/post_sensor_only_run.cuh'],
        [('post_local::entry',values,'block')],worlds,Path(root)/'sensor_only',device=device,
        block_size=64,dynamic_shared_bytes=(18*b+6)*8,stage_barriers=False)
    owners=tuple(arrays.items());scalar_items=tuple(scalars.items());legacy_spans=tuple(span(a) for a in legacy)
    def validate():
        for name,a in owners:
            value=force_mask if name=='force_mask' else acc_mask if name=='acc_mask' else raw[name]
            if value is not a or span(a)!=ranges[name]:raise ValueError('sensor-only owner drift: '+name)
        if any(raw[n]!=value for n,value in scalar_items):raise ValueError('sensor-only scalar drift')
        if tuple(span(a) for a in legacy)!=legacy_spans:raise ValueError('legacy held owner drift')
    def enqueue(stream=None):validate();plan.launch(stream)
    return SimpleNamespace(abi='post_sensor_only_v1',plan=plan,enqueue=enqueue,validate=validate,
        sensor=raw['c_ps'],valid=raw['c_valid'],written=raw['c_written'],first_error=raw['c_first'],
        validity_fields=('sensor',),legacy_force_validity_updated=False,
        qualification='raw P09 positive; actual new binding P09I pending',manifest=plan.manifest)
