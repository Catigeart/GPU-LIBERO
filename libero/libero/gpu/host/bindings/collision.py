"""Handwritten collision math; cold Python inspects only the retained pointer ABI.

One block owns each world. Bounds, SAP, stages and dispatch cooperate within
the block; live BVH stages traverse independent body pairs in parallel, then
publish the original request order. Diagnostic traversal retains its owner.
The native detector retains a separate pair-parallel kernel launch boundary.
"""
import libero.gpu.precision as scalar
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
import ast
import inspect
import textwrap
from pathlib import Path
from types import MethodType
from collections import UserDict
import numpy as np
import warp as wp
from libero.gpu.host.build.native import NativePlan
from libero.gpu.host.runtime.collision import chain
from libero.gpu.reference.collision.contact_finalize import kernel as finalize_module
from libero.gpu.reference.collision.contact_params import kernel as params_module
from libero.gpu.reference.collision.collision_tree import kernel as tree_module
from libero.gpu.reference.collision.collision_dispatch import kernel as dispatch_module

_NAMES = {
    '_input_error_kernel': ('cf_input_error','lane0'),
    '_effective_active_kernel': ('cf_effective','lane0'),
    '_bounds_kernel': ('cf_bounds','lane0'),
    '_broadphase_kernel': ('cf_broadphase_block','block'),
    '_stagebuild_kernel': ('cf_stagebuild_block','block'),
    '_tree_kernel': ('cf_tree','block'),
    '_req_to_dispatch_kernel': ('cf_req_dispatch','lane0'),
    '_dispatch': ('cf_dispatch_block','block'),
    '_dispatch_to_calls_kernel': ('cf_dispatch_calls','lane0'),
    '_reset_error_kernel': ('cf_reset_error','lane0'),
    '_detect_error_kernel': ('cf_detect_error','lane0'),
    '_calls_to_finalize_kernel': ('cf_calls_finalize','block'),
    '_dedup': ('cf_dedup','block'),
    '_complete': ('cf_params','block'),
    '_assemble': ('cf_assemble','lane0'),
    '_chain_error_kernel': ('cf_chain_error','lane0'),
    '_restore_failed_contacts_kernel': ('cf_restore_contacts','block'),
    '_body_counters_kernel': ('cf_body_counters_block','block'),
}
_SUFFIX={wp.int32:'i32',scalar.wp_real:'f64',wp.float32:'f32'}


def _body(method):
    return ast.parse(textwrap.dedent(inspect.getsource(method))).body[0].body


def _eval(node, owner, globals_):
    return eval(compile(ast.Expression(node),'<collision retained ABI>','eval'),
                globals_,{'self':owner,'s':getattr(owner,'static',None),'N':owner.nworld})


def _stage_call(node,owner,globals_):
    call=node.value
    kernel=call.args[0].id
    if kernel not in _NAMES:
        raise RuntimeError('unreviewed collision kernel ABI: '+kernel)
    args=next(k.value for k in call.keywords if k.arg=='inputs')
    name,mode=_NAMES[kernel]
    return ('collision_cuda::'+name,[_eval(a,owner,globals_) for a in args.elts],mode)


def _workspace_stages(ws):
    """Read launch argument lists only; no Python arithmetic becomes CUDA."""
    stages=[]
    for node in _body(chain.CollisionChainWorkspace.launch):
        if isinstance(node,ast.For) and ast.unparse(node.iter)=='self.outputs.items()':
            restore=any(isinstance(n,ast.Assign) for n in node.body)
            for name,arr in ws.outputs.items():
                suffix=_SUFFIX[arr.dtype];before=ws.before_outputs[name]
                if restore:
                    stages.append(('collision_cuda::cf_restore_'+suffix,[ws.active,before,arr,arr.size//ws.nworld],'block'))
                else: stages.append(('collision_cuda::cf_copy_'+suffix,[arr,before,arr.size//ws.nworld],'block'))
            continue
        if not isinstance(node,ast.Expr) or not isinstance(node.value,ast.Call):continue
        call=node.value;name=ast.unparse(call.func)
        if name=='wp.launch':stages.append(_stage_call(node,ws,vars(chain)))
        elif name=='wp.copy':
            dst,src=(_eval(a,ws,vars(chain)) for a in call.args[:2])
            if dst.size!=src.size:raise ValueError('collision copy size changed')
            stages.append(('collision_cuda::cf_copy_'+_SUFFIX[dst.dtype],[src,dst,dst.size//ws.nworld],'block'))
        elif name=='self._stage_guard':
            status=_eval(call.args[0],ws,vars(chain));stage=_eval(call.args[1],ws,vars(chain))
            codes=tuple(_eval(call.args[2],ws,vars(chain)))+(-999,)*3
            stages.append(('collision_cuda::cf_stage_error',[status,ws.active,stage,*codes[:3],
                           ws.fe_stage,ws.fe_code,ws.fe_item,ws.effective_active],'lane0'))
        elif name=='self.detect.launch':stages.append(None)
        elif name in ('self.tree.launch','self.dispatch.launch'):
            owner=ws.tree if name=='self.tree.launch' else ws.dispatch
            module=tree_module if name=='self.tree.launch' else dispatch_module
            method=module.CollisionTreeWorkspace.launch if name=='self.tree.launch' else module.CollisionDispatchWorkspace.launch
            for inner in _body(method):
                if isinstance(inner,ast.Expr) and isinstance(inner.value,ast.Call) and ast.unparse(inner.value.func)=='wp.launch':
                    stages.append(_stage_call(inner,owner,vars(module)))
        elif name=='self.finalize.launch':
            for inner in _body(finalize_module.ContactFinalizeWorkspace.launch):
                if not isinstance(inner,ast.Expr) or not isinstance(inner.value,ast.Call):continue
                if ast.unparse(inner.value.func)=='wp.launch':stages.append(_stage_call(inner,ws.finalize,vars(finalize_module)))
                elif ast.unparse(inner.value.func)=='self.params.launch':
                    for p in _body(params_module.ContactParamsWorkspace.launch):
                        if isinstance(p,ast.Expr) and isinstance(p.value,ast.Call) and ast.unparse(p.value.func)=='wp.launch':
                            stages.append(_stage_call(p,ws.finalize.params,vars(params_module)))
    if stages.count(None)!=1:raise RuntimeError('exactly one collision detector boundary required')
    return stages


def _dataflow_storage(ws):
    """Private transactional contacts; public witnesses retain their owners."""
    if not hasattr(ws,'_cuda_collision_dataflow_storage'):
        n=ws.nworld*ws.contact_capacity
        fields=65 if (getattr(ws,'_cuda_collision_dataflow_profile','compatible')!='fast' or
                      getattr(ws,'_cuda_collision_diagnostics',True)) else 28
        # A retained private owner is outside the immutable public ABI
        # inventory. NativePlan validates each arena pointer independently.
        ws._cuda_collision_dataflow_storage=UserDict({
            'contact_f':wp.empty(n*fields,dtype=scalar.wp_real,device=ws.device),
            'contact_i':wp.empty(n*5,dtype=wp.int32,device=ws.device),
            'order':wp.empty(n,dtype=wp.int32,device=ws.device),
            'order_scratch':wp.empty(n,dtype=wp.int32,device=ws.device),
            'offsets':wp.empty(ws.nworld*ws.call_capacity,dtype=wp.int32,device=ws.device),
            'frame_error':wp.empty(ws.nworld*ws.call_capacity,dtype=wp.int32,device=ws.device)})
    return ws._cuda_collision_dataflow_storage


def _pair_parameter_cache(ws,params,storage):
    """Cold immutable-model cache; general mutable workspaces mix on demand.

    Native model updates construct new GPU owners and therefore a new table.
    Explicit pairs and caller margins use the original device mixer. Cap the
    optional dense table at 17 MiB; large models simply keep the direct path.
    """
    ngeom=params[27]
    root=getattr(ws,'_cuda_collision_pair_parameter_root',None)
    immutable=(getattr(ws,'_cuda_collision_static_parameters',False) or
               getattr(ws,'_collision_cold_admission',None) is not None)
    if root is None or not immutable or not 0<ngeom*ngeom<=131072:
        return [storage['contact_f'],storage['contact_i'],0]
    if not hasattr(ws,'_cuda_collision_pair_parameter_cache'):
        f=wp.empty(ngeom*ngeom*16,dtype=scalar.wp_real,device=ws.device)
        i=wp.empty(ngeom*ngeom*2,dtype=wp.int32,device=ws.device)
        plan=NativePlan('collision_pair_parameters',
            [GPU_ROOT/'cuda/collision/collision_params.cuh'],
            [('collision_cuda::cf_pair_params_table',[*params[5:24],ngeom,params[28],f,i],'block')],
            1,root,device=ws.device)
        plan.launch(wp.get_stream(ws.device))
        # One cold synchronization publishes this model-only table to any
        # later consumer stream. There is no per-step snapshot or comparison.
        wp.synchronize_stream(wp.get_stream(ws.device))
        # Private owners follow the same arena boundary as candidate storage;
        # each NativePlan independently retains and validates their pointers.
        ws._cuda_collision_pair_parameter_cache=UserDict({'f':f,'i':i,'plan':plan})
        ws._cuda_collision_pair_parameter_plan=plan
    cache=ws._cuda_collision_pair_parameter_cache
    return [cache['f'],cache['i'],ngeom]


def prepare_bounds_cache(ws,dynamics):
    """Cold descriptor for F's shared-geom tail and C's guarded consumer."""
    if hasattr(ws,'_cuda_collision_plans'):
        raise ValueError('F bounds producer must be connected before C compilation')
    geom=dynamics.position.frames.geom_xpos
    if geom.ptr!=ws.xpos.ptr or geom.size!=ws.nworld*ws.ngeom*3 or dynamics.capacity!=ws.nworld:
        ws._cuda_collision_bounds_cache_reason='C geom owner/world stride differs from the F producer'
        return None
    if not hasattr(ws,'_cuda_collision_bounds_cache'):
        ws._cuda_collision_bounds_cache=UserDict({name:wp.empty(ws.nworld*width,dtype=dtype,device=ws.device)
            for name,width,dtype in (
                ('aabb',ws.nbody*6,scalar.wp_real),('frame',9,scalar.wp_real),('scratch',34,scalar.wp_real),
                ('entered',1,wp.int32),('count',1,wp.int32),('status',1,wp.int32),('options',4,wp.int32))})
        ws._cuda_collision_bounds_cache['epoch']=wp.full(ws.nworld,-1,dtype=wp.int64,device=ws.device)
        for name in ('prepare_count','hit_count','fallback_count'):
            ws._cuda_collision_bounds_cache[name]=wp.zeros(ws.nworld,dtype=wp.int64,device=ws.device)
        ws._cuda_collision_bounds_cache['dynamics']=dynamics
    b=ws._cuda_collision_bounds_cache
    args=[ws.xpos,dynamics.valid['position'],dynamics.written['position'],dynamics.forward_epoch,dynamics.first_error,
          ws.gates,ws.collision_mode,ws.geom_rbound,ws.geom_margin,ws.body_geomadr,ws.body_geomnum,ws.geom_bodyid,
          ws.aabb,ws.frame,ws.bounds_scratch,b['aabb'],b['frame'],b['scratch'],b['entered'],b['count'],b['status'],
          b['epoch'],b['options'],b['prepare_count'],ws.ngeom,ws.nbody]
    return {'bound':True,'function':'collision_cuda::cf_bounds_prepare_cached_block','args':args,
            'geom_position_arg':0,'sources':[(GPU_ROOT / 'cuda/collision/collision_bounds.cuh')],
            'cache_epoch':b['epoch'],'arithmetic':'original PCA/order; shared world-local geom input'}


def _temporal_position_signature(dynamics):
    owners=(dynamics.position.frames.geom_xpos,dynamics.position.frames.geom_xmat,
            dynamics.valid['position'],dynamics.written['position'],
            dynamics.forward_epoch,dynamics.first_error)
    return (id(dynamics),tuple((id(a),int(a.ptr),int(a.size),str(a.dtype),str(a.device)) for a in owners))


def configure_temporal_front(ws,dynamics,*,dataflow_profile='compatible',temporal=False,
                             absolute_padding=0.005,relative_padding=0.01,max_age=32,
                             max_cache_bytes=8*1024**2):
    """Admit before F compilation so an admitted C front owns its bounds work.

    The helper validates real original stage argument owners and capacities.
    A declined request keeps F's PCA producer and the original C consumer.
    """
    from libero.gpu.host.runtime.core.algorithm_config import CollisionAlgorithmOptions
    policy=CollisionAlgorithmOptions(temporal,absolute_padding,relative_padding,max_age,max_cache_bytes)
    if dataflow_profile not in ('compatible','fast'):
        raise ValueError('collision dataflow profile')
    settings=(policy,dataflow_profile,id(dynamics))
    previous=getattr(ws,'_cuda_collision_temporal_configuration',None)
    if previous is not None:
        if previous!=settings:
            raise ValueError('temporal collision config/position owner change requires a cold rebuild')
        _validate_temporal_position_owner(ws)
        return temporal_front_manifest(ws)
    if temporal and hasattr(ws,'_cuda_collision_plans'):
        raise ValueError('temporal collision must be configured before C compilation')
    ws._cuda_collision_temporal_configuration=settings
    ws._cuda_collision_temporal_options=policy
    ws._cuda_collision_temporal_dynamics=dynamics
    ws._cuda_collision_temporal_sources=()
    if temporal:
        from .collision_temporal import prepare_temporal_collision_stages
        _,sources=prepare_temporal_collision_stages(ws,_workspace_stages(ws),dynamics=dynamics,
            profile=dataflow_profile,absolute_padding=policy.absolute_padding,
            relative_padding=policy.relative_padding,max_age=policy.max_age,
            max_cache_bytes=policy.max_cache_bytes)
        ws._cuda_collision_temporal_sources=tuple(sources)
        if sources:
            ws._cuda_collision_temporal_position_signature=_temporal_position_signature(dynamics)
    else:
        ws._cuda_collision_temporal_reason='temporal broadphase not requested'
    return temporal_front_manifest(ws)


def _validate_temporal_position_owner(ws):
    if getattr(ws,'_cuda_collision_temporal_sources',()):
        if _temporal_position_signature(ws._cuda_collision_temporal_dynamics)!=ws._cuda_collision_temporal_position_signature:
            raise ValueError('temporal position owner/schema changed; cold rebuild required')


def temporal_front_manifest(ws):
    storage=getattr(ws,'_cuda_collision_temporal',None)
    policy=getattr(ws,'_cuda_collision_temporal_options',None)
    return dict(requested=bool(policy and policy.temporal),
        admitted=bool(getattr(ws,'_cuda_collision_temporal_sources',())),
        reason=getattr(ws,'_cuda_collision_temporal_reason','temporal broadphase not configured'),
        proof=storage['proof'] if storage is not None else None,
        position_owner='sealed dynamics position arrays and epoch' if storage is not None else None,
        f_pca_producer_omitted=bool(getattr(ws,'_cuda_collision_temporal_f_pca_omitted',False)),
        threshold_status='experimental configurable spatial/age policy; not a measured winner')


def _dataflow_stages(ws,transport=None):
    """Retain the public ABI while changing the actual producer/consumer path."""
    from libero.gpu.host.build.collision_metadata import prepare_collision_metadata
    metadata=prepare_collision_metadata(ws)
    stages=_workspace_stages(ws);storage=_dataflow_storage(ws);result=[]
    diagnostics=(getattr(ws,'_cuda_collision_dataflow_profile','compatible')!='fast' or
                 getattr(ws,'_cuda_collision_diagnostics',True))
    shadows={id(a) for a in ws.before_outputs.values()}
    index=0
    while index<len(stages):
        stage=stages[index];index+=1
        if stage is None:result.append(None);continue
        fn,args,mode=stage[:3]
        if fn.startswith('collision_cuda::cf_copy_'):
            src,dst,_=args
            if src.ptr==dst.ptr:continue
            # Fast requested-world guards already preserve every mask-0 output.
            # These shadows only feed the now-redundant inactive restore below;
            # contact failure transactions use private candidates instead.
            if not diagnostics and id(dst) in shadows:continue
            # Tree stage/count/mask are retained inputs, not public outputs.
            # Pass the actual producer owner to cf_tree instead of copying.
            if any(dst is value for value in (ws.tree.stage,ws.tree.stage_count,ws.tree.active_in)):
                continue
        if not diagnostics and fn in ('collision_cuda::cf_restore_f64',
                'collision_cuda::cf_restore_i32','collision_cuda::cf_restore_f32'):
            if args[0] is not ws.active or id(args[1]) not in shadows:
                raise ValueError('unexpected inactive collision restore owner')
            continue
        if fn=='collision_cuda::cf_tree':
            args=[ws.stage,ws.stage_count,ws.effective_active,*args[3:]]
            if 'pair_key_offsets' in metadata:
                args += [metadata['pair_key_offsets'],metadata['pair_key_indices'],metadata['pair_shadow']]
            stage=(fn if diagnostics else 'collision_cuda::cf_tree_live',args,mode)
        elif fn=='collision_cuda::cf_dispatch_block' and 'geom_pair_route' in metadata:
            stage=(fn,[*args,metadata['geom_pair_route'],metadata['route_shadow']],mode)
        elif fn=='collision_cuda::cf_bounds' and hasattr(ws,'_cuda_collision_bounds_cache'):
            b=ws._cuda_collision_bounds_cache;d=b['dynamics']
            extra=[b['aabb'],b['frame'],b['scratch'],b['entered'],b['count'],b['status'],b['epoch'],b['options'],
                   d.valid['position'],d.written['position'],d.forward_epoch,d.first_error,b['hit_count'],b['fallback_count']]
            stage=('collision_cuda::cf_bounds_cached_block',[*args,*extra],'block')
        elif fn=='collision_cuda::cf_bounds':
            stage=('collision_cuda::cf_bounds_block',args,'block')
        elif fn=='collision_cuda::cf_req_dispatch':
            stage=('collision_cuda::cf_req_dispatch_ordered_block',args,'block')
        elif fn=='collision_cuda::cf_dispatch_calls':
            stage=('collision_cuda::cf_dispatch_calls_block',args,'block')
        elif fn=='collision_cuda::cf_calls_finalize' and not diagnostics:
            stage=('collision_cuda::cf_calls_finalize_live',args,'block')
        if fn=='collision_cuda::cf_dedup':
            following=stages[index]
            if following is None or following[0]!='collision_cuda::cf_params':
                raise RuntimeError('dedup / params stage adjacency changed')
            p=following[1]
            if p[0] is not args[5] or p[1] is not args[8] or p[3] is not args[3]:
                raise ValueError('dedup / params retained producer owners changed')
            if p[26]!=args[12]*50 or p[27]!=args[7]:
                raise ValueError('chain pairwise parameter shape changed')
            stage=('collision_cuda::cf_dedup_params'+('' if diagnostics else '_live'),args+[p[2],*p[4:27],p[28],storage['offsets'],
                    storage['frame_error'],storage['contact_f'],storage['contact_i'],ws.contact_capacity]+
                    ([] if diagnostics else _pair_parameter_cache(ws,p,storage)),'block')
            index+=1
        elif fn=='collision_cuda::cf_assemble':
            args=[args[0],*args[2:7],args[8],args[9],storage['frame_error'],storage['contact_f'],
                  storage['contact_i'],*args[12:21],storage['order'],storage['order_scratch']]
            stage=('collision_cuda::cf_assemble_candidates_block' if diagnostics else
                   'collision_cuda::cf_assemble_candidates_live',args,'block')
        elif fn=='collision_cuda::cf_restore_contacts':
            # No public contact was changed before the world-wide verdict.
            continue
        result.append(stage)
    if transport is None:
        result.append(('collision_cuda::cf_commit_contacts' if diagnostics else
                       'collision_cuda::cf_commit_contacts_live',
            [ws.active,ws.fe_code,ws.finalize.ncon,storage['contact_f'],storage['contact_i'],
             ws.finalize.contact_f,ws.finalize.contact_i,ws.contact_capacity],'block'))
    else:
        d,x=transport.dynamics,transport.inputs
        if transport.collision.workspace is not ws:
            raise ValueError('direct transport belongs to another collision workspace')
        args=[d.active,d.valid['position'],d.first_error,ws.fe_code,ws.effective_active,ws.finalize.ncon,
              storage['contact_f'],storage['contact_i'],ws.finalize.contact_f,ws.finalize.contact_i,
              transport.active,transport.ncon,
              *[x[n] for n in ('dist','pos','frame','margin','friction','solref','solreffriction','solimp','dim','geom1','geom2','exclude')],
              transport.source_capacity,transport.capacity]
        result.append(('collision_cuda::cf_commit_contacts_transport' if diagnostics else
                       'collision_cuda::cf_commit_contacts_transport_live',args,'block'))
    if getattr(ws,'_cuda_collision_temporal_sources',()):
        from .collision_temporal import prepare_temporal_collision_stages
        policy=ws._cuda_collision_temporal_options
        result,sources=prepare_temporal_collision_stages(ws,result,
            dynamics=ws._cuda_collision_temporal_dynamics,
            profile=ws._cuda_collision_temporal_configuration[1],
            absolute_padding=policy.absolute_padding,relative_padding=policy.relative_padding,
            max_age=policy.max_age,max_cache_bytes=policy.max_cache_bytes)
        if tuple(sources)!=ws._cuda_collision_temporal_sources:
            raise ValueError('cold temporal front admission changed during C construction')
    return result


def _source_files(ws=None):
    return [GPU_ROOT / 'cuda/collision' / ('collision_'+name+'.cuh') for name in
            ('front','bounds','tree','dispatch','finalize','params','pipeline','transport')]+list(
                getattr(ws,'_cuda_collision_temporal_sources',()))


def _guard_inactive(ws,stages):
    """Cold ABI wrappers apply a uniform requested-world write predicate.

    These wrappers contain no generated arithmetic. The effective mask and
    the consumer admission tail remain unconditional so stale success cannot
    survive a masked invocation. Compatible diagnostic mode is unchanged.
    """
    if getattr(ws,'_cuda_collision_dataflow_profile','compatible')!='fast':return stages,''
    mapped=[];source=['namespace collision_cuda {']
    types={wp.int32:'int32_t*',wp.uint32:'uint32_t*',wp.int64:'int64_t*',wp.float32:'float*',scalar.wp_real:'real_t*'}
    for index,stage in enumerate(stages):
        if stage is None or stage[0] in ('collision_cuda::cf_effective',
                'collision_cuda::cf_commit_contacts_transport',
                'collision_cuda::cf_commit_contacts_transport_live'):
            mapped.append(stage);continue
        fn,args,mode=stage[:3];specs=stage[3] if len(stage)>3 else None
        c_types=[(specs[i] if specs else None) or
                 (types[value.dtype] if isinstance(value,wp.array) else
                  ('real_t' if isinstance(value,(float,np.floating)) else 'int32_t'))
                 for i,value in enumerate(args)]
        name='cf_requested_stage_'+str(index)
        declarations=', '.join(t+' a'+str(i) for i,t in enumerate(c_types))
        source.append('__device__ inline void '+name+'(int w,int* requested'+
                      (', '+declarations if declarations else '')+') {\n'+
                      '    if(requested[w]==0)'+ ('{if(threadIdx.x==0)((libero_collision_cuda::Params*)a0)->task_count[w]=0;return;}' if fn=='libero_collision_cuda::detect_prepare_pointer' else 'return;')+'\n    '+fn+'(w'+
                      (', '+', '.join('a'+str(i) for i in range(len(args))) if args else '')+');\n}')
        mapped.append(('collision_cuda::'+name,[ws.active,*args],mode,['int32_t*',*c_types]))
    source.append('}')
    return mapped,'\n'.join(source)


def install_workspace(ws,root,*,dataflow_profile=None,complete_only=False,diagnostics=True):
    if type(complete_only) is not bool:raise TypeError('complete_only must be bool')
    if type(diagnostics) is not bool:raise TypeError('diagnostics must be bool')
    if complete_only and dataflow_profile!='fast':
        raise ValueError('complete_only requires explicit fast cold dataflow')
    if hasattr(ws,'_cuda_collision_plans') and getattr(ws,'_cuda_collision_complete_only',False)!=complete_only:
        raise ValueError('complete_only mode change requires a cold workspace rebuild')
    if hasattr(ws,'_cuda_collision_plans') and ws._cuda_collision_diagnostics!=diagnostics:
        raise ValueError('collision diagnostics change requires a cold workspace rebuild')
    ws._cuda_collision_diagnostics=diagnostics
    if dataflow_profile is not None:
        if dataflow_profile not in ('compatible','fast'):raise ValueError('collision dataflow profile')
        if hasattr(ws,'_cuda_collision_plans') and getattr(ws,'_cuda_collision_dataflow_profile','compatible')!=dataflow_profile:
            raise ValueError('dataflow profile change requires a cold workspace rebuild')
        ws._cuda_collision_dataflow_profile=dataflow_profile
    ws._cuda_collision_pair_parameter_root=Path(root)/'pair_parameters'
    if hasattr(ws,'_cuda_collision_plans'):return ws._cuda_collision_manifest
    stages,guards=_guard_inactive(ws,_dataflow_stages(ws));cut=stages.index(None)
    files=_source_files(ws)
    if guards:files.append(guards)
    # Combined installation consumes these exact stages directly, for both
    # retained legacy storage and the pending handwritten-only storage ABI.
    # Before finalization, the temporary launch rejects incomplete bindings.
    # Keep the reset plan: it remains a public lifecycle consumer.
    plans=[] if complete_only else [NativePlan('collision_before_detect',files,stages[:cut],ws.nworld,Path(root)/'before',device=ws.device),
           NativePlan('collision_after_detect',files,stages[cut+1:],ws.nworld,Path(root)/'after',device=ws.device)]
    reset_plan=NativePlan('collision_reset_errors',[GPU_ROOT/'cuda/collision/collision_front.cuh'],
        [('collision_cuda::cf_reset_error',[ws.reset_mask,ws.fe_stage,ws.fe_code,ws.fe_item],'lane0')],
        ws.nworld,Path(root)/'reset',device=ws.device)
    old_launch=ws.launch
    def launch(self,stream=None):
        if self.borrow_detector_raw and (self._borrowed_raw_owner is not self.detect or self.finalize.raw is not self.detect.raw):
            raise ValueError('borrowed detector/finalizer raw identity changed')
        binding=self._device_input_binding
        stream=stream or (binding.stream if binding is not None else wp.get_stream(self.device))
        self.validate_device_bindings(stream)
        _validate_temporal_position_owner(self)
        if not plans:raise RuntimeError('complete-only workspace requires finalized complete plan')
        self._launch_started=True
        plans[0].launch(stream);self.detect.launch(stream=stream);plans[1].launch(stream)
    ws._cuda_collision_old_launch=old_launch
    ws._cuda_collision_complete_only=complete_only
    ws._cuda_collision_plans=plans
    ws.launch=MethodType(launch,ws)
    def reset_errors(self,mask,*,stream=None):
        value=np.asarray(mask)
        if value.dtype!=np.int32 or value.shape!=(self.nworld,) or not np.all((value==0)|(value==1)):
            raise ValueError('reset mask schema')
        binding=self._device_input_binding
        stream=stream or (binding.stream if binding is not None else wp.get_stream(self.device))
        self.validate_device_bindings(stream)
        with wp.ScopedStream(stream):self.reset_mask.assign(value)
        reset_plan.launch(stream)
    ws._cuda_collision_old_reset_errors=ws.reset_errors
    ws._cuda_collision_reset=reset_plan
    ws.reset_errors=MethodType(reset_errors,ws)
    ws._cuda_collision_manifest={'schema':'handwritten-collision-r01','plans':[p.manifest for p in plans],
        'stages_before_detect':[s[0] for s in stages[:cut]],'stages_after_detect':[s[0] for s in stages[cut+1:]],
        'detector_boundary':'pair parallel native CUDA launch','capacities':{k:getattr(ws,k) for k in
          ('nworld','nbody','ngeom','call_capacity','pair_capacity','stage_slots','req_slots','contact_capacity')},
        'source_math_translation':False,
        'capacity_diagnostics':getattr(ws,'_cuda_collision_dataflow_profile','compatible')!='fast' or diagnostics,
        'dataflow_profile':getattr(ws,'_cuda_collision_dataflow_profile','compatible'),
        'private_candidate_bytes':ws.nworld*(ws.contact_capacity*((65 if diagnostics or
            getattr(ws,'_cuda_collision_dataflow_profile','compatible')!='fast' else 28)*8+5*4+2*4)+ws.call_capacity*8)}
    ws._cuda_collision_manifest['reset_plan']=reset_plan.manifest
    cache=getattr(ws,'_cuda_collision_pair_parameter_cache',None)
    ws._cuda_collision_manifest['candidate_float_fields']=65 if (
        diagnostics or getattr(ws,'_cuda_collision_dataflow_profile','compatible')!='fast') else 28
    ws._cuda_collision_manifest['pair_parameter_cache']={
        'enabled':cache is not None,
        'bytes':cache['f'].size*8+cache['i'].size*4 if cache is not None else 0,
        'lifetime':'immutable model; rebuilt by native model replacement',
        'plan':cache['plan'].manifest if cache is not None else None}
    if complete_only:ws._cuda_collision_manifest['cold_installation_abi']='complete-only-v1'
    ws._cuda_collision_manifest['temporal_front']=temporal_front_manifest(ws)
    metadata=getattr(ws,'_cuda_collision_metadata',{})
    ws._cuda_collision_manifest['branch_metadata']=metadata.get('proof',{
        'admission':'scalar fallback','reason':getattr(ws,'_cuda_collision_metadata_reason','unavailable')})
    return ws._cuda_collision_manifest


def install(bundle,root,*,dataflow_profile='compatible',temporal=False,
            absolute_padding=0.005,relative_padding=0.01,max_age=32,max_cache_bytes=8*1024**2,
            diagnostics=True,static_parameters=False,complete_only=False):
    if dataflow_profile not in ('compatible','fast'):raise ValueError('collision dataflow profile')
    adapter=bundle.runtime.forward.c
    ws=getattr(adapter,'workspace',adapter)
    # Native batch ownership changes physical parameters only by rebuilding
    # the bundle. General component callers retain on-demand mixing by default.
    ws._cuda_collision_static_parameters=static_parameters
    configure_temporal_front(ws,bundle.dynamics,dataflow_profile=dataflow_profile,temporal=temporal,
        absolute_padding=absolute_padding,relative_padding=relative_padding,max_age=max_age,
        max_cache_bytes=max_cache_bytes)
    result=install_workspace(ws,Path(root)/'collision',dataflow_profile=dataflow_profile,
                             diagnostics=diagnostics,complete_only=complete_only)
    transport=getattr(bundle.runtime.forward,'transport',None)
    if transport is not None and not complete_only:
        result['transport']=install_transport(transport,Path(root)/'collision_transport')
    return result


def install_transport(transport,root):
    d,x=transport.dynamics,transport.inputs
    args=[d.active,d.valid['position'],d.first_error,transport.sources['error'],transport.sources['active'],
          transport.sources['ncon'],transport._flat_f,transport._flat_i,transport.active,transport.ncon,
          *[x[n] for n in ('dist','pos','frame','margin','friction','solref','solreffriction','solimp','dim','geom1','geom2','exclude')],
          transport.source_capacity,transport.capacity]
    plan=NativePlan('collision_contact_transport',[(GPU_ROOT / 'cuda/collision/collision_transport.cuh')],
          [('collision_cuda::cf_transport',args,'block')],transport.nworld,root,device=transport.device)
    def enqueue(self):self.validate();plan.launch(self.stream)
    transport._cuda_collision_transport=plan
    transport.enqueue=MethodType(enqueue,transport)
    return plan.manifest


def make_compact_query_plans(ws,root,*,transport=None,producer_blocks=32,query_schedule='compact',world_guard=None):
    """Cold opt-in mapping; no host count download or cross-CTA barrier.

    Same-stream boundaries publish the private producer results before the
    existing ordered detector commit and C transaction finalizer.
    """
    if isinstance(producer_blocks,bool) or not isinstance(producer_blocks,int) or not 1<=producer_blocks<=2147483647:
        raise ValueError('positive integer producer blocks required')
    from .collision_detect import device_source
    detector=ws.detect._cuda_detector_plan
    stages=_dataflow_stages(ws,transport);cut=stages.index(None)
    before=stages[:cut]+[('libero_collision_cuda::detect_prepare_pointer',[detector.device_params],'block',['void*'])]
    after=[('libero_collision_cuda::detect_commit_pointer',[detector.device_params],'block',['void*'])]+stages[cut+1:]
    before,g0=_guard_inactive(ws,before);after,g1=_guard_inactive(ws,after)
    # Predicate wrapper names are local to each compilation unit.
    sources=_source_files(ws)+[device_source()]
    front=NativePlan('collision_query_front',sources+([g0] if g0 else []),before,
        ws.nworld,Path(root)/'front',block_size=64,device=ws.device,extra_flags=('-rdc=true',),world_guard=world_guard)
    tail=NativePlan('collision_query_tail',sources+([g1] if g1 else []),after,
        ws.nworld,Path(root)/'tail',block_size=64,device=ws.device,extra_flags=('-rdc=true',),world_guard=world_guard)
    def launch(stream):
        ws.validate_device_bindings(stream);detector.validate(stream)
        _validate_temporal_position_owner(ws)
        if transport is not None:transport.validate()
        if ws.detect._cuda_detector_plan is not detector:raise ValueError('compact detector owner changed')
        # Prevalidate both NativePlan bindings before the first GPU write.
        front.validate_bindings();tail.validate_bindings()
        front.launch(stream)
        producer=(detector.launch_parallel_producer if query_schedule=='parallel' else detector.launch_compact_producer)
        producer(producer_blocks,stream,world_guard)
        tail.launch(stream)
    return launch,{'front':front,'tail':tail,'detector':detector,'producer_blocks':producer_blocks,
        'grid_mapping':'front=(1,W); producer=(blocks,W,1); tail=(1,W)',
        'task_order':'device compact route tiles; original-slot output; ordered first-error commit',
        'task_capacity':detector.task_capacity,'kernel_launches':3}


def install_combined_workspace(ws,root,*,transport=None,query_schedule='fused',producer_blocks=32,world_guard=None):
    """Fuse the complete C function; all stages share one world/block.

    Device detector queries distribute over the same 64 lanes. Their outputs
    become visible at the detector's internal barrier, and the NativePlan
    stage barrier completes its ordered public commit before finalization.
    No stage reads another world's scratch or requires a grid-wide barrier.
    """
    if query_schedule not in ('fused','compact','parallel'):raise ValueError('collision query schedule')
    if hasattr(ws,'_cuda_collision_complete'):
        if getattr(ws,'_cuda_collision_world_guard',(None,))[0] is not world_guard:raise ValueError('collision world guard owner changed; cold rebuild required')
        if getattr(ws,'_cuda_collision_query_schedule','fused')!=query_schedule:raise ValueError('query schedule change requires cold rebuild')
        if query_schedule!='fused' and ws._cuda_collision_compact['producer_blocks']!=producer_blocks:raise ValueError('producer blocks change requires cold rebuild')
        if transport is not None and getattr(ws,'_cuda_collision_direct_transport',None) is not transport:
            raise ValueError('direct C/S transport change requires a cold workspace rebuild')
        return ws._cuda_collision_manifest
    if not hasattr(ws.detect,'_cuda_detector_plan'):
        raise ValueError('install the handwritten detector before complete collision fusion')
    if not hasattr(ws,'_cuda_collision_plans'):install_workspace(ws,Path(root)/'partial')
    from .collision_detect import device_source
    detector=ws.detect._cuda_detector_plan
    detector_params=detector.device_params
    stages=_dataflow_stages(ws,transport);cut=stages.index(None)
    merged=stages[:cut]+[('libero_collision_cuda::detect_pointer',[detector_params],'block',['void*'])]+stages[cut+1:]
    merged,guards=_guard_inactive(ws,merged)
    files=_source_files(ws)
    # Detector source defines detect_pointer before predicate wrappers call it.
    sources=files+[device_source()]+([guards] if guards else [])
    plan=(NativePlan('collision_complete',sources,merged,ws.nworld,root,
                    block_size=64,device=ws.device,extra_flags=('-rdc=true',),world_guard=world_guard)
          if query_schedule=='fused' else None)
    ws._cuda_collision_partial_plans=ws._cuda_collision_plans
    ws._cuda_collision_complete=plan
    # Retained plan owns this pointer. Do not extend the sealed public array ABI.
    ws._cuda_collision_world_guard=(world_guard,)
    ws._cuda_collision_plans=[plan]
    def launch(self,stream=None):
        if self.borrow_detector_raw and (self._borrowed_raw_owner is not self.detect or self.finalize.raw is not self.detect.raw):
            raise ValueError('borrowed detector/finalizer raw identity changed')
        binding=self._device_input_binding
        stream=stream or (binding.stream if binding is not None else wp.get_stream(self.device))
        self.validate_device_bindings(stream)
        _validate_temporal_position_owner(self)
        if self.detect._cuda_detector_plan is not detector:raise ValueError('complete collision detector owner changed')
        if detector.device_params is not detector_params:raise ValueError('complete collision detector params owner changed')
        if transport is not None:transport.validate()
        detector.validate(stream)
        self._launch_started=True
        plan.launch(stream)
    ws._cuda_collision_query_schedule=query_schedule
    if query_schedule!='fused':
        compact_launch,compact=make_compact_query_plans(ws,Path(root)/query_schedule,transport=transport,
            producer_blocks=producer_blocks,query_schedule=query_schedule,world_guard=world_guard)
        ws._cuda_collision_compact=compact
        ws._cuda_collision_plans=[compact['front'],compact['tail']]
        def launch(self,stream=None):
            if self.borrow_detector_raw and (self._borrowed_raw_owner is not self.detect or self.finalize.raw is not self.detect.raw):
                raise ValueError('borrowed detector/finalizer raw identity changed')
            binding=self._device_input_binding
            stream=stream or (binding.stream if binding is not None else wp.get_stream(self.device))
            self._launch_started=True;compact_launch(stream)
    ws.launch=MethodType(launch,ws)
    prior=ws._cuda_collision_manifest
    ws._cuda_collision_manifest={**prior,'schema':'handwritten-complete-collision-r02',
        'plans':[plan.manifest] if plan is not None else [],'detector_boundary':'intra-block device function',
        'stage_order':[s[0] for s in merged],'kernel_launches':1,
        'detector_params_owner':'retained DetectorPlan.device_params',
        'ownership_checks':['CollisionChainWorkspace.validate_device_bindings','DetectorPlan.validate','NativePlan.validate_bindings'],
        'dataflow':{'profile':getattr(ws,'_cuda_collision_dataflow_profile','compatible'),
                    'public_diagnostics':('complete fixed-capacity ABI' if
                        ws._cuda_collision_manifest['capacity_diagnostics'] else
                        'live contact rows; capacity diagnostic tails unspecified'),
                    'raw_alias_self_copy':False,'tree_inputs':'direct producer owners',
                    'candidate_pairs':'ordered request cursor; block-cooperative descriptor publication',
                    'live_tree':'thread-per-body-pair DFS; ordered prefix commit; bounded visit-arena scratch; ordered fallback',
                    'params':'one mix per pair; kept raw indices feed local contact builder',
                    'sort':'stable index merge; one record permutation',
                    'shared_geom_bounds_from_f':hasattr(ws,'_cuda_collision_bounds_cache'),
                    'bounds_cache_witnesses':(['prepare_count','hit_count','fallback_count','epoch','options']
                                               if hasattr(ws,'_cuda_collision_bounds_cache') else []),
                    'contact_commit':'private candidate then successful public prefix',
                    'direct_s_transport':transport is not None}}
    ws._cuda_collision_manifest['dataflow']['inactive_requested_no_write']=bool(guards)
    ws._cuda_collision_manifest['temporal_front']=temporal_front_manifest(ws)
    ws._cuda_collision_manifest['dataflow']['fast_diagnostic_exclusions']=(
        ['inactive before-output shadows','inactive detector execution count'] if guards else [])
    if not prior['capacity_diagnostics']:
        ws._cuda_collision_manifest['dataflow']['fast_diagnostic_exclusions'] += [
            'before-output backup copies','inactive output restores',
            'unused dedup and parameter rows','compact raw and params_f materialization',
            'BVH visit/stack/leaf tapes and visualization flags','unused call counter tails']
    if transport is not None:
        ws._cuda_collision_direct_transport=transport
        def enqueue_direct(self):self.validate()
        transport._cuda_collision_direct_owner=ws
        transport._cuda_collision_direct_plan=compact['tail'] if query_schedule!='fused' else plan
        transport._cuda_collision_old_enqueue=transport.enqueue
        transport.enqueue=MethodType(enqueue_direct,transport)
    if query_schedule!='fused':
        ws._cuda_collision_manifest.update(query_schedule=query_schedule,kernel_launches=7 if query_schedule=='parallel' else 3,detector_boundary='same-stream kernel boundaries',producer_blocks=producer_blocks,plans=[compact['front'].manifest,detector.manifest,compact['tail'].manifest])
    return ws._cuda_collision_manifest


def install_combined(bundle,root,*,query_schedule='fused',producer_blocks=32,world_guard=None):
    """Factory hook after C and detector installation, before graph capture."""
    forward=bundle.runtime.forward
    return install_combined_workspace(forward.c.workspace,Path(root)/'collision_complete',transport=forward.transport,query_schedule=query_schedule,producer_blocks=producer_blocks,world_guard=world_guard)
