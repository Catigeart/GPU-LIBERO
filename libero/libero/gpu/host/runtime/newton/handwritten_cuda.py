"""Direct-pointer handwritten Newton core; Python only binds and compiles.

Retained stage metadata supplies the call order and actual buffer bindings.
No Warp ModuleBuilder, generated arithmetic, array descriptors or Warp C++
headers are used. The old lowered core remains an independent reference.
"""
import libero.gpu.precision as scalar
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
from libero.gpu.host.runtime.core.product_paths import nvcc_path, serialized_build
import ctypes as ct
from libero.gpu.host.build.library_lifetime import load_cdll
import hashlib
import inspect
import json
import os
import re
import subprocess
import threading
from pathlib import Path

import numpy as np
import warp as wp
from .persistent_cuda import PersistentCudaCore, Slot, _record
from libero.gpu.host.build.product_cache import current_product_authority
from libero.gpu.host.build.build_artifacts import receipted_artifact_build
import uuid
from libero.gpu.host.bindings.capacity import align_bytes, array_layout, LAYOUT_VERSION
from libero.gpu.host.runtime.core.resource_budget import newton_shared_budget
from libero.gpu.host.runtime.core.execution_config import newton_execution_config
from .algorithm_options import effective_newton_algorithm_configuration
from . import device_phase_profile as phase_profile

_RECORD_LOCK = threading.RLock()
_FUNCTIONS = {
    '_update':'constraint_update', '_transpose':'constraint_transpose',
    'validate':'hessian_validate', 'assemble':'hessian_assemble',
    'factor_update':'hessian_factor_update', '_direction':'direction',
    '_warm_select':'warm_select', '_commit_step':'commit_step',
    '_stats_stop':'stats_stop', '_finalize':'finalize',
    '_init_direction':'direction_commit', '_parallel_tape_ls_buffers':'parallel_tape_ls_buffers',
    '_final_event':'final_event', '_publish':'publish',
    '_publication_gate':'publication_gate', '_finish_publication':'finish_publication',
}
_GLUE = ('jacvec_sub mulM gauss_compute count_active_cone save_old gate_ls gate_status '
         'gate_stats gate_maxiter direction_commit empty_fast_path copy_qacc copy_cost '
         'zero_rows solver_iter_reset finalize_exit seed_eval tape_commit tape_ls_input '
         'validate_entry_counts validate_contact_rows').split()
_FUNCTIONS.update({'_'+k:k for k in _GLUE})


def _project_header_source(paths, project_root):
    """Inline quoted project headers and return their complete source closure.

    A synthetic guard replaces each file's pragma once. Keeping the guard at
    every inclusion site preserves conditional inclusion: an inactive first
    include must not suppress a later active one. Angle-bracket SDK includes
    remain compiler inputs. Resolved quoted paths must stay inside the project.
    """
    project_root=Path(project_root).resolve()
    quoted=re.compile(r'^[ \t]*#[ \t]*include[ \t]*"([^"\r\n]+)"[^\r\n]*$',re.MULTILINE)
    once=re.compile(r'^[ \t]*#[ \t]*pragma[ \t]+once[ \t]*(?://[^\r\n]*)?\r?$',re.MULTILINE)
    contents={};active=[]

    def expand(path):
        path=Path(path).resolve()
        if not path.is_relative_to(project_root):
            raise ValueError(('quoted CUDA header outside project',str(path),str(project_root)))
        if path not in contents:contents[path]=path.read_bytes()
        text=contents[path].decode('utf-8')
        single=bool(once.search(text))
        if path in active:
            # A pragma-once parent has already defined its guard before this
            # recursive inclusion. An unguarded cycle is not a valid bundle.
            if single:return '// recursive pragma-once include: '+str(path.relative_to(project_root))+'\n'
            raise ValueError(('unguarded CUDA include cycle',tuple(map(str,active+[path]))))
        active.append(path)
        try:
            def include(match):
                child=path.parent/match.group(1)
                if match.group(1) == 'precision.cuh':
                    child=GPU_ROOT/'cuda/common/precision.cuh'
                return expand(child)
            body=quoted.sub(include,once.sub('',text))
        finally:
            active.pop()
        relative=path.relative_to(project_root).as_posix()
        if single:
            guard='LIBERO_INLINE_HEADER_'+hashlib.sha256(relative.encode()).hexdigest().upper()
            body=f'\n#ifndef {guard}\n#define {guard}\n// project source: {relative}\n'+body+f'\n#endif // {guard}\n'
        return body

    source='\n'.join(expand(path) for path in paths)
    hashes={str(path):hashlib.sha256(data).hexdigest() for path,data in contents.items()}
    return source,hashes


def _expand(kernel, values):
    """Unpack compose metadata; no translation of any arithmetic body."""
    from libero.gpu.host.assembly.compose.compose import _CACHE
    entry = next((key for key, value in tuple(_CACHE.items()) if value is kernel), None)
    if entry is None:
        if kernel.key not in _FUNCTIONS:
            raise ValueError(('handwritten stage missing', kernel.key))
        return [(kernel, values)]
    names = kernel.fusion_argument_names
    bindings = dict(zip(names, values))
    result = []
    for child, mapping in entry[1]:
        result.extend(_expand(child, [bindings[target] for _, target in mapping]))
    return result


def _head_metadata_source(metadata, bind, ws, heads, mutable_arrays, *,
                          diagnostic_profile, execution_config, has_blas,
                          combined_entry):
    """Bind optional ordinal operands without moving the S/N callback boundary.

    This cold helper emits only pointer ABI and a post-validation gate. It does
    not treat producer validity flags as an admission for current N contents.
    """
    result=dict(arguments=['nullptr']*3,gate='',shared='',enabled=False,
                fallback_reason=None)
    if metadata is None:return result
    if (diagnostic_profile!='fast' or ws.nworld!=8 or
            execution_config.strategy!='single_cta' or not has_blas or not combined_entry):
        raise ValueError('ordinal head metadata requires fast N=8 single_cta cuBLASDx combined entry')
    if (metadata.nworld,metadata.rows,metadata.contacts,metadata.nv)!=(
            ws.nworld,ws.rows,ws.contacts,ws.qacc.shape[1]):
        raise ValueError('ordinal head metadata dimensions differ from Newton')
    metadata.validate()
    if any(str(metadata[key].device)!=str(ws.device) for key in metadata):
        raise ValueError('ordinal head metadata device differs from Newton')
    pitches=dict(source_counts=5,source_ncon=1,compact_heads=ws.rows,
        compact_count=1,compact_valid=1,head_type=ws.rows,head_id=ws.rows,
        head_dim=ws.rows,head_mu=ws.rows,head_friction=5*ws.rows,valid=1)
    if dict(metadata.world_strides)!=pitches:
        raise ValueError('ordinal head metadata world pitches differ')
    # Original callers may admit aliases. Only this optional reuse route is
    # narrowed: a mutable N/S07 owner could otherwise change a compared operand
    # later in the same invocation. Aliases retain the original nullptr path.
    readonly=('J','aref','D','R','loss','efc_type','efc_id','cone_dim','cone_mu',
              'cone_friction','qM','qfrc_smooth','qacc_smooth','nefc','ne','nf','ncon',
              'contact_dim','contact_mu','contact_friction')
    protected=[getattr(ws,key) for key in readonly]+[metadata[key] for key in pitches]
    def span(value):
        return value.ptr,value.ptr+value.size*int(value.strides[-1])
    for value in mutable_arrays:
        lo,hi=span(value)
        for operand in protected:
            left,right=span(operand)
            if lo<right and left<hi:
                result['fallback_reason']='mutable Newton/S07 owner overlaps an ordinal or current input owner'
                return result
    pointer={key:bind(metadata[key],'ordinal_'+key,world_stride=pitch)
             for key,pitch in pitches.items()}
    original=[bind(getattr(ws,key),key) for key in
        ('active','status','count','nefc','ne','nf','ncon','efc_type','efc_id',
         'contact_dim','contact_mu','contact_friction','cone_dim','cone_mu','cone_friction')]
    original.insert(2,'&invariant_valid')
    original.extend(bind(heads[key],key) for key in ('group_head','head_rows','head_count'))
    order=('source_counts','source_ncon','compact_heads','compact_count','compact_valid',
           'head_type','head_id','head_dim','head_mu','head_friction','valid')
    gate_arguments=original+[pointer[key] for key in order]+[
        '&ordinal_head_selected','ROWS',str(ws.contacts),'NV']
    result.update(enabled=True,
        arguments=['(ordinal_head_selected?'+pointer[key]+':nullptr)'
                   for key in ('head_dim','head_mu','head_friction')],
        gate='constraint_head_metadata_gate('+','.join(gate_arguments)+');',
        shared='__shared__ int ordinal_head_selected;\nif(threadIdx.x==0)ordinal_head_selected=0;\n')
    return result


class HandwrittenCudaCore(PersistentCudaCore):
    @serialized_build
    def __init__(self, solver, build_root, *, block_size=None, shared_state=True,
                 cholesky_backend='source', mathdx_root=None, blas_backend=None,
                 diagnostic_profile='compatible',adapter=None,external_prefix=None,external_tail=None,
                 phase_profiling=False, phase_trace_capacity=0, execution_config=None,
                 cone_hessian_strategy='auto',cone_batch_threshold=None,
                 rank_rebuild_threshold=None,newton_islands='off',head_metadata=None):
        self.execution_config=newton_execution_config(block_size,execution_config)
        block_size=self.execution_config.block_threads
        ws=solver.ws
        # Fused stages bypass their standalone launchers, so preserve their
        # sealed private-owner checks at cold binding and every actual launch.
        self.external_binding_validators=tuple(
            schema['validate_private_bindings']
            for schema in (external_prefix,external_tail)
            if schema is not None and schema.get('validate_private_bindings') is not None)
        for validate in self.external_binding_validators:
            if not callable(validate):raise TypeError('external private binding validator must be callable')
            validate()
        if diagnostic_profile=='compatibility':diagnostic_profile='compatible'
        if diagnostic_profile not in ('compatible', 'fast'):
            raise ValueError('diagnostic profile must be compatible or fast')
        self.diagnostic_profile=diagnostic_profile
        self.adapter=adapter
        self.phase_profiling=bool(phase_profiling)
        if type(phase_trace_capacity) is not int or phase_trace_capacity<0:
            raise ValueError('nonnegative integer phase trace capacity required')
        if phase_trace_capacity and not self.phase_profiling:
            raise ValueError('phase trace requires explicit phase profiling')
        self.phase_trace_capacity=phase_trace_capacity
        if diagnostic_profile=='fast':solver.configure_diagnostic_profile('fast')
        elif getattr(solver,'diagnostic_profile','compatible')!='compatible':raise ValueError('compatible core requires compatible solver history')
        if block_size not in (32,64,128) or ws.maxiter<1:
            raise ValueError('positive maxiter; block size 32/64/128 required')
        self.solver=solver;self.ws=ws;self.block_size=block_size
        self.original=solver.launch_device
        self.event_owner=inspect.getclosurevars(solver.launch_device.__func__).nonlocals.get('owner')
        with _RECORD_LOCK:
            records=_record(solver)
        stages=[]
        for mode,kernel,dim,args in records:
            if kernel=='S07':stages.append((mode,kernel,dim,args));continue
            children=_expand(kernel,args)
            stages.extend((mode,k,dim,v) for k,v in children)
        nv,nm=ws.qacc.shape[1],ws.qM.shape[1]
        self.scalar_io=nv in (21,43)
        if cholesky_backend not in ('source','cusolverdx'):
            raise ValueError('unknown Cholesky backend')
        use_dx=cholesky_backend=='cusolverdx'
        blas_backend=blas_backend or ('cublasdx' if use_dx else 'source')
        if blas_backend not in ('source','cublasdx'):
            raise ValueError('unknown Hessian BLAS backend')
        use_blas=blas_backend=='cublasdx'
        if use_blas and not use_dx:
            raise ValueError('collective BLAS Hessian currently requires the cuSOLVERDx factor path')
        self.hessian_algorithm_configuration=effective_newton_algorithm_configuration(
            dict(cone_hessian_strategy=cone_hessian_strategy,cone_batch_threshold=cone_batch_threshold,
                 rank_rebuild_threshold=rank_rebuild_threshold,newton_islands=newton_islands),
            nv,has_blas=use_blas,has_dx=use_dx)
        cone_threshold=self.hessian_algorithm_configuration['cone_batch_threshold']
        rebuild_threshold=self.hessian_algorithm_configuration['rank_rebuild_threshold']
        effective_cone_strategy=self.hessian_algorithm_configuration['cone_strategy']
        budget=newton_shared_budget(nv,cholesky_backend=cholesky_backend,blas_backend=blas_backend,scalar_bytes=scalar.itemsize,
            prefix_static_bytes=int((external_prefix or {}).get('static_shared_bytes',0))+
                (4 if head_metadata is not None else 0)+
                (phase_profile.STATIC_SHARED_BYTES if self.phase_profiling else 0),
            tail_static_bytes=int((external_tail or {}).get('static_shared_bytes',0)),
            newton_islands=newton_islands)
        hessian_k=budget['hessian_tile_k'] or (16 if nv<=43 else 8)
        static_reserve=budget['static_reserve_bytes'];dx_bytes=budget['dx_workspace_bytes']
        if not budget['admitted_by_formula']:
            raise ValueError('cuSOLVERDx matrix exceeds this single-block shared capacity')
        tail_elements=int((external_tail or {}).get('workspace_elements',0))
        if tail_elements>dx_bytes//scalar.itemsize:
            raise ValueError(('external tail workspace exceeds lifetime-reused Dx storage',tail_elements,dx_bytes//scalar.itemsize))
        from libero.gpu.host.runtime.core.product_paths import private_product_root
        self.root=private_product_root(build_root);self.root.mkdir(parents=True,exist_ok=True)
        fields=[];values=[];arrays={};names=set();owners=[]
        # Row topology is regenerated whenever invariant_valid is reset at entry.
        self.constraint_group_heads=wp.zeros((ws.nworld,ws.rows),dtype=wp.int32,device=ws.device)
        self.constraint_hessian_owner=wp.zeros((ws.nworld,ws.contacts),dtype=wp.int32,device=ws.device)
        # Invocation-local topology: populated only after complete row validation.
        self.constraint_head_rows=wp.zeros((ws.nworld,ws.rows),dtype=wp.int32,device=ws.device)
        self.constraint_head_count=wp.zeros((ws.nworld,),dtype=wp.int32,device=ws.device)
        self.constraint_tile_head_offset=wp.zeros((ws.nworld,(ws.rows+15)//16+1),dtype=wp.int32,device=ws.device)
        self.constraint_typed_head_rows=wp.zeros((ws.nworld,ws.rows),dtype=wp.int32,device=ws.device)
        self.constraint_typed_head_offsets=wp.zeros((ws.nworld,8),dtype=wp.int32,device=ws.device)
        named={}
        for owner,prefix in ((ws,''),(solver,'solver_')):
            for key,value in vars(owner).items():
                if isinstance(value,wp.array):named.setdefault((value.ptr,value.dtype),prefix+key.lstrip('_'))
        invariant={ws.madr.ptr,ws.parentid.ptr,ws.simplenum.ptr,ws.model_scalars.ptr}
        def bind(value, suggested,world_stride=None,owner_only=False):
            if isinstance(value,Slot):return 'iteration_slot'
            if isinstance(value,wp.array):
                if not value.is_contiguous:raise ValueError(('contiguous core array required',suggested))
                token=(value.ptr,value.dtype)
                # External functions accept the original owner plus their own
                # world index. Their pointer parameter must not establish a
                # local-view policy, even when the same owner is later used by
                # canonical publication (or a shared model has an odd length).
                stride=None
                if not owner_only:
                    stride=(0 if value.ptr in invariant else value.size//ws.nworld) if world_stride is None else int(world_stride)
                    if stride and value.size%ws.nworld:raise ValueError(('world stride',suggested))
                if token in arrays:
                    entry=arrays[token]
                    if not owner_only:
                        if entry['view_required'] and entry['stride']!=stride:
                            raise ValueError(('conflicting local world stride',entry['name'],entry['stride'],stride))
                        entry['view_required']=True;entry['stride']=stride
                    return entry['name']
                ctype={scalar.wp_real:'real_t',wp.int32:'int32_t',wp.int64:'int64_t'}.get(value.dtype)
                if ctype is None:raise TypeError(('core dtype',value.dtype))
                name=named.get(token,suggested)
                name=''.join(c if c.isalnum() or c=='_' else '_' for c in name)
                while name in names:name+='_' 
                names.add(name)
                fields.append((name,ct.c_void_p,ctype+'*'));values.append(ct.c_void_p(value.ptr));owners.append(value)
                arrays[token]=dict(name=name,stride=stride,ctype=ctype,array=value,shared=False,view_required=not owner_only)
                return name
            if isinstance(value,(float,np.floating)):
                fields.append((f'scalar_{len(fields)}',scalar.ct_real,'real_t'));values.append(scalar.ct_real(value))
            else:
                fields.append((f'scalar_{len(fields)}',ct.c_int32,'int32_t'));values.append(ct.c_int32(value))
            return 'p->'+fields[-1][0]
        pending_argument='nullptr';algorithm_argument='nullptr';islands_argument='nullptr';upper_argument='nullptr'
        if use_dx:
            self.hessian_algorithm_diagnostic=wp.zeros((ws.nworld,8),dtype=wp.int32,device=ws.device)
            algorithm_argument=bind(self.hessian_algorithm_diagnostic,'hessian_algorithm_diagnostic')
            if rebuild_threshold:
                self.hessian_upper_backup=wp.empty((ws.nworld,nv*nv),dtype=scalar.wp_real,device=ws.device)
                upper_argument=bind(self.hessian_upper_backup,'hessian_upper_backup')
        if use_blas and effective_cone_strategy!='rank':
            # Lifetime overlaps local POTRF/L^T J scratch. Keep a separate owner
            # instead of aliasing vendor workspace or pending contact operands.
            self.cone_pending_rows=wp.empty((ws.nworld,hessian_k*nv),dtype=scalar.wp_real,device=ws.device)
            pending_argument=bind(self.cone_pending_rows,'cone_pending_rows')
        if newton_islands=='exact':
            self.islands_diagnostic=wp.zeros((ws.nworld,8),dtype=wp.int32,device=ws.device)
            islands_argument=bind(self.islands_diagnostic,'islands_diagnostic')
        self.head_metadata=head_metadata
        if head_metadata is not None:
            from .head_metadata import HeadMetadata
            if not isinstance(head_metadata,HeadMetadata):raise TypeError('sealed HeadMetadata descriptor required')
            self.external_binding_validators+=(head_metadata.validate,)
        readonly={'J','aref','D','R','loss','efc_type','efc_id','cone_dim','cone_mu',
            'cone_friction','qM','qfrc_smooth','qacc_smooth','nefc','ne','nf','ncon',
            'contact_dim','contact_mu','contact_friction','madr','parentid','simplenum','model_scalars',
            'mass_plan_offsets','mass_plan_coefficient','mass_plan_source','mass_plan_assign',
            'mass_plan_madr_snapshot','mass_plan_parent_snapshot','mass_plan_simple_snapshot','smooth_qacc'}
        mutable=[value for key,value in vars(ws).items()
                 if isinstance(value,wp.array) and key not in readonly]
        readonly_views={'_J_flat','_aref_flat','_D_flat','_R_flat','_loss_flat',
            '_type_flat','_id_flat','_dim_flat','_mu_flat','_fr_flat',
            '_cone_dim_flat','_cone_mu_flat','_cone_fr_flat','_qM_flat','_qacc_smooth_flat'}
        mutable.extend(value for key,value in vars(solver).items()
                       if isinstance(value,wp.array) and key not in readonly_views)
        mutable.extend((self.constraint_group_heads,self.constraint_hessian_owner,
            self.constraint_head_rows,self.constraint_head_count,self.constraint_tile_head_offset,
            self.constraint_typed_head_rows,self.constraint_typed_head_offsets))
        head_source=_head_metadata_source(head_metadata,bind,ws,
            dict(group_head=self.constraint_group_heads,head_rows=self.constraint_head_rows,
                 head_count=self.constraint_head_count),mutable,
            diagnostic_profile=diagnostic_profile,execution_config=self.execution_config,
            has_blas=use_blas,combined_entry=adapter is not None)
        head_arguments=head_source['arguments']
        mass_plan=[]
        for key in ('mass_plan_offsets','mass_plan_coefficient','mass_plan_source','mass_plan_assign'):
            a=getattr(ws,key,None)
            if a is not None:mass_plan.append(bind(a,key,world_stride=0))
        if mass_plan and len(mass_plan)!=4:raise ValueError('incomplete packed mass plan')
        mass_guard=[]
        if mass_plan:
            for live,snapshot in (('madr','mass_plan_madr_snapshot'),('parentid','mass_plan_parent_snapshot'),('simplenum','mass_plan_simple_snapshot')):
                mass_guard.append((bind(getattr(ws,live),live,world_stride=0),bind(getattr(ws,snapshot),snapshot,world_stride=0)))
            mass_plan=['(mass_plan_valid?'+x+':nullptr)' for x in mass_plan]

        calls={k:[] for k in ('prefix','loop','tail')}
        recorded_tail={}
        head_gate_inserted=False
        tape_keys={'_tape_ls_input','_parallel_tape_ls_buffers','_tape_commit'}
        for mode,kernel,dim,args in stages:
            if kernel=='S07':
                group_head = bind(self.constraint_group_heads, 'constraint_group_heads')
                plan=[bind(self.constraint_head_rows,'constraint_head_rows'),
                      bind(self.constraint_head_count,'constraint_head_count'),
                      bind(self.constraint_tile_head_offset,'constraint_tile_head_offset')]
                call=('search_kernel_blas_dx(0,ls,1,NV,NM,ROWS,52,dx_matrix,blas_diagnostic,'+','.join([group_head]+plan+head_arguments)+');') if use_blas else 'if(threadIdx.x==0) search_kernel(0,ls,1,NV,NM,ROWS,52);'
                if self.phase_profiling:call=phase_profile.wrap_stage(call,'S07',mode)
                calls[mode].append(call+'\n__syncthreads();')
                continue
            params=list(inspect.signature(kernel.func).parameters)
            if len(params)!=len(args):raise ValueError(('stage parameters',kernel.key))
            if diagnostic_profile=='fast' and kernel.key in tape_keys:continue
            if kernel.key=='_count_active_cone':continue # generated with S04 states
            bound=[bind(value,key) for value,key in zip(args,params)]
            shape=(dim,) if isinstance(dim,int) else tuple(dim)
            perworld=int(np.prod(shape))//ws.nworld
            fn=_FUNCTIONS[kernel.key]
            if kernel.key=='_update':
                counts=(args[5].ptr==solver._flag_on.ptr)
                extras=['&invariant_valid',bind(ws.nactive,'nactive') if counts else 'nullptr',
                        bind(ws.ncone,'ncone') if counts else 'nullptr',
                        bind(ws.oldstate,'oldstate') if counts and mode=='loop' else 'nullptr',
                        '&changed_count' if counts and mode=='loop' else 'nullptr',
                        bind(self.constraint_group_heads,'constraint_group_heads'),
                        bind(self.constraint_hessian_owner,'constraint_hessian_owner'),
                        bind(self.constraint_head_rows,'constraint_head_rows'),
                        bind(self.constraint_head_count,'constraint_head_count'),
                        bind(self.constraint_tile_head_offset,'constraint_tile_head_offset'),
                        bind(self.constraint_typed_head_rows,'constraint_typed_head_rows'),
                        bind(self.constraint_typed_head_offsets,'constraint_typed_head_offsets'),*head_arguments]
                code='constraint_update_block(0,'+','.join(bound+extras)+');'
            elif kernel.key=='validate':
                code='hessian_validate_block(0,'+','.join(bound+['&invariant_valid'])+');'
            elif kernel.key=='_stats_stop':
                code='stats_stop_block(0,'+','.join(bound+['&changed_count',bind(ws.Ma,'Ma'),bind(ws.qfrc_smooth,'qfrc_smooth'),bind(ws.J,'J'),bind(ws.force,'force')])+');'
            elif kernel.key=='_tape_commit':
                code='tape_commit_block(0,'+','.join(bound)+');'
            elif kernel.key=='_gauss_compute':
                code='gauss_compute_block(0,'+','.join(bound)+');'
            elif ws.warmstart_enabled and mode=='prefix' and kernel.key=='_jacvec_sub' and args[3].ptr==ws.qacc.ptr:
                code='warm_init_jacvec('+','.join([bound[0],bound[1],bind(ws.choice,'choice'),
                    bind(ws.jar_warm,'jar_warm'),bind(ws.efc_b,'efc_b'),bound[5]])+');'
            elif ws.warmstart_enabled and mode=='prefix' and kernel.key=='_mulM' and args[1].ptr==ws.qacc.ptr:
                code='warm_init_mass('+','.join([bound[0],bind(ws.choice,'choice'),
                    bind(ws.Ma_warm,'Ma_warm'),*bound[1:],*mass_plan])+');'
            elif kernel.key=='_mulM' and mass_plan:
                code='mulM_plan_block(0,'+','.join(bound+mass_plan)+');'
            elif adapter is not None and kernel.key=='_publish':
                code='if(threadIdx.x==0) publish(0,entry_upstream+w,entry_enabled+w,'+','.join(bound[2:])+');'
            elif use_dx and kernel.key=='factor_update':
                code='hessian_factor_update_dx(0,'+','.join(bound)+',dx_matrix,potrf_diagnostic,'+','.join(
                    ['blas_diagnostic' if use_blas else 'nullptr',pending_argument,algorithm_argument,islands_argument,upper_argument])+');'
            elif use_blas and kernel.key=='assemble':
                code='hessian_assemble_blas_dx(0,'+','.join(bound)+',dx_matrix,blas_diagnostic);'
            elif use_blas and kernel.key=='_jacvec_sub':
                code='jacvec_sub_blas_dx(0,'+','.join(bound)+',dx_matrix,blas_diagnostic);'
            elif use_blas and kernel.key=='_transpose':
                code='constraint_transpose_blas_dx(0,'+','.join(bound)+',dx_matrix,blas_diagnostic);'
            elif use_blas and kernel.key=='_direction':
                code='direction_blas_dx(0,'+','.join(bound)+',dx_matrix,blas_diagnostic,DIRECTION_BASE_SHARED,DIRECTION_CONE_SHARED,'+islands_argument+');'
            elif kernel.key in ('_warm_select', '_commit_step', '_finalize', '_validate_contact_rows'):
                code=f'{fn}_block(0,'+','.join(bound)+');'
            elif perworld==1:code=f'if(threadIdx.x==0) {fn}(0,'+','.join(bound)+');'
            else:code=f'for(int local=threadIdx.x;local<{perworld};local+=blockDim.x) {fn}(local,'+','.join(bound)+');'
            if self.phase_profiling:code=phase_profile.wrap_stage(code,kernel.key,mode)
            calls[mode].append(code+'\n__syncthreads();')
            if kernel.key=='_update' and head_source['enabled'] and not head_gate_inserted:
                if mode!='prefix':raise ValueError('ordinal metadata gate requires an ordinary prefix update')
                # The first call used nullptr operands: only its completed
                # original validation/list construction can admit the late gate.
                calls[mode].append(head_source['gate']+'\n__syncthreads();')
                head_gate_inserted=True
            if mode=='loop' and kernel.key in tape_keys:recorded_tail[kernel.key]=bound
        if head_source['enabled'] and not head_gate_inserted:raise ValueError('ordinal metadata prefix update missing')
        # S07 is already handwritten CUDA. Bind its pointer ABI to local views.
        ls_fields=[]
        byptr={value.ptr:value for owner in (ws,solver) for value in vars(owner).values() if isinstance(value,wp.array)}
        for key,_ in ws._s07_args._fields_:
            ptr=getattr(ws._s07_args,key)
            address=ptr if isinstance(ptr,int) else ct.cast(ptr,ct.c_void_p).value
            if address not in byptr:raise ValueError(('S07 root missing',key))
            ls_fields.append(f'ls.{key}='+bind(byptr[address],'ls_'+key)+';')
        active_name=bind(ws.active,'active')
        status_name=bind(ws.status,'status')
        route_name=bind(ws.route,'route')
        s07_status_name=bind(ws.s07_status,'s07_status')
        # The inactive source schedule clears these statuses and LS markers,
        # while preserving all solver/tape fields that its gated stages hold.
        inactive=f'''if(threadIdx.x==0){{
{s07_status_name}[0]=0;{status_name}[0]=0;{route_name}[0]=0;
seed_eval(0,{bind(ws.eval_trace,'eval_trace')},{bind(ws.eval_tag,'eval_tag')});
}}\n__syncthreads();\n'''
        if diagnostic_profile=='compatible':
            if set(recorded_tail)!=tape_keys:raise ValueError('inactive tape schedule missing')
            buffers=recorded_tail['_parallel_tape_ls_buffers']
            # Original signature: slot, slots, rows, Mv, Jv, quad, qG, tapes, nv.
            inactive+='inactive_tape_ls_buffers(iteration_slot,'+str(ws.maxiter)+','+','.join(buffers[1:])+');\n'
            history_calls=[]
            for key in ('_tape_ls_input','_tape_commit'):
                arguments=recorded_tail[key].copy();arguments[0]='history_slot'
                fn='tape_commit_block' if key=='_tape_commit' else _FUNCTIONS[key]
                guard='' if key=='_tape_commit' else 'if(threadIdx.x==0) '
                history_calls.append(guard+fn+'(0,'+','.join(arguments)+');')
            inactive+=f'for(int history_slot=iteration_slot;history_slot<{min(ws.maxiter,solver.tape_slots)};++history_slot){{\n'+ '\n'.join(history_calls)+'\n__syncthreads();\n}\n'
        external_headers=[]
        def external_stages(schema,allow_local=False):
            code=[]
            if schema is None:return code
            external_headers.extend(Path(p) for p in schema.get('headers',()))
            for stage in schema.get('stages',()):
                values=[]
                for name,value in stage['arguments']:
                    expression=stage.get('argument_expressions',{}).get(name)
                    if expression:
                        if not allow_local:raise ValueError('prefix cannot consume a private Newton view')
                        if expression=='private_qacc':
                            if not isinstance(value,wp.array) or value.ptr!=ws.qacc.ptr:raise ValueError('private qacc expression owner')
                            values.append(bind(value,'qacc'));continue
                        if expression=='dx_matrix' and use_dx:values.append('dx_matrix');continue
                        raise ValueError('unsupported external expression')
                    token=bind(value,'external_'+name,owner_only=isinstance(value,wp.array))
                    # External stages explicitly retain original owner pointers
                    # and consume their own world index, rather than local views.
                    values.append('p->'+token if isinstance(value,wp.array) else token)
                world_argument='w,' if schema.get('world_index_argument',True) else ''
                invocation=stage['function']+'('+world_argument+','.join(values)+');'
                if stage.get('collective',False):code.append(invocation)
                elif int(stage.get('perworld',1))==1:code.append('if(threadIdx.x==0) '+invocation)
                else:raise ValueError('external prefix only supports scalar or collective stages')
                code.append('__syncthreads();')
            return code
        external_code=external_stages(external_prefix)
        external_tail_code=external_stages(external_tail,True)
        adapter_begin=[];adapter_commit=[];seeded=set()
        if adapter is not None:
            from .runtime_adapter import PUBLIC_FIELDS, RESET_INT, RESET_FLOAT
            if adapter.ws is not ws or not adapter._sealed:raise ValueError('adapter must own the sealed Newton workspace')
            local=lambda owner,key:bind(getattr(owner,key),key)
            pub=local(adapter,'publication_error');attempt=local(adapter,'n_attempt');expected=local(adapter,'n_expected')
            adapter_begin.append(f'if(threadIdx.x==0){{{pub}[0]=entry_publication[w];if(entry_enabled[w]){{++{attempt}[0];{expected}[0]=entry_supplied?entry_plan[w]:{attempt}[0];}}}}\n__syncthreads();')
            for name in PUBLIC_FIELDS:
                work=getattr(ws,name);canonical=adapter._canonical[name]
                dst=bind(work,name);src=bind(canonical,'canonical_'+name);stride=work.size//ws.nworld
                # Public row tails are held by prefix publication, not by the
                # private seed. Clamp before validation so malformed counts can
                # never turn the entry copy into an out-of-bounds access.
                seed_extent=str(stride)
                if diagnostic_profile=='fast':
                    if name in ('force','state'):
                        seed_extent=f'max(0,min({local(ws,"nefc")}[0],{stride}))'
                    elif name in ('solver_stats_f','solver_stats_i'):
                        # stats_stop overwrites complete records before commit;
                        # base/iteration reset to zero in this combined entry.
                        seed_extent=None
                if seed_extent is not None:
                    adapter_begin.append(f'libero_newton_io::copy_block({src},{dst},{seed_extent});')
                # Publication has checked nefc. Records outside these extents
                # retain their canonical bytes without reading private tails.
                extent=str(stride)
                if name in ('force','state'):
                    extent=f'{local(ws,"nefc")}[0]'
                elif name in ('solver_stats_f','solver_stats_i'):
                    # The entry resets base=0. iteration counts complete stats
                    # records; alpha==0 and empty entries can write no records.
                    width=3 if name=='solver_stats_f' else 4
                    extent=f'{width}*min({local(ws,"iteration")}[0],1000)'
                adapter_commit.append(f'libero_newton_io::copy_block({dst},{src},{extent});')
                seeded.add(work.ptr)
            reset=[f'{local(ws,name)}[0]=0;' for name in RESET_INT+RESET_FLOAT]
            reset.extend([f'{local(ws,"error_item")}[0]=-1;',f'{local(ws,"stage")}[0]=1;',
                f'{local(ws,"maxiter_arr")}[0]={ws.maxiter};',f'{local(ws,"solver_iter_base")}[0]=0;',
                f'{local(ws,"tolerance")}[0]={local(ws,"model_scalars")}[0];',
                f'{local(ws,"meaninertia")}[0]={local(ws,"model_scalars")}[1];',
                f'{local(ws,"empty")}[0]=0;{local(ws,"newton_world")}[0]=0;{active_name}[0]=0;',
                f'{local(ws,"count")}[0]={local(ws,"nefc")}[0];{local(ws,"err_code")}[0]=entry_upstream[w];',
                f'{local(adapter,"valid")}[0]=0;{local(adapter,"written")}[0]=0;',
                f'if(entry_enabled[w]&&!entry_upstream[w]){{if({local(ws,"nefc")}[0]==0){local(ws,"empty")}[0]=1;else if({local(ws,"nefc")}[0]>0){{{local(ws,"newton_world")}[0]=1;{active_name}[0]=1;}}}}'])
            adapter_begin.append('__syncthreads();\nif(threadIdx.x==0){\n'+'\n'.join(reset)+'\n}\n__syncthreads();')
            adapter_commit=['if('+local(adapter,'valid')+'[0]){\n'+'\n'.join(adapter_commit)+'\n}\n__syncthreads();']
            if not any(mode=='tail' and kernel!='S07' and kernel.key=='_publish' for mode,kernel,_,_ in stages):
                valid=local(adapter,'valid');written=local(adapter,'written');nwritten=local(adapter,'n_written')
                calls['tail'].extend([
                    f'if(threadIdx.x==0)publish(0,entry_upstream+w,entry_enabled+w,{local(ws,"err_code")},{valid},{written});\n__syncthreads();',
                    f'if(threadIdx.x==0)publication_gate(0,{valid},{pub});\n__syncthreads();',
                    f'if(threadIdx.x==0)finish_publication(0,{valid},{written},{expected},{nwritten});\n__syncthreads();'])
        if use_dx:
            self.potrf_diagnostic=wp.zeros((ws.nworld,4),dtype=wp.int32,device=ws.device)
            bind(self.potrf_diagnostic,'potrf_diagnostic')
        if use_blas:
            self.blas_diagnostic=wp.zeros((ws.nworld,7),dtype=wp.int32,device=ws.device)
            bind(self.blas_diagnostic,'blas_diagnostic')
        # Preserve exact aliases by selecting storage for pointer identities.
        hot=('qacc','search','grad','Mgrad','Ma','Mv',
             'scratch_x','scratch_grad','scratch_fwd','H_vec','H_local','H_ltj','H','Hcone')
        selected=[];shared_bytes=0
        for key in hot:
            a=getattr(ws,key);entry=arrays.get((a.ptr,a.dtype))
            if not shared_state or entry is None or not entry['view_required'] or entry['shared']:continue
            size=entry['stride']*scalar.itemsize
            offset=align_bytes(shared_bytes)
            end=offset+size
            if not size or align_bytes(end,16)+dx_bytes+static_reserve>48*1024:continue
            entry['shared']=True;entry['offset']=offset//scalar.itemsize
            shared_bytes=end;selected.append(entry)
        if self.phase_profiling:
            self.phase_profile_values=wp.zeros((ws.nworld,len(phase_profile.PHASES),2),dtype=wp.int64,device=ws.device)
            profile_pointer=bind(self.phase_profile_values,'phase_profile_values',owner_only=True)
            if self.phase_trace_capacity:
                self.phase_trace_fields=list(phase_profile.TRACE_WORKSPACE_FIELDS)
                metric_operands=[bind(getattr(ws,key),key)+'[0]'
                                 for key in phase_profile.TRACE_WORKSPACE_FIELDS]
                for name,width in (('potrf_diagnostic',4),('blas_diagnostic',7),
                                   ('hessian_algorithm_diagnostic',8)):
                    value=getattr(self,name,None)
                    if value is None:continue
                    pointer=bind(value,name)
                    for index in range(width):
                        self.phase_trace_fields.append(name+'.'+str(index))
                        metric_operands.append(pointer+'['+str(index)+']')
                self.phase_profile_trace=wp.zeros((ws.nworld,self.phase_trace_capacity,len(phase_profile.PHASES)*2),dtype=wp.int64,device=ws.device)
                self.phase_trace_counts=wp.zeros(ws.nworld,dtype=wp.int64,device=ws.device)
                self.phase_trace_metrics=wp.zeros((ws.nworld,self.phase_trace_capacity,len(metric_operands)),dtype=wp.int64,device=ws.device)
                trace_pointer=bind(self.phase_profile_trace,'phase_profile_trace',owner_only=True)
                counter_pointer=bind(self.phase_trace_counts,'phase_trace_counts',owner_only=True)
                metric_pointer=bind(self.phase_trace_metrics,'phase_trace_metrics',owner_only=True)
                trace_store=phase_profile.trace_store_source(profile_pointer,trace_pointer,
                    counter_pointer,metric_pointer,self.phase_trace_capacity,metric_operands)
        Param=type('HandwrittenParameters',(ct.Structure,),{'_fields_':[(n,t) for n,t,_ in fields]})
        self.params=Param(*values)
        declarations='\n'.join(f'{kind} {name};' for name,_,kind in fields)
        views=[];loads=[];stores=[]
        for e in arrays.values():
            if not e['view_required']:continue
            name,stride=e['name'],e['stride']
            globalptr=f'p->{name}'+(f'+w*{stride}' if stride else '')
            localptr=f'shared+{e["offset"]}' if e['shared'] else globalptr
            views.append(f'{e["ctype"]}* {name}={localptr};')
            if e['shared']:
                if e['array'].ptr not in seeded:loads.append(f'libero_newton_io::copy_block(({globalptr}),{name},{stride});')
                stores.append(f'libero_newton_io::copy_block({name},({globalptr}),{stride});')
        header_dir=(GPU_ROOT / 'cuda/newton')
        paths=[GPU_ROOT/'cuda/common/vector.cuh',GPU_ROOT/'cuda/common/warp.cuh',GPU_ROOT/'cuda/common/packed_mass_gather.cuh']+[header_dir/(name+'.cuh') for name in ('math','constraint','hessian','state','glue')]+external_headers
        dependency=None
        if use_dx:
            if use_blas: paths.extend([(GPU_ROOT / 'cuda/newton/hessian_blas_dx.cuh'),(GPU_ROOT / 'cuda/newton/direction_blas_dx.cuh'),(GPU_ROOT / 'cuda/newton/line_search_warp.cuh'),(GPU_ROOT / 'cuda/newton/s07_blas_dx.cuh')])
            paths.append((GPU_ROOT / 'cuda/newton/potrf_dx.cuh'))
            if newton_islands=='exact':paths.append(GPU_ROOT/'cuda/newton/islands.cuh')
            default='/srv/libero-split-receive-20261004-r01/dependencies/mathdx-cuda12/nvidia-mathdx-25.12.1-cuda12/nvidia/mathdx/25.12'
            dependency=Path(mathdx_root or os.environ.get('LIBERO_MATHDX_ROOT',default)).resolve()
            for name in ('include/cusolverdx.hpp','lib/libcusolverdx.a','external/cutlass/include/cutlass/cutlass.h'):
                if not (dependency/name).is_file():raise FileNotFoundError(dependency/name)
            if use_blas and not (dependency/'include/cublasdx.hpp').is_file():raise FileNotFoundError(dependency/'include/cublasdx.hpp')
        native_path=(GPU_ROOT / 'cuda/newton/linesearch_scalar.cuh')
        native=native_path.read_text()
        factor_shared=[arrays.get((a.ptr,a.dtype),{}).get('shared',False) for a in (ws.H,ws.Hcone)]
        source=f'#define LIBERO_NEWTON_SCALAR_IO {int(self.scalar_io)}\n#define NEWTON_BLOCK_THREADS {block_size}\n#define NV {nv}\n#define NM {nm}\n#define ROWS {ws.rows}\n#define DATAFLOW_FAST_PROFILE {int(diagnostic_profile=="fast")}\n#define DIRECTION_BASE_SHARED {int(factor_shared[0])}\n#define DIRECTION_CONE_SHARED {int(factor_shared[1])}\n'+native
        if use_dx:
            source=f'#define DX_SM {wp.get_device(ws.device).arch*10}\n#define DX_THREADS {block_size}\n#define DX_MATRIX_BYTES {dx_bytes}\n#define DX_HESSIAN_K {hessian_k}\n#define LIBERO_HESSIAN_BLAS {int(use_blas)}\n'+source
            source=('#define LIBERO_CONE_HESSIAN_STRATEGY '+str({'rank':0,'batch':1,'auto':2}[effective_cone_strategy])+'\n'
                '#define LIBERO_CONE_BATCH_THRESHOLD '+str(cone_threshold)+'\n'
                '#define LIBERO_RANK_REBUILD_THRESHOLD '+str(rebuild_threshold)+'\n'
                '#define LIBERO_NEWTON_ISLANDS '+str(int(newton_islands=='exact'))+'\n'+source)
        source+='#define CF_S_KBIP_PLANES '+str(int(diagnostic_profile=='fast'))+'\n'
        header_source,stage_sources=_project_header_source(paths,GPU_ROOT)
        source+=header_source
        if self.phase_profiling:source=phase_profile.instrument_headers(source,use_dx=use_dx,use_blas=use_blas)
        source+='\nstruct Parameters {\n'+declarations+'\n};\n'
        source+=f'static_assert(sizeof(Parameters)=={ct.sizeof(Param)},"pointer ABI");\n'
        entry_decl=',const int* entry_upstream,const int* entry_enabled,const int* entry_publication,const int64_t* entry_plan,int entry_supplied' if adapter is not None else ''
        source+='__global__ void handwritten_newton(const Parameters* p'+entry_decl+'){\nint w=blockIdx.y;\n'
        if self.phase_profiling:
            source+='libero_profile_init();\nlibero_profile_begin('+str(phase_profile.PHASES.index('kernel_total'))+');\n'
            loads=[phase_profile.wrap_stage('\n'.join(loads),'resident_load','prefix')] if loads else []
            stores=[phase_profile.wrap_stage('\n'.join(stores),'resident_store','tail')] if stores else []
            adapter_begin=[phase_profile.wrap_stage('\n'.join(adapter_begin),'entry_seed','prefix')] if adapter_begin else []
            adapter_commit=[phase_profile.wrap_stage('\n'.join(adapter_commit),'public_commit','tail')] if adapter_commit else []
        profile_external_code=[phase_profile.wrap_stage('\n'.join(external_code),'external_prefix','prefix')] if self.phase_profiling and external_code else external_code
        profile_external_tail_code=[phase_profile.wrap_stage('\n'.join(external_tail_code),'external_tail','tail')] if self.phase_profiling and external_tail_code else external_tail_code
        source+="""extern __shared__ __align__(32) real_t shared[];
__shared__ int invariant_valid,changed_count;
if(threadIdx.x==0){invariant_valid=0;changed_count=0;}
"""+head_source['shared']+'\n'.join(views)+'\n'+'\n'.join(profile_external_code+loads+adapter_begin)+'\n__syncthreads();\nS07Args ls={};\n'+'\n'.join(ls_fields)+'\n'
        if use_dx:
            # Align the device-library matrix workspace independently of views.
            shared_bytes=((shared_bytes+15)//16)*16
            source+=f'real_t* dx_matrix=shared+{shared_bytes//scalar.itemsize};\n'
            shared_bytes+=dx_bytes
        if mass_guard:
            source+='__shared__ int mass_plan_valid;\nif(threadIdx.x==0)mass_plan_valid=1;\n__syncthreads();\n'
            condition='||'.join(a+'[i]!='+b+'[i]' for a,b in mass_guard)
            source+='for(int i=threadIdx.x;i<NV;i+=blockDim.x)if('+condition+')atomicExch(&mass_plan_valid,0);\n__syncthreads();\n'
        source+='\n'.join(calls['prefix'])+f'\nint iteration_slot=0;\nfor(;iteration_slot<{ws.maxiter};++iteration_slot){{\nif(!{active_name}[0])break;\n'
        source+='\n'.join(calls['loop'])+f'\n}}\nif(iteration_slot<{ws.maxiter}){{\n'+inactive+'\n}\n'+'\n'.join(calls['tail']+adapter_commit+profile_external_tail_code)+'\n'+'\n'.join(stores)+'\n}\n'
        if self.phase_profiling:
            end=source.rfind('\n}\n')
            if end<0:raise RuntimeError('profile kernel end missing')
            finish='\nlibero_profile_end('+str(phase_profile.PHASES.index('kernel_total'))+');\n'
            finish+=f'libero_profile_store(p->{profile_pointer}+w*{len(phase_profile.PHASES)*2});'
            if self.phase_trace_capacity:finish+='\n'+trace_store
            source=source[:end]+finish+source[end:]
        if adapter is None:
            source+=f'extern "C" int launch_persistent(const void* p,int worlds,void* stream){{handwritten_newton<<<dim3(1,worlds),{block_size},{shared_bytes},(cudaStream_t)stream>>>((const Parameters*)p);return (int)cudaGetLastError();}}\n'
        else:
            source+='extern "C" int launch_persistent(const void*,int,void*){return (int)cudaErrorInvalidValue;}\n'
            source+=f'extern "C" int launch_entry(const void* p,int worlds,const int* upstream,const int* enabled,const int* publication,const int64_t* plan,int supplied,void* stream){{handwritten_newton<<<dim3(1,worlds),{block_size},{shared_bytes},(cudaStream_t)stream>>>((const Parameters*)p,upstream,enabled,publication,plan,supplied);return (int)cudaGetLastError();}}\n'
        dependency_sha=None if dependency is None else hashlib.sha256((dependency/'lib/libcusolverdx.a').read_bytes()).hexdigest()
        bridge_path=(GPU_ROOT / 'cuda/newton/potrf_dx_bridge.cu')
        bridge_source=''
        if use_dx:
            bridge_source=f'#define NV {nv}\n#define DX_SM {wp.get_device(ws.device).arch*10}\n#define DX_THREADS {block_size}\n#define DX_MATRIX_BYTES {dx_bytes}\n'+bridge_path.read_text()
        blas_bridge_path=(GPU_ROOT / 'cuda/newton/hessian_blas_dx_bridge.cu')
        if use_blas:
            bridge_source+=f'\n#define DX_HESSIAN_K {hessian_k}\n'+blas_bridge_path.read_text()
            bridge_source+='\n'+((GPU_ROOT / 'cuda/newton/direction_blas_dx_bridge.cu')).read_text()
        if newton_islands=='exact':
            bridge_source+='\n'+(GPU_ROOT/'cuda/newton/islands_dx_bridge.cu').read_text()
        bridge_paths=([bridge_path] if use_dx else [])+([blas_bridge_path,GPU_ROOT/'cuda/newton/direction_blas_dx_bridge.cu'] if use_blas else [])+([GPU_ROOT/'cuda/newton/islands_dx_bridge.cu'] if newton_islands=='exact' else [])
        stage_sources.update({str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [native_path]+bridge_paths})
        # Bind the byte hashes as well as expanded text: editing a dependency's
        # pragma/comment/line endings must never silently reuse an old build.
        source_closure=json.dumps(stage_sources,sort_keys=True,separators=(',',':'))
        digest=hashlib.sha256((source+bridge_source+source_closure+(dependency_sha or '')+'|fma=true|prec-div=true|prec-sqrt=true|real='+str(scalar.bits)).encode()).hexdigest();source_path=self.root/(digest+'.cu');target=self.root/(digest+'.so')
        source_path.write_text(source)
        product_authority=current_product_authority()
        if product_authority is not None:
            target=self.root/(digest+'_'+uuid.uuid4().hex+'.so')
        compiler=nvcc_path()
        command=[compiler,*scalar.native_flags(),'-I'+str(GPU_ROOT/'cuda/common'),'-shared','-Xcompiler','-fPIC','-O3','--fmad=true','--prec-div=true','--prec-sqrt=true',
                 '-arch=sm_%d'%wp.get_device(ws.device).arch,str(source_path),'-o',str(target)]
        commands=[command]
        if use_dx:
            bridge=self.root/(digest+'_bridge.cu');bridge.write_text(bridge_source)
            paths.append(bridge_path)
            if use_blas: paths.append(blas_bridge_path)
            if use_blas: paths.append((GPU_ROOT / 'cuda/newton/direction_blas_dx_bridge.cu'))
            if newton_islands=='exact':paths.append(GPU_ROOT/'cuda/newton/islands_dx_bridge.cu')
            architecture='-arch=sm_%d'%wp.get_device(ws.device).arch
            common=[compiler,*scalar.native_flags(),'-I'+str(GPU_ROOT/'cuda/common'),'-Xcompiler','-fPIC','-O3','-std=c++17','--expt-relaxed-constexpr',architecture]
            if product_authority is not None:
                # Private signed products must be unloadable after their last
                # borrower; GNU UNIQUE CCCL variables pin the DSO in glibc.
                common += ['-Xcompiler=-fno-gnu-unique',
                           '-Xlinker=--exclude-libs=libcusolverdx.a']
            main_object=self.root/(digest+'_main.o');bridge_object=self.root/(digest+'_bridge.o')
            commands=[common+['-dc','--fmad=true','--prec-div=true','--prec-sqrt=true',str(source_path),'-o',str(main_object)],
                      common+['-dc','-dlto','--fmad=true','-I'+str(dependency/'include'),
                              '-I'+str(dependency/'external/cutlass/include'),str(bridge),'-o',str(bridge_object)],
                      common+['-shared','-dlto','--fmad=true',str(main_object),str(bridge_object),
                              str(dependency/'lib/libcusolverdx.a'),'-o',str(target)]]
        resource_usage=None
        product_result=None
        dependency_inputs=None
        dependency_receipt_file=None
        from libero.gpu.host.runtime.core.product_paths import reuse_newton_binary, record_build, current_build_project
        if product_authority is None:
            project = current_build_project()
            if project is not None and not target.exists() and dependency is not None:
                from libero.gpu.host.build.aot import migrate_newton_path_alias, collect_newton_dependency_inputs
                from libero.gpu.host.build.compilation import compilation_allowed
                if compilation_allowed():
                    dependency_inputs=collect_newton_dependency_inputs(commands,
                        architecture=wp.get_device(ws.device).arch,
                        mathdx_archive=dependency/'lib/libcusolverdx.a')
                migrate_newton_path_alias(project['root'], target=target, identity=digest,
                    source=source, bridge_source=bridge_source, stage_sources=stage_sources,
                    mathdx_archive=dependency/'lib/libcusolverdx.a',
                    architecture=wp.get_device(ws.device).arch, commands=commands,
                    launch_command=command, nv=nv, nM=nm, dependency_inputs=dependency_inputs)
            reuse_newton_binary(target, digest, hashlib.sha256(source.encode()).hexdigest(),
                                wp.get_device(ws.device).arch)
            from libero.gpu.host.build.aot import admit_newton_cached_product
            admit_newton_cached_product(target, identity=digest,
                source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                architecture=wp.get_device(ws.device).arch,
                preserve_root=project['root'] if project is not None else self.root)
        reused_product = target.exists()
        def verify_completed_product():
            nonlocal resource_usage
            if use_dx:
                resources=subprocess.run([str(Path(compiler).parent/'cuobjdump'),'--dump-resource-usage',str(target)],capture_output=True,text=True)
                if resources.returncode:raise RuntimeError('CUDA resource inspection failed: '+resources.stderr)
                (self.root/'RESOURCE_USAGE.txt').write_text(resources.stdout)
                import re
                section=next((part for part in resources.stdout.split('Function ')
                              if part.splitlines() and 'handwritten_newton' in part.splitlines()[0]),None)
                if not section:raise RuntimeError('compiled Newton kernel missing from CUDA resource report')
                resource_usage={key.lower():int(value) for key,value in re.findall(r'(REG|STACK|SHARED|LOCAL|CONSTANT\[0\]):(\d+)',section)}
                if 'shared' not in resource_usage:raise RuntimeError('compiled shared-memory usage unavailable')
                if shared_bytes+resource_usage['shared']>48*1024:
                    raise ValueError(('Newton static+dynamic shared exceeds 48 KiB',shared_bytes,resource_usage))
            completed=load_cdll(target)
            if not hasattr(completed,'launch_persistent') or (adapter is not None and not hasattr(completed,'launch_entry')):
                raise RuntimeError('handwritten CUDA launcher symbol ABI missing')
        if product_authority is not None:
            def compile_log(index,result):
                mode='w' if index==0 else 'a'
                with (self.root/'compile.log').open(mode) as stream:
                    stream.write(result.stdout+result.stderr+'\n')
            product_result=receipted_artifact_build(commands=commands,
                units=[source_path]+([bridge] if use_dx else []),
                outputs={'newton.so':target},architecture=wp.get_device(ws.device).arch,
                source_closure=stage_sources,verify=verify_completed_product,
                logs=compile_log,required_symbols=('launch_persistent','launch_entry') if adapter is not None else ('launch_persistent',))
            record_build('newton-cuda', digest, product_result['compile_commands_executed'])
        else:
            try:
                if not target.exists():
                    from libero.gpu.host.build.compilation import require_compilation
                    require_compilation('Newton CUDA library', target)
                    if dependency is not None and dependency_inputs is None:
                        from libero.gpu.host.build.aot import collect_newton_dependency_inputs
                        dependency_inputs=collect_newton_dependency_inputs(commands,
                            architecture=wp.get_device(ws.device).arch,
                            mathdx_archive=dependency/'lib/libcusolverdx.a')
                    logs=[]
                    for invocation in commands:
                        result=subprocess.run(invocation,capture_output=True,text=True)
                        logs.append(result.stdout+result.stderr)
                        (self.root/'compile.log').write_text('\n'.join(logs))
                        if result.returncode:raise RuntimeError('handwritten CUDA compilation failed: '+str(self.root/'compile.log'))
                verify_completed_product()
                if not reused_product and dependency_inputs is not None:
                    from libero.gpu.host.build.aot import publish_newton_dependency_receipt
                    dependency_receipt_file=publish_newton_dependency_receipt(target,
                        identity=digest, inputs=dependency_inputs)
            except BaseException:
                # A failed fresh build/receipt may not become a later exact-key
                # cache hit. Existing verified binaries are never removed.
                if not reused_product:target.unlink(missing_ok=True)
                raise
            record_build('newton-cuda', digest, () if reused_product else commands)
        self.library=load_cdll(target);self.library.launch_persistent.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p];self.library.launch_persistent.restype=ct.c_int
        if adapter is not None:
            self.library.launch_entry.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_void_p]
            self.library.launch_entry.restype=ct.c_int
        with wp.ScopedStream(ws.stream):self.device_params=wp.array(np.frombuffer(bytes(self.params),dtype=np.uint8).copy(),dtype=wp.uint8,device=ws.device)
        self.owners=tuple(owners);self.tokens=tuple((a.ptr,a.shape,a.dtype) for a in self.owners)
        self.layout_tokens=tuple((a.ptr,tuple(a.shape),tuple(a.strides),a.dtype) for a in self.owners)
        self.layout_descriptors=tuple(array_layout(e['array'],world_stride_elements=e['stride'],owner_name=e['name'],
            shared_offset_bytes=e['offset']*scalar.itemsize if e['shared'] else None) for e in arrays.values())
        self.bindings=tuple((owner,key,value) for owner in (solver,ws,self) for key,value in vars(owner).items() if isinstance(value,wp.array))
        self.capacity=(ws.nworld,ws.rows,ws.contacts,ws.maxiter,ws.warmstart_enabled)
        self.manifest=dict(floating_policy={'fma':True,'fast_math':False,'prec_div':True,'prec_sqrt':True},source=str(source_path),sha256=digest,command=command,nv=nv,nM=nm,rows=ws.rows,contacts=ws.contacts,
            compile_commands=commands,device_bridge=None if not use_dx else str(bridge),
            nworld=ws.nworld,maxiter=ws.maxiter,block_size=block_size,
            execution_config={"strategy":self.execution_config.strategy,"block_threads":block_size,"tuning_status":self.execution_config.tuning_status,"selection_scope":"user-selected Newton default or explicit caller configuration; no automatic winner selection"},
            entry_specialization={'block_threads':block_size,'world_mapping':'blockIdx.y',
                'independent_stop':True,'minimum_iterations':1,'warmstart':bool(ws.warmstart_enabled),
                'diagnostic_profile':diagnostic_profile,'nv':nv,'nm':nm,'rows':ws.rows,'contacts':ws.contacts},
            stage_counts={k:len(v) for k,v in calls.items()},
            layout_version=LAYOUT_VERSION,private_layouts=self.layout_descriptors,
            shared_lifetime_contract={'views':'prefix/loop/tail; no overlap reuse of public owners',
                'dx_matrix':'loop MathDx scratch then external tail scratch after loop completes',
                'qacc':'external private_qacc consumes current resident view; stride0=0',
                'barrier':'collective stages separated by block synchronization'},
            handwritten=True,warp_cpp_dependency=False,shared_bytes=shared_bytes,shared_fields=[e['name'] for e in selected],
            static_shared_reserve=static_reserve,compiled_resources=resource_usage,resource_budget=budget,
            phase_profiling={'enabled':self.phase_profiling,'phases':list(phase_profile.PHASES) if self.phase_profiling else [],
                'trace_capacity':self.phase_trace_capacity,
                'trace_fields':getattr(self,'phase_trace_fields',[]),
                'clock':'globaltimer nanoseconds','scope':'world-private single CTA invocation; parent/child times overlap',
                'perturbation':'extra shared/synchronization/register state; never substitute for uninstrumented step timing'},
            cholesky_backend=cholesky_backend,mathdx_root=None if dependency is None else str(dependency),
            blas_backend=blas_backend,hessian_tile_k=hessian_k,
            hessian_algorithm=self.hessian_algorithm_configuration,
            hessian_algorithm_diagnostic_fields=['cone_batch_calls','cone_rank_calls','cone_gram_tiles',
                'base_reconstruct_tiles','adaptive_rebuild_calls','quadratic_changed_rows','cone_rank_rows','reserved'],
            islands_diagnostic_fields=['partition_calls','partitioned_calls','factor_calls','dense_fallbacks',
                'padded_cells_total','max_components','max_component_size','direction_calls'] if newton_islands=='exact' else [],
            blas_workspace_bytes=dx_bytes if use_blas else 0,
            blas_diagnostic_fields=['assembly_calls','gemm_tiles','direction_calls','jv_calls','transpose_calls','cone_gemm_calls','S07_jv_calls'],
            mathdx_library_sha256=dependency_sha,source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            potrf_diagnostic_fields=['main_calls','contact_calls','source_fallbacks','nonpositive_info'],
            source_bytes=len(source.encode()),diagnostic_tail_preserved=diagnostic_profile=='compatible',diagnostic_profile=diagnostic_profile,
            adapter_prefix_tail=adapter is not None,external_prefix_stages=len(external_code)//2,
            external_tail_stages=len(external_tail_code)//2,witness_slots=solver.tape_slots,
            ordinal_head_metadata=dict(requested=head_metadata is not None,enabled=head_source['enabled'],
                fallback_reason=head_source['fallback_reason'],
                scope='fast single_cta N=8 cuBLASDx combined entry; qualification candidate',
                gate='same invocation after first ordinary S04 update; exact current heads/G and both N numeric views'),
            scalar_io=self.scalar_io,
            io_execution_policy={'qualification_shapes':[21,43],
                'path':'pre_vector_scalar' if self.scalar_io else 'algorithm_packet',
                'preserves_algorithm_configuration':True,
                'whole_dx_current_shapes':[21,43,57],
                'large_capacity_extension':'BlockedFactorPlan T16 retained; no full Newton capacity claim'},
            dataflow_features=['invocation-static-validation-cache','state-produced-counts','state-produced-change-count','validated-compact-row-topology','completion-tile-head-offsets',
                'selected-warm-init-reuse','stats-produced-gradient-norm-reuse','resident-factor-trsm','uniform-math-stop','compatible-inactive-history-fanout' if diagnostic_profile=='compatible' else 'explicit-witness-disabled',
                'Jv-tile-to-quad' if diagnostic_profile=='fast' and use_blas else 'compatible-quad-materialization'],
            project_header_binding='recursive quoted project headers; conditional pragma-once guards',
            stage_sources=stage_sources)
        self.manifest['library_file']=str(target.resolve())
        self.manifest['library_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
        retained_dependency_receipt=target.with_suffix('.dependencies.json')
        if dependency_receipt_file is not None or retained_dependency_receipt.is_file():
            self.manifest['dependency_receipt_file']=str(retained_dependency_receipt)
        self.manifest['compile_commands_executed'] = (product_result['compile_commands_executed']
            if product_result is not None else ([] if reused_product else commands))
        if product_result is not None:self.manifest['product_build']=product_result
        target.with_suffix('.json').write_text(json.dumps(self.manifest, indent=2)+'\n')
        (self.root/'MANIFEST.json').write_text(json.dumps(self.manifest,indent=2)+'\n')

    def profile_snapshot(self, *, stream=None):
        """Download completed diagnostics outside measured intervals."""
        if not self.phase_profiling:raise RuntimeError('phase profiling not enabled')
        wp.synchronize_stream(stream or (self.adapter.stream if self.adapter is not None else self.ws.stream))
        return phase_profile.decode(self.phase_profile_values.numpy())

    def profile_trace_snapshot(self, *, stream=None):
        """Download bounded invocation records after the measured GPU work."""
        if not self.phase_trace_capacity:raise RuntimeError('phase trace not enabled')
        wp.synchronize_stream(stream or (self.adapter.stream if self.adapter is not None else self.ws.stream))
        return dict(counts=self.phase_trace_counts.numpy(),
                    phases=self.phase_profile_trace.numpy(),
                    metrics=self.phase_trace_metrics.numpy(),
                    phase_names=phase_profile.PHASES,metric_names=self.phase_trace_fields,
                    capacity=self.phase_trace_capacity)

    def _validate_launch(self):
        # Retain inherited admission/capacity/root checks before external seals.
        super()._validate_launch()
        if tuple((a.ptr,tuple(a.shape),tuple(a.strides),a.dtype) for a in self.owners)!=self.layout_tokens:
            raise ValueError('physical layout changed; cold rebuild required')
        for validate in self.external_binding_validators:
            validate()

    def launch_entry(self,upstream,enabled,publication,epoch_plan,supplied=False):
        """One block/world owns admission, work, publication and exact public hold.

        Scalars and pointer roots travel as launch arguments; no hot host physics
        or parameter-buffer upload is introduced. The adapter retains its cold
        source/owner checks and validates live upstream masks before this call.
        """
        if self.adapter is None:raise RuntimeError('cold adapter fusion was not enabled')
        adapter=self.adapter;ws=self.ws
        for name,value,template in (('upstream',upstream,ws.err_code),('enabled',enabled,ws.active),
            ('publication',publication,adapter.publication_error),('epoch_plan',epoch_plan,adapter.n_expected)):
            adapter._check_array(name,value,template)
        self._validate_launch()
        self.entry_owners=(upstream,enabled,publication,epoch_plan)
        result=self.library.launch_entry(self.device_params.ptr,ws.nworld,upstream.ptr,enabled.ptr,
            publication.ptr,epoch_plan.ptr,int(bool(supplied)),ws.stream.cuda_stream)
        if result:raise RuntimeError(('fused Newton entry launch',result))
