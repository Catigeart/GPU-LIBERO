"""Handwritten S position/reference CUDA, with source-derived ABI bindings only.

The AST reader recovers launch operands, never arithmetic. Mathematical bodies
are manually maintained in constraints*.cuh. Existing adapter ownership and
all-or-nothing public commits are retained. Dense elliptic S families only,
as already admitted by the native dimension specialization factory.
"""
import libero.gpu.precision as scalar
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
import ast
import inspect
import textwrap
import types
import os
from collections import UserDict
from pathlib import Path
import warp as wp
import numpy as np
from libero.gpu.host.build.native import NativePlan
from libero.gpu.host.runtime.core.tiled_execution import TiledExecution, TiledPhaseContract
from libero.gpu.host.runtime.core.execution_config import ExecutionConfig
from libero.gpu.model.constraint_pairs import build_constraint_pair_support

ROOT = (GPU_ROOT / 'cuda/constraints')
BLOCK = {'_preflight', '_jacobian_rows', '_metadata', '_reference', '_impedance',
         '_commit_position', '_commit_reference', '_produce', '_assemble', '_reference_gate', '_prediag'}
ORDER = ('_position_gate', '_reset_stage_status', '_produce', '_contact_incoming',
         '_gate_active_status', '_preflight', '_jacobian_rows', '_metadata',
         '_assemble', '_prediag', '_impedance', '_merge_stage_status', '_commit_position')
REF_ORDER = ('_reference_gate', '_reference', '_commit_reference')

def constraint_feature_selection(profile, nv, nworld, mapping, *, qualification_mode=None, environ=None):
    """Resolve cold candidate flags without claiming new GPU qualification.

    Unset mode retains the previous shape defaults and per-flag overrides.
    Explicit N8 modes isolate unified metadata and group parameter reuse; a
    conflicting experimental flag is rejected rather than silently combined.
    """
    env = os.environ if environ is None else environ
    mode = qualification_mode if qualification_mode is not None else env.get('LIBERO_CUDA_S_QUALIFICATION_MODE')
    candidates = {'baseline': (0, 0), 'unified': (1, 0),
                  'group': (0, 1), 'unified_group': (1, 1)}
    if mode is not None:
        if mode not in candidates:
            raise ValueError('S qualification mode must be baseline/unified/group/unified_group')
        if profile != 'fast' or nworld != 8 or mapping != 'single_cta':
            raise ValueError('S qualification modes require fast N8 single_cta')
    names = ('BATCH_COMPACT', 'GROUP_PARAMETER_REUSE', 'UNIFIED_METADATA',
             'STATIC_CANDIDATES', 'INTERNAL_OWNER_RECEIPT', 'COMMON_ANCESTOR_ZERO')
    selected_defaults = ('BATCH_COMPACT', 'GROUP_PARAMETER_REUSE',
                         'UNIFIED_METADATA', 'INTERNAL_OWNER_RECEIPT')
    retained_shape_default = profile == 'fast' and nv == 43 and nworld in (1, 2)
    desired = {name: 0 for name in names}
    if mode is not None:
        desired['UNIFIED_METADATA'], desired['GROUP_PARAMETER_REUSE'] = candidates[mode]
    flags = {}
    for name in names:
        default = desired[name] if mode is not None else int(retained_shape_default and name in selected_defaults)
        env_name = 'COMMON_ZERO' if name == 'COMMON_ANCESTOR_ZERO' else name
        value = int(env.get('LIBERO_CUDA_S_' + env_name, default))
        if value not in (0, 1):
            raise ValueError('S ' + name + ' must be 0 or 1')
        if mode is not None and value != desired[name]:
            raise ValueError('S qualification mode conflicts with ' + name)
        flags[name] = value
    return dict(mode=mode or 'retained_defaults', flags=flags,
                candidate_pending_gpu=mode is not None,
                domain=dict(profile=profile, nv=int(nv), nworld=int(nworld), mapping=mapping),
                qualification='existing defaults retained; explicit N8 modes require separate numerical qualification')


def _original_method(obj, name):
    # Cold fused subclasses wrap the accepted schedule; inspect its base method.
    for cls in type(obj).__mro__:
        method = cls.__dict__.get(name)
        if method is not None and 'wp.launch(' in inspect.getsource(method):
            return method
    raise ValueError('original S launch schedule unavailable: '+name)

def _binding_schedule(obj, name):
    function = _original_method(obj, name)
    tree = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
    globals_ = dict(function.__globals__)
    local = {'self': obj, 'stream': None}
    result = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            # Only local launch bindings/options; never execute self mutations.
            if all(isinstance(n, (ast.Name, ast.Tuple)) for n in node.targets):
                exec(compile(ast.Module(body=[node], type_ignores=[]), '<S ABI locals>', 'exec'), globals_, local)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if ast.unparse(call.func) == 'wp.launch':
                kernel = eval(compile(ast.Expression(call.args[0]), '<S ABI kernel>', 'eval'), globals_, local)
                operands = next(k.value for k in call.keywords if k.arg == 'inputs')
                values = eval(compile(ast.Expression(operands), '<S ABI operands>', 'eval'), globals_, local)
                result.append((kernel, values))
            elif isinstance(call.func, ast.Attribute) and call.func.attr == 'launch':
                child = eval(compile(ast.Expression(call.func.value), '<S ABI segment>', 'eval'), globals_, local)
                result.extend(_binding_schedule(child, 'launch'))
    return result

def _ctype(value):
    if isinstance(value, wp.array):
        return {scalar.wp_real:'real_t*', wp.int32:'int32_t*', wp.int64:'int64_t*'}[value.dtype], value
    if isinstance(value, scalar.wp_real):
        return 'real_t', float(value.value)
    if isinstance(value, (float, np.floating)):
        return 'real_t', float(value)
    return 'int32_t', int(value)

def _private_token(a):
    return (id(a),int(a.ptr),tuple(a.shape),tuple(a.strides),a.dtype,str(a.device))


def _validate_private_storage(workspace,storage,owners,tokens):
    if getattr(workspace,'_cuda_constraints_storage',None) is not storage:
        raise ValueError('S private storage owner changed')
    if set(storage)!=set(owners):
        raise ValueError('S private storage keyset changed')
    for key,a in storage.items():
        if a is not owners[key] or _private_token(a)!=tokens[key]:
            raise ValueError('S private storage binding changed: '+key)


class ContactTilePositionPlan(TiledExecution):
    """Ordered prepare/worker/commit launches; only commit owns publication.

    This deliberately tiles Jacobian contacts only. Prefix/preflight and all
    metadata/impedance/finite checks remain the accepted world coordinator.
    No new arithmetic, atomic row allocation, or host hot-count readback.
    """
    def __init__(self, prepare, workers, commit, split, contact_tile):
        contract=TiledPhaseContract('constraints_contact_jacobian', workers.work_tiles,
            contact_tile)
        super().__init__(prepare, workers, commit, contract)
        self.plans = (prepare, workers, commit)
        self._split = split
        self.manifest.update(name='constraints_position_contact_tiles',
            kernel_count=3, execution='prepare_worker_commit',
            worker_scope='complete_contact_jacobian',
            public_commit_owner='commit_only',
            phase_sync='same_stream_kernel_boundaries',
            plans=[plan.manifest for plan in self.plans])

    def set_scalar(self, stage, index, value, stream=None):
        if stage < self._split:
            return self.prepare.set_scalar(stage, index, value, stream)
        if stage == self._split:
            return self.workers.set_scalar(0, index, value, stream)
        return self.commit.set_scalar(stage-self._split-1, index, value, stream)

    def launch(self, stream=None):
        super().launch(stream)


class ContactGroupPositionPlan:
    """Two disjoint-private worker waves, five same-stream kernel boundaries.

    The first wave owns Jacobian contact tiles. The second owns validated
    impedance-group ordinals; its prepare computes stable heads and mu owners.
    Only the final coordinator writes public arrays.
    """
    def __init__(self, plans, scalar_routes, contact_tile):
        self.plans=tuple(plans)
        self.prepare,self.workers,self.group_prepare,self.group_workers,self.commit=self.plans
        self._scalar_routes=scalar_routes
        self._sealed=self._structure()
        if any(p.nworld!=self.prepare.nworld or p.device!=self.prepare.device for p in self.plans):
            raise ValueError('S group phases worlds/device mismatch')
        if any(p.work_tiles!=1 for p in (self.prepare,self.group_prepare,self.commit)):
            raise ValueError('S coordinators must be unique')
        for worker in (self.workers,self.group_workers):
            if any(m!='tile' for m in worker.manifest['stage_execution_modes']):
                raise ValueError('S workers require explicit tile ABI')
        self.manifest=dict(name='constraints_position_contact_group_tiles',kernel_count=5,
            worker_scope='contact_Jacobian_and_impedance_groups',contact_tile=contact_tile,
            public_commit_owner='final_coordinator_only',phase_sync='same_stream_kernel_boundaries',
            device_ownership_proof=False,plans=[p.manifest for p in self.plans])
    def _structure(self):
        return (tuple((id(p),p.nworld,p.work_tiles,p.block_size,p.dynamic_shared_bytes,tuple(p.manifest.get('stage_execution_modes',()))) for p in self.plans),
                tuple(id(p) for p in (self.prepare,self.workers,self.group_prepare,self.group_workers,self.commit)),
                tuple((key,tuple((id(p),i) for p,i in value)) for key,value in sorted(self._scalar_routes.items())))
    def validate_bindings(self):
        if self._sealed!=self._structure():
            raise ValueError('S group execution structure changed')
        for p in self.plans:p.validate_bindings()
    def set_scalar(self, stage, index, value, stream=None):
        for plan,local in self._scalar_routes[stage]:plan.set_scalar(local,index,value,stream)
    def launch(self,stream=None):
        self.validate_bindings()
        for p in self.plans:p.launch(stream=stream)


class ConstraintPlans:
    def __init__(self, workspace, build_root, adapter=None, dataflow_profile=None, execution_config=None, qualification_mode=None):
        self.workspace = workspace
        profile = dataflow_profile or os.environ.get('LIBERO_CUDA_DATAFLOW_PROFILE', 'compatible')
        if profile == 'compatibility': profile = 'compatible'
        if profile not in ('compatible', 'fast'):
            raise ValueError('S dataflow profile must be compatible or fast')
        self.profile = profile
        mapping = os.environ.get('LIBERO_CUDA_S_MAPPING', 'single_cta')
        if mapping not in ('single_cta', 'contact_tiles'):
            raise ValueError('S mapping must be single_cta or contact_tiles')
        if execution_config is not None:
            if not isinstance(execution_config, ExecutionConfig):
                raise TypeError('S execution_config must be ExecutionConfig')
            if execution_config.strategy == 'cluster':
                raise ValueError('S cluster execution is not implemented')
            mapping = 'contact_tiles' if execution_config.strategy == 'multi_cta' else 'single_cta'
        self.execution_config = execution_config
        self.mapping = mapping
        group_tiles=os.environ.get("LIBERO_CUDA_S_WORKER_SCOPE", "jacobian") == "groups"
        if os.environ.get("LIBERO_CUDA_S_WORKER_SCOPE", "jacobian") not in ("jacobian","groups"):
            raise ValueError("S worker scope must be jacobian or groups")
        contact_tile = int(os.environ.get('LIBERO_CUDA_S_CONTACT_TILE', '32'))
        worker_threads = int(os.environ.get('LIBERO_CUDA_S_BLOCK_SIZE', '128'))
        if execution_config is not None:
            contact_tile = execution_config.contact_tile
            worker_threads = execution_config.block_threads
        if contact_tile not in (16, 32, 64, 128):
            raise ValueError('S contact tile must be 16/32/64/128')
        if worker_threads not in (32, 64, 128, 256):
            raise ValueError('S worker block must be 32/64/128/256')
        work_tiles=max(1,(workspace.contacts+contact_tile-1)//contact_tile)
        if mapping == 'contact_tiles' and execution_config is not None and execution_config.workers_per_world != work_tiles:
            raise ValueError('S workers_per_world must equal ceil(contact_capacity/contact_tile); use contact_tile_config')
        self.contact_tile = contact_tile
        self.worker_threads = worker_threads
        self.work_tiles = work_tiles
        positions = _binding_schedule(workspace, 'launch_position')
        reference = _binding_schedule(workspace, 'launch_reference')
        if tuple(k.func.__name__ for k,v in positions) != ORDER:
            raise ValueError('S position source launch order changed')
        if tuple(k.func.__name__ for k,v in reference) != REF_ORDER:
            raise ValueError('S reference source launch order changed')
        facts = workspace.model_facts
        nv,nq,nj,nb = (facts[k] for k in ('nv','nq','njnt','nbody'))
        nf = workspace.s02.friction_pos.size // workspace.nworld
        nl = workspace.s02.limit_pos.size // workspace.nworld
        self.feature_selection = constraint_feature_selection(profile, nv, workspace.nworld,
            mapping, qualification_mode=qualification_mode)
        experiment_flag = self.feature_selection['flags'].__getitem__
        common_zero = experiment_flag('COMMON_ANCESTOR_ZERO')
        defines = dict(NV=nv,NQ=nq,NJ=nj,NF_CAP=nf,NL_CAP=nl,
                       S_NBODY=nb,S_ROWS=workspace.rows,S_CONTACTS=workspace.contacts,
                       CF_S_COMPATIBLE=int(profile == 'compatible'),CF_S_KBIP_PLANES=int(profile == 'fast'),CF_S_WARPS=2,
                       CF_S_COMMON_ANCESTOR_ZERO=common_zero,
                       CF_S_BATCH_COMPACT=experiment_flag('BATCH_COMPACT'),
                       CF_S_GROUP_PARAMETER_REUSE=experiment_flag('GROUP_PARAMETER_REUSE'),
                       CF_S_UNIFIED_METADATA=experiment_flag('UNIFIED_METADATA'),
                       CF_S_STATIC_CANDIDATES=experiment_flag('STATIC_CANDIDATES'),
                       CF_S_INTERNAL_OWNER_RECEIPT=experiment_flag('INTERNAL_OWNER_RECEIPT'))
        # The original ContactStatic seal already qualified this dense ancestor
        # owner. Upload exactly its ordered CSR; no new topology interpretation.
        ancestor=np.asarray(workspace.s01.static.ancestor_mask,dtype=np.int32)
        if ancestor.shape!=(nb,nv) or np.any((ancestor!=0)&(ancestor!=1)):
            raise ValueError('S ancestor CSR source schema')
        offsets=np.concatenate((np.zeros(1,np.int32),np.cumsum(np.count_nonzero(ancestor,axis=1),dtype=np.int32)))
        indices=np.nonzero(ancestor)[1].astype(np.int32)
        # Enable bounded structural reuse by default; zero budget disables
        # the cache, and missing/budgeted-out pairs retain ordered CSR merge.
        pair_budget=int(os.environ.get('LIBERO_CUDA_S_PAIR_CACHE_BYTES',str(8*1024*1024)))
        pair_support=build_constraint_pair_support(ancestor,byte_budget=pair_budget)
        self.pair_support=pair_support
        # Conservative structural support is available to future solver graph
        # planning; a body-pair union alone does not establish independence.
        workspace._cuda_constraint_pair_metadata=pair_support
        def nonempty(value):
            return value if value.size else np.asarray([-1],dtype=np.int32)
        storage=UserDict({
            'cold_friction_map':wp.array(nonempty(np.flatnonzero(workspace.static_checked['dof_frictionloss']>0).astype(np.int32)),dtype=wp.int32,device=workspace.device),
            'cold_friction_mask':wp.array((workspace.static_checked['dof_frictionloss']>0).astype(np.int32),dtype=wp.int32,device=workspace.device),
            'cold_limit_sides':wp.array(nonempty(np.flatnonzero(np.repeat(workspace.static_checked['jnt_limited']!=0,2)).astype(np.int32)),dtype=wp.int32,device=workspace.device),
            'cold_limited_mask':wp.array((workspace.static_checked['jnt_limited']!=0).astype(np.int32),dtype=wp.int32,device=workspace.device),
            'body_dof_offsets':wp.array(offsets,dtype=wp.int32,device=workspace.device),
            'body_dof_indices':wp.array(indices,dtype=wp.int32,device=workspace.device),
            'body_pair_lookup':wp.array(nonempty(pair_support.pair_lookup),dtype=wp.int32,device=workspace.device),
            'pair_support_offsets':wp.array(pair_support.pair_offsets,dtype=wp.int32,device=workspace.device),
            'pair_support_columns':wp.array(nonempty(pair_support.pair_columns),dtype=wp.int32,device=workspace.device),
            'pair_support_members':wp.array(nonempty(pair_support.pair_members),dtype=wp.int32,device=workspace.device),
            'contact_queue':wp.zeros(workspace.nworld*workspace.contacts,dtype=wp.int32,device=workspace.device),
            'contact_dim_offsets':wp.zeros(workspace.nworld*5,dtype=wp.int32,device=workspace.device),
            'contact_queue_valid':wp.zeros(workspace.nworld,dtype=wp.int32,device=workspace.device),
            'compact_heads':wp.zeros(workspace.nworld*workspace.rows,dtype=wp.int32,device=workspace.device),
            'compact_count':wp.zeros(workspace.nworld,dtype=wp.int32,device=workspace.device),
            'compact_valid':wp.zeros(workspace.nworld,dtype=wp.int32,device=workspace.device),
            'head_type':wp.zeros(workspace.nworld*workspace.rows,dtype=wp.int32,device=workspace.device),
            'head_id':wp.zeros(workspace.nworld*workspace.rows,dtype=wp.int32,device=workspace.device),
            'head_dim':wp.zeros(workspace.nworld*workspace.rows,dtype=wp.int32,device=workspace.device),
            'head_cone_valid':wp.zeros(workspace.nworld,dtype=wp.int32,device=workspace.device),
            'head_mu':wp.zeros(workspace.nworld*workspace.rows,dtype=scalar.wp_real,device=workspace.device),
            'head_friction':wp.zeros(workspace.nworld*workspace.rows*5,dtype=scalar.wp_real,device=workspace.device),
            'group_math_error':wp.zeros(workspace.nworld,dtype=wp.int32,device=workspace.device),
            'producer_receipt':wp.zeros(workspace.nworld*4,dtype=wp.int32,device=workspace.device),
            'mu_owner':wp.zeros(workspace.nworld*workspace.contacts,dtype=wp.int32,device=workspace.device),
        })
        # UserDict is deliberately outside the original adapter's dict/array
        # public inventory. Its private owners have a separate strict seal.
        workspace._cuda_constraints_storage=storage
        self._storage=storage
        self._storage_owners=dict(storage)
        self._storage_tokens={k:_private_token(a) for k,a in storage.items()}
        queue=[('int32_t*','contact_queue',storage['contact_queue']),
               ('int32_t*','contact_dim_offsets',storage['contact_dim_offsets']),
               ('int32_t*','contact_queue_valid',storage['contact_queue_valid'])]
        signatures=[];stage_parameters={}
        def stages(entries):
            result=[]
            for kernel,values in entries:
                fname=kernel.func.__name__.lstrip('_')
                parameters=tuple(inspect.signature(kernel.func).parameters)
                if len(parameters)!=len(values):raise ValueError('S ABI parameter count')
                operands=[_ctype(v) for v in values]
                ctypes_,values=map(list,zip(*operands))
                # Only pointer/scalar bindings are synthesized here. Row math
                # stays handwritten; producers own the final private layout.
                row_metadata=[('real_t*','unified_pos',workspace.stage_pos),
                              ('real_t*','unified_margin',workspace.stage_margin),
                              ('real_t*','unified_loss',workspace.stage_frictionloss),
                              ('int32_t*','unified_type',workspace.stage_type),
                              ('int32_t*','unified_id',workspace.stage_id)]
                receipt=[('int32_t*','producer_receipt',storage['producer_receipt'])]
                extras = {
                    'position_gate': receipt,
                    'produce': [('real_t*', 'unified_J', workspace.stage_J),*row_metadata,
                                ('const int32_t*','cold_friction_map',storage['cold_friction_map']),
                                ('const int32_t*','cold_friction_mask',storage['cold_friction_mask']),
                                ('const int32_t*','cold_limit_sides',storage['cold_limit_sides']),
                                ('const int32_t*','cold_limited_mask',storage['cold_limited_mask']),
                                ('int32_t','cold_friction_count',int(np.count_nonzero(workspace.static_checked['dof_frictionloss']>0))),
                                ('int32_t','cold_limit_count',int(2*np.count_nonzero(workspace.static_checked['jnt_limited'])))],
                    'preflight': queue,
                    'metadata': [*queue,*row_metadata,*receipt],
                    'assemble': receipt,
                    'prediag': receipt,
                    'jacobian_rows': [('real_t*', 'unified_J', workspace.stage_J),
                                      ('int32_t*', 'unified_incoming', workspace.incoming_contact),
                                      ('const int32_t*','body_dof_offsets',storage['body_dof_offsets']),
                                      ('const int32_t*','body_dof_indices',storage['body_dof_indices']),
                                      ('const int32_t*','body_pair_lookup',storage['body_pair_lookup']),
                                      ('const int32_t*','pair_support_offsets',storage['pair_support_offsets']),
                                      ('const int32_t*','pair_support_columns',storage['pair_support_columns']),
                                      ('const int32_t*','pair_support_members',storage['pair_support_members']),
                                      ('int32_t','pair_cache_enabled',int(bool(pair_support.pair_lookup.size))),
                                      ('const real_t*','cdof_planes',workspace.s01.cdof),
                                      ('int32_t','cdof_pitch',nv),('int32_t','cdof_layout',0),*queue],
                    'impedance': [('int32_t*', 'group_heads', workspace._cuda_impedance_heads),
                                  ('int32_t*','compact_heads',storage['compact_heads']),
                                  ('int32_t*','compact_count',storage['compact_count']),
                                  ('int32_t*','compact_valid',storage['compact_valid']),
                                  ('int32_t*','mu_owner',storage['mu_owner']),
                                  ('int32_t*','head_type',storage['head_type']),
                                  ('int32_t*','head_id',storage['head_id']),
                                  ('int32_t*','head_dim',storage['head_dim'])],
                }.get(fname, [])
                parameters = list(parameters)
                for ctype, parameter, value in extras:
                    ctypes_.append(ctype); parameters.append(parameter); values.append(value)
                stage_parameters[fname]=tuple(parameters)
                signature=f'#define CF_S_{fname} {fname}(int w, '+', '.join(t+' '+p for p,t in zip(parameters,ctypes_))+')'
                signatures.append(signature)
                if fname == 'impedance':
                    signatures.append(signature.replace('CF_S_impedance impedance(',
                        'CF_S_impedance_validate impedance_validate('))
                    signatures.append('#define CF_S_CALL_impedance_validate impedance_validate(w, '+
                        ', '.join(parameters)+')')
                    for part in ('prepare','math'):
                        signatures.append(signature.replace('CF_S_impedance impedance(',
                            'CF_S_impedance_'+part+' impedance_'+part+'('))
                        signatures.append('#define CF_S_CALL_impedance_'+part+' impedance_'+part+'(w, '+
                            ', '.join(parameters)+')')
                result.append(('cf_s::'+fname,values,'block' if kernel.func.__name__ in BLOCK else 'lane0',ctypes_))
            return result
        pstages=stages(positions);rstages=stages(reference)
        if adapter is not None:
            pstages.append(('cf_s::publish_exclude',[workspace.active,workspace.status,
                workspace.committed,workspace.ncon,adapter._exclude_source,adapter._exclude_target,
                workspace.contacts],'block'))
        source=(GPU_ROOT / 'cuda/common/warp.cuh').read_text()+'\n'
        source+='\n'.join(signatures)+'\n'
        source+=((GPU_ROOT / 'cuda/constraints/constraints.cuh')).read_text()+'\n'+((GPU_ROOT / 'cuda/constraints/constraints_impedance.cuh')).read_text()
        # Adjacent lane-0 producers execute in one ordered thread, and contact
        # metadata does not read Jacobian values. Synchronize at the actual
        # block hand-offs rather than after every source-stage name.
        position_barriers=[False,True,True,False,True,True,False,True,
                           True,True,True,True,True]
        if adapter is not None: position_barriers.append(True)
        if mapping == 'single_cta':
            self.position=NativePlan('constraints_position',[source],pstages,workspace.nworld,
                build_root,block_size=worker_threads if execution_config is not None else 64,
                defines=dict(defines,CF_S_WARPS=(worker_threads//32 if execution_config is not None else 2)),
                stage_barriers=position_barriers)
        else:
            split=ORDER.index('_jacobian_rows')
            worker_source=source.replace(
                '#define CF_S_jacobian_rows jacobian_rows(int w, ',
                '#define CF_S_jacobian_rows jacobian_rows(int w, int tile, int tile_count, ')
            worker_defines=dict(defines,CF_S_CONTACT_TILES=1,
                CF_S_CONTACT_TILE=contact_tile,CF_S_WARPS=worker_threads//32)
            prepare=NativePlan('constraints_position_prepare',[source],pstages[:split],
                workspace.nworld,build_root,defines=defines,
                stage_barriers=position_barriers[:split])
            function,values,mode,types_=pstages[split]
            workers=NativePlan('constraints_position_jacobian_tiles',[worker_source],
                [(function,values,'tile',types_)],workspace.nworld,build_root,
                block_size=worker_threads,defines=worker_defines,
                work_tiles=work_tiles)
            commit=None if group_tiles else NativePlan('constraints_position_commit',[source],pstages[split+1:],
                workspace.nworld,build_root,defines=defines,
                stage_barriers=position_barriers[split+1:])
            if group_tiles:
                impedance_stage=ORDER.index('_impedance')
                f,iv,im,it=pstages[impedance_stage]
                init=('cf_s::impedance_math_reset',[storage['group_math_error']], 'block',['int32_t*'])
                group_prepare=NativePlan('constraints_position_group_prepare',[source],
                    pstages[split+1:impedance_stage]+[(f.replace('impedance','impedance_prepare'),iv,im,it),init],
                    workspace.nworld,build_root,defines=defines,
                    stage_barriers=[True]*(impedance_stage-split+1))
                group_source=source.replace('#define CF_S_impedance_math impedance_math(int w, ',
                    '#define CF_S_impedance_math impedance_math(int w, int tile, int tile_count, ')
                group_source=group_source.replace('int32_t* head_dim)', 'int32_t* head_dim, int32_t* group_math_error)')
                # The CALL macro in the unused ordinary wrapper still requires
                # the tile ABI when this source is compiled as a worker unit.
                group_source=group_source.replace('#define CF_S_CALL_impedance_math impedance_math(w, ',
                    '#define CF_S_CALL_impedance_math impedance_math(w, tile, tile_count, ')
                group_source=group_source.replace('head_dim)', 'head_dim, group_math_error)')
                # Worker translation unit never instantiates the ordinary entry.
                group_source=group_source.replace('__device__ void CF_S_impedance {', '#if !CF_S_GROUP_TILES\n__device__ void CF_S_impedance {')
                group_source=group_source.replace('__device__ void impedance_math_reset', '#endif\n__device__ void impedance_math_reset')
                group_capacity=min(workspace.rows,workspace.contacts+nf+nl)
                group_tile=int(os.environ.get('LIBERO_CUDA_S_GROUP_TILE','128'))
                if group_tile not in (32,64,128,256):raise ValueError('S group tile must be 32/64/128/256')
                group_count=max(1,(group_capacity+group_tile-1)//group_tile)
                group_workers=NativePlan('constraints_position_impedance_tiles',[group_source],
                    [(f.replace('impedance','impedance_math'),iv+[storage['group_math_error']],'tile',it+['int32_t*'])],
                    workspace.nworld,build_root,block_size=worker_threads,
                    defines=dict(worker_defines,CF_S_CONTACT_TILES=0,CF_S_GROUP_TILES=1,CF_S_GROUP_TILE=group_tile),work_tiles=group_count)
                merge=('cf_s::impedance_math_merge',[storage['group_math_error'],workspace.s03.status],
                    'block',['const int32_t*','int32_t*'])
                commit=NativePlan('constraints_position_group_commit',[source],
                    [merge]+pstages[impedance_stage+1:],workspace.nworld,build_root,
                    defines=defines,stage_barriers=[True]*(len(pstages)-impedance_stage))
                routes={i:[] for i in range(len(pstages))}
                for i in range(split):routes[i]=[(prepare,i)]
                routes[split]=[(workers,0)]
                for i in range(split+1,impedance_stage):routes[i]=[(group_prepare,i-split-1)]
                routes[impedance_stage]=[(group_prepare,impedance_stage-split-1),(group_workers,0)]
                for i in range(impedance_stage+1,len(pstages)):routes[i]=[(commit,i-impedance_stage)]
                self.position=ContactGroupPositionPlan((prepare,workers,group_prepare,group_workers,commit),routes,contact_tile)
                self.position.manifest.update(group_capacity=group_capacity,group_work_tiles=group_count,group_tile=group_tile,
                    runtime_extent='validated_compact_count',mu_owner='last_original_row_head_per_contact',
                    error_selection='minimum_original_group_ordinal_private_atomic_then_coordinator')
            else:
                self.position=ContactTilePositionPlan(prepare,workers,commit,split,contact_tile)
            self.position.manifest.update(contact_tile=contact_tile,
                worker_threads=worker_threads, tiled_capacity=workspace.contacts,
                actual_work_tiles=work_tiles,worker_policy='one_CTA_per_capacity_tile',
                runtime_extent='validated_contact_queue_count',
                offsets='single_world_preflight', metadata='single_world_group_prepare' if group_tiles else 'single_world_commit')
        ref_defines=dict(defines,CF_S_DX_SM=wp.get_device(workspace.device).arch*10)
        self.reference=NativePlan('constraints_reference',[source,(GPU_ROOT / 'cuda/constraints/constraints_reference.cuh')],rstages,
            workspace.nworld,build_root,defines=ref_defines,mathdx=True)
        self.scalar_stage=ORDER.index('_impedance')
        self.scalar_indices=tuple(stage_parameters['impedance'].index(k) for k in ('timestep','impratio','refsafe'))
        for plan in (self.position, self.reference):
            plan.manifest['feature_selection'] = dict(self.feature_selection,
                model_sha256=workspace.s01.static.model_sha256)
            plan.manifest['dataflow'] = dict(profile=profile,
                unified_private_J=True, group_impedance=True,
                parallel_transaction_commit=True, diagnostic_writes=profile == 'compatible')
        self.position.manifest['pair_support_cache']=dict(
            lookup_layout='dense',
            cached_body_pairs=pair_support.cached_body_pairs,
            total_body_pairs=nb*nb,bytes=pair_support.nbytes,
            byte_budget=pair_support.byte_budget,source_sha256=pair_support.source_sha256,
            column_order='ascending_original_dof',fallback='ordered_body_CSR_merge',
            exact_common_ancestor_zero=profile=='fast' and bool(common_zero),
            common_ancestor_zero_guard='finite_frame_offset<=2e100_cdof<=1e100',
            sparse_support='conservative_union_not_independence_proof')

    def validate_private_bindings(self):
        _validate_private_storage(self.workspace,self._storage,self._storage_owners,self._storage_tokens)
        if getattr(self.workspace,'_cuda_constraint_pair_metadata',None) is not self.pair_support:
            raise ValueError('S pair support metadata owner changed')

    def launch_position(self, stream=None):
        self.validate_private_bindings()
        s=self.workspace.s03
        s.last_launch_scalars=dict(timestep=np.asarray(s.timestep,np.float64),
            impratio=np.asarray(s.impratio,np.float64),disableflags=np.asarray(s.disableflags,np.int32),
            refsafe=np.asarray(int(s.refsafe),np.int32))
        # By-value option snapshots follow the original launch semantics;
        # captured graphs retain the options recorded at capture time.
        for j,value in enumerate((float(s.timestep),float(s.impratio),int(s.refsafe))):
            self.position.set_scalar(self.scalar_stage,self.scalar_indices[j],value,stream)
        self.position.launch(stream)

    def launch_reference(self, stream=None):
        self.validate_private_bindings()
        self.reference.launch(stream)

def _install_private_validation(adapter, plans):
    # Retain the full generated public seal; extend its host preflight so
    # cached prefix/recovery graphs reject scratch drift before replay.
    original_validate = adapter.validate
    def validate(self):
        if self is not adapter or getattr(self.validate, '__func__', None) is not validate:
            raise ValueError('S validation wrapper changed')
        original_validate()
        plans.validate_private_bindings()
    adapter.validate = types.MethodType(validate, adapter)

def contact_tile_config(contact_capacity, *, contact_tile=32, block_threads=128):
    """Explicit S candidate with capacity-derived grid, never an tuned default."""
    if isinstance(contact_capacity, bool) or not isinstance(contact_capacity, int) or contact_capacity < 0:
        raise ValueError('nonnegative contact capacity required')
    if contact_tile not in (16,32,64,128) or block_threads not in (32,64,128,256):
        raise ValueError('unsupported S tile/block candidate')
    return ExecutionConfig(strategy='multi_cta',block_threads=block_threads,
        contact_tile=contact_tile,workers_per_world=max(1,(contact_capacity+contact_tile-1)//contact_tile))


def install(bundle, build_root, dataflow_profile=None, execution_config=None, qualification_mode=None):
    adapter=bundle.runtime.forward.s
    workspace=adapter.workspace
    # Validate the original seal before changing its explicitly selected entries.
    adapter.validate()
    plans=ConstraintPlans(workspace,build_root,adapter,dataflow_profile=dataflow_profile,execution_config=execution_config,qualification_mode=qualification_mode)
    workspace._cuda_constraint_plans=plans
    # Keep the actual installed launch handles reachable without inspecting
    # method closures. The central manifest/trace walker has a bounded depth.
    adapter._cuda_position_plan=plans.position
    adapter._cuda_reference_plan=plans.reference
    bundle._cuda_s_position_plan=plans.position
    bundle._cuda_s_reference_plan=plans.reference
    workspace.launch_position=types.MethodType(lambda self,stream=None: plans.launch_position(stream),workspace)
    workspace.launch_reference=types.MethodType(lambda self,stream=None: plans.launch_reference(stream),workspace)
    adapter._methods=(workspace.launch_position.__func__,workspace.launch_reference.__func__)
    def adapter_position(self):
        self.validate()
        self.workspace.launch_position(stream=self.stream)
    adapter.launch_position=types.MethodType(adapter_position,adapter)
    _install_private_validation(adapter, plans)
    adapter.validate()
    forward=bundle.runtime.forward
    metadata=newton_head_metadata(forward)
    projection=projection_plan(forward,build_root)
    forward._cuda_s_projection=projection
    forward._fusion_project_newton=types.MethodType(lambda self: projection.launch(self.stream),forward)
    return {'position':plans.position.manifest,'reference':plans.reference.manifest,
            'projection':projection.manifest,
            'kernels':len(plans.position.plans)+2 if plans.mapping == 'contact_tiles' else 3,'family':'dense elliptic friction/hinge-slide limits/contact',
            'arithmetic':'handwritten','fma':True,
            'dataflow_profile':plans.profile,'position_mapping':plans.mapping,
            'feature_selection':plans.feature_selection,'head_metadata':metadata.manifest(),
            'projection_admission_interface':'projection_stages(forward)'}

def projection_stages(forward):
    """Prepared S->N admission stages for a combined Newton entry.

    Consumers may prepend these stages (in this exact order) to the Newton
    entry using constraints_projection.cuh. This does not copy numeric S
    arrays, skip validation, stamp an unperformed solve, or change ownership.
    """
    n,x,sw=forward.n.ws,forward._s_views,forward.s.workspace
    stages=[('cf_s::n_mask',[forward.d.active,forward.d.first_error,forward.d.valid['smooth'],
       sw.reference_ready,sw.status,forward.n_admitted,forward.n_upstream],'lane0'),
      ('cf_s::project_counts',[forward.n_admitted,forward._counts,forward.transport.ncon,
       n.nefc,n.ne,n.nf,n.ncon,n.count,n.empty,n.newton_world,n.active,
       forward.projection_status,forward.projection_item,n.rows,n.contacts,forward.runtime_shape.nv],'lane0'),
      ('cf_s::project_cone_heads_verified',[forward.n_admitted,n.nefc,n.ne,n.nf,n.ncon,x['type'],x['id'],
       x['dim'],x['mu'],x['friction'],n.cone_dim,n.cone_mu,n.cone_friction,
       forward.projection_status,forward.projection_item,n.rows,n.contacts,
       sw._cuda_constraints_storage['compact_heads'],sw._cuda_constraints_storage['compact_count'],sw._cuda_constraints_storage['compact_valid'],
       sw._cuda_constraints_storage['head_dim'],sw._cuda_constraints_storage['head_mu'],
       sw._cuda_constraints_storage['head_friction'],sw._cuda_constraints_storage['head_cone_valid']],'block'),
      ('cf_s::increment_plan',[forward.mask,forward.d.first_error,forward.expected_newton],'lane0')]
    return stages

def projection_plan(forward,build_root):
    """Standalone compatible entry; a combined N entry uses projection_stages."""
    plan=NativePlan('constraints_projection',[(GPU_ROOT / 'cuda/constraints/constraints_projection.cuh')],
        projection_stages(forward),forward.s.workspace.nworld,build_root,stage_barriers=False)
    owner=forward.s.workspace._cuda_constraint_plans
    original_launch=plan.launch
    def launch(self,stream=None):
        owner.validate_private_bindings()
        original_launch(stream)
    plan.launch=types.MethodType(launch,plan)
    return plan

def _head_metadata_arrays(workspace):
    storage = workspace._cuda_constraints_storage
    return dict(source_counts=workspace.counts, source_ncon=workspace.ncon,
                compact_heads=storage['compact_heads'], compact_count=storage['compact_count'],
                compact_valid=storage['compact_valid'], head_type=storage['head_type'],
                head_id=storage['head_id'], head_dim=storage['head_dim'],
                head_mu=storage['head_mu'], head_friction=storage['head_friction'],
                valid=storage['head_cone_valid'])


def newton_head_metadata(forward):
    """Return sealed original head owners independently of prefix migration.

    This cold descriptor does not read values or authenticate current numeric
    data. N retains its admission, current validation and per-launch fallback.
    """
    from libero.gpu.host.runtime.newton.head_metadata import HeadMetadata
    existing = getattr(forward, '_cuda_s_head_metadata', None)
    if existing is not None:
        if type(existing) is not HeadMetadata or existing.owner is not forward:
            raise ValueError('S head metadata descriptor owner changed')
        existing.validate()
        return existing
    adapter = forward.s
    workspace = adapter.workspace
    plans = workspace._cuda_constraint_plans
    dimensions = (workspace.nworld, workspace.rows, workspace.contacts, int(forward.runtime_shape.nv))
    def validate_owner():
        if forward.s is not adapter or adapter.workspace is not workspace or workspace._cuda_constraint_plans is not plans:
            raise ValueError('S head metadata producer root changed')
        if (workspace.nworld, workspace.rows, workspace.contacts, int(forward.runtime_shape.nv)) != dimensions:
            raise ValueError('S head metadata producer dimensions changed')
        forward.s.validate()
        plans.validate_private_bindings()
    result = HeadMetadata.create(owner=forward, nworld=dimensions[0], rows=dimensions[1],
        contacts=dimensions[2], nv=dimensions[3], provider=lambda: _head_metadata_arrays(workspace),
        validate_owner=validate_owner, array_type=wp.array, int32_dtype=wp.int32, float64_dtype=scalar.wp_real)
    forward._cuda_s_head_metadata = result
    return result


def projection_prefix(forward):
    """Cold typed prefix schema consumed by a combined handwritten N entry.

    All array strides are explicit. The four per-world stage arguments use
    their original flattened owner; no array view is created here.
    """
    names = (
        ('active','first','smooth','reference','sstatus','admitted','error'),
        ('admitted','source_counts','source_ncon','nefc','ne','nf','ncon','count',
         'empty','newton','active','status','item','rows','contacts','nv'),
        ('admitted','nefc','ne','nf','ncon','typ','id','dim','mu','friction',
         'out_dim','out_mu','out_friction','status','item','rows','contacts',
         'compact_heads','compact_count','compact_valid','head_dim','head_mu','head_friction','head_cone_valid'),
        ('mask','first','epoch'),
    )
    stages = []
    for (function, values, mode), parameters in zip(projection_stages(forward), names):
        if len(parameters) != len(values): raise ValueError('S projection prefix ABI')
        stages.append(dict(function=function, arguments=list(zip(parameters,values)),
            collective=mode == 'block', perworld=1,
            world_strides={name:(value.size // forward.capacity)
                for name,value in zip(parameters,values) if isinstance(value,wp.array)}))
    metadata = newton_head_metadata(forward)
    return dict(headers=[(GPU_ROOT / 'cuda/constraints/constraints_projection.cuh')],stages=stages,
                validate_private_bindings=metadata.validate,
                world_index_argument=True, numeric_input_copy=False,
                static_shared_bytes=660, static_shared_scope="constraints_projection_header_upper_bound",
                newton_head_metadata=metadata)
