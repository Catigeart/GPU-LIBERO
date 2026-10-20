"""Handwritten native D velocity, bias and smooth-force CUDA chains.

The retained host launcher performs all admission and binding checks. Its cold
schedule supplies only pointer bindings and stage order; arithmetic is provided
by the handwritten headers, with no generated Warp C++ or mathematical lowering.
"""
import libero.gpu.precision as scalar
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
from pathlib import Path
import types
import hashlib
import numpy as np
import warp as wp

from libero.gpu.host.assembly.pipeline import dynamics as reference
from libero.gpu.host.runtime.dynamics.generic_runtime import GenericRuntimeDynamicsAdapter
from libero.gpu.host.build.native import NativePlan
from libero.gpu.host.runtime.dynamics.selected_local import (prepare_selection, validate_selection, selected_stages, scalar_prefix)

ROOT = (GPU_ROOT / 'cuda/dynamics')
METHODS = ('launch_velocity', 'launch_bias', 'launch_force')
FUNCTIONS = {
    '_gate': 'd_gate', '_finish': 'd_finish',
    '_reference_gate': 'd_reference_gate', '_velocity_status': 'd_velocity_status',
    '_force_gate': 'd_force_gate', '_actuation_finish': 'd_actuation_finish',
    '_copy_admitted': 'd_copy_admitted_block', 'commit_bias': 'd_commit_bias_block',
    'commit_smooth': 'd_commit_smooth_block',
    'scalar_transmission_velocity': 'd_scalar_transmission_velocity',
    '_comvel': 'd_comvel', '_rne_bias': 'd_rne_bias',
    'joint_passive_count': 'd_joint_passive_count', 'fluid_count': 'd_fluid_count',
    'actuation_count': 'd_actuation_count', '_three_rn': 'd_three_rn',
    '_project': 'd_project', '_seed_rhs': 'd_seed_rhs',
    '_factor_solve': 'd_factor_solve', '_route': 'd_route',
}

COLLECTIVE_KEYS = {'actuation_count', '_comvel', '_rne_bias',
                   'joint_passive_count', 'fluid_count', '_three_rn', '_project',
                   '_copy_admitted', 'commit_bias', 'commit_smooth'}


def _branch_source_hash(adapter):
    source=adapter.passive.static
    digest=hashlib.sha256()
    for key in ('body_parentid','body_dofadr','body_dofnum','dof_parentid',
                'jnt_type','jnt_qposadr','jnt_dofadr'):
        value=np.asarray(source[key])
        digest.update(key.encode());digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    digest.update(str((adapter.nq,adapter.nv,adapter.nbody)).encode())
    return digest.hexdigest()


def _prepare_branch_topology(adapter):
    # This entry accepts only the source-qualified immutable prepared adapter.
    # Bare mutable workspaces continue to use the original scalar/dense helpers.
    from libero.gpu.model.topology import (prepare_body_topology,
        prepare_ancestor_topology,prepare_joint_topology)
    source=adapter.passive.static
    result=types.SimpleNamespace(body=None,ancestor=None,joint=None,tokens={})
    try:
        result.body=prepare_body_topology(source['body_parentid'],adapter.device)
        if result.body.proof['nbody']!=adapter.nbody:result.body=None
    except ValueError:
        pass
    try:
        mask=np.zeros((adapter.nbody,adapter.nv),np.int32)
        for body in range(1,adapter.nbody):
            owner=body;seen=set()
            while owner and source['body_dofnum'][owner]==0:
                if owner in seen:raise ValueError('ancestor owner cycle')
                seen.add(owner);owner=int(source['body_parentid'][owner])
            if not owner:continue
            dof=int(source['body_dofadr'][owner]+source['body_dofnum'][owner]-1)
            seen=set()
            while dof>=0:
                if dof>=adapter.nv or dof in seen:raise ValueError('ancestor dof domain')
                seen.add(dof);mask[body,dof]=1;dof=int(source['dof_parentid'][dof])
        result.ancestor=prepare_ancestor_topology(mask,adapter.device)
    except (ValueError,IndexError):
        pass
    try:
        result.joint=prepare_joint_topology(source['jnt_type'],source['jnt_qposadr'],
            source['jnt_dofadr'],adapter.device,nq=adapter.nq,nv=adapter.nv)
    except ValueError:
        pass
    for kind in ('body','ancestor','joint'):
        metadata=getattr(result,kind)
        if metadata is not None:
            for key,value in metadata.arrays.items():
                result.tokens[(kind,key)]=(value.ptr,value.shape,value.dtype,value.device)
    result.source_hash=_branch_source_hash(adapter)
    adapter.cuda_branch_topology=result


def _branch_arguments(adapter,key):
    metadata=adapter.cuda_branch_topology
    if key in ('_comvel','_rne_bias') and metadata.body is not None:
        arrays=metadata.body.arrays
        args=[arrays['level_offsets'],arrays['level_bodies'],metadata.body.proof['nlevels']]
        if key=='_rne_bias':args += [arrays['children_offsets'],arrays['children_indices']]
        return args
    if key in ('_project','fluid_count') and metadata.ancestor is not None:
        arrays=metadata.ancestor.arrays
        return [arrays['body_dof_offsets'],arrays['body_dof_indices']]
    if key=='joint_passive_count' and metadata.joint is not None:
        arrays=metadata.joint.arrays
        return [int(metadata.joint.proof['joint_dof_disjoint'] and
                    metadata.joint.proof['joint_qpos_disjoint']),
                arrays['joint_type_offsets'],arrays['joint_type_indices']]
    return []


def _validate_branch_topology(adapter):
    metadata=adapter.cuda_branch_topology
    if metadata.source_hash!=_branch_source_hash(adapter):
        raise ValueError('prepared immutable dynamics topology changed; cold rebuild required')
    for (kind,key),token in metadata.tokens.items():
        value=getattr(metadata,kind).arrays[key]
        if (value.ptr,value.shape,value.dtype,value.device)!=token:
            raise ValueError('dynamics topology metadata owner drift')
    # This validates host content and device owners. It does not detect direct
    # in-place GPU writes to immutable topology; those are outside this API.


def _smooth_arguments(adapter):
    a=adapter.acceleration;s=a.solver;f=adapter.position.factor
    return [a.passive,a.bias,a.applied,a.actuator,a.active,a.entry_status,a.smooth,a.seed,
        adapter.nv,a.xfrc,a.xipos,a.subtree_com,a.cdof,a.body_ids,a.body_count,a.rootid,
        a.ancestor,a.scratch,a.stages,a.qfrc,a.d02_status,adapter.nbody,a.qm,s.count,
        s.dof_Madr,s.dof_parentid,s.ld,s.inv,s.sqrt_inv,s.solution,s.pivot,s.status,
        s.warning_count,s.warning_last,s.nM,s.rhs,a.qacc_smooth,a.status,
        f.ld,f.inv,f.sqrt_inv,f.pivot,f.status,f.warning_count,f.warning_last,f.active,
        adapter.valid['position'],adapter.written['position'],adapter.forward_epoch,
        adapter.cuda_factor_reuse.count]+_branch_arguments(adapter,'_project')


def _cache_arguments(adapter):
    c=adapter.cuda_velocity_bias_cache
    return [c.acc,c.body_force,c.status,c.valid,c.state_written,c.forward_written,
            adapter.state_epoch,adapter.forward_epoch]


class _NativeContext(reference._Context):
    def __init__(self, adapter, name, build_root, dataflow_profile='compatible'):
        super().__init__(adapter.capacity, cold=True, label='CUDA_D_'+name)
        self.adapter, self.build_root = adapter, Path(build_root)/name
        self.dataflow_profile=dataflow_profile
        self.original_plans = []
        self.forward_launch_router = None

    def method(self,owner,name,*args,**kwargs):
        if self.dataflow_profile=='fast' and name in ('snapshot','restore_unrequested'):
            # Uniform requested-mask guards precede every device write. No
            # inactive-zero work needs to be undone for an unrequested world.
            self.flush();return None
        # The separately installed forward router replaces adapter._launch for
        # begin-forward dispatch. Inside a D phase, use the already reviewed
        # world-launch ABI to record our stage rather than cloning that later
        # router (whose closure and semantics belong to another component).
        if owner is self.adapter and name == '_launch':
            current=getattr(getattr(owner,name),'__func__',None)
            if current not in (GenericRuntimeDynamicsAdapter._launch,self.forward_launch_router):
                raise ValueError('native dynamics qualified launcher replaced')
            return self.clone(GenericRuntimeDynamicsAdapter._launch)(owner,*args,**kwargs)
        return super().method(owner,name,*args,**kwargs)

    def record(self, kernel, dim, arguments, kwargs):
        if dim != self.capacity:
            raise ValueError('native dynamics requires scalar world launch schedule: '+kernel.key)
        if kernel.key not in FUNCTIONS:
            raise ValueError('unimplemented handwritten dynamics stage: '+kernel.key)
        self.pending.append((kernel, arguments, kwargs))

    def _stages(self):
        if self.dataflow_profile!='fast':
            return [(FUNCTIONS[k.key],list(args)+_branch_arguments(self.adapter,k.key),'block' if k.key in COLLECTIVE_KEYS else 'lane0')
                    for k,args,_ in self.pending]
        stages=[];i=0;a=self.adapter
        smooth_keys=('_three_rn','_project','_seed_rhs','_factor_solve','_route')
        while i<len(self.pending):
            kernel,args,_=self.pending[i];key=kernel.key
            if key=='_three_rn':
                actual=tuple(k.key for k,_,_ in self.pending[i:i+5])
                if actual!=smooth_keys:raise ValueError('smooth dataflow schedule drift')
                stages.append(('d_smooth_chain',_smooth_arguments(a),'block'));i+=5;continue
            if key=='_comvel':
                args=list(args)+[a.position.cinert,a.rne_model.gravity,a.rne_model.gravity_disabled]+_cache_arguments(a)
                fn='d_comvel_bias_cached'
            elif key=='_rne_bias':
                args=list(args)+_cache_arguments(a);fn='d_rne_bias_cached'
            elif key=='commit_smooth':fn='d_commit_smooth_shared'
            else:fn=FUNCTIONS[key]
            args=list(args)+_branch_arguments(a,key)
            stages.append((fn,args,'block' if key in COLLECTIVE_KEYS and fn!='d_commit_smooth_shared' else 'lane0'))
            i+=1
        return stages

    def flush(self):
        if not self.pending:
            return
        sequence = tuple(k for k, _, _ in self.pending)
        stages=self._stages()
        selection=getattr(self.adapter,"cuda_selected_local",None)
        if selection is not None:stages=selected_stages(self,stages)
        options = self.pending[0][2]
        if any(opts.get('stream') is not options.get('stream') for _, _, opts in self.pending):
            raise ValueError('native dynamics stream boundary drift')
        if self.cold:
            uses_blas = any(k.key == 'actuation_count' for k in sequence)
            headers = [(GPU_ROOT / 'cuda/common/packed_mass.cuh'),(GPU_ROOT / 'cuda/dynamics/dynamics.cuh'), (GPU_ROOT / 'cuda/dynamics/dynamics_passive.cuh'), (GPU_ROOT / 'cuda/dynamics/dynamics_smooth.cuh')]
            # BLAS device code is present only in groups which call actuation.
            if not uses_blas:
                # The scalar passive implementation is independently usable;
                # its collective actuation section is guarded by this define.
                defines = {'D_NO_ACTUATION_DX': 1}
            else:
                defines = {}
            defines.update(D_NV=self.adapter.nv,D_NU=self.adapter.actuation.nu,
                           D_NM=self.adapter.acceleration.solver.nM,
                           D_THREADS=64,D_SM=self.adapter.device.arch*10)
            fast=self.dataflow_profile=='fast'
            if fast:defines['D_FAST_DATAFLOW']=1
            bridges=[(GPU_ROOT / 'cuda/dynamics/dynamics_blas.cu')] if uses_blas else ()
            if selection is not None:
                if selection.fluid_parallel_payload:headers.append(GPU_ROOT/'cuda/dynamics/dynamics_selected.cuh')
                if selection.scalar_support:
                    prefix=scalar_prefix(selection.support)
                    headers.insert(0,prefix)
                    # Match the actual tested same-job prefix in every CU.
                    bridges=tuple(prefix+path.read_text() for path in bridges)
                if selection.fused_rne_finite:defines['D_FUSED_RNE_FINITE']=1
            plan = NativePlan(self.label+'_group%d'%len(self.plans),headers,stages,
                self.capacity,self.build_root,block_size=64,defines=defines,
                bridges=bridges,mathdx=uses_blas,
                device=self.adapter.device,world_guard=self.adapter.requested_mask if fast else None)
            self.plans.append((sequence,plan))
        else:
            if self.cursor >= len(self.plans):
                raise ValueError('native dynamics schedule grew')
            expected, plan = self.plans[self.cursor]
            if sequence != expected:
                raise ValueError('native dynamics schedule drift')
            plan.set_arguments(stages,stream=options.get('stream',self.adapter.stream))
            plan.launch(stream=options.get('stream',self.adapter.stream))
            self.cursor += 1
        self.pending.clear()


def _wrapper(name):
    original = getattr(GenericRuntimeDynamicsAdapter,name)
    def native(self,*args,**kwargs):
        self.validate(kwargs.get('stream'))
        _validate_branch_topology(self)
        if getattr(self,"cuda_selected_local",None) is not None:validate_selection(self)
        if getattr(self,'cuda_dynamics_profile','compatible')=='fast':
            if self.cuda_velocity_bias_cache.layout!=('component_body',1,self.nbody):
                raise ValueError('dataflow D spatial cache layout drift')
            for namespace in (self.cuda_velocity_bias_cache,self.cuda_factor_reuse):
                actual={key:(value.ptr,value.shape,value.dtype,value.device)
                        for key,value in vars(namespace).items() if isinstance(value,wp.array)}
                if actual!=namespace.tokens:raise ValueError('dataflow D cache owner drift')
        # Shifted overlap can create cross-world dependencies. The explicitly
        # retained reference path handles that case rather than silently
        # changing its execution ordering inside one block/world.
        if reference._shifted_external_alias(self,(*args,*kwargs.values())):
            return original(self,*args,**kwargs)
        context = self._cuda_dynamics_contexts[name]
        if getattr(context,'running',False):
            raise ValueError('native dynamics reentrant dispatch')
        context.running=True;context.cursor=0
        try:
            result=context.function(original,self,*args,**kwargs)
            context.finish()
            return result
        finally:
            context.pending.clear();context.running=False
    native.__name__=name
    return native


def install(bundle,build_root,*,dataflow_profile='compatible',selected_local_domains=()):
    adapter = getattr(bundle,'dynamics',bundle)
    if not isinstance(adapter,GenericRuntimeDynamicsAdapter):
        raise TypeError('native-count dynamics adapter required')
    if dataflow_profile=='compatibility':dataflow_profile='compatible'
    if dataflow_profile not in ('compatible','fast'):
        raise ValueError('unknown dynamics dataflow profile')
    adapter.cuda_dynamics_profile=dataflow_profile
    _prepare_branch_topology(adapter)
    prepare_selection(adapter,selected_local_domains,wp,dataflow_profile)
    if dataflow_profile=='fast':
        # These private owners do not enter the frozen original public/atomic
        # field set. Their bindings have an independent checked token inventory.
        c=types.SimpleNamespace(layout=('component_body',1,adapter.nbody))
        for key,width,dtype in (('acc',adapter.nbody*6,scalar.wp_real),
            ('body_force',adapter.nbody*6,scalar.wp_real),('status',None,wp.int32),
            ('valid',None,wp.int32),('state_written',None,wp.int64),
            ('forward_written',None,wp.int64)):
            setattr(c,key,wp.zeros((adapter.capacity,width) if width else adapter.capacity,
                                  dtype=dtype,device=adapter.device))
        r=types.SimpleNamespace(count=wp.zeros(adapter.capacity,dtype=wp.int64,device=adapter.device))
        for namespace in (c,r):
            namespace.tokens={key:(value.ptr,value.shape,value.dtype,value.device)
                for key,value in vars(namespace).items() if isinstance(value,wp.array)}
        adapter.cuda_velocity_bias_cache=c;adapter.cuda_factor_reuse=r
        if adapter.acceleration.qm.ptr!=adapter.position.factor.qm.ptr:
            raise ValueError('F/D factor reuse requires the same prepared qM owner')
    calls = {
        'launch_velocity': {},
        'launch_bias': dict(reference_done=adapter.gate_bias,reference_error=adapter.gate_bias,
                           reference_written=adapter.forward_epoch,reference_expected=adapter.forward_epoch),
        'launch_force': dict(ctrl=adapter.actuation.ctrl,ctrl_valid=adapter.actuation_active),
    }
    contexts={}
    for name in METHODS:
        context=_NativeContext(adapter,name,build_root,dataflow_profile)
        context.function(getattr(GenericRuntimeDynamicsAdapter,name),adapter,**calls[name])
        context.finish();context.cold=False;contexts[name]=context
    adapter._cuda_dynamics_contexts=contexts
    adapter._d_fusion_ranges=reference._array_ranges(adapter)
    for name in METHODS:
        setattr(adapter,name,types.MethodType(_wrapper(name),adapter))
    adapter.cuda_dynamics_manifest = {
        name:[plan.manifest for _,plan in context.plans] for name,context in contexts.items()}
    return dict(installed=True,plans=adapter.cuda_dynamics_manifest,
                scope='velocity, RNE bias, passive, actuation and smooth acceleration',
                profile=dataflow_profile,
                spatial_cache_layout=('component_body_v1' if dataflow_profile=='fast' else None),
                dataflow=('shared actuator/smooth single RHS; same-forward F factor; velocity-bias cache; requested no-write'
                          if dataflow_profile=='fast' else 'compatible ordered stages'),
                unresolved=[],shifted_alias_fallback='original ordered launch path')
