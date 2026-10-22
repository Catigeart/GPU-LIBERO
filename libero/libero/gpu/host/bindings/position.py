"""Handwritten native-count CUDA position chain, one block per world.

Python only binds stable arrays and retains prepared admission/history boundaries.
The body recurrence and mass tree retain their native topological order; frame
objects run cooperatively between block barriers. No generated arithmetic is used.
"""
import libero.gpu.precision as scalar
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
from pathlib import Path
from types import MethodType
from libero.gpu.host.build.native import NativePlan

_ROOT = (GPU_ROOT / 'cuda/position')
_MODEL = 'qpos0 parent jnum jadr mocapid bpos bquat jtype qadr jaxis jpos'.split()
_DATA = 'active qpos mocap_pos mocap_quat qtmp mqtmp xpos xquat xmat xanchor xaxis world_xipos world_ximat status'.split()
_DIAG = 'norm_before mocap_before pre_pos pre_quat joint_pos joint_quat body_norm_before body_route joint_route body_mat_identity joint_rot_identity hinge_zero offcenter_rot_identity trig_arg trig_sin trig_cos norm_branch mocap_branch body_norm_branch'.split()


def _stage(name, args, mode='lane0'):
    if name == 'check_frames': mode = 'block'
    return ('libero_position::'+name, list(args), mode)



def _topology(p):
    """Prepare immutable integer schedules once, from admitted model owners."""
    cached=getattr(p,'_branch_topology',None)
    if cached is not None:return cached
    import numpy as np
    from libero.gpu.model.topology import (prepare_body_topology,prepare_joint_topology,
        prepare_packed_topology,prepare_frame_topology)
    def cold(a):return a.numpy() if hasattr(a,'numpy') else np.asarray(a)
    b,f,c,r=p.body,p.frames,p.compos,p.crb_model
    model=b.static;device=b.data.active.device
    try:
        if (model.nb,model.nj)!=(c.nbody,c.njnt) or model.nb!=r.nbody:
            raise ValueError('position phase dimensions disagree')
        parent=cold(model.parent).reshape(-1)
        if not np.array_equal(parent,cold(c.parent).reshape(-1)) or not np.array_equal(parent,cold(r.body_parent).reshape(-1)):
            raise ValueError('position phase parent descriptors disagree')
        if not np.array_equal(cold(model.jtype).reshape(-1),cold(c.jtype).reshape(-1)):
            raise ValueError('position phase joint types disagree')
        jnum,jadr=cold(model.jnum).reshape(-1),cold(model.jadr).reshape(-1)
        joints=[j for body in range(1,model.nb) for j in range(int(jadr[body]),int(jadr[body]+jnum[body]))]
        if sorted(joints)!=list(range(model.nj)):
            raise ValueError('body joint ranges must cover every joint exactly once')
        body=prepare_body_topology(cold(model.parent),device)
        joint=prepare_joint_topology(cold(model.jtype),cold(model.qadr),cold(c.jdof),device,nq=model.nq,nv=c.nv)
        if not joint.proof['joint_qpos_disjoint'] or not joint.proof['joint_dof_disjoint']:
            raise ValueError('joint state/output ranges overlap')
        packed=prepare_packed_topology(cold(r.dof_parent),cold(r.dof_madr),device,
            simple=np.asarray(cold(r.dof_simple)!=0,dtype=np.int32))
        if len(cold(r.dof_madr)) and int(cold(r.dof_madr).reshape(-1)[-1])+len(packed.host.arrays['dof_chain_indices'])-int(packed.host.arrays['dof_chain_offsets'][-2])>r.nM:
            raise ValueError('last packed ancestor row exceeds nM')
        cached=dict(body=body,joint=joint,packed=packed,
            frames={kind:prepare_frame_topology(cold(f.static[kind+'_sameframe']),device)
                    for kind in ('body','geom','site')},qualified=True)
    except ValueError as error:
        cached=dict(qualified=False,reason=str(error))
    p._branch_topology=cached
    return cached


def _body_schedule(topology):
    body=topology['body'];a=body.arrays
    return [a['level_offsets'],a['level_bodies'],int(body.proof['nlevels'])]


def _frame_schedule(topology,kind):
    a=topology['frames'][kind].arrays
    return [a['frame_offsets'],a['frame_indices']]

def body_stages(p):
    b, f = p.body, p.frames
    topology=_topology(p);typed=topology['qualified']
    m, d, g = b.static, b.data, b.diag
    dims = [m.nq, m.nb, m.nj, m.nm]
    diag = [getattr(g, k) for k in _DIAG]
    out = [_stage('body_block' if typed else 'body', [*[getattr(m,k) for k in _MODEL], *dims,
                          *[getattr(d,k) for k in _DATA], *diag,*(_body_schedule(topology) if typed else [])],
                         'block' if typed else 'lane0')]
    out.append(_stage('commit', [*dims, d.active,d.status,d.qpos,d.mocap_quat,d.qtmp,d.mqtmp,
        d.xpos,d.xquat,d.xmat,d.xanchor,d.xaxis,d.world_xipos,d.world_ximat,*diag]))
    out.append(_stage('after_body',[p.active,b.status,p.frame_active]))
    out.append(_stage('frame_status',[f.active_world,f.status]))
    out.append(_stage('frames_body_typed' if typed else 'frames_body',[f.active_world,f.body_xpos,f.body_xquat,f.body_xmat,
        f.static['body_ipos'],f.static['body_iquat'],f.static['body_sameframe'],
        f.xipos,f.ximat,f.body_tmpquat,f.nbody,*(_frame_schedule(topology,'body') if typed else [])],'block'))
    for kind, count in [('geom',f.ngeom),('site',f.nsite)]:
        out.append(_stage('frames_objects_typed' if typed else 'frames_objects',[f.active_world,f.body_xpos,f.body_xquat,f.body_xmat,
            f.xipos,f.ximat,f.static[kind+'_bodyid'],f.static[kind+'_pos'],
            f.static[kind+'_quat'],f.static[kind+'_sameframe'],getattr(f,kind+'_xpos'),
            getattr(f,kind+'_xmat'),getattr(f,kind+'_tmpquat'),f.nbody,count,*(_frame_schedule(topology,kind) if typed else [])],'block'))
    out.append(_stage('check_frames',[p.frame_active,f.status,f.xipos,f.ximat,
        f.geom_xpos,f.geom_xmat,f.site_xpos,f.site_xmat,f.body_tmpquat,
        f.geom_tmpquat,f.site_tmpquat,f.nbody,f.ngeom,f.nsite]))
    out.append(_stage('after_frames',[p.frame_active,f.status,p.mass_active]))
    return out


def position_stages(p):
    from .position_mass import stages as mass_stages
    out=body_stages(p)+mass_stages(p)
    out.append(_stage('position_finish',[p.active,p.body.status,p.frames.status,p.mass_active,
        p.mass_first_stage,p.mass_first_code,p.mass_result_kind,p.first_stage,p.first_code,p.result_kind]))
    return out


def _finite_vector_defines(d):
    """Cold static counts select only the measured 342-element finite scan.

    Each scan keeps its runtime extent argument and original owner guards.
    Fast dataflow may use 128 threads for larger frame sets; standalone plans
    retain the 64-thread default. Unconfigured headers keep scalar scans.
    """
    p=d.position
    return {'CF_PM_STATIC_NBODY':int(p.body.static.nb),
            'CF_PM_STATIC_NJNT':int(p.body.static.nj),
            'CF_PM_STATIC_NV':int(p.factor.nv),
            'CF_PM_STATIC_NM':int(p.factor.nM)}


def create_plan(d, build_root):
    p,t=d.position,d.transmission
    stages=[_stage('dynamics_gate',[d.active,d.first_error,d.state_epoch,d.source_state_epoch,
        d.forward_epoch,d.valid['position'],d.written['position'],d.gate_position])]
    stages+=position_stages(p)
    stages.append(_stage('transmission',[t.qpos,t.position_active,t.qadr,t.dadr,t.gear,
        t.length,t.moment,t.position_status,t.nq,t.nv,t.nu]))
    stages.append(_stage('dynamics_status',[p.result_kind,p.first_code,t.position_status,d.status_position]))
    stages.append(_stage('dynamics_finish',[d.gate_position,d.status_position,d.forward_epoch,
        d.valid['position'],d.written['position'],d.first_error]))
    return NativePlan('handwritten_position',[(GPU_ROOT / 'cuda/position/position.cuh'),(GPU_ROOT / 'cuda/position/position_mass.cuh')],
        stages,d.capacity,build_root,device=d.device,defines=_finite_vector_defines(d))


def _create_position_snapshot_plan(d, build_root):
    """Save only the worlds consumed by the unchanged mask-zero restore.

    The retained hold owns the field inventory, aliases and history storage.
    This plan binds those same buffers once; its device predicate reads the
    current requested mask on every replay. F still executes all its original
    clears, diagnostics and selected-failure writes between save and restore.
    """
    import warp as wp
    hold=d._holds['position']
    hold.validate(hold.fields)
    mask=d.requested_mask
    if (mask.dtype!=wp.int32 or mask.shape!=(d.capacity,) or
            mask.device!=d.device or not mask.is_contiguous):
        raise ValueError('position snapshot requires the retained world mask')
    kinds={scalar.wp_real:'f64',wp.int32:'i32',wp.int64:'i64'}
    stages=[]
    for name,source in hold.views.items():
        before=hold.before[name]
        if (source.ndim!=2 or source.shape[0]!=d.capacity or
                source.shape!=before.shape or source.dtype!=before.dtype or
                source.dtype not in kinds):
            raise ValueError('position snapshot field schema: '+name)
        stages.append(('cf_copy::hold_'+kinds[source.dtype],
            [mask,source,before,source.shape[1]],'block'))
    plan=NativePlan('position_masked_snapshot',
        [GPU_ROOT/'cuda/runtime/copies.cuh'],stages,d.capacity,build_root,
        device=d.device,stage_barriers=False)
    plan.manifest.update(snapshot_condition='requested_mask==0',
        snapshot_fields=list(hold.views),
        history='original hold buffers; requested worlds leave history untouched')
    return hold,plan


def install(bundle,build_root):
    """Install real dynamics entry; Foundation forwards through this bound method."""
    d=bundle.dynamics
    plan=create_plan(d,build_root)
    position_hold,snapshot=_create_position_snapshot_plan(d,Path(build_root)/'snapshot')
    def launch_position(self,stream=None):
        self.validate(stream)
        self.position.validate_bindings()
        self.transmission.validate_bindings()
        if self._holds['position'] is not position_hold:
            raise ValueError('position snapshot hold owner changed')
        position_hold.validate(position_hold.fields)
        snapshot.launch(stream=self.stream)
        plan.launch(stream=self.stream)
        self._holds['position'].restore_unrequested(self.requested_mask,self.stream)
    d.launch_position=MethodType(launch_position,d)
    d._cuda_position_plan=plan
    d._cuda_position_snapshot_plan=snapshot
    from .position_scene import install_scene_from_bundle
    scene=install_scene_from_bundle(bundle,Path(build_root)/'scene')
    manifest=dict(plan.manifest,stages=len(position_stages(d.position))+4,
        execution='one block/world; ordered tree; cooperative body/object frame grids',
        history='mask0 snapshot and original restore_unrequested boundaries',
        history_snapshot=snapshot.manifest,
        numerics='handwritten FP64; FMA; native full-range sin/cos',
        validation='10 bounded component cases passed; true-forward comparison recorded by integrated factory',
        observer_scene=scene)
    return manifest



def _dataflow_stages(d):
    """Retained ABI only: arithmetic and topology remain handwritten CUDA."""
    p,t=d.position,d.transmission
    stages=[_stage('dynamics_gate',[d.active,d.first_error,d.state_epoch,d.source_state_epoch,
        d.forward_epoch,d.valid['position'],d.written['position'],d.gate_position])]
    stages+=position_stages(p)
    stages.append(_stage('transmission',[t.qpos,t.position_active,t.qadr,t.dadr,t.gear,
        t.length,t.moment,t.position_status,t.nq,t.nv,t.nu]))
    stages.append(_stage('dynamics_status',[p.result_kind,p.first_code,t.position_status,d.status_position]))
    stages.append(_stage('dynamics_finish',[d.gate_position,d.status_position,d.forward_epoch,
        d.valid['position'],d.written['position'],d.first_error]))
    return stages


def _dataflow_source(d,*,bounds_cache=None,subtree_mass=False):
    """Bind world slices into one handwritten block dataflow orchestrator.

    The generator performs no mathematical rewriting. It substitutes shared
    storage for phase-local pointers and emits explicitly chosen phase barriers.
    Stable external arrays still receive every legacy value, including traces.
    """
    import warp as wp
    p=d.position;b=p.body;f=p.frames;m=p.factor
    nq,nb,nj,nm=b.static.nq,b.static.nb,b.static.nj,b.static.nm
    nv,nM=m.nv,m.nM
    intervals=None
    if subtree_mass and _topology(p)['qualified']:
        from libero.gpu.model.topology import contiguous_subtree_sizes
        intervals=contiguous_subtree_sizes(p.compos.parent.numpy())
    geom_shared=3*f.ngeom if bounds_cache else 0
    fixed=21*nb+6*nj+geom_shared  # body matrix, joint frames, inertial + geom positions
    body_phase=7*nb+nq+4*nm+12
    mass_phase=23*nb+12*nv+2*nM
    shared_doubles=fixed+max(body_phase,mass_phase)
    originals=_dataflow_stages(d)
    if bounds_cache:
        originals.append((bounds_cache['function'],bounds_cache['args'],'lane0'))
    values=[];specs=[];seen={};rows={};objects=0
    ctypes={scalar.wp_real:'real_t*',wp.float32:'float*',wp.int32:'int32_t*',wp.int64:'int64_t*'}
    for name,args,mode in originals:
        key=name.split('::')[-1]
        if key in ('frames_objects','frames_objects_typed'):key=('geom','site')[objects];objects+=1
        if key=='body_block':key='body'
        if key=='frames_body_typed':key='frames_body'
        names=[]
        for value in args:
            token=('array',id(value)) if isinstance(value,wp.array) else ('scalar',type(value),value)
            if token not in seen:
                seen[token]=len(values);values.append(value)
                specs.append(ctypes[value.dtype] if isinstance(value,wp.array) else 'int32_t')
            names.append('a%d'%seen[token])
        rows[key]=(name,args,names)
    def shifted(key,index):
        value=rows[key][1][index];name=rows[key][2][index]
        if not isinstance(value,wp.array) or value.size%d.capacity:
            raise ValueError('world slice expected: %s[%d]'%(key,index))
        return '%s + w*%d'%(name,value.size//d.capacity)
    def call(key,world='w',override=None,function=None):
        name,_,names=rows[key];args=list(names)
        for index,value in (override or {}).items():args[index]=value
        if key in ('compos','crb'):
            # Fill optional static schedules for unqualified topology before final flag.
            count=7 if key=='compos' else 10
            supplied=len(args)-(21 if key=='compos' else 16)
            defaults=['nullptr']*count
            defaults[-3]='0'
            args.extend(defaults[supplied:])
            args.append('true')
            if intervals is not None:
                args.append('libero_position::mass_subtree_sizes')
        if key=='cf_bounds_prepare_cached_block':args.append('true')
        return '%s(%s, %s);'%(function or name,world,', '.join(args))
    def lane(statement):return '    if(threadIdx.x==0) { '+statement+' }'
    def publish(key,index,source):
        value=rows[key][1][index]
        if key=='compos' and index in (17,18,19) or key=='crb' and index==13:
            count,width=(nv,6) if key=='compos' and index==19 else (nb,3 if key=='compos' and index==17 else 10)
            return '    cf_pm::publish_planes(%s,%s,%d,%d);'%(shifted(key,index),source,count,width)
        return '    libero_position::copy_block(%s,%s,%d);'%(shifted(key,index),source,value.size//d.capacity)
    # This shared prefix lives through frames -> compos; body quaternions and
    # normalization scratch expire before the packed mass phase reuses memory.
    code=['extern __shared__ real_t position_scratch[];']
    offset=0
    for name,count in [('s_xmat',9*nb),('s_anchor',3*nj),('s_axis',3*nj),
                       ('s_xipos',3*nb),('s_ximat',9*nb)]:
        code.append('real_t* %s=position_scratch+%d;'%(name,offset));offset+=count
    if bounds_cache:
        code.append('real_t* s_geom_positions=position_scratch+%d;'%offset);offset+=geom_shared
    code.append('real_t* phase=position_scratch+%d;'%offset)
    offset=0
    for name,count in [('s_xpos',3*nb),('s_xquat',4*nb),('s_qtmp',nq),
                       ('s_mqtmp',4*nm),('s_world_xipos',3),('s_world_ximat',9)]:
        code.append('real_t* %s=phase+%d;'%(name,offset));offset+=count
    body_base_count=len(_MODEL)+4+len(_DATA)+len(_DIAG)
    body={i:shifted('body',i) for i in range(15,body_base_count)}
    body.update({19:'s_qtmp',20:'s_mqtmp',21:'s_xpos',22:'s_xquat',23:'s_xmat',
                 24:'s_anchor',25:'s_axis',26:'s_world_xipos',27:'s_world_ximat'})
    commit={i:shifted('commit',i) for i in range(4,len(rows['commit'][1]))}
    commit.update({8:'s_qtmp',9:'s_mqtmp',10:'s_xpos',11:'s_xquat',12:'s_xmat',
                   13:'s_anchor',14:'s_axis',15:'s_world_xipos',16:'s_world_ximat'})
    body_call=call('body','0',body)
    body_collective=rows['body'][0].endswith('body_block')
    code.extend([lane(call('dynamics_gate')),('    '+body_call if body_collective else lane(body_call)),
                 lane(call('commit','0',commit)),lane(call('after_body')),lane(call('frame_status')),
                 '    __syncthreads(); // normalized body inputs -> cooperative frames'])
    for i,source in [(19,'s_qtmp'),(20,'s_mqtmp'),(21,'s_xpos'),(22,'s_xquat'),
                     (23,'s_xmat'),(24,'s_anchor'),(25,'s_axis'),(26,'s_world_xipos'),(27,'s_world_ximat')]:
        code.append(publish('body',i,source))
    frames={0:shifted('frames_body',0),1:'s_xpos',2:'s_xquat',3:'s_xmat',
            7:'s_xipos',8:'s_ximat',9:shifted('frames_body',9)}
    code.extend(['    '+call('frames_body','0',frames),
                 '    __syncthreads(); // inertial frames -> sameframe=2 objects'])
    for key,geometry in [('geom',1),('site',0)]:
        override={0:shifted(key,0),1:'s_xpos',2:'s_xquat',3:'s_xmat',4:'s_xipos',5:'s_ximat',
                  10:shifted(key,10),11:shifted(key,11),12:shifted(key,12)}
        typed=rows[key][0].endswith('frames_objects_typed')
        _,_,names=rows[key];args=list(names)
        for index,value in override.items():args[index]=value
        extra=[str(geometry),'w',('s_geom_positions' if bounds_cache and geometry else 'nullptr')]
        function='libero_position::frames_objects_dataflow_typed' if typed else 'libero_position::frames_objects_dataflow'
        final=args[:15]+extra+(args[15:] if typed else [])+['true']
        code.append('    '+function+'(0, '+', '.join(final)+');')
    code.extend([publish('frames_body',7,'s_xipos'),publish('frames_body',8,'s_ximat'),
                 '    __syncthreads(); // all geometry/site writes complete; retire body scratch'])
    check={i:shifted('check_frames',i) for i in range(11)}
    check.update({2:'s_xipos',3:'s_ximat'})
    code.extend(['    '+call('check_frames','0',check),
                 '    __syncthreads();',lane(call('after_frames'))])
    offset=0
    for name,count in [('s_com',3*nb),('s_cinert',10*nb),('s_crb',10*nb),('s_cdof',6*nv),
                       ('s_qm',nM),('s_ld',nM),('s_inv',nv),('s_sqrt_inv',nv),('s_solution',4*nv)]:
        code.append('real_t* %s=phase+%d;'%(name,offset));offset+=count
    compos={0:shifted('compos',0),1:'s_xipos',2:'s_ximat',3:'s_anchor',4:'s_xmat',5:'s_axis',
            17:'s_com',18:'s_cinert',19:'s_cdof',20:shifted('compos',20)}
    crb={0:shifted('crb',0),1:'s_cinert',2:'s_cdof',13:'s_crb',14:'s_qm',15:shifted('crb',15)}
    factor={0:'s_qm',1:shifted('factor_solve',1),2:shifted('factor_solve',2),
            3:shifted('factor_solve',3),6:'s_ld',7:'s_inv',8:'s_sqrt_inv',9:'s_solution',
            10:shifted('factor_solve',10),11:shifted('factor_solve',11),
            12:shifted('factor_solve',12),13:shifted('factor_solve',13)}
    code.extend(['    '+call('compos','0',compos),'    __syncthreads();',lane(call('gate1')),
                 '    '+call('crb','0',crb),'    __syncthreads();',lane(call('gate2')),
                 '    '+call('factor_solve','0',factor,'cf_pm::factor_solve_shared'),'    __syncthreads();',
                 lane(call('finish')),lane(call('position_finish')),lane(call('transmission')),
                 lane(call('dynamics_status')),lane(call('dynamics_finish')),
                 '    __syncthreads(); // packed factor -> durable output publication'])
    for key,index,source in [('compos',17,'s_com'),('compos',18,'s_cinert'),('compos',19,'s_cdof'),
                             ('crb',13,'s_crb'),('crb',14,'s_qm'),('factor_solve',6,'s_ld'),
                             ('factor_solve',7,'s_inv'),('factor_solve',8,'s_sqrt_inv'),
                             ('factor_solve',9,'s_solution')]:
        code.append(publish(key,index,source))
    if bounds_cache:
        key=bounds_cache['function'].split('::')[-1]
        code.append('    '+call(key,'w',{bounds_cache['geom_position_arg']:'s_geom_positions'}))
        code.append('    __syncthreads();')
    signature=', '.join('%s a%d'%(kind,index) for index,kind in enumerate(specs))
    # Only immutable model integers are embedded. The CUDA mathematical bodies
    # remain in position_mass.cuh; no arithmetic is generated here.
    constants=('__device__ __constant__ int mass_subtree_sizes[%d] = {%s};\n' %
        (nb,','.join(map(str,intervals)))) if intervals is not None else ''
    source='namespace libero_position {\n'+constants+'__device__ __forceinline__ void dataflow(int w, '+signature+') {\n'+'\n'.join(code)+'\n}\n}\n'
    return source,values,specs,dict(shared_bytes=8*shared_doubles,
        subtree_mass=intervals is not None,
        phase_barriers=sum(statement.count('__syncthreads()') for statement in code),
        legacy_phase_barriers=len(_dataflow_stages(d)),fixed_shared_doubles=fixed,
        geometry_shared_doubles=geom_shared,bounds_cache=bool(bounds_cache),
        mass_private_layout="component planes; public rawbit interleaved publication",mass_global_bytes_delta=0,geometry_private_layout="component planes replacing 3*ngeom shared AoS",
        body_phase_doubles=body_phase,mass_phase_doubles=mass_phase,
        native_dimensions=dict(nq=nq,nbody=nb,njnt=nj,nmocap=nm,nv=nv,nM=nM))


def create_dataflow_plan(d,build_root,*,dataflow_profile='compatible',bounds_cache=None,subtree_mass=None):
    if dataflow_profile not in ('compatible','fast'):
        raise ValueError('position dataflow profile must be compatible or fast')
    if subtree_mass is None:
        subtree_mass=dataflow_profile=='fast' and scalar.bits==64
    if type(subtree_mass) is not bool or subtree_mass and scalar.bits!=64:
        raise ValueError('subtree_mass must be bool and requires FP64 when enabled')
    source,values,specs,workspace=_dataflow_source(d,bounds_cache=bounds_cache,subtree_mass=subtree_mass)
    # Geometry frames and durable publication have independent owners. More
    # warps hide their memory latency; small/compatible cases keep the default.
    original_threads=128 if dataflow_profile=='fast' and d.position.frames.ngeom>=128 else 64
    threads=256 if workspace['subtree_mass'] else original_threads
    plan=NativePlan('position_dataflow_'+dataflow_profile,
        [(GPU_ROOT / 'cuda/common/packed_mass.cuh'),(GPU_ROOT / 'cuda/position/position.cuh'),(GPU_ROOT / 'cuda/position/position_mass.cuh'),
         *(bounds_cache.get('sources',[]) if bounds_cache else []),source],
        [('libero_position::dataflow',values,'block',specs)],d.capacity,build_root,
        device=d.device,block_size=threads,defines={'LIBERO_POSITION_DATAFLOW':1,
            'LIBERO_POSITION_BODY_DIAGNOSTICS':int(dataflow_profile!='fast'),
            **_finite_vector_defines(d)},stage_barriers=False,
        dynamic_shared_bytes=workspace['shared_bytes'])
    plan.manifest.update(dataflow_profile=dataflow_profile,workspace=workspace,
        body_diagnostics=dataflow_profile!='fast',threads_per_world=threads,
        original_threads_per_world=original_threads,
        normalization_policy=('component_divisions' if dataflow_profile!='fast' or int(d.position.factor.nv) in (21,27)
                              else 'reciprocal_multiply'),
        execution='one block/world; shared body/frame/mass dataflow; explicit phase barriers',
        geometry_consumer=('fresh local geom pose -> shared -> original PCA/bounds into private C arena; epoch-checked original C stage publication' if bounds_cache else 'component-only; bounds cache not bound'),
        geometry_hook='optional LIBERO_POSITION_GEOM_HOOK; default no-op and not counted as fusion completion',
        factor_output='existing position.factor ld/inv/sqrt_inv/pivot/status/warnings; D may consume by current position lineage')
    return plan


def install_dataflow(bundle,build_root,*,dataflow_profile='compatible',subtree_mass=None):
    """Explicit dataflow candidate; legacy install remains the comparison route."""
    d=bundle.dynamics
    bounds_cache=None
    temporal_bounds=False
    collision=getattr(getattr(bundle.runtime,'forward',None),'c',None)
    if collision is not None:
        ws=getattr(collision,'workspace',collision)
        temporal_bounds=bool(getattr(ws,'_cuda_collision_temporal_sources',()))
        if temporal_bounds:
            from .collision import _validate_temporal_position_owner
            _validate_temporal_position_owner(ws)
            if ws._cuda_collision_temporal_dynamics is not d:
                raise ValueError('temporal C bounds owner differs from F position owner')
            ws._cuda_collision_temporal_f_pca_omitted=True
        else:
            from .collision import prepare_bounds_cache
            bounds_cache=prepare_bounds_cache(ws,d)
    plan=create_dataflow_plan(d,build_root,dataflow_profile=dataflow_profile,bounds_cache=bounds_cache,
                              subtree_mass=subtree_mass)
    position_hold,snapshot=_create_position_snapshot_plan(d,Path(build_root)/'snapshot')
    if temporal_bounds:
        plan.manifest['geometry_consumer']='fresh geom pose -> C temporal bounds; F PCA bounds producer omitted after cold admission'
    def launch_position(self,stream=None):
        self.validate(stream);self.position.validate_bindings();self.transmission.validate_bindings()
        if self._holds['position'] is not position_hold:
            raise ValueError('position snapshot hold owner changed')
        position_hold.validate(position_hold.fields)
        snapshot.launch(stream=self.stream)
        plan.launch(stream=self.stream)
        self._holds['position'].restore_unrequested(self.requested_mask,self.stream)
    d.launch_position=MethodType(launch_position,d)
    d._cuda_position_plan=plan
    d._cuda_position_snapshot_plan=snapshot
    d.cuda_packed_factor_source=d.position.factor
    # Scene callbacks remain separately installed by the assembly owner.
    return dict(plan.manifest,installed=True,protected_history='mask0 snapshot/original restore_unrequested',
        history_snapshot=snapshot.manifest,
        implementation='shared phase workspace + register local vectors; compatible profile retains body traces',
        bounds_cache=bool(bounds_cache),
        temporal_bounds=temporal_bounds,
        unresolved=[] if bounds_cache or temporal_bounds else ['F geom/PCA bounds cache not bound for this component-only owner'])
