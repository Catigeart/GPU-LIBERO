"""Shipped-domain CUDA ownership and original MJWarp N=2 control comparison.

One imported model is shared read-only by two independent Data/controller/
observation owners. Each physical path is measured once, warmup5/control20;
initialization/reset/capture are outside timing. No camera or task predicate
packaging is timed. Original numerical operators are prepared offline first.
"""
import argparse,gc,hashlib,inspect,json,subprocess,sys,time
from pathlib import Path
from libero.gpu.host.build.task_catalog import load_catalog
from types import SimpleNamespace as NS
import numpy as np
from .mjwarp_public_forward import ROOTS,OUTPUTS,compare,graph_audit
from .mjwarp_camlight import digest


def add_arguments(p):
    p.add_argument('--build-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tasks',nargs='+',default=['all'])
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--prepare-and-run',action='store_true',help='prepare offline, then freeze compilation and reuse the same model/Graphs for one measurement')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--correctness-only',action='store_true',help='observe warmup5/control20 without timing events or a performance measurement')
    p.add_argument('--recheck-leaves',action='store_true',help='replay common-input cvel/mass solve on saved endpoints without repeating timing')
    p.add_argument('--reassess-saved',action='store_true',help='recheck retained endpoints after validation-only changes; do not repeat timing')
    p.add_argument('--worker-task',help=argparse.SUPPRESS)
    return p


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n'
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(text);temp.replace(path)


def source_signature():
    from libero.gpu.host.backends.mjwarp.build import source_files
    flags={key:True for key in inspect.signature(source_files).parameters}
    return dict(sources=source_files(**flags),tool=digest(__file__),forward_helper=digest(Path(__file__).with_name('mjwarp_public_forward.py')),
                repeat_helper=digest(Path(__file__).with_name('mjwarp_trajectory_reference.py')),
                scene_helper=digest(Path(__file__).with_name('mjwarp_scene_inputs.py')),
                actions_helper=digest(Path(__file__).with_name('common.py')),protocol=dict(worlds=2,warmup=5,steps=20,repeats=1,cameras=False,scope='device control Graph with identical OSC and observation sampler; no host packaging'))


def gpu_state():
    result={}
    for name,fields in [('gpu','utilization.gpu,memory.used'),('processes','pid,process_name,used_memory')]:
        kind='--query-gpu=' if name=='gpu' else '--query-compute-apps='
        p=subprocess.run(['nvidia-smi',kind+fields,'--format=csv,noheader'],capture_output=True,text=True)
        result[name]=p.stdout.strip() if p.returncode==0 else p.stderr.strip()
    return result


class ReferenceOwner:
    """Original physics with the same maintained OSC/observation bridge."""
    def __init__(self,env):
        import warp as wp
        import mujoco_warp as mjw
        from libero.gpu.host.backends.mjwarp.runtime import Runtime
        from libero.gpu.host.backends.mjwarp.newton import Newton
        from libero.gpu.host.adapters.observations.adapter import GpuObservationAdapter
        self.stream,self.built,self.model=env.stream,env.built,env.model
        self.num_envs,self.substeps,self.capacities=2,env.substeps,env.capacities
        self.identity=dict(mjwarp_commit=env.identity['mjwarp_commit'],preset='upstream')
        self.public_forward_implementation='upstream';self._held_contacts={}
        self.data=mjw.make_data(env.built.native_model,nworld=2,nccdmax=env.capacities['nconmax'],**env.capacities)
        self.observer=GpuObservationAdapter(env.built.native_model,env.built.metadata,2,device=self.stream.device)
        self.runtime=Runtime(self,self.model,self.data,env.library_path,newton=Newton(self.model,self.data,self.stream))
        self._fields=None
    def observation_fields(self):
        import warp as wp
        if self._fields is None:
            m,d=self.built.native_model,self.data
            def alias(a,width):return wp.array(ptr=a.ptr,shape=(2,width),dtype=float,device=self.stream.device)
            self._fields=dict(qpos=d.qpos,qvel=d.qvel,body_xpos=alias(d.xpos,m.nbody*3),body_xquat=alias(d.xquat,m.nbody*4),body_xmat=alias(d.xmat,m.nbody*9),geom_xpos=alias(d.geom_xpos,m.ngeom*3),geom_xmat=alias(d.geom_xmat,m.ngeom*9),site_xpos=alias(d.site_xpos,m.nsite*3),site_xmat=alias(d.site_xmat,m.nsite*9),subtree_com=alias(d.subtree_com,m.nbody*3))
        return self._fields
    def initialize(self,env):
        import warp as wp
        for name in ROOTS:wp.copy(getattr(self.data,name),getattr(env.data,name))
        self.runtime.forward()
        self.runtime.controller.bootstrap(env._selection)
        self.observer.reset(env._selection,stream=self.stream)
        self.observer.sample(self.observation_fields(),force=True,world_mask=env._selection,stream=self.stream)
        self.initial_roots={name:wp.clone(getattr(self.data,name)) for name in ROOTS}
        self.initial_controller={name:wp.clone(value) for name,value in self.runtime.controller.buffers.items()}
    def close(self):
        import warp as wp
        wp.synchronize_stream(self.stream);self.observer.close()
        self.runtime._control_graph=None;self.runtime._control_graph_owners.clear();self.runtime.env=None
        self._fields=None;self.runtime=self.observer=self.model=self.data=self.built=None
        self.initial_roots=self.initial_controller=None


def selected_paths(tree,prefix=''):
    rows=[]
    if isinstance(tree,dict):
        for k,v in tree.items():
            if k=='selected':rows.append(dict(path=prefix,selected=v))
            else:rows+=selected_paths(v,prefix+'.'+k)
    return rows


def measure(runtime,stream):
    import warp as wp
    from .common import actions
    for k in range(5):runtime.enqueue_action(actions(k,2))
    wp.synchronize_stream(stream)
    pairs=[(wp.Event(device=stream.device,enable_timing=True),wp.Event(device=stream.device,enable_timing=True)) for _ in range(20)]
    start=time.perf_counter()
    for k,(begin,end) in enumerate(pairs):
        wp.record_event(begin,stream);runtime.enqueue_action(actions(k+5,2));wp.record_event(end,stream)
    wp.synchronize_stream(stream);wall=time.perf_counter()-start
    values=[float(wp.get_event_elapsed_time(a,b,synchronize=False)) for a,b in pairs];total=sum(values)/1000
    return dict(control_ms=values,total_gpu_seconds=total,total_wall_seconds=wall,mean_control_ms=float(np.mean(values)),std_control_ms=float(np.std(values)),minimum_control_ms=min(values),maximum_control_ms=max(values),environment_steps_per_second=40/total,physical_world_steps_per_second=40*runtime.substeps/total)


def run_controls(runtime,stream):
    """Same5+20 action sequence, without timing events or measurements."""
    import warp as wp
    from .common import actions
    for k in range(25):runtime.enqueue_action(actions(k,2))
    wp.synchronize_stream(stream)



def root_state_checks(host,actual,reference):
    """Existing project FP32 sensitivity screen, frozen reference inputs only.

    Preserve fixed1024eps state comparisons as diagnostics for acceleration;
    use the already established precision.sensitivity_floor for ill-conditioned
    mass solves. Neither constants nor reference inputs depend on the candidate.
    """
    from .precision import sensitivity_floor
    import mujoco
    rows=[]
    for name in ROOTS:
        if name in ('qacc','qacc_warmstart'):continue
        x,y=getattr(actual,name).numpy(),getattr(reference,name).numpy()
        error=float(abs(x.astype(float)-y).max(initial=0.));exact=name in ('time','qfrc_applied','xfrc_applied')
        bound=0. if exact else float(1024*np.finfo(np.float32).eps*max(1.,float(abs(y).max(initial=0.))))
        rows.append(dict(field='root '+name,error=error,bound=bound,passed=bool(np.all(np.isfinite(x)) and error<=bound)))
    for w in range(2):
        matrix=np.empty((host.nv,host.nv));mujoco.mju_sym2dense(matrix,reference.M.numpy()[w].astype(float),host.M_rownnz,host.M_rowadr,host.M_colind)
        for name in ('qacc','qacc_warmstart'):
            x,y=getattr(actual,name).numpy()[w],getattr(reference,name).numpy()[w]
            base=float(1024*np.finfo(np.float32).eps*max(1.,float(abs(y).max(initial=0.))))
            try:floor,condition=sensitivity_floor(matrix,y,'max')
            except ValueError as error:
                rows.append(dict(field=name+' frozen sensitivity',world=w,passed=False,unbounded=str(error)));continue
            bound=max(base,floor);error=float(abs(x.astype(float)-y).max(initial=0.))
            rows.append(dict(field=name+' frozen sensitivity',world=w,error=error,bound=bound,fixed_state_bound=base,frozen_reference_floor=floor,reference_mass_condition=condition,passed=bool(np.all(np.isfinite(x)) and error<=bound)))
    for mode,data in (('cuda',actual),('upstream',reference)):
        rows.append(dict(field=mode+' public warmstart copy',passed=np.array_equal(data.qacc.numpy().view(np.uint8),data.qacc_warmstart.numpy().view(np.uint8))))
    return rows


def initial_numerical_agreement(record):
    return bool(record.get('initial_numerical_passed',True) and all(v['passed'] for v in record.get('initial_checks',[]) if not v.get('diagnostic',False)))


def reassess_saved(args,task,sig):
    """Re-evaluate saved outputs only when every executed source still matches."""
    import mujoco
    key=task['suite']+'-'+str(task['task_id']);folder=args.output/key;path=folder/'summary.json';old=json.loads(path.read_text())
    if old['signature']['sources']!=sig['sources'] or old['signature']['forward_helper']!=sig['forward_helper']:raise RuntimeError('Physics/reference source changed; saved measurements cannot be readmitted')
    archive=folder/'strict-first-run-summary.json'
    if not archive.exists():archive.write_bytes(path.read_bytes())
    wanted=old['signature']['identity']['imported_model_sha256'];found=[]
    for receipt in (args.build_root/'mjwarp/models').glob('*/receipt.json'):
        record=json.loads(receipt.read_text())
        if record['model_sha256']==wanted:found.append(receipt.parent/'model.mjb')
    if not found or digest(found[0])!=wanted:raise RuntimeError('Matching source-bound modern model is absent')
    model=mujoco.MjModel.from_binary_path(str(found[0]));endpoint=folder/'endpoints.npz'
    with np.load(endpoint,allow_pickle=False) as f:arrays={k:f[k].copy() for k in f.files}
    data={mode:NS(**{name:NS(numpy=lambda value=arrays[mode+'.'+name]:value) for name in ROOTS+OUTPUTS}) for mode in ('cuda','upstream')}
    checks=[v for v in old['checks'] if not v['field'].startswith('root ') and 'frozen sensitivity' not in v['field'] and 'public warmstart copy' not in v['field']]
    for row in checks:
        if row['field']=='qacc':row['diagnostic']=True
    checks+=root_state_checks(model,data['cuda'],data['upstream']);old['checks']=checks
    old['strict_first_run_passed']=old.get('strict_first_run_passed',old['passed']);old['passed']=initial_numerical_agreement(old) and all(v['passed'] for v in checks if not v.get('diagnostic',False)) and not old['runtime_compilations'];old['assessment_signature']=sig
    old['assessment']=dict(source='retained endpoints, no new integrations or timing',original_report_sha256=digest(archive),endpoints_sha256=digest(endpoint),policy='existing precision.sensitivity_floor from frozen original mass/acceleration; strict acceleration mismatch remains a diagnostic; roots and exact warmstart copy also checked')
    save(path,old);print(json.dumps(dict(task=key,reassessed=True,passed=old['passed'],speedup=old['speedup'],new_checks=checks[-14:])),flush=True);return 0 if old['passed'] else 2



def endpoint_leaf_checks(env,reference):
    """Mandatory same-input local replays, after endpoint snapshots/timing.

    cvel is sampled before Euler, qvel after Euler. Cross-trajectory cvel and
    unconstrained acceleration deltas are retained as observations; the migrated
    functions still pass the unchanged common-input numerical screen here.
    """
    import warp as wp
    import mujoco
    from mujoco_warp._src import smooth
    from .precision import balance_residual
    m,a,b=env.model,env.data,reference.data;rows=[]
    for mode,data in (('cuda',a),('upstream',b)):
        values={name:getattr(data,name).numpy() for name in ('M','qacc_smooth','qfrc_smooth')}
        for w in range(2):
            matrix=np.empty((m.nv,m.nv));host=env.built.native_model;mujoco.mju_sym2dense(matrix,values['M'][w].astype(float),host.M_rownnz,host.M_rowadr,host.M_colind)
            rows.append(dict(field=mode+' unconstrained mass backward error',world=w,**balance_residual(matrix,values['qacc_smooth'][w],values['qfrc_smooth'][w],np.zeros(m.nv))))
    for name in ('qvel','cdof'):wp.copy(getattr(b,name),getattr(a,name),stream=env.stream)
    env.runtime.physics.velocity_driver.com_vel.enqueue();smooth.com_vel(m,b)
    wp.synchronize_stream(env.stream)
    for name in ('cvel','cdof_dot'):
        x,y=getattr(a,name).numpy(),getattr(b,name).numpy();error=float(abs(x.astype(float)-y).max(initial=0.));bound=float(1024*np.finfo(np.float32).eps*max(1.,float(abs(y).max(initial=0.))));rows.append(dict(field='common-input '+name,error=error,bound=bound,passed=bool(np.all(np.isfinite(x)) and error<=bound)))
    for name in ('M','qLD','qLDiagInv','qfrc_smooth'):wp.copy(getattr(b,name),getattr(a,name),stream=env.stream)
    env.runtime.physics.acceleration_driver.binding.enqueue();smooth.solve_m(m,b,b.qacc_smooth,b.qfrc_smooth)
    wp.synchronize_stream(env.stream);x,y=a.qacc_smooth.numpy(),b.qacc_smooth.numpy();error=float(abs(x.astype(float)-y).max(initial=0.));bound=float(1024*np.finfo(np.float32).eps*max(1.,float(abs(y).max(initial=0.))));rows.append(dict(field='common-input unconstrained mass solve',error=error,bound=bound,passed=bool(np.all(np.isfinite(x)) and error<=bound)))
    return dict(passed=all(v['passed'] for v in rows),checks=rows,integrations=0,performance=None)


def initial_runtime_contracts(actual,reference):
    """Structural/finite admission, separate from strict numerical agreement.

    A failed finite numerical screen remains a failed comparison, but its
    source-bound runtime can still be measured once for diagnosis. Invalid
    storage counts, nonfinite state or altered public inputs stop execution.
    """
    rows=[]
    for mode,data in (('cuda',actual),('upstream',reference)):
        for name in dict.fromkeys(ROOTS+OUTPUTS):
            rows.append(dict(field=mode+' finite '+name,passed=bool(np.all(np.isfinite(getattr(data,name).numpy())))))
        n=int(data.nacon.numpy()[0]);nefc=data.nefc.numpy()
        rows.append(dict(field=mode+' contact count storage',passed=bool(0<=n<=data.naconmax)))
        rows.append(dict(field=mode+' row count storage',passed=bool(np.all((nefc>=0)&(nefc<=data.njmax)))))
    for name in ROOTS:
        if name=='qacc':continue  # Public forward computes acceleration.
        a,b=(getattr(data,name).numpy() for data in (actual,reference))
        rows.append(dict(field='common initial input '+name,passed=np.array_equal(a.view(np.uint8),b.view(np.uint8))))
    return rows


def recheck_leaves(args,task,sig):
    from libero.libero import get_libero_path
    from libero.libero.envs import GpuBatchEnv
    from libero.gpu.host.build.compilation import compilation_scope
    import warp as wp
    key=task['suite']+'-'+str(task['task_id']);folder=args.output/key;path=folder/'summary.json';old=json.loads(path.read_text())
    if old['signature']['sources']!=sig['sources']:raise RuntimeError('Executed physics sources changed; cannot recheck saved trajectory')
    archive=folder/'strict-first-run-summary.json'
    if not archive.exists():archive.write_bytes(path.read_bytes())
    with compilation_scope(False) as scope:
        env=GpuBatchEnv(str(Path(get_libero_path('bddl_files'))/task['key']),2,physics_engine='mjwarp',precision='fp32',initialize=False,allow_compile=False,build_root=args.build_root,use_camera_obs=False,ignore_done=True,mjwarp_options=dict(preset='cuda_libero',fast_compile=True));reference=None
        try:
            if env.identity!=old['signature']['identity']:raise RuntimeError('Saved endpoint scene/product identity differs')
            with wp.ScopedStream(env.stream):
                reference=ReferenceOwner(env)
                with np.load(folder/'endpoints.npz',allow_pickle=False) as data:
                    for mode,target in (('cuda',env.data),('upstream',reference.data)):
                        for name in dict.fromkeys(ROOTS+OUTPUTS):getattr(target,name).assign(data[mode+'.'+name])
                leaf=endpoint_leaf_checks(env,reference)
            checks=old['checks']
            for row in checks:
                if row['field'] in ('cvel','qacc_smooth'):row['diagnostic']=True
            old['strict_first_run_passed']=old.get('strict_first_run_passed',old['passed']);old['endpoint_leaf_checks']=leaf;old['assessment_signature']=sig;old['passed']=initial_numerical_agreement(old) and leaf['passed'] and all(v['passed'] for v in checks if not v.get('diagnostic',False)) and not scope['events'];old['leaf_recheck']=dict(original_report_sha256=digest(archive),endpoints_sha256=digest(folder/'endpoints.npz'),runtime_compilations=scope['events'],additional_integrations=0,timing_repeated=False)
            save(path,old);print(json.dumps(dict(task=key,passed=old['passed'],leaf=leaf)),flush=True);return 0 if old['passed'] else 2
        finally:
            if reference is not None:reference.close()
            env.close()


def run_prepared(env,reference,signature,phases,audit,scope,competition,task,folder,*,correctness_only=False):
    import warp as wp
    key=task['suite']+'-'+str(task['task_id']);paths=selected_paths(env.identity)
    initial=compare(env.built.native_model,env.data,reference.data)
    for mode,data in (('cuda',env.data),('upstream',reference.data)):
        initial.append(dict(field=mode+' initial contact capacity',actual=int(data.nacon.numpy()[0]),capacity=data.naconmax,passed=int(data.nacon.numpy()[0])<=data.naconmax))
        initial.append(dict(field=mode+' initial row capacity',actual=data.nefc.numpy().tolist(),capacity=data.njmax,passed=bool(np.all(data.nefc.numpy()<=data.njmax))))
    initial_passed=all(v['passed'] for v in initial if not v.get('diagnostic',False))
    runtime_checks=initial_runtime_contracts(env.data,reference.data)
    runtime_usable=all(v['passed'] for v in runtime_checks)
    if not initial_passed or not runtime_usable:
        np.savez(folder/'initial-failure.npz',**{mode+'.'+name:getattr(data,name).numpy() for mode,data in (('cuda',env.data),('upstream',reference.data)) for name in ROOTS+OUTPUTS})
    if not runtime_usable:
        result=dict(passed=False,stage='initial-forward',signature=signature,checks=initial,audit=audit,phases_seconds=phases,runtime_compilations=scope['events']);save(folder/'summary.json',result);print(json.dumps(dict(task=key,passed=False,stage='initial-forward',failures=[v for v in initial if not v['passed'] and not v.get('diagnostic',False)])),flush=True);return 2
    order=('upstream','cuda') if task['task_id']%2==0 else ('cuda','upstream');timings={};completed=[]
    for mode in order:
        runtime=reference.runtime if mode=='upstream' else env.runtime
        if correctness_only:run_controls(runtime,env.stream)
        else:timings[mode]=measure(runtime,env.stream)
        completed.append(mode)
        record=dict(signature=signature,timings=timings,order=order,phases_seconds=phases,competition_before=competition,completed_modes=completed)
        if correctness_only:record.update(kind='correctness_only',controls_per_mode=25,performance=None)
        save(folder/('execution.json' if correctness_only else 'measurement.json'),record)
    checks=compare(env.built.native_model,env.data,reference.data)
    # Evolved contact counts are observations, not frozen-input
    # ownership contracts: close states may cross contact thresholds.
    for row in checks:
        if row['field'] in ('nefc','nacon','qacc','cvel','qacc_smooth'):row['diagnostic']=True
    checks+=root_state_checks(env.built.native_model,env.data,reference.data)
    for mode,runtime in (('cuda',env.runtime),('upstream',reference.runtime)):
        counts=runtime.integration_count.numpy();bits=runtime.data.overflow.numpy()
        from mujoco_warp._src import types
        legal=int(types.OverflowType.ITERATIONS|types.OverflowType.LS_ITERATIONS)
        checks.append(dict(field=mode+' actual integrations',expected=625,actual=counts.tolist(),passed=bool(np.all(counts==625))))
        checks.append(dict(field=mode+' capacity',actual=bits.tolist(),passed=bool(not np.any(bits & ~legal))))
    endpoint={mode+'.'+name:getattr(data,name).numpy() for mode,data in (('cuda',env.data),('upstream',reference.data)) for name in ROOTS+OUTPUTS};np.savez(folder/'endpoints.npz',**endpoint)
    repeated=None
    if any(not v['passed'] for v in checks if not v.get('diagnostic',False)):
        from .mjwarp_trajectory_reference import replay
        repeated=replay(env,reference,endpoint,folder,tuple(dict.fromkeys(ROOTS+OUTPUTS)))
    leaf=endpoint_leaf_checks(env,reference)
    passed=initial_passed and leaf['passed'] and all(v['passed'] for v in checks if not v.get('diagnostic',False)) and not scope['events'];result=dict(passed=passed,signature=signature,initial_numerical_passed=initial_passed,initial_runtime_contracts=runtime_checks,initial_checks=initial,checks=checks,endpoint_leaf_checks=leaf,endpoints_sha256=digest(folder/'endpoints.npz'),audit=audit,selected=paths,phases_seconds=phases,timings=timings,speedup=None if correctness_only else timings['upstream']['total_gpu_seconds']/timings['cuda']['total_gpu_seconds'],order=order,competition_before=competition,competition_after=gpu_state(),runtime_compilations=scope['events'],worlds=2,warmup=5,steps=20,repeats=1)
    if correctness_only:result.update(execution=dict(kind='correctness_only',controls_per_mode=25,completed_modes=completed),performance=None)
    result['original_repeat_diagnostic']=repeated
    save(folder/'summary.json',result);print(json.dumps(dict(task=key,passed=passed,speedup=result['speedup'],cuda_ms=timings.get('cuda',{}).get('mean_control_ms'),upstream_ms=timings.get('upstream',{}).get('mean_control_ms'),failures=[v for v in checks if not v['passed'] and not v.get('diagnostic',False)])),flush=True);return 0 if passed else 2


def worker(args,task,sig):
    from libero.gpu import precision
    precision.select('fp32')
    import warp as wp
    from libero.libero import get_libero_path
    from libero.libero.envs import GpuBatchEnv
    from libero.gpu.host.build.compilation import compilation_scope
    from libero.gpu.host.backends.mjwarp.build import profile_sources,save_prepared_profile
    key=task['suite']+'-'+str(task['task_id']);folder=args.output/key;folder.mkdir(parents=True,exist_ok=True);receipt=folder/'preparation.json';env=reference=None;start=time.perf_counter();competition=gpu_state()
    with compilation_scope(args.prepare) as scope:
        try:
            env=GpuBatchEnv(str(Path(get_libero_path('bddl_files'))/task['key']),2,physics_engine='mjwarp',precision='fp32',initialize=False,allow_compile=args.prepare,build_root=args.build_root,use_camera_obs=False,ignore_done=True,mjwarp_options=dict(preset='cuda_libero',fast_compile=True));initialized=time.perf_counter()-start
            from libero.gpu.host.backends.mjwarp.scene_inputs import read as read_scene_inputs
            from .mjwarp_scene_inputs import verify_imported_model
            original_scene,_=read_scene_inputs(args.build_root,task)
            scene_check=verify_imported_model(env.built.native_model,original_scene);save(folder/'scene-input-check.json',scene_check)
            if not scene_check['passed']:raise RuntimeError('Imported body/site/primitive poses differ from original prepared CPU scene: '+str([v for v in scene_check['checks'] if not v['passed']]))
            paths=selected_paths(env.identity)
            if any(v['selected'] not in ('cuda','cuda_newton') for v in paths) or env.runtime.forward_selected!='cuda' or env.runtime.physics.queries.sensors is None:raise RuntimeError('Complete preset has a visible unsupported numerical domain: '+str(paths))
            if env.model.nhistory or env.model.na:raise RuntimeError('Shipped stateless/no-history premise differs')
            signature=dict(**sig,identity=env.identity)
            if not args.prepare and (not receipt.is_file() or json.loads(receipt.read_text())['signature']!=signature):raise RuntimeError('Prepare exact full original/CUDA operators offline first')
            with wp.ScopedStream(env.stream):
                begin=time.perf_counter();env.reset(seed=0,init_states=env.initial_states[:2]);wp.synchronize_stream(env.stream);reset_seconds=time.perf_counter()-begin
                begin=time.perf_counter();audit=graph_audit(env);wp.synchronize_stream(env.stream);cuda_capture_seconds=time.perf_counter()-begin
                if not audit['passed']:raise RuntimeError('Upstream numerical work remains in complete CUDA Graph: '+str(audit['upstream_numerical_launches']))
                begin=time.perf_counter();reference=ReferenceOwner(env);reference.initialize(env);wp.synchronize_stream(env.stream);reference_init_seconds=time.perf_counter()-begin
                begin=time.perf_counter();reference.runtime.prepare_control_graph();wp.synchronize_stream(env.stream);reference_capture_seconds=time.perf_counter()-begin
                phases=dict(environment_creation=initialized,reset=reset_seconds,cuda_capture=cuda_capture_seconds,reference_creation_forward=reference_init_seconds,reference_capture=reference_capture_seconds)
                if args.prepare:
                    path=save_prepared_profile(env,sig['sources'],scope['events'])
                    save(receipt,dict(signature=signature,profile=str(path),phases_seconds=phases,selected=paths,audit=audit,compilation_events=scope['events'],physical_integrations=0));print(json.dumps(dict(task=key,prepared=True,compilations=len(scope['events']),phases=phases)),flush=True)
                    if not args.prepare_and_run:return 0
                with compilation_scope(False) as frozen:
                    env.allow_compile=False
                    return run_prepared(env,reference,signature,phases,audit,frozen,competition,task,folder,correctness_only=getattr(args,'correctness_only',False))
        finally:
            if reference is not None:reference.close()
            if env is not None:env.close()
            gc.collect()


def main(argv=None,*,args=None):
    if args is None:args=add_arguments(argparse.ArgumentParser(description=__doc__)).parse_args(argv)
    from libero.gpu import precision
    precision.select('fp32')
    if args.prepare_and_run:args.prepare=True
    preparation_only=args.prepare and not args.prepare_and_run
    sig=source_signature();tasks=load_catalog(args.build_root)
    sig['protocol']['correctness_only']=getattr(args,'correctness_only',False)
    selected=[t for t in tasks if any(v in ('all',t['suite'],t['suite']+':'+str(t['task_id'])) for v in ([args.worker_task] if args.worker_task else args.tasks))]
    if not selected:raise ValueError('No selected tasks')
    args.output.mkdir(parents=True,exist_ok=True)
    if args.recheck_leaves:
        return 0 if all(recheck_leaves(args,task,sig)==0 for task in selected) else 2
    if args.reassess_saved:
        return 0 if all(reassess_saved(args,task,sig)==0 for task in selected) else 2
    if args.worker_task:return worker(args,selected[0],sig)
    rows=[]
    for task in selected:
        key=task['suite']+'-'+str(task['task_id']);folder=args.output/key;name='preparation.json' if preparation_only else 'summary.json';record=folder/name
        previous=json.loads(record.read_text()) if args.resume and record.is_file() else None
        assessed=previous.get('assessment_signature',previous.get('signature',{})) if previous else {}
        if previous and all(assessed.get(k)==sig.get(k) for k in ('sources','tool','forward_helper','repeat_helper','scene_helper','actions_helper','protocol')):
            row=dict(task=key,returncode=0 if preparation_only or previous.get('passed') else 2,resumed=True)
        else:
            folder.mkdir(parents=True,exist_ok=True);cmd=[sys.executable,str(Path(__file__).parents[1]/'validate_gpu.py'),'mjwarp-complete','--build-root',str(args.build_root),'--output',str(args.output),'--worker-task',task['suite']+':'+str(task['task_id'])]
            if args.prepare_and_run:cmd.append('--prepare-and-run')
            elif args.prepare:cmd.append('--prepare')
            if getattr(args,'correctness_only',False):cmd.append('--correctness-only')
            with (folder/('prepare.log' if preparation_only else 'run.log')).open('w') as log:process=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
            row=dict(task=key,returncode=process.returncode,resumed=False)
        if record.is_file():
            value=json.loads(record.read_text());row.update(passed=value.get('passed',row['returncode']==0),speedup=value.get('speedup'))
        rows.append(row);save(args.output/('preparation-index.json' if preparation_only else 'index.json'),dict(passed=all(v['returncode']==0 for v in rows),expected=len(selected),completed=len(rows),signature=sig,rows=rows));print(json.dumps(dict(completed=len(rows),expected=len(selected),**row)),flush=True)
    return 0 if all(v['returncode']==0 for v in rows) else 2
if __name__=='__main__':raise SystemExit(main())
