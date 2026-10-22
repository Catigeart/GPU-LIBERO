"""Load-only N=2 FP64 motion replacement and complete-control comparison."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace


def add_arguments(p):
    p.add_argument('--build-root', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True, help='frozen native substep .so')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--task', type=int, default=4)
    p.add_argument('--state', type=Path, help='optional previously recorded FP64 endpoint')
    p.add_argument('--performance', action='store_true')
    p.add_argument('--host-state', action='store_true',
                   help='hold timed versions on the host; transfers stay outside CUDA events')
    p.add_argument('--producer', choices=('motion', 'force'), default='motion')
    p.add_argument('--baseline-motion', action='store_true',
                   help='compare force replacement against a frozen motion product')
    p.add_argument('--baseline-force', action='store_true',
                   help='compare against a frozen accepted force product')
    p.add_argument('--baseline-aligned', action='store_true',
                   help='compare against a frozen accepted init/resume product')
    p.add_argument('--baseline-force-threads', type=int, default=256, choices=(64,128,256))
    p.add_argument('--newton-split', action='store_true',
                   help='evaluate independent Newton init/resume with accepted motion/force producers')
    p.add_argument('--baseline-euler', action='store_true', help='frozen aligned baseline uses its separate Euler entry')
    p.add_argument('--controller-split', action='store_true', help='separate controller from Post/tail')
    p.add_argument('--euler-split', action='store_true',
                   help='evaluate independent Euler after the explicit init/resume chain')
    p.add_argument('--verify-native-workspace', action='store_true',
                   help='also check native Full-Newton/Post workspace reuse, without timing it')
    p.add_argument('--prior-force', type=Path,
                   help='optional frozen producer for same-input local tuning')
    p.add_argument('--threads', type=int, nargs='+',
                   help='limit producer variants when its mathematics is unchanged')


def gpu_sample():
    return {name: subprocess.check_output(['nvidia-smi', query, '--format=csv,noheader'], text=True).strip()
        for name, query in (('gpu', '--query-gpu=utilization.gpu,memory.used'),
                            ('processes', '--query-compute-apps=pid,process_name,used_memory'))}


def arrays_of(*roots):
    import warp as wp
    seen, arrays = set(), {}
    def walk(obj):
        if id(obj) in seen: return
        seen.add(id(obj))
        if isinstance(obj, wp.array):
            if obj.size and obj.device.is_cuda: arrays[(obj.ptr,obj.shape,str(obj.dtype))] = obj
        elif isinstance(obj, dict):
            for x in obj.values(): walk(x)
        elif isinstance(obj, (list,tuple)):
            for x in obj: walk(x)
        elif isinstance(obj,SimpleNamespace) or (type(obj).__module__.startswith('libero.gpu') and hasattr(obj,'__dict__')):
            for x in vars(obj).values(): walk(x)
    for x in roots: walk(x)
    return list(arrays.values())


def save(path, data):
    path.write_text(json.dumps(data,indent=2)+'\n')


def compare(ref, out):
    import numpy as np
    rows=[]
    for key,a in ref.items():
        b=out[key];scale=float(np.max(np.abs(a))) if a.size else 0.
        error=float(np.max(np.abs(a-b))) if a.size else 0.
        limit=1.e-9*(1.+scale)
        rows.append(dict(field=key,max_abs=error,scale=scale,limit=limit,
            passed=bool(np.isfinite(a).all() and np.isfinite(b).all() and error<=limit)))
    return rows


def export_stage_fixture(path, env, candidate, arrays):
    """Freeze actual Post/Euler inputs after one forward, before integration.

    Byte owners retain pointer aliases and element representation. Replayers
    allocate each owner once and patch field offsets; recorded addresses are
    never dereferenced by a later process. No extra timed control is run.
    """
    import numpy as np
    import warp as wp
    from libero.gpu.host.runtime.step.newton_binding import NewtonArguments
    forward = env.runtime.forward
    # AlignedSubstep configures Init/Resume, not the diagnostic Full-Newton
    # kernel below. Wide shapes exceed CUDA's default dynamic shared limit;
    # opt this separate entry in before capture, exactly as StagedSubstep does.
    configure = candidate.library.flash_libero_staged_configure
    configure.argtypes = [ct.c_void_p]*4+[ct.c_int]
    configure.restype = ct.c_int
    resources = [(ct.c_int*5)() for _ in range(4)]
    candidate._check(configure(*resources, candidate.arguments.body_velocity.nb),
                     'configure fixture forward')
    launch = candidate.library.flash_libero_launch_staged_forward
    launch.argtypes = [ct.c_void_p]+[ct.c_int]*4+[ct.c_void_p]
    launch.restype = ct.c_int
    with wp.ScopedCapture(stream=env.stream) as capture:
        wp.copy(forward.mask, env.runtime.intent, stream=env.stream)
        forward.f.begin_forward_with_plan(forward.mask, stream=env.stream,
                                          plan=forward.expected_forward)
        forward._dataflow_reuse.enqueue()
        candidate._check(launch(candidate.aligned_device_arguments.ptr, candidate.worlds,
            1, 0, candidate.post_shared_bytes, env.stream.cuda_stream), 'fixture forward')
    wp.capture_launch(capture.graph, stream=env.stream)
    wp.synchronize_stream(env.stream)
    def read_bytes(pointer, size):
        return wp.array(ptr=pointer, shape=(size,), dtype=wp.uint8,
                        device=env.stream.device, deleter=None).numpy()
    request = NewtonArguments.from_buffer_copy(read_bytes(candidate.arguments.newton.parameters,
                                                          ct.sizeof(NewtonArguments)).tobytes())
    owners = [(a, int(a.ptr), int(a.ptr)+a.size*wp.types.type_size_in_bytes(a.dtype))
              for a in arrays if a.is_contiguous]
    payload, descriptors, owner_keys = {}, {}, {}
    def record(structure):
        fields = {}
        for name, kind in structure._fields_:
            value = getattr(structure, name)
            if kind is not ct.c_void_p:
                fields[name] = dict(value=value)
                continue
            if not value:
                fields[name] = dict(pointer=None)
                continue
            matches = [(hi-lo, lo, a) for a, lo, hi in owners if lo <= value < hi]
            if not matches:
                raise ValueError('fixture owner not retained: '+name)
            # A retained one-world view may alias a full batched owner. Keep
            # the widest containing allocation so later worlds remain valid.
            size, lo, owner = max(matches, key=lambda row: row[0])
            key = owner_keys.get((lo, size))
            if key is None:
                key = 'owner_'+str(len(owner_keys))
                owner_keys[lo, size] = key
                payload[key] = read_bytes(lo, size)
                descriptors[key] = dict(bytes=size, shape=list(owner.shape), dtype=str(owner.dtype))
            fields[name] = dict(owner=key, offset=int(value)-lo)
        return dict(bytes=ct.sizeof(structure), fields=fields)
    schema = dict(worlds=candidate.worlds, scalar_bytes=8,
        phase='one actual standalone forward; no controller or Euler',
        post=record(request.post), euler=record(candidate.arguments.euler),
        state=record(candidate.arguments.state), owners=descriptors)
    payload['private_qacc'] = candidate.qacc_handoff.numpy()
    payload['schema'] = np.array(json.dumps(schema))
    np.savez_compressed(path, **payload)
    return dict(path=str(path), owners=len(descriptors), bytes=path.stat().st_size)


def main(args):
    import faulthandler
    import numpy as np
    from libero.gpu import precision
    precision.select('fp64')
    import warp as wp
    from libero.libero import get_libero_path
    from libero.libero.envs import GpuBatchEnv
    from libero.gpu.host.build.compilation import compilation_scope,install_warp_guard
    from libero.gpu.host.runtime.step.fused_substep import FusedSubstep
    from libero.gpu.host.runtime.step.peeled_motion import PeeledMotionSubstep
    from libero.gpu.host.runtime.step.peeled_force import PeeledForceSubstep
    from libero.gpu.host.runtime.step.physics import _reference_mask
    if sum((args.baseline_motion, args.baseline_force, args.baseline_aligned)) > 1:
        raise ValueError('select one frozen baseline kind')
    if args.euler_split and not args.newton_split:
        raise ValueError('--euler-split requires --newton-split')
    if args.baseline_euler and not args.baseline_aligned:
        raise ValueError('--baseline-euler requires --baseline-aligned')
    if args.controller_split and not args.euler_split:
        raise ValueError('--controller-split requires --euler-split')
    install_warp_guard();wp.init()
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/f'task{args.task:02d}.json'
    report=dict(task=args.task,worlds=2,producer=args.producer,region={},cases={},performance={},passed=False)
    def event(name,**kw):print(json.dumps(dict(event=name,**kw)),flush=True)
    with compilation_scope(False) as scope:
        task=next(t for t in json.loads((args.build_root/'TASK_INSTANCES.json').read_text())['tasks']
                  if t['suite']=='libero_10' and t['task_id']==args.task)
        report['inputs']=dict(task_key=task['key'],model_directory=task['directory'],
            state=str(args.state) if args.state else 'official_initial_states',
            state_sha256=hashlib.sha256(args.state.read_bytes()).hexdigest() if args.state else None,
            action=[.02,-.015,.01,.005,-.003,.002,.1])
        tick=time.perf_counter();event('constructing',task=args.task)
        faulthandler.dump_traceback_later(180,repeat=True)
        env=GpuBatchEnv(Path(get_libero_path('bddl_files'))/task['key'],2,
            init_states=np.load(Path(task['directory'])/'initial_states.npy'), initialize=False,
            allow_compile=False,build_root=args.build_root,debug=False,precision='fp64',
            physics_engine='native',use_camera_obs=False,ignore_done=True)
        faulthandler.cancel_dump_traceback_later()
        report['construction_s']=time.perf_counter()-tick
        try:
            if args.state:
                with np.load(args.state) as z:
                    ep={k:z['candidate.'+k].copy() for k in ('qpos','qvel','time','qacc_warmstart')}
                env.reset(seed=0,init_states=np.column_stack((ep['time'],ep['qpos'],ep['qvel'])))
                env.runtime.euler.qacc_warmstart.assign(ep['qacc_warmstart'])
            else:env.reset(seed=0)
            env._physics.stage_action(np.tile([.02,-.015,.01,.005,-.003,.002,.1],(2,1)))
            baseline_class=PeeledForceSubstep if args.baseline_force else (
                PeeledMotionSubstep if args.baseline_motion else FusedSubstep)
            if args.baseline_aligned:
                from libero.gpu.host.runtime.step.aligned_substep import AlignedSubstep
                baseline_class=AlignedSubstep
            base=baseline_class(env.runtime,args.build_root/'cmake-control',allow_build=False,fast_compile=True)
            if args.baseline_force:base.force_threads=args.baseline_force_threads
            old=ct.CDLL(str(args.baseline.resolve()))
            prefix='aligned' if args.baseline_aligned else 'force' if args.baseline_force else 'motion' if args.baseline_motion else 'substep'
            size=getattr(old,'flash_libero_'+prefix+'_arguments_size');size.restype=ct.c_size_t
            assert size()==ct.sizeof(base.aligned_arguments if args.baseline_aligned else base.arguments)
            if prefix in ('motion','force'):
                if getattr(old,'flash_libero_'+prefix+'_nv')()!=base.variant.nv:
                    raise ValueError('frozen baseline NV differs')
            elif args.baseline_aligned:
                if old.flash_libero_staged_nv()!=base.variant.nv:
                    raise ValueError('frozen baseline NV differs')
            elif old.flash_libero_substep_dimension(0)!=base.variant.nv:
                raise ValueError('frozen baseline NV differs')
            base.library=old
            if args.baseline_aligned:
                configure=old.flash_libero_aligned_configure
                configure.argtypes=[ct.c_void_p]*4+[ct.c_int]; configure.restype=ct.c_int
                values=[(ct.c_int*5)() for _ in range(4)]
                with wp.ScopedDevice(env.stream.device):
                    assert configure(*values,base.arguments.body_velocity.nb)==0
                base.post_shared_bytes=values[3][3]
                base.aligned_launch=old.flash_libero_launch_aligned_substep
                if args.baseline_euler:
                    configure_euler=old.flash_libero_aligned_euler_configure
                    configure_euler.argtypes=[ct.POINTER(ct.c_int),ct.c_int]
                    with wp.ScopedDevice(env.stream.device):
                        assert configure_euler((ct.c_int*8)(),base.arguments.body_velocity.nb)==0
                    base.aligned_launch=old.flash_libero_launch_aligned_euler_substep
                base.aligned_launch.argtypes=[ct.c_void_p]+[ct.c_int]*8+[ct.c_void_p]
                base.aligned_launch.restype=ct.c_int
            elif args.baseline_force:
                configure=old.flash_libero_force_configure;configure.argtypes=[ct.c_void_p]*3
                values=[(ct.c_int*2)() for _ in range(3)]
                with wp.ScopedDevice(env.stream.device):assert configure(*values)==0
                base.launch_force_substep=old.flash_libero_launch_force_substep
                base.launch_force_substep.argtypes=[ct.c_void_p]*4+[ct.c_int]*6+[ct.c_void_p]
                base.launch_force_substep.restype=ct.c_int
            elif args.baseline_motion:
                configure=old.flash_libero_motion_configure;configure.argtypes=[ct.c_void_p]*3
                values=[(ct.c_int*2)() for _ in range(3)]
                with wp.ScopedDevice(env.stream.device):assert configure(*values)==0
                base.launch_peeled=old.flash_libero_launch_motion_substep
                base.launch_peeled.argtypes=[ct.c_void_p]*3+[ct.c_int]*5+[ct.c_void_p]
                base.launch_peeled.restype=ct.c_int
            else:
                configure=old.flash_libero_substep_configure;configure.argtypes=[ct.c_void_p]*2
                with wp.ScopedDevice(env.stream.device):assert configure(ct.byref(ct.c_int()),ct.byref(ct.c_int()))==0
                base.launch_post_entry=old.flash_libero_launch_substep_with_post
                base.launch_post_entry.argtypes=[ct.c_void_p]+[ct.c_int]*4+[ct.c_void_p]
                base.launch_entry=old.flash_libero_launch_substep
                base.launch_entry.argtypes=[ct.c_void_p]+[ct.c_int]*3+[ct.c_void_p]
                base.select_entry=old.flash_libero_select_substep_continuation
                base.select_entry.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ct.c_void_p,ct.c_void_p]
            candidate_class=PeeledForceSubstep if args.producer=='force' else PeeledMotionSubstep
            if args.newton_split:
                if args.producer != 'force': raise ValueError('--newton-split requires --producer force')
                from libero.gpu.host.runtime.step.aligned_substep import AlignedSubstep
                candidate_class=AlignedSubstep
                if args.euler_split:
                    from libero.gpu.host.runtime.step.aligned_substep import EulerAlignedSubstep
                    candidate_class=EulerAlignedSubstep
                if args.controller_split:
                    from libero.gpu.host.runtime.step.aligned_substep import ControllerAlignedSubstep
                    candidate_class=ControllerAlignedSubstep
            candidate=candidate_class(env.runtime,args.build_root/'cmake-control',
                allow_build=False,fast_compile=True,motion_threads=512)
            assert candidate.eligible,candidate.resources
            report['resources']=candidate.resources
            implementations=[('baseline',base),('candidate',candidate)]
            if args.verify_native_workspace:
                if not args.newton_split:
                    raise ValueError('--verify-native-workspace requires --newton-split')
                native=PeeledForceSubstep(env.runtime,args.build_root/'cmake-control',
                    allow_build=False,fast_compile=True,motion_threads=512)
                assert native.eligible,native.resources
                implementations.append(('native_workspace',native))
                report['native_workspace_resources']=native.resources
            prior_force=None
            if args.prior_force:
                if args.producer!='force':raise ValueError('--prior-force requires --producer force')
                prior_library=ct.CDLL(str(args.prior_force.resolve()))
                size=prior_library.flash_libero_force_arguments_size;size.restype=ct.c_size_t
                if size()!=ct.sizeof(candidate.arguments) or prior_library.flash_libero_force_nv()!=candidate.variant.nv:
                    raise ValueError('prior force ABI/shape differs')
                prior_force=prior_library.flash_libero_launch_force
                prior_force.argtypes=candidate.launch_force.argtypes;prior_force.restype=ct.c_int
            report['libraries']={name:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                for name,p in (('baseline',args.baseline),('candidate',Path(candidate.library._name)))}
            f=env.runtime.forward;d=f.d
            force_cached=wp.array(np.array([1,0],np.int32),dtype=wp.int32,device=env.stream.device)
            arrays=arrays_of(env._physics.bundle,*(x for _,x in implementations),force_cached)
            # Borrow the already admitted scalar owner, including plans whose
            # closure retains it outside the ordinary object inventory.
            factor_reuse=wp.array(ptr=base.arguments.smooth_acceleration.factor_reuse_count,
                dtype=wp.int64,shape=(2,),device=env.stream.device,deleter=None)
            if not any(a.ptr==factor_reuse.ptr and a.dtype==wp.int64 and a.shape==(2,) for a in arrays):
                arrays.append(factor_reuse)
            def snapshot():
                wp.synchronize_stream(env.stream)
                return [a.numpy().copy() for a in arrays]
            def restore(values):
                with wp.ScopedStream(env.stream):
                    for a,v in zip(arrays,values):a.assign(v)
                wp.synchronize_stream(env.stream)
            def region():
                if args.producer=='force':
                    result={'actuation.'+k:getattr(d.actuation,k).numpy().copy() for k in
                        ('force','force_pre','ctrl_used','ctrl_route','force_route','qfrc_actuator')}
                    result.update({'smooth.'+k:getattr(d.acceleration,k).numpy().copy() for k in
                        ('smooth','seed','scratch','qfrc','qacc_smooth')})
                    return result
                result={name:getattr(d,name).numpy().copy() for name in ('cvel','cdof_dot','cacc','frc','qfrc_bias')}
                result['passive']=d.passive.qfrc.numpy().copy();return result
            def endpoint():
                e=env.runtime.euler
                out={name:getattr(e,name).numpy().copy() for name in
                     ('qpos','qvel','time','qacc','qacc_warmstart','qfrc_constraint')}
                out.update({'dynamics.'+k:v.numpy().copy() for k,v in d.public.items()})
                out.update({'post.'+k:v.numpy().copy() for k,v in env._physics.bundle.post.public.items()})
                if args.producer=='force':
                    out.update({'warning.'+k:getattr(d.actuation,k).numpy().copy() for k in
                                ('warning_number','warning_lastinfo')})
                return out
            def counters():
                r=env.runtime
                return {k:a.numpy().copy() for k,a in dict(integrated=r.integration_count,
                    newton=f.n.n_attempt,position=f._dataflow_reuse.math,mass_reuse=factor_reuse,
                    reset=r.reset_events,error=r.first_error).items()}
            seed=snapshot();event('ready',resources=report['resources'])
            with wp.ScopedCapture(stream=env.stream) as cap:
                wp.copy(f.mask,env.runtime.intent,stream=env.stream)
                f.f.begin_forward_with_plan(f.mask,stream=env.stream,plan=f.expected_forward)
                f._dataflow_reuse.enqueue()
            wp.capture_launch(cap.graph,stream=env.stream);prepared=snapshot()
            # The original admitted native velocity/reference/bias functions;
            # no Newton, controller, integration or trajectory is required.
            with wp.ScopedCapture(stream=env.stream) as cap:
                f.f.launch_velocity(stream=env.stream)
                f._launch(_reference_mask,[d.valid['velocity'],d.written['velocity'],f.expected_forward,
                    d.first_error,f.transport.active])
                f.s.launch_reference()
                sw=f.s.workspace
                d.launch_bias(reference_done=sw.reference_ready,reference_error=sw.status,
                    reference_written=sw.ref_epoch,reference_expected=f.expected_forward,stream=env.stream)
                if args.producer=='force':
                    d.launch_force(ctrl=f.ctrl,ctrl_valid=f.ctrl_valid,stream=env.stream)
            reference_graph=cap.graph
            wp.capture_launch(reference_graph,stream=env.stream);wp.synchronize_stream(env.stream)
            expected=region();assert not np.any(counters()['error'])
            np.savez(args.output/f'task{args.task:02d}-region-reference.npz',
                **expected, qpos=d.qpos.numpy(), qvel=d.qvel.numpy(), ctrl=f.ctrl.numpy(),
                applied=d.acceleration.applied.numpy(), xfrc=d.acceleration.xfrc.numpy(),
                model_path=str(Path(task['directory'])/'model.mjb'))
            choices=(64,128,256) if args.producer=='force' else (512,256,128)
            if args.threads:
                if any(n not in choices for n in args.threads):raise ValueError('unsupported producer thread count')
                choices=tuple(dict.fromkeys(args.threads))
            for threads in choices:
                restore(prepared)
                if args.producer=='force':
                    candidate.force_threads=threads
                    with wp.ScopedCapture(stream=env.stream) as motion_capture:candidate.enqueue_motion()
                    wp.capture_launch(motion_capture.graph,stream=env.stream)
                    wp.synchronize_stream(env.stream)
                    with wp.ScopedCapture(stream=env.stream) as cap:candidate.enqueue_force()
                else:
                    candidate.motion_threads=threads
                    with wp.ScopedCapture(stream=env.stream) as cap:candidate.enqueue_motion()
                wp.capture_launch(cap.graph,stream=env.stream);wp.synchronize_stream(env.stream)
                readiness=candidate.force_ready if args.producer=='force' else candidate.ready
                assert readiness.numpy().tolist()==[1,1],readiness.numpy()
                checks=compare(expected,region());report['region'][str(threads)]=checks;save(path,report)
                assert all(x['passed'] for x in checks),checks
                event('region_passed',threads=threads,max_abs=max(x['max_abs'] for x in checks))
                if args.performance:
                    for _ in range(5):wp.capture_launch(cap.graph,stream=env.stream)
                    starts=[wp.Event(env.stream.device,enable_timing=True) for _ in range(20)]
                    stops=[wp.Event(env.stream.device,enable_timing=True) for _ in range(20)]
                    for start,stop in zip(starts,stops):
                        env.stream.record_event(start);wp.capture_launch(cap.graph,stream=env.stream);env.stream.record_event(stop)
                    wp.synchronize_stream(env.stream)
                    us=[1000*wp.get_event_elapsed_time(x,y,synchronize=False) for x,y in zip(starts,stops)]
                    report['performance'][args.producer+'_'+str(threads)]=dict(us=us,mean_us=float(np.mean(us)),median_us=float(np.median(us)))
                    if prior_force:
                        with wp.ScopedCapture(stream=env.stream) as prior_capture:
                            candidate._check(prior_force(candidate.device_arguments.ptr,candidate.ready.ptr,
                                candidate.force_ready.ptr,candidate.worlds,candidate.START,threads,
                                env.stream.cuda_stream),'prior force replay')
                        for _ in range(5):wp.capture_launch(prior_capture.graph,stream=env.stream)
                        starts=[wp.Event(env.stream.device,enable_timing=True) for _ in range(20)]
                        stops=[wp.Event(env.stream.device,enable_timing=True) for _ in range(20)]
                        for start,stop in zip(starts,stops):
                            env.stream.record_event(start);wp.capture_launch(prior_capture.graph,stream=env.stream)
                            env.stream.record_event(stop)
                        wp.synchronize_stream(env.stream)
                        us=[1000*wp.get_event_elapsed_time(x,y,synchronize=False) for x,y in zip(starts,stops)]
                        report['performance']['prior_force_'+str(threads)]=dict(us=us,mean_us=float(np.mean(us)),median_us=float(np.median(us)))
            selected=min(choices,key=lambda n:report['performance'][args.producer+'_'+str(n)]['median_us']) if args.performance else choices[0]
            setattr(candidate,args.producer+'_threads',selected);report['selected_threads']=selected
            report['resources'][args.producer+'_threads']=selected
            if args.producer=='force':
                report['force_inputs']={}
                for case in ('external_wrench','control_limits','badctrl_fallback'):
                    restore(prepared)
                    if case=='external_wrench':
                        wrench=np.zeros_like(d.acceleration.xfrc.numpy())
                        wrench[0,-1]=[.7,-.3,.4,.05,-.02,.03]
                        wrench[0,5]=[-.15,.2,.3,.04,.01,-.03]
                        wrench[1,-2]=[-.2,.6,-.1,-.03,.04,-.01]
                        wrench[1,9]=[.25,-.1,.2,-.02,.03,.01]
                        d.acceleration.xfrc.assign(wrench)
                    else:
                        control=f.ctrl.numpy()
                        control[:,0]=1.e6 if case=='control_limits' else np.nan
                        f.ctrl.assign(control)
                    local_seed=snapshot()
                    if case!='badctrl_fallback':
                        wp.capture_launch(reference_graph,stream=env.stream)
                        wp.synchronize_stream(env.stream);reference=region()
                    restore(local_seed)
                    with wp.ScopedCapture(stream=env.stream) as cap:
                        candidate.enqueue_motion();candidate.enqueue_force()
                    wp.capture_launch(cap.graph,stream=env.stream);wp.synchronize_stream(env.stream)
                    if case=='badctrl_fallback':
                        assert candidate.force_ready.numpy().tolist()==[0,0]
                        report['force_inputs'][case]=dict(ready=[0,0],passed=True)
                    else:
                        assert candidate.force_ready.numpy().tolist()==[1,1]
                        checks=compare(reference,region())
                        report['force_inputs'][case]=dict(checks=checks,passed=all(x['passed'] for x in checks))
                        assert all(x['passed'] for x in checks),checks
                    save(path,report);event('force_input_passed',case=case)
            original={name:bytes(x.arguments) for name,x in implementations}
            cases=[('ordinary',False,False,20),('policy_post',True,True,1)]
            if args.task==4:cases.append(('mixed_cache_miss',False,True,5))
            if args.producer=='force':cases.append(('badctrl',False,False,1))
            for case,policy,post,steps in cases:
                end={};counts={}
                for name,x in implementations:
                    restore(seed);x.arguments=type(x.arguments).from_buffer_copy(original[name])
                    if case=='badctrl':
                        control=f.ctrl.numpy();control[:,0]=np.nan;f.ctrl.assign(control)
                    if case=='mixed_cache_miss':x.arguments.position.cached=force_cached.ptr
                    x.device_arguments.assign(np.frombuffer(bytes(x.arguments),np.uint8).copy())
                    with wp.ScopedCapture(stream=env.stream) as cap:x._enqueue(policy,post)
                    before=counters()
                    for _ in range(steps):wp.capture_launch(cap.graph,stream=env.stream)
                    wp.synchronize_stream(env.stream);after=counters();end[name]=endpoint()
                    counts[name]={k:(after[k]-before[k]).tolist() for k in before if k!='error'}
                    assert not np.any(after['error']),(case,name,after)
                    assert counts[name]['integrated']==[steps,steps]
                checks=compare(end['baseline'],end['candidate'])
                report['cases'][case]=dict(checks=checks,counters=counts);save(path,report)
                assert all(x['passed'] for x in checks),checks
                # Numerical stopping choices can change, but invocation count
                # and physical flow remain exact for these replay cases.
                assert counts['baseline']==counts['candidate'],counts
                if args.verify_native_workspace:
                    native_checks=compare(end['baseline'],end['native_workspace'])
                    report['cases'][case]['native_workspace_checks']=native_checks
                    save(path,report)
                    assert all(x['passed'] for x in native_checks),native_checks
                    assert counts['baseline']==counts['native_workspace'],counts
                event('substeps_passed',case=case,max_abs=max(x['max_abs'] for x in checks))
            if args.performance:
                restore(seed);graphs={};device_states={};initial_counts=counters()
                for name,x in (('baseline',base),('candidate',candidate)):
                    x.arguments=type(x.arguments).from_buffer_copy(original[name])
                    x.device_arguments.assign(np.frombuffer(bytes(x.arguments),np.uint8).copy())
                    with wp.ScopedCapture(stream=env.stream) as cap:
                        for sub in range(env.runtime.substeps):
                            x._enqueue(sub==0,sub==env.runtime.substeps-1)
                    graphs[name]=cap.graph
                    restore(seed)
                    for _ in range(5):wp.capture_launch(cap.graph,stream=env.stream)
                    if args.host_state:
                        # Pinned owners permit asynchronous copies on this
                        # stream. Downloading each array with numpy() would
                        # introduce thousands of host/device rendezvous.
                        device_states[name]=[wp.empty_like(a,device='cpu',pinned=True) for a in arrays]
                        for a,s in zip(arrays,device_states[name]):wp.copy(s,a,stream=env.stream)
                    else:
                        with wp.ScopedStream(env.stream):
                            device_states[name]=[wp.clone(a) for a in arrays]
                starts={k:[wp.Event(env.stream.device,enable_timing=True) for _ in range(20)] for k in graphs}
                stops={k:[wp.Event(env.stream.device,enable_timing=True) for _ in range(20)] for k in graphs}
                report['gpu_before']=gpu_sample()
                # Each version has its own continuing 20-control-step state.
                # Restore/save copies are outside events; alternate order to
                # reduce bias from a changing co-tenant load.
                for i in range(20):
                    for name in (('baseline','candidate') if i%2==0 else ('candidate','baseline')):
                        for a,s in zip(arrays,device_states[name]):wp.copy(a,s,stream=env.stream)
                        env.stream.record_event(starts[name][i]);wp.capture_launch(graphs[name],stream=env.stream)
                        env.stream.record_event(stops[name][i])
                        for a,s in zip(arrays,device_states[name]):wp.copy(s,a,stream=env.stream)
                wp.synchronize_stream(env.stream);report['gpu_after']=gpu_sample()
                for name in graphs:
                    ms=[wp.get_event_elapsed_time(x,y,synchronize=False) for x,y in zip(starts[name],stops[name])]
                    report['performance']['control_'+name]=dict(ms=ms,mean_ms=float(np.mean(ms)),median_ms=float(np.median(ms)),std_ms=float(np.std(ms)),
                        warmup=5,steps=20,physical_substeps_per_control=env.runtime.substeps,
                        restore_save_inside_events=False,continuing_per_version_state=True,
                        snapshot_location='host_pinned' if args.host_state else 'device')
                report['performance_work']={}
                for name in graphs:
                    for a,s in zip(arrays,device_states[name]):wp.copy(a,s,stream=env.stream)
                    wp.synchronize_stream(env.stream);final_counts=counters()
                    work={k:(final_counts[k]-initial_counts[k]).tolist() for k in final_counts if k!='error'}
                    work['final_error']=final_counts['error'].tolist()
                    if args.producer=='force':
                        implementation=base if name=='baseline' else candidate
                        handoff=getattr(implementation,'force_ready',None)
                        work['force_ready']=handoff.numpy().tolist() if handoff is not None else None
                        if name=='candidate' and candidate.force_threads:
                            expected=2 if candidate.resources.get('force_after_controller') else 1
                            assert work['force_ready']==[expected,expected],work
                    report['performance_work'][name]=work
                    assert work['integrated']==[25*env.runtime.substeps]*2,work
                    assert not np.any(final_counts['error']),work
                a=report['performance']['control_baseline']['mean_ms'];b=report['performance']['control_candidate']['mean_ms']
                report['control_speedup']=a/b;event('performance',speedup=a/b,baseline_ms=a,candidate_ms=b)
            assert not [x for x in scope['events']+env.build_report['events'] if x.get('commands_executed')]
            report.update(passed=True,runtime_compilations=0);save(path,report)
            if args.newton_split:
                # Save reusable regional inputs while this costly environment
                # is alive. The export is outside all correctness/timing gates.
                restore(seed)
                candidate.arguments=type(candidate.arguments).from_buffer_copy(original['candidate'])
                candidate.device_arguments.assign(np.frombuffer(bytes(candidate.arguments),np.uint8).copy())
                try:
                    report['stage_fixture']=export_stage_fixture(
                        args.output/f'task{args.task:02d}-stage-inputs.npz',env,candidate,arrays)
                except Exception as error:
                    report['stage_fixture']=dict(exported=False,error=repr(error))
                save(path,report)

            event('passed',task=args.task,output=str(path));return 0
        finally:env.close()
