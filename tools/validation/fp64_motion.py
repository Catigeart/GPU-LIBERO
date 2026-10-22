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
    from libero.gpu.host.runtime.step.physics import _reference_mask
    install_warp_guard();wp.init()
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/f'task{args.task:02d}.json'
    report=dict(task=args.task,worlds=2,region={},cases={},performance={},passed=False)
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
            base=FusedSubstep(env.runtime,args.build_root/'cmake-control',allow_build=False,fast_compile=True)
            old=ct.CDLL(str(args.baseline.resolve()))
            size=old.flash_libero_substep_arguments_size;size.restype=ct.c_size_t
            assert size()==ct.sizeof(base.arguments)
            configure=old.flash_libero_substep_configure;configure.argtypes=[ct.c_void_p]*2
            with wp.ScopedDevice(env.stream.device):assert configure(ct.byref(ct.c_int()),ct.byref(ct.c_int()))==0
            base.library=old
            base.launch_post_entry=old.flash_libero_launch_substep_with_post
            base.launch_post_entry.argtypes=[ct.c_void_p]+[ct.c_int]*4+[ct.c_void_p]
            base.launch_entry=old.flash_libero_launch_substep
            base.launch_entry.argtypes=[ct.c_void_p]+[ct.c_int]*3+[ct.c_void_p]
            base.select_entry=old.flash_libero_select_substep_continuation
            base.select_entry.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ct.c_void_p,ct.c_void_p]
            candidate=PeeledMotionSubstep(env.runtime,args.build_root/'cmake-control',
                allow_build=False,fast_compile=True,motion_threads=512)
            assert candidate.eligible,candidate.resources
            report['resources']=candidate.resources
            report['libraries']={name:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                for name,p in (('baseline',args.baseline),('candidate',Path(candidate.library._name)))}
            f=env.runtime.forward;d=f.d
            force_cached=wp.array(np.array([1,0],np.int32),dtype=wp.int32,device=env.stream.device)
            arrays=arrays_of(env._physics.bundle,base,candidate,force_cached)
            def snapshot():
                wp.synchronize_stream(env.stream)
                return [a.numpy().copy() for a in arrays]
            def restore(values):
                with wp.ScopedStream(env.stream):
                    for a,v in zip(arrays,values):a.assign(v)
                wp.synchronize_stream(env.stream)
            def region():
                result={name:getattr(d,name).numpy().copy() for name in ('cvel','cdof_dot','cacc','frc','qfrc_bias')}
                result['passive']=d.passive.qfrc.numpy().copy();return result
            def endpoint():
                e=env.runtime.euler
                out={name:getattr(e,name).numpy().copy() for name in
                     ('qpos','qvel','time','qacc','qacc_warmstart','qfrc_constraint')}
                out.update({'dynamics.'+k:v.numpy().copy() for k,v in d.public.items()})
                out.update({'post.'+k:v.numpy().copy() for k,v in env._physics.bundle.post.public.items()})
                return out
            def counters():
                r=env.runtime
                return {k:a.numpy().copy() for k,a in dict(integrated=r.integration_count,
                    newton=f.n.n_attempt,position=f._dataflow_reuse.math,reset=r.reset_events,error=r.first_error).items()}
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
            wp.capture_launch(cap.graph,stream=env.stream);wp.synchronize_stream(env.stream)
            expected=region();assert not np.any(counters()['error'])
            np.savez(args.output/f'task{args.task:02d}-region-reference.npz',
                **expected, qpos=d.qpos.numpy(), qvel=d.qvel.numpy(), model_path=str(Path(task['directory'])/'model.mjb'))
            for threads in (512,256,128):
                restore(prepared);candidate.motion_threads=threads
                with wp.ScopedCapture(stream=env.stream) as cap:candidate.enqueue_motion()
                wp.capture_launch(cap.graph,stream=env.stream);wp.synchronize_stream(env.stream)
                assert candidate.ready.numpy().tolist()==[1,1],candidate.ready.numpy()
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
                    report['performance']['motion_'+str(threads)]=dict(us=us,mean_us=float(np.mean(us)),median_us=float(np.median(us)))
            selected=min((128,256,512),key=lambda n:report['performance']['motion_'+str(n)]['median_us']) if args.performance else 512
            candidate.motion_threads=selected;report['selected_threads']=selected
            report['resources']['motion_threads']=selected
            original={name:bytes(x.arguments) for name,x in (('baseline',base),('candidate',candidate))}
            cases=[('ordinary',False,False,20),('policy_post',True,True,1)]
            if args.task==4:cases.append(('mixed_cache_miss',False,True,5))
            for case,policy,post,steps in cases:
                end={};counts={}
                for name,x in (('baseline',base),('candidate',candidate)):
                    restore(seed);x.arguments=type(x.arguments).from_buffer_copy(original[name])
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
                        restore_save_inside_events=False,continuing_per_version_state=True)
                report['performance_work']={}
                for name in graphs:
                    for a,s in zip(arrays,device_states[name]):wp.copy(a,s,stream=env.stream)
                    wp.synchronize_stream(env.stream);final_counts=counters()
                    work={k:(final_counts[k]-initial_counts[k]).tolist() for k in final_counts if k!='error'}
                    work['final_error']=final_counts['error'].tolist();report['performance_work'][name]=work
                    assert work['integrated']==[25*env.runtime.substeps]*2,work
                    assert not np.any(final_counts['error']),work
                a=report['performance']['control_baseline']['mean_ms'];b=report['performance']['control_candidate']['mean_ms']
                report['control_speedup']=a/b;event('performance',speedup=a/b,baseline_ms=a,candidate_ms=b)
            assert not [x for x in scope['events']+env.build_report['events'] if x.get('commands_executed')]
            report.update(passed=True,runtime_compilations=0);save(path,report)
            event('passed',task=args.task,output=str(path));return 0
        finally:env.close()
