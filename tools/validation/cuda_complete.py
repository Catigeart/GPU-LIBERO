"""Source-bound FP32 reference export and independent Warp-free trajectories."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import inspect
import json
import subprocess
import sys
import time
import numpy as np
from .common import actions,select_tasks

FIELDS=('qpos','qvel','qacc','qacc_warmstart','ctrl','time','xpos','xquat','site_xpos','site_xmat',
        'qfrc_bias','qfrc_smooth','qfrc_constraint','cfrc_ext','cfrc_int','sensordata')


def add_arguments(parser):
    parser.add_argument('--build-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--tasks',nargs='+',default=['all'])
    parser.add_argument('--reference-export',action='store_true')
    parser.add_argument('--worlds',type=int,default=2)
    parser.add_argument('--warmup',type=int,default=5)
    parser.add_argument('--steps',type=int,default=20)
    parser.add_argument('--performance',action='store_true')
    parser.add_argument('--repeat-original',action='store_true',help='untimed original-against-original diagnostic, not a performance repeat')
    parser.add_argument('--resume',action='store_true',help='reuse only complete source/protocol/output-bound task records')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,default=lambda x:x.item())+'\n')
    temporary.replace(path)


def signatures(candidate=False):
    from libero.gpu.host.paths import GPU_ROOT,REPO_ROOT
    paths=list((GPU_ROOT/'host/backends/cuda_fp32' if candidate else GPU_ROOT/'host/backends/mjwarp').glob('*.py'))
    for folder in ('dynamics','newton','collision'):paths+=list((GPU_ROOT/'host/runtime'/folder).glob('mjwarp*.py'))
    paths+=list((GPU_ROOT/'host/adapters/observations').glob('*.py'))
    paths += [GPU_ROOT/n for n in ('__init__.py','precision.py','host/runtime/cuda_api.py','host/runtime/dynamics/__init__.py',
        'host/adapters/environment/batch_env.py','model/factory.py')]
    paths+=list((GPU_ROOT/'cuda').rglob('mjwarp*.cu'))+list((GPU_ROOT/'cuda').rglob('mjwarp*.cuh'))
    if candidate:
        paths+=list((GPU_ROOT/'cuda/runtime').glob('*.cu'))
        paths+=list((GPU_ROOT/'cuda/fp32').glob('*.cu'))+list((GPU_ROOT/'cuda/fp32').glob('*.cuh'))
        paths+=list((GPU_ROOT/'cmake').glob('FP32*.cmake'))
        paths += [GPU_ROOT/n for n in ('host/backends/selection.py','data/cpu_scene_input_source_pin.json',
            'cmake/CudaRuntime.cmake','cmake/CudaBridges.cmake','cmake/FP32Fusion.cmake','CMakeLists.txt')]
    return {str(p.relative_to(REPO_ROOT)):digest(p) for p in sorted(set(paths))}


def gpu_state():
    p=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],capture_output=True,text=True)
    return p.stdout.strip() if p.returncode==0 else p.stderr.strip()


def snapshot(owner):return {name:getattr(owner.data,name).numpy() for name in FIELDS}


def rollout(owner,args,mode):
    pairs=[];stream=owner.stream
    if mode=='reference':
        import warp as wp
        def event():return wp.Event(device=stream.device,enable_timing=True)
        def record(e):wp.record_event(e,stream)
        def elapsed(a,b):return float(wp.get_event_elapsed_time(a,b,synchronize=False))
        synchronize=lambda:wp.synchronize_stream(stream)
    else:
        import ctypes as ct
        def event():
            e=ct.c_void_p();stream.runtime.call('event_create',ct.byref(e));return e
        def record(e):stream.runtime.call('event_record',e,stream.cuda_stream)
        def elapsed(a,b):
            value=ct.c_float();stream.runtime.call('event_elapsed',a,b,ct.byref(value));return value.value
        synchronize=stream.synchronize
    for k in range(args.warmup):owner.runtime.enqueue_action(actions(k,args.worlds))
    synchronize();begin=time.perf_counter()
    try:
        for k in range(args.steps):
            if args.performance:
                first,last=event(),event();pairs.append((first,last));record(first)
            owner.runtime.enqueue_action(actions(k+args.warmup,args.worlds))
            if args.performance:record(last)
        synchronize();wall=time.perf_counter()-begin
        values=[elapsed(a,b) for a,b in pairs]
        performance=None if not values else dict(control_ms=values,mean_control_ms=float(np.mean(values)),
            environment_steps_per_second=args.worlds*args.steps/(sum(values)/1000),wall_seconds=wall)
        return snapshot(owner),performance
    finally:
        if mode!='reference':
            for a,b in pairs:stream.runtime.call('event_destroy',a);stream.runtime.call('event_destroy',b)


def reference_task(args,task,folder):
    if (folder/'reference.json').exists():raise RuntimeError('Preserve completed reference; choose a new output')
    import warp as wp
    from libero.libero.envs import GpuBatchEnv
    from libero.libero import get_libero_path
    from libero.gpu.host.build.compilation import compilation_scope
    from .mjwarp_complete import ReferenceOwner
    from .mjwarp_public_forward import ROOTS
    folder.mkdir(parents=True,exist_ok=True);signature=signatures();env=raw=None;result={};timings={};initial={}
    before=gpu_state()
    try:
        with compilation_scope(True) as preparation:
            env=GpuBatchEnv(str(Path(get_libero_path('bddl_files'))/task['key']),args.worlds,
                physics_engine='mjwarp',precision='fp32',build_root=args.build_root,allow_compile=True,
                initialize=False,ignore_done=True,use_camera_obs=False,mjwarp_options=dict(preset='cuda_libero',fast_compile=True))
            bank=env.initial_states[np.arange(args.worlds)%len(env.initial_states)]
            env.reset(init_states=bank);env.runtime.prepare_control_graph()
            raw=ReferenceOwner(env);raw.initialize(env);raw.runtime.prepare_control_graph()
            for label,owner in (('cuda_reference',env),('upstream',raw)):
                for name in ROOTS:initial[label+'.'+name]=getattr(owner.data,name).numpy()
                for name,value in owner.runtime.controller.buffers.items():initial[label+'.controller.'+name]=value.numpy()
            assert np.all(env.runtime.integration_count.numpy()==0) and np.all(raw.runtime.integration_count.numpy()==0)
        with compilation_scope(False) as execution:
            for label,owner in (('cuda_reference',env),('upstream',raw)):
                values,timing=rollout(owner,args,'reference');result.update({label+'.'+k:v for k,v in values.items()});timings[label]=timing
            if args.repeat_original:
                with wp.ScopedStream(env.stream):
                    for name in ROOTS:getattr(raw.data,name).assign(initial['upstream.'+name])
                    for name,value in raw.runtime.controller.buffers.items():value.assign(initial['upstream.controller.'+name])
                    raw.observer.reset(env._selection,stream=env.stream)
                diagnostic_args=SimpleNamespace(**dict(vars(args),performance=False))
                values,_=rollout(raw,diagnostic_args,'reference');result.update({'upstream_repeat.'+k:v for k,v in values.items()})
            env._check()
        np.savez(folder/'initial.npz',**initial,bank=bank);np.savez(folder/'endpoints.npz',**result)
        report=dict(task=task['key'],worlds=args.worlds,warmup=args.warmup,steps=args.steps,
            signatures=signature,tool_sha256=digest(__file__),identity=env.identity,
            initial_sha256=digest(folder/'initial.npz'),endpoint_sha256=digest(folder/'endpoints.npz'),
            offline_compilations=preparation['events'],runtime_compilations=execution['events'],timings=timings,
            repeat_original=args.repeat_original,competition_before=before,competition_after=gpu_state())
        if execution['events']:raise RuntimeError('Runtime compilation detected')
        save(folder/'reference.json',report)
        return dict(task=task['key'],exported=True,runtime_compilations=0)
    finally:
        if raw is not None:raw.close()
        if env is not None:env.close()


def independent_task(args,task,folder):
    from .cuda_runtime import ForbidWarp
    sys.meta_path.insert(0,ForbidWarp())
    from libero.libero.envs import GpuBatchEnv
    from libero.libero import get_libero_path
    from libero.gpu.host.build.compilation import compilation_scope
    from .mjwarp_policy import compare_replays
    source=args.reference/(task['suite']+'-'+str(task['task_id']))
    record=json.loads((source/'reference.json').read_text())
    if record['signatures']!=signatures():raise RuntimeError('Reference dependencies changed')
    if record['initial_sha256']!=digest(source/'initial.npz') or record['endpoint_sha256']!=digest(source/'endpoints.npz'):raise RuntimeError('Reference arrays changed')
    if any(record[k]!=getattr(args,k) for k in ('worlds','warmup','steps')):raise ValueError('Reference protocol differs')
    if (folder/'summary.json').exists():raise RuntimeError('Preserve completed trajectory; choose new output')
    before=gpu_state();env=None
    try:
        with compilation_scope(False) as execution:
            phase=time.perf_counter()
            env=GpuBatchEnv(str(Path(get_libero_path('bddl_files'))/task['key']),args.worlds,
                physics_engine='cuda_fp32',precision='fp32',build_root=args.build_root,allow_compile=False,
                initialize=False,ignore_done=True,use_camera_obs=False)
            creation_seconds=time.perf_counter()-phase
            phase=time.perf_counter()
            with np.load(source/'initial.npz',allow_pickle=False) as a:
                bank=a['bank'].copy();env.reset(init_states=bank)
                for name in ('time','qpos','qvel','qacc','qacc_warmstart','ctrl','qfrc_applied','xfrc_applied'):
                    if not np.array_equal(getattr(env.data,name).numpy(),a['cuda_reference.'+name]):
                        if name not in ('qacc',):raise RuntimeError('Common initial root differs: '+name)
            reset_seconds=time.perf_counter()-phase
            phase=time.perf_counter();env.runtime.prepare_control_graph();env.stream.synchronize()
            capture_seconds=time.perf_counter()-phase
            values,timing=rollout(env,args,'independent');env._check()
            public_obs=env._observations(tuple(range(args.worlds)))
            public_success=env.check_success()
            public_finite=all(np.isfinite(v).all() and len(v)==args.worlds for v in public_obs.values())
            if not public_finite or public_success.shape!=(args.worlds,):
                raise RuntimeError('Independent public observation/predicate contract failed')
        with np.load(source/'endpoints.npz',allow_pickle=False) as a:
            def fields(prefix):return {k:a[prefix+'.'+k].copy() for k in FIELDS}
            old,raw=fields('cuda_reference'),fields('upstream')
            repeated=fields('upstream_repeat') if record['repeat_original'] else raw
        def state(v):return dict(qpos=v['qpos'],qvel=v['qvel'],controller_output=v['ctrl'],time=v['time'])
        current_checks=compare_replays(state(values),state(old),state(old))
        original_checks=compare_replays(state(values),state(raw),state(repeated))
        if execution['events']:raise RuntimeError('Unexpected compilation')
        if any(k.split('.')[0] in ('warp','mujoco_warp') for k in sys.modules):raise RuntimeError('Warp leaked into independent process')
        folder.mkdir(parents=True,exist_ok=True);np.savez(folder/'endpoints.npz',**values)
        report=dict(task=task['key'],passed=all(c['passed'] for c in current_checks),
            original_fp32_passed=all(c['passed'] for c in original_checks),checks=current_checks,original_checks=original_checks,
            full_state_scope='all qpos/qvel, closed-loop controller outputs and time; acceleration/forces remain separate diagnostics',
            endpoint_sha256=digest(folder/'endpoints.npz'),reference_sha256=digest(source/'reference.json'),
            candidate_sources=signatures(True),runtime_compilations=execution['events'],warp_imported=False,
            worlds=args.worlds,warmup=args.warmup,steps=args.steps,performance=timing,
            competition_before=before,competition_after=gpu_state(),integrations=env.runtime.integration_count.numpy().tolist())
        report['phases_seconds']=dict(environment_creation=creation_seconds,reset=reset_seconds,control_graph=capture_seconds)
        report['public_boundary']=dict(passed=True,observation_shapes={k:list(v.shape) for k,v in public_obs.items()},success=public_success.tolist(),
            constraint_rows=env.data.nefc.numpy().tolist(),solver_iterations=env.data.solver_niter.numpy().tolist(),
            contacts=env._snapshot(('ncon',))['ncon'].tolist())
        save(folder/'summary.json',report)
        return {k:report[k] for k in ('task','passed','original_fp32_passed','warp_imported','integrations')}
    finally:
        if env is not None:env.close()


def main(args):
    from libero.gpu import precision
    precision.select('fp32')
    if args.worlds!=2:raise ValueError('This qualification protocol starts with N=2')
    tasks=select_tasks(json.loads((args.build_root/'TASK_INSTANCES.json').read_text())['tasks'],args.tasks)
    args.output.mkdir(parents=True,exist_ok=True);rows=[]
    for task in tasks:
        folder=args.output/(task['suite']+'-'+str(task['task_id']))
        record_path=folder/('reference.json' if args.reference_export else 'summary.json')
        if args.resume and record_path.is_file():
            record=json.loads(record_path.read_text())
            if any(record[k]!=getattr(args,k) for k in ('worlds','warmup','steps')):
                raise ValueError('Resume protocol differs: '+str(record_path))
            if args.reference_export:
                if record['signatures']!=signatures() or record['repeat_original']!=args.repeat_original:
                    raise ValueError('Resume reference sources/diagnostics changed')
                for name,key in (('initial.npz','initial_sha256'),('endpoints.npz','endpoint_sha256')):
                    if digest(folder/name)!=record[key]:raise ValueError('Resume reference arrays changed')
                if bool(record['timings']['cuda_reference'])!=args.performance:
                    raise ValueError('Resume timing protocol differs')
                row=dict(task=task['key'],exported=True,runtime_compilations=0)
            else:
                if record['candidate_sources']!=signatures(True) or digest(folder/'endpoints.npz')!=record['endpoint_sha256']:
                    raise ValueError('Resume candidate sources/arrays changed')
                if digest(args.reference/(task['suite']+'-'+str(task['task_id']))/'reference.json')!=record['reference_sha256']:
                    raise ValueError('Resume independent reference changed')
                if bool(record['performance'])!=args.performance:raise ValueError('Resume timing differs')
                row={k:record[k] for k in ('task','passed','original_fp32_passed','warp_imported','integrations')}
        else:
            row=reference_task(args,task,folder) if args.reference_export else independent_task(args,task,folder)
        rows.append(row);print(json.dumps(row),flush=True)
        save(args.output/'index.json',rows)
        import gc
        gc.collect()
    return 0 if args.reference_export or all(row['passed'] and row['original_fp32_passed'] for row in rows) else 2
