"""Common-input prefix/controller/Newton/step2 qualification without retiming."""
from pathlib import Path
from libero.gpu.host.build.task_catalog import load_catalog
import ctypes as ct
import json
import sys
import numpy as np
from .cuda_complete import digest,save,signatures
from .common import actions,select_tasks


def add_arguments(parser):
    parser.add_argument('--build-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--tasks',nargs='+',default=['libero_spatial:0','libero_90:50','libero_90:79','libero_90:84'])
    parser.add_argument('--reference-export',action='store_true')
    parser.add_argument('--controls',type=int,default=6,help='freeze after this many controls;25 covers the standard5+20 endpoint')
    parser.add_argument('--prefix-coupling',action='store_true',help='also verify rebuilt collision/constraint invariants and a substep using that rebuilt prefix')


def arrays(data):
    return {prefix+name:a for prefix,owner in (('',data),('contact.',data.contact),('efc.',data.efc))
            for name,a in vars(owner).items() if hasattr(a,'ptr') and hasattr(a,'numpy')}


def snapshot(data):return {k:a.numpy() for k,a in arrays(data).items()}


def restore(data,values):
    target=arrays(data)
    for name,value in values.items():
        if name not in target:
            if value.size:raise ValueError('Unrepresented nonempty reference state: '+name)
            continue
        if target[name].numpy().shape!=value.shape:raise ValueError('Reference state shape differs: '+name)
        target[name].assign(value)


def export_task(args,task,folder):
    import warp as wp
    from libero.libero.envs import GpuBatchEnv
    from libero.libero import get_libero_path
    from libero.gpu.host.build.compilation import compilation_scope
    env=None;folder.mkdir(parents=True,exist_ok=True)
    try:
        with compilation_scope(True):
            env=GpuBatchEnv(str(Path(get_libero_path('bddl_files'))/task['key']),2,precision='fp32',physics_engine='mjwarp',
                build_root=args.build_root,initialize=False,allow_compile=True,ignore_done=True,use_camera_obs=False,
                mjwarp_options=dict(preset='cuda_libero',fast_compile=True))
            env.reset(init_states=env.initial_states[:2]);env.runtime.prepare_control_graph()
        results={}
        with compilation_scope(False) as scope,wp.ScopedStream(env.stream):
            for k in range(args.controls):env.runtime.enqueue_action(actions(k,2))
            wp.synchronize_stream(env.stream)
            def capture(name):
                values=snapshot(env.data);np.savez_compressed(folder/(name+'.npz'),**values);results[name]=digest(folder/(name+'.npz'));return values
            capture('prefix-input');env.runtime.physics.step1();capture('prefix-output')
            initial_controller={k:a.numpy() for k,a in env.runtime.controller.buffers.items()}
            np.savez_compressed(folder/'controller-input.npz',**initial_controller);results['controller-input']=digest(folder/'controller-input.npz')
            env.runtime.controller.enqueue(False)
            controller={k:a.numpy() for k,a in env.runtime.controller.buffers.items()}
            np.savez_compressed(folder/'controller-output.npz',**controller);results['controller-output']=digest(folder/'controller-output.npz')
            step2=capture('step2-input')
            env.runtime.physics.actuation();env.runtime.physics.acceleration();capture('qp-input')
            with env.runtime.newton.capture_owners() as owners,wp.ScopedCapture(stream=env.stream) as cap:env.runtime.newton.solve()
            wp.capture_launch(cap.graph,stream=env.stream);capture('qp-output')
            restore(env.data,step2)
            with env.runtime.newton.capture_owners() as owners2,wp.ScopedCapture(stream=env.stream) as cap2:env.runtime.physics.step2()
            wp.capture_launch(cap2.graph,stream=env.stream);capture('step2-output')
        if scope['events']:raise RuntimeError('Unexpected compilation')
        save(folder/'reference.json',dict(task=task['key'],files=results,reference_sources=signatures(),
            tool_sha256=digest(__file__),identity=env.identity,performance=None,controls=args.controls,runtime_compilations=scope['events']))
        return dict(task=task['key'],exported=True)
    finally:
        if env is not None:env.close()


def check_task(args,task,folder):
    from .cuda_runtime import ForbidWarp
    sys.meta_path.insert(0,ForbidWarp())
    from libero.libero.envs import GpuBatchEnv
    from libero.libero import get_libero_path
    from libero.gpu.host.backends.cuda_fp32 import device as cuda
    source=args.reference/(task['suite']+'-'+str(task['task_id']));record=json.loads((source/'reference.json').read_text())
    if record['reference_sources']!=signatures():raise RuntimeError('Reference function dependencies changed')
    if record.get('controls',6)!=args.controls:raise ValueError('Frozen control window differs')
    def read(name):
        path=source/(name+'.npz')
        if digest(path)!=record['files'][name]:raise RuntimeError('Changed frozen function inputs')
        with np.load(path,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
    env=None;rows=[];folder.mkdir(parents=True,exist_ok=True)
    try:
        env=GpuBatchEnv(str(Path(get_libero_path('bddl_files'))/task['key']),2,physics_engine='cuda_fp32',precision='fp32',
            build_root=args.build_root,initialize=False,allow_compile=False,ignore_done=True,use_camera_obs=False)
        with env.stream:
            def compare(stage,actual,expected,names):
                for name in names:
                    a,b=np.asarray(actual[name]),np.asarray(expected[name]);error=float(np.max(abs(a.astype(float)-b.astype(float)),initial=0))
                    bound=1024*np.finfo(np.float32).eps*max(1.,float(np.max(abs(b),initial=0)))
                    if name in ('time','qpos','qvel') and stage=='prefix':bound=0.
                    rows.append(dict(stage=stage,field=name,error=error,bound=bound,
                        passed=bool(a.shape==b.shape and np.isfinite(a).all() and error<=bound)))
            restore(env.data,read('prefix-input'))
            env.runtime.physics.step1();env.stream.synchronize();native_prefix=snapshot(env.data)
            compare('prefix',native_prefix,read('prefix-output'),('qpos','qvel','time','xpos','xquat','xmat','geom_xpos','geom_xmat','site_xpos','site_xmat',
                'subtree_com','cinert','cdof','M','qLD','qLDiagInv','cvel','cdof_dot','qfrc_spring','qfrc_damper','qfrc_passive','qfrc_bias'))
            restore(env.data,read('prefix-output'))
            for k,v in read('controller-input').items():env.runtime.controller.buffers[k].assign(v)
            env.runtime.controller.enqueue(False);env.stream.synchronize()
            compare('controller',{k:a.numpy() for k,a in env.runtime.controller.buffers.items()},read('controller-output'),
                ('pm','pj','pq','pv','bias','pp','pr','pe','mass','jacobian','q','qvel','ee_pos','ee_rot','ee_velocity','ctrl','raw_tau','nullspace_torque'))
            for stage in ('qp','step2'):
                restore(env.data,read(stage+'-input'))
                with cuda.Graph(env.stream,owners=[env.runtime.physics]) as graph:
                    env.runtime.newton.solve() if stage=='qp' else env.runtime.physics.step2()
                graph.launch();env.stream.synchronize();actual=snapshot(env.data);expected=read(stage+'-output')
                compare(stage,actual,expected,('qacc','qfrc_constraint') if stage=='qp' else ('qacc','qacc_warmstart','qvel','qpos','time','qfrc_constraint','cfrc_int','sensordata'))
                np.savez_compressed(folder/(stage+'-output.npz'),**{k:actual[k] for k in ('qacc','qacc_warmstart','qvel','qpos','qfrc_constraint')})
                graph.close()
            if args.prefix_coupling:
                reference_prefix=read('prefix-output')
                compare('collision-counts',native_prefix,reference_prefix,('nefc','ne','nf','nacon','ncollision'))
                def invariants(value):
                    gram=[];affine=[]
                    for w,count in enumerate(value['nefc']):
                        count=int(count);j=value['efc.J'][w,:count,:env.model.nv].astype(float)
                        diagonal=value['efc.D'][w,:count].astype(float)
                        gram.append(j.T@(diagonal[:,None]*j))
                        affine.append(j.T@(diagonal*value['efc.aref'][w,:count]))
                    return dict(gram=np.asarray(gram),affine=np.asarray(affine))
                compare('constraint-invariants',invariants(native_prefix),invariants(reference_prefix),('gram','affine'))
                restore(env.data,native_prefix)
                for k,v in read('controller-input').items():env.runtime.controller.buffers[k].assign(v)
                env.runtime.controller.enqueue(False)
                with cuda.Graph(env.stream,owners=[env.runtime.physics]) as graph:env.runtime.physics.step2()
                graph.launch();env.stream.synchronize()
                compare('coupled-substep',snapshot(env.data),read('step2-output'),('qvel','qpos','time','ctrl'))
                graph.close()
        result=dict(task=task['key'],passed=all(r['passed'] for r in rows),checks=rows,warp_imported=False,
            reference_sha256=digest(source/'reference.json'),candidate_sources=signatures(True),performance=None,prefix_coupling=args.prefix_coupling)
        save(folder/'summary.json',result)
        return {k:result[k] for k in ('task','passed','warp_imported')}|dict(failed=[r for r in rows if not r['passed']])
    finally:
        if env is not None:env.close()


def main(args):
    from libero.gpu import precision
    precision.select('fp32')
    if args.controls<1:raise ValueError('Positive frozen control window required')
    tasks=select_tasks(load_catalog(args.build_root),args.tasks);rows=[]
    for task in tasks:
        row=(export_task if args.reference_export else check_task)(args,task,args.output/(task['suite']+'-'+str(task['task_id'])))
        rows.append(row);print(json.dumps(row),flush=True);save(args.output/'index.json',rows)
        import gc
        gc.collect()
    return 0 if args.reference_export or all(r['passed'] for r in rows) else 2
