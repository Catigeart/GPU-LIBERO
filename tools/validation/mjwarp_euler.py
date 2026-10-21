"""Frozen N=2 Euler leaf qualification, independent of collision/Newton; no timing."""
import argparse,hashlib,json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from .mjwarp_euler_reference import reference
from .mjwarp_motion_reference import check_arrays
PHASES=('derivative','diagonal','velocity','position','composed')


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def add_arguments(p):
    for name in ('build-root','inputs','output','library'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--cpu-reference',action='store_true')
    return p


def cases(args):
    result=[]
    for p in sorted(args.inputs.glob('*.npz')):
        with np.load(p,allow_pickle=False) as f:c={k:f[k].copy() for k in f.files}
        for key in ('nq','nv','njnt','nC'):c[key]=int(c[key])
        c['name']=p.stem;result.append(c)
    if not result:raise ValueError('No frozen Euler inputs')
    return result


def current_case(original,updated):
    c={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in original.items()}
    if updated:
        c['qvel']*=np.float32(-.7);c['acceleration']*=np.float32(1.3)
        c['damping']*=np.float32(.8);c['dampingpoly']*=np.float32(1.2)
        c['timestep']*=np.float32(.9);c['matrix']*=np.float32(1.1)
        c['derivative']*=np.float32(.6)
        # Preserve quaternion fields; mutate scalar/translation coordinates.
        for kind,q in zip(c['joint_type'],c['qposadr']):
            width=3 if kind==0 else 0 if kind==1 else 1
            c['qpos'][:,q:q+width]+=np.float32(.013)
    return c


def input_identity(args):return {p.name:digest(p) for p in sorted(args.inputs.glob('*')) if p.is_file()}


def cpu_reference(args):
    import mujoco
    checks=[]
    for original in cases(args):
        for updated in (False,True):
            c=current_case(original,updated);expected,_=reference(c,'composed');dt=c['timestep']
            qv=c['qvel'].astype(float)+c['acceleration'].astype(float)*dt[np.arange(2)%len(dt),None]
            q=c['qpos'].astype(float).copy();degenerate=[]
            for w in range(2):
                for j,(kind,p,d) in enumerate(zip(c['joint_type'],c['qposadr'],c['dofadr'])):
                    p,d=int(p),int(d)
                    if kind==0:q[w,p:p+3]+=float(dt[w%len(dt)])*qv[w,d:d+3];p+=3;d+=3
                    if kind<2:
                        if np.linalg.norm(q[w,p:p+4])==0:
                            # Pinned Warp's zero-quaternion storage fallback is
                            # intentionally different from CPU2.3.7; checked
                            # against the raw pinned GPU comparator instead.
                            degenerate.extend((w,x) for x in range(p,p+4));continue
                        mujoco.mju_normalize4(q[w,p:p+4])
                        mujoco.mju_quatIntegrate(q[w,p:p+4],qv[w,d:d+3],float(dt[w%len(dt)]))
                    else:q[w,p]+=float(dt[w%len(dt)])*qv[w,d]
            mask=np.ones(q.shape,bool)
            for w,p in degenerate:mask[w,p]=False
            error=float(abs(q[mask]-expected['qpos'][mask]).max(initial=0.))
            limit=4096*np.finfo(float).eps*max(1.,float(abs(q[mask]).max(initial=0.)))
            checks.append(dict(name=c['name'],updated=updated,field='CPU quaternion/scalar integration',error=error,limit=limit,degenerate_components_excluded=len(degenerate),passed=bool(error<=limit)))
    result=dict(passed=all(x['passed'] for x in checks),checks=checks,version=mujoco.__version__,inputs=input_identity(args),
        tool_sha256=digest(__file__),reference_sha256=digest(Path(__file__).with_name('mjwarp_euler_reference.py')),newton_solves=0)
    (args.output/'cpu-oracle.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'],cpu_checks=len(checks))),flush=True);return 0 if result['passed'] else 2


def construct(c,stream):
    import warp as wp
    from mujoco_warp._src import types
    arr=lambda a,dtype:wp.array(np.ascontiguousarray(a),dtype=dtype,device=stream.device)
    m=NS(nv=c['nv'],nq=c['nq'],njnt=c['njnt'],nC=c['nC'],na=0,
        opt=NS(integrator=types.IntegratorType.EULER,enableflags=0,disableflags=0,timestep=arr(c['timestep'],float)))
    for field,key,dtype in (('jnt_type','joint_type',int),('jnt_qposadr','qposadr',int),('jnt_dofadr','dofadr',int),('dof_jntid','dof_joint',int),
            ('M_rowadr','row',int),('M_rownnz','nnz',int),('M_colind','column',int),('dof_damping','damping',float),('dof_dampingpoly','dampingpoly',wp.vec2)):
        setattr(m,field,arr(c[key],dtype))
    d=NS(nworld=2,qpos=arr(c['qpos'],float),qvel=arr(c['qvel'],float),M=arr(c['matrix'],float),
        qacc=wp.full((2,m.nv),37.,dtype=float),qacc_warmstart=wp.full((2,m.nv),-23.,dtype=float))
    return m,d


class Replay:
    def __init__(self,c,args,stream,phase):
        import warp as wp
        from mujoco_warp._src import forward
        from libero.gpu.host.runtime.dynamics.mjwarp_euler import EulerNumerics
        self.case,self.stream,self.phase=c,stream,phase;self.model,self.data=m,d=construct(c,stream)
        arr=lambda a:wp.array(np.ascontiguousarray(a),dtype=float,device=stream.device)
        self.acceleration=arr(c['acceleration']);self.outputs={};self.graphs={}
        keys=PHASES[:-1] if phase=='composed' else (phase,)
        names=[dict(derivative='derivative',diagonal='matrix',velocity='qvel',position='qpos')[k] for k in keys]
        for mode in ('upstream','cuda'):
            w=NS(damping=arr(c['derivative']),matrix=arr(c['matrix']),acceleration=self.acceleration)
            d.qvel=arr(c['qvel']);d.qpos=arr(c['qpos']);self.outputs[mode]={k:dict(derivative=w.damping,matrix=w.matrix,qvel=d.qvel,qpos=d.qpos)[k] for k in names}
            if mode=='cuda':self.binding=EulerNumerics(m,d,args.library,stream,workspace=w)
            with wp.ScopedCapture(stream=stream) as cap:
                for k in keys:
                    if mode=='cuda':self.binding.enqueue(PHASES.index(k))
                    elif k=='derivative':wp.launch(forward._compute_damping_deriv,dim=(2,m.nv),inputs=[m.dof_damping,m.dof_dampingpoly,d.qvel],outputs=[w.damping])
                    elif k=='diagonal':wp.launch(forward._euler_damp_qfrc,dim=(2,m.nv),inputs=[m.opt.timestep,m.M_rownnz,m.M_rowadr,w.damping],outputs=[w.matrix])
                    elif k=='velocity':wp.launch(forward._next_velocity,dim=(2,m.nv),inputs=[m.opt.timestep,d.qvel,self.acceleration,1.],outputs=[d.qvel])
                    else:wp.launch(forward._next_position,dim=(2,m.njnt),inputs=[m.opt.timestep,m.jnt_type,m.jnt_qposadr,m.jnt_dofadr,d.qpos,d.qvel,1.],outputs=[d.qpos])
            self.graphs[mode]=cap.graph
            # Retain intermediate/input owners even when this phase doesn't output them.
            self.outputs[mode+'-all']=dict(derivative=w.damping,matrix=w.matrix,qvel=d.qvel,qpos=d.qpos)

    def check(self,updated):
        import warp as wp
        c=current_case(self.case,updated);m,d=self.model,self.data
        for field,key in (('dof_damping','damping'),('dof_dampingpoly','dampingpoly')):getattr(m,field).assign(c[key])
        m.opt.timestep.assign(c['timestep']);self.acceleration.assign(c['acceleration']);d.M.assign(c['matrix'])
        before={k:a.numpy().view(np.uint8).copy() for k,a in self.binding.arrays.items() if k not in self.outputs['cuda']}
        holds={k:getattr(d,k).numpy().view(np.uint8).copy() for k in ('M','qacc','qacc_warmstart')}
        expected,bounds=reference(c,self.phase);checks=[];values={}
        for mode,graph in self.graphs.items():
            for k,a in self.outputs[mode+'-all'].items():a.assign(c[k])
            # Capture each current input, after restoring state/workspace.
            if mode=='cuda':before={k:a.numpy().view(np.uint8).copy() for k,a in self.binding.arrays.items() if k not in self.outputs['cuda']}
            wp.capture_launch(graph,stream=self.stream);wp.synchronize_stream(self.stream)
            values[mode]={k:a.numpy() for k,a in self.outputs[mode].items()}
            checks.extend(dict(mode=mode,**x) for x in check_arrays(values[mode],expected,bounds))
        held=all(np.array_equal(a.numpy().view(np.uint8),before[k]) for k,a in self.binding.arrays.items() if k in before)
        held &= all(np.array_equal(getattr(d,k).numpy().view(np.uint8),a) for k,a in holds.items())
        return dict(name=c['name'],phase=self.phase,updated=updated,inputs_held=bool(held),checks=checks,passed=bool(held and all(x['passed'] for x in checks)),
            max_difference={k:float(abs(values['cuda'][k]-values['upstream'][k]).max()) for k in expected}),values


def signature(args):
    from libero.gpu.host.backends.mjwarp.build import source_files
    sources=source_files(include_euler=True)
    sources={k:v for k,v in sources.items() if k.startswith('upstream/') or 'mjwarp_euler' in k or k in (
        'host/runtime/dynamics/mjwarp_motion.py','host/runtime/dynamics/mjwarp_passive.py','cmake/MjwarpEuler.cmake','project_source_identity','data/mjwarp_source_pin.json')}
    helpers={name:digest(Path(__file__).with_name(name)) for name in ('mjwarp_motion_reference.py','mjwarp_passive_reference.py')}
    return dict(sources=sources,helpers=helpers,inputs=input_identity(args),library_sha256=digest(args.library),tool_sha256=digest(__file__),reference_sha256=digest(Path(__file__).with_name('mjwarp_euler_reference.py')))


def main(argv=None,*,args=None):
    if args is None:args=add_arguments(argparse.ArgumentParser(description=__doc__)).parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.cpu_reference:return cpu_reference(args)
    from libero.gpu import precision
    precision.select('fp32')
    import warp as wp
    from warp._src.build import init_kernel_cache
    from libero.gpu.host.backends.mjwarp.model import require_upstream
    from libero.gpu.host.build.compilation import compilation_scope,install_warp_guard
    from validation.common import save_json
    require_upstream();init_kernel_cache(args.build_root/'mjwarp/warp-cache');install_warp_guard();wp.init();stream=wp.Stream('cuda:0')
    sig=signature(args);prepared=args.output/'preparation.json'
    if not args.prepare:
        if not prepared.is_file() or json.loads(prepared.read_text())['signature']!=sig:raise RuntimeError('Prepare exact Euler operators offline first')
        cpu=json.loads((args.output/'cpu-oracle.json').read_text())
        if not cpu['passed'] or any(cpu[k]!=sig[k] for k in ('inputs','tool_sha256','reference_sha256')):raise RuntimeError('Prepare matching CPU Euler oracle first')
    with compilation_scope(args.prepare) as scope,wp.ScopedStream(stream):
        replays=[Replay(c,args,stream,p) for p in PHASES for c in cases(args)]
        if args.prepare:
            wp.synchronize_stream(stream);save_json(prepared,dict(signature=sig,compilation_events=scope['events'],newton_solves=0))
            print(json.dumps(dict(prepared=True,captures=len(replays)*2,compilations=len(scope['events']))),flush=True);return 0
        rows=[]
        for replay in replays:
            for updated in (False,True):
                row,values=replay.check(updated);rows.append(row)
                np.savez(args.output/(row['name']+'-'+row['phase']+('-updated' if updated else '')+'-output.npz'),**{mode+'.'+k:v for mode,vv in values.items() for k,v in vv.items()})
                print(json.dumps({k:row[k] for k in ('name','phase','updated','passed')}),flush=True)
                if not row['passed']:break
            if not row['passed']:break
        passed=all(r['passed'] for r in rows)
        save_json(args.output/'summary.json',dict(passed=passed,signature=sig,cases=rows,runtime_compilations=scope['events'],newton_solves=0,performance=None,
            policy='independent FP64 arithmetic, norm/division/quaternion sensitivity and sin/cos argument bounds; fixed8x safety; exact input holds',worlds=2))
        print(json.dumps(dict(passed=passed,cases=len(rows))),flush=True);return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())
