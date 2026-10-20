"""N=2 frozen acceleration parity, CPU 2.3.7 wrench oracle and local timing."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_arguments(p):
    for key in ('build-root','mass-library','force-library','catalogue','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--cpu-python',type=Path,help='CPU reference launcher, needed for first offline preparation')
    p.add_argument('--prepare',action='store_true');p.add_argument('--performance',action='store_true')
    p.add_argument('--warmup',type=int,default=5);p.add_argument('--steps',type=int,default=20)
    return p


def freeze(args,stream):
    import warp as wp
    import mujoco_warp as mjw
    from mujoco_warp._src import forward
    from libero.libero import get_libero_path
    from libero.gpu.host.backends.mjwarp.model import build_scene,capacity_budget
    catalogue=json.loads(args.catalogue.read_text())
    for task_id in (0,3):
        path=args.output/f'task{task_id}.npz'
        if path.exists():
            if json.loads(path.with_suffix('.json').read_text())['sha256']!=digest(path):raise RuntimeError('Frozen acceleration input changed')
            continue
        row=next(r for r in catalogue['rows'] if r['suite']=='libero_10' and r['task_id']==task_id)
        built,m,bank,task,identity=build_scene(args.build_root,str(Path(get_libero_path('bddl_files'))/row['task']),0,
            dict(use_camera_obs=False,has_renderer=False,has_offscreen_renderer=False,hard_reset=False,ignore_done=True))
        budget=capacity_budget(built.native_model)
        d=mjw.make_data(built.native_model,nworld=2,nccdmax=budget['nconmax'],**budget)
        d.time.assign(bank[:2,0].astype(np.float32));d.qpos.assign(np.ascontiguousarray(bank[:2,1:1+m.nq],np.float32))
        d.qvel.assign(np.ascontiguousarray(bank[:2,1+m.nq:],np.float32))
        roots={k:getattr(d,k).numpy() for k in ('time','qpos','qvel')}
        forward.fwd_position(m,d);forward.fwd_velocity(m,d);forward.fwd_actuation(m,d)
        wp.synchronize_stream(stream)
        arrays=dict(nv=np.array(m.nv),nbody=np.array(m.nbody),factor_total=np.array(m.qLD_block_total),
            factor=d.qLD.numpy(),diagonal=d.qLDiagInv.numpy(),address=m.qLD_block_adr.numpy())
        for name in ('body_rootid','dof_bodyid','body_subtreenum','body_treeid'):
            arrays[name]=getattr(m,name).numpy()
        for name,expected in (('body_rootid','body_rootid'),('dof_bodyid','dof_bodyid'),('body_subtreenum','subtree_counts')):
            if not np.array_equal(arrays[name],row[expected]):raise RuntimeError('Imported GPU topology differs from CPU catalogue')
        for name in ('qfrc_passive','qfrc_bias','qfrc_actuator','qfrc_applied','xfrc_applied','xipos','subtree_com','cdof'):
            arrays[name]=getattr(d,name).numpy()
        for i,t in enumerate(m.M_tiles):
            arrays[f'starts{i}']=t.adr.numpy();arrays[f'size{i}']=np.array(t.size);arrays[f'kind{i}']=np.array(int(t.elemid.size!=0))
        if not all(np.array_equal(getattr(d,k).numpy(),v) for k,v in roots.items()):raise RuntimeError('Freeze changed physical roots')
        np.savez(path,**arrays)
        path.with_suffix('.json').write_text(json.dumps(dict(task=row['task'],identity=identity,sha256=digest(path),
            model_path=row['model_path'],model_sha256=row['model_sha256'],physics_integrations=0,newton_solves=0),indent=2)+'\n')
        print(json.dumps(dict(frozen=str(path),nv=m.nv,nbody=m.nbody)),flush=True)
        del built,m,d


def export_cpu_oracle(argv):
    """Use actual CPU reference mj_applyFT on the same frozen physical arrays."""
    import mujoco
    assert mujoco.__version__=='2.3.7'
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args(argv)
    result={};inputs={}
    for task_id in (0,3):
        path=args.output/f'task{task_id}.npz';receipt=json.loads(path.with_suffix('.json').read_text())
        assert digest(path)==receipt['sha256'] and digest(receipt['model_path'])==receipt['model_sha256']
        c=dict(np.load(path,allow_pickle=False));m=mujoco.MjModel.from_binary_path(receipt['model_path']);d=mujoco.MjData(m)
        assert m.nv==int(c['nv']) and m.nbody==int(c['nbody'])
        assert np.array_equal(m.body_rootid,c['body_rootid']) and np.array_equal(m.dof_bodyid,c['dof_bodyid'])
        jac=np.zeros((2,m.nbody,6,m.nv),np.float64)
        for w in range(2):
            d.cdof[:]=c['cdof'][w];d.subtree_com[:]=c['subtree_com'][w];d.xipos[:]=c['xipos'][w]
            for b in range(m.nbody):
                for k in range(6):
                    ft=np.eye(6)[k];target=np.zeros(m.nv)
                    mujoco.mj_applyFT(m,d,ft[:3],ft[3:],d.xipos[b],b,target);jac[w,b,k]=target
        result[f'task{task_id}']=jac;inputs[path.name]=digest(path)
    path=args.output/'cpu-oracle.npz';np.savez(path,**result)
    path.with_suffix('.json').write_text(json.dumps(dict(dependency=mujoco.__version__,function='mj_applyFT',
        sha256=digest(path),inputs=inputs,physics_integrations=0,tool_sha256=digest(__file__)),indent=2)+'\n')
    print(json.dumps(dict(cpu_oracle=True,physics_integrations=0)),flush=True)


def cases(output):
    result=[]
    for task_id in (0,3):
        source=dict(np.load(output/f'task{task_id}.npz',allow_pickle=False));nb=int(source['nbody'])
        for variant in ('real','single','multiple','asymmetric','force-only','torque-only','cancellation','padded'):
            c={k:v.copy() for k,v in source.items()};rng=np.random.default_rng(991+task_id)
            c.update(name=f'task{task_id}-{variant}',task_id=task_id,nv=int(c['nv']),nbody=nb,
                     factor_total=int(c['factor_total']),padding=3 if variant=='padded' else 0)
            ft=c['xfrc_applied'];ft[:]=0
            if variant=='single':ft[:,int(c['dof_bodyid'][0]),:]=rng.normal(size=(2,6))
            elif variant not in ('real','cancellation'):
                ft[:]=rng.normal(size=ft.shape).astype(np.float32)
                # Fixed and world-body inputs are included; they must contribute only through actual ancestors.
                if variant=='asymmetric':ft[0]=0
                elif variant=='force-only':ft[:,:,3:]=0
                elif variant=='torque-only':ft[:,:,:3]=0
            if variant=='cancellation':
                c['qfrc_passive'][:]=np.float32(1e4);c['qfrc_bias'][:]=np.float32(1e4)
                c['qfrc_actuator'][:]=rng.normal(size=(2,c['nv']));c['qfrc_applied']=-c['qfrc_actuator'].copy()
            result.append(c)
    return result


def reference(c,jac):
    """CPU Jacobian oracle plus a priori operation-magnitude error bounds."""
    passive,bias,act,applied=[c[k].astype(np.float64) for k in ('qfrc_passive','qfrc_bias','qfrc_actuator','qfrc_applied')]
    ft=c['xfrc_applied'].astype(np.float64);nv=c['nv']
    projection=np.einsum('wbki,wbk->wi',jac,ft)
    rhs=passive-bias+act+applied+projection
    unit=np.finfo(np.float32).eps/2
    base_abs=abs(passive)+abs(bias)+abs(act)+abs(applied)
    limit=3*unit/(1-3*unit)*base_abs
    for w in range(2):
        for i in range(nv):
            body=int(c['dof_bodyid'][i]);size=int(c['body_subtreenum'][body]);root=int(c['body_rootid'][body])
            angular=abs(c['cdof'][w,i,:3].astype(np.float64));linear=abs(c['cdof'][w,i,3:].astype(np.float64))
            major=0.
            for b in range(body,body+size):
                offset=abs(c['xipos'][w,b].astype(np.float64))+abs(c['subtree_com'][w,root].astype(np.float64))
                cross=np.array([angular[1]*offset[2]+angular[2]*offset[1],angular[2]*offset[0]+angular[0]*offset[2],
                                angular[0]*offset[1]+angular[1]*offset[0]])
                major+=float((linear+cross)@abs(ft[w,b,:3])+angular@abs(ft[w,b,3:]))
            gamma=(32+size)*unit/(1-(32+size)*unit)
            limit[w,i]+=8*gamma*major+unit*(base_abs[w,i]+abs(projection[w,i]))
    return rhs,limit


class Replay:
    MODES=('upstream','mass-only','split','fused')
    def __init__(self,c,args,stream,jac):
        import warp as wp
        from mujoco_warp._src import types
        from libero.gpu.host.runtime.dynamics.mjwarp_mass_solve import MassSolvePlan
        from libero.gpu.host.runtime.dynamics.mjwarp_smooth_force import SmoothForce
        self.case,self.stream,self.jac=c,stream,jac
        arr=lambda a,dtype:wp.array(np.ascontiguousarray(a),dtype=dtype,device=stream.device)
        self.model=m=NS(nv=c['nv'],nbody=c['nbody'],qLD_block_total=c['factor_total'],
            qLD_block_adr=arr(c['address'],int),block_dim=types.BlockDim(),M_tiles=[],
            opt=NS(enableflags=0,disableflags=0,integrator=types.IntegratorType.EULER))
        for k in ('body_rootid','dof_bodyid','body_subtreenum','body_treeid'):setattr(m,k,arr(c[k],int))
        i=0
        while f'starts{i}' in c:
            m.M_tiles.append(types.TileSet(adr=arr(c[f'starts{i}'],int),size=int(c[f'size{i}']),
                elemid=arr(np.array([0],np.int32) if c[f'kind{i}'] else np.empty(0,np.int32),int)));i+=1
        self.data=d=NS(nworld=2,qLD=arr(c['factor'],float),qLDiagInv=arr(c['diagonal'],float),
                       tree_awake=wp.empty((2,0),dtype=int,device=stream.device))
        for key in ('qfrc_passive','qfrc_bias','qfrc_actuator','qfrc_applied'):
            value=np.pad(c[key],((0,0),(0,c['padding'])),constant_values=37.)
            setattr(d,key,arr(value,float))
        for key,dtype in (('cdof',wp.spatial_vector),('xfrc_applied',wp.spatial_vector),
                          ('xipos',wp.vec3),('subtree_com',wp.vec3)):setattr(d,key,arr(c[key],dtype))
        self.inputs={k:getattr(d,k).numpy() for k in ('qLD','qLDiagInv','qfrc_passive','qfrc_bias',
            'qfrc_actuator','qfrc_applied','cdof','xfrc_applied','xipos','subtree_com')}
        self.plan=MassSolvePlan(m,d,args.mass_library,stream);self.graphs={};self.timing={};self.outputs={};self.bindings={}
        for mode in self.MODES:
            d.qfrc_smooth=wp.full((2,m.nv+c['padding']),37.,dtype=float,device=stream.device)
            d.qacc_smooth=wp.full_like(d.qfrc_smooth,37.)
            binding=self.plan.bind(d.qacc_smooth,d.qfrc_smooth)
            force=SmoothForce(m,d,args.force_library,stream)
            self.bindings[mode]=(binding,force);self.outputs[mode]=(d.qfrc_smooth,d.qacc_smooth)
            with wp.ScopedCapture(stream=stream) as capture:self.enqueue(mode)
            self.graphs[mode]=capture.graph
            if c['name'].endswith('-real'):
                with wp.ScopedCapture(stream=stream) as capture:
                    for _ in range(32):self.enqueue(mode)
                self.timing[mode]=capture.graph

    def enqueue(self,mode):
        import warp as wp
        from mujoco_warp._src import forward
        d,m=self.data,self.model
        d.qfrc_smooth,d.qacc_smooth=self.outputs[mode]
        binding,force=self.bindings[mode]
        if mode=='upstream':forward.fwd_acceleration(m,d)
        else:
            if mode=='mass-only':
                wp.launch(forward._qfrc_smooth(False),dim=(2,m.nv),inputs=[m.body_treeid,m.dof_bodyid,
                    d.qfrc_applied,d.tree_awake,d.qfrc_bias,d.qfrc_passive,d.qfrc_actuator],outputs=[d.qfrc_smooth])
                forward.xfrc_accumulate(m,d,d.qfrc_smooth)
            else:force.enqueue(split=mode=='split')
            binding.enqueue()

    def check(self,updated=False):
        import warp as wp
        from validation.precision import balance_residual
        c={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in self.case.items()};nv=c['nv']
        if updated:
            c['xfrc_applied']*=np.float32(-.25);c['xfrc_applied'][0,-1,0]=1.5
            c['qfrc_applied']*=np.float32(-.5)
        self.data.xfrc_applied.assign(c['xfrc_applied'])
        applied=np.pad(c['qfrc_applied'],((0,0),(0,c['padding'])),constant_values=37.)
        self.data.qfrc_applied.assign(applied)
        for graph in self.graphs.values():wp.capture_launch(graph,stream=self.stream)
        wp.synchronize_stream(self.stream)
        outputs={mode:dict(force=f.numpy(),acceleration=x.numpy()) for mode,(f,x) in self.outputs.items()}
        rhs,force_limit=reference(c,self.jac);rows=[]
        for mode,out in outputs.items():
            force_error=abs(out['force'][:,:nv]-rhs)
            force_ok=bool(np.all(force_error<=force_limit))
            checks=[];i=0
            while f'starts{i}' in c:
                size=int(c[f'size{i}'])
                for start in c[f'starts{i}']:
                    start=int(start);sl=slice(start,start+size);adr=int(c['address'][start])
                    for w in range(2):
                        if adr==-2:a=np.diag(1/c['diagonal'][w,sl].astype(np.float64))
                        else:
                            u=np.triu(c['factor'][w,adr:adr+size*size].reshape(size,size)).astype(np.float64);a=u.T@u
                        res=balance_residual(a,out['acceleration'][w,sl],out['force'][w,sl],np.zeros(size))
                        ref=np.linalg.solve(a,rhs[w,sl]);condition=float(np.linalg.cond(a,np.inf))
                        unit=np.finfo(np.float32).eps/2;eta=8*size*unit/(1-size*unit)
                        if condition*eta>=1:raise RuntimeError('Unqualified forward sensitivity')
                        bound=2*condition*eta/(1-condition*eta)*max(float(abs(ref).max()),np.finfo(np.float32).tiny)
                        bound+=float(np.linalg.norm(np.linalg.inv(a),np.inf))*float(force_limit[w,sl].max())
                        error=float(abs(out['acceleration'][w,sl]-ref).max())
                        checks.append(dict(world=w,start=start,**res,forward_error=error,forward_limit=bound,
                                           qualified=bool(res['passed'] and error<=bound)))
                i+=1
            held=all(np.all(out[k][:,nv:]==37.) for k in out)
            rows.append(dict(mode=mode,force_max_error=float(force_error.max()),
                force_max_budget_fraction=float(np.max(np.divide(force_error,force_limit,out=np.zeros_like(force_error),where=force_limit>0))),
                force_passed=force_ok,held=held,checks=checks,
                passed=bool(force_ok and held and all(x['qualified'] for x in checks))))
        inputs_held=all(np.array_equal(getattr(self.data,k).numpy(),applied if k=='qfrc_applied' else
            c['xfrc_applied'] if k=='xfrc_applied' else v,equal_nan=True) for k,v in self.inputs.items())
        differences={mode:{k:float(abs(out[k]-outputs['upstream'][k]).max()) for k in out}
                     for mode,out in outputs.items() if mode!='upstream'}
        return dict(name=c['name'],updated=updated,rows=rows,inputs_held=inputs_held,differences=differences,
                    passed=bool(inputs_held and all(x['passed'] for x in rows))),outputs

    def benchmark(self,args):
        import warp as wp
        # Restore the actual frozen zero-external-force case, outside timing.
        self.data.xfrc_applied.assign(self.case['xfrc_applied']);self.data.qfrc_applied.assign(self.case['qfrc_applied'])
        result={}
        for mode,graph in self.timing.items():
            for _ in range(args.warmup):wp.capture_launch(graph,stream=self.stream)
            start,end=wp.Event(enable_timing=True),wp.Event(enable_timing=True);values=[]
            for _ in range(args.steps):
                self.stream.record_event(start);wp.capture_launch(graph,stream=self.stream);self.stream.record_event(end)
                wp.synchronize_event(end);values.append(wp.get_event_elapsed_time(start,end,synchronize=False)*1000/32)
            result[mode]=dict(mean_us=float(np.mean(values)),std_us=float(np.std(values)),microseconds=values)
        result.update(name=self.case['name'],calls_per_sample=32,warmup=args.warmup,samples=args.steps,
            speedup=result['upstream']['mean_us']/result['fused']['mean_us'],
            speedup_vs_previous=result['mass-only']['mean_us']/result['fused']['mean_us'])
        return result


def main(argv=None,*,args=None):
    if args is None:args=add_arguments(argparse.ArgumentParser(description=__doc__)).parse_args(argv)
    if args.warmup<0 or args.steps<1:raise ValueError('Invalid measurement counts')
    from libero.gpu import precision
    precision.select('fp32')
    import warp as wp
    from warp._src.build import init_kernel_cache
    from libero.gpu.host.backends.mjwarp.build import source_files
    from libero.gpu.host.build.compilation import compilation_scope,install_warp_guard
    from validation.common import save_json
    from validation.runner import gpu_load_snapshot
    args.output.mkdir(parents=True,exist_ok=True);init_kernel_cache(args.build_root/'mjwarp/warp-cache')
    install_warp_guard();wp.init();stream=wp.Stream('cuda:0');prepared=args.output/'preparation.json'
    with compilation_scope(args.prepare) as scope,wp.ScopedStream(stream):
        if args.prepare:
            freeze(args,stream)
            if not (args.output/'cpu-oracle.npz').exists():
                import subprocess
                if args.cpu_python is None:raise ValueError('First preparation needs --cpu-python')
                subprocess.run([str(args.cpu_python.resolve()),str(Path(__file__).resolve()),'cpu-oracle',
                                '--output',str(args.output.resolve())],check=True)
        inputs=[args.output/f'task{i}.npz' for i in (0,3)];oracle=args.output/'cpu-oracle.npz'
        receipt=json.loads(oracle.with_suffix('.json').read_text())
        if receipt['sha256']!=digest(oracle) or receipt['inputs']!={p.name:digest(p) for p in inputs}:
            raise RuntimeError('CPU oracle does not match frozen inputs')
        signature=dict(sources=source_files(include_force=True),tool_sha256=digest(__file__),
            mass_library_sha256=digest(args.mass_library),force_library_sha256=digest(args.force_library),
            catalogue_sha256=digest(args.catalogue),inputs={p.name:digest(p) for p in inputs},
            oracle_sha256=digest(oracle),oracle_receipt_sha256=digest(oracle.with_suffix('.json')))
        if not args.prepare and (not prepared.exists() or json.loads(prepared.read_text())['signature']!=signature):
            raise RuntimeError('Prepare exact acceleration operators offline first')
        with np.load(oracle,allow_pickle=False) as saved:jac={k:saved[k] for k in saved.files}
        replays=[Replay(c,args,stream,jac[f'task{c["task_id"]}']) for c in cases(args.output)]
        if args.prepare:
            wp.synchronize_stream(stream);save_json(prepared,dict(signature=signature,compilation_events=scope['events'],physics_integrations=0))
            print(json.dumps(dict(prepared=True,cases=len(replays),compilations=len(scope['events']))),flush=True);return 0
        before=gpu_load_snapshot();rows=[]
        for replay in replays:
            for updated in (False,True):
                row,outputs=replay.check(updated);rows.append(row)
                np.savez(args.output/(row['name']+('-updated' if updated else '')+'-output.npz'),
                         **{mode+'.'+k:v for mode,out in outputs.items() for k,v in out.items()})
                print(json.dumps(dict(name=row['name'],updated=updated,passed=row['passed'])),flush=True)
        passed=all(x['passed'] for x in rows)
        timing=[x.benchmark(args) for x in replays if x.timing] if args.performance and passed else None
        save_json(args.output/'summary.json',dict(passed=passed,signature=signature,cases=rows,performance=timing,
            runtime_compilations=scope['events'],physics_integrations=0,newton_solves=0,
            gpu_load_before=before,gpu_load_after=gpu_load_snapshot(),policy='a priori force operation majorant; existing 8 gamma_block solve residual and normwise sensitivity'))
        print(json.dumps(dict(passed=passed,performance=timing)),flush=True);return 0 if passed else 2


if __name__=='__main__':
    import sys
    if sys.argv[1:2]==['cpu-oracle']:export_cpu_oracle(sys.argv[2:])
    else:raise SystemExit(main())
