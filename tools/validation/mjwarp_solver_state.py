"""N=2 local initialization/stopping replay and immutable task-0 Newton input.

Offline preparation may compile pinned reference operators and perform one raw
forward without integration. Ordinary replay only consumes frozen inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
from libero.gpu.host.build.task_catalog import load_catalog
from types import SimpleNamespace as NS

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_arguments(parser):
    for name in ('input','build-root','library','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--prepare',action='store_true')
    return parser


def initial_arrays(m,d):
    from mujoco_warp._src import types
    a={k:getattr(d,k).numpy() for k in ('M','nefc','qacc_warmstart','qacc_smooth','qacc','qfrc_constraint','overflow')}
    a.update({k:getattr(d.efc,k).numpy() for k in ('J','aref')})
    a.update({k:getattr(m,k).numpy() for k in ('M_mulm_rowadr','M_mulm_col','M_mulm_madr')})
    n,nv,rows=d.nworld,m.nv,d.njmax
    a.update(nv=np.array(nv),warmstart=np.array(not(m.opt.disableflags & types.DisableBit.WARMSTART)),iterations=np.array(m.opt.iterations),
        tolerance=m.opt.tolerance.numpy(),inertia=m.stat.meaninertia.numpy(),
        Ma=np.full((n,nv),17.,np.float32),Jaref=np.full((n,rows),19.,np.float32),
        search_dot=np.full(n,11.,np.float32),niter=np.full(n,9,np.int32),done=np.ones(n,np.bool_),
        grad_dot=np.ones(n,np.float32),decrement=np.ones(n,np.float32),
        improvement=np.ones(n,np.float32),alpha=np.ones(n,np.float32))
    return a


def freeze_real(args):
    import warp as wp
    import mujoco_warp as mjw
    from validation.mjwarp_newton import model_for,fields
    from libero.gpu.host.backends.mjwarp.model import build_scene,capacity_budget
    from libero.gpu.host.backends.mjwarp.build import source_files
    from libero.libero import get_libero_path
    folder=args.output/'frozen';folder.mkdir(exist_ok=True);receipt=folder/'receipt.json'
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if old['input_sha256']!=digest(args.input):raise ValueError('another frozen Newton boundary')
        for k,v in old['files'].items():
            if digest(folder/k)!=v:raise ValueError('frozen solver state data changed: '+k)
        return
    if not args.prepare:raise RuntimeError('freeze initialization inputs offline first')
    tasks=load_catalog(args.build_root);identities={}
    for task_id in (4,0):
        t=next(t for t in tasks if t['suite']=='libero_10' and t['task_id']==task_id)
        if task_id==4:
            host,m,identity=model_for(args.build_root,t)
            boundary=json.loads(args.input.with_name('summary.json').read_text())
            for key in ('scene_sha256','imported_xml_sha256','imported_model_sha256','mjwarp_commit'):
                if identity[key]!=boundary['identity'][key]:raise ValueError('frozen model differs: '+key)
            layout=boundary['frozen_newton']['layout']
            d=mjw.make_data(host,nworld=2,nconmax=layout['pooled_contacts']//2,
                           nccdmax=layout['pooled_contacts']//2,njmax=layout['rows'])
            saved=dict(np.load(args.input,allow_pickle=False))
            for name,target in fields(d).items():
                if 'input.'+name in saved:target.assign(saved['input.'+name])
        else:
            built,m,bank,_,identity=build_scene(args.build_root,
                str(Path(get_libero_path('bddl_files'))/t['key']),0,
                dict(use_camera_obs=False,has_renderer=False,has_offscreen_renderer=False,hard_reset=False,ignore_done=True))
            cap=capacity_budget(built.native_model)
            d=mjw.make_data(built.native_model,nworld=2,nccdmax=cap['nconmax'],**cap)
            d.qpos.assign(bank[:2,1:1+m.nq].astype(np.float32))
            d.qvel.assign(bank[:2,1+m.nq:1+m.nq+m.nv].astype(np.float32))
            d.time.assign(bank[:2,0].astype(np.float32))
            held={k:getattr(d,k).numpy() for k in ('qpos','qvel','time')}
            mjw.forward(m,d);wp.synchronize()
            assert all(np.array_equal(getattr(d,k).numpy(),v) for k,v in held.items())
            arrays={'input.'+k:v.numpy() for k,v in fields(d).items()}
            arrays.update({'model.'+k:getattr(m,k).numpy() for k in ('M_hinit_i','M_colind','dof_parentid')})
            np.savez(folder/'task0-newton.npz',**arrays)
            (folder/'task0-newton.json').write_text(json.dumps(dict(identity=identity,
                input_sha256=digest(folder/'task0-newton.npz'),worlds=2,
                solver={k:int(getattr(m.opt,k)) for k in ('iterations','ls_iterations','cone','disableflags')},
                physics_integrations=0),indent=2)+'\n')
        np.savez(folder/f'task{task_id}-init.npz',**initial_arrays(m,d));identities[str(task_id)]=identity
    files=[f'task{t}-init.npz' for t in (4,0)]+['task0-newton.npz','task0-newton.json']
    receipt.write_text(json.dumps(dict(input_sha256=digest(args.input),files={k:digest(folder/k) for k in files},
        identities=identities,reference_sources=source_files(),collector_sha256=digest(__file__),
        reference_forward_replays=1,physics_integrations=0),indent=2)+'\n')


def init_case(nv,warm=True,variant='ordinary'):
    from validation.mjwarp_linesearch import synthetic
    a=synthetic(nv)['arrays'];rng=np.random.default_rng(87000+nv)
    n,rows,pad=a['J'].shape
    a.update(qacc_warmstart=rng.normal(0.,.5,(n,nv)).astype(np.float32),
        qacc_smooth=rng.normal(0.,.5,(n,nv)).astype(np.float32),
        qfrc_constraint=np.full((n,nv),29.,np.float32),aref=rng.normal(0.,.5,(n,rows)).astype(np.float32),
        warmstart=np.array(warm),iterations=np.array(100),stop_calls=np.array(1),
        Ma=np.full((n,nv),17.,np.float32),Jaref=np.full((n,rows),19.,np.float32),
        search_dot=np.full(n,11.,np.float32),niter=np.full(n,9,np.int32),done=np.ones(n,np.bool_),
        grad_dot=np.ones(n,np.float32),decrement=np.ones(n,np.float32),
        improvement=np.ones(n,np.float32),alpha=np.ones(n,np.float32))
    if variant=='empty-world0':a['nefc'][0]=0
    elif variant=='empty-both':a['nefc'][:]=0
    return dict(name=f'nv{nv}-init-{"warm" if warm else "cold"}-{variant}',kind='initialize',arrays=a)


def stop_case(nv,variant='continue'):
    a=init_case(nv)['arrays'];scale=np.float32(nv)*a['inertia'][0];tol=np.float32(1.e-4)
    a['tolerance']=np.array([tol],np.float32)
    a.update(done=np.zeros(2,np.bool_),niter=np.zeros(2,np.int32),search_dot=np.ones(2,np.float32),
        alpha=np.ones(2,np.float32),improvement=np.full(2,2*tol*scale,np.float32),
        grad_dot=np.full(2,(2*tol*scale)**2,np.float32),decrement=np.full(2,4*tol*scale,np.float32),
        stop_calls=np.array(2))
    if variant=='zero-alpha':a['alpha'][0]=0.
    elif variant=='improvement':a['improvement'][0]=.5*tol*scale
    elif variant=='zero-improvement':a['improvement'][0]=0.
    elif variant=='negative-improvement':a['improvement'][0]=-tol*scale
    elif variant=='gradient':a['grad_dot'][0]=(.5*tol*scale)**2
    elif variant=='decrement':a['decrement'][0]=tol*scale
    elif variant=='iteration-limit':a['niter'][0]=99
    elif variant=='already-done':a['done'][0]=True
    elif variant=='all-done':a['done'][:]=True
    elif variant=='both-finish':a['alpha'][:]=0.
    elif variant=='per-world-scalars':
        a['inertia']=np.array([a['inertia'][0],2*a['inertia'][0]],np.float32)
        a['tolerance']=np.array([tol,4*tol],np.float32)
    elif variant.startswith('improvement-boundary'):
        value=np.float32(tol*scale)
        a['improvement'][0]=np.nextafter(value,np.float32(0. if variant.endswith('below') else np.inf))
    elif variant.startswith('gradient-boundary'):
        value=np.float32((tol*scale)**2)
        a['grad_dot'][0]=np.nextafter(value,np.float32(0. if variant.endswith('below') else np.inf))
    elif variant.startswith('decrement-boundary'):
        value=np.float32(2*tol*scale)
        a['decrement'][0]=np.nextafter(value,np.float32(0. if variant.endswith('below') else np.inf))
    return dict(name=f'nv{nv}-stop-{variant}',kind='stop',arrays=a)


def cases(args):
    from libero.gpu.host.runtime.newton.mjwarp_direction import FAMILIES
    result=[dict(name=f'task{t}-init-frozen',kind='initialize',arrays=dict(np.load(
        args.output/'frozen'/f'task{t}-init.npz',allow_pickle=False))) for t in (4,0)]
    result += [init_case(nv,warm) for nv in FAMILIES for warm in (True,False)]
    result += [init_case(nv,True,v) for nv in (21,39,57) for v in ('empty-world0','empty-both')]
    result += [stop_case(nv) for nv in FAMILIES]
    result += [stop_case(39,v) for v in ('zero-alpha','improvement','zero-improvement','negative-improvement',
        'gradient','decrement','iteration-limit','already-done','all-done','both-finish','per-world-scalars',
        'improvement-boundary-below','improvement-boundary-above','gradient-boundary-below',
        'gradient-boundary-above','decrement-boundary-below','decrement-boundary-above')]
    return result


def expected_stop(a):
    done=a['done'].copy();niter=a['niter'].copy();overflow=a['overflow'].copy();live=np.array([2-int(done.sum())],np.int32)
    for _ in range(int(a.get('stop_calls',1))):
        for w in range(2):
            if done[w]:continue
            niter[w]+=1
            scale=np.float32(a['inertia'][w % a['inertia'].size]*np.float32(int(a['nv'])))
            tolerance=a['tolerance'][w % a['tolerance'].size]
            improvement=np.float32(a['improvement'][w]/scale)
            gradient=np.float32(np.sqrt(a['grad_dot'][w])/scale)
            model=np.float32(np.float32(.5)*a['decrement'][w]/scale)
            finish=a['alpha'][w]==0 or (0<improvement<tolerance) or gradient<tolerance or model<tolerance
            if finish or niter[w]==int(a['iterations']):
                if not finish:overflow[w]|=512
                done[w]=True;live[0]-=1
    return dict(done=done,niter=niter,overflow=overflow,nsolving=live)


class Replay:
    def __init__(self,case,library,stream):
        import warp as wp
        from mujoco_warp._src import types
        from libero.gpu.host.runtime.newton.mjwarp_solver_state import SolverStatePlan
        self.case,self.stream=case,stream
        a=case['arrays'];nv=int(a['nv']);n,rows,pad=a['J'].shape
        excluded={'nv','warmstart','iterations','stop_calls','ls_iterations'}
        dev={k:wp.array(np.ascontiguousarray(v),dtype=wp.bool if v.dtype==np.bool_ else
            wp.int32 if v.dtype==np.int32 else wp.float32,device=stream.device)
            for k,v in a.items() if k not in excluded and k not in ('contact_friction',)}
        # No contact metadata is read by this stage.
        self.model=NS(nv=nv,nv_pad=pad,is_sparse=False,opt=NS(cone=types.ConeType.ELLIPTIC,
            solver=types.SolverType.NEWTON,integrator=types.IntegratorType.EULER,enableflags=0,
            iterations=int(a['iterations']),warn_overflow=0,tolerance=dev['tolerance'],
            disableflags=0 if a['warmstart'] else types.DisableBit.WARMSTART),stat=NS(meaninertia=dev['inertia']),
            **{k:dev[k] for k in ('M_mulm_rowadr','M_mulm_col','M_mulm_madr')})
        self.inputs,self.data,self.contexts,self.graphs,self.restores,self.outputs,self.restore_sources=dev,{},{},{},{},{},{}
        for mode in ('upstream','cuda'):
            output={k:wp.clone(dev[k]) for k in ('qacc','Ma','Jaref','search_dot','niter','done','qfrc_constraint','overflow')}
            output['nsolving']=wp.array(np.array([2-int(a['done'].sum())] if case['kind']=='stop' else [2],np.int32),
                                       dtype=wp.int32,device=stream.device)
            d=NS(nworld=n,njmax=rows,nefc=dev['nefc'],M=dev['M'],qacc=output['qacc'],
                qacc_warmstart=dev['qacc_warmstart'],qacc_smooth=dev['qacc_smooth'],dof_cdof=None,
                qfrc_constraint=output['qfrc_constraint'],solver_niter=output['niter'],overflow=output['overflow'],
                efc=NS(J=dev['J'],aref=dev['aref'],Ma=output['Ma'],J_rownnz=None,J_rowadr=None,J_colind=None))
            ctx=NS(Jaref=output['Jaref'],search_dot=output['search_dot'],done=output['done'],
                grad_dot=dev['grad_dot'],newton_decrement=dev['decrement'],improvement=dev['improvement'],
                alpha=dev['alpha'],compact_d_full=None,compact_m_full=None)
            self.data[mode],self.contexts[mode],self.outputs[mode]=d,ctx,output
            src={k:wp.clone(v) for k,v in output.items()};self.restore_sources[mode]=src
            with wp.ScopedCapture(stream=stream) as capture:
                for k,v in output.items():wp.copy(v,src[k],stream=stream)
            self.restores[mode]=capture.graph
            if mode=='cuda':
                self.plan=SolverStatePlan(self.model,d,library,stream);self.binding=self.plan.bind(ctx)
                self.binding.bind_stopping(output['nsolving'])
            with wp.ScopedCapture(stream=stream) as capture:self.enqueue(mode)
            self.graphs[mode]=capture.graph

    def enqueue(self,mode):
        import warp as wp
        from mujoco_warp._src import solver
        m,d,c=self.model,self.data[mode],self.contexts[mode]
        if self.case['kind']=='initialize':
            if mode=='cuda':self.binding.initialize();return
            wp.launch(solver._solve_init_dof(bool(self.case['arrays']['warmstart']),False),dim=(d.nworld,m.nv),
                inputs=[d.nefc,d.qacc_warmstart,d.qacc_smooth],outputs=[d.qacc,d.qfrc_constraint])
            wp.launch(solver._solve_init_efc,dim=d.nworld,outputs=[d.solver_niter,c.search_dot,c.done])
            chunk=20 if m.nv>50 else 50;parts=(m.nv+chunk-1)//chunk
            if parts>1:c.Jaref.zero_()
            wp.launch(solver._solve_init_jaref_kernel(False,m.nv,chunk,False),dim=(d.nworld,d.njmax,parts),
                inputs=[d.nefc,d.qacc,None,None,None,d.efc.J,d.efc.aref,None],outputs=[c.Jaref])
            solver._mul_m_compact_aware(m,d,c,d.efc.Ma,d.qacc,c.done)
        else:
            for _ in range(int(self.case['arrays'].get('stop_calls',1))):
                if mode=='cuda':self.binding.stop()
                else:wp.launch(solver._solve_done(0),dim=d.nworld,inputs=[m.nv,m.opt.tolerance,m.opt.iterations,
                    m.stat.meaninertia,c.grad_dot,c.newton_decrement,c.improvement,c.alpha,c.done],
                    outputs=[d.solver_niter,d.overflow,self.outputs[mode]['nsolving'],c.done])

    def check(self):
        import warp as wp
        from validation.mjwarp_linesearch import matrix
        a=self.case['arrays'];nv=int(a['nv']);eps=np.finfo(np.float32).eps
        out={}
        for mode in ('upstream','cuda'):
            wp.capture_launch(self.restores[mode],stream=self.stream);wp.capture_launch(self.graphs[mode],stream=self.stream)
            wp.synchronize_stream(self.stream);out[mode]={k:v.numpy() for k,v in self.outputs[mode].items()}
        checks=[]
        if self.case['kind']=='initialize':
            q=np.where((a['nefc']>0)[:,None] & bool(a['warmstart']),a['qacc_warmstart'],a['qacc_smooth'])
            mass=matrix(a);j=a['J'][:,:,:nv].astype(np.float64);q64=q.astype(np.float64)
            ma=np.einsum('wij,wj->wi',mass,q64);jr=np.einsum('wri,wi->wr',j,q64)-a['aref']
            gamma=nv*eps/(1-nv*eps)
            mb=8*gamma*np.einsum('wij,wj->wi',abs(mass),abs(q64))
            jb=8*gamma*(np.einsum('wri,wi->wr',abs(j),abs(q64))+abs(a['aref']))
            for mode,o in out.items():
                worlds=[]
                for w in range(2):
                    used=int(a['nefc'][w]);unused=o['Jaref'][w,used:]
                    holds=np.all(unused==0.) if nv>50 else np.array_equal(unused,a['Jaref'][w,used:])
                    me=abs(o['Ma'][w]-ma[w]);je=abs(o['Jaref'][w,:used]-jr[w,:used])
                    worlds.append(dict(world=w,mass_max_error=float(me.max()),jaref_max_error=float(je.max(initial=0.)),
                        unused_holds=bool(holds),passed=bool(holds and np.all(me<=mb[w]) and np.all(je<=jb[w,:used]))))
                scalars=bool(np.array_equal(o['qacc'],q) and np.all(o['niter']==0) and not o['done'].any()
                    and np.all(o['search_dot']==0.) and np.array_equal(o['qfrc_constraint'],a['qfrc_constraint'])
                    and np.array_equal(o['overflow'],a['overflow']) and o['nsolving'][0]==2)
                checks.append(dict(mode=mode,worlds=worlds,exact_copy_and_reset=scalars,
                    passed=bool(scalars and all(w['passed'] for w in worlds))))
        else:
            expected=expected_stop(a)
            for mode,o in out.items():
                status={k:bool(np.array_equal(o[k],v)) for k,v in expected.items()}
                hold=all(np.array_equal(v,self.restore_sources[mode][k].numpy()) for k,v in o.items() if k not in expected)
                checks.append(dict(mode=mode,status=status,expected={k:v.tolist() for k,v in expected.items()},
                    held=hold,passed=bool(hold and all(status.values()))))
        hold=all(np.array_equal(np.ascontiguousarray(v.numpy()).view(np.uint8),
                              np.ascontiguousarray(a[k]).view(np.uint8)) for k,v in self.inputs.items())
        return dict(name=self.case['name'],kind=self.case['kind'],nv=nv,worlds=2,checks=checks,input_holds=hold,
                    passed=bool(hold and all(c['passed'] for c in checks))),out


def main(argv=None,*,args=None):
    if args is None:args=add_arguments(argparse.ArgumentParser(description=__doc__)).parse_args(argv)
    from libero.gpu import precision
    precision.select('fp32')
    import warp as wp
    from warp._src.build import init_kernel_cache
    from libero.gpu.host.backends.mjwarp.build import source_files
    from libero.gpu.host.build.compilation import compilation_scope,install_warp_guard
    from validation.common import save_json
    from validation.runner import gpu_load_snapshot
    args.output.mkdir(parents=True,exist_ok=True)
    init_kernel_cache(args.build_root/'mjwarp/warp-cache');install_warp_guard();wp.init();stream=wp.Stream('cuda:0')
    with compilation_scope(args.prepare) as scope,wp.ScopedStream(stream):
        freeze_real(args)
        signature=dict(sources=source_files(include_state=True),input_sha256=digest(args.input),
            library_sha256=digest(args.library),tool_sha256=digest(__file__),
            frozen_receipt_sha256=digest(args.output/'frozen/receipt.json'))
        preparation=args.output/'preparation.json'
        if not args.prepare and (not preparation.is_file() or json.loads(preparation.read_text())['signature']!=signature):
            raise RuntimeError('prepare these exact solver state graphs offline first')
        replays=[Replay(c,args.library,stream) for c in cases(args)]
        if args.prepare:
            wp.synchronize_stream(stream);save_json(preparation,dict(signature=signature,
                compilation_events=scope['events'],cases=len(replays),physics_integrations=0,local_graph_replays=0))
            print(json.dumps(dict(prepared=True,cases=len(replays),compilations=len(scope['events']))),flush=True);return 0
        before=gpu_load_snapshot();rows=[]
        for replay in replays:
            row,out=replay.check();rows.append(row)
            np.savez(args.output/(row['name']+'.npz'),**{mode+'.'+k:v for mode,o in out.items() for k,v in o.items()})
            print(json.dumps(dict(name=row['name'],passed=row['passed'])),flush=True)
        passed=all(row['passed'] for row in rows)
        save_json(args.output/'summary.json',dict(passed=passed,cases=rows,signature=signature,
            runtime_compilations=scope['events'],physics_integrations=0,gpu_load_before=before,gpu_load_after=gpu_load_snapshot(),
            policy='independent FP64 product bounds; exact copy/reset/hold and FP32 stopping/counter transitions'))
        print(json.dumps(dict(passed=passed,cases=len(rows))),flush=True);return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())
