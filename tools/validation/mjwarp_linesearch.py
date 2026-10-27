"""Frozen local line-search replay, with explicit offline preparation.

No physics integration. Products and line search are qualified separately;
all numerical candidates consume identical resident inputs and restore outside
measured graphs. Every real timing sample contains one N=2 invocation.
"""
import argparse
import dataclasses
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
    parser.add_argument('--products-only',action='store_true')
    parser.add_argument('--performance',action='store_true')
    parser.add_argument('--warmup',type=int,default=5)
    parser.add_argument('--steps',type=int,default=20)
    return parser


def snapshot(m,d,ctx):
    a={name:getattr(d,name).numpy() for name in ('ne','nf','nefc','nacon','M','qacc','qfrc_smooth','overflow')}
    a.update({'contact_'+k:getattr(d.contact,k).numpy() for k in ('dim','friction','efc_address')})
    a.update({k:getattr(d.efc,k).numpy() for k in ('J','D','frictionloss','type','id','Ma')})
    a.update({k:getattr(ctx,k).numpy() for k in ('search','search_dot','done','Jaref')})
    a.update({k:getattr(m,k).numpy() for k in ('M_mulm_rowadr','M_mulm_col','M_mulm_madr')})
    a.update({k:getattr(m.opt,k).numpy() for k in ('tolerance','ls_tolerance','impratio_invsqrt')})
    a.update(inertia=m.stat.meaninertia.numpy(),nv=np.array(m.nv),ls_iterations=np.array(m.opt.ls_iterations))
    return a


def freeze_real(args):
    import warp as wp
    import mujoco_warp as mjw
    from mujoco_warp._src import solver,types
    from validation.mjwarp_newton import make_fixtures
    from libero.gpu.host.backends.mjwarp.build import source_files
    from libero.gpu.host.backends.mjwarp.model import build_scene,capacity_budget
    from libero.libero import get_libero_path
    folder=args.output/'frozen';folder.mkdir(exist_ok=True);receipt=folder/'receipt.json'
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if old['input_sha256']!=digest(args.input):raise ValueError('another frozen Newton boundary')
        for k,v in old['files'].items():
            if digest(folder/k)!=v:raise ValueError('frozen line-search data changed: '+k)
        return
    if not args.prepare:raise RuntimeError('freeze raw line-search inputs offline first')
    f=make_fixtures(args)[0];m,d=f['model'],f['data'];identities={'4':f['model_identity']}
    tasks=load_catalog(args.build_root)
    for task_id in (4,0):
        if task_id==0:
            t=next(t for t in tasks if t['suite']=='libero_10' and t['task_id']==0)
            built,m,bank,_,identity=build_scene(args.build_root,
                str(Path(get_libero_path('bddl_files'))/t['key']),0,
                dict(use_camera_obs=False,has_renderer=False,has_offscreen_renderer=False,hard_reset=False,ignore_done=True))
            cap=capacity_budget(built.native_model)
            d=mjw.make_data(built.native_model,nworld=2,nccdmax=cap['nconmax'],**cap)
            nq,nv=m.nq,m.nv
            d.qpos.assign(bank[:2,1:1+nq].astype(np.float32))
            d.qvel.assign(bank[:2,1+nq:1+nq+nv].astype(np.float32))
            d.time.assign(bank[:2,0].astype(np.float32))
            held={k:getattr(d,k).numpy() for k in ('qpos','qvel','time')}
            mjw.forward(m,d)
            assert all(np.array_equal(getattr(d,k).numpy(),v) for k,v in held.items())
            identities['0']=identity
        ctx=solver._create_solver_context(m,d)
        warm=not(m.opt.disableflags & types.DisableBit.WARMSTART)
        wp.launch(solver._solve_init_dof(warm,False),dim=(2,m.nv),
            inputs=[d.nefc,d.qacc_warmstart,d.qacc_smooth],outputs=[d.qacc,d.qfrc_constraint])
        solver.init_context(m,d,ctx,grad=True)
        wp.synchronize()
        np.savez(folder/f'task{task_id}.npz',**snapshot(m,d,ctx))
    receipt.write_text(json.dumps(dict(input_sha256=digest(args.input),files={
        f'task{t}.npz':digest(folder/f'task{t}.npz') for t in (4,0)},model_identities=identities,
        reference_sources=source_files(),collector_sha256=digest(__file__),physics_integrations=0,
        upstream_forward_replays=1),indent=2)+'\n')


def synthetic(nv,variant='ordinary'):
    from validation.mjwarp_constraint import synthetic as constraints,reference
    c=constraints(nv,variant if variant in ('world1-done','all-done','empty-world0','top-boundary',
        'bottom-boundary','zero-tangent','friction-boundary','broadcast-impratio') else 'ordinary')
    a=c['arrays'];a['Jaref']=a.pop('jaref');a['impratio_invsqrt']=a.pop('impratio')
    rng=np.random.default_rng(99000+nv)
    b=rng.normal(0.,.1,(2,nv,nv));mass=b@b.transpose(0,2,1)+np.eye(nv)[None,:,:]*2.
    rr,cc=np.tril_indices(nv);a['M']=mass[:,rr,cc].astype(np.float32)
    adr=np.zeros((nv,nv),np.int32);adr[rr,cc]=np.arange(len(rr));adr[cc,rr]=np.arange(len(rr))
    a['M_mulm_rowadr']=np.arange(nv+1,dtype=np.int32)*nv
    a['M_mulm_col']=np.tile(np.arange(nv,dtype=np.int32),nv)
    a['M_mulm_madr']=adr.reshape(-1)
    mass=matrix(a)
    a['qacc']=rng.normal(0.,.3,(2,nv)).astype(np.float32)
    a['Ma']=np.einsum('wij,wj->wi',mass,a['qacc']).astype(np.float32)
    a['qfrc_smooth']=rng.normal(0.,.5,(2,nv)).astype(np.float32)
    projection=dict(a,jaref=a['Jaref'],impratio=a['impratio_invsqrt'])
    force=reference(projection)[0];search=[]
    for w in range(2):
        used=int(a['nefc'][w]);j=a['J'][w,:used,:nv].astype(np.float64)
        g=a['Ma'][w]-a['qfrc_smooth'][w]-j.T@force[w,:used]
        scale=np.diag(mass[w])+np.sum(j*j*a['D'][w,:used,None],axis=0)
        search.append(-g/scale)
    a['search']=np.array(search,np.float32)
    if variant=='zero-search':a['search'][:]=0.
    elif variant=='uphill-search':a['search']*=-1.
    elif variant=='small-search':a['search']*=1.e-6
    a['search_dot']=np.sum(a['search'].astype(np.float64)**2,axis=1).astype(np.float32)
    a['overflow']=np.full(2,2,np.int32)  # unrelated flags must survive atomic OR
    a['tolerance']=np.array([1.e-8],np.float32);a['ls_tolerance']=np.array([.01],np.float32)
    a['inertia']=np.array([np.mean(np.diagonal(mass,axis1=1,axis2=2))],np.float32)
    a['ls_iterations']=np.array(0 if variant=='zero-iterations' else 1 if variant=='one-iteration' else 50)
    if variant=='per-world-scalars':
        for k in ('tolerance','ls_tolerance','inertia'):a[k]=np.r_[a[k],a[k]*2].astype(np.float32)
    return dict(name=f'nv{nv}-{variant}',origin='manufactured physical cone/QP local boundary',arrays=a)


def cases(args):
    from libero.gpu.host.runtime.newton.mjwarp_direction import FAMILIES
    cs=[dict(name=f'task{t}-frozen-boundary',origin='raw upstream pre-line-search boundary',
        arrays=dict(np.load(args.output/'frozen'/f'task{t}.npz',allow_pickle=False))) for t in (4,0)]
    cs += [synthetic(nv) for nv in FAMILIES]
    cs += [synthetic(39,v) for v in ('world1-done','all-done','empty-world0','top-boundary','bottom-boundary',
        'zero-tangent','friction-boundary','broadcast-impratio','zero-search','uphill-search','small-search',
        'zero-iterations','one-iteration','per-world-scalars')]
    cs += [synthetic(57,v) for v in ('world1-done','empty-world0','zero-search','zero-iterations')]
    return cs


def matrix(a):
    n,nv=a['qacc'].shape if 'qacc' in a else (2,int(a['nv']))
    result=np.zeros((n,nv,nv),np.float64)
    for i in range(nv):
        start,end=a['M_mulm_rowadr'][i:i+2]
        result[:,i,a['M_mulm_col'][start:end]]=a['M'][:,a['M_mulm_madr'][start:end]]
    return result


def objective(a,w,alpha,mass,jv):
    """Independent FP64 grouped physical cone energy on the exact input ray."""
    used=int(a['nefc'][w]);ne,nf=int(a['ne'][w]),int(a['nf'][w])
    x=a['Jaref'][w].astype(np.float64)+alpha*jv[w];d=a['D'][w].astype(np.float64)
    v=a['search'][w].astype(np.float64)
    smooth=alpha*(v@(a['Ma'][w].astype(np.float64)-a['qfrc_smooth'][w]))+.5*alpha*alpha*(v@mass[w]@v)
    cost=.5*np.sum(d[:ne]*x[:ne]**2)
    for row in range(ne,ne+nf):
        loss=float(a['frictionloss'][w,row]);rf=loss/(d[row] if d[row] else 1.e-15)
        cost += .5*d[row]*x[row]**2 if abs(x[row])<rf else loss*(abs(x[row])-.5*rf)
    contacts=set()
    for row in range(ne+nf,used):
        if a['type'][w,row]==7:contacts.add(int(a['id'][w,row]))
        elif x[row]<0:cost+=.5*d[row]*x[row]**2
    for c in contacts:
        dim=int(a['contact_dim'][c]);ids=a['contact_efc_address'][c,:dim]
        f=a['contact_friction'][c,:dim-1].astype(np.float64)
        mu=float(f[0])*float(a['impratio_invsqrt'][w % a['impratio_invsqrt'].size])
        n=x[ids[0]]*mu;t=np.linalg.norm(x[ids[1:]]*f)
        if n>=mu*t:pass
        elif mu*n+t<=0:cost+=.5*np.sum(d[ids]*x[ids]**2)
        else:cost+=.5*d[ids[0]]/(mu*mu*(1+mu*mu))*(n-mu*t)**2
    return float(smooth+cost)


class Replay:
    def __init__(self,case,library,stream,products_only=False):
        import warp as wp
        from mujoco_warp._src import types
        from libero.gpu.host.runtime.newton.mjwarp_linesearch import LinesearchPlan
        self.case,self.stream,self.products_only=case,stream,products_only
        a=case['arrays'];nv=int(a['nv']);n,rows,pad=a['J'].shape;cap=a['contact_dim'].size
        dev={k:wp.array(np.ascontiguousarray(v),dtype=types.vec5 if k=='contact_friction' else
            wp.bool if v.dtype==np.bool_ else wp.int32 if v.dtype==np.int32 else wp.float32,device=stream.device)
            for k,v in a.items() if k not in ('nv','ls_iterations')}
        opt=NS(cone=types.ConeType.ELLIPTIC,solver=types.SolverType.NEWTON,enableflags=0,
            integrator=types.IntegratorType.EULER,ls_iterations=int(a['ls_iterations']),warn_overflow=0,
            **{k:dev[k] for k in ('tolerance','ls_tolerance','impratio_invsqrt')})
        self.model=NS(nv=nv,nv_pad=pad,is_sparse=False,opt=opt,stat=NS(meaninertia=dev['inertia']),
            block_dim=NS(linesearch_iterative=32),**{k:dev[k] for k in ('M_mulm_rowadr','M_mulm_col','M_mulm_madr')})
        self.inputs,self.data,self.contexts,self.graphs,self.restores,self.outputs=dev,{},{},{},{},{}
        for mode in ('upstream','cuda'):
            output={k:wp.clone(dev[k]) for k in ('qacc','Ma','Jaref','overflow')}
            output.update(mv=wp.full((n,nv),17.,dtype=float,device=stream.device),
                jv=wp.full((n,rows),19.,dtype=float,device=stream.device),
                quad=wp.full((n,rows),wp.vec3(23.),dtype=wp.vec3,device=stream.device),
                alpha=wp.full(n,-3.,dtype=float,device=stream.device),
                improvement=wp.full(n,-4.,dtype=float,device=stream.device))
            d=NS(nworld=n,njmax=rows,naconmax=cap,M=dev['M'],ne=dev['ne'],nf=dev['nf'],nefc=dev['nefc'],
                nacon=dev['nacon'],qfrc_smooth=dev['qfrc_smooth'],qacc=output['qacc'],overflow=output['overflow'],
                contact=NS(dim=dev['contact_dim'],friction=dev['contact_friction'],efc_address=dev['contact_efc_address']),
                efc=NS(J=dev['J'],J_rownnz=None,J_rowadr=None,J_colind=None,Ma=output['Ma'],
                    **{k:dev[k] for k in ('D','frictionloss','type','id')}),dof_cdof=None)
            ctx=NS(done=dev['done'],search=dev['search'],search_dot=dev['search_dot'],
                ls_exhausted=wp.zeros(n,dtype=bool,device=stream.device),compact_d_full=None,compact_m_full=None,
                search_unchanged=wp.zeros(n,dtype=bool,device=stream.device),
                **{k:output[k] for k in ('Jaref','mv','jv','quad','alpha','improvement')})
            self.data[mode],self.contexts[mode],self.outputs[mode]=d,ctx,output
            self.restore_sources=getattr(self,'restore_sources',{})
            sources={k:wp.clone(v) for k,v in output.items()};self.restore_sources[mode]=sources
            with wp.ScopedCapture(stream=stream) as capture:
                for k,v in output.items():wp.copy(v,sources[k],stream=stream)
            self.restores[mode]=capture.graph
            if mode=='cuda':self.plan=LinesearchPlan(self.model,d,library,stream);self.binding=self.plan.bind(ctx)
            with wp.ScopedCapture(stream=stream) as capture:self.enqueue(mode)
            self.graphs[mode]=capture.graph

    def enqueue(self,mode):
        from mujoco_warp._src import solver,support
        import warp as wp
        if mode=='cuda':
            self.binding.enqueue(1 if self.products_only else 0)
            if self.products_only and self.model.nv<=50:self.binding.enqueue(3)
        elif not self.products_only:solver._linesearch(self.model,self.data[mode],self.contexts[mode])
        else:
            d,c,m=self.data[mode],self.contexts[mode],self.model
            support.mul_m(m,d,c.mv,c.search,skip=c.done)
            chunk=20 if m.nv>50 else 50;parts=(m.nv+chunk-1)//chunk
            if parts>1:wp.launch(solver._linesearch_zero_jv,dim=(d.nworld,d.njmax),inputs=[d.nefc,c.done],outputs=[c.jv])
            wp.launch(solver._linesearch_jv_fused_kernel(False,m.nv,chunk,False),dim=(d.nworld,d.njmax,parts),
                inputs=[d.nefc,None,None,None,d.efc.J,None,c.search,c.done],outputs=[c.jv])

    def run(self,mode):
        import warp as wp
        wp.capture_launch(self.restores[mode],stream=self.stream)
        wp.capture_launch(self.graphs[mode],stream=self.stream);wp.synchronize_stream(self.stream)
        return {k:v.numpy() for k,v in self.outputs[mode].items()}

    def check(self):
        from validation.mjwarp_policy import compare_replays
        a=self.case['arrays'];nv=int(a['nv']);eps=np.finfo(np.float32).eps
        out={k:self.run(k) for k in ('upstream','cuda')};out['raw-b']=self.run('upstream')
        mass=matrix(a);mv=np.einsum('wij,wj->wi',mass,a['search'].astype(np.float64))
        j=a['J'][:,:,:nv].astype(np.float64);jv=np.einsum('wri,wi->wr',j,a['search'].astype(np.float64))
        gamma=nv*eps/(1.-nv*eps)
        mb=8*gamma*np.einsum('wij,wj->wi',abs(mass),abs(a['search'].astype(np.float64)))
        jb=8*gamma*np.einsum('wri,wi->wr',abs(j),abs(a['search'].astype(np.float64)))
        checks=[]
        for mode in ('upstream','cuda'):
            o=out[mode];worlds=[]
            for w in range(2):
                if a['done'][w]:
                    held=all(np.array_equal(v[w],self.restore_sources[mode][k].numpy()[w]) for k,v in o.items())
                    worlds.append(dict(world=w,held=held,passed=held));continue
                used=int(a['nefc'][w]);me=abs(o['mv'][w]-mv[w]);je=abs(o['jv'][w,:used]-jv[w,:used])
                passed=bool(np.all(me<=mb[w]) and np.all(je<=jb[w,:used]) and np.all(o['jv'][w,used:]==19.))
                row=dict(world=w,products_passed=passed,mv_max_error=float(me.max()),
                    jv_max_error=float(je.max(initial=0.)),alpha=float(o['alpha'][w]))
                if not self.products_only:
                    alpha=float(o['alpha'][w]);initial=objective(a,w,0.,mass,jv);after=objective(a,w,alpha,mass,jv)
                    # FP32 input-operation majorant: cone energy plus smooth
                    # ray terms, amplified for row and dot-product reductions.
                    v=a['search'][w].astype(np.float64)
                    ray=abs(alpha)*float(abs(v)@(abs(a['Ma'][w])+abs(a['qfrc_smooth'][w])))
                    ray+=.5*alpha*alpha*float(abs(v)@abs(mass[w])@abs(v))
                    energy_bound=64*eps*(nv+used+32)*max(abs(initial)+abs(after)+ray,1.e-30)
                    gain=initial-after;gain_error=abs(float(o['improvement'][w])-gain)
                    descent=after<=initial+energy_bound and gain_error<=energy_bound
                    qref=a['qacc'][w].astype(np.float64)+alpha*a['search'][w]
                    qbound=4*eps*(abs(a['qacc'][w])+abs(alpha)*abs(a['search'][w]))
                    mref=a['Ma'][w].astype(np.float64)+alpha*o['mv'][w]
                    ml=4*eps*(abs(a['Ma'][w])+abs(alpha)*abs(o['mv'][w]))
                    jr=a['Jaref'][w,:used].astype(np.float64)+alpha*o['jv'][w,:used]
                    jl=4*eps*(abs(a['Jaref'][w,:used])+abs(alpha)*abs(o['jv'][w,:used]))
                    commit=bool(np.all(abs(o['qacc'][w]-qref)<=qbound) and np.all(abs(o['Ma'][w]-mref)<=ml)
                        and np.all(abs(o['Jaref'][w,:used]-jr)<=jl)
                        and np.array_equal(o['Jaref'][w,used:],a['Jaref'][w,used:]))
                    row.update(independent_energy_change=after-initial,reported_improvement=float(o['improvement'][w]),
                        improvement_error=gain_error,energy_error_bound=energy_bound,descent=bool(descent),commit=commit,
                        overflow=int(o['overflow'][w]));passed &= descent and commit
                row['passed']=bool(passed);worlds.append(row)
            checks.append(dict(mode=mode,worlds=worlds,passed=all(w['passed'] for w in worlds)))
        fields=('mv','jv') if self.products_only else ('qacc','Ma','Jaref','alpha','improvement')
        equivalence=compare_replays({k:out['cuda'][k] for k in fields},{k:out['upstream'][k] for k in fields},
                                   {k:out['raw-b'][k] for k in fields})
        flags=bool(np.array_equal(out['cuda']['overflow'],out['upstream']['overflow']))
        holds=all(np.array_equal(np.ascontiguousarray(v.numpy()).view(np.uint8),np.ascontiguousarray(a[k]).view(np.uint8)) for k,v in self.inputs.items())
        result=dict(name=self.case['name'],nv=nv,worlds=2,rows=a['nefc'].tolist(),origin=self.case['origin'],
            checks=checks,equivalence=equivalence,flags_equal=flags,input_holds=holds,
            passed=bool(all(c['passed'] for c in checks) and all(c['passed'] for c in equivalence) and flags and holds))
        return result,out

    def benchmark(self,args):
        import warp as wp
        result={}
        for mode,graph in self.graphs.items():
            for _ in range(args.warmup):
                wp.capture_launch(self.restores[mode],stream=self.stream);wp.capture_launch(graph,stream=self.stream)
            values=[];start=wp.Event(enable_timing=True);end=wp.Event(enable_timing=True)
            for _ in range(args.steps):
                wp.capture_launch(self.restores[mode],stream=self.stream)
                self.stream.record_event(start);wp.capture_launch(graph,stream=self.stream);self.stream.record_event(end)
                wp.synchronize_event(end);values.append(wp.get_event_elapsed_time(start,end,synchronize=False)*1000)
            result[mode]=dict(mean_us=float(np.mean(values)),std_us=float(np.std(values)),microseconds=values)
        return dict(name=self.case['name'],calls_per_sample=1,warmup=args.warmup,samples=args.steps,
            **result,speedup=result['upstream']['mean_us']/result['cuda']['mean_us'])


def main(argv=None,*,args=None):
    if args is None:args=add_arguments(argparse.ArgumentParser(description=__doc__)).parse_args(argv)
    if args.warmup<0 or args.steps<1:raise ValueError('invalid timing counts')
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
        signature=dict(sources=source_files(include_linesearch=True),input_sha256=digest(args.input),
            library_sha256=digest(args.library),tool_sha256=digest(__file__),products_only=args.products_only,
            frozen_receipt_sha256=digest(args.output/'frozen/receipt.json'))
        prepared=args.output/('products-preparation.json' if args.products_only else 'preparation.json')
        if not args.prepare and (not prepared.is_file() or json.loads(prepared.read_text())['signature']!=signature):
            raise RuntimeError('prepare these exact line-search graphs offline first')
        replays=[Replay(c,args.library,stream,args.products_only) for c in cases(args)]
        if args.prepare:
            wp.synchronize_stream(stream)
            save_json(prepared,dict(signature=signature,compilation_events=scope['events'],physics_integrations=0,
                local_graph_replays=0,cases=len(replays)))
            print(json.dumps(dict(prepared=True,cases=len(replays),compilations=len(scope['events']))),flush=True);return 0
        before=gpu_load_snapshot();rows=[]
        for r in replays:
            row,out=r.check();rows.append(row)
            np.savez(args.output/(('products-' if args.products_only else '')+row['name']+'.npz'),
                     **{mode+'.'+k:v for mode,values in out.items() for k,v in values.items()})
            print(json.dumps(dict(name=row['name'],passed=row['passed'])),flush=True)
        passed=all(r['passed'] for r in rows)
        timing=[r.benchmark(args) for r in replays if r.case['name'].startswith('task')] if passed and args.performance else None
        save_json(args.output/('products-summary.json' if args.products_only else 'summary.json'),dict(passed=passed,
            signature=signature,cases=rows,performance=timing,runtime_compilations=scope['events'],physics_integrations=0,
            gpu_load_before=before,gpu_load_after=gpu_load_snapshot(),
            policy='independent FP64 products/physical cone energy/commit roundoff; unchanged same-engine FP32 envelope'))
        print(json.dumps(dict(passed=passed,performance=timing)),flush=True)
        return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())
