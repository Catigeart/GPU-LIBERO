"""Frozen N=2 factor/factor-solve checks and independent FP64 residuals."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_arguments(p):
    for name in ('build-root','library','catalogue','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--performance',action='store_true')
    p.add_argument('--phase',choices=('factor','solve'),default='factor')
    p.add_argument('--warmup',type=int,default=5);p.add_argument('--steps',type=int,default=20)
    return p


def freeze(args,stream):
    import warp as wp
    import mujoco_warp as mjw
    from mujoco_warp._src import forward,solver
    from libero.libero import get_libero_path
    from libero.gpu.host.backends.mjwarp.model import build_scene,capacity_budget
    rows=json.loads(args.catalogue.read_text())['rows']
    for task_id in (0,3):
        path=args.output/f'task{task_id}.npz'
        if path.exists():
            if json.loads(path.with_suffix('.json').read_text())['sha256']!=digest(path):raise RuntimeError('Frozen matrix input changed')
            continue
        row=next(r for r in rows if r['suite']=='libero_10' and r['task_id']==task_id)
        built,m,bank,task,identity=build_scene(args.build_root,str(Path(get_libero_path('bddl_files'))/row['task']),0,
            dict(use_camera_obs=False,has_renderer=False,has_offscreen_renderer=False,hard_reset=False,ignore_done=True))
        budget=capacity_budget(built.native_model)
        d=mjw.make_data(built.native_model,nworld=2,nccdmax=budget['nconmax'],**budget)
        d.time.assign(bank[:2,0].astype(np.float32));d.qpos.assign(np.ascontiguousarray(bank[:2,1:1+m.nq],np.float32))
        d.qvel.assign(np.ascontiguousarray(bank[:2,1+m.nq:],np.float32));roots={k:getattr(d,k).numpy() for k in ('time','qpos','qvel')}
        forward.fwd_position(m,d);forward.fwd_velocity(m,d);forward.fwd_actuation(m,d)
        forward.fwd_acceleration(m,d);solver.solve(m,d)
        effective=wp.clone(d.M);damping=wp.empty((2,m.nv),dtype=float,device=stream.device)
        wp.launch(forward._compute_damping_deriv,dim=(2,m.nv),inputs=[m.dof_damping,m.dof_dampingpoly,d.qvel],outputs=[damping])
        wp.launch(forward._euler_damp_qfrc,dim=(2,m.nv),inputs=[m.opt.timestep,m.M_rownnz,m.M_rowadr,damping],outputs=[effective])
        wp.synchronize_stream(stream)
        arrays=dict(nv=np.array(m.nv),nC=np.array(m.nC),factor_total=np.array(m.qLD_block_total),
            matrix=d.M.numpy(),euler_matrix=effective.numpy(),rhs=d.efc.Ma.numpy(),
            damping=damping.numpy(),timestep=m.opt.timestep.numpy(),address=m.qLD_block_adr.numpy(),
            row=m.M_rowadr.numpy(),nnz=m.M_rownnz.numpy(),column=m.M_colind.numpy())
        blocks=[]
        for i,t in enumerate(m.M_tiles):
            arrays[f'starts{i}']=t.adr.numpy();arrays[f'size{i}']=np.array(t.size)
            arrays[f'gather{i}']=t.elemid.numpy();arrays[f'kind{i}']=np.array(int(t.elemid.size!=0))
            for start in arrays[f'starts{i}']:
                adr=int(arrays['address'][start]);blocks.append(dict(kind='compact' if adr==-2 else
                    'tile' if t.elemid.size else 'scalar',size=t.size,start=int(start),factor_address=adr))
        if (m.nv!=row['nv'] or m.nC!=row['nC'] or m.qLD_block_total!=row['factor_total'] or
                sorted(blocks,key=lambda b:b['start'])!=sorted(row['blocks'],key=lambda b:b['start'])):
            raise RuntimeError('Frozen model block layout differs from 130-task catalogue')
        if not all(np.array_equal(getattr(d,k).numpy(),v) for k,v in roots.items()):raise RuntimeError('Freeze advanced physical roots')
        np.savez(path,**arrays)
        path.with_suffix('.json').write_text(json.dumps(dict(task=row['task'],identity=identity,sha256=digest(path),
            catalogue_sha256=digest(args.catalogue),physics_integrations=0,newton_solves=1,
            origin='raw upstream initial-state forward, then Euler matrix preparation only; no advance'),indent=2)+'\n')
        print(json.dumps(dict(frozen=str(path),nv=m.nv,nC=m.nC)),flush=True)
        del built,m,d


def synthetic(layout,name,*,condition=30.):
    """Build CSR independently of gather maps, then classify three block families."""
    rng=np.random.default_rng(7601+sum(x[1] for x in layout));nv=sum(x[1] for x in layout)
    dense=np.zeros((2,nv,nv),np.float64);address=np.empty(nv,np.int32);blocks=[];start=0;total=0
    for kind,size in layout:
        if kind=='compact':
            for w in range(2):dense[w,start:start+size,start:start+size]=np.diag(np.geomspace(1,3,size)*(w+1))
            adr=-2
        else:
            for w in range(2):
                q,_=np.linalg.qr(rng.normal(size=(size,size)))
                a=q@np.diag(np.geomspace(1,condition,size))@q.T
                if kind=='sparse9':
                    a=np.eye(size)*5+rng.normal(scale=.05,size=(size,size));a=(a+a.T)/2;a[8,7]=a[7,8]=0.
                dense[w,start:start+size,start:start+size]=a*(1.e-3 if w==0 else 1.e3)
            adr=total;total+=size*size
        address[start:start+size]=adr;blocks.append((kind,size,start));start+=size
    rows=[];columns=[];values=[[],[]];pair={};group={}
    for kind,size,start in blocks:
        group.setdefault((0 if kind in ('compact','scalar') else 1,size),[]).append(start)
        for i in range(size):
            rows.append(len(columns))
            for j in range(i+1):
                if kind=='compact' and i!=j:continue
                if kind=='sparse9' and (i,j)==(8,7):continue
                pair[start+i,start+j]=len(columns);columns.append(start+j)
                for w in range(2):values[w].append(dense[w,start+i,start+j])
    row=np.array(rows,np.int32);nc=len(columns);nnz=np.diff(np.r_[row,nc]).astype(np.int32)
    c=dict(name=name,nv=nv,nC=nc,factor_total=total,matrix=np.array(values,np.float32),
        rhs=rng.normal(size=(2,nv)).astype(np.float32),address=address,row=row,nnz=nnz,column=np.array(columns,np.int32),
        alias=False,padding=0,origin='manufactured SPD blocks with independent CSR')
    for k,((kind,size),starts) in enumerate(group.items()):
        c[f'starts{k}']=np.array(starts,np.int32);c[f'size{k}']=np.array(size);c[f'kind{k}']=np.array(kind)
        gather=[pair.get((max(s+i,s+j),min(s+i,s+j)),-1) for s in starts for i in range(size) for j in range(size)]
        c[f'gather{k}']=np.array(gather,np.int32) if kind else np.empty(0,np.int32)
    return c


def cases(output):
    result=[]
    for task_id in (0,3):
        with np.load(output/f'task{task_id}.npz',allow_pickle=False) as saved:source={k:saved[k].copy() for k in saved.files}
        for phase in ('matrix','euler'):
            c={k:v.copy() for k,v in source.items()};c.update(name=f'task{task_id}-{phase}',alias=False,padding=0,origin='real frozen upstream boundary')
            for key in ('nv','nC','factor_total'):c[key]=int(c[key])
            if phase=='euler':c['matrix']=c['euler_matrix'].copy()
            result.append(c)
    layouts=[(('compact',1),),(('scalar',6),),(('sparse9',9),),
             (('compact',1),('scalar',6),('sparse9',9),('scalar',6)),(('compact',6),('scalar',1),('tile',9))]
    for i,layout in enumerate(layouts):
        source=synthetic(layout,f'synthetic{i}')
        for variant in ('ordinary','alias','padded'):
            c={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in source.items()}
            c.update(name=source['name']+'-'+variant,alias=variant=='alias',padding=3 if variant=='padded' else 0);result.append(c)
    result.append(synthetic((('tile',9),),'synthetic-conditioned',condition=1.e4))
    c=synthetic((('scalar',6),('sparse9',9)),'synthetic-zero');c['rhs'][:]=0;result.append(c)
    return result


def dense_reference(c):
    a=np.zeros((2,c['nv'],c['nv']),np.float64)
    rows=np.repeat(np.arange(c['nv']),c['nnz']);cols=c['column']
    a[:,rows,cols]=c['matrix'];a[:,cols,rows]=c['matrix']
    return a


class Replay:
    def __init__(self,c,args,stream):
        import warp as wp
        from mujoco_warp._src import types
        from libero.gpu.host.runtime.dynamics.mjwarp_mass_factor import MassFactorPlan
        arr=lambda a,dtype:wp.array(np.ascontiguousarray(a),dtype=dtype,device=stream.device)
        self.case,self.stream=c,stream
        self.model=m=NS(nv=c['nv'],nC=c['nC'],qLD_block_total=c['factor_total'],
            M_rowadr=arr(c['row'],int),M_rownnz=arr(c['nnz'],int),M_colind=arr(c['column'],int),
            qLD_block_adr=arr(c['address'],int),block_dim=types.BlockDim(),M_tiles=[])
        i=0
        while f'starts{i}' in c:
            m.M_tiles.append(types.TileSet(adr=arr(c[f'starts{i}'],int),size=int(c[f'size{i}']),elemid=arr(c[f'gather{i}'],int)));i+=1
        self.data=d=NS(nworld=2,M=arr(c['matrix'],float),qLD=wp.empty((2,c['factor_total']),dtype=float,device=stream.device))
        self.rhs=arr(np.pad(c['rhs'],((0,0),(0,c['padding'])),constant_values=37.),float)
        self.outputs,self.bindings,self.graphs,self.timing={},{},{},{}
        self.plan=MassFactorPlan(m,d,args.library,stream)
        for phase in ('factor','solve'):
            for mode in ('upstream','cuda'):
                key=phase+'-'+mode
                factor=wp.full((2,c['factor_total']),37.,dtype=float,device=stream.device)
                diagonal=wp.full((2,m.nv),37.,dtype=float,device=stream.device)
                x=self.rhs if c['alias'] else wp.full((2,m.nv+c['padding']),37.,dtype=float,device=stream.device)
                self.outputs[key]=(factor,diagonal,x)
                self.bindings[key]=self.plan.bind(d.M,factor,diagonal,x if phase=='solve' else None,self.rhs if phase=='solve' else None)
                with wp.ScopedCapture(stream=stream) as capture:self.enqueue(phase,mode)
                self.graphs[key]=capture.graph
                if c['name'].startswith('task') and ((phase=='factor' and c['name'].endswith('-matrix')) or
                                                      (phase=='solve' and c['name'].endswith('-euler'))):
                    with wp.ScopedCapture(stream=stream) as capture:
                        for _ in range(32):self.enqueue(phase,mode)
                    self.timing[key]=capture.graph

    def enqueue(self,phase,mode):
        from mujoco_warp._src import smooth
        key=phase+'-'+mode;factor,diagonal,x=self.outputs[key]
        if mode=='cuda':self.bindings[key].enqueue()
        elif phase=='factor':smooth._factor_blocks(self.model,self.data,self.data.M,factor,diagonal)
        else:smooth.factor_solve_i(self.model,self.data,self.data.M,factor,diagonal,x,self.rhs)

    def check(self,phase,updated):
        import warp as wp
        from validation.precision import balance_residual
        c={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in self.case.items()}
        if updated:c['matrix']*=np.float32(1.25);c['rhs']*=np.float32(-.5)
        self.data.M.assign(c['matrix']);nv=c['nv'];seed=np.pad(c['rhs'],((0,0),(0,c['padding'])),constant_values=37.)
        outputs={}
        for mode in ('upstream','cuda'):
            key=phase+'-'+mode;factor,diagonal,x=self.outputs[key]
            factor.fill_(37.);diagonal.fill_(37.);self.rhs.assign(seed)
            if not c['alias']:x.fill_(37.)
            wp.capture_launch(self.graphs[key],stream=self.stream);wp.synchronize_stream(self.stream)
            outputs[mode]=dict(factor=factor.numpy(),diagonal=diagonal.numpy(),solution=x.numpy(),rhs=self.rhs.numpy())
        a=dense_reference(c);checks=[];mask=np.zeros((c['factor_total'],),bool);dmask=np.zeros(nv,bool);i=0
        unit=np.finfo(np.float32).eps/2
        while f'starts{i}' in c:
            size=int(c[f'size{i}']);kind=int(c[f'kind{i}'])
            for first in c[f'starts{i}']:
                first=int(first);sl=slice(first,first+size);adr=int(c['address'][first])
                if adr==-2:dmask[sl]=True
                else:
                    r,col=np.triu_indices(size);mask[adr+r*size+col]=True
                    if kind:mask[adr:adr+size*size]=True
                for w in range(2):
                    block=a[w,sl,sl];gamma=size*unit/(1-size*unit)
                    for mode,out in outputs.items():
                        if adr==-2:reconstruction=np.diag(1/out['diagonal'][w,sl].astype(np.float64));positive=np.all(out['diagonal'][w,sl]>0)
                        else:
                            u=np.triu(out['factor'][w,adr:adr+size*size].reshape(size,size)).astype(np.float64)
                            reconstruction=u.T@u;positive=np.all(np.diag(u)>0)
                        error=float(np.linalg.norm(block-reconstruction,np.inf)/max(np.linalg.norm(block,np.inf),np.finfo(np.float64).tiny))
                        row=dict(mode=mode,world=w,start=first,size=size,factor_backward_error=error,factor_limit=8*gamma,
                                 passed=bool(positive and np.isfinite(error) and error<=8*gamma))
                        if phase=='solve':
                            b=c['rhs'][w,sl].astype(np.float64);x=out['solution'][w,sl]
                            res=balance_residual(block,x,b,np.zeros(size));ref=np.linalg.solve(block,b)
                            condition=float(np.linalg.cond(block,np.inf));eta=8*gamma
                            if condition*eta>=1:raise RuntimeError('Input cannot qualify forward sensitivity')
                            limit=2*condition*eta/(1-condition*eta)*max(float(abs(ref).max()),np.finfo(np.float32).tiny)
                            forward_error=float(abs(x-ref).max())
                            row.update(solve=res,forward_error=forward_error,forward_limit=limit,condition=condition)
                            row['passed'] &= bool(res['passed'] and forward_error<=limit)
                        checks.append(row)
            i+=1
        holds=np.array_equal(self.data.M.numpy(),c['matrix']);differences={}
        for mode,out in outputs.items():
            holds &= np.all(out['factor'][:,~mask]==37.) and np.all(out['diagonal'][:,~dmask]==37.)
            if not (phase=='solve' and c['alias']):holds &= np.array_equal(out['rhs'],seed)
            holds &= np.all(out['solution'][:,nv:]==37.)
            if phase=='factor' and not c['alias']:holds &= np.all(out['solution']==37.)
            i=0
            while f'starts{i}' in c:
                size=int(c[f'size{i}'])
                if c[f'kind{i}']:
                    r,col=np.tril_indices(size,-1)
                    for first in c[f'starts{i}']:
                        adr=int(c['address'][first]);holds &= np.all(out['factor'][:,adr+r*size+col]==0.)
                i+=1
        for key in ('factor','diagonal','solution'):
            if phase=='factor' and key=='solution':continue
            dif=abs(outputs['cuda'][key]-outputs['upstream'][key]);differences[key]=float(dif.max(initial=0.))
        return dict(name=c['name'],phase=phase,updated=updated,holds=bool(holds),checks=checks,differences=differences,
                    passed=bool(holds and all(x['passed'] for x in checks))),outputs

    def benchmark(self,args,phase):
        import warp as wp
        self.data.M.assign(self.case['matrix']);self.rhs.assign(self.case['rhs']);result={}
        for mode in ('upstream','cuda'):
            graph=self.timing[phase+'-'+mode]
            for _ in range(args.warmup):wp.capture_launch(graph,stream=self.stream)
            start,end=wp.Event(enable_timing=True),wp.Event(enable_timing=True);values=[]
            for _ in range(args.steps):
                self.stream.record_event(start);wp.capture_launch(graph,stream=self.stream);self.stream.record_event(end)
                wp.synchronize_event(end);values.append(wp.get_event_elapsed_time(start,end,synchronize=False)*1000/32)
            result[mode]=dict(mean_us=float(np.mean(values)),std_us=float(np.std(values)),microseconds=values)
        result.update(name=self.case['name'],phase=phase,warmup=args.warmup,samples=args.steps,calls_per_sample=32,
                      speedup=result['upstream']['mean_us']/result['cuda']['mean_us'])
        return result


def main(argv=None,*,args=None):
    if args is None:args=add_arguments(argparse.ArgumentParser(description=__doc__)).parse_args(argv)
    if args.warmup<0 or args.steps<1:raise ValueError('Invalid measurement counts')
    from libero.gpu import precision
    precision.select('fp32')
    import warp as wp
    from warp._src.build import init_kernel_cache
    from libero.gpu.host.backends.mjwarp.model import require_upstream
    from libero.gpu.host.backends.mjwarp.build import source_files
    from libero.gpu.host.build.compilation import compilation_scope,install_warp_guard
    from validation.common import save_json
    from validation.runner import gpu_load_snapshot
    require_upstream();init_kernel_cache(args.build_root/'mjwarp/warp-cache');install_warp_guard();wp.init()
    args.output.mkdir(parents=True,exist_ok=True);stream=wp.Stream('cuda:0');prepared=args.output/'preparation.json'
    with compilation_scope(args.prepare) as scope,wp.ScopedStream(stream):
        if args.prepare:freeze(args,stream)
        # Local admission binds actual leaf/freeze dependencies. Unrelated host
        # stage scheduling edits do not invalidate frozen mathematical inputs.
        sources=source_files(include_factor=True)
        sources={k:v for k,v in sources.items() if k.startswith('upstream/') or 'mjwarp_mass_factor' in k or
            k in ('cmake/MjwarpMassFactor.cmake','host/backends/mjwarp/model.py','project_source_identity','data/mjwarp_source_pin.json')}
        signature=dict(sources=sources,tool_sha256=digest(__file__),library_sha256=digest(args.library),
            catalogue_sha256=digest(args.catalogue),inputs={f'task{i}':digest(args.output/f'task{i}.npz') for i in (0,3)},
            receipts={f'task{i}':digest(args.output/f'task{i}.json') for i in (0,3)})
        if not args.prepare and (not prepared.exists() or json.loads(prepared.read_text())['signature']!=signature):
            raise RuntimeError('Prepare exact factor operators offline first')
        replays=[Replay(c,args,stream) for c in cases(args.output)]
        if args.prepare:
            wp.synchronize_stream(stream);save_json(prepared,dict(signature=signature,compilation_events=scope['events'],physics_integrations=0))
            print(json.dumps(dict(prepared=True,cases=len(replays),compilations=len(scope['events']))),flush=True);return 0
        before=gpu_load_snapshot();rows=[]
        if args.phase=='solve':
            first=json.loads((args.output/'factor-summary.json').read_text())
            if not first['passed'] or first['signature']!=signature:raise RuntimeError('Qualify factorization first on the same inputs')
        for replay in replays:
            if args.phase=='factor' and (replay.case['alias'] or replay.case['name'].endswith('-zero')):continue
            for updated in (False,True):
                row,outputs=replay.check(args.phase,updated);rows.append(row)
                np.savez(args.output/(row['name']+'-'+args.phase+('-updated' if updated else '')+'-output.npz'),
                    **{mode+'.'+k:v for mode,out in outputs.items() for k,v in out.items()})
                print(json.dumps(dict(name=row['name'],phase=args.phase,updated=updated,passed=row['passed'])),flush=True)
        passed=all(x['passed'] for x in rows);timing=None
        if args.performance and passed:
            if args.phase!='solve':raise ValueError('Time both leaves once after factor and solve pass')
            timing=[x.benchmark(args,phase) for phase in ('factor','solve') for x in replays if phase+'-upstream' in x.timing]
        result=dict(passed=passed,signature=signature,cases=rows,performance=timing,runtime_compilations=scope['events'],
            physics_integrations=0,gpu_load_before=before,gpu_load_after=gpu_load_snapshot(),
            policy='existing 8 gamma_block factor reconstruction/solve residual; 2*kappa*eta/(1-kappa*eta) solve sensitivity; exact hold contracts')
        save_json(args.output/(args.phase+'-summary.json'),result)
        print(json.dumps(dict(passed=passed,phase=args.phase,performance=timing)),flush=True);return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())
