"""Frozen FP32 solve_m parity, independent residuals and local timing (N=2).

--prepare explicitly freezes two real upstream boundaries and admits kernels
offline. Ordinary replay cannot compile or advance physics. Timing uses 32
idempotent calls per sample to resolve these microsecond-scale functions.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np


def add_arguments(parser):
    for key in ('build-root', 'library', 'output', 'catalogue'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--performance', action='store_true')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--steps', type=int, default=20)
    return parser


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(args, stream):
    """Upstream position/factor/velocity/forces only; no Newton or integration."""
    import warp as wp
    import mujoco_warp as mjw
    from mujoco_warp._src import forward
    from libero.libero import get_libero_path
    from libero.gpu.host.backends.mjwarp.model import build_scene, capacity_budget
    catalogue=json.loads(args.catalogue.read_text())
    receipts=[]
    for task_id in (0,3):
        path=args.output/f'task{task_id}.npz'
        if path.exists():
            receipt=json.loads(path.with_suffix('.json').read_text())
            if receipt['sha256']!=digest(path):raise RuntimeError('Frozen mass input changed')
            receipts.append(receipt)
            continue
        row=next(r for r in catalogue['rows'] if r['suite']=='libero_10' and r['task_id']==task_id)
        bddl=str(Path(get_libero_path('bddl_files'))/row['task'])
        built,m,bank,task,identity=build_scene(args.build_root,bddl,0,dict(
            use_camera_obs=False,has_renderer=False,has_offscreen_renderer=False,hard_reset=False,ignore_done=True))
        budget=capacity_budget(built.native_model)
        d=mjw.make_data(built.native_model,nworld=2,nccdmax=budget['nconmax'],**budget)
        states=bank[:2]
        d.time.assign(states[:,0].astype(np.float32))
        d.qpos.assign(np.ascontiguousarray(states[:,1:1+m.nq],dtype=np.float32))
        d.qvel.assign(np.ascontiguousarray(states[:,1+m.nq:],dtype=np.float32))
        roots={k:getattr(d,k).numpy() for k in ('time','qpos','qvel')}
        forward.fwd_position(m,d)
        forward.fwd_velocity(m,d)
        forward.fwd_actuation(m,d)
        forward.fwd_acceleration(m,d)
        wp.synchronize_stream(stream)
        blocks=[];address=m.qLD_block_adr.numpy()
        arrays=dict(nv=np.array(m.nv),factor_total=np.array(m.qLD_block_total),
                    factor=d.qLD.numpy(),diagonal=d.qLDiagInv.numpy(),rhs=d.qfrc_smooth.numpy(),
                    address=address)
        for i,tile in enumerate(m.M_tiles):
            arrays[f'starts{i}']=tile.adr.numpy()
            arrays[f'size{i}']=np.array(tile.size)
            arrays[f'kind{i}']=np.array(int(tile.elemid.size!=0))
            for start in arrays[f'starts{i}']:
                adr=int(address[start]);blocks.append(dict(kind='compact' if adr==-2 else
                    'tile' if tile.elemid.size else 'scalar',size=tile.size,start=int(start),factor_address=adr))
        if (sorted(blocks,key=lambda b:b['start'])!=sorted(row['blocks'],key=lambda b:b['start']) or
                identity['imported_xml_sha256']!=row['imported_xml_sha256'] or m.qLD_block_total!=row['factor_total']):
            raise RuntimeError('GPU imported mass layout differs from CPU-only catalogue')
        if not all(np.array_equal(getattr(d,k).numpy(),v) for k,v in roots.items()):
            raise RuntimeError('Mass input collection changed physical roots')
        np.savez(path,**arrays)
        receipt=dict(task=row['task'],nv=m.nv,worlds=2,identity=identity,blocks=blocks,sha256=digest(path),
                     physics_integrations=0,newton_solves=0,catalogue_sha256=digest(args.catalogue),
                     origin='raw upstream initial-state position/factor, velocity, actuation, acceleration')
        path.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
        receipts.append(receipt)
        print(json.dumps(dict(frozen=str(path),nv=m.nv)),flush=True)
        del built,m,d
    return receipts


def load_case(path):
    with np.load(path,allow_pickle=False) as saved:
        c={k:saved[k].copy() for k in saved.files}
    c.update(name=path.stem,origin=json.loads(path.with_suffix('.json').read_text())['origin'])
    c['nv'],c['factor_total']=int(c['nv']),int(c['factor_total'])
    return c


def manufactured():
    """Three real block families, plus supported scalar branch/alias edge cases."""
    rng=np.random.default_rng(19371)
    result=[]
    layouts=(((0,1,-2),),((0,6,0),),((1,9,0),),
             ((0,1,-2),(0,6,0),(1,9,36),(0,6,117)),
             ((0,6,-2),(0,1,0),(1,9,1)))
    for index,layout in enumerate(layouts):
        nv=sum(size for _,size,_ in layout)
        total=max((adr+size*size for _,size,adr in layout if adr>=0),default=0)
        # NaN guards ensure neither unused lower factors nor unrelated diagonal
        # entries accidentally become physical inputs.
        c=dict(nv=nv,factor_total=total,factor=np.full((2,total),np.nan,np.float32),
               diagonal=np.full((2,nv),np.nan,np.float32),address=np.empty(nv,np.int32),
               rhs=rng.normal(size=(2,nv)).astype(np.float32),origin='manufactured upper-factor block system')
        start=0
        for i,(kind,size,adr) in enumerate(layout):
            c[f'starts{i}']=np.array([start],np.int32);c[f'size{i}']=np.array(size);c[f'kind{i}']=np.array(kind)
            c['address'][start:start+size]=adr
            for w in range(2):
                if adr==-2:c['diagonal'][w,start:start+size]=np.geomspace(.1,10,size)*(w+1)
                else:
                    q=rng.normal(scale=.15,size=(size,size));u=np.triu(q)
                    u[np.diag_indices(size)]=np.geomspace(1,3,size)
                    packed=c['factor'][w,adr:adr+size*size].reshape(size,size)
                    r,col=np.triu_indices(size);packed[r,col]=u[r,col]
            start+=size
        for variant in ('ordinary','alias','padded','zero'):
            item={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in c.items()}
            item.update(name=f'synthetic{index}-{variant}',alias=variant=='alias',padding=3 if variant=='padded' else 0)
            if variant=='zero':item['rhs'][:]=0
            result.append(item)
    return result


class Replay:
    def __init__(self,case,library,stream):
        import warp as wp
        from mujoco_warp._src import types
        from libero.gpu.host.runtime.dynamics.mjwarp_mass_solve import MassSolvePlan
        self.case,self.stream=case,stream
        array=lambda a:wp.array(np.ascontiguousarray(a),dtype=wp.int32 if a.dtype==np.int32 else wp.float32,device=stream.device)
        self.model=NS(nv=case['nv'],qLD_block_total=case['factor_total'],
                      qLD_block_adr=array(case['address']),block_dim=types.BlockDim(),M_tiles=[])
        i=0
        while f'starts{i}' in case:
            self.model.M_tiles.append(types.TileSet(adr=array(case[f'starts{i}']),size=int(case[f'size{i}']),
                elemid=array(np.array([0],np.int32) if case[f'kind{i}'] else np.empty(0,np.int32))))
            i+=1
        self.data=NS(nworld=2,qLD=array(case['factor']),qLDiagInv=array(case['diagonal']))
        self.plan=MassSolvePlan(self.model,self.data,library,stream)
        self.vectors,self.graphs,self.timing_graphs={},{},{}
        padding=case.get('padding',0);nv=case['nv']
        self.seed=np.full((2,nv+padding),37.,np.float32);self.seed[:,:nv]=case['rhs']
        for mode in ('upstream','cuda'):
            y=array(self.seed)
            x=y if case.get('alias') else wp.full((2,nv+padding),37.,dtype=float,device=stream.device)
            self.vectors[mode]=(x,y)
            if mode=='cuda':self.binding=self.plan.bind(x,y)
            with wp.ScopedCapture(stream=stream) as capture:self.enqueue(mode)
            self.graphs[mode]=capture.graph
            if case['name'].startswith('task'):
                with wp.ScopedCapture(stream=stream) as capture:
                    for _ in range(32):self.enqueue(mode)
                self.timing_graphs[mode]=capture.graph

    def enqueue(self,mode):
        from mujoco_warp._src import smooth
        if mode=='cuda':self.binding.enqueue()
        else:smooth.solve_m(self.model,self.data,*self.vectors[mode])

    def check(self,updated=False):
        import warp as wp
        from validation.precision import balance_residual
        c,nv=self.case,self.model.nv
        # Reusing the SAME binding/graphs must observe new factors and RHS.
        factor=c['factor']*(np.float32(1.25) if updated else np.float32(1))
        diagonal=c['diagonal']*(np.float32(.75) if updated else np.float32(1))
        rhs=c['rhs']*(np.float32(-.5) if updated else np.float32(1))
        self.data.qLD.assign(factor);self.data.qLDiagInv.assign(diagonal)
        for mode,(x,y) in self.vectors.items():
            seed=self.seed.copy();seed[:,:nv]=rhs;y.assign(seed)
            if not c.get('alias'):x.fill_(37.)
            wp.capture_launch(self.graphs[mode],stream=self.stream)
        wp.synchronize_stream(self.stream)
        outputs={k:x.numpy() for k,(x,y) in self.vectors.items()}
        holds=all(np.array_equal(y.numpy()[:,nv:],self.seed[:,nv:]) and
            np.all(x.numpy()[:,nv:]==37.) and (c.get('alias') or np.array_equal(y.numpy()[:,:nv],rhs))
            for x,y in self.vectors.values())
        holds &= np.array_equal(self.data.qLD.numpy(),factor,equal_nan=True)
        holds &= np.array_equal(self.data.qLDiagInv.numpy(),diagonal,equal_nan=True)
        rows=[];i=0
        while f'starts{i}' in c:
            size=int(c[f'size{i}'])
            for start in c[f'starts{i}']:
                start=int(start);adr=int(c['address'][start]);sl=slice(start,start+size)
                for w in range(2):
                    if adr==-2:a=np.diag(1/diagonal[w,sl].astype(np.float64))
                    else:
                        u=np.triu(factor[w,adr:adr+size*size].reshape(size,size)).astype(np.float64)
                        a=u.T@u
                    b=rhs[w,sl].astype(np.float64);ref=np.linalg.solve(a,b)
                    condition=float(np.linalg.cond(a,np.inf));checks={}
                    # Fixed policy: 8 gamma_n backward error, with its a-priori
                    # normwise sensitivity bound 2*k*eta/(1-k*eta).
                    unit=np.finfo(np.float32).eps/2;eta=8*size*unit/(1-size*unit)
                    if condition*eta>=1:raise RuntimeError('Input cannot support forward sensitivity qualification')
                    limit=2*condition*eta/(1-condition*eta)*max(float(abs(ref).max()),np.finfo(np.float32).tiny)
                    for mode,x in outputs.items():
                        res=balance_residual(a,x[w,sl],b,np.zeros(size))
                        error=float(abs(x[w,sl]-ref).max())
                        checks[mode]=dict(**res,forward_error=error,forward_limit=limit,
                                          qualified=bool(res['passed'] and error<=limit))
                    difference=float(abs(outputs['cuda'][w,sl]-outputs['upstream'][w,sl]).max())
                    rows.append(dict(world=w,start=start,size=size,condition=condition,checks=checks,
                        difference=difference,passed=bool(all(v['qualified'] for v in checks.values()) and difference<=2*limit)))
            i+=1
        return dict(name=c['name'],updated=updated,nv=nv,worlds=2,holds=bool(holds),blocks=rows,
                    passed=bool(holds and all(r['passed'] for r in rows))),outputs

    def benchmark(self,args):
        import warp as wp
        result={}
        for mode,graph in self.timing_graphs.items():
            for _ in range(args.warmup):wp.capture_launch(graph,stream=self.stream)
            start,end=wp.Event(enable_timing=True),wp.Event(enable_timing=True);values=[]
            for _ in range(args.steps):
                self.stream.record_event(start);wp.capture_launch(graph,stream=self.stream);self.stream.record_event(end)
                wp.synchronize_event(end)
                values.append(wp.get_event_elapsed_time(start,end,synchronize=False)*1000/32)
            result[mode]=dict(mean_us=float(np.mean(values)),std_us=float(np.std(values)),microseconds=values)
        result.update(name=self.case['name'],calls_per_sample=32,warmup=args.warmup,samples=args.steps,
                      speedup=result['upstream']['mean_us']/result['cuda']['mean_us'])
        return result


def main(argv=None,*,args=None):
    if args is None:args=add_arguments(argparse.ArgumentParser(description=__doc__)).parse_args(argv)
    if args.warmup<0 or args.steps<1:raise ValueError('invalid measurement count')
    from libero.gpu import precision
    precision.select('fp32')
    import warp as wp
    from warp._src.build import init_kernel_cache
    from libero.gpu.host.backends.mjwarp.build import source_files
    from libero.gpu.host.build.compilation import compilation_scope,install_warp_guard
    from validation.common import save_json
    from validation.runner import gpu_load_snapshot
    args.output.mkdir(parents=True,exist_ok=True)
    init_kernel_cache(args.build_root/'mjwarp/warp-cache');install_warp_guard();wp.init()
    prepared=args.output/'preparation.json';stream=wp.Stream('cuda:0')
    with compilation_scope(args.prepare) as scope,wp.ScopedStream(stream):
        if args.prepare:freeze(args,stream)
        inputs=[args.output/f'task{i}.npz' for i in (0,3)]
        signature=dict(sources=source_files(include_mass=True),tool_sha256=digest(__file__),
            catalogue_sha256=digest(args.catalogue),library_sha256=digest(args.library),
            inputs={p.name:digest(p) for p in inputs},receipts={p.stem:digest(p.with_suffix('.json')) for p in inputs})
        if not args.prepare and (not prepared.is_file() or json.loads(prepared.read_text())['signature']!=signature):
            raise RuntimeError('Prepare these exact mass solve graphs offline first')
        replays=[Replay(c,args.library,stream) for c in [load_case(p) for p in inputs]+manufactured()]
        if args.prepare:
            wp.synchronize_stream(stream)
            save_json(prepared,dict(signature=signature,compilation_events=scope['events'],physics_integrations=0))
            print(json.dumps(dict(prepared=True,cases=len(replays),compilations=len(scope['events']))),flush=True)
            return 0
        before=gpu_load_snapshot();rows=[]
        for replay in replays:
            for updated in (False,True):
                row,outputs=replay.check(updated);rows.append(row)
                np.savez(args.output/(row['name']+('-updated' if updated else '')+'-output.npz'),**outputs)
                print(json.dumps(dict(name=row['name'],updated=updated,passed=row['passed'])),flush=True)
        passed=all(r['passed'] for r in rows)
        # Restore original frozen factors after the lifetime test, outside timing.
        for r in replays[:2]:r.check()
        timing=[r.benchmark(args) for r in replays[:2]] if args.performance and passed else None
        save_json(args.output/'summary.json',dict(passed=passed,signature=signature,cases=rows,performance=timing,
            runtime_compilations=scope['events'],physics_integrations=0,newton_solves=0,
            gpu_load_before=before,gpu_load_after=gpu_load_snapshot(),
            policy='8 gamma_block backward residual; 2 cond_inf eta/(1-cond_inf eta) forward sensitivity; exact hold/alias contracts'))
        print(json.dumps(dict(passed=passed,performance=timing)),flush=True)
        return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())
