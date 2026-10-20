"""Same-input gradient/direction qualification; no physical integration.

Compile the independent CUDA target first. --prepare admits upstream operators
offline; the ordinary run cannot compile. FP64 solve/backward error checks are
independent of the upstream comparison and use only the frozen FP32 inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np


def add_arguments(parser):
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--build-root', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--small-input', type=Path, help='frozen real task5 gradient boundary .npz')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--performance', action='store_true')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--calls-per-sample', type=int, default=32)
    return parser


def cases(path,small_input=None):
    from validation.mjwarp_hessian import real_case, matrix_reference
    from libero.gpu.host.runtime.newton.mjwarp_direction import FAMILIES
    real = real_case(path)
    h, _, _ = matrix_reference(real)
    saved = dict(np.load(path, allow_pickle=False))
    nv, n = real['nv'], 2
    mass = np.zeros((n,nv,nv),np.float64)
    r,c = real['arrays']['mass_row'],real['arrays']['mass_column']
    mass[:,r,c] = real['arrays']['mass']; mass[:,c,r] = real['arrays']['mass']
    first = dict(name='task4-derived-final-boundary', origin=real['origin'],nv=nv,
        h=h.astype(np.float32), ma=np.einsum('wij,wj->wi',mass,saved['output.qacc']).astype(np.float32),
        smooth=saved['input.qfrc_smooth'], force=saved['output.qfrc_constraint'],done=np.zeros(n,np.bool_))
    result = [first]
    if small_input is not None:
        with np.load(small_input,allow_pickle=False) as saved_small:
            small={k:saved_small[k].copy() for k in ('h','ma','smooth','force','done')}
            small.update(nv=int(saved_small['nv']),name='task5-frozen-initial-boundary',
                origin='raw upstream forward followed by init_context at task5 initial states; no integration')
        if small['nv']!=21:raise ValueError('expected real task5 NV21 boundary')
        result.append(small)
    for nv in FAMILIES:
        pad = ((nv+3)//4)*4 if nv<=32 else ((nv+15)//16)*16
        rng = np.random.default_rng(11000+nv)
        h = np.zeros((n,pad,pad),np.float32)
        for w in range(n):
            q,_ = np.linalg.qr(rng.normal(size=(nv,nv)))
            h[w,:nv,:nv] = q @ np.diag(np.geomspace(1.,100.,nv)) @ q.T
            h[w,nv:,nv:] = np.eye(pad-nv)
            # Lower storage is not a second input triangle.
            h[w][np.tril_indices(pad,-1)] = 1234.
        result.append(dict(name=f'nv{nv}-spd',origin='manufactured SPD input',nv=nv,h=h,
            ma=rng.normal(size=(n,nv)).astype(np.float32),smooth=rng.normal(size=(n,nv)).astype(np.float32),
            force=rng.normal(size=(n,nv)).astype(np.float32),done=np.zeros(n,np.bool_)))
    for nv in (21,39,57):
        source=next(c for c in result if c['name']==f'nv{nv}-spd')
        for variant in ('world1-done','all-done','zero-gradient'):
            item={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in source.items()}
            item['name']=f'nv{nv}-{variant}'
            if variant=='world1-done':item['done'][1]=True
            elif variant=='all-done':item['done'][:]=True
            else:item['ma']=item['smooth']+item['force'];item['force'][:]=0.;item['smooth']=item['ma'].copy()
            result.append(item)
    return result


class Replay:
    def __init__(self,case,library,stream):
        import warp as wp
        from mujoco_warp._src import types
        from libero.gpu.host.runtime.newton.mjwarp_direction import DirectionPlan
        self.case,self.stream=case,stream
        nv,pad=case['nv'],case['h'].shape[1]
        array=lambda a:wp.array(np.ascontiguousarray(a),dtype=wp.bool if a.dtype==np.bool_ else wp.float32,device=stream.device)
        self.model=NS(nv=nv,nv_pad=pad,block_dim=types.BlockDim())
        self.data=NS(nworld=2,nefc=wp.zeros(2,dtype=int,device=stream.device),
            efc=NS(Ma=array(case['ma'])),qfrc_smooth=array(case['smooth']),qfrc_constraint=array(case['force']))
        self.contexts,self.graphs={},{}
        self.plan=DirectionPlan(self.model,self.data,library,stream)
        for mode in ('upstream','cuda'):
            vec=lambda width:wp.full((2,width),37.,dtype=float,device=stream.device)
            scalar=lambda:wp.full(2,37.,dtype=float,device=stream.device)
            ctx=NS(done=array(case['done']),h=array(case['h']),grad=wp.zeros((2,pad),dtype=float,device=stream.device),
                grad_dot=scalar(),newton_decrement=scalar(),grad_scale=scalar(),
                search_unchanged=wp.full(2,True,dtype=bool,device=stream.device),
                hfactor=wp.full((2,pad,pad) if nv>32 else (2,0,0),37.,dtype=float,device=stream.device),
                search=vec(nv),search_dot=scalar(),alpha=scalar())
            self.contexts[mode]=ctx
            if mode=='cuda':self.binding=self.plan.bind(ctx)
            with wp.ScopedCapture(stream=stream) as capture:
                for _ in range(32):self.enqueue(mode)
            self.graphs[mode]=capture.graph

    def enqueue(self,mode):
        import warp as wp
        from mujoco_warp._src import solver
        if mode=='cuda':self.binding.enqueue();return
        m,d,ctx=self.model,self.data,self.contexts[mode]
        wp.launch(solver._update_gradient_zero_grad_dot(False),dim=2,inputs=[d.nefc,ctx.alpha,ctx.done],
            outputs=[ctx.grad_dot,ctx.newton_decrement,ctx.grad_scale,ctx.search_unchanged])
        wp.launch(solver._update_gradient_grad(False),dim=(2,m.nv),
            inputs=[d.qfrc_smooth,d.qfrc_constraint,d.efc.Ma,d.nefc,ctx.done],outputs=[ctx.grad,ctx.grad_dot])
        solver._cholesky_factorize_solve(m,d,ctx)

    def check(self):
        import warp as wp
        for mode in self.graphs:wp.capture_launch(self.graphs[mode],stream=self.stream)
        wp.synchronize_stream(self.stream)
        outputs={mode:{k:getattr(ctx,k).numpy() for k in
            ('grad','grad_dot','search','search_dot','newton_decrement','grad_scale','search_unchanged','hfactor')}
            for mode,ctx in self.contexts.items()}
        a,nv=self.case,self.model.nv
        rows=[];eps=np.finfo(np.float32).eps;gamma=nv*eps/(1.-nv*eps)
        for w in range(2):
            if a['done'][w]:
                held=all(np.array_equal(outputs['cuda'][k][w],outputs['upstream'][k][w]) for k in outputs['cuda'])
                held &= all(np.all(outputs['cuda'][k][w]==(0. if k=='grad' else False if k=='search_unchanged' else 37.))
                    for k in outputs['cuda'])
                rows.append(dict(world=w,done=True,held=bool(held),passed=bool(held)));continue
            h=np.triu(a['h'][w,:nv,:nv]).astype(np.float64);h=h+h.T-np.diag(np.diag(h))
            g=a['ma'][w].astype(np.float64)-a['smooth'][w]-a['force'][w]
            ref=-np.linalg.solve(h,g);condition=float(np.linalg.cond(h,np.inf))
            bound=32.*gamma*condition*max(float(abs(ref).max()),np.finfo(np.float32).tiny)
            checks={}
            for mode,out in outputs.items():
                s=out['search'][w].astype(np.float64);cg=out['grad'][w,:nv].astype(np.float64)
                scale=np.linalg.norm(h,np.inf)*np.linalg.norm(s,np.inf)+np.linalg.norm(cg,np.inf)
                backward=float(np.linalg.norm(h@s+cg,np.inf)/max(scale,np.finfo(np.float64).tiny))
                error=float(abs(s-ref).max())
                dot=float(s@s);dec=float(-cg@s)
                scalar_error=max(abs(float(out['search_dot'][w])-dot)/max(abs(dot),1.e-30),
                    abs(float(out['newton_decrement'][w])-dec)/max(abs(dec),1.e-30))
                grad_error=float(abs(cg-g).max())
                grad_bound=4.*eps*float(np.max(abs(a['ma'][w])+abs(a['smooth'][w])+abs(a['force'][w])))
                checks[mode]=dict(backward_error=backward,backward_limit=16.*gamma,
                    search_error=error,search_limit=bound,scalar_relative_error=scalar_error,
                    gradient_error=grad_error,gradient_limit=grad_bound,
                    passed=bool(backward<=16.*gamma and error<=bound and scalar_error<=16.*gamma and
                                grad_error<=grad_bound and out['grad_scale'][w]==1 and not out['search_unchanged'][w]))
            difference=float(abs(outputs['cuda']['search'][w]-outputs['upstream']['search'][w]).max())
            rows.append(dict(world=w,condition=condition,checks=checks,search_difference=difference,
                passed=bool(all(v['passed'] for v in checks.values()) and difference<=2*bound)))
        # H and padded gradients remain unchanged, including skipped worlds.
        holds=all(np.array_equal(ctx.h.numpy(),a['h']) and np.all(ctx.grad.numpy()[:,nv:]==0)
                  for ctx in self.contexts.values())
        return dict(name=a['name'],origin=a['origin'],nv=nv,worlds=2,checks=rows,holds=holds,
                    passed=bool(holds and all(r['passed'] for r in rows))),outputs

    def benchmark(self,args):
        import warp as wp
        if args.calls_per_sample!=32:raise ValueError('this admitted graph contains exactly 32 calls')
        result={}
        for mode,graph in self.graphs.items():
            for _ in range(args.warmup):wp.capture_launch(graph,stream=self.stream)
            values=[];start=wp.Event(enable_timing=True);end=wp.Event(enable_timing=True)
            for _ in range(args.steps):
                self.stream.record_event(start);wp.capture_launch(graph,stream=self.stream);self.stream.record_event(end)
                wp.synchronize_event(end)
                values.append(wp.get_event_elapsed_time(start,end,synchronize=False)*1000/32)
            result[mode]=dict(mean_us=float(np.mean(values)),std_us=float(np.std(values)),microseconds=values)
        result['speedup']=result['upstream']['mean_us']/result['cuda']['mean_us']
        result.update(name=self.case['name'],calls_per_sample=32,warmup=args.warmup,samples=args.steps)
        return result


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
    init_kernel_cache(args.build_root/'mjwarp/warp-cache');install_warp_guard();wp.init()
    signature=dict(sources=source_files(include_direction=True),
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest())
    if args.small_input is not None:
        signature['small_input_sha256']=hashlib.sha256(args.small_input.read_bytes()).hexdigest()
        signature['small_receipt_sha256']=hashlib.sha256(args.small_input.with_name('summary.json').read_bytes()).hexdigest()
    prepared=args.output/'preparation.json'
    if not args.prepare and (not prepared.is_file() or json.loads(prepared.read_text())['signature']!=signature):
        raise RuntimeError('prepare these exact local direction graphs offline first')
    stream=wp.Stream('cuda:0')
    with compilation_scope(args.prepare) as scope,wp.ScopedStream(stream):
        replays=[Replay(c,args.library,stream) for c in cases(args.input,args.small_input)]
        if args.prepare:
            wp.synchronize_stream(stream)
            save_json(prepared,dict(signature=signature,compilation_events=scope['events'],physics_integrations=0,replays=0))
            print(json.dumps(dict(prepared=True,cases=len(replays),compilations=len(scope['events']))),flush=True);return 0
        before=gpu_load_snapshot();rows=[]
        for replay in replays:
            row,outputs=replay.check();rows.append(row)
            np.savez(args.output/(row['name']+'.npz'),**{mode+'.'+k:v for mode,out in outputs.items() for k,v in out.items()})
            print(json.dumps(dict(name=row['name'],passed=row['passed'])),flush=True)
        passed=all(r['passed'] for r in rows)
        timing=([r.benchmark(args) for r in replays if r.case['name'].startswith('task')]
                if args.performance and passed else None)
        result=dict(passed=passed,signature=signature,cases=rows,performance=timing,
            runtime_compilations=scope['events'],physics_integrations=0,gpu_load_before=before,gpu_load_after=gpu_load_snapshot(),
            policy='FP64 solve sensitivity 32*gamma_NV*cond_inf; backward/scalar 16*gamma_NV; gradient 4eps arithmetic majorant')
        save_json(args.output/'summary.json',result)
        print(json.dumps(dict(passed=passed,performance=timing)),flush=True)
        return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())
