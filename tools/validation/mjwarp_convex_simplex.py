"""Reachable ordered GJK simplex probe, independent convex QP and raw parity."""
import argparse,functools,json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from .mjwarp_convex_simplex_reference import closest,check
from .mjwarp_convex_support import digest,add_arguments,input_identity
FIELDS=('weights','point','distance','mask')


def cases(args):
    rows=[]
    for p in sorted(args.inputs.glob('*.npz')):
        if digest(p)!=json.loads(p.with_suffix('.json').read_text())['sha256']:raise RuntimeError('Reachable simplex input changed')
        with np.load(p,allow_pickle=False) as f:c={k:f[k].copy() for k in f.files}
        c['name']=p.stem;rows.append(c)
    if not rows:raise ValueError('No reachable simplex inputs')
    return rows


def current_case(c,updated):
    c={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in c.items()}
    if updated:c['simplex']=(c['simplex'][:,:,[2,0,1]]*np.float32(.97)).copy()
    return c

@functools.lru_cache(maxsize=1)
def raw_kernel():
    import warp as wp
    from mujoco_warp._src.collision_gjk import _subdistance,_linear_combine
    from mujoco_warp._src.types import mat43
    @wp.kernel(module='unique',enable_backward=False)
    def kernel(simplex:wp.array2d[wp.vec3],count:wp.array[int],weights:wp.array2d[float],point:wp.array[wp.vec3],distance:wp.array[float],mask:wp.array[int]):
        i=wp.tid();s=mat43()
        for k in range(4):s[k]=simplex[i,k]
        w=_subdistance(count[i],s);p=_linear_combine(count[i],w,s);bits=int(0)
        for k in range(4):
            weights[i,k]=w[k]
            if w[k]!=0.0:bits|=1<<k
        point[i]=p;distance[i]=wp.length(p);mask[i]=bits
    return kernel


def construct(c,stream):
    import warp as wp
    arr=lambda value,dtype:wp.array(np.ascontiguousarray(value),dtype=dtype,device=stream.device)
    inputs=NS(simplex=arr(c['simplex'],wp.vec3),count=arr(c['count'],wp.int32));outputs=NS(weights=arr(c['initial_weights'],wp.float32),point=arr(c['initial_point'],wp.vec3),distance=arr(c['initial_distance'],wp.float32),mask=arr(c['initial_mask'],wp.int32));return inputs,outputs

def assign(c,inputs,outputs):
    inputs.simplex.assign(c['simplex']);inputs.count.assign(c['count'])
    for key in FIELDS:getattr(outputs,key).assign(c['initial_'+key])


class Replay:
    def __init__(self,c,args,stream):
        import warp as wp
        from libero.gpu.host.runtime.collision.mjwarp_convex import SimplexProbe
        self.case,self.stream=c,stream;self.owners={};self.graphs={}
        for mode in ('upstream','cuda'):
            inputs,outputs=construct(c,stream);self.owners[mode]=(inputs,outputs)
            if mode=='cuda':self.binding=SimplexProbe(inputs.simplex,inputs.count,outputs,args.library,stream)
            with wp.ScopedCapture(stream=stream) as cap:
                if mode=='cuda':self.binding.enqueue()
                elif inputs.count.size:wp.launch(raw_kernel(),inputs.count.size,block_dim=128,inputs=[inputs.simplex,inputs.count],outputs=[getattr(outputs,k) for k in FIELDS])
            self.graphs[mode]=cap.graph
    def check(self,updated):
        import warp as wp
        c=current_case(self.case,updated);rows=[];values=[]
        for mode,(inputs,outputs) in self.owners.items():
            assign(c,inputs,outputs);before={k:a.numpy().view(np.uint8).copy() for k,a in vars(inputs).items()};wp.capture_launch(self.graphs[mode],stream=self.stream);wp.synchronize_stream(self.stream);got={k:getattr(outputs,k).numpy() for k in FIELDS};values.append(got);checks=[]
            for i in range(len(c['count'])):checks += [dict(v,request=i) for v in check(c['simplex'][i],int(c['count'][i]),{k:got[k][i] for k in FIELDS})]
            held=all(np.array_equal(a.numpy().view(np.uint8),before[k]) for k,a in vars(inputs).items());rows.append(dict(mode=mode,passed=bool(held and all(v['passed'] for v in checks)),checks=checks,inputs_held=held))
        # Coordinate/active-set choices can differ at a degenerate face. Both
        # candidates must independently attain the QP optimum and KKT residual.
        error=abs(values[0]['distance'].astype(float)-values[1]['distance']);scale=1+np.max(abs(c['simplex']),axis=(1,2));u=np.finfo(np.float32).eps/2;bound=16*(512*u/(1-512*u))*scale;parity=bool(np.all(error<=bound))
        return dict(name=c['name'],updated=updated,modes=rows,distance_parity=parity,maximum_distance_difference=float(error.max(initial=0)),passed=parity and all(row['passed'] for row in rows)),values


def signature(args):
    from libero.gpu.host.backends.mjwarp.build import source_files
    files=source_files(include_convex=True);files={k:v for k,v in files.items() if k.startswith('upstream/') or any(n in k for n in ('mjwarp_convex','MjwarpConvex','mjwarp_contact.cuh','mjwarp_plane.py')) or k in ('host/runtime/dynamics/mjwarp_motion.py','project_source_identity','data/mjwarp_source_pin.json')}
    return dict(sources=files,inputs=input_identity(args),library_sha256=digest(args.library),tool_sha256=digest(__file__),reference_sha256=digest(Path(__file__).with_name('mjwarp_convex_simplex_reference.py')),helpers={n:digest(Path(__file__).with_name(n)) for n in ('mjwarp_convex_support.py',)})


def cpu_reference(args):
    rows=[]
    for original in cases(args):
        for updated in (False,True):
            c=current_case(original,updated)
            for i,count in enumerate(c['count']):
                s=c['simplex'][i,:int(count)].astype(float);ref=closest(s);p=ref['point'];w=ref['weights'];scale=1+np.max(abs(s));eps=2048*np.finfo(float).eps*scale**2
                valid=np.isfinite(ref['distance']) and np.all(w>=-eps) and abs(sum(w)-1)<=eps and np.min((s-p)@p)>=-eps
                rows.append(dict(name=c['name'],updated=updated,request=i,passed=bool(valid),distance=ref['distance']))
    result=dict(passed=bool(rows) and all(v['passed'] for v in rows),checks=rows,inputs=input_identity(args),tool_sha256=digest(__file__),reference_sha256=digest(Path(__file__).with_name('mjwarp_convex_simplex_reference.py')),physics_integrations=0,performance=None,domain='independent FP64 affine-face convex-QP enumeration/KKT; CPU2.3.7 uses MPR, not this newer GJK signed-volume subalgorithm')
    (args.output/'cpu-oracle.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=result['passed'],checks=len(rows))),flush=True);return 0 if result['passed'] else 2


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
    require_upstream();init_kernel_cache(args.build_root/'mjwarp/warp-cache');install_warp_guard();wp.init();stream=wp.Stream('cuda:0');sig=signature(args);prep=args.output/'preparation.json'
    if not args.prepare:
        if not prep.is_file() or json.loads(prep.read_text())['signature']!=sig:raise RuntimeError('Prepare exact simplex probe offline first')
        cpu=json.loads((args.output/'cpu-oracle.json').read_text())
        if not cpu['passed'] or any(cpu[k]!=sig[k] for k in ('inputs','tool_sha256','reference_sha256')):raise RuntimeError('Prepare matching simplex QP oracle first')
    with compilation_scope(args.prepare) as scope,wp.ScopedStream(stream):
        replays=[Replay(c,args,stream) for c in cases(args)]
        if args.prepare:
            wp.synchronize_stream(stream);prep.write_text(json.dumps(dict(signature=sig,compilation_events=scope['events'],physics_integrations=0),indent=2)+'\n');print(json.dumps(dict(prepared=True,captures=2*len(replays),compilations=len(scope['events']))),flush=True);return 0
        rows=[]
        for replay in replays:
            for updated in (False,True):
                row,values=replay.check(updated);rows.append(row);print(json.dumps({k:row[k] for k in ('name','updated','passed')}),flush=True)
                if not row['passed']:np.savez(args.output/'failure.npz',**{mode+'.'+k:v for mode,got in zip(('upstream','cuda'),values) for k,v in got.items()});break
            if not row['passed']:break
        passed=all(row['passed'] for row in rows);(args.output/'summary.json').write_text(json.dumps(dict(passed=passed,signature=sig,cases=rows,runtime_compilations=scope['events'],worlds=2,physics_integrations=0,performance=None,production_collision_replaced=False),indent=2)+'\n');print(json.dumps(dict(passed=passed,cases=len(rows))),flush=True);return 0 if passed else 2
if __name__=='__main__':raise SystemExit(main())
