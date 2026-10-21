"""Independent regression for GJK tuned-direction separation certificates."""
from pathlib import Path
from types import SimpleNamespace as NS
import json
import numpy as np
from .cuda_complete import save,digest
from . import mjwarp_convex_gjk as gjk,mjwarp_convex_ccd as ccd
from .mjwarp_convex_support import STATIC
from .mjwarp_box_reference import box_separation


def add_arguments(parser):
    parser.add_argument('--build-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--fixture',type=Path,default=Path(__file__).resolve().parents[2]/'tests/data/gjk_tuned_direction.json')
    parser.add_argument('--prepare',action='store_true')


def fixture_case(path,variant='overlap'):
    from libero.gpu.host.runtime.collision.mjwarp_convex import GJK_OUTPUTS
    f=json.loads(Path(path).read_text());a=np.asarray(f['position1'],np.float32);b=np.asarray(f['position2'],np.float32)
    if variant=='separated':b=b+np.array([0,0,.02],np.float32)
    elif variant=='deeper':b=b-np.array([0,0,.0002],np.float32)
    elif variant=='translated':
        shift=np.array([.013,-.017,.019],np.float32);a=a+shift;b=b+shift
    c=dict(name=variant,ngeom=2,world=np.array([0],np.int32),pair=np.array([[0,1]],np.int32),
        geom_type=np.array([6,6],np.int32),geom_dataid=np.full((1,2),-1,np.int32),
        size=np.asarray(f['sizes'],np.float32)[None],xpos=np.tile(np.stack([a,b])[None],(2,1,1)),
        xmat=np.tile(np.asarray(f['rotations'],np.float32)[None],(2,1,1,1)),
        geom=np.zeros(1,np.int32),cached=np.full(1,-1,np.int32),direction=np.zeros((1,3),np.float32),request_margin=np.zeros(1,np.float32),
        initial_point=np.zeros((1,3),np.float32),initial_cached_out=np.zeros(1,np.int32),initial_vertex_out=np.zeros(1,np.int32),
        position1=a[None],initial1=a[None],position2=b[None],initial2=b[None],
        tolerance=np.full(1,f['tolerance'],np.float32),iterations=np.full(1,f['iterations'],np.int32),
        discrete=np.ones(1,np.int32),epa_iterations=f['epa_iterations'],face_capacity=6+5*f['epa_iterations'],horizon_capacity=24,overflow=np.zeros(2,np.int32))
    for k in STATIC:
        if k not in c:c[k]=np.zeros(1,np.int32)
    c['mesh_vertnum']=np.ones(1,np.int32);c['mesh_graphadr']=np.full(1,-1,np.int32)
    for k in ('mesh_vert','mesh_polynormal'):c[k]=np.zeros((1,3),np.float32)
    for k in ('cache1','cache2'):c[k]=np.full(1,-1,np.int32)
    for k in ('margin1','margin2','cutoff'):c[k]=np.zeros(1,np.float32)
    for k,kind,width in GJK_OUTPUTS:
        shape=(1,4) if width==4 else (1,)
        if kind=='vec3':shape+=(3,)
        c['initial_'+k]=np.zeros(shape,np.int32 if kind=='int32' else np.float32)
    return c


def main(args):
    import warp as wp
    from warp._src.build import init_kernel_cache
    from libero.gpu import precision
    from libero.gpu.host.backends.mjwarp.model import require_upstream
    from libero.gpu.host.build.compilation import compilation_scope,install_warp_guard
    from libero.gpu.host.runtime.collision.mjwarp_convex import GJK_REQUESTS,GJK_OUTPUTS,CCD_OUTPUTS,GJKProbe
    precision.select('fp32');require_upstream();init_kernel_cache(args.build_root/'mjwarp/warp-cache');install_warp_guard()
    gjk.GJK_REQUESTS,gjk.GJK_OUTPUTS=GJK_REQUESTS,GJK_OUTPUTS
    wp.init();stream=wp.Stream('cuda:0');args.output.mkdir(parents=True,exist_ok=True)
    with compilation_scope(args.prepare) as scope,wp.ScopedStream(stream):
        replays=[]
        for variant in ('overlap','deeper','separated','translated'):
            c=fixture_case(args.fixture,variant);replay=ccd.Replay(c,args,stream)
            owner=gjk.construct(c,stream);m,d,q,o=owner;binding=GJKProbe(m,d,q,o,args.library,stream)
            with wp.ScopedCapture(stream=stream) as cap:binding.enqueue()
            replays.append((c,replay,owner,binding,cap.graph))
        if args.prepare:
            save(args.output/'prepare.json',dict(fixture_sha256=digest(args.fixture),library_sha256=digest(args.library),
                compilation_events=scope['events'],integrations=0));return 0
        rows=[]
        for c,replay,owner,binding,graph in replays:
            gap=box_separation(c['position1'][0],c['position2'][0],*c['xmat'][0],*c['size'][0])
            for mode,values in replay.owners.items():
                wp.capture_launch(replay.graphs[mode],stream=stream);wp.synchronize_stream(stream)
                m,d,q,o,s,frozen=values;got={k:getattr(o,k).numpy() for k in gjk.RAW+CCD_OUTPUTS}
                np.savez_compressed(args.output/(c['name']+'-'+mode+'.npz'),**got,**{'scratch.'+k:v.numpy() for k,v in vars(s).items()})
                dist=float(got['dist_out'][0]);nc=int(got['ncontact_out'][0])
                passed=bool(nc==1 and dist<0) if gap < -float(c['tolerance'][0]) else bool(dist>=0)
                rows.append(dict(case=c['name'],mode=mode,sat_gap=gap,distance=dist,contacts=nc,passed=passed))
            wp.capture_launch(graph,stream=stream);wp.synchronize_stream(stream)
            got={k:getattr(owner[3],k).numpy() for k,_,_ in GJK_OUTPUTS}
            np.savez_compressed(args.output/(c['name']+'-gjk.npz'),**got)
            if int(got['separated_out'][0]):
                direction=got['direction_out'][0].astype(float)
                minimum=np.dot(direction,c['position1'][0].astype(float)-c['position2'][0].astype(float))
                minimum-=sum(np.dot(abs(r.astype(float).T@direction),s.astype(float)) for r,s in zip(c['xmat'][0],c['size'][0]))
                rows.append(dict(case=c['name'],mode='cuda-certificate',passed=bool(minimum>0),
                    actual_minimum=float(minimum),claimed_lower=float(got['lower_out'][0])))
        result=dict(rows=rows,passed=all(r['passed'] for r in rows if r['mode'].startswith('cuda')),
            fixture_sha256=digest(args.fixture),library_sha256=digest(args.library),runtime_compilations=scope['events'],integrations=0,performance=None)
        save(args.output/'summary.json',result);print(json.dumps(result),flush=True)
        return 0 if result['passed'] else 2
