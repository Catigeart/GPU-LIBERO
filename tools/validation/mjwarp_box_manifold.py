"""Box-only convex manifold from frozen original EPA faces, N=2."""
import argparse,functools,json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from . import mjwarp_convex_gjk as gjk
from .mjwarp_convex_support import add_arguments,cases,digest
from .mjwarp_convex_reference import U
FIELDS=('count','witness1','witness2','distance')


def current_case(c,updated):
    c={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in c.items()}
    if updated:
        shift=np.array([.013,-.017,.019],np.float32)
        for key in ('position1','position2','initial1','initial2','frozen_vertices','frozen_x1','frozen_x2'):c[key]+=shift
    return c


@functools.lru_cache(maxsize=1)
def reference_kernel():
    import warp as wp
    from mujoco_warp._src.collision_core import Geom
    from mujoco_warp._src.collision_gjk import multicontact
    @wp.kernel(module='unique',enable_backward=False)
    def kernel(world:wp.array[int],pair:wp.array[wp.vec2i],size:wp.array2d[wp.vec3],R:wp.array2d[wp.mat33],position1:wp.array[wp.vec3],position2:wp.array[wp.vec3],vert:wp.array2d[wp.vec3],index:wp.array2d[int],face:wp.array2d[int],face_index:wp.array[int],x1:wp.array[wp.vec3],x2:wp.array[wp.vec3],polygon:wp.array2d[wp.vec3],clipped:wp.array2d[wp.vec3],pn:wp.array2d[wp.vec3],pd:wp.array2d[float],idx1:wp.array2d[int],idx2:wp.array2d[int],n1:wp.array2d[wp.vec3],n2:wp.array2d[wp.vec3],end:wp.array2d[wp.vec3],face1:wp.array2d[wp.vec3],face2:wp.array2d[wp.vec3],count:wp.array[int],witness1:wp.array2d[wp.vec3],witness2:wp.array2d[wp.vec3],distance:wp.array2d[float]):
        i=wp.tid();w=world[i];g1=pair[i][0];g2=pair[i][1];a=Geom();b=Geom();a.pos=position1[i];b.pos=position2[i];a.rot=R[w,g1];b.rot=R[w,g2];a.size=size[w%size.shape[0],g1];b.size=size[w%size.shape[0],g2]
        nc,one,two,d=multicontact(polygon[i],clipped[i],pn[i],pd[i],idx1[i],idx2[i],n1[i],n2[i],end[i],face1[i],face2[i],vert[i],index[i],face[i,face_index[i]],x1[i],x2[i],a,b,6,6);count[i]=nc
        for k in range(4):witness1[i,k]=one[k];witness2[i,k]=two[k];distance[i,k]=d[k]
    return kernel


def construct(c,stream):
    import warp as wp
    m,d,q,query_outputs=gjk.construct(c,stream);arr=lambda v,t:wp.array(np.ascontiguousarray(v),dtype=t,device=stream.device)
    frozen=NS(**{key:arr(c['frozen_'+key],wp.vec3 if key in ('vertices','x1','x2') else wp.int32) for key in ('vertices','vertex_index','face','face_index','x1','x2')});out=NS(**{key:arr(c['initial_'+key],wp.int32 if key=='count' else wp.float32 if key=='distance' else wp.vec3) for key in FIELDS});nr=len(c['world']);scratch=NS()
    for key,width,dtype in (('polygon',8,wp.vec3),('clipped',8,wp.vec3),('pn',4,wp.vec3),('pd',4,wp.float32),('idx1',3,wp.int32),('idx2',3,wp.int32),('n1',3,wp.vec3),('n2',3,wp.vec3),('end',3,wp.vec3),('face1',4,wp.vec3),('face2',4,wp.vec3)):setattr(scratch,key,wp.zeros((nr,width),dtype=dtype,device=stream.device))
    return m,d,q,query_outputs,frozen,out,scratch


def assign(c,m,d,q,query_outputs,frozen,out,scratch):
    gjk.assign(c,m,d,q,query_outputs)
    for key,v in vars(frozen).items():v.assign(c['frozen_'+key])
    for key in FIELDS:getattr(out,key).assign(c['initial_'+key])


def upstream(m,d,q,query_outputs,frozen,out,scratch):
    import warp as wp
    if q.world.size:wp.launch(reference_kernel(),q.world.size,block_dim=128,inputs=[q.world,q.pair,m.geom_size,d.geom_xmat,q.position1,q.position2]+[getattr(frozen,k) for k in ('vertices','vertex_index','face','face_index','x1','x2')]+list(vars(scratch).values())+[getattr(out,k) for k in FIELDS])


def check(c,i,got):
    n=int(got['count']);rows=[];scale=1+np.max(abs(c['size']))+max(np.max(abs(c['position1'][i])),np.max(abs(c['position2'][i])));budget=8*(512*U/(1-512*U))*scale
    def add(name,passed,**v):rows.append(dict(name=name,passed=bool(passed),**v))
    add('finite result and manifold capacity',0<=n<=4 and all(np.all(np.isfinite(got[k])) for k in FIELDS if k!='count'))
    if n==0:add('empty manifold zero result',all(np.all(got[k]==0) for k in FIELDS if k!='count'))
    for k in range(n):
        one=got['witness1'][k].astype(float);two=got['witness2'][k].astype(float);distance=float(got['distance'][k])
        fallback=n==1 and distance==0 and np.array_equal(got['witness1'][k],c['frozen_x1'][i]) and np.array_equal(got['witness2'][k],c['frozen_x2'][i])
        if not fallback:add('signed witness distance',abs(abs(distance)-np.linalg.norm(one-two))<=budget and distance<=budget,maximum_error=abs(abs(distance)-np.linalg.norm(one-two)))
        for side,point in ((1,one),(2,two)):
            w=int(c['world'][i]);g=int(c['pair'][i,side-1]);R=c['xmat'][w,g].astype(float);half=c['size'][w%len(c['size']),g].astype(float);local=R.T@(point-c['position'+str(side)][i]);outside=np.max(abs(local)-half);surface=np.min(abs(abs(local)-half))
            # Face-face clipping treats opposing normals within FACE_TOL as
            # aligned, then projects along the second face's normal. If v is
            # clipped into the first face's infinite prism, exact projection
            # is v-d*n1; pinned projection is v+d*n2. Their displacement is
            # |d|*||n1+n2|| <= |d|*sqrt(2*(1-FACE_TOL)). This is a source-level
            # geometric approximation, separate from the FP32 error allowance.
            approximation=0. if side==2 or fallback else abs(distance)*np.sqrt(2*(1-float(np.float32(np.cos(.0889)))))
            add('box slab/surface within pinned face-alignment approximation '+str(side),outside<=budget+approximation and surface<=budget+approximation,maximum_outside=float(outside),surface_error=float(surface),floating_point_budget=float(budget),algorithmic_projection_bound=float(approximation))
    return rows,float(budget)


class Replay:
    def __init__(self,c,args,stream):
        import warp as wp
        from libero.gpu.host.runtime.collision.mjwarp_convex import ManifoldProbe
        self.case,self.stream=c,stream;self.owners={};self.graphs={}
        for mode in ('upstream','cuda'):
            owner=construct(c,stream);self.owners[mode]=owner;m,d,q,qo,f,o,s=owner
            if mode=='cuda':self.binding=ManifoldProbe(m,d,q,qo,f,o,args.library,stream)
            with wp.ScopedCapture(stream=stream) as cap:
                if mode=='cuda':self.binding.enqueue()
                else:upstream(*owner)
            self.graphs[mode]=cap.graph
    def check(self,updated):
        import warp as wp
        c=current_case(self.case,updated);rows=[];values=[];budgets=[]
        for mode,owner in self.owners.items():
            m,d,q,qo,f,o,s=owner;assign(c,*owner);readonly={**vars(m),**vars(d),**{'request.'+k:v for k,v in vars(q).items()},**{'frozen.'+k:v for k,v in vars(f).items()},**{'query-hold.'+k:v for k,v in vars(qo).items()}};readonly={k:v for k,v in readonly.items() if hasattr(v,'numpy')};before={k:v.numpy().view(np.uint8).copy() for k,v in readonly.items()};wp.capture_launch(self.graphs[mode],stream=self.stream);wp.synchronize_stream(self.stream);got={k:getattr(o,k).numpy() for k in FIELDS};values.append(got);checks=[]
            for i in range(len(c['world'])):
                local,budget=check(c,i,{k:v[i] for k,v in got.items()});checks += [dict(v,request=i) for v in local]
                if mode=='cuda':budgets.append(budget)
            held=all(np.array_equal(v.numpy().view(np.uint8),before[k]) for k,v in readonly.items());rows.append(dict(mode=mode,passed=held and all(v['passed'] for v in checks),checks=checks,readonly_inputs_held=held,manifold_counts=np.bincount(got['count'],minlength=5).tolist()))
        parity=[]
        for i in range(len(c['world'])):
            a,b=values;na,nb=int(a['count'][i]),int(b['count'][i]);error=0.
            if na and nb:
                pa=(a['witness1'][i,:na].astype(float)+a['witness2'][i,:na].astype(float))*.5;pb=(b['witness1'][i,:nb].astype(float)+b['witness2'][i,:nb].astype(float))*.5;d=np.linalg.norm(pa[:,None]-pb,axis=2);error=max(float(np.min(d,axis=0).max()),float(np.min(d,axis=1).max()))
            parity.append(dict(request=i,passed=bool(na==nb and error<=2*budgets[i]),count_equal=na==nb,point_set_error=error,budget=2*budgets[i]))
        return dict(name=c['name'],updated=updated,modes=rows,parity=parity,passed=all(v['passed'] for v in rows+parity)),values


def signature(args):
    s=gjk.signature(args);s['tool_sha256']=digest(__file__);return s


def main(argv=None,*,args=None):
    if args is None:args=add_arguments(argparse.ArgumentParser(description=__doc__)).parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.cpu_reference:raise ValueError('CPU2.3.7 box manifold has a different depth/contact algorithm; use independent slab/surface checks')
    from libero.gpu import precision
    precision.select('fp32')
    from libero.gpu.host.runtime.collision.mjwarp_convex import GJK_REQUESTS,GJK_OUTPUTS
    gjk.GJK_REQUESTS,gjk.GJK_OUTPUTS=GJK_REQUESTS,GJK_OUTPUTS
    import warp as wp
    from warp._src.build import init_kernel_cache
    from libero.gpu.host.backends.mjwarp.model import require_upstream
    from libero.gpu.host.build.compilation import compilation_scope,install_warp_guard
    require_upstream();init_kernel_cache(args.build_root/'mjwarp/warp-cache');install_warp_guard();wp.init();stream=wp.Stream('cuda:0');sig=signature(args);prep=args.output/'preparation.json'
    if not args.prepare and (not prep.is_file() or json.loads(prep.read_text())['signature']!=sig):raise RuntimeError('Prepare exact box manifold probe offline first')
    with compilation_scope(args.prepare) as scope,wp.ScopedStream(stream):
        replays=[Replay(c,args,stream) for c in cases(args)]
        if args.prepare:
            wp.synchronize_stream(stream);prep.write_text(json.dumps(dict(signature=sig,compilation_events=scope['events'],physics_integrations=0),indent=2)+'\n');print(json.dumps(dict(prepared=True,captures=2*len(replays),compilations=len(scope['events']))),flush=True);return 0
        rows=[]
        for replay in replays:
            for updated in (False,True):
                row,values=replay.check(updated);rows.append(row);failed=[dict(mode=m['mode'],**v) for m in row['modes'] for v in m['checks'] if not v['passed']];print(json.dumps(dict(name=row['name'],updated=updated,passed=row['passed'],failures=failed[:8],parity_failures=[v for v in row['parity'] if not v['passed']][:8])),flush=True)
                if not row['passed']:np.savez(args.output/'failure.npz',**{mode+'.'+k:v for mode,got in zip(('upstream','cuda'),values) for k,v in got.items()});break
            if not row['passed']:break
        passed=all(row['passed'] for row in rows);(args.output/'summary.json').write_text(json.dumps(dict(passed=passed,signature=sig,cases=rows,runtime_compilations=scope['events'],worlds=2,physics_integrations=0,performance=None,production_collision_replaced=False),indent=2)+'\n');print(json.dumps(dict(passed=passed,cases=len(rows))),flush=True);return 0 if passed else 2
if __name__=='__main__':raise SystemExit(main())
