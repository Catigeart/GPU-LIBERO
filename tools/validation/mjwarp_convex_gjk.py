"""Frozen N=2 GJK loop conformance, independent support/convex-QP bounds."""
import argparse,functools,json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from .mjwarp_convex_support import add_arguments,cases,digest,input_identity,construct as geometry_construct
from .mjwarp_convex_reference import support,U
from .mjwarp_convex_simplex_reference import closest
RAW=('separated_out','dim_out','index1_out','index2_out','vertex1_out','vertex2_out','dist_out','x1_out','x2_out','simplex_out','simplex1_out','simplex2_out')


def current_case(c,updated):
    c={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in c.items()}
    if updated:
        shift=np.array([.013,-.017,.019],np.float32)
        for key in ('position1','position2','initial1','initial2'):c[key]+=shift
        c['size']*=np.float32(.97)
    return c


@functools.lru_cache(maxsize=1)
def reference_kernel():
    import warp as wp
    from mujoco_warp._src.collision_core import Geom
    from mujoco_warp._src.collision_gjk import gjk
    @wp.struct
    class Output:
        separated_out:wp.array[int]
        dim_out:wp.array[int]
        index1_out:wp.array[int]
        index2_out:wp.array[int]
        vertex1_out:wp.array2d[int]
        vertex2_out:wp.array2d[int]
        dist_out:wp.array[float]
        x1_out:wp.array[wp.vec3]
        x2_out:wp.array[wp.vec3]
        simplex_out:wp.array2d[wp.vec3]
        simplex1_out:wp.array2d[wp.vec3]
        simplex2_out:wp.array2d[wp.vec3]
    @wp.func
    def geom_at(w:int,g:int,position:wp.vec3,cache:int,margin:float,types:wp.array[int],ids:wp.array2d[int],size:wp.array2d[wp.vec3],R:wp.array2d[wp.mat33],adr:wp.array[int],num:wp.array[int],gadr:wp.array[int],vertices:wp.array[wp.vec3],graph:wp.array[int]):
        a=Geom();a.pos=position;a.rot=R[w,g];a.size=size[w%size.shape[0],g];a.index=cache;a.margin=margin
        if types[g]==7:
            m=ids[w%ids.shape[0],g];a.vertadr=adr[m];a.vertnum=num[m];a.graphadr=gadr[m];a.vert=vertices;a.graph=graph
        return a
    @wp.kernel(module='unique',enable_backward=False)
    def kernel(types:wp.array[int],ids:wp.array2d[int],size:wp.array2d[wp.vec3],R:wp.array2d[wp.mat33],adr:wp.array[int],num:wp.array[int],gadr:wp.array[int],vertices:wp.array[wp.vec3],graph:wp.array[int],world:wp.array[int],pair:wp.array[wp.vec2i],iterations:wp.array[int],discrete:wp.array[int],cache1:wp.array[int],cache2:wp.array[int],position1:wp.array[wp.vec3],position2:wp.array[wp.vec3],initial1:wp.array[wp.vec3],initial2:wp.array[wp.vec3],margin1:wp.array[float],margin2:wp.array[float],tolerance:wp.array[float],cutoff:wp.array[float],out:Output):
        i=wp.tid();w=world[i];g1=pair[i][0];g2=pair[i][1]
        a=geom_at(w,g1,position1[i],cache1[i],margin1[i],types,ids,size,R,adr,num,gadr,vertices,graph);b=geom_at(w,g2,position2[i],cache2[i],margin2[i],types,ids,size,R,adr,num,gadr,vertices,graph)
        result=gjk(tolerance[i],iterations[i],a,b,initial1[i],initial2[i],types[g1],types[g2],cutoff[i],discrete[i]!=0)
        out.separated_out[i]=int(result.separated);out.dim_out[i]=result.dim;out.index1_out[i]=result.index1;out.index2_out[i]=result.index2;out.dist_out[i]=result.dist;out.x1_out[i]=result.x1;out.x2_out[i]=result.x2
        for k in range(4):
            out.vertex1_out[i,k]=result.simplex_index1[k];out.vertex2_out[i,k]=result.simplex_index2[k];out.simplex_out[i,k]=result.simplex[k];out.simplex1_out[i,k]=result.simplex1[k];out.simplex2_out[i,k]=result.simplex2[k]
    return kernel,Output


def construct(c,stream):
    import warp as wp
    # Reuse cold geometry fixture layout; its support request arrays are unused.
    m,d,_,_=geometry_construct(c,stream);arr=lambda value,kind:wp.array(np.ascontiguousarray(value),dtype=getattr(wp,kind),device=stream.device)
    q=NS(**{k:arr(c[k],kind) for k,kind in GJK_REQUESTS});o=NS(**{k:arr(c['initial_'+k],kind) for k,kind,_ in GJK_OUTPUTS});return m,d,q,o


def assign(c,m,d,q,o):
    m.geom_size.assign(c['size'])
    for k,_ in GJK_REQUESTS:getattr(q,k).assign(c[k])
    for k,_,_ in GJK_OUTPUTS:getattr(o,k).assign(c['initial_'+k])


def upstream(m,d,q,o):
    import warp as wp
    kernel,Outputs=reference_kernel();out=Outputs()
    for k in RAW:setattr(out,k,getattr(o,k))
    if q.world.size:wp.launch(kernel,q.world.size,block_dim=128,inputs=[m.geom_type,m.geom_dataid,m.geom_size,d.geom_xmat,m.mesh_vertadr,m.mesh_vertnum,m.mesh_graphadr,m.mesh_vert,m.mesh_graph]+[getattr(q,k) for k,_ in GJK_REQUESTS]+[out])


def support_at(c,i,side,direction):
    w=int(c['world'][i]);g=int(c['pair'][i,side-1]);p=c['xpos'].copy();p[w,g]=c['position'+str(side)][i]
    a={**c,'world':np.array([w]),'geom':np.array([g]),'xpos':p,'direction':np.array([direction]),'request_margin':np.array([c['margin'+str(side)][i]])}
    return support(a,0)


def check(c,i,got,mode):
    n=int(got['dim_out']);distance=float(got['dist_out']);x1=np.asarray(got['x1_out'],float);x2=np.asarray(got['x2_out'],float)
    scale=1+max(np.max(abs(c['position1'][i])),np.max(abs(c['position2'][i])),np.max(abs(c['size'])))
    gamma=512*U/(1-512*U);budget=8*gamma*scale
    rows=[]
    def add(name,passed,**v):rows.append(dict(name=name,passed=bool(passed),**v))
    add('finite result and simplex dimension',0<=n<=4 and np.isfinite(distance) and np.all(np.isfinite(x1)) and np.all(np.isfinite(x2)))
    cutoff_return=distance==float(np.float32(1.e30))
    if cutoff_return:
        add('cutoff empty/zero witness contract',n==0 and int(got['separated_out'])==1 and np.all(x1==0) and np.all(x2==0))
        add('cutoff was enabled',c['cutoff'][i]<np.float32(1.e30))
    else:
        if n:
            s=got['simplex_out'][:n].astype(float);a=got['simplex1_out'][:n].astype(float);b=got['simplex2_out'][:n].astype(float);qp=closest(s)
            add('Minkowski active simplex identity',np.max(abs(s-(a-b)))<=budget)
            add('active simplex independent QP distance',abs(distance-qp['distance'])<=budget,maximum_error=abs(distance-qp['distance']),budget=budget)
            for side,points,x in ((1,a,x1),(2,b,x2)):
                # Independent feasibility permits legal different witness choices
                # on a nonunique exposed feature; no signed-volume code reused.
                error=closest(points-x)['distance'];add('witness in active support hull '+str(side),error<=budget,maximum_error=error)
                g=int(c['pair'][i,side-1]);kind=int(c['geom_type'][g]);index=got['vertex'+str(side)+'_out'][:n]
                if kind==7:
                    w=int(c['world'][i]);mesh=int(c['geom_dataid'][w%len(c['geom_dataid']),g]);num=int(c['mesh_vertnum'][mesh]);valid=np.all((index>=0)&(index<num));add('active mesh vertex range '+str(side),valid)
                    if valid:
                        adr=int(c['mesh_vertadr'][mesh]);expected=c['mesh_vert'][adr+index].astype(float)@c['xmat'][w,g].astype(float).T+c['position'+str(side)][i]
                        offset=np.linalg.norm(points-expected,axis=1);margin=max(0,float(c['margin'+str(side)][i]))*.5
                        add('active mesh points match indexed vertices/margin '+str(side),np.max(abs(offset-margin))<=budget)
                else:add('active primitive feature range '+str(side),np.all((index==-1) if kind in (2,3,4) else ((index>=0)&(index<=(1 if kind==5 else 7)))))
        else:add('immediate return initial witnesses',np.array_equal(got['x1_out'],c['initial1'][i]) and np.array_equal(got['x2_out'],c['initial2'][i]))
        error=abs(distance-np.linalg.norm(x1-x2));add('distance/witness norm identity',error<=budget,maximum_error=error)
        direction=x1-x2;length=np.linalg.norm(direction)
        if length>budget:
            direction/=length;one=support_at(c,i,1,-direction);two=support_at(c,i,2,direction);lower=-one['projection']-two['projection'];bound=one['budget']+two['budget']+budget
            add('independent support lower bound below primal distance',lower<=distance+bound,lower=lower,upper=distance,budget=bound)
            if abs(lower)>bound:add('separation flag outside FP32 support interval',bool(got['separated_out'])==(lower>0))
    # Final caches are from the last support query, which need not survive
    # simplex compression, and are not the uncached final separation test.
    from libero.gpu.host.runtime.collision.mjwarp_convex import validate_requests
    for side in (1,2):
        try:validate_requests(c['world'][i:i+1],c['pair'][i:i+1,side-1],np.array([got['index'+str(side)+'_out']]),*(c[k] for k in ('geom_type','geom_dataid','mesh_vertadr','mesh_vertnum','mesh_graphadr','mesh_graph')),2);valid=True
        except ValueError:valid=False
        add('final mode-specific cache range '+str(side),valid)
    if mode=='cuda':
        used=int(got['iterations_out']);queries=int(got['queries_out']);stop=int(got['stop_out']);add('iteration/query/stop controls',0<=used<=int(c['iterations'][i]) and 0<=queries<=2*(used+1) and 0<=stop<=7)
        if cutoff_return:
            add('cutoff probe certificate',int(got['valid_out'])==1 and stop in (4,5))
            direction=got['direction_out'].astype(float);one=support_at(c,i,1,-direction);two=support_at(c,i,2,direction);lower=-one['projection']-two['projection'];bound=one['budget']+two['budget']+budget
            # Discrete near-zero direction tuning can make the algorithm's
            # selected projection exceed the exact directional support minimum.
            add('independent cutoff separating support plane',lower+bound>=float(c['cutoff'][i]),lower=lower,budget=bound)
    return rows,budget


class Replay:
    def __init__(self,c,args,stream):
        import warp as wp
        from libero.gpu.host.runtime.collision.mjwarp_convex import GJKProbe
        self.case,self.stream=c,stream;self.owners={};self.graphs={}
        for mode in ('upstream','cuda'):
            owner=construct(c,stream);self.owners[mode]=owner;m,d,q,o=owner
            if mode=='cuda':self.binding=GJKProbe(m,d,q,o,args.library,stream)
            with wp.ScopedCapture(stream=stream) as cap:
                if mode=='cuda':self.binding.enqueue()
                else:upstream(*owner)
            self.graphs[mode]=cap.graph
    def check(self,updated):
        import warp as wp
        c=current_case(self.case,updated);rows=[];values=[];budgets=[]
        for mode,(m,d,q,o) in self.owners.items():
            assign(c,m,d,q,o);readonly={**vars(m),**vars(d),**{'request.'+k:v for k,v in vars(q).items()}};readonly={k:v for k,v in readonly.items() if hasattr(v,'numpy')};before={k:v.numpy().view(np.uint8).copy() for k,v in readonly.items()}
            wp.capture_launch(self.graphs[mode],stream=self.stream);wp.synchronize_stream(self.stream);got={k:getattr(o,k).numpy() for k,_,_ in GJK_OUTPUTS};values.append(got);checks=[]
            for i in range(len(c['world'])):
                one,budget=check(c,i,{k:v[i] for k,v in got.items()},mode);checks += [dict(v,request=i) for v in one]
                if mode=='cuda':budgets.append(budget)
            held=all(np.array_equal(v.numpy().view(np.uint8),before[k]) for k,v in readonly.items())
            # Keep original mathematical defects visible. Its expanded triangle
            # determinants can violate barycentric feasibility; candidate math
            # remains a mandatory independent gate at the unchanged bound.
            math_names=('active simplex independent QP distance','witness in active support hull 1','witness in active support hull 2','independent support lower bound below primal distance','separation flag outside FP32 support interval')
            contracts=held and all(v['passed'] for v in checks if v['name'] not in math_names)
            rows.append(dict(mode=mode,passed=held and all(v['passed'] for v in checks),contracts_passed=contracts,checks=checks,readonly_inputs_held=held))
        parity=[]
        for i in range(len(c['world'])):
            a,b=(v['dist_out'][i] for v in values);cut_a=a==np.float32(1.e30);cut_b=b==np.float32(1.e30)
            if cut_a and cut_b:valid=True;error=0.
            elif cut_a or cut_b:
                finite=float(b if cut_a else a);error=abs(finite-float(c['cutoff'][i]));valid=error<=2*budgets[i]
            else:error=abs(float(a)-float(b));valid=error<=2*budgets[i]
            parity.append(dict(request=i,passed=bool(valid),distance_error=error,budget=2*budgets[i]))
        return dict(name=c['name'],updated=updated,modes=rows,parity=parity,passed=all(v['passed'] for v in parity) and all(v['passed'] if v['mode']=='cuda' else v['contracts_passed'] for v in rows)),values


def signature(args):
    from .mjwarp_convex_simplex import signature as prior
    s=prior(args);s.update(tool_sha256=digest(__file__),reference_sha256=digest(Path(__file__).with_name('mjwarp_convex_reference.py')),simplex_reference_sha256=digest(Path(__file__).with_name('mjwarp_convex_simplex_reference.py')));return s


def main(argv=None,*,args=None):
    if args is None:args=add_arguments(argparse.ArgumentParser(description=__doc__)).parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.cpu_reference:raise ValueError('GJK uses independent FP64 support/QP checks inline; CPU2.3.7 implements MPR rather than this function')
    from libero.gpu import precision
    precision.select('fp32')
    global GJK_REQUESTS,GJK_OUTPUTS
    from libero.gpu.host.runtime.collision.mjwarp_convex import GJK_REQUESTS,GJK_OUTPUTS
    import warp as wp
    from warp._src.build import init_kernel_cache
    from libero.gpu.host.backends.mjwarp.model import require_upstream
    from libero.gpu.host.build.compilation import compilation_scope,install_warp_guard
    require_upstream();init_kernel_cache(args.build_root/'mjwarp/warp-cache');install_warp_guard();wp.init();stream=wp.Stream('cuda:0');sig=signature(args);prep=args.output/'preparation.json'
    if not args.prepare and (not prep.is_file() or json.loads(prep.read_text())['signature']!=sig):raise RuntimeError('Prepare exact GJK probe offline first')
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
