"""Rigid shape-aware GJK/EPA, frozen N=2 requests and caller-owned scratch."""
import argparse,functools,json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from . import mjwarp_convex_gjk as gjk
from .mjwarp_convex_support import add_arguments,cases,digest
from .mjwarp_convex_reference import U
from .mjwarp_convex_simplex_reference import closest



def current_case(c,updated):
    value=gjk.current_case(c,updated)
    if updated and c.get('use_frozen',False):
        # A frozen simplex belongs to fixed shape sizes. Translate both witness
        # hulls together, retaining its Minkowski simplex and barycentric scale.
        value['size']=c['size'].copy();shift=np.array([.013,-.017,.019],np.float32)
        for key in ('simplex1','simplex2','x1','x2'):value['frozen_'+key]=c['frozen_'+key]+shift
    return value

@functools.lru_cache(maxsize=1)
def reference_kernel():
    import warp as wp
    from mujoco_warp._src.collision_core import Geom
    from mujoco_warp._src.collision_gjk import gjk_phase,epa_phase,GJKResult
    _,Output=gjk.reference_kernel()
    @wp.struct
    class Query:
        world:wp.array[int]
        pair:wp.array[wp.vec2i]
        iterations:wp.array[int]
        discrete:wp.array[int]
        cache1:wp.array[int]
        cache2:wp.array[int]
        position1:wp.array[wp.vec3]
        position2:wp.array[wp.vec3]
        initial1:wp.array[wp.vec3]
        initial2:wp.array[wp.vec3]
        margin1:wp.array[float]
        margin2:wp.array[float]
        tolerance:wp.array[float]
        cutoff:wp.array[float]
    @wp.struct
    class Frozen:
        simplex:wp.array2d[wp.vec3]
        simplex1:wp.array2d[wp.vec3]
        simplex2:wp.array2d[wp.vec3]
        distance:wp.array[float]
        x1:wp.array[wp.vec3]
        x2:wp.array[wp.vec3]
        count:wp.array[int]
        vertex1:wp.array2d[int]
        vertex2:wp.array2d[int]
    @wp.func
    def geom_at(w:int,g:int,position:wp.vec3,cache:int,margin:float,types:wp.array[int],ids:wp.array2d[int],size:wp.array2d[wp.vec3],R:wp.array2d[wp.mat33],adr:wp.array[int],num:wp.array[int],gadr:wp.array[int],vertices:wp.array[wp.vec3],graph:wp.array[int]):
        a=Geom();a.pos=position;a.rot=R[w,g];a.size=size[w%size.shape[0],g];a.index=cache;a.margin=margin
        if types[g]==7:
            m=ids[w%ids.shape[0],g];a.vertadr=adr[m];a.vertnum=num[m];a.graphadr=gadr[m];a.vert=vertices;a.graph=graph
        return a
    @wp.kernel(module='unique',enable_backward=False)
    def kernel(types:wp.array[int],ids:wp.array2d[int],size:wp.array2d[wp.vec3],R:wp.array2d[wp.mat33],adr:wp.array[int],num:wp.array[int],gadr:wp.array[int],vertices:wp.array[wp.vec3],graph:wp.array[int],q:Query,out:Output,vert:wp.array2d[wp.vec3],vertex_index:wp.array2d[int],face:wp.array2d[int],pr:wp.array2d[wp.vec3],norm2:wp.array2d[float],horizon:wp.array2d[int],overflow:wp.array[int],needs_out:wp.array[int],ncontact_out:wp.array[int],face_index_out:wp.array[int],epa_iterations:int,use_frozen:bool,f:Frozen):
        i=wp.tid();w=q.world[i];g1=q.pair[i][0];g2=q.pair[i][1]
        a=geom_at(w,g1,q.position1[i],q.cache1[i],q.margin1[i],types,ids,size,R,adr,num,gadr,vertices,graph);b=geom_at(w,g2,q.position2[i],q.cache2[i],q.margin2[i],types,ids,size,R,adr,num,gadr,vertices,graph)
        needs=bool(True);dist=float(0.0);ncontact=int(1);x1=wp.vec3();x2=wp.vec3();result=GJKResult()
        if use_frozen:
            result.dim=f.count[i];result.dist=f.distance[i];result.x1=f.x1[i];result.x2=f.x2[i]
            for k in range(4):
                result.simplex[k]=f.simplex[i,k];result.simplex1[k]=f.simplex1[i,k];result.simplex2[k]=f.simplex2[i,k];result.simplex_index1[k]=f.vertex1[i,k];result.simplex_index2[k]=f.vertex2[i,k]
            dist=result.dist;x1=result.x1;x2=result.x2
        else:needs,dist,ncontact,x1,x2,result,a,b=gjk_phase(q.tolerance[i],q.cutoff[i],q.iterations[i],a,b,types[g1],types[g2],q.initial1[i],q.initial2[i])
        index=int(-1)
        if needs:dist,ncontact,x1,x2,index=epa_phase(q.tolerance[i],epa_iterations,result,a,b,types[g1],types[g2],vert[i],vertex_index[i],face[i],pr[i],norm2[i],horizon[i],False,w,overflow)
        out.separated_out[i]=int(result.separated);out.dim_out[i]=result.dim;out.index1_out[i]=a.index;out.index2_out[i]=b.index;out.dist_out[i]=dist;out.x1_out[i]=x1;out.x2_out[i]=x2
        for k in range(4):
            out.vertex1_out[i,k]=result.simplex_index1[k];out.vertex2_out[i,k]=result.simplex_index2[k];out.simplex_out[i,k]=result.simplex[k];out.simplex1_out[i,k]=result.simplex1[k];out.simplex2_out[i,k]=result.simplex2[k]
        needs_out[i]=int(needs);ncontact_out[i]=ncontact;face_index_out[i]=index
    return kernel,Query,Output,Frozen


def construct(c,stream):
    import warp as wp
    from libero.gpu.host.runtime.collision.mjwarp_convex import CCD_OUTPUTS
    m,d,q,o=gjk.construct(c,stream);nr=len(c['world']);iterations=int(c['epa_iterations']);nv=10+2*iterations;nf=int(c['face_capacity']);nh=int(c['horizon_capacity'])
    for key in CCD_OUTPUTS:setattr(o,key,wp.full(nr,-73,dtype=wp.int32,device=stream.device))
    scratch=NS()
    for key,dtype,width in (('vertices',wp.vec3,nv),('projection',wp.vec3,nf),('norm2',wp.float32,nf),('vertex_index',wp.int32,nv),('face',wp.int32,nf),('horizon',wp.int32,nh)):setattr(scratch,key,wp.full((nr,width),-79,dtype=dtype,device=stream.device))
    scratch.overflow=wp.array(c['overflow'],dtype=wp.int32,device=stream.device)
    frozen=NS()
    for key,dtype,width in (('simplex',wp.vec3,4),('simplex1',wp.vec3,4),('simplex2',wp.vec3,4),('distance',wp.float32,1),('x1',wp.vec3,1),('x2',wp.vec3,1),('count',wp.int32,1),('vertex1',wp.int32,4),('vertex2',wp.int32,4)):
        shape=(nr,4) if width==4 else (nr,)
        value=c.get('frozen_'+key)
        setattr(frozen,key,wp.zeros(shape,dtype=dtype,device=stream.device) if value is None else wp.array(value,dtype=dtype,device=stream.device))
    return m,d,q,o,scratch,frozen


def upstream(m,d,q,o,scratch,frozen,iterations,use_frozen):
    import warp as wp
    kernel,Query,Outputs,Frozen=reference_kernel();query=Query();out=Outputs();f=Frozen()
    for k,v in vars(frozen).items():setattr(f,k,v)
    for k,_ in gjk.GJK_REQUESTS:setattr(query,k,getattr(q,k))
    for k in gjk.RAW:setattr(out,k,getattr(o,k))
    if q.world.size:wp.launch(kernel,q.world.size,block_dim=128,inputs=[m.geom_type,m.geom_dataid,m.geom_size,d.geom_xmat,m.mesh_vertadr,m.mesh_vertnum,m.mesh_graphadr,m.mesh_vert,m.mesh_graph,query,out]+[getattr(scratch,k) for k in ('vertices','vertex_index','face','projection','norm2','horizon','overflow')]+[o.needs_epa_out,o.ncontact_out,o.face_index_out,iterations,use_frozen,f])


def check(c,i,got,workspace,mode):
    from libero.gpu.host.runtime.collision.mjwarp_convex import validate_requests
    distance=float(got['dist_out']);x1=got['x1_out'].astype(float);x2=got['x2_out'].astype(float);nc=int(got['ncontact_out']);need=int(got['needs_epa_out']);index=int(got['face_index_out']);rows=[]
    scale=1+max(np.max(abs(c['size'])),np.max(abs(c['mesh_vert'])),np.max(abs(c['position1'][i])),np.max(abs(c['position2'][i])));budget=8*(512*U/(1-512*U))*scale
    def add(name,passed,**v):rows.append(dict(name=name,passed=bool(passed),**v))
    add('finite result/count/phase controls',np.isfinite(distance) and np.all(np.isfinite(x1)) and np.all(np.isfinite(x2)) and nc in (0,1) and need in (0,1) and index>=-1)
    if nc==0:add('empty EPA failure result',distance==float(np.float32(1.e30)) and np.all(x1==0) and np.all(x2==0) and index==-1 and need==1)
    elif distance!=float(np.float32(1.e30)):
        # Inflating a center-line result can cross its witnesses at deep but
        # nonzero separation; the signed distance still has this absolute norm.
        error=abs(abs(distance)-np.linalg.norm(x1-x2));add('signed distance/witness norm',error<=budget,maximum_error=error,budget=budget)
    if nc==1 and distance!=float(np.float32(1.e30)) and not (need==0 and distance==0 and np.array_equal(c['initial1'][i],c['initial2'][i])):
        w=int(c['world'][i]);a,b=map(int,c['pair'][i]);type1,type2=map(int,c['geom_type'][[a,b]]);one=c['size'][w%len(c['size']),a].astype(float);two=c['size'][w%len(c['size']),b].astype(float)
        R1=c['xmat'][w,a].astype(float);R2=c['xmat'][w,b].astype(float);delta=c['position2'][i].astype(float)-c['position1'][i].astype(float);analytic=None
        margins=float(c['margin1'][i]+c['margin2'][i])*.5
        if type1==6 and type2==2:
            q=abs(R1.T@delta)-one;analytic=np.linalg.norm(np.maximum(q,0))+min(float(max(q)),0)-two[0]-margins
        elif type1==6 and type2 in (3,6) and np.array_equal(R1,np.eye(3)) and np.array_equal(R2,np.eye(3)):
            if type2==6:
                q=abs(delta)-(one+two)
                if max(q)<=0:analytic=float(max(q))-margins
            else:
                half=one.copy();half[2]+=two[1];q=abs(delta)-half;analytic=np.linalg.norm(np.maximum(q,0))+min(float(max(q)),0)-two[0]-margins
        if analytic is not None:add('independent rounded-box/SAT signed distance',abs(distance-analytic)<=budget,maximum_error=abs(distance-analytic),budget=budget)
    if c.get('verify_box_overlap',False):
        from .mjwarp_box_reference import box_separation
        w=int(c['world'][i]);a,b=map(int,c['pair'][i])
        if tuple(c['geom_type'][[a,b]])!=(6,6) or c['margin1'][i]!=0 or c['margin2'][i]!=0:
            raise ValueError('Box overlap regression requires zero-margin BOX pairs')
        gap=box_separation(c['position1'][i],c['position2'][i],c['xmat'][w,a],c['xmat'][w,b],
                           c['size'][w%len(c['size']),a],c['size'][w%len(c['size']),b])
        tolerance=float(c['tolerance'][i])
        if gap < -tolerance:
            add('confirmed box overlap emits penetration',nc==1 and distance<0,independent_gap=gap)
        elif gap > tolerance:
            add('confirmed box separation emits no penetration',distance>=0,independent_gap=gap)
    for side in (1,2):
        try:validate_requests(c['world'][i:i+1],c['pair'][i:i+1,side-1],np.array([got['index'+str(side)+'_out']]),*(c[k] for k in ('geom_type','geom_dataid','mesh_vertadr','mesh_vertnum','mesh_graphadr','mesh_graph')),2);valid=True
        except ValueError:valid=False
        add('phase output cache range '+str(side),valid)
    if mode=='cuda':add('polytope capacities',0<=got['nvert_out']<=(10+2*int(c['epa_iterations']))//2 and 0<=got['nface_out']<=int(c['face_capacity']) and -1<=got['status_out']<=12)
    if index>=0:
        nf=int(c['face_capacity']);add('selected face index range',index<nf)
        if index<nf:
            packed=int(workspace['face'][i,index]);ids=np.array([packed&1023,(packed>>10)&1023,(packed>>20)&1023]);limit=workspace['vertices'].shape[1]//2;valid=np.all(ids<limit);add('packed face vertex ranges',valid)
            if valid:
                one=workspace['vertices'][i,2*ids].astype(float);two=workspace['vertices'][i,2*ids+1].astype(float);s=one-two;point=workspace['projection'][i,index].astype(float);w=np.linalg.lstsq(np.r_[s.T,np.ones((1,3))],np.r_[point,1.],rcond=None)[0];projection=w@s
                add('EPA plane affine projection residual',np.linalg.norm(projection-point)<=budget and abs(sum(w)-1)<=budget)
                add('EPA witness affine identity',max(np.max(abs(w@one-x1)),np.max(abs(w@two-x2)))<=budget)
                add('EPA face distance identity',abs(abs(distance)-np.sqrt(float(workspace['norm2'][i,index])))<=budget)
    return rows,budget


class Replay:
    def __init__(self,c,args,stream):
        import warp as wp
        from libero.gpu.host.runtime.collision.mjwarp_convex import CCDProbe
        self.case,self.stream=c,stream;self.owners={};self.graphs={}
        for mode in ('upstream','cuda'):
            owner=construct(c,stream);self.owners[mode]=owner;m,d,q,o,s,frozen=owner
            if mode=='cuda':self.binding=CCDProbe(m,d,q,o,s,int(c['epa_iterations']),args.library,stream,frozen=frozen if c.get('use_frozen',False) else None)
            with wp.ScopedCapture(stream=stream) as cap:
                if mode=='cuda':self.binding.enqueue()
                else:upstream(*owner,int(c['epa_iterations']),bool(c.get('use_frozen',False)))
            self.graphs[mode]=cap.graph
    def check(self,updated):
        import warp as wp
        from libero.gpu.host.runtime.collision.mjwarp_convex import CCD_OUTPUTS
        c=current_case(self.case,updated);rows=[];values=[];budgets=[]
        for mode,(m,d,q,o,s,frozen) in self.owners.items():
            gjk.assign(c,m,d,q,o);s.overflow.assign(c['overflow'])
            if c.get('use_frozen',False):
                for key,value in vars(frozen).items():value.assign(c['frozen_'+key])
            for key in CCD_OUTPUTS:getattr(o,key).fill_(-73)
            readonly={**vars(m),**vars(d),**{'request.'+k:v for k,v in vars(q).items()},**{'frozen.'+k:v for k,v in vars(frozen).items()}};readonly={k:v for k,v in readonly.items() if hasattr(v,'numpy')};before={k:v.numpy().view(np.uint8).copy() for k,v in readonly.items()}
            wp.capture_launch(self.graphs[mode],stream=self.stream);wp.synchronize_stream(self.stream);got={k:getattr(o,k).numpy() for k in gjk.RAW+CCD_OUTPUTS};scratch={k:v.numpy() for k,v in vars(s).items()};values.append({**got,**{'scratch.'+k:v for k,v in scratch.items()}});checks=[]
            for i in range(len(c['world'])):
                one,budget=check(c,i,{k:v[i] for k,v in got.items()},scratch,mode);checks += [dict(v,request=i) for v in one]
                if mode=='cuda':budgets.append(budget)
            held=all(np.array_equal(v.numpy().view(np.uint8),before[k]) for k,v in readonly.items());rows.append(dict(mode=mode,passed=held and all(v['passed'] for v in checks),checks=checks,readonly_inputs_held=held,epa_requests=int(got['needs_epa_out'].sum()),empty_returns=int(np.count_nonzero(got['ncontact_out']==0))))
        parity=[]
        for i in range(len(c['world'])):
            a,b=(v['dist_out'][i] for v in values);cut_a=a==np.float32(1.e30);cut_b=b==np.float32(1.e30);error=0. if cut_a and cut_b else abs(float(a)-float(b));samecount=values[0]['ncontact_out'][i]==values[1]['ncontact_out'][i]
            parity.append(dict(request=i,passed=bool(error<=2*budgets[i] and samecount),distance_error=error,budget=2*budgets[i]))
        overflow=bool(np.array_equal(values[0]['scratch.overflow'],values[1]['scratch.overflow']));return dict(name=c['name'],updated=updated,modes=rows,parity=parity,overflow_parity=overflow,passed=overflow and all(v['passed'] for v in parity) and all(v['passed'] for v in rows)),values


def signature(args):
    s=gjk.signature(args);s['tool_sha256']=digest(__file__);s['gjk_tool_sha256']=digest(Path(gjk.__file__))
    s['box_reference_sha256']=digest(Path(__file__).with_name('mjwarp_box_reference.py'));return s


def cpu_reference(args):
    import mujoco
    if mujoco.__version__!='2.3.7':raise RuntimeError('CCD CPU oracle requires actual CPU2.3.7')
    from .mjwarp_convex_support import input_identity
    m=mujoco.MjModel.from_xml_string('<mujoco><size nconmax="128"/><worldbody><body><freejoint/><geom type="sphere" size=".2"/></body><body><freejoint/><geom type="sphere" size=".2"/></body></worldbody></mujoco>');d=mujoco.MjData(m);mujoco.mj_kinematics(m,d);m.geom_rbound[:]=100;m.geom_aabb[:]=[0,0,0,100,100,100];rows=[];excluded=0
    for original in cases(args):
        for updated in (False,True):
            c=current_case(original,updated)
            for i,(a,b) in enumerate(c['pair']):
                w=int(c['world'][i]);kind=c['geom_type'][[a,b]]
                if np.any(kind==7) or c['margin1'][i]!=0 or c['margin2'][i]!=0:excluded+=1;continue
                m.geom_type[:]=kind
                for side,g in enumerate((a,b)):
                    m.geom_size[side]=c['size'][w%len(c['size']),g];position=c['position'+str(side+1)][i];R=c['xmat'][w,g].ravel();d.geom_xpos[side]=position;d.geom_xmat[side]=R;d.xpos[side+1]=position;d.xmat[side+1]=R
                mujoco.mj_collision(m,d)
                # The old engine reports penetrating contacts only. Preserve
                # absence as an excluded distance, not a zero-distance oracle.
                if not d.ncon:excluded+=1;continue
                distance=min(float(d.contact[j].dist) for j in range(d.ncon));rows.append(dict(case=c['name'],updated=updated,request=i,types=kind.tolist(),distance=distance,passed=bool(np.isfinite(distance))))
    library=next(Path(mujoco.__file__).parent.glob('libmujoco.so*'));result=dict(passed=bool(rows) and all(v['passed'] for v in rows),checks=rows,excluded_mesh_margin_nonpenetrating=excluded,inputs=input_identity(args),tool_sha256=digest(__file__),version=mujoco.__version__,library=str(library),library_sha256=digest(library),physics_integrations=0,domain='actual CPU2.3.7 public mj_collision common rigid primitives; older MPR/SAT/analytic paths, no GJK/EPA equivalence claim')
    (args.output/'cpu-oracle.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=result['passed'],checks=len(rows),excluded=excluded)),flush=True);return 0 if result['passed'] else 2


def main(argv=None,*,args=None):
    if args is None:args=add_arguments(argparse.ArgumentParser(description=__doc__)).parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.cpu_reference:return cpu_reference(args)
    from libero.gpu import precision
    precision.select('fp32')
    from libero.gpu.host.runtime.collision.mjwarp_convex import GJK_REQUESTS,GJK_OUTPUTS
    gjk.GJK_REQUESTS,gjk.GJK_OUTPUTS=GJK_REQUESTS,GJK_OUTPUTS
    import warp as wp
    from warp._src.build import init_kernel_cache
    from libero.gpu.host.backends.mjwarp.model import require_upstream
    from libero.gpu.host.build.compilation import compilation_scope,install_warp_guard
    require_upstream();init_kernel_cache(args.build_root/'mjwarp/warp-cache');install_warp_guard();wp.init();stream=wp.Stream('cuda:0');sig=signature(args);prep=args.output/'preparation.json'
    if not args.prepare and (not prep.is_file() or json.loads(prep.read_text())['signature']!=sig):raise RuntimeError('Prepare exact CCD probe offline first')
    with compilation_scope(args.prepare) as scope,wp.ScopedStream(stream):
        replays=[Replay(c,args,stream) for c in cases(args)]
        if args.prepare:
            wp.synchronize_stream(stream);prep.write_text(json.dumps(dict(signature=sig,compilation_events=scope['events'],physics_integrations=0),indent=2)+'\n');print(json.dumps(dict(prepared=True,captures=2*len(replays),compilations=len(scope['events']))),flush=True);return 0
        oracle=json.loads((args.output/'cpu-oracle.json').read_text())
        if not oracle['passed'] or oracle['inputs']!=sig['inputs'] or oracle['tool_sha256']!=digest(__file__):raise RuntimeError('Prepare matching actual CPU collision oracle')
        cpu_checks=[];rows=[]
        for replay in replays:
            for updated in (False,True):
                row,values=replay.check(updated)
                for ref in oracle['checks']:
                    if ref['case']!=row['name'] or ref['updated']!=updated:continue
                    i=ref['request'];distance=float(values[1]['dist_out'][i]);budget=row['parity'][i]['budget']
                    # Capacity exhaustion and explicit zero EPA iterations are
                    # semantics tests, not complete penetration-depth solves.
                    if values[1]['ncontact_out'][i]==0 or int(replay.case['epa_iterations'])==0:continue
                    error=abs(distance-ref['distance']);cpu_checks.append(dict(case=row['name'],updated=updated,request=i,within_error_bound=bool(error<=budget),error=error,budget=budget,role='version/algorithm observation, not an EPA depth oracle'))
                rows.append(row);failed=[dict(mode=m['mode'],**v) for m in row['modes'] for v in m['checks'] if not v['passed']];print(json.dumps(dict(name=row['name'],updated=updated,passed=row['passed'],failures=failed[:8],parity_failures=[v for v in row['parity'] if not v['passed']][:8],overflow_parity=row['overflow_parity'])),flush=True)
                if not row['passed']:np.savez(args.output/'failure.npz',**{mode+'.'+k:v for mode,got in zip(('upstream','cuda'),values) for k,v in got.items()});break
            if not row['passed']:break
        passed=all(row['passed'] for row in rows);(args.output/'summary.json').write_text(json.dumps(dict(passed=passed,signature=sig,cases=rows,cpu_observations=cpu_checks,runtime_compilations=scope['events'],worlds=2,physics_integrations=0,performance=None,production_collision_replaced=False),indent=2)+'\n');print(json.dumps(dict(passed=passed,cases=len(rows))),flush=True);return 0 if passed else 2
if __name__=='__main__':raise SystemExit(main())
