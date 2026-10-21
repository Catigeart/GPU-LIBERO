"""N=2 frozen rigid support points; no production convex migration claim."""
import argparse,functools,hashlib,json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from .mjwarp_convex_reference import support,check
STATIC=('geom_type','geom_dataid','mesh_vertadr','mesh_vertnum','mesh_graphadr','mesh_graph','mesh_polynum','mesh_polyadr','mesh_polyvertadr','mesh_polyvertnum','mesh_polyvert','mesh_polymapadr','mesh_polymapnum','mesh_polymap')
REQUESTS=(('world','world','int32'),('geom','geom','int32'),('cached','cached','int32'),('direction','direction','vec3'),('request_margin','margin','float32'))
OUTPUTS=(('point','vec3'),('cached_out','int32'),('vertex_out','int32'))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def add_arguments(p):
    for key in ('build-root','inputs','output','library'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--cpu-reference',action='store_true');return p

def cases(args):
    rows=[]
    for p in sorted(args.inputs.glob('*.npz')):
        if digest(p)!=json.loads(p.with_suffix('.json').read_text())['sha256']:raise RuntimeError('Frozen support input changed')
        with np.load(p,allow_pickle=False) as f:c={k:f[k].copy() for k in f.files}
        c['ngeom']=int(c['ngeom']);c['name']=p.stem;rows.append(c)
    if not rows:raise ValueError('No frozen support inputs')
    return rows

def current_case(c,updated):
    c={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in c.items()}
    if updated:
        c['size']*=np.float32(.93);c['xpos']+=np.array([.013,-.017,.019],np.float32);c['xpos'][1]*=np.float32(.98);c['request_margin']*=np.float32(1.11);c['direction']=c['direction'][:,[2,0,1]].copy()
    return c

@functools.lru_cache(maxsize=1)
def reference_kernel():
    import warp as wp
    from mujoco_warp._src.collision_core import Geom
    from mujoco_warp._src.collision_gjk import support as raw_support
    @wp.kernel(module='unique',enable_backward=False)
    def kernel(geom_type:wp.array[int],geom_dataid:wp.array2d[int],size:wp.array2d[wp.vec3],vertadr:wp.array[int],vertnum:wp.array[int],graphadr:wp.array[int],vertices:wp.array[wp.vec3],graph:wp.array[int],xpos:wp.array2d[wp.vec3],xmat:wp.array2d[wp.mat33],world:wp.array[int],geom:wp.array[int],cached:wp.array[int],direction:wp.array[wp.vec3],margin:wp.array[float],point:wp.array[wp.vec3],cached_out:wp.array[int],vertex_out:wp.array[int]):
        i=wp.tid();w=world[i];g=geom[i];kind=geom_type[g];value=Geom();value.pos=xpos[w,g];value.rot=xmat[w,g];value.size=size[w%size.shape[0],g];value.index=cached[i];value.margin=margin[i]
        if kind==7:
            mesh=geom_dataid[w%geom_dataid.shape[0],g];value.vertadr=vertadr[mesh];value.vertnum=vertnum[mesh];value.graphadr=graphadr[mesh];value.vert=vertices;value.graph=graph
        result=raw_support(value,kind,direction[i]);point[i]=result.point;cached_out[i]=result.cached_index;vertex_out[i]=result.vertex_index
    return kernel


def construct(c,stream):
    import warp as wp
    arr=lambda value,kind:wp.array(np.ascontiguousarray(value),dtype=getattr(wp,kind),device=stream.device)
    m=NS(ngeom=c['ngeom']);d=NS(nworld=2,geom_xpos=arr(c['xpos'],'vec3'),geom_xmat=arr(c['xmat'],'mat33'))
    for k in STATIC:setattr(m,k,arr(c[k],'int32'))
    m.geom_size=arr(c['size'],'vec3')
    for k in ('mesh_vert','mesh_polynormal'):setattr(m,k,arr(c[k],'vec3'))
    requests=NS(**{field:arr(c[key],kind) for key,field,kind in REQUESTS});outputs=NS(**{key:arr(c['initial_'+key],kind) for key,kind in OUTPUTS});return m,d,requests,outputs


def upstream(m,d,r,o):
    import warp as wp
    if r.world.size:wp.launch(reference_kernel(),dim=r.world.size,block_dim=128,inputs=[m.geom_type,m.geom_dataid,m.geom_size,m.mesh_vertadr,m.mesh_vertnum,m.mesh_graphadr,m.mesh_vert,m.mesh_graph,d.geom_xpos,d.geom_xmat,r.world,r.geom,r.cached,r.direction,r.margin],outputs=[o.point,o.cached_out,o.vertex_out])


def assign(c,m,d,r,o):
    m.geom_size.assign(c['size']);d.geom_xpos.assign(c['xpos']);d.geom_xmat.assign(c['xmat'])
    for key,field,_ in REQUESTS:getattr(r,field).assign(c[key])
    for key,_ in OUTPUTS:getattr(o,key).assign(c['initial_'+key])


class Replay:
    def __init__(self,c,args,stream):
        import warp as wp
        from libero.gpu.host.runtime.collision.mjwarp_convex import SupportProbe
        self.case,self.stream=c,stream;self.owners={};self.graphs={}
        for mode in ('upstream','cuda'):
            owner=construct(c,stream);m,d,r,o=owner;self.owners[mode]=owner
            if mode=='cuda':self.binding=SupportProbe(m,d,r,o,args.library,stream)
            with wp.ScopedCapture(stream=stream) as cap:
                if mode=='cuda':self.binding.enqueue()
                else:upstream(*owner)
            self.graphs[mode]=cap.graph
    def check(self,updated):
        import warp as wp
        c=current_case(self.case,updated);refs=[support(c,i) for i in range(len(c['world']))];rows=[];outputs=[]
        for mode,(m,d,r,o) in self.owners.items():
            assign(c,m,d,r,o);readonly={**vars(m),**vars(d),**{'request.'+k:v for k,v in vars(r).items()}};readonly={k:v for k,v in readonly.items() if hasattr(v,'numpy')};before={k:v.numpy().view(np.uint8).copy() for k,v in readonly.items()}
            wp.capture_launch(self.graphs[mode],stream=self.stream);wp.synchronize_stream(self.stream);got={k:getattr(o,k).numpy() for k,_ in OUTPUTS};outputs.append(got);checks=[]
            for i,ref in enumerate(refs):checks += [dict(v,request=i) for v in check(c,i,got['point'][i],int(got['cached_out'][i]),int(got['vertex_out'][i]),ref,require_indexed_corner=mode=='cuda')]
            held=all(np.array_equal(v.numpy().view(np.uint8),before[k]) for k,v in readonly.items());rows.append(dict(mode=mode,passed=bool(held and all(v['passed'] for v in checks)),checks=checks,readonly_inputs_held=held))
        # Each point must solve the independent support problem. Flat faces may
        # choose distinct legal vertices; compare projected support, not bytes.
        errors=np.array([abs(np.dot((outputs[0]['point'][i]-outputs[1]['point'][i]).astype(float),ref['direction'])) for i,ref in enumerate(refs)]);budgets=np.array([2*ref['budget'] for ref in refs]);parity=bool(np.all(errors<=budgets))
        return dict(name=c['name'],updated=updated,modes=rows,projection_parity=parity,maximum_projection_difference=float(errors.max(initial=0)),passed=parity and all(row['passed'] for row in rows)),outputs


def input_identity(args):return {p.name:digest(p) for p in sorted(args.inputs.glob('*')) if p.is_file()}
def signature(args):
    from libero.gpu.host.backends.mjwarp.build import source_files
    files=source_files(include_convex=True);files={k:v for k,v in files.items() if k.startswith('upstream/') or any(n in k for n in ('mjwarp_convex','MjwarpConvex','mjwarp_contact.cuh','mjwarp_plane.py')) or k in ('host/runtime/dynamics/mjwarp_motion.py','project_source_identity','data/mjwarp_source_pin.json')}
    return dict(sources=files,inputs=input_identity(args),library_sha256=digest(args.library),tool_sha256=digest(__file__),reference_sha256=digest(Path(__file__).with_name('mjwarp_convex_reference.py')))


def cpu_reference(args):
    import mujoco
    if mujoco.__version__!='2.3.7':raise RuntimeError('Support CPU oracle requires actual CPU2.3.7')
    path=next(Path(mujoco.__file__).parent.glob('libmujoco.so*'));rows=[];skipped=0
    m=mujoco.MjModel.from_xml_string('<mujoco><size nconmax="128"/><worldbody><geom type="plane" size="1 1 .1"/><body><freejoint/><geom type="sphere" size=".2"/></body></worldbody></mujoco>');d=mujoco.MjData(m);mujoco.mj_kinematics(m,d);m.geom_rbound[1]=100;m.geom_aabb[1]=[0,0,0,100,100,100];m.geom_margin[:]=1.e6
    for original in cases(args):
        for updated in (False,True):
            c=current_case(original,updated)
            for i in range(len(c['world'])):
                ref=support(c,i);kind=ref['kind'];direction=ref['direction'];length=np.linalg.norm(direction)
                if kind==7 or length==0 or ref['margin']<0:skipped+=1;continue
                n=-direction/length;axis=ref['R'][:,2]
                if kind==5 and np.linalg.norm(axis*np.dot(n,axis)-n)**2<1.e-12:skipped+=1;continue
                b=np.eye(3)[int(np.argmin(abs(n)))];b-=n*np.dot(n,b);b/=np.linalg.norm(b);plane=np.stack((b,np.cross(n,b),n),axis=1)
                m.geom_type[1]=kind;m.geom_dataid[1]=-1;m.geom_size[1]=ref['size'];d.geom_xpos[0]=0;d.geom_xpos[1]=ref['pos'];d.geom_xmat[0]=plane.ravel();d.geom_xmat[1]=ref['R'].ravel();d.xpos[1]=ref['pos'];d.xmat[1]=ref['R'].ravel();mujoco.mj_collision(m,d)
                if d.ncon==0:rows.append(dict(name=c['name'],updated=updated,request=i,passed=False,reason='CPU plane oracle produced no support witness'));continue
                value=-min(float(d.contact[j].dist) for j in range(d.ncon))*length
                if ref['margin']>0 or kind==2:value+=.5*ref['margin']*length**2
                error=abs(value-ref['projection']);rows.append(dict(name=c['name'],updated=updated,request=i,passed=bool(error<=ref['budget']),maximum_error=error))
    result=dict(passed=bool(rows) and all(v['passed'] for v in rows),checks=rows,excluded_mesh_zero_direction_negative_margin_axial_cylinder=skipped,inputs=input_identity(args),tool_sha256=digest(__file__),reference_sha256=digest(Path(__file__).with_name('mjwarp_convex_reference.py')),cpu_library=str(path),cpu_library_sha256=digest(path),version=mujoco.__version__,physics_integrations=0,performance=None,domain='actual public CPU2.3.7 plane collision minimum projection for common primitive support; mesh and version-different/degenerate paths independently and pinned-GPU qualified')
    (args.output/'cpu-oracle.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=result['passed'],checks=len(rows),excluded=skipped)),flush=True);return 0 if result['passed'] else 2


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
        if not prep.is_file() or json.loads(prep.read_text())['signature']!=sig:raise RuntimeError('Prepare exact support probe offline first')
        cpu=json.loads((args.output/'cpu-oracle.json').read_text())
        if not cpu['passed'] or any(cpu[k]!=sig[k] for k in ('inputs','tool_sha256','reference_sha256')):raise RuntimeError('Prepare matching CPU support oracle first')
    with compilation_scope(args.prepare) as scope,wp.ScopedStream(stream):
        replays=[Replay(c,args,stream) for c in cases(args)]
        if args.prepare:
            wp.synchronize_stream(stream);prep.write_text(json.dumps(dict(signature=sig,compilation_events=scope['events'],physics_integrations=0,production_collision_replaced=False),indent=2)+'\n');print(json.dumps(dict(prepared=True,captures=2*len(replays),compilations=len(scope['events']))),flush=True);return 0
        rows=[]
        for replay in replays:
            for updated in (False,True):
                row,values=replay.check(updated);rows.append(row);print(json.dumps({k:row[k] for k in ('name','updated','passed')}),flush=True)
                if not row['passed']:np.savez(args.output/'failure.npz',**{mode+'.'+k:v for mode,got in zip(('upstream','cuda'),values) for k,v in got.items()});break
            if not row['passed']:break
        passed=all(row['passed'] for row in rows);(args.output/'summary.json').write_text(json.dumps(dict(passed=passed,signature=sig,cases=rows,runtime_compilations=scope['events'],worlds=2,physics_integrations=0,performance=None,production_collision_replaced=False),indent=2)+'\n');print(json.dumps(dict(passed=passed,cases=len(rows))),flush=True);return 0 if passed else 2
if __name__=='__main__':raise SystemExit(main())
