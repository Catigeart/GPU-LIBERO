"""N=2 frozen plane geometry/material/contact ownership, without integration."""
import argparse,ctypes as ct,hashlib,json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from .mjwarp_plane_reference import reference,manifold,material
PARAMETERS=(('size','geom_size','vec3'),('solmix','geom_solmix','float32'),('solref','geom_solref','vec2'),('solimp','geom_solimp','vec5'),('friction','geom_friction','vec3'),('margin','geom_margin','float32'),('gap','geom_gap','float32'),('adhesion','geom_adhesion','float32'),('pair_solref','pair_solref','vec2'),('pair_solreffriction','pair_solreffriction','vec2'),('pair_solimp','pair_solimp','vec5'),('pair_friction','pair_friction','vec5'),('pair_margin','pair_margin','float32'),('pair_gap','pair_gap','float32'),('pair_adhesion','pair_adhesion','float32'))
CONTACTS=(('dist','float32'),('pos','vec3'),('frame','mat33'),('includemargin','float32'),('friction','vec5'),('solref','vec2'),('solreffriction','vec2'),('solimp','vec5'),('adhesion','float32'),('dim','int32'),('geom','vec2i'),('worldid','int32'),('type','int32'),('geomcollisionid','int32'),('efc_address','int32'))
STATIC=('geom_type','condim','priority','pair_dim','geom_dataid','mesh_vertadr','mesh_vertnum','mesh_graphadr','mesh_graph','mesh_polynum','mesh_polyadr','mesh_polyvertadr','mesh_polyvertnum','mesh_polyvert','mesh_polymapadr','mesh_polymapnum','mesh_polymap')
MESH=STATIC[5:]+('mesh_vert','mesh_polynormal')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def add_arguments(p):
    for name in ('build-root','inputs','output','library'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--cpu-reference',action='store_true');return p

def cases(args):
    out=[]
    for p in sorted(args.inputs.glob('*.npz')):
        if digest(p)!=json.loads(p.with_suffix('.json').read_text())['sha256']:raise RuntimeError('Plane frozen input changed')
        with np.load(p,allow_pickle=False) as f:c={k:f[k].copy() for k in f.files}
        for k in ('ngeom','npair','capacity','columns'):c[k]=int(c[k])
        c['name']=p.stem;out.append(c)
    if not out:raise ValueError('No frozen plane inputs')
    return out

def current_case(c,updated):
    c={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in c.items()}
    if updated:
        for key,*_ in PARAMETERS:c[key]*=np.float32(.97)
        c['xpos']+=np.float32(.006);c['xpos'][1]*=np.float32(.98)
    return c

def construct(c,stream):
    import warp as wp
    from mujoco_warp._src import types
    def arr(value,kind):return wp.array(np.ascontiguousarray(value),dtype=getattr(wp,kind,None) or getattr(types,kind),device=stream.device)
    m=NS(ngeom=c['ngeom'],npair=c['npair'],nflex=0,opt=NS(integrator=types.IntegratorType.EULER,enableflags=0),geom_pair_type_count=(1,)*45)
    for key,field,kind in PARAMETERS:setattr(m,field,arr(c[key],kind))
    for key in STATIC:setattr(m,{'condim':'geom_condim','priority':'geom_priority'}.get(key,key),arr(c[key],'int32'))
    for key in ('mesh_vert','mesh_polynormal'):setattr(m,key,arr(c[key],'vec3'))
    d=NS(nworld=2,naconmax=c['capacity'],geom_xpos=arr(c['xpos'],'vec3'),geom_xmat=arr(c['xmat'],'mat33'),ncollision=arr(c['candidate_count'],'int32'),nacon=arr(c['initial_count'],'int32'),contact=NS())
    for key,kind in CONTACTS:setattr(d.contact,key,arr(c['initial_'+key],kind))
    ctx=NS(collision_pair=arr(c['candidate_geom'],'vec2i'),collision_pairid=arr(c['candidate_ids'],'vec2i'),collision_worldid=arr(c['candidate_world'],'int32'))
    return m,d,ctx

def assign(c,m,d,ctx):
    for key,field,_ in PARAMETERS:getattr(m,field).assign(c[key])
    for key,field in (('xpos','geom_xpos'),('xmat','geom_xmat'),('candidate_count','ncollision'),('initial_count','nacon')):getattr(d,field).assign(c[key])
    for key,_ in CONTACTS:getattr(d.contact,key).assign(c['initial_'+key])

def upstream(m,d,ctx):
    from mujoco_warp._src import collision_primitive,types
    table=[(types.GeomType.PLANE,t) for t in (types.GeomType.SPHERE,types.GeomType.CAPSULE,types.GeomType.ELLIPSOID,types.GeomType.CYLINDER,types.GeomType.BOX,types.GeomType.MESH)]
    collision_primitive.primitive_narrowphase(m,d,ctx,table)

def check_output(c,got,expected):
    n=int(got['count'][0]);initial=expected['initial'];cap=c['capacity'];begin=min(initial,cap);end=min(n,cap);checks=[];contracts=[]
    keys=[(int(got['worldid'][i]),*map(int,got['geom'][i]),int(got['geomcollisionid'][i])) for i in range(begin,end)];actual=set(keys)
    contracts.append(dict(name='unique contact/world/geom/manifold ownership',passed=len(keys)==len(actual) and actual<=expected['possible']))
    contracts.append(dict(name='attempted count within independent activation interval',passed=initial+len(expected['must'])<=n<=initial+len(expected['possible'])))
    contracts.append(dict(name='all certain contacts emitted without overflow',passed=n>cap or expected['must']<=actual))
    holds=True
    for field,_ in CONTACTS:
        value=got[field];holds &= np.array_equal(value[:begin].view(np.uint8),c['initial_'+field][:begin].view(np.uint8)) and np.array_equal(value[end:].view(np.uint8),c['initial_'+field][end:].view(np.uint8))
    contracts.append(dict(name='prefix and unwritten tails exact bit holds',passed=bool(holds)))
    for i,key in enumerate(keys,begin):
        if key not in expected['rows']:continue
        row,budget,uncertain=expected['rows'][key]
        for field,_ in CONTACTS:
            value=np.asarray(got[field][i]);ref=np.asarray(row[field]);err=abs(value.astype(float)-ref);exact=value.dtype.kind in 'iu';allowed=0. if exact else budget+8*(256*np.finfo(np.float32).eps/2)/(1-256*np.finfo(np.float32).eps/2)*abs(ref)
            ok=bool(np.all(err<=allowed))
            if field=='type' and uncertain:ok=bool(int(value) in (int(ref),int(ref)^1))
            if field=='dim' and uncertain and row['adhesion']!=0:ok=bool(int(value) in (1,row['dim']))
            checks.append(dict(key=key,field=field,passed=ok,maximum_error=float(np.max(err)),maximum_budget_fraction=float(np.max(err/(np.asarray(allowed)+np.finfo(float).tiny))) if not exact else None))
    return dict(passed=all(x['passed'] for x in checks+contracts),count=n,checks=checks,contracts=contracts)

class Replay:
    def __init__(self,c,args,stream):
        import warp as wp
        from libero.gpu.host.runtime.collision.mjwarp_plane import PlaneContacts,qualify
        self.case,self.stream=c,stream;self.owners={};self.graphs={}
        for mode in ('upstream','cuda'):
            m,d,ctx=construct(c,stream);self.owners[mode]=(m,d,ctx)
            if mode=='cuda':self.binding=PlaneContacts(m,d,ctx,args.library,stream,admission=qualify(m,d,implementation='cuda'))
            with wp.ScopedCapture(stream=stream) as cap:
                if mode=='cuda':self.binding.enqueue()
                else:upstream(m,d,ctx)
            self.graphs[mode]=cap.graph
    def check(self,updated):
        import warp as wp
        c=current_case(self.case,updated);expected=reference(c);rows=[]
        for mode,(m,d,ctx) in self.owners.items():
            assign(c,m,d,ctx);readonly={key:getattr(m,field) for key,field,_ in PARAMETERS};readonly.update({key:getattr(m,{'condim':'geom_condim','priority':'geom_priority'}.get(key,key)) for key in STATIC});readonly.update({key:getattr(m,key) for key in ('mesh_vert','mesh_polynormal')});readonly.update(xpos=d.geom_xpos,xmat=d.geom_xmat,candidate_count=d.ncollision,**vars(ctx))
            before={key:value.numpy().view(np.uint8).copy() for key,value in readonly.items()};wp.capture_launch(self.graphs[mode],stream=self.stream);wp.synchronize_stream(self.stream)
            got={key:getattr(d.contact,key).numpy() for key,_ in CONTACTS};got['count']=d.nacon.numpy();row=check_output(c,got,expected);held=all(np.array_equal(a.numpy().view(np.uint8),before[k]) for k,a in readonly.items());row.update(mode=mode,readonly_inputs_held=held);row['passed'] &=held;rows.append(row)
        return dict(name=c['name'],updated=updated,modes=rows,passed=all(x['passed'] for x in rows))

def signature(args):
    from libero.gpu.host.backends.mjwarp.build import source_files
    sources=source_files(include_plane=True);sources={k:v for k,v in sources.items() if k.startswith('upstream/') or 'plane' in k or 'Plane' in k or 'mjwarp_contact.cuh' in k or k in ('host/runtime/dynamics/mjwarp_motion.py','data/mjwarp_source_pin.json','project_source_identity')}
    return dict(sources=sources,inputs={p.name:digest(p) for p in sorted(args.inputs.glob('*')) if p.is_file()},library_sha256=digest(args.library),tool_sha256=digest(__file__),reference_sha256=digest(Path(__file__).with_name('mjwarp_plane_reference.py')))

def cpu_reference(args):
    import mujoco
    if mujoco.__version__!='2.3.7':raise RuntimeError('Plane CPU oracle requires actual CPU2.3.7')
    path=next(Path(mujoco.__file__).parent.glob('libmujoco.so*'));lib=ct.CDLL(str(path));rows=[];skipped=0
    m=mujoco.MjModel.from_xml_string('<mujoco><size nconmax="128"/><worldbody><geom type="plane" size="1 1 .1"/><body pos="0 0 .1"><freejoint/><geom type="sphere" size=".2"/></body></worldbody></mujoco>');d=mujoco.MjData(m);mujoco.mj_kinematics(m,d);mujoco.mj_collision(m,d);m.geom_margin[:]=100.;m.geom_rbound[1]=100.;m.geom_aabb[1]=[0,0,0,100,100,100]
    for original in cases(args):
        for updated in (False,True):
            c=current_case(original,updated)
            for i in range(min(c['capacity'],int(c['candidate_count'][0]))):
                g1,g2=map(int,c['candidate_geom'][i]);w=int(c['candidate_world'][i]);kind=int(c['geom_type'][g2])
                if c['geom_type'][g1]!=0 or kind not in (2,3,4,5,6):continue
                n=c['xmat'][w,g1,:,2].astype(float);axis=c['xmat'][w,g2,:,2].astype(float)
                if kind==5 and np.linalg.norm(axis*np.dot(n,axis)-n)**2<1.e-12:skipped+=1;continue
                m.geom_type[:]=[0,kind];m.geom_dataid[:]=-1;m.geom_size[1]=c['size'][w%len(c['size']),g2];d.geom_xpos[:]=c['xpos'][w,[g1,g2]];d.geom_xmat[:]=c['xmat'][w,[g1,g2]].reshape(2,9)
                d.xpos[1]=d.geom_xpos[1];d.xmat[1]=d.geom_xmat[1];mujoco.mj_collision(m,d);count=d.ncon
                p=material(c,w,g1,g2,int(c['candidate_ids'][i,0]));ref=manifold(c,w,g1,g2,p)
                for j in range(count):
                    values=np.r_[d.contact[j].dist,d.contact[j].pos].copy();error=min(float(np.max(abs(values-np.r_[dist,pos]))) for dist,pos,_ in ref);scale=1.+max(abs(values));budget=4096*np.finfo(float).eps*scale;rows.append(dict(name=c['name'],updated=updated,kind=kind,passed=bool(error<=budget),maximum_error=error))
    result=dict(passed=bool(rows) and all(x['passed'] for x in rows),checks=rows,skipped_degenerate_cylinders=skipped,version=mujoco.__version__,cpu_library=str(path),cpu_library_sha256=digest(path),inputs={p.name:digest(p) for p in sorted(args.inputs.glob('*')) if p.is_file()},tool_sha256=digest(__file__),reference_sha256=digest(Path(__file__).with_name('mjwarp_plane_reference.py')),physics_integrations=0,performance=None,domain='actual CPU2.3.7 mj_collision plane-sphere/capsule/ellipsoid/cylinder/box dist+pos; mesh manifold and degenerate cylinder version differences checked independently and vs pinned GPU')
    (args.output/'cpu-oracle.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=result['passed'],checks=len(rows),skipped=skipped)),flush=True);return 0 if result['passed'] else 2

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
        if not prep.is_file() or json.loads(prep.read_text())['signature']!=sig:raise RuntimeError('Prepare exact plane operators offline first')
        cpu=json.loads((args.output/'cpu-oracle.json').read_text())
        if not cpu['passed'] or any(cpu[k]!=sig[k] for k in ('inputs','tool_sha256','reference_sha256')):raise RuntimeError('Prepare matching CPU plane oracle first')
    with compilation_scope(args.prepare) as scope,wp.ScopedStream(stream):
        replays=[Replay(c,args,stream) for c in cases(args)]
        if args.prepare:
            wp.synchronize_stream(stream);prep.write_text(json.dumps(dict(signature=sig,compilation_events=scope['events'],physics_integrations=0),indent=2)+'\n');print(json.dumps(dict(prepared=True,captures=2*len(replays),compilations=len(scope['events']))),flush=True);return 0
        rows=[]
        for replay in replays:
            for updated in (False,True):
                row=replay.check(updated);rows.append(row);print(json.dumps({k:row[k] for k in ('name','updated','passed')}),flush=True)
                if not row['passed']:break
            if not row['passed']:break
        passed=all(x['passed'] for x in rows);(args.output/'summary.json').write_text(json.dumps(dict(passed=passed,signature=sig,cases=rows,runtime_compilations=scope['events'],worlds=2,physics_integrations=0,performance=None),indent=2)+'\n');print(json.dumps(dict(passed=passed,cases=len(rows))),flush=True);return 0 if passed else 2
if __name__=='__main__':raise SystemExit(main())
