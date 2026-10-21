"""Frozen N=2 box contact manifolds: pinned comparison and independent geometry."""
import argparse,json
from pathlib import Path
import numpy as np
from .mjwarp_plane import PARAMETERS,CONTACTS,STATIC,MESH,digest,add_arguments,cases as plane_cases,current_case,construct as plane_construct,assign
from .mjwarp_box_reference import pair_math,geometry_checks,materials


def cases(args):
    rows=plane_cases(args)
    for c in rows:c['boxbox']=int(c['boxbox'])
    return rows

def construct(c,stream):
    m,d,ctx=plane_construct(c,stream)
    from mujoco_warp._src import types
    m.geom_pair_type_count=(1,)*(len(types.GeomType)*(len(types.GeomType)+1)//2)
    m.has_boxbox_margin=bool(c['boxbox']);m.opt.disableflags=0;return m,d,ctx

def upstream(m,d,ctx):
    from mujoco_warp._src import collision_primitive,types
    table=[(types.GeomType.SPHERE,types.GeomType.BOX),(types.GeomType.CAPSULE,types.GeomType.BOX)]
    if m.has_boxbox_margin:table.append((types.GeomType.BOX,types.GeomType.BOX))
    collision_primitive.primitive_narrowphase(m,d,ctx,table)

def pair_references(c):
    rows={}
    for i in range(min(c['capacity'],int(c['candidate_count'][0]))):
        g1,g2=map(int,c['candidate_geom'][i]);w=int(c['candidate_world'][i]);t1,t2=c['geom_type'][[g1,g2]]
        if t2!=6 or not (t1 in (2,3) or (t1==6 and c['boxbox'])):continue
        info=pair_math(c,w,g1,g2);pair,sensor=map(int,c['candidate_ids'][i]);p=materials(c,w,g1,g2,pair);info.update(material=p,pair=pair,sensor=sensor)
        boundary=info['separation'] if t1==6 else info['closest']-float(p['margin']+p['gap']);info['near_boundary']=abs(boundary)<=info['budget'];rows[(w,g1,g2)]=info
    return rows

def check_output(c,got,refs):
    n=int(got['count'][0]);initial=int(c['initial_count'][0]);begin=min(initial,c['capacity']);end=min(n,c['capacity']);keys=[];checks=[];diagnostics=[];holds=True;seen={}
    for field,_ in CONTACTS:
        holds &= np.array_equal(got[field][:begin].view(np.uint8),c['initial_'+field][:begin].view(np.uint8)) and np.array_equal(got[field][end:].view(np.uint8),c['initial_'+field][end:].view(np.uint8))
    for i in range(begin,end):
        key=(int(got['worldid'][i]),*map(int,got['geom'][i]),int(got['geomcollisionid'][i]));keys.append(key);pairkey=key[:3]
        if pairkey not in refs:checks.append(dict(name='contact belongs to selected candidate family',passed=False));continue
        info=refs[pairkey];seen[pairkey]=seen.get(pairkey,0)+1;p=info['material'];dist=float(got['dist'][i]);active=dist<float(p['margin']+p['gap']);adhesive=p['adhesion']!=0;kind=(1 if info['pair']>=-1 and active else 0)|(2 if info['sensor']>=0 else 0);dim=1 if adhesive and dist>=p['margin'] else p['dim'];budget=info['budget']
        checks.extend(geometry_checks(info,dist,got['pos'][i],got['frame'][i],key[-1]))
        checks.extend([dict(name='stored contact is detected or has collision sensor',passed=bool((info['pair']!=-2 and active) or info['sensor']>=0)),dict(name='contact kind/adhesion dim/address exact',passed=bool(int(got['type'][i])==kind and int(got['dim'][i])==dim and np.all(got['efc_address'][i]==-1)))])
        for field in ('includemargin','friction','solref','solreffriction','solimp','adhesion'):
            ref=np.asarray(p['margin'] if field=='includemargin' else p[field]);error=float(np.max(abs(got[field][i].astype(float)-ref)));checks.append(dict(name='material '+field,passed=bool(np.all(abs(got[field][i].astype(float)-ref)<=budget*(1+abs(ref)))),maximum_error=error))
    if n<=c['capacity']:
        for key,info in refs.items():
            p=info['material'];certain=info['separation']<-info['budget'] if info['kind']==6 else info['closest']<float(p['margin']+p['gap'])-info['budget']
            if certain and info['pair']!=-2:
                check=dict(name='definite overlap has contact',key=key,passed=seen.get(key,0)>0)
                if info['kind']==3 and info['inside']:
                    check['upstream_limitation']='centerline crossing a box face interior is not considered by the legacy endpoint/edge search'
                    diagnostics.append(check)
                else:checks.append(check)
    checks.extend([dict(name='attempt count monotone',passed=n>=initial),dict(name='unique world/geom/manifold ownership',passed=len(keys)==len(set(keys))),dict(name='prefix and unwritten bit holds',passed=bool(holds))])
    return dict(passed=all(v['passed'] for v in checks),count=n,checks=checks,diagnostics=diagnostics,keys=keys)


def compare(c,outputs,refs):
    rows=[];a,b=outputs;ia=min(int(c['initial_count'][0]),c['capacity'])
    def map_rows(g):return {(int(g['worldid'][i]),*map(int,g['geom'][i]),int(g['geomcollisionid'][i])):i for i in range(ia,min(int(g['count'][0]),c['capacity']))}
    x,y=map_rows(a),map_rows(b);overflow=any(int(z['count'][0])>c['capacity'] for z in outputs)
    if not overflow:
        difference=set(x)^set(y);rows.append(dict(name='same manifold keys outside activation boundary',passed=all(refs[key[:3]]['near_boundary'] for key in difference),different_keys=len(difference)))
    else:rows.append(dict(name='overflow attempted count comparison',passed=int(a['count'][0])==int(b['count'][0])))
    for key in x.keys()&y.keys():
        info=refs[key[:3]];budget=info['budget'];i,j=x[key],y[key]
        for field,_ in CONTACTS:
            va,vb=a[field][i],b[field][j];error=abs(va.astype(float)-vb);exact=va.dtype.kind in 'iu';allowed=0 if exact else budget*(1+np.abs(va.astype(float)))
            rows.append(dict(name='pinned contact '+field,key=key,passed=bool(np.all(error<=allowed)),maximum_error=float(np.max(error))))
    return dict(passed=all(v['passed'] for v in rows),checks=rows)


class Replay:
    def __init__(self,c,args,stream):
        import warp as wp
        from libero.gpu.host.runtime.collision.mjwarp_box import BoxContacts,qualify
        self.case,self.stream=c,stream;self.owners={};self.graphs={}
        for mode in ('upstream','cuda'):
            m,d,ctx=construct(c,stream);self.owners[mode]=(m,d,ctx)
            if mode=='cuda':self.binding=BoxContacts(m,d,ctx,args.library,stream,admission=qualify(m,d,implementation='cuda'))
            with wp.ScopedCapture(stream=stream) as cap:
                if mode=='cuda':self.binding.enqueue()
                else:upstream(m,d,ctx)
            self.graphs[mode]=cap.graph
    def check(self,updated):
        import warp as wp
        c=current_case(self.case,updated);refs=pair_references(c);rows=[];outputs=[]
        for mode,(m,d,ctx) in self.owners.items():
            assign(c,m,d,ctx);readonly={key:getattr(m,field) for key,field,_ in PARAMETERS};readonly.update({key:getattr(m,{'condim':'geom_condim','priority':'geom_priority'}.get(key,key)) for key in STATIC});readonly.update(xpos=d.geom_xpos,xmat=d.geom_xmat,candidate_count=d.ncollision,**vars(ctx))
            before={key:a.numpy().view(np.uint8).copy() for key,a in readonly.items()};wp.capture_launch(self.graphs[mode],stream=self.stream);wp.synchronize_stream(self.stream);got={key:getattr(d.contact,key).numpy() for key,_ in CONTACTS};got['count']=d.nacon.numpy();outputs.append(got)
            row=check_output(c,got,refs);row.pop('keys');held=all(np.array_equal(a.numpy().view(np.uint8),before[key]) for key,a in readonly.items());row.update(mode=mode,readonly_inputs_held=held);row['passed'] &=held;rows.append(row)
        parity=compare(c,outputs,refs);return dict(name=c['name'],updated=updated,modes=rows,comparison=parity,passed=parity['passed'] and all(v['passed'] for v in rows)),outputs


def helpers():return {n:digest(Path(__file__).with_name(n)) for n in ('mjwarp_plane.py','mjwarp_plane_reference.py')}
def input_identity(args):return {p.name:digest(p) for p in sorted(args.inputs.glob('*')) if p.is_file()}
def signature(args):
    from libero.gpu.host.backends.mjwarp.build import source_files
    files=source_files(include_box=True);files={k:v for k,v in files.items() if k.startswith('upstream/') or any(n in k for n in ('mjwarp_box','MjwarpBox','mjwarp_contact.cuh','mjwarp_plane.py')) or k in ('host/runtime/dynamics/mjwarp_motion.py','project_source_identity','data/mjwarp_source_pin.json')}
    return dict(sources=files,inputs=input_identity(args),library_sha256=digest(args.library),tool_sha256=digest(__file__),reference_sha256=digest(Path(__file__).with_name('mjwarp_box_reference.py')),helpers=helpers())


def cpu_reference(args):
    import mujoco
    if mujoco.__version__!='2.3.7':raise RuntimeError('Box CPU oracle requires actual CPU2.3.7')
    path=next(Path(mujoco.__file__).parent.glob('libmujoco.so*'));rows=[];skipped=0
    m=mujoco.MjModel.from_xml_string('<mujoco><size nconmax="128"/><worldbody><body><freejoint/><geom type="sphere" size=".2"/></body><body><freejoint/><geom type="box" size=".2 .2 .2"/></body></worldbody></mujoco>');d=mujoco.MjData(m);mujoco.mj_kinematics(m,d);m.geom_rbound[:]=100;m.geom_aabb[:]=[0,0,0,100,100,100]
    for original in cases(args):
        for updated in (False,True):
            c=current_case(original,updated)
            for key,info in pair_references(c).items():
                w,g1,g2=key;p=info['material'];kind=info['kind'];m.geom_type[:]=[kind,6];m.geom_size[:]=[info['size1'],info['size2']];d.geom_xpos[:]=[info['p1'],info['p2']];d.geom_xmat[:]=[info['R1'].ravel(),info['R2'].ravel()];d.xpos[1:3]=d.geom_xpos;d.xmat[1:3]=d.geom_xmat
                if kind==6:
                    m.geom_margin[:]=0;mujoco.mj_collision(m,d);ok=d.ncon>0 if info['separation']<-info['budget'] else True;rows.append(dict(name=c['name'],updated=updated,kind=kind,passed=bool(ok),domain='definite SAT overlap detection'));continue
                if info['inside'] or np.min(info['size1'][:2])==0:skipped+=1;continue
                m.geom_margin[:]=100;mujoco.mj_collision(m,d)
                for j in range(d.ncon):
                    check=geometry_checks(info,d.contact[j].dist,d.contact[j].pos,d.contact[j].frame.reshape(3,3),j);rows.extend(dict(v,name=c['name'],updated=updated,kind=kind) for v in check)
    result=dict(passed=bool(rows) and all(v['passed'] for v in rows),checks=rows,skipped_inside_or_zero_size=skipped,inputs=input_identity(args),tool_sha256=digest(__file__),reference_sha256=digest(Path(__file__).with_name('mjwarp_box_reference.py')),helpers=helpers(),cpu_library=str(path),cpu_library_sha256=digest(path),version=mujoco.__version__,physics_integrations=0,performance=None,domain='CPU2.3.7 mj_collision sphere/capsule outside-box distances/witnesses and box-box definite SAT overlap; old inside distance/manifold differences excluded')
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
        if not prep.is_file() or json.loads(prep.read_text())['signature']!=sig:raise RuntimeError('Prepare exact box operators offline first')
        cpu=json.loads((args.output/'cpu-oracle.json').read_text())
        if not cpu['passed'] or any(cpu[k]!=sig[k] for k in ('inputs','tool_sha256','reference_sha256','helpers')):raise RuntimeError('Prepare matching CPU box oracle first')
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
        passed=all(v['passed'] for v in rows);(args.output/'summary.json').write_text(json.dumps(dict(passed=passed,signature=sig,cases=rows,runtime_compilations=scope['events'],worlds=2,physics_integrations=0,performance=None),indent=2)+'\n');print(json.dumps(dict(passed=passed,cases=len(rows))),flush=True);return 0 if passed else 2
if __name__=='__main__':raise SystemExit(main())
