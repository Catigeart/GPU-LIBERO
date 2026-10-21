"""Frozen N=2 applied/contact wrench qualification; no solves or trajectories."""
import argparse,hashlib,json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from .mjwarp_external_wrench_reference import reference,decode
from .mjwarp_motion_reference import check_arrays

CONTACTS={'contact_dim':('dim','int'),'contact_geom':('geom','vec2i'),
    'contact_address':('efc_address','int'),'contact_world':('worldid','int'),
    'contact_pos':('pos','vec3'),'contact_frame':('frame','mat33'),
    'contact_friction':('friction','vec5'),'contact_adhesion':('adhesion','float32')}
DATA={'applied':('xfrc_applied','spatial_vector'),'xipos':('xipos','vec3'),
    'subtree_com':('subtree_com','vec3'),'external':('cfrc_ext','spatial_vector'),'contact_count':('nacon','int')}


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def helpers():return {'mjwarp_motion_reference.py':digest(Path(__file__).with_name('mjwarp_motion_reference.py'))}
def input_identity(args):return {p.name:digest(p) for p in sorted(args.inputs.glob('*')) if p.is_file()}


def add_arguments(p):
    for name in ('build-root','inputs','output','library'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--cpu-reference',action='store_true')
    return p


def cases(args):
    result=[]
    for p in sorted(args.inputs.glob('*.npz')):
        if digest(p)!=json.loads(p.with_suffix('.json').read_text())['sha256']:raise RuntimeError('Frozen external wrench input changed')
        with np.load(p,allow_pickle=False) as f:c={k:f[k].copy() for k in f.files}
        for k in ('nv','nbody','ngeom','naconmax','njmax','nmaxcondim','cone'):c[k]=int(c[k])
        c['name']=p.stem;result.append(c)
    if not result:raise ValueError('No frozen external wrench inputs')
    return result


def current_case(original,updated):
    c={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in original.items()}
    if updated:
        c['applied']*=np.array([.7,-1.1,.9,-.6,.8,1.3],np.float32)
        c['constraint_force']*=np.array([-.83,1.17],np.float32)[:,None]
        c['xipos']+=np.array([.013,-.007,.009],np.float32)
        c['subtree_com']+=np.array([-.003,.005,-.011],np.float32)
        n=min(int(c['contact_count'][0]),c['naconmax'])
        c['contact_pos'][:n]+=np.array([.017,-.011,.023],np.float32)
        c['contact_frame'][:n]=c['contact_frame'][:n,...,[1,2,0]].copy()
        c['contact_friction'][:n]*=np.float32(.93);c['contact_adhesion'][:n]*=np.float32(1.13)
        c['contact_world'][:n]=1-c['contact_world'][:n]
        # Active count is a current device input, not a captured host scalar.
        c['contact_count'][0]=(n+1)//2
        c['external']*=np.float32(-.71)
    return c


def cpu_reference(args):
    import mujoco
    if mujoco.__version__!='2.3.7':raise RuntimeError('External wrench CPU oracle requires CPU reference2.3.7')
    checks=[];outputs={}
    for original in cases(args):
        receipt=json.loads((args.inputs/(original['name']+'.json')).read_text())
        if 'model_path' in receipt:
            if digest(receipt['model_path'])!=receipt['model_sha256']:raise RuntimeError('CPU external wrench model changed')
            m=mujoco.MjModel.from_binary_path(receipt['model_path'])
        else:m=mujoco.MjModel.from_xml_string(receipt['xml'])
        assert m.neq==m.ntendon==0
        if not np.array_equal(m.body_rootid,original['root']) or not np.array_equal(m.geom_bodyid,original['geom_body']):raise RuntimeError('CPU external wrench topology differs')
        m.opt.cone=original['cone']
        for updated in (False,True):
            c=current_case(original,updated)
            # CPU2.3.7 has no contact adhesion. Qualify its actual mathematics
            # on the common zero-adhesion domain; pinned adhesion is checked
            # separately against the independent FP64 formula on the GPU.
            common=dict(c,contact_adhesion=np.zeros_like(c['contact_adhesion']))
            expected,_=reference(common);got=[]
            for w in range(2):
                d=mujoco.MjData(m);mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
                indices=[i for i in range(min(int(c['contact_count'][0]),c['naconmax'])) if c['contact_world'][i]==w]
                # Allocate actual CPU constraint storage without running a solver.
                # CPU uses contiguous rows. Repack equivalent force components
                # from MJWarp's per-component addresses, with clipped rows zero.
                dynamic=np.flatnonzero(m.body_dofnum[m.geom_bodyid]>0)
                for i in indices:
                    contact=mujoco.MjContact();contact.dim=6;contact.geom1=int(dynamic[0]);contact.geom2=int(dynamic[-1])
                    contact.dist=-.01;contact.frame[:]=np.eye(3).ravel();contact.friction[:]=1.
                    contact.solref[:]=[.02,1.];contact.solimp[:]=[.9,.95,.001,.5,2.]
                    if mujoco.mj_addContact(m,d,contact):raise RuntimeError('CPU oracle contact storage exhausted')
                mujoco.mj_makeConstraint(m,d);d.efc_force[:]=0.
                width=10 if c['cone']==0 else 6
                if d.nefc<len(indices)*width:raise RuntimeError('CPU oracle row allocation differs')
                for slot,i in enumerate(indices):
                    con=d.contact[slot];con.dim=int(c['contact_dim'][i]);con.geom1=int(c['contact_geom'][i,0]);con.geom2=int(c['contact_geom'][i,1])
                    con.pos[:]=c['contact_pos'][i];con.frame[:]=c['contact_frame'][i].ravel();con.friction[:]=c['contact_friction'][i]
                    adr=c['contact_address'][i];con.efc_address=slot*width if adr[0]>=0 else -1
                    if adr[0]>=0:
                        size=1 if con.dim==1 else 2*(con.dim-1) if c['cone']==0 else con.dim
                        for j in range(size):
                            source=int(adr[0])+j if c['cone']==0 else int(adr[j])
                            d.efc_force[slot*width+j]=float(c['constraint_force'][w,source]) if source<c['njmax'] else 0.
                    force=np.empty(6);mujoco.mj_contactForce(m,d,slot,force);target=decode(common,i,adhesion=False)[0]
                    error=float(abs(force-target).max());limit=128*np.finfo(float).eps*max(1.,float(abs(target).max()))
                    checks.append(dict(name=c['name'],updated=updated,world=w,phase='contact-decode',contact=i,error=error,limit=limit,passed=bool(error<=limit)))
                d.subtree_com[:]=c['subtree_com'][w];d.xipos[:]=c['xipos'][w];d.xfrc_applied[:]=c['applied'][w]
                mujoco.mj_rnePostConstraint(m,d);got.append(d.cfrc_ext.copy())
            got=np.array(got);target=expected['cfrc_ext'];error=float(abs(got-target).max())
            # FP64 summation in the actual CPU must agree up to its own
            # roundoff, independently of the looser FP32 GPU acceptance.
            scale=max(1.,float(abs(c['applied']).sum()),float(abs(c['constraint_force']).sum()))
            limit=8192*np.finfo(float).eps*scale
            checks.append(dict(name=c['name'],updated=updated,phase='composed',error=error,limit=limit,passed=bool(error<=limit)))
            outputs[c['name']+('.updated' if updated else '')]=got
    np.savez(args.output/'cpu-oracle.npz',**outputs)
    result=dict(passed=all(r['passed'] for r in checks),version=mujoco.__version__,checks=checks,inputs=input_identity(args),tool_sha256=digest(__file__),
        reference_sha256=digest(Path(__file__).with_name('mjwarp_external_wrench_reference.py')),helpers=helpers(),
        physics_integrations=0,newton_solves=0,domain='common zero adhesion; equivalent contiguous CPU force rows')
    (args.output/'cpu-oracle.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'],cpu_checks=len(checks))),flush=True);return 0 if result['passed'] else 2


def construct(c,stream):
    import warp as wp
    from mujoco_warp._src import types
    arr=lambda value,dtype:wp.array(np.ascontiguousarray(value),dtype=(types.vec5 if dtype=='vec5' else wp.int32 if dtype=='int' else getattr(wp,dtype)),device=stream.device)
    m=NS(nv=c['nv'],nbody=c['nbody'],ngeom=c['ngeom'],nmaxcondim=c['nmaxcondim'],neq=0,ntendon=0,
        nflex=0,nflexedge=0,nflexelem=0,nflexbend_interp=0,nflexintcell=0,
        opt=NS(cone=c['cone'],integrator=types.IntegratorType.EULER,enableflags=0))
    for key,field in (('parent','body_parentid'),('root','body_rootid'),('geom_body','geom_bodyid')):setattr(m,field,arr(c[key],'int'))
    d=NS(nworld=2,naconmax=c['naconmax'],njmax=c['njmax'],contact=NS(),efc=NS(force=arr(c['constraint_force'],'float32')))
    for key,(field,dtype) in DATA.items():setattr(d,field,arr(c[key],dtype))
    for key,(field,dtype) in CONTACTS.items():setattr(d.contact,field,arr(c[key],dtype))
    return m,d


def upstream_applied(m,d):
    import warp as wp
    from mujoco_warp._src import smooth
    wp.launch(smooth._cfrc_ext,dim=(d.nworld,m.nbody),inputs=[m.body_rootid,d.xfrc_applied,d.xipos,d.subtree_com],outputs=[d.cfrc_ext])


def upstream_contacts(m,d):
    import warp as wp
    from mujoco_warp._src import smooth
    wp.launch(smooth._cfrc_ext_contact,dim=d.naconmax,inputs=[m.opt.cone,m.body_rootid,m.geom_bodyid,d.subtree_com,d.contact.pos,d.contact.frame,
        d.contact.friction,d.contact.dim,d.contact.geom,d.contact.efc_address,d.contact.worldid,d.contact.adhesion,d.efc.force,d.njmax,d.nacon],outputs=[d.cfrc_ext])


def assign(c,m,d):
    for key,(field,_) in DATA.items():getattr(d,field).assign(c[key])
    for key,(field,_) in CONTACTS.items():getattr(d.contact,field).assign(c[key])
    d.efc.force.assign(c['constraint_force'])


class Replay:
    def __init__(self,c,args,stream,phase):
        import warp as wp
        from libero.gpu.host.runtime.dynamics.mjwarp_external_wrench import ExternalWrenchNumerics,qualify
        self.case,self.stream,self.phase=c,stream,phase;self.models={};self.datas={};self.graphs={}
        for mode in ('upstream','cuda'):
            m,d=construct(c,stream);self.models[mode]=m;self.datas[mode]=d
            if mode=='cuda':self.binding=ExternalWrenchNumerics(m,d,args.library,stream,admission=qualify(m,d,implementation='cuda'))
            with wp.ScopedCapture(stream=stream) as cap:
                if phase!='contact':
                    if mode=='cuda':self.binding.enqueue(0)
                    else:upstream_applied(m,d)
                if phase!='applied':
                    if mode=='cuda':self.binding.enqueue(1)
                    else:upstream_contacts(m,d)
            self.graphs[mode]=cap.graph

    def check(self,updated):
        import warp as wp
        c=current_case(self.case,updated);expected,errors=reference(c,self.phase);checks=[];held=True;values={}
        for mode in ('upstream','cuda'):
            m,d=self.models[mode],self.datas[mode];assign(c,m,d)
            readonly={**{k:v for k,v in vars(m).items() if hasattr(v,'numpy')},
                **{k:getattr(d,field) for k,(field,_) in DATA.items() if k!='external'},
                **{k:getattr(d.contact,field) for k,(field,_) in CONTACTS.items()},'constraint_force':d.efc.force}
            before={key:a.numpy().view(np.uint8).copy() for key,a in readonly.items()}
            wp.capture_launch(self.graphs[mode],stream=self.stream);wp.synchronize_stream(self.stream)
            values[mode]={'cfrc_ext':d.cfrc_ext.numpy()};checks += [dict(mode=mode,**x) for x in check_arrays(values[mode],expected,errors)]
            held &= all(np.array_equal(a.numpy().view(np.uint8),before[key]) for key,a in readonly.items())
            if self.phase=='contact':held &= np.array_equal(values[mode]['cfrc_ext'][:,0].view(np.uint8),c['external'][:,0].view(np.uint8))
            else:held &= np.array_equal(values[mode]['cfrc_ext'][:,0],np.zeros((2,6),np.float32))
        return dict(name=c['name'],phase=self.phase,updated=updated,inputs_and_holds=bool(held),checks=checks,
            maximum_difference=float(abs(values['cuda']['cfrc_ext']-values['upstream']['cfrc_ext']).max()),passed=bool(held and all(x['passed'] for x in checks)))


def signature(args):
    from libero.gpu.host.backends.mjwarp.build import source_files
    sources=source_files(include_external_wrench=True)
    sources={k:v for k,v in sources.items() if k.startswith('upstream/') or 'external_wrench' in k or 'ExternalWrench' in k or k in ('host/runtime/dynamics/mjwarp_motion.py','data/mjwarp_source_pin.json','project_source_identity')}
    return dict(sources=sources,inputs=input_identity(args),library_sha256=digest(args.library),tool_sha256=digest(__file__),
        reference_sha256=digest(Path(__file__).with_name('mjwarp_external_wrench_reference.py')),helpers=helpers())


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
    from validation.common import save_json
    require_upstream();init_kernel_cache(args.build_root/'mjwarp/warp-cache');install_warp_guard();wp.init();stream=wp.Stream('cuda:0');sig=signature(args);prep=args.output/'preparation.json'
    if not args.prepare:
        if not prep.is_file() or json.loads(prep.read_text())['signature']!=sig:raise RuntimeError('Prepare exact external wrench operators offline first')
        cpu=json.loads((args.output/'cpu-oracle.json').read_text())
        if not cpu['passed'] or any(cpu[k]!=sig[k] for k in ('inputs','tool_sha256','reference_sha256','helpers')):raise RuntimeError('Prepare matching CPU external wrench oracle first')
    with compilation_scope(args.prepare) as scope,wp.ScopedStream(stream):
        replays=[Replay(c,args,stream,p) for p in ('applied','contact','composed') for c in cases(args)]
        if args.prepare:
            wp.synchronize_stream(stream);save_json(prep,dict(signature=sig,compilation_events=scope['events'],physics_integrations=0,newton_solves=0))
            print(json.dumps(dict(prepared=True,captures=2*len(replays),compilations=len(scope['events']))),flush=True);return 0
        rows=[]
        for replay in replays:
            for updated in (False,True):
                row=replay.check(updated);rows.append(row);print(json.dumps({k:row[k] for k in ('name','phase','updated','passed')}),flush=True)
                if not row['passed']:break
            if not row['passed']:break
        passed=all(x['passed'] for x in rows)
        save_json(args.output/'summary.json',dict(passed=passed,signature=sig,cases=rows,runtime_compilations=scope['events'],physics_integrations=0,newton_solves=0,performance=None,worlds=2,
            policy='independent FP64 decode/rotation/translation; componentwise arithmetic bounds plus arbitrary-order atomic absolute-contribution bound; fixed8x safety; exact readonly/hold contracts'))
        print(json.dumps(dict(passed=passed,cases=len(rows))),flush=True);return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())
