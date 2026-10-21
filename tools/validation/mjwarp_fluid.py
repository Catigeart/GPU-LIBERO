"""Frozen N=2 fluid wrench/projection/passive checks; no timing or dynamics."""
import argparse,hashlib,json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from .mjwarp_fluid_reference import reference
from .mjwarp_motion_reference import check_arrays
from .mjwarp_passive import construct as passive_construct
PARAMS=('body_mass','body_inertia','wind','density','viscosity')
FRAMES=('xipos','ximat','subtree_com','cvel','cdof')
PASSIVE_FIELDS=('qfrc_spring','qfrc_damper','qfrc_gravcomp','qfrc_fluid','qfrc_adhesion','qfrc_passive')


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def helpers():return {n:digest(Path(__file__).with_name(n)) for n in ('mjwarp_motion_reference.py','mjwarp_passive_reference.py','mjwarp_passive.py')}


def add_arguments(p):
    for name in ('build-root','inputs','output','library'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--passive-library',type=Path)
    p.add_argument('--prepare',action='store_true');p.add_argument('--cpu-reference',action='store_true')
    return p


def cases(args):
    result=[]
    for p in sorted(args.inputs.glob('*.npz')):
        if digest(p)!=json.loads(p.with_suffix('.json').read_text())['sha256']:raise RuntimeError('Frozen fluid input changed')
        with np.load(p,allow_pickle=False) as f:c={k:f[k].copy() for k in f.files}
        for k in ('nq','nv','njnt','nbody','ngeom'):c[k]=int(c[k])
        for k in ('disable_spring','disable_damper','gravity_enabled','has_fluid','has_adhesion'):c[k]=bool(c[k])
        c['name']=p.stem;result.append(c)
    if not result:raise ValueError('No frozen fluid inputs')
    return result


def current_case(original,updated):
    c={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in original.items()}
    if updated:
        c['body_mass']*=np.float32(1.07);c['body_inertia']*=np.float32(.91)
        c['density']*=np.float32(.83);c['viscosity']*=np.float32(1.13)
        c['wind']=c['wind'][...,[2,0,1]].copy()*np.float32(.89)
        c['xipos']+=np.array([.013,-.007,.009],np.float32);c['subtree_com']+=np.array([-.003,.005,-.011],np.float32)
        c['ximat']=c['ximat'][...,[1,2,0]].copy();c['cvel']*=np.float32(.97);c['cdof']*=np.float32(1.03)
        c['body_force']*=np.array([.7,-1.1,.9,-.6,.8,1.3],np.float32)
        c['qvel']*=np.float32(.93);c['damping']*=np.float32(1.17)
    return c


def input_identity(args):return {p.name:digest(p) for p in sorted(args.inputs.glob('*')) if p.is_file()}


def cpu_reference(args):
    import mujoco
    if mujoco.__version__!='2.3.7':raise RuntimeError('Fluid CPU oracle requires CPU reference2.3.7')
    rows=[]
    for original in cases(args):
        receipt=json.loads((args.inputs/(original['name']+'.json')).read_text())
        if 'model_path' in receipt:
            if digest(receipt['model_path'])!=receipt['model_sha256']:raise RuntimeError('Fluid CPU model changed')
            m=mujoco.MjModel.from_binary_path(receipt['model_path'])
        else:m=mujoco.MjModel.from_xml_string(receipt['xml'])
        d=mujoco.MjData(m);m.jnt_stiffness[:]=0;m.dof_damping[:]=0;m.body_gravcomp[:]=0;m.opt.disableflags=0
        for updated in (False,True):
            c=current_case(original,updated);fv,_=reference(c,'composed');pv,_=reference(c,'projection')
            for w in range(2):
                m.body_mass[:]=c['body_mass'][w%len(c['body_mass'])];m.body_inertia[:]=c['body_inertia'][w%len(c['body_inertia'])]
                m.opt.wind[:]=c['wind'][w%len(c['wind'])];m.opt.density=float(c['density'][w%len(c['density'])]);m.opt.viscosity=float(c['viscosity'][w%len(c['viscosity'])])
                d.qpos[:]=c['qpos'][w];d.qvel[:]=c['qvel'][w]
                for key in FRAMES:
                    a=c[key][w];getattr(d,key)[:]=a.reshape(-1,9) if key=='ximat' else a
                # Actual CPU passive with spring/damping/gravcomp coefficients
                # zero isolates its inertia-box fluid, including mj_applyFT.
                mujoco.mj_passive(m,d);values={'fluid':d.qfrc_passive.copy()}
                project=np.zeros(m.nv)
                for b in range(1,m.nbody):
                    force=c['body_force'][w,b].astype(float)
                    mujoco.mj_applyFT(m,d,force[:3].copy(),force[3:].copy(),c['xipos'][w,b].astype(float),b,project)
                values['projection']=project
                for phase,value in values.items():
                    target=(fv if phase=='fluid' else pv)['qfrc_fluid'][w]
                    error=float(abs(value-target).max(initial=0.));limit=8192*np.finfo(float).eps*max(1.,float(abs(target).max(initial=0.)))
                    rows.append(dict(name=c['name'],updated=updated,world=w,phase=phase,error=error,limit=limit,passed=bool(error<=limit)))
    result=dict(passed=all(r['passed'] for r in rows),version=mujoco.__version__,checks=rows,inputs=input_identity(args),tool_sha256=digest(__file__),
        reference_sha256=digest(Path(__file__).with_name('mjwarp_fluid_reference.py')),helpers=helpers(),physics_integrations=0,newton_solves=0)
    (args.output/'cpu-oracle.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=result['passed'],cpu_checks=len(rows))),flush=True)
    return 0 if result['passed'] else 2


def construct(c,stream):
    import warp as wp
    from mujoco_warp._src import types
    m,d=passive_construct(c,stream)
    arr=lambda value,dtype:wp.array(np.ascontiguousarray(value),dtype=dtype,device=stream.device)
    for key in ('nbody','ngeom'):setattr(m,key,c[key])
    for key in ('ntendon','nflex','nflexedge','nflexelem','nflexbend_interp','nflexintcell'):setattr(m,key,0)
    m.opt.integrator=types.IntegratorType.EULER;m.opt.enableflags=0
    for key in ('wind','density','viscosity'):setattr(m.opt,key,arr(c[key],wp.vec3 if key=='wind' else float))
    for field,key,dtype in (('body_parentid','parent',int),('body_rootid','root',int),('body_subtreenum','subtree_size',int),('dof_bodyid','dof_body',int),
            ('body_fluid_ellipsoid','ellipsoid',wp.bool),('body_geomnum','body_geomnum',int),('body_geomadr','body_geomadr',int),('geom_type','geom_type',int),
            ('body_mass','body_mass',float),('body_inertia','body_inertia',wp.vec3),('geom_size','geom_size',wp.vec3),('geom_fluid','geom_fluid',float)):
        setattr(m,field,arr(c[key],dtype))
    for key in (*FRAMES,'geom_xpos','geom_xmat'):
        dtype=wp.mat33 if key.endswith('mat') else wp.spatial_vector if key in ('cvel','cdof') else wp.vec3
        setattr(d,key,arr(c[key],dtype))
    return m,d


def upstream_force(m,d,force):
    import warp as wp
    from mujoco_warp._src import passive
    wp.launch(passive._fluid_force,dim=(2,m.nbody),inputs=[m.opt.wind,m.opt.density,m.opt.viscosity,m.body_rootid,m.body_geomnum,m.body_geomadr,
        m.body_mass,m.body_inertia,m.geom_type,m.geom_size,m.geom_fluid,m.body_fluid_ellipsoid,d.xipos,d.ximat,d.geom_xpos,d.geom_xmat,d.subtree_com,d.cvel],outputs=[force])


def upstream_joint(m,d):
    import warp as wp
    from mujoco_warp._src import passive
    wp.launch(passive._spring_damper_dof_passive,dim=(2,m.njnt),inputs=[m.opt.disableflags,m.qpos_spring,m.jnt_type,m.jnt_qposadr,m.jnt_dofadr,
        m.jnt_stiffness,m.jnt_stiffnesspoly,m.dof_damping,m.dof_dampingpoly,d.qpos,d.qvel],outputs=[d.qfrc_spring,d.qfrc_damper])


class Replay:
    def __init__(self,c,args,stream,phase):
        import warp as wp
        from mujoco_warp._src import passive,support
        from libero.gpu.host.runtime.dynamics.mjwarp_fluid import FluidNumerics,qualify
        from libero.gpu.host.runtime.dynamics.mjwarp_passive import PassiveForces
        self.case,self.stream,self.phase=c,stream,phase;self.models={};self.datas={};self.graphs={};self.forces={};self.keys=tuple(reference(c,phase)[0])
        for mode in ('upstream','cuda'):
            m,d=construct(c,stream);self.models[mode]=m;self.datas[mode]=d
            if mode=='cuda':
                self.binding=FluidNumerics(m,d,args.library,stream,admission=qualify(m,d,implementation='cuda'));self.forces[mode]=self.binding.arrays['body_force']
                if phase=='passive':self.passive=PassiveForces(m,d,args.passive_library,stream)
            else:self.forces[mode]=wp.empty((2,m.nbody),dtype=wp.spatial_vector)
            with wp.ScopedCapture(stream=stream) as cap:
                if phase=='passive':
                    if mode=='cuda':self.passive.enqueue(0)
                    else:upstream_joint(m,d)
                if phase!='projection':
                    if mode=='cuda':self.binding.enqueue(0)
                    else:upstream_force(m,d,self.forces[mode])
                if phase!='force':
                    if mode=='cuda':self.binding.enqueue(1)
                    else:support.apply_ft(m,d,self.forces[mode],d.qfrc_fluid,False)
                if phase=='passive':
                    if mode=='cuda':self.passive.enqueue(1)
                    else:wp.launch(passive._qfrc_passive_kernel(True,False,False),dim=(2,m.nv),inputs=[m.jnt_actgravcomp,m.dof_jntid,d.qfrc_spring,d.qfrc_damper,d.qfrc_gravcomp,d.qfrc_fluid,d.qfrc_adhesion],outputs=[d.qfrc_passive])
            self.graphs[mode]=cap.graph

    def check(self,updated):
        import warp as wp
        c=current_case(self.case,updated);expected,errors=reference(c,self.phase);checks=[];held=True;values={}
        for mode in ('upstream','cuda'):
            m,d=self.models[mode],self.datas[mode]
            for key in ('body_mass','body_inertia'):getattr(m,key).assign(c[key])
            for key in ('wind','density','viscosity'):getattr(m.opt,key).assign(c[key])
            m.dof_damping.assign(c['damping'])
            for key in (*FRAMES,*PASSIVE_FIELDS,'qvel'):getattr(d,key).assign(c[key])
            self.forces[mode].assign(c['body_force'])
            arrays={**{'model.'+k:v for k,v in vars(m).items() if hasattr(v,'numpy')},**{'option.'+k:v for k,v in vars(m.opt).items() if hasattr(v,'numpy')},
                **{'data.'+k:v for k,v in vars(d).items() if hasattr(v,'numpy') and k not in self.keys}}
            if 'body_force' not in self.keys:arrays['body_force']=self.forces[mode]
            before={key:a.numpy().view(np.uint8).copy() for key,a in arrays.items()}
            wp.capture_launch(self.graphs[mode],stream=self.stream);wp.synchronize_stream(self.stream)
            values[mode]={key:(self.forces[mode] if key=='body_force' else getattr(d,key)).numpy() for key in self.keys}
            checks += [dict(mode=mode,**x) for x in check_arrays(values[mode],expected,errors)]
            held &= all(np.array_equal(a.numpy().view(np.uint8),before[key]) for key,a in arrays.items())
        return dict(name=c['name'],phase=self.phase,updated=updated,inputs_and_holds=bool(held),checks=checks,
            maximum_difference={key:float(abs(values['cuda'][key]-values['upstream'][key]).max(initial=0.)) for key in self.keys},passed=bool(held and all(x['passed'] for x in checks)))


def signature(args):
    from libero.gpu.host.backends.mjwarp.build import source_files
    sources=source_files(include_fluid=True,include_passive=args.passive_library is not None)
    sources={k:v for k,v in sources.items() if k.startswith('upstream/') or 'mjwarp_fluid' in k or 'MjwarpFluid' in k or 'mjwarp_passive' in k or 'MjwarpPassive' in k or k in ('host/runtime/dynamics/mjwarp_motion.py','data/mjwarp_source_pin.json','project_source_identity')}
    return dict(sources=sources,inputs=input_identity(args),library_sha256=digest(args.library),passive_sha256=digest(args.passive_library) if args.passive_library else None,
        tool_sha256=digest(__file__),reference_sha256=digest(Path(__file__).with_name('mjwarp_fluid_reference.py')),helpers=helpers())


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
        if not prep.is_file() or json.loads(prep.read_text())['signature']!=sig:raise RuntimeError('Prepare exact fluid operators offline first')
        cpu=json.loads((args.output/'cpu-oracle.json').read_text())
        if not cpu['passed'] or any(cpu[k]!=sig[k] for k in ('inputs','tool_sha256','reference_sha256','helpers')):raise RuntimeError('Prepare matching CPU fluid oracle first')
    with compilation_scope(args.prepare) as scope,wp.ScopedStream(stream):
        replays=[Replay(c,args,stream,p) for p in ('force','projection','composed') for c in cases(args)]
        if args.passive_library:replays += [Replay(c,args,stream,'passive') for c in cases(args) if c['name'].startswith('libero')]
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
            policy='independent FP64 inertia-box and spatial/Jacobian algebra, propagated FP32 arithmetic/clamp/sqrt/powf bounds, fixed8x safety; exact readonly/hold contracts'))
        print(json.dumps(dict(passed=passed,cases=len(rows))),flush=True);return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())
