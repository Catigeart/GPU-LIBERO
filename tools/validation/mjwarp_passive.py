"""Frozen N=2 joint passive forces, final sum, and composition; no timing."""
import argparse,hashlib,json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from .mjwarp_passive_reference import reference
from .mjwarp_motion_reference import check_arrays

TASKS=(('libero_10',0),('libero_10',3),('libero_spatial',0))
PHASES=('joint','combine','composed')
FIELDS=('qfrc_spring','qfrc_damper','qfrc_gravcomp','qfrc_fluid','qfrc_adhesion','qfrc_passive')


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def add_arguments(p):
    for name in ('build-root','output','library'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--catalogue',type=Path,required=True)
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--cpu-reference',action='store_true')
    return p


def contributions(c):
    idx=np.arange(2*c['nv'],dtype=np.float64).reshape(2,c['nv'])
    for i,key in enumerate(FIELDS):c[key]=(np.sin(idx*.2+i+.3)*(i+1)).astype(np.float32)


def freeze_real(args):
    import mujoco
    rows=json.loads(args.catalogue.read_text())['rows']
    for suite,task in TASKS:
        name=f'{suite}-{task}';p=args.output/(name+'.npz')
        if p.exists():
            if digest(p)!=json.loads(p.with_suffix('.json').read_text())['sha256']:raise RuntimeError('Passive input changed')
            continue
        row=next(r for r in rows if r['suite']==suite and r['task_id']==task)
        if digest(row['model_path'])!=row['model_sha256']:raise RuntimeError('Passive model changed')
        m=mujoco.MjModel.from_binary_path(row['model_path']);bank=np.load(Path(row['model_path']).parent/'initial_states.npy')
        c=dict(nq=np.array(m.nq),nv=np.array(m.nv),njnt=np.array(m.njnt),
            qpos=bank[:2,1:1+m.nq].astype(np.float32),qpos_spring=m.qpos_spring[None].astype(np.float32),
            stiffness=m.jnt_stiffness[None].astype(np.float32),damping=m.dof_damping[None].astype(np.float32),
            stiffnesspoly=np.zeros((1,m.njnt,2),np.float32),dampingpoly=np.zeros((1,m.nv,2),np.float32),
            actgravcomp=np.zeros(m.njnt,np.int32),disable_spring=np.array(False),disable_damper=np.array(False),
            gravity_enabled=np.array(True),has_fluid=np.array(True),has_adhesion=np.array(False))
        for key,field in (('joint_type','jnt_type'),('qposadr','jnt_qposadr'),('dofadr','jnt_dofadr'),('dof_joint','dof_jntid')):
            c[key]=getattr(m,field).astype(np.int32)
        idx=np.arange(m.nv,dtype=float)
        c['qvel']=np.stack((.35*np.sin(idx+.2)+.12,.6*np.cos(idx*.7+.4)-.2)).astype(np.float32)
        contributions(c);np.savez(p,**c)
        p.with_suffix('.json').write_text(json.dumps(dict(sha256=digest(p),model_path=row['model_path'],model_sha256=row['model_sha256'],
            catalogue_sha256=digest(args.catalogue),initial_states_sha256=digest(Path(row['model_path']).parent/'initial_states.npy'),
            note='Real joint/parameter/state inputs; other already-computed force contributions are manufactured frozen vectors'),indent=2)+'\n')


def real_cases(args):
    result=[]
    for suite,task in TASKS:
        name=f'{suite}-{task}';p=args.output/(name+'.npz');receipt=json.loads(p.with_suffix('.json').read_text())
        if digest(p)!=receipt['sha256'] or digest(args.catalogue)!=receipt['catalogue_sha256']:raise RuntimeError('Passive input receipt differs')
        with np.load(p,allow_pickle=False) as f:c={k:f[k].copy() for k in f.files}
        for key in ('nq','nv','njnt'):c[key]=int(c[key])
        for key in ('disable_spring','disable_damper','gravity_enabled','has_fluid','has_adhesion'):c[key]=bool(c[key])
        c.update(name=name,model_path=receipt['model_path']);result.append(c)
    return result


def synthetic():
    c=dict(name='mixed-linear',nq=14,nv=12,njnt=5,joint_type=np.array([0,1,2,3,3],np.int32),
        qposadr=np.array([0,7,11,12,13],np.int32),dofadr=np.array([0,6,9,10,11],np.int32),
        dof_joint=np.array([0]*6+[1]*3+[2,3,4],np.int32),actgravcomp=np.array([0,1,0,1,0],np.int32),
        qpos=np.zeros((2,14),np.float32),qpos_spring=np.zeros((1,14),np.float32),
        stiffness=np.array([[.8,1.3,.5,2.2,0.]],np.float32),stiffnesspoly=np.zeros((1,5,2),np.float32),
        damping=np.linspace(.01,.3,24,dtype=np.float32).reshape(2,12),dampingpoly=np.zeros((1,12,2),np.float32),
        qvel=np.linspace(-.8,1.2,24,dtype=np.float32).reshape(2,12),
        disable_spring=False,disable_damper=False,gravity_enabled=True,has_fluid=True,has_adhesion=True)
    c['qpos_spring'][0,3]=1;c['qpos_spring'][0,7]=1
    c['qpos'][:,:3]=np.array([[.3,-.2,.1],[-.4,.7,.2]],np.float32)
    for w in range(2):
        for p,angle in ((3,.4+w*.6),(7,-.7-w*.2)):
            axis=np.array([.3,.4,.5]);axis/=np.linalg.norm(axis)
            c['qpos'][w,p:p+4]=np.r_[np.cos(angle/2),axis*np.sin(angle/2)]
    c['qpos'][:,11:]=np.array([[.3,-.2,.1],[-.7,.4,-.3]],np.float32)
    contributions(c);return c


def cases(args):
    result=real_cases(args);base=synthetic();result.append(base)
    names=('mixed-polynomial','small-angles','near-halfturn-below','near-halfturn-above','zero-quaternion',
        'disable-spring','disable-damper','disable-both','gravity-off','no-fluid-no-adhesion','zero-coefficients')
    for name in names:
        c={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in base.items()};c['name']=name
        if name=='mixed-polynomial':
            c['qpos_spring']=np.repeat(c['qpos_spring'],2,axis=0);c['qpos_spring'][1,:3]=[.2,-.1,.4]
            c['stiffnesspoly']=np.linspace(-.03,.08,20,dtype=np.float32).reshape(2,5,2)
            c['dampingpoly']=np.linspace(.01,.05,24,dtype=np.float32).reshape(1,12,2)
            c['stiffness']=np.repeat(c['stiffness'],2,axis=0);c['stiffness'][1]*=.7
        elif name in ('small-angles','near-halfturn-below','near-halfturn-above'):
            angle={'small-angles':1e-7,'near-halfturn-below':np.pi-1e-3,'near-halfturn-above':np.pi+1e-3}[name]
            for w in range(2):
                for p in (3,7):c['qpos'][w,p:p+4]=np.array([np.cos(angle/2),np.sin(angle/2),0,0],np.float32)*(-1 if w else 1)
        elif name=='zero-quaternion':c['qpos'][:,3:7]=0;c['qpos'][:,7:11]=0
        elif name=='disable-spring':c['disable_spring']=True
        elif name=='disable-damper':c['disable_damper']=True
        elif name=='disable-both':c['disable_spring']=True;c['disable_damper']=True
        elif name=='gravity-off':c['gravity_enabled']=False
        elif name=='no-fluid-no-adhesion':
            c['has_fluid']=False;c['has_adhesion']=False;c['qfrc_fluid'][:]=np.nan;c['qfrc_adhesion'][:]=np.nan
        elif name=='zero-coefficients':
            c['stiffness'][:]=0;c['damping'][:]=0;c['qpos'][:]=np.nan;c['qvel'][:]=np.nan
        result.append(c)
    return result


def current_case(original,updated):
    c={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in original.items()}
    if updated:
        c['qvel']*=np.array([.625,-1.25],np.float32)[:,None]
        for key in ('stiffness','stiffnesspoly','damping','dampingpoly'):c[key]*=np.float32(1.125)
        # Keep quaternion boundary families fixed; scalar coordinates and all
        # other resident parameters/contributions still change in the same Graph.
        for j,kind in enumerate(c['joint_type']):
            p=c['qposadr'][j]
            if kind==0:c['qpos'][:,p:p+3]+=np.float32(.125)
            elif kind>=2:c['qpos'][:,p]-=np.float32(.175)
        c['qpos_spring'][:,:3]+=np.float32(.03125)
        for key in FIELDS:c[key]*=np.array([.75,-1.5],np.float32)[:,None]
    return c


def input_identity(args):return {f'{s}-{t}':digest(args.output/f'{s}-{t}.npz') for s,t in TASKS}


def cpu_reference(args):
    import mujoco
    if mujoco.__version__!='2.3.7':raise RuntimeError('CPU passive oracle requires MuJoCo2.3.7')
    freeze_real(args);checks=[];saved={}
    for original in real_cases(args):
        receipt=json.loads((args.output/(original['name']+'.json')).read_text())
        if digest(original['model_path'])!=receipt['model_sha256']:raise RuntimeError('CPU passive model changed')
        m=mujoco.MjModel.from_binary_path(original['model_path']);d=mujoco.MjData(m)
        m.opt.density=0;m.opt.viscosity=0;m.body_gravcomp[:]=0;m.opt.disableflags=0
        for updated in (False,True):
            c=current_case(original,updated);v,_=reference(c,'joint');expected=v['qfrc_spring']+v['qfrc_damper'];values=[]
            for w in range(2):
                d.qpos[:]=c['qpos'][w];d.qvel[:]=c['qvel'][w];m.qpos_spring[:]=c['qpos_spring'][w%len(c['qpos_spring'])]
                m.jnt_stiffness[:]=c['stiffness'][w%len(c['stiffness'])];m.dof_damping[:]=c['damping'][w%len(c['damping'])]
                mujoco.mj_passive(m,d);values.append(d.qfrc_passive.copy())
            values=np.array(values);error=float(np.max(abs(values-expected)));limit=4096*np.finfo(float).eps*max(1.,float(np.max(abs(expected))))
            assert error<=limit,(c['name'],error,limit)
            checks.append(dict(name=c['name'],updated=updated,field='linear-joint-passive',error=error,limit=limit))
            saved[c['name']+str(updated)]=values
    # Real models have no springs/BALL: validate the quaternion mathematics with
    # the installed CPU utility on the manufactured nonzero-spring inputs too.
    from .mjwarp_passive_reference import quaternion_difference
    for c in cases(args)[len(TASKS):]:
        if c['name']=='zero-quaternion' or c['name']=='zero-coefficients':continue
        for w in range(2):
            for j,kind in enumerate(c['joint_type']):
                if kind not in (0,1):continue
                p=int(c['qposadr'][j])+(3 if kind==0 else 0);q=c['qpos'][w,p:p+4].astype(float);q/=np.linalg.norm(q)
                ref=c['qpos_spring'][w%len(c['qpos_spring']),p:p+4].astype(float);out=np.empty(3)
                mujoco.mju_subQuat(out,q,ref);target,_=quaternion_difference(c['qpos'][w,p:p+4],ref)
                error=float(abs(out-target).max());limit=4096*np.finfo(float).eps*max(1.,float(abs(target).max()))
                assert error<=limit,(c['name'],error,limit)
                checks.append(dict(name=c['name'],world=w,joint=j,field='quaternion-difference',error=error,limit=limit))
    p=args.output/'cpu-oracle.npz';np.savez(p,**saved)
    receipt=dict(version=mujoco.__version__,sha256=digest(p),tool_sha256=digest(__file__),
        reference_sha256=digest(Path(__file__).with_name('mjwarp_passive_reference.py')),inputs=input_identity(args),checks=checks,
        physics_integrations=0,newton_solves=0)
    (args.output/'cpu-oracle.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(passed=True,cpu_checks=len(checks))),flush=True);return 0


def construct(c,stream):
    import warp as wp
    from mujoco_warp._src import types
    arr=lambda a,dtype:wp.array(np.ascontiguousarray(a),dtype=dtype,device=stream.device)
    m=NS(nv=c['nv'],nq=c['nq'],njnt=c['njnt'],has_fluid=c['has_fluid'],flg_adhesion=c['has_adhesion'])
    for field,key in (('jnt_type','joint_type'),('jnt_qposadr','qposadr'),('jnt_dofadr','dofadr'),('dof_jntid','dof_joint'),('jnt_actgravcomp','actgravcomp')):
        setattr(m,field,arr(c[key],int))
    flags=(int(types.DisableBit.SPRING) if c['disable_spring'] else 0)|(int(types.DisableBit.DAMPER) if c['disable_damper'] else 0)|(0 if c['gravity_enabled'] else int(types.DisableBit.GRAVITY))
    m.opt=NS(disableflags=flags)
    for field,key,dtype in (('qpos_spring','qpos_spring',float),('jnt_stiffness','stiffness',float),('jnt_stiffnesspoly','stiffnesspoly',wp.vec2),('dof_damping','damping',float),('dof_dampingpoly','dampingpoly',wp.vec2)):
        setattr(m,field,arr(c[key],dtype))
    d=NS(nworld=2,qpos=arr(c['qpos'],float),qvel=arr(c['qvel'],float))
    for key in FIELDS:setattr(d,key,arr(c[key],float))
    d.qacc=wp.full((2,m.nv),float('nan'),dtype=float,device=stream.device)
    d.qacc_warmstart=wp.full((2,m.nv),37.,dtype=float,device=stream.device)
    return m,d


class Replay:
    def __init__(self,c,args,stream,phase):
        import warp as wp
        from mujoco_warp._src import passive
        from libero.gpu.host.runtime.dynamics.mjwarp_passive import PassiveForces
        self.case,self.phase,self.stream=c,phase,stream;self.outputs={};self.graphs={}
        self.model,self.data=m,d=construct(c,stream)
        names=('qfrc_spring','qfrc_damper') if phase=='joint' else ('qfrc_passive',) if phase=='combine' else ('qfrc_spring','qfrc_damper','qfrc_passive')
        for mode in ('upstream','cuda'):
            for key in names:setattr(d,key,wp.empty((2,m.nv),dtype=float,device=stream.device))
            self.outputs[mode]={k:getattr(d,k) for k in names}
            if mode=='cuda':self.binding=PassiveForces(m,d,args.library,stream)
            with wp.ScopedCapture(stream=stream) as capture:
                if phase!='combine':
                    if mode=='cuda':self.binding.enqueue(0)
                    else:wp.launch(passive._spring_damper_dof_passive,dim=(2,m.njnt),
                        inputs=[m.opt.disableflags,m.qpos_spring,m.jnt_type,m.jnt_qposadr,m.jnt_dofadr,m.jnt_stiffness,m.jnt_stiffnesspoly,m.dof_damping,m.dof_dampingpoly,d.qpos,d.qvel],outputs=[d.qfrc_spring,d.qfrc_damper])
                if phase!='joint':
                    if mode=='cuda':self.binding.enqueue(1)
                    else:wp.launch(passive._qfrc_passive_kernel(m.has_fluid,m.flg_adhesion,c['gravity_enabled']),dim=(2,m.nv),
                        inputs=[m.jnt_actgravcomp,m.dof_jntid,d.qfrc_spring,d.qfrc_damper,d.qfrc_gravcomp,d.qfrc_fluid,d.qfrc_adhesion],outputs=[d.qfrc_passive])
            self.graphs[mode]=capture.graph

    def check(self,updated):
        import warp as wp
        c=current_case(self.case,updated);m,d=self.model,self.data
        for field,key in (('qpos_spring','qpos_spring'),('jnt_stiffness','stiffness'),('jnt_stiffnesspoly','stiffnesspoly'),('dof_damping','damping'),('dof_dampingpoly','dampingpoly')):getattr(m,field).assign(c[key])
        d.qpos.assign(c['qpos']);d.qvel.assign(c['qvel'])
        readonly={k:a for k,a in self.binding.arrays.items() if k not in ('spring','damper','passive')}
        for key in FIELDS:
            if key not in self.outputs['cuda']:getattr(d,key).assign(c[key]);readonly[key]=getattr(d,key)
        readonly.update(qacc=d.qacc,qacc_warmstart=d.qacc_warmstart)
        before={k:a.numpy().view(np.uint8).copy() for k,a in readonly.items()}
        expected,bounds=reference(c,self.phase);checks=[];values={}
        for mode,g in self.graphs.items():
            for a in self.outputs[mode].values():a.fill_(37.)
            wp.capture_launch(g,stream=self.stream);wp.synchronize_stream(self.stream)
            values[mode]={k:a.numpy() for k,a in self.outputs[mode].items()}
            checks.extend(dict(mode=mode,**r) for r in check_arrays(values[mode],expected,bounds))
        held=all(np.array_equal(a.numpy().view(np.uint8),before[k]) for k,a in readonly.items())
        return dict(name=c['name'],phase=self.phase,updated=updated,inputs_held=bool(held),checks=checks,
            max_difference={k:float(np.max(abs(values['cuda'][k]-values['upstream'][k]))) for k in expected},
            passed=bool(held and all(r['passed'] for r in checks))),values


def signature(args):
    from libero.gpu.host.backends.mjwarp.build import source_files
    names=source_files(include_passive=True)
    names={k:v for k,v in names.items() if k.startswith('upstream/') or 'mjwarp_passive' in k or k in (
        'cmake/MjwarpPassive.cmake','host/runtime/dynamics/mjwarp_motion.py','data/mjwarp_source_pin.json','project_source_identity')}
    return dict(sources=names,tool_sha256=digest(__file__),reference_sha256=digest(Path(__file__).with_name('mjwarp_passive_reference.py')),
        roundoff_helpers_sha256=digest(Path(__file__).with_name('mjwarp_motion_reference.py')),inputs=input_identity(args),library_sha256=digest(args.library))


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
    require_upstream();init_kernel_cache(args.build_root/'mjwarp/warp-cache');install_warp_guard();wp.init()
    stream=wp.Stream('cuda:0');sig=signature(args);prepared=args.output/'preparation.json'
    if not args.prepare:
        if not prepared.is_file() or json.loads(prepared.read_text())['signature']!=sig:raise RuntimeError('Prepare exact passive operators offline first')
        receipt=json.loads((args.output/'cpu-oracle.json').read_text())
        if receipt['sha256']!=digest(args.output/'cpu-oracle.npz') or any(receipt[k]!=sig[k] for k in ('tool_sha256','reference_sha256','inputs')):
            raise RuntimeError('Prepare exact CPU passive oracle first')
    with compilation_scope(args.prepare) as scope,wp.ScopedStream(stream):
        replays=[Replay(c,args,stream,p) for p in PHASES for c in cases(args)]
        if args.prepare:
            wp.synchronize_stream(stream);save_json(prepared,dict(signature=sig,compilation_events=scope['events'],physics_integrations=0,newton_solves=0))
            print(json.dumps(dict(prepared=True,captures=len(replays)*2,compilations=len(scope['events']))),flush=True);return 0
        rows=[];passed=True
        for r in replays:
            if not passed:break
            for updated in (False,True):
                row,values=r.check(updated);rows.append(row)
                np.savez(args.output/(row['name']+'-'+row['phase']+('-updated' if updated else '')+'-output.npz'),**{mode+'.'+k:v for mode,vv in values.items() for k,v in vv.items()})
                print(json.dumps({k:row[k] for k in ('name','phase','updated','passed')}),flush=True)
                passed &= row['passed']
        save_json(args.output/'summary.json',dict(passed=passed,signature=sig,cases=rows,runtime_compilations=scope['events'],newton_solves=0,physics_integrations=0,performance=None,
            policy='independent FP64 algebra; input-derived operation and quaternion sensitivity bounds; fixed8x safety; exact readonly/ownership'))
        print(json.dumps(dict(passed=passed,cases=len(rows))),flush=True);return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())
