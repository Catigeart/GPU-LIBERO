"""Frozen N=2 stateless actuation, moment projection and limits; no timing."""
import argparse,hashlib,json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from .mjwarp_actuation_reference import reference,dense_moment
from .mjwarp_motion_reference import check_arrays

TASKS=(('libero_10',0),('libero_10',3),('libero_spatial',0))
PHASES=('force','projection','limits','composed')


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def add_arguments(p):
    for name in ('build-root','catalogue','output','library'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--cpu-reference',action='store_true')
    return p


def extras(c):
    c['actuator_force']=(np.sin(np.arange(2*c['nactuator']).reshape(2,c['nactuator'])*.7+.3)*3).astype(np.float32)
    c['raw_qfrc']=(np.cos(np.arange(2*c['nv']).reshape(2,c['nv'])*.4+.1)*4).astype(np.float32)
    c['gravcomp']=(np.sin(np.arange(2*c['nv']).reshape(2,c['nv'])*.2+.8)*2).astype(np.float32)


def pack_moments(dense):
    n,na,nv=dense.shape;counts=np.count_nonzero(dense,axis=2).astype(np.int32)
    capacity=max(1,int(counts.sum(axis=1).max()));columns=np.full((n,capacity),-1,np.int32);values=np.zeros((n,capacity),np.float32);addresses=np.zeros((n,na),np.int32)
    for w in range(n):
        at=0
        for u in range(na):
            addresses[w,u]=at;ids=np.flatnonzero(dense[w,u]);columns[w,at:at+len(ids)]=ids;values[w,at:at+len(ids)]=dense[w,u,ids];at+=len(ids)
    return dict(moment_rownnz=counts,moment_rowadr=addresses,moment_colind=columns,moment=values)


def freeze_real(args):
    import mujoco
    catalogue=json.loads(args.catalogue.read_text());assert catalogue['passed']
    for suite,task in TASKS:
        name=f'{suite}-{task}';p=args.output/(name+'.npz')
        if p.exists():
            if digest(p)!=json.loads(p.with_suffix('.json').read_text())['sha256']:raise RuntimeError('Actuation input changed')
            continue
        row=next(r for r in catalogue['rows'] if r['suite']==suite and r['task_id']==task)
        if digest(row['model_path'])!=row['model_sha256']:raise RuntimeError('Actuation cached model changed')
        m=mujoco.MjModel.from_binary_path(row['model_path']);d=mujoco.MjData(m)
        assert m.nu==9 and m.na==0 and m.ntendon==0
        c=dict(nv=m.nv,nu=m.nu,nactuator=m.nu,njnt=m.njnt,na=0,disable_clampctrl=False,gravity_enabled=True)
        for key,field in (('gain_type','actuator_gaintype'),('bias_type','actuator_biastype'),('dyntype','actuator_dyntype'),
                ('trntype','actuator_trntype'),('trnid','actuator_trnid'),('joint_type','jnt_type'),('dof_joint','dof_jntid')):
            c[key]=getattr(m,field).astype(np.int32)
        c.update(ctrladr=np.arange(m.nu,dtype=np.int32),ctrlnum=np.ones(m.nu,np.int32),actadr=m.actuator_actadr.astype(np.int32),actgravcomp=np.zeros(m.njnt,np.int32))
        for key,field in (('ctrllimited','actuator_ctrllimited'),('forcelimited','actuator_forcelimited'),('jointlimited','jnt_actfrclimited')):
            c[key]=getattr(m,field).astype(bool)
        for key,field in (('gainprm','actuator_gainprm'),('biasprm','actuator_biasprm'),('ctrlrange','actuator_ctrlrange'),('forcerange','actuator_forcerange'),('jointrange','jnt_actfrcrange')):
            c[key]=getattr(m,field)[None].astype(np.float32)
        banks=np.load(Path(row['model_path']).parent/'initial_states.npy');mom=[];length=[];velocity=[]
        for w in range(2):
            d.qpos[:]=banks[w,1:1+m.nq]
            d.qvel[:]=.4*np.sin(np.arange(m.nv)*.7+.2+w)
            mujoco.mj_transmission(m,d)
            mom.append(d.actuator_moment.copy());length.append(d.actuator_length.copy());velocity.append(d.actuator_moment@d.qvel)
        c.update(pack_moments(np.array(mom)));c['length']=np.array(length,np.float32);c['velocity']=np.array(velocity,np.float32)
        fraction=np.linspace(-.25,1.25,m.nu,dtype=np.float32);lo=c['ctrlrange'][0,:,0];hi=c['ctrlrange'][0,:,1]
        c['ctrl']=np.stack((lo+fraction*(hi-lo),lo+fraction[::-1]*(hi-lo))).astype(np.float32);extras(c)
        assert np.array_equal(c['gain_type'],np.zeros(9,int)) and np.array_equal(c['bias_type'],[0]*7+[1,1])
        np.savez(p,**c);p.with_suffix('.json').write_text(json.dumps(dict(sha256=digest(p),model_path=row['model_path'],model_sha256=row['model_sha256'],catalogue_sha256=digest(args.catalogue),
            note='Actual CPU2.3.7 parameters and transmission moments/lengths on cached states; converted fields checked in production admission',physics_integrations=0,newton_solves=0),indent=2)+'\n')


def real_cases(args):
    result=[]
    for suite,task in TASKS:
        name=f'{suite}-{task}';p=args.output/(name+'.npz');receipt=json.loads(p.with_suffix('.json').read_text())
        if digest(p)!=receipt['sha256'] or digest(args.catalogue)!=receipt['catalogue_sha256']:raise RuntimeError('Actuation input receipt differs')
        with np.load(p,allow_pickle=False) as f:c={k:f[k].copy() for k in f.files}
        for key in ('nv','nu','nactuator','njnt','na'):c[key]=int(c[key])
        for key in ('disable_clampctrl','gravity_enabled'):c[key]=bool(c[key])
        c.update(name=name,model_path=receipt['model_path']);result.append(c)
    return result


def synthetic():
    na,nv,nu=6,6,4;c=dict(name='mixed-affine',nv=nv,nu=nu,nactuator=na,njnt=nv,na=0,
        gain_type=np.array([0,1,0,1,1,0],np.int32),bias_type=np.array([0,1,1,0,1,1],np.int32),
        dyntype=np.zeros(na,np.int32),trntype=np.zeros(na,np.int32),trnid=np.stack((np.arange(na),-np.ones(na)),axis=1).astype(np.int32),
        joint_type=np.array([2,3,2,3,2,3],np.int32),dof_joint=np.arange(nv,dtype=np.int32),
        ctrladr=np.array([0,1,2,3,-1,1],np.int32),ctrlnum=np.array([1,1,1,1,0,1],np.int32),actadr=np.full(na,-1,np.int32),
        ctrllimited=np.array([1,0,1,1],bool),forcelimited=np.array([0,1,1,0,1,1],bool),jointlimited=np.array([1,1,0,0,1,0],bool),
        actgravcomp=np.array([0,1,0,1,1,0],np.int32),disable_clampctrl=False,gravity_enabled=True,
        gainprm=np.zeros((1,na,10),np.float32),biasprm=np.zeros((2,na,10),np.float32),
        ctrlrange=np.array([[[-1,1],[-2,2],[-.2,.5],[0,.04]]],np.float32),
        forcerange=np.tile(np.array([-2,2],np.float32),(2,na,1)),jointrange=np.tile(np.array([-3,3],np.float32),(1,nv,1)),
        ctrl=np.array([[-1.2,.3,.8,.07],[1.5,-.4,-.3,-.1]],np.float32),
        length=np.linspace(-.6,.8,12,dtype=np.float32).reshape(2,na),velocity=np.linspace(.7,-.9,12,dtype=np.float32).reshape(2,na))
    c['gainprm'][0,:,:3]=np.array([[1.3,0,0],[.4,.5,-.2],[.8,0,0],[1.1,-.3,.4],[.6,.2,-.1],[1.7,0,0]],np.float32)
    c['biasprm'][:,:,:3]=np.array([[[.2,-.3,.1]]*na,[[.1,.2,-.4]]*na],np.float32)
    dense=np.zeros((2,na,nv),np.float32)
    for w in range(2):
        for u in range(na):
            if u==4:continue
            dense[w,u,(u+w)%nv]=1.+.1*u;dense[w,u,(u+2+w)%nv]=(-1 if u%2 else 1)*.3
    c.update(pack_moments(dense));extras(c);return c


def cases(args):
    result=real_cases(args);base=synthetic();result.append(base)
    for name in ('clamp-disabled','gravity-disabled','all-limits-off','control-boundary','force-boundary','joint-boundary',
                 'world-broadcast','overlap-cancellation','all-rows-empty','fixed-ignores-length','zero-controls'):
        c={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in base.items()};c['name']=name
        if name=='clamp-disabled':c['disable_clampctrl']=True
        elif name=='gravity-disabled':c['gravity_enabled']=False
        elif name=='all-limits-off':c['ctrllimited'][:]=0;c['forcelimited'][:]=0;c['jointlimited'][:]=0
        elif name=='control-boundary':
            c['ctrl'][0]=c['ctrlrange'][0,:,0];c['ctrl'][1]=np.nextafter(c['ctrlrange'][0,:,1],np.float32(np.inf))
        elif name=='force-boundary':
            c['gain_type'][:]=0;c['bias_type'][:]=0;c['gainprm'][:]=0;c['gainprm'][:,:,0]=1;c['forcelimited'][:]=1;c['forcerange'][:]=[0,.5]
            c['ctrl'][:]=[0,.5,np.nextafter(np.float32(.5),np.float32(np.inf)),-.1]
        elif name=='joint-boundary':
            c['jointlimited'][:]=1;c['raw_qfrc'][0]=c['jointrange'][0,:,0];c['raw_qfrc'][1]=np.nextafter(c['jointrange'][0,:,1],np.float32(np.inf));c['actgravcomp'][:]=0
        elif name=='world-broadcast':
            c['gainprm']=np.repeat(c['gainprm'],2,axis=0);c['gainprm'][1]*=1.5
            c['ctrlrange']=np.repeat(c['ctrlrange'],2,axis=0);c['ctrlrange'][1]*=.75
            c['jointrange']=np.repeat(c['jointrange'],2,axis=0);c['jointrange'][1]*=.5
        elif name=='overlap-cancellation':
            c['moment_colind'][:]=0;c['moment']*=np.float32(1e3);c['actuator_force'][1]*=np.float32(-1e3)
        elif name=='all-rows-empty':c['moment_rownnz'][:]=0;c['moment_colind'][:]=-1;c['moment'][:]=np.nan
        elif name=='fixed-ignores-length':c['gain_type'][:]=0;c['bias_type'][:]=0;c['length'][:]=np.nan;c['velocity'][:]=np.nan
        elif name=='zero-controls':c['ctrl'][:]=0;c['gainprm'][:]=0;c['biasprm'][:]=0;c['gravcomp'][:]=0
        result.append(c)
    return result


def current_case(original,updated):
    c={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in original.items()}
    if updated:
        c['ctrl']*=np.array([.75,-1.25],np.float32)[:,None]
        c['length']+=np.float32(.125);c['velocity']*=np.float32(-.75)
        c['gainprm']*=np.float32(.875);c['biasprm']*=np.float32(1.125)
        c['ctrlrange']*=np.float32(.9);c['forcerange']*=np.float32(.8);c['jointrange']*=np.float32(1.2)
        for key in ('moment','actuator_force','raw_qfrc','gravcomp'):c[key]*=np.array([.625,-1.5],np.float32)[:,None]
        if not c['name'].startswith('libero_'):
            valid=c['moment_colind']>=0;c['moment_colind'][valid]=(c['moment_colind'][valid]+1)%c['nv']
    return c


def input_identity(args):return {f'{s}-{t}':digest(args.output/f'{s}-{t}.npz') for s,t in TASKS}


def cpu_reference(args):
    import mujoco
    if mujoco.__version__!='2.3.7':raise RuntimeError('Actuation CPU oracle requires2.3.7')
    freeze_real(args);saved={};checks=[]
    for original in real_cases(args):
        receipt=json.loads((args.output/(original['name']+'.json')).read_text())
        if digest(original['model_path'])!=receipt['model_sha256']:raise RuntimeError('CPU actuation model differs')
        m=mujoco.MjModel.from_binary_path(original['model_path']);d=mujoco.MjData(m)
        for updated in (False,True):
            c=current_case(original,updated);expected,_=reference(c,'composed');out={k:[] for k in expected}
            for w in range(2):
                for field,key in (('actuator_gainprm','gainprm'),('actuator_biasprm','biasprm'),('actuator_ctrlrange','ctrlrange'),('actuator_forcerange','forcerange'),('jnt_actfrcrange','jointrange')):
                    getattr(m,field)[:]=c[key][w%len(c[key])]
                for field,key in (('actuator_ctrllimited','ctrllimited'),('actuator_forcelimited','forcelimited'),('jnt_actfrclimited','jointlimited')):getattr(m,field)[:]=c[key]
                bit=int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL);m.opt.disableflags=bit if c['disable_clampctrl'] else 0
                d.ctrl[:]=c['ctrl'][w];d.actuator_length[:]=c['length'][w];d.actuator_velocity[:]=c['velocity'][w];d.actuator_moment[:]=dense_moment(c)[w]
                mujoco.mj_fwdActuation(m,d)
                for key in expected:out[key].append(getattr(d,key).copy())
            for key,value in out.items():
                value=np.array(value);target=expected[key];error=float(abs(value-target).max());limit=4096*np.finfo(float).eps*max(1.,float(abs(target).max()))
                assert error<=limit,(c['name'],key,error,limit)
                checks.append(dict(name=c['name'],updated=updated,field=key,error=error,limit=limit));saved[c['name']+str(updated)+key]=value
    for original in cases(args):
        for updated in (False,True):
            c=current_case(original,updated);target,_=reference(c,'projection');dense=dense_moment(c);values=[]
            for w in range(2):
                out=np.empty(c['nv']);mujoco.mju_mulMatTVec(out,dense[w],c['actuator_force'][w].astype(float));values.append(out)
            error=float(abs(np.array(values)-target['qfrc_actuator']).max());limit=4096*np.finfo(float).eps*max(1.,float(abs(target['qfrc_actuator']).max()))
            assert error<=limit,(c['name'],error,limit);checks.append(dict(name=c['name'],updated=updated,field='dense-projection',error=error,limit=limit))
    p=args.output/'cpu-oracle.npz';np.savez(p,**saved)
    (args.output/'cpu-oracle.json').write_text(json.dumps(dict(version=mujoco.__version__,sha256=digest(p),tool_sha256=digest(__file__),reference_sha256=digest(Path(__file__).with_name('mjwarp_actuation_reference.py')),inputs=input_identity(args),checks=checks,newton_solves=0,physics_integrations=0),indent=2)+'\n')
    print(json.dumps(dict(passed=True,cpu_checks=len(checks))),flush=True);return 0


def construct(c,stream):
    import warp as wp
    from mujoco_warp._src import types
    arr=lambda a,dtype:wp.array(np.ascontiguousarray(a),dtype=dtype,device=stream.device)
    m=NS(nv=c['nv'],nu=c['nu'],nactuator=c['nactuator'],njnt=c['njnt'],na=0,ntendon=0,nhistory=0)
    flags=(int(types.DisableBit.CLAMPCTRL) if c['disable_clampctrl'] else 0)|(0 if c['gravity_enabled'] else int(types.DisableBit.GRAVITY))
    m.opt=NS(integrator=types.IntegratorType.EULER,enableflags=0,disableflags=flags,timestep=arr(np.array([.002],np.float32),float))
    for field,key,dtype in (('actuator_gaintype','gain_type',int),('actuator_biastype','bias_type',int),('actuator_dyntype','dyntype',int),
            ('actuator_trntype','trntype',int),('actuator_trnid','trnid',wp.vec2i),('jnt_type','joint_type',int),
            ('actuator_ctrladr','ctrladr',int),('actuator_ctrlnum','ctrlnum',int),('actuator_actadr','actadr',int),
            ('dof_jntid','dof_joint',int),('jnt_actgravcomp','actgravcomp',int),('actuator_ctrllimited','ctrllimited',wp.bool),
            ('actuator_forcelimited','forcelimited',wp.bool),('jnt_actfrclimited','jointlimited',wp.bool),
            ('actuator_gainprm','gainprm',types.vec10),('actuator_biasprm','biasprm',types.vec10),
            ('actuator_ctrlrange','ctrlrange',wp.vec2),('actuator_forcerange','forcerange',wp.vec2),('jnt_actfrcrange','jointrange',wp.vec2)):
        setattr(m,field,arr(c[key],dtype))
    # Unused state/dynamics inputs of the independent pinned comparator.
    m.actuator_ctrlspec=wp.zeros(m.nactuator,dtype=int);m.actuator_actnum=wp.zeros(m.nactuator,dtype=int)
    m.actuator_dynprm=wp.zeros((1,m.nactuator),dtype=types.vec10);m.actuator_actlimited=wp.zeros(m.nactuator,dtype=wp.bool)
    m.actuator_actrange=wp.zeros((1,m.nactuator),dtype=wp.vec2);m.actuator_actearly=wp.zeros(m.nactuator,dtype=wp.bool)
    m.actuator_gear=wp.zeros((1,m.nactuator),dtype=wp.spatial_vector);m.actuator_acc0=wp.zeros((1,m.nactuator),dtype=float)
    m.actuator_lengthrange=wp.zeros((1,m.nactuator),dtype=wp.vec2)
    d=NS(nworld=2)
    for field,key,dtype in (('ctrl','ctrl',float),('actuator_length','length',float),('actuator_velocity','velocity',float),
            ('moment_rownnz','moment_rownnz',int),('moment_rowadr','moment_rowadr',int),('moment_colind','moment_colind',int),('actuator_moment','moment',float),
            ('qfrc_gravcomp','gravcomp',float),('actuator_force','actuator_force',float),('qfrc_actuator','raw_qfrc',float)):
        setattr(d,field,arr(c[key],dtype))
    d.act=wp.empty((2,0),dtype=float);d.act_dot=wp.empty((2,0),dtype=float)
    d.qacc=wp.full((2,m.nv),float('nan'),dtype=float);d.qacc_warmstart=wp.full((2,m.nv),37.,dtype=float)
    return m,d


class Replay:
    def __init__(self,c,args,stream,phase):
        import warp as wp
        from mujoco_warp._src import forward,types
        from libero.gpu.host.runtime.dynamics.mjwarp_actuation import ActuatorForces
        self.case,self.stream,self.phase=c,stream,phase;self.model,self.data=m,d=construct(c,stream);self.outputs={};self.graphs={}
        names=('actuator_force',) if phase=='force' else ('actuator_force','qfrc_actuator') if phase=='composed' else ('qfrc_actuator',)
        for mode in ('upstream','cuda'):
            for key in names:setattr(d,key,wp.empty((2,m.nactuator if key=='actuator_force' else m.nv),dtype=float))
            self.outputs[mode]={k:getattr(d,k) for k in names}
            if mode=='cuda':self.binding=ActuatorForces(m,d,args.library,stream)
            with wp.ScopedCapture(stream=stream) as capture:
                if phase in ('force','composed'):
                    if mode=='cuda':self.binding.enqueue(0)
                    else:wp.launch(forward._actuator_force,dim=(2,m.nactuator),inputs=[m.na,m.opt.timestep,m.jnt_type,m.actuator_trntype,m.actuator_dyntype,m.actuator_gaintype,m.actuator_biastype,m.actuator_ctrladr,m.actuator_ctrlnum,m.actuator_ctrlspec,m.actuator_actadr,m.actuator_actnum,m.actuator_trnid,m.actuator_dynprm,m.actuator_gainprm,m.actuator_biasprm,m.actuator_actlimited,m.actuator_actrange,m.actuator_actearly,m.actuator_forcelimited,m.actuator_forcerange,m.actuator_ctrllimited,m.actuator_ctrlrange,m.actuator_gear,m.actuator_acc0,m.actuator_lengthrange,d.act,d.ctrl,d.actuator_length,d.actuator_velocity,m.opt.disableflags & types.DisableBit.CLAMPCTRL],outputs=[d.act_dot,d.actuator_force])
                if phase in ('projection','composed'):
                    if mode=='cuda':self.binding.enqueue(1)
                    else:
                        d.qfrc_actuator.zero_();wp.launch(forward._qfrc_actuator,dim=(2,m.nactuator),inputs=[d.moment_rownnz,d.moment_rowadr,d.moment_colind,d.actuator_moment,d.actuator_force],outputs=[d.qfrc_actuator])
                if phase in ('limits','composed'):
                    if mode=='cuda':self.binding.enqueue(2)
                    else:wp.launch(forward._qfrc_actuator_gravcomp_limits,dim=(2,m.nv),inputs=[m.jnt_actfrclimited,m.jnt_actgravcomp,m.jnt_actfrcrange,m.dof_jntid,d.qfrc_gravcomp,d.qfrc_actuator,c['gravity_enabled']],outputs=[d.qfrc_actuator])
            self.graphs[mode]=capture.graph

    def check(self,updated):
        import warp as wp
        c=current_case(self.case,updated);m,d=self.model,self.data
        for field,key in (('actuator_gainprm','gainprm'),('actuator_biasprm','biasprm'),('actuator_ctrlrange','ctrlrange'),('actuator_forcerange','forcerange'),('jnt_actfrcrange','jointrange')):getattr(m,field).assign(c[key])
        for field,key in (('ctrl','ctrl'),('actuator_length','length'),('actuator_velocity','velocity'),('moment_rownnz','moment_rownnz'),('moment_rowadr','moment_rowadr'),('moment_colind','moment_colind'),('actuator_moment','moment'),('qfrc_gravcomp','gravcomp')):getattr(d,field).assign(c[key])
        if 'actuator_force' not in self.outputs['cuda']:d.actuator_force.assign(c['actuator_force'])
        readonly={k:a for k,a in self.binding.arrays.items() if k not in ('force','qfrc')}
        readonly.update(qacc=d.qacc,qacc_warmstart=d.qacc_warmstart,act=d.act,act_dot=d.act_dot)
        if 'actuator_force' not in self.outputs['cuda']:readonly['force']=d.actuator_force
        if 'qfrc_actuator' not in self.outputs['cuda']:readonly['qfrc']=d.qfrc_actuator
        before={k:a.numpy().view(np.uint8).copy() for k,a in readonly.items()}
        expected,bounds=reference(c,self.phase);checks=[];values={}
        for mode,g in self.graphs.items():
            for key,a in self.outputs[mode].items():
                if self.phase=='limits' and key=='qfrc_actuator':a.assign(c['raw_qfrc'])
                else:a.fill_(37.)
            wp.capture_launch(g,stream=self.stream);wp.synchronize_stream(self.stream)
            values[mode]={k:a.numpy() for k,a in self.outputs[mode].items()}
            checks.extend(dict(mode=mode,**r) for r in check_arrays(values[mode],expected,bounds))
        held=all(np.array_equal(a.numpy().view(np.uint8),before[k]) for k,a in readonly.items())
        return dict(name=c['name'],phase=self.phase,updated=updated,inputs_held=bool(held),checks=checks,
            max_difference={k:float(abs(values['cuda'][k]-values['upstream'][k]).max()) for k in expected},passed=bool(held and all(r['passed'] for r in checks))),values


def signature(args):
    from libero.gpu.host.backends.mjwarp.build import source_files
    names=source_files(include_actuation=True);names={k:v for k,v in names.items() if k.startswith('upstream/') or 'mjwarp_actuation' in k or k in ('cmake/MjwarpActuation.cmake','host/runtime/dynamics/mjwarp_motion.py','data/mjwarp_source_pin.json','project_source_identity')}
    return dict(sources=names,tool_sha256=digest(__file__),reference_sha256=digest(Path(__file__).with_name('mjwarp_actuation_reference.py')),roundoff_helpers_sha256=digest(Path(__file__).with_name('mjwarp_motion_reference.py')),inputs=input_identity(args),library_sha256=digest(args.library))


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
    require_upstream();init_kernel_cache(args.build_root/'mjwarp/warp-cache');install_warp_guard();wp.init();stream=wp.Stream('cuda:0');sig=signature(args);prepared=args.output/'preparation.json'
    if not args.prepare:
        if not prepared.is_file() or json.loads(prepared.read_text())['signature']!=sig:raise RuntimeError('Prepare exact actuation operators offline first')
        receipt=json.loads((args.output/'cpu-oracle.json').read_text())
        if receipt['sha256']!=digest(args.output/'cpu-oracle.npz') or any(receipt[k]!=sig[k] for k in ('tool_sha256','reference_sha256','inputs')):raise RuntimeError('Prepare exact CPU actuation oracle first')
    with compilation_scope(args.prepare) as scope,wp.ScopedStream(stream):
        replays=[Replay(c,args,stream,p) for p in PHASES for c in cases(args)]
        if args.prepare:
            wp.synchronize_stream(stream);save_json(prepared,dict(signature=sig,compilation_events=scope['events'],physics_integrations=0,newton_solves=0));print(json.dumps(dict(prepared=True,captures=len(replays)*2,compilations=len(scope['events']))),flush=True);return 0
        rows=[];passed=True
        for replay in replays:
            if not passed:break
            for updated in (False,True):
                row,values=replay.check(updated);rows.append(row)
                np.savez(args.output/(row['name']+'-'+row['phase']+('-updated' if updated else '')+'-output.npz'),**{mode+'.'+k:v for mode,vv in values.items() for k,v in vv.items()})
                print(json.dumps({k:row[k] for k in ('name','phase','updated','passed')}),flush=True);passed &= row['passed']
        save_json(args.output/'summary.json',dict(passed=passed,signature=sig,cases=rows,runtime_compilations=scope['events'],newton_solves=0,physics_integrations=0,performance=None,
            policy='independent FP64 fixed/affine arithmetic, 1-Lipschitz clamping, absolute-contribution atomic sum bounds; fixed8x safety; exact input holds'))
        print(json.dumps(dict(passed=passed,cases=len(rows))),flush=True);return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())
