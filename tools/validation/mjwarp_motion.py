"""Frozen N=2 com_vel, bias RNE and composed checks; no timing/trajectories."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from .mjwarp_motion_reference import reference, velocity, check_arrays

TASKS = (('libero_10',0), ('libero_10',3), ('libero_spatial',0))


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def add_arguments(p):
    for name in ('build-root','kinematics','output','velocity-library'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--bias-library',type=Path)
    p.add_argument('--phase',choices=('com_vel','bias','composed'),default='com_vel')
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--cpu-reference',action='store_true')
    return p


def freeze(args,stream):
    import warp as wp
    import mujoco_warp as mjw
    from mujoco_warp._src import smooth,types
    from libero.libero import get_libero_path
    from libero.gpu.host.backends.mjwarp.model import build_scene,capacity_budget
    for suite,task in TASKS:
        name=f'{suite}-{task}';path=args.output/(name+'.npz')
        if path.exists():
            if json.loads(path.with_suffix('.json').read_text())['sha256']!=digest(path):
                raise RuntimeError('Frozen motion input changed')
            continue
        source=args.kinematics/(name+'.npz');receipt=json.loads(source.with_suffix('.json').read_text())
        if receipt['sha256']!=digest(source):raise RuntimeError('Frozen kinematics changed')
        with np.load(source,allow_pickle=False) as f:kin={k:f[k].copy() for k in f.files}
        catalogue=json.loads((args.build_root/'TASK_INSTANCES.json').read_text())['tasks']
        taskrow=next(t for t in catalogue if t['suite']==suite and t['task_id']==task)
        built,m,_,_,identity=build_scene(args.build_root,str(Path(get_libero_path('bddl_files'))/taskrow['key']),0,
            dict(use_camera_obs=False,has_renderer=False,has_offscreen_renderer=False,hard_reset=False,ignore_done=True))
        if identity!=receipt['identity'] or m.nv!=kin['nv'] or m.nbody!=kin['nbody']:
            raise RuntimeError('Motion model differs from frozen kinematics')
        budget=capacity_budget(built.native_model)
        d=mjw.make_data(built.native_model,nworld=2,nccdmax=budget['nconmax'],**budget)
        # Nonzero distinct velocities exercise all cross terms; no position or
        # collision computation/solver/integration is needed at this boundary.
        idx=np.arange(m.nv,dtype=np.float64)
        qvel=np.stack((.35*np.sin(idx+.2)+.12,.6*np.cos(idx*.7+.4)-.2)).astype(np.float32)
        d.qvel.assign(qvel);d.cdof.assign(kin['cdof']);d.cinert.assign(kin['cinert'])
        smooth.com_vel(m,d);smooth.rne(m,d);wp.synchronize_stream(stream)
        c=dict(nv=np.array(m.nv),nbody=np.array(m.nbody),njnt=np.array(m.njnt),parent=m.body_parentid.numpy(),
            jnum=m.body_jntnum.numpy(),jadr=m.body_jntadr.numpy(),dnum=m.body_dofnum.numpy(),dadr=m.body_dofadr.numpy(),
            joint=m.jnt_type.numpy(),joint_dof=m.jnt_dofadr.numpy(),dof_body=m.dof_bodyid.numpy(),
            branches=m.body_branches.numpy(),starts=m.body_branch_start.numpy(),qvel=qvel,
            cdof=kin['cdof'],cinert=kin['cinert'],gravity=m.opt.gravity.numpy(),
            gravity_disabled=np.array(bool(m.opt.disableflags & types.DisableBit.GRAVITY)))
        for field in ('cvel','cdof_dot','cacc','cfrc_int','qfrc_bias'):c[field]=getattr(d,field).numpy()
        for i,layer in enumerate(m.body_tree):c['layer'+str(i)]=layer.numpy()
        np.savez(path,**c)
        path.with_suffix('.json').write_text(json.dumps(dict(sha256=digest(path),identity=identity,
            model_path=receipt['model_path'],model_sha256=receipt['model_sha256'],kinematics_sha256=digest(source),
            physics_integrations=0,newton_solves=0),indent=2)+'\n')
        print(json.dumps(dict(frozen=name,nv=m.nv,nbody=m.nbody,branches=m.nbranch)),flush=True)


def real_cases(output):
    result=[]
    for suite,task in TASKS:
        name=f'{suite}-{task}';p=output/(name+'.npz');receipt=json.loads(p.with_suffix('.json').read_text())
        if receipt['sha256']!=digest(p):raise RuntimeError('Frozen motion input changed')
        with np.load(p,allow_pickle=False) as f:c={k:f[k].copy() for k in f.files}
        for key in ('nv','nbody','njnt'):c[key]=int(c[key])
        c['gravity_disabled']=bool(c['gravity_disabled']);c.update(name=name,model_path=receipt['model_path'])
        result.append(c)
    return result


def synthetic(parent,joints,name):
    rng=np.random.default_rng(381+len(parent));parent=np.array(parent,np.int32);nb=len(parent)
    types=[];joint_dof=[];dof_body=[];jnum=[];jadr=[];dnum=[];dadr=[];widths=(6,3,1,1)
    for b,js in enumerate(joints):
        jnum.append(len(js));jadr.append(len(types) if js else -1);dadr.append(len(dof_body) if js else -1)
        count=0
        for j in js:
            joint_dof.append(len(dof_body));types.append(j);count+=widths[j];dof_body.extend([b]*widths[j])
        dnum.append(count)
    nv=len(dof_body);flat=[];starts=[0];depth=np.zeros(nb,int)
    for b in range(1,nb):depth[b]=depth[parent[b]]+1
    child_count=np.bincount(parent[1:],minlength=nb)
    for leaf in np.flatnonzero(child_count[1:]==0)+1:
        chain=[];b=int(leaf)
        while b:chain.append(b);b=int(parent[b])
        flat.extend(chain[::-1]);starts.append(len(flat))
    c=dict(name=name,nv=nv,nbody=nb,njnt=len(types),parent=parent,gravity_disabled=False,
        qvel=rng.normal(scale=.4,size=(2,nv)).astype(np.float32),cdof=rng.normal(scale=.3,size=(2,nv,6)).astype(np.float32),
        cinert=rng.normal(scale=.02,size=(2,nb,10)).astype(np.float32),
        gravity=np.array([[0,0,-9.81],[1.25,-.75,-8.4]],np.float32))
    for key,value in (('jnum',jnum),('jadr',jadr),('dnum',dnum),('dadr',dadr),('joint',types),
            ('joint_dof',joint_dof),('dof_body',dof_body),('branches',flat),('starts',starts)):
        c[key]=np.array(value,np.int32)
    for level in range(int(depth.max())+1):c['layer'+str(level)]=np.flatnonzero(depth==level).astype(np.int32)
    for w in range(2):
        c['cinert'][w,:,:3]=rng.uniform(.1,.6,(nb,3));c['cinert'][w,:,9]=rng.uniform(.5,2,nb)
        c['cinert'][w]*=1.e-3 if w==0 else 1.e3
    motion,_=velocity(c)
    c.update({k:v.astype(np.float32) for k,v in motion.items()})
    return c


def cases(output):
    real=real_cases(output)
    mixed=synthetic([0,0,1,2,2,0],[[],[0],[1],[2,3],[],[3]],'mixed-joints')
    wide=synthetic([0,0]+[1]*36,[[]]+[[3]]+[[3,2] if b<22 else [3] for b in range(2,38)],'wide-fanin')
    deep=synthetic([0,0]+list(range(1,13))+[1]*24,[[]]+[[0]]+[[3,2] if b<17 else [3] for b in range(2,38)],'deep-branching')
    independent=synthetic([0,0,1,2,0,4,4,0,7,8],[[],[0],[1],[3],[0],[],[2],[0],[1],[3]],'independent-trees')
    result=real+[mixed,wide,deep,independent]
    for name,zero,disabled in (('zero-velocity',True,False),('zero-gravity-off',True,True),('nonzero-gravity-off',False,True)):
        c={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in mixed.items()};c.update(name=name,gravity_disabled=disabled)
        if zero:c['qvel'][:]=0;c['cvel'][:]=0;c['cdof_dot'][:]=0
        result.append(c)
    return result


def cpu_reference(args):
    import mujoco
    if mujoco.__version__!='2.3.7':raise RuntimeError('CPU oracle requires MuJoCo 2.3.7')
    saved={};checks=[]
    for c in real_cases(args.output):
        receipt=json.loads((args.output/(c['name']+'.json')).read_text())
        if digest(c['model_path'])!=receipt['model_sha256']:raise RuntimeError('CPU reference model changed')
        m=mujoco.MjModel.from_binary_path(c['model_path']);d=mujoco.MjData(m)
        for field,key in (('body_parentid','parent'),('body_jntnum','jnum'),('body_jntadr','jadr'),
                ('body_dofnum','dnum'),('body_dofadr','dadr'),('jnt_type','joint'),('jnt_dofadr','joint_dof'),('dof_bodyid','dof_body')):
            if not np.array_equal(getattr(m,field),c[key]):raise RuntimeError('CPU motion topology differs: '+field)
        outs={k:[] for k in ('cvel','cdof_dot','bias','composed')}
        gravity_bit=int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)
        m.opt.disableflags=(m.opt.disableflags | gravity_bit) if c['gravity_disabled'] else (m.opt.disableflags & ~gravity_bit)
        for w in range(2):
            d.cdof[:]=c['cdof'][w];d.cinert[:]=c['cinert'][w];d.qvel[:]=c['qvel'][w]
            m.opt.gravity[:]=c['gravity'][w%len(c['gravity'])];mujoco.mj_comVel(m,d)
            outs['cvel'].append(d.cvel.copy());outs['cdof_dot'].append(d.cdof_dot.copy())
            result=np.empty(m.nv);mujoco.mj_rne(m,d,0,result);outs['composed'].append(result.copy())
            d.cvel[:]=c['cvel'][w];d.cdof_dot[:]=c['cdof_dot'][w]
            mujoco.mj_rne(m,d,0,result);outs['bias'].append(result.copy())
        vr,_=reference(c,'com_vel');br,_=reference(c,'bias');cr,_=reference(c,'composed')
        expected=vr|{'bias':br['qfrc_bias'],'composed':cr['qfrc_bias']}
        for key,values in outs.items():
            values=np.array(values);target=expected[key];error=float(abs(values-target).max())
            limit=1024*np.finfo(float).eps*max(float(abs(target).max()),np.finfo(float).tiny)
            if error>limit:raise RuntimeError('FP64 motion reference differs from CPU: '+c['name']+'/'+key)
            checks.append(dict(name=c['name']+'/'+key,error=error,limit=limit));saved[c['name']+'.'+key]=values
    p=args.output/'cpu-oracle.npz';np.savez(p,**saved)
    (args.output/'cpu-oracle.json').write_text(json.dumps(dict(sha256=digest(p),tool_sha256=digest(__file__),
        reference_sha256=digest(Path(__file__).with_name('mjwarp_motion_reference.py')),version=mujoco.__version__,
        inputs={c['name']:digest(args.output/(c['name']+'.npz')) for c in real_cases(args.output)},checks=checks,
        newton_solves=0,physics_integrations=0),indent=2)+'\n')
    print(json.dumps(dict(cpu_oracle=True,checks=len(checks),passed=True)),flush=True);return 0


class Replay:
    def __init__(self,c,args,stream):
        import warp as wp
        from mujoco_warp._src import types
        from libero.gpu.host.runtime.dynamics.mjwarp_motion import MotionTopology
        from libero.gpu.host.runtime.dynamics.mjwarp_velocity import ComVelocity
        self.case,self.stream,self.phase=c,stream,args.phase
        arr=lambda a,dtype:wp.array(np.ascontiguousarray(a),dtype=dtype,device=stream.device)
        m=self.model=NS(nv=c['nv'],nbody=c['nbody'],njnt=c['njnt'],nbranch=len(c['starts'])-1)
        for field,key in (('body_parentid','parent'),('body_jntnum','jnum'),('body_jntadr','jadr'),('body_dofnum','dnum'),
                ('body_dofadr','dadr'),('jnt_type','joint'),('jnt_dofadr','joint_dof'),('dof_bodyid','dof_body'),
                ('body_branches','branches'),('body_branch_start','starts')):setattr(m,field,arr(c[key],int))
        m.body_tree=tuple(arr(c['layer'+str(i)],int) for i in range(sum(k.startswith('layer') for k in c)))
        m.opt=NS(disableflags=int(types.DisableBit.GRAVITY) if c['gravity_disabled'] else 0,gravity=arr(c['gravity'],wp.vec3))
        d=self.data=NS(nworld=2,qvel=arr(c['qvel'],float),cdof=arr(c['cdof'],wp.spatial_vector),
            cinert=arr(c['cinert'],types.vec10),qacc=wp.full((2,m.nv),float('nan'),dtype=float,device=stream.device),
            cfrc_ext=wp.full((2,m.nbody),37.,dtype=wp.spatial_vector,device=stream.device))
        self.topology=MotionTopology(m,d,stream);self.outputs={};self.graphs={};self.bindings={}
        for mode in ('upstream','cuda'):
            for key,width in (('cvel',m.nbody),('cdof_dot',m.nv),('cacc',m.nbody),('cfrc_int',m.nbody)):
                setattr(d,key,wp.empty((2,width),dtype=wp.spatial_vector,device=stream.device))
            d.qfrc_bias=wp.empty((2,m.nv),dtype=float,device=stream.device)
            names=('cvel','cdof_dot') if args.phase=='com_vel' else ('cacc','cfrc_int','qfrc_bias') if args.phase=='bias' else ('cvel','cdof_dot','cacc','cfrc_int','qfrc_bias')
            self.outputs[mode]={k:getattr(d,k) for k in names}
            if args.phase=='bias':d.cvel.assign(c['cvel']);d.cdof_dot.assign(c['cdof_dot'])
            if mode=='cuda':
                velocity_binding=ComVelocity(m,d,args.velocity_library,stream,self.topology) if args.phase!='bias' else None
                bias_binding=None
                if args.phase!='com_vel':
                    from libero.gpu.host.runtime.dynamics.mjwarp_bias import Bias
                    bias_binding=Bias(m,d,args.bias_library,stream,self.topology)
                self.bindings[mode]=(velocity_binding,bias_binding)
            # Retain per-mode fixed bias inputs, not just the final Data aliases.
            self.bindings['inputs-'+mode]=(d.cvel,d.cdof_dot)
            with wp.ScopedCapture(stream=stream) as capture:
                from mujoco_warp._src import smooth
                if args.phase!='bias':
                    if mode=='cuda':velocity_binding.enqueue()
                    else:smooth.com_vel(m,d)
                if args.phase!='com_vel':
                    if mode=='cuda':bias_binding.enqueue()
                    else:smooth.rne(m,d)
            self.graphs[mode]=capture.graph

    def check(self,updated):
        import warp as wp
        c={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in self.case.items()}
        if updated:
            c['qvel']*=np.array([.625,-1.25],np.float32)[:,None]
            c['cdof']*=np.array([1.125,.875],np.float32)[:,None,None]
            c['cinert']*=np.array([.75,1.5],np.float32)[:,None,None]
            c['gravity']*=np.float32(.8);c['cvel']*=np.float32(.75);c['cdof_dot']*=np.float32(1.5)
        d,m=self.data,self.model
        for k in ('qvel','cdof','cinert'):getattr(d,k).assign(c[k])
        m.opt.gravity.assign(c['gravity'])
        readonly=dict(self.topology.arrays,qvel=d.qvel,cdof=d.cdof,cinert=d.cinert,gravity=m.opt.gravity,
                      unused_qacc=d.qacc,unused_external=d.cfrc_ext)
        inputs={k:a.numpy().view(np.uint8).copy() for k,a in readonly.items()}
        expected,bounds=reference(c,self.phase);rows=[];values={};held=True
        for mode,graph in self.graphs.items():
            for a in self.outputs[mode].values():a.fill_(37.)
            if self.phase=='bias':
                v,dot=self.bindings['inputs-'+mode];v.assign(c['cvel']);dot.assign(c['cdof_dot'])
            wp.capture_launch(graph,stream=self.stream);wp.synchronize_stream(self.stream)
            values[mode]={k:a.numpy() for k,a in self.outputs[mode].items()}
            checks=check_arrays(values[mode],expected,bounds)
            rows.extend(dict(mode=mode,**r) for r in checks)
            if self.phase=='bias':held &= np.array_equal(v.numpy(),c['cvel']) and np.array_equal(dot.numpy(),c['cdof_dot'])
        held &= all(np.array_equal(a.numpy().view(np.uint8),inputs[k]) for k,a in readonly.items())
        return dict(name=c['name'],updated=updated,checks=rows,inputs_held=bool(held),
            max_difference={k:float(abs(values['cuda'][k]-values['upstream'][k]).max()) for k in values['cuda']},
            passed=bool(held and all(r['passed'] for r in rows))),values


def leaf_sources(phase):
    from libero.gpu.host.backends.mjwarp.build import source_files
    from libero.gpu.host.paths import GPU_ROOT
    source=source_files()
    source={k:v for k,v in source.items() if k.startswith('upstream/') or k in ('host/backends/mjwarp/model.py','project_source_identity','data/mjwarp_source_pin.json')}
    names=['host/runtime/dynamics/mjwarp_motion.py','cuda/dynamics/mjwarp_spatial.cuh']
    for part in (('velocity',) if phase=='com_vel' else ('bias',) if phase=='bias' else ('velocity','bias')):
        names.extend((f'cuda/dynamics/mjwarp_{part}.cu',f'cuda/dynamics/mjwarp_{part}.cuh',
            f'host/runtime/dynamics/mjwarp_{part}.py',f'cmake/Mjwarp{part.title()}.cmake'))
    return source|{name:digest(GPU_ROOT/name) for name in names}


def main(argv=None,*,args=None):
    if args is None:args=add_arguments(argparse.ArgumentParser(description=__doc__)).parse_args(argv)
    if args.cpu_reference:return cpu_reference(args)
    if args.phase!='com_vel' and (args.bias_library is None or not args.bias_library.is_file()):raise ValueError('Build/provide bias library offline')
    from libero.gpu import precision
    precision.select('fp32')
    import warp as wp
    from warp._src.build import init_kernel_cache
    from libero.gpu.host.backends.mjwarp.model import require_upstream
    from libero.gpu.host.build.compilation import compilation_scope,install_warp_guard
    from validation.common import save_json
    from validation.runner import gpu_load_snapshot
    require_upstream();init_kernel_cache(args.build_root/'mjwarp/warp-cache');install_warp_guard();wp.init()
    args.output.mkdir(parents=True,exist_ok=True);stream=wp.Stream('cuda:0')
    with compilation_scope(args.prepare) as scope,wp.ScopedStream(stream):
        if args.prepare:freeze(args,stream)
        signature=dict(sources=leaf_sources(args.phase),tool_sha256=digest(__file__),
            reference_sha256=digest(Path(__file__).with_name('mjwarp_motion_reference.py')),
            libraries={k:digest(v) for k,v in (('velocity',args.velocity_library),('bias',args.bias_library)) if v is not None and (k=='velocity' or args.phase!='com_vel')},
            inputs={f'{s}-{t}':digest(args.output/f'{s}-{t}.npz') for s,t in TASKS})
        prepared=args.output/(args.phase+'-preparation.json')
        if not args.prepare and (not prepared.exists() or json.loads(prepared.read_text())['signature']!=signature):raise RuntimeError('Prepare exact motion operators offline first')
        selected=real_cases(args.output) if args.phase=='composed' else cases(args.output)
        replays=[Replay(c,args,stream) for c in selected]
        if args.prepare:
            wp.synchronize_stream(stream);save_json(prepared,dict(signature=signature,compilation_events=scope['events'],physics_integrations=0,newton_solves=0))
            print(json.dumps(dict(prepared=True,phase=args.phase,cases=len(replays),compilations=len(scope['events']))),flush=True);return 0
        oracle=args.output/'cpu-oracle.npz';receipt=json.loads((args.output/'cpu-oracle.json').read_text())
        if (receipt['sha256']!=digest(oracle) or receipt['tool_sha256']!=signature['tool_sha256'] or
                receipt['reference_sha256']!=signature['reference_sha256'] or receipt['inputs']!=signature['inputs']):raise RuntimeError('Prepare exact CPU motion reference first')
        if args.phase!='com_vel':
            previous=json.loads((args.output/'com_vel-summary.json').read_text())
            prior_signature=signature|dict(sources=leaf_sources('com_vel'),libraries={'velocity':digest(args.velocity_library)})
            if not previous['passed'] or previous['signature']!=prior_signature:raise RuntimeError('Qualify exact com_vel on these inputs first')
        if args.phase=='composed':
            previous=json.loads((args.output/'bias-summary.json').read_text())
            if not previous['passed'] or previous['signature']!=(signature|dict(sources=leaf_sources('bias'))):raise RuntimeError('Qualify exact bias first')
        before=gpu_load_snapshot();rows=[]
        for r in replays:
            for updated in (False,True):
                row,outputs=r.check(updated);rows.append(row)
                np.savez(args.output/(row['name']+'-'+args.phase+('-updated' if updated else '')+'-output.npz'),**{mode+'.'+k:v for mode,out in outputs.items() for k,v in out.items()})
                print(json.dumps(dict(phase=args.phase,name=row['name'],updated=updated,passed=row['passed'])),flush=True)
        passed=all(r['passed'] for r in rows)
        save_json(args.output/(args.phase+'-summary.json'),dict(passed=passed,signature=signature,cases=rows,runtime_compilations=scope['events'],
            newton_solves=0,physics_integrations=0,performance=None,gpu_load_before=before,gpu_load_after=gpu_load_snapshot(),
            policy='independent FP64 spatial algebra; propagated componentwise roundoff with input majorants; 8x safety; exact readonly/ownership contracts'))
        print(json.dumps(dict(passed=passed,phase=args.phase,cases=len(rows))),flush=True);return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())
