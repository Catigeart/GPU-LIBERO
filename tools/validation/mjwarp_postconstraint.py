"""Frozen N=2 postconstraint RNE suffix; no Newton, trajectories or timing."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from .mjwarp_motion import TASKS, cases as motion_cases
from .mjwarp_motion_reference import check_arrays
from .mjwarp_postconstraint_reference import reference


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def add_arguments(p):
    for name in ('build-root','motion-inputs','output','library'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--cpu-reference',action='store_true')
    return p


def cases(args):
    result=motion_cases(args.motion_inputs)
    for c in result:
        idx=np.arange(c['nv'],dtype=np.float64)
        c['qacc']=np.stack((1.7*np.sin(idx*.3+.4),2.1*np.cos(idx*.6+.2))).astype(np.float32)
        idx=np.arange(c['nbody']*6,dtype=np.float64).reshape(c['nbody'],6)
        c['cfrc_ext']=np.stack((.7*np.cos(idx*.2+.3),1.9*np.sin(idx*.4+.7))).astype(np.float32)
        c['cfrc_ext'][:,0,:]=37.  # Public world wrench is ignored by RNE.
        if c['name']=='zero-gravity-off':c['qacc'][:]=0;c['cfrc_ext'][:]=0
    base=next(c for c in result if c['name']=='mixed-joints')
    for name in ('acceleration-only','external-only'):
        c={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in base.items()}
        c.update(name=name,gravity_disabled=True)
        for field in ('qvel','cvel','cdof_dot'):c[field][:]=0
        if name=='acceleration-only':c['cfrc_ext'][:]=0
        else:c['qacc'][:]=0
        result.append(c)
    return result


def current_case(original,updated):
    c={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in original.items()}
    if updated:
        c['qvel']*=np.array([.625,-1.25],np.float32)[:,None]
        c['qacc']*=np.array([.25,-1.5],np.float32)[:,None]
        c['cfrc_ext']*=np.array([-.75,1.25],np.float32)[:,None,None]
        c['cdof']*=np.array([1.125,.875],np.float32)[:,None,None]
        c['cinert']*=np.array([.75,1.5],np.float32)[:,None,None]
        c['gravity']*=np.float32(.8);c['cvel']*=np.float32(.75);c['cdof_dot']*=np.float32(1.5)
    return c


def input_identity(args):
    return {f'{s}-{t}':digest(args.motion_inputs/f'{s}-{t}.npz') for s,t in TASKS}


def cpu_reference(args):
    import mujoco
    if mujoco.__version__!='2.3.7':raise RuntimeError('CPU oracle requires MuJoCo 2.3.7')
    saved={};checks=[]
    for original in cases(args)[:len(TASKS)]:
        receipt=json.loads((args.motion_inputs/(original['name']+'.json')).read_text())
        if digest(original['model_path'])!=receipt['model_sha256']:raise RuntimeError('CPU model changed')
        m=mujoco.MjModel.from_binary_path(original['model_path']);d=mujoco.MjData(m)
        if not np.array_equal(m.body_parentid,original['parent']):raise RuntimeError('CPU body tree differs')
        for updated in (False,True):
            c=current_case(original,updated);expected,_=reference(c);outputs={k:[] for k in expected}
            ext=[]
            for w in range(2):
                for field in ('cdof','cdof_dot','cinert','qvel','qacc','cvel'):getattr(d,field)[:]=c[field][w]
                # With coincident application/subtree points, the CPU's actual
                # external-force builder publishes precisely our frozen wrench.
                d.subtree_com[:]=0;d.xipos[:]=0
                d.xfrc_applied[:,:3]=c['cfrc_ext'][w,:,3:]
                d.xfrc_applied[:,3:]=c['cfrc_ext'][w,:,:3]
                bit=int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)
                m.opt.disableflags=(m.opt.disableflags|bit) if c['gravity_disabled'] else (m.opt.disableflags&~bit)
                m.opt.gravity[:]=c['gravity'][w%len(c['gravity'])]
                mujoco.mj_rnePostConstraint(m,d)
                for key in expected:outputs[key].append(getattr(d,key).copy())
                ext.append(d.cfrc_ext.copy())
            target_ext=c['cfrc_ext'].copy();target_ext[:,0,:]=0
            if not np.array_equal(np.array(ext),target_ext):raise RuntimeError('CPU external wrench mapping differs')
            for key,values in outputs.items():
                values=np.array(values);target=expected[key];error=float(abs(values-target).max())
                limit=4096*np.finfo(float).eps*max(float(abs(target).max()),np.finfo(float).tiny)
                if error>limit:raise RuntimeError('FP64 postconstraint reference differs: '+c['name']+'/'+key)
                checks.append(dict(name=c['name'],updated=updated,field=key,error=error,limit=limit))
                saved[c['name']+('.updated.' if updated else '.')+key]=values
    p=args.output/'cpu-oracle.npz';np.savez(p,**saved)
    (args.output/'cpu-oracle.json').write_text(json.dumps(dict(sha256=digest(p),
        tool_sha256=digest(__file__),reference_sha256=digest(Path(__file__).with_name('mjwarp_postconstraint_reference.py')),
        motion_tool_sha256=digest(Path(__file__).with_name('mjwarp_motion.py')),
        motion_reference_sha256=digest(Path(__file__).with_name('mjwarp_motion_reference.py')),
        inputs=input_identity(args),version=mujoco.__version__,checks=checks,newton_solves=0,physics_integrations=0),indent=2)+'\n')
    print(json.dumps(dict(cpu_oracle=True,checks=len(checks),passed=True)),flush=True)
    return 0


class Replay:
    def __init__(self,c,args,stream):
        import warp as wp
        from mujoco_warp._src import smooth,types
        from libero.gpu.host.runtime.dynamics.mjwarp_motion import MotionTopology
        from libero.gpu.host.runtime.dynamics.mjwarp_postconstraint import PostconstraintRNE
        self.case,self.stream=c,stream
        arr=lambda a,dtype:wp.array(np.ascontiguousarray(a),dtype=dtype,device=stream.device)
        m=self.model=NS(nv=c['nv'],nbody=c['nbody'],njnt=c['njnt'],nbranch=len(c['starts'])-1)
        for field,key in (('body_parentid','parent'),('body_jntnum','jnum'),('body_jntadr','jadr'),('body_dofnum','dnum'),
                ('body_dofadr','dadr'),('jnt_type','joint'),('jnt_dofadr','joint_dof'),('dof_bodyid','dof_body'),
                ('body_branches','branches'),('body_branch_start','starts')):setattr(m,field,arr(c[key],int))
        m.body_tree=tuple(arr(c['layer'+str(i)],int) for i in range(sum(k.startswith('layer') for k in c)))
        m.opt=NS(disableflags=int(types.DisableBit.GRAVITY) if c['gravity_disabled'] else 0,gravity=arr(c['gravity'],wp.vec3))
        d=self.data=NS(nworld=2)
        for field,dtype in (('qvel',float),('qacc',float),('cdof',wp.spatial_vector),('cdof_dot',wp.spatial_vector),
                ('cinert',types.vec10),('cvel',wp.spatial_vector),('cfrc_ext',wp.spatial_vector)):
            setattr(d,field,arr(c[field],dtype))
        d.qfrc_bias=wp.full((2,m.nv),float('nan'),dtype=float,device=stream.device)
        d.qacc_warmstart=arr(np.arange(2*m.nv,dtype=np.float32).reshape(2,m.nv)*.25,float)
        self.topology=MotionTopology(m,d,stream);self.outputs={};self.graphs={}
        for mode in ('upstream','cuda'):
            for key in ('cacc','cfrc_int'):setattr(d,key,wp.empty((2,m.nbody),dtype=wp.spatial_vector,device=stream.device))
            self.outputs[mode]={k:getattr(d,k) for k in ('cacc','cfrc_int')}
            if mode=='cuda':self.binding=PostconstraintRNE(m,d,args.library,stream,self.topology)
            with wp.ScopedCapture(stream=stream) as capture:
                if mode=='cuda':self.binding.enqueue()
                else:
                    smooth._rne_cacc_world(m,d);smooth._rne_cacc_forward(m,d,flg_acc=True)
                    smooth._rne_cfrc(m,d,flg_cfrc_ext=True);smooth._rne_cfrc_backward(m,d)
            self.graphs[mode]=capture.graph

    def check(self,updated):
        import warp as wp
        c=current_case(self.case,updated);d=self.data
        readonly=dict(self.topology.arrays,gravity=self.model.opt.gravity,
            qfrc_bias=d.qfrc_bias,qacc_warmstart=d.qacc_warmstart)
        for k in ('qvel','qacc','cdof','cdof_dot','cinert','cvel','cfrc_ext'):
            getattr(d,k).assign(c[k]);readonly[k]=getattr(d,k)
        self.model.opt.gravity.assign(c['gravity'])
        before={k:a.numpy().view(np.uint8).copy() for k,a in readonly.items()}
        expected,bounds=reference(c);rows=[];values={}
        for mode,graph in self.graphs.items():
            for a in self.outputs[mode].values():a.fill_(37.)
            wp.capture_launch(graph,stream=self.stream);wp.synchronize_stream(self.stream)
            values[mode]={k:a.numpy() for k,a in self.outputs[mode].items()}
            rows.extend(dict(mode=mode,**r) for r in check_arrays(values[mode],expected,bounds))
        held=all(np.array_equal(a.numpy().view(np.uint8),before[k]) for k,a in readonly.items())
        return dict(name=c['name'],updated=updated,checks=rows,inputs_held=bool(held),
            max_difference={k:float(abs(values['cuda'][k]-values['upstream'][k]).max()) for k in expected},
            passed=bool(held and all(r['passed'] for r in rows))),values


def signature(args):
    from libero.gpu.host.backends.mjwarp.build import source_files
    from libero.gpu.host.paths import GPU_ROOT
    source=source_files(include_postconstraint=True)
    source={k:v for k,v in source.items() if k.startswith('upstream/') or k in (
        'project_source_identity','data/mjwarp_source_pin.json','host/backends/mjwarp/model.py',
        'host/backends/mjwarp/postconstraint.py','host/backends/mjwarp/postconstraint_external.py',
        'host/runtime/dynamics/mjwarp_motion.py','host/runtime/dynamics/mjwarp_postconstraint.py',
        'cuda/dynamics/mjwarp_postconstraint.cu','cuda/dynamics/mjwarp_postconstraint.cuh',
        'cuda/dynamics/mjwarp_spatial.cuh','cmake/MjwarpPostconstraint.cmake')}
    return dict(sources=source,tool_sha256=digest(__file__),
        reference_sha256=digest(Path(__file__).with_name('mjwarp_postconstraint_reference.py')),
        motion_tool_sha256=digest(Path(__file__).with_name('mjwarp_motion.py')),
        motion_reference_sha256=digest(Path(__file__).with_name('mjwarp_motion_reference.py')),
        inputs=input_identity(args),library_sha256=digest(args.library))


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
    from validation.runner import gpu_load_snapshot
    require_upstream();init_kernel_cache(args.build_root/'mjwarp/warp-cache');install_warp_guard();wp.init()
    stream=wp.Stream('cuda:0');sig=signature(args);prepared=args.output/'preparation.json'
    if not args.prepare:
        if not prepared.is_file() or json.loads(prepared.read_text())['signature']!=sig:
            raise RuntimeError('Prepare exact postconstraint operators offline first')
        receipt=json.loads((args.output/'cpu-oracle.json').read_text())
        keys=('tool_sha256','reference_sha256','motion_tool_sha256','motion_reference_sha256','inputs')
        if receipt['sha256']!=digest(args.output/'cpu-oracle.npz') or any(receipt[k]!=sig[k] for k in keys):
            raise RuntimeError('Prepare exact CPU postconstraint reference first')
    with compilation_scope(args.prepare) as scope,wp.ScopedStream(stream):
        replays=[Replay(c,args,stream) for c in cases(args)]
        if args.prepare:
            wp.synchronize_stream(stream)
            save_json(prepared,dict(signature=sig,compilation_events=scope['events'],physics_integrations=0,newton_solves=0))
            print(json.dumps(dict(prepared=True,cases=len(replays),compilations=len(scope['events']))),flush=True)
            return 0
        before=gpu_load_snapshot();rows=[]
        for replay in replays:
            for updated in (False,True):
                row,outputs=replay.check(updated);rows.append(row)
                np.savez(args.output/(row['name']+('-updated' if updated else '')+'-output.npz'),
                    **{mode+'.'+k:v for mode,out in outputs.items() for k,v in out.items()})
                print(json.dumps(dict(name=row['name'],updated=updated,passed=row['passed'])),flush=True)
        passed=all(r['passed'] for r in rows)
        save_json(args.output/'summary.json',dict(passed=passed,signature=sig,cases=rows,runtime_compilations=scope['events'],
            newton_solves=0,physics_integrations=0,performance=None,gpu_load_before=before,gpu_load_after=gpu_load_snapshot(),
            policy='independent FP64 spatial algebra; componentwise input-derived roundoff, 8x safety; exact readonly/ownership'))
        print(json.dumps(dict(passed=passed,cases=len(rows))),flush=True)
        return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())
