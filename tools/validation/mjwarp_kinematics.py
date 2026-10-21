"""Frozen N=2 rigid kinematics leaves and composition; no timing or dynamics."""
import argparse,hashlib,json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from .mjwarp_kinematics_reference import reference
from .mjwarp_motion_reference import check_arrays
PHASES=('matrices','inertial','geom','site','body','composed')
OUTPUTS=('xpos','xquat','xanchor','xaxis','xmat','xipos','ximat','geom_xpos','geom_xmat','site_xpos','site_xmat')
FIELDS=dict(qpos0='qpos0',body_pos='body_pos',body_quat='body_quat',jnt_pos='jnt_pos',jnt_axis='jnt_axis',
    body_ipos='body_ipos',body_iquat='body_iquat',geom_pos='geom_pos',geom_quat='geom_quat',site_pos='site_pos',site_quat='site_quat')


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def add_arguments(p):
    for name in ('build-root','inputs','output','library'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--cpu-reference',action='store_true')
    return p


def cases(args):
    result=[]
    for p in sorted(args.inputs.glob('*.npz')):
        if digest(p)!=json.loads(p.with_suffix('.json').read_text())['sha256']:raise RuntimeError('Frozen kinematics input changed')
        with np.load(p,allow_pickle=False) as f:c={k:f[k].copy() for k in f.files}
        for k in ('nq','nv','nbody','njnt','ngeom','nsite','nmocap'):c[k]=int(c[k])
        c['name']=p.stem;result.append(c)
    if not result:raise ValueError('No frozen kinematics inputs')
    return result


def current_case(original,updated):
    c={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in original.items()}
    if updated:
        c['qpos']*=np.float32(.91)
        for k in ('body_pos','body_ipos'):
            c[k][:,1:]+=np.array([.003,-.002,.004],np.float32)
        c['jnt_pos']+=np.array([.001,-.002,.003],np.float32)
        c['geom_pos']+=np.array([-.002,.003,.001],np.float32)
        c['site_pos']+=np.array([.003,.001,-.002],np.float32)
        c['mocap_pos']+=np.array([-.003,.002,.001],np.float32);c['mocap_quat']*=np.float32(1.07)
        for kind,q in zip(c['joint_type'],c['qposadr']):
            if kind>=2:c['qpos0'][:,q]+=np.float32(.019)
        # Independently updated frozen body frames exercise derived-only leaves.
        c['xpos'][:,1:]+=np.array([.004,-.003,.002],np.float32)
        c['xquat'][:,1:]*=np.float32(.97)
    return c


def input_identity(args):return {p.name:digest(p) for p in sorted(args.inputs.glob('*')) if p.is_file()}


def cpu_reference(args):
    import mujoco
    checks=[];excluded=[]
    for original in cases(args):
        receipt=json.loads((args.inputs/(original['name']+'.json')).read_text())
        if receipt.get('cpu_degenerate_excluded'):
            excluded.append(dict(name=original['name'],reason='pinned Warp all-zero quaternion fallback differs from CPU; GPU/formula checks still required'));continue
        if 'model_path' in receipt:
            if digest(receipt['model_path'])!=receipt['model_sha256']:raise RuntimeError('CPU kinematics model changed')
            m=mujoco.MjModel.from_binary_path(receipt['model_path'])
        else:m=mujoco.MjModel.from_xml_string(receipt['xml'])
        d=mujoco.MjData(m)
        # Actual CPU local2Global has compile-time sameframe shortcuts. Disable
        # them for the oracle when mutable local transforms are supplied.
        m.body_sameframe[:]=0;m.geom_sameframe[:]=0;m.site_sameframe[:]=0
        for updated in (False,True):
            c=current_case(original,updated)
            # CPU2.3.7 rotVecQuat assumes unit quaternions, whereas the pinned
            # Warp polynomial is homogeneous in q. The FP32-cast local model
            # quaternions need not be exactly unit. Qualify the shared ideal
            # rigid-frame mathematics on normalized FP64 parameters here;
            # raw FP32 inputs are kept unchanged in all GPU/formula checks.
            for field in ('body_quat','body_iquat','geom_quat','site_quat'):
                value=c[field].astype(np.float64);length=np.linalg.norm(value,axis=-1,keepdims=True)
                c[field]=value/np.where(length>0,length,1.)
            value=c['jnt_axis'].astype(np.float64);length=np.linalg.norm(value,axis=-1,keepdims=True)
            c['jnt_axis']=value/np.where(length>0,length,1.)
            expected,_=reference(c,'composed')
            for w in range(2):
                for field in FIELDS:getattr(m,field)[:]=c[field][w%len(c[field])]
                d.qpos[:]=c['qpos'][w];d.mocap_pos[:]=c['mocap_pos'][w];d.mocap_quat[:]=c['mocap_quat'][w]
                mujoco.mj_kinematics(m,d)
                for k in OUTPUTS:
                    actual=getattr(d,k).copy();target=expected[k][w]
                    if k.endswith('mat'):actual=actual.reshape(-1,3,3)
                    if k in ('xipos','ximat'):actual=actual[1:];target=target[1:] # CPU hardcodes world inertial frame
                    elif k.startswith('geom_'):
                        active=~((c['weld'][c['geom_body']]==0)&(c['mocap'][c['root'][c['geom_body']]]==-1));actual=actual[active];target=target[active]
                    error=float(abs(actual-target).max(initial=0.));limit=4096*np.finfo(float).eps*max(1.,float(abs(target).max(initial=0.)))
                    checks.append(dict(name=c['name'],updated=updated,world=w,field=k,error=error,limit=limit,passed=bool(error<=limit)))
        del d,m
    result=dict(passed=all(x['passed'] for x in checks),version=mujoco.__version__,checks=checks,excluded=excluded,inputs=input_identity(args),
        tool_sha256=digest(__file__),reference_sha256=digest(Path(__file__).with_name('mjwarp_kinematics_reference.py')),newton_solves=0,physics_integrations=0)
    (args.output/'cpu-oracle.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'],cpu_checks=len(checks))),flush=True);return 0 if result['passed'] else 2


def construct(c,stream):
    import warp as wp
    from mujoco_warp._src import types
    arr=lambda a,dtype:wp.array(np.ascontiguousarray(a),dtype=dtype,device=stream.device)
    m=NS(**{k:c[k] for k in ('nq','nv','nbody','njnt','ngeom','nsite','nmocap')},nbranch=len(c['starts'])-1,
        opt=NS(integrator=types.IntegratorType.EULER,enableflags=0))
    for field,key in (('body_parentid','parent'),('body_jntnum','body_jntnum'),('body_jntadr','body_jntadr'),('body_dofnum','body_dofnum'),('body_dofadr','body_dofadr'),
            ('jnt_type','joint_type'),('jnt_qposadr','qposadr'),('jnt_dofadr','dofadr'),('dof_bodyid','dof_body'),('body_mocapid','mocap'),('body_rootid','root'),
            ('body_weldid','weld'),('geom_bodyid','geom_body'),('site_bodyid','site_body'),('body_branches','branches'),('body_branch_start','starts')):
        setattr(m,field,arr(c[key],int))
    m.body_tree=tuple(arr(np.flatnonzero(c['depth']==i).astype(np.int32),int) for i in range(int(c['depth'].max())+1))
    for field in FIELDS:
        dtype=wp.float32 if field=='qpos0' else wp.quat if 'quat' in field else wp.vec3
        setattr(m,field,arr(c[field],dtype))
    d=NS(nworld=2)
    for field in ('qpos','mocap_pos','mocap_quat',*OUTPUTS):
        dtype=wp.float32 if field=='qpos' else wp.mat33 if field.endswith('mat') else wp.quat if 'quat' in field else wp.vec3
        setattr(d,field,arr(c[field],dtype))
    d.qvel=wp.full((2,m.nv),31.,dtype=float);d.qacc=wp.full((2,m.nv),float('nan'),dtype=float);d.qacc_warmstart=wp.full((2,m.nv),-19.,dtype=float)
    return m,d


def upstream(m,d,phase):
    import warp as wp
    from mujoco_warp._src import smooth
    if phase==0:wp.launch(smooth._kinematics_branch,dim=(2,m.nbranch),inputs=[m.qpos0,m.body_parentid,m.body_mocapid,m.body_jntnum,m.body_jntadr,m.body_pos,m.body_quat,m.jnt_type,m.jnt_qposadr,m.jnt_pos,m.jnt_axis,m.body_branches,m.body_branch_start,d.qpos,d.mocap_pos,d.mocap_quat],outputs=[d.xpos,d.xquat,d.xanchor,d.xaxis])
    elif phase==1:wp.launch(smooth._compute_body_matrices,dim=(2,m.nbody),inputs=[d.xquat],outputs=[d.xmat])
    elif phase==2:wp.launch(smooth._compute_body_inertial_frames,dim=(2,m.nbody),inputs=[m.body_ipos,m.body_iquat,d.xpos,d.xquat],outputs=[d.xipos,d.ximat])
    elif phase==3:wp.launch(smooth._geom_local_to_global,dim=(2,m.ngeom),inputs=[m.body_rootid,m.body_weldid,m.body_mocapid,m.geom_bodyid,m.geom_pos,m.geom_quat,d.xpos,d.xquat],outputs=[d.geom_xpos,d.geom_xmat])
    else:wp.launch(smooth._site_local_to_global,dim=(2,m.nsite),inputs=[m.site_bodyid,m.site_pos,m.site_quat,d.xpos,d.xquat],outputs=[d.site_xpos,d.site_xmat])


def check(outputs,values,errors):
    # Empty geometry/site arrays are a valid no-op domain.
    return check_arrays({k:v for k,v in outputs.items() if v.size},{k:v for k,v in values.items() if v.size},{k:v for k,v in errors.items() if v.size})


class Replay:
    def __init__(self,c,args,stream,phase):
        import warp as wp
        from libero.gpu.host.runtime.dynamics.mjwarp_kinematics import KinematicFrames
        self.case,self.stream,self.phase=c,stream,phase;self.models={};self.datas={};self.graphs={}
        phases=range(5) if phase=='composed' else ({'body':0,'matrices':1,'inertial':2,'geom':3,'site':4}[phase],)
        self.keys=tuple(reference(c,phase)[0])
        for mode in ('upstream','cuda'):
            m,d=construct(c,stream);self.models[mode]=m;self.datas[mode]=d
            if mode=='cuda':self.binding=KinematicFrames(m,d,args.library,stream)
            with wp.ScopedCapture(stream=stream) as cap:
                for p in phases:
                    if mode=='cuda':self.binding.enqueue(p)
                    else:upstream(m,d,p)
            self.graphs[mode]=cap.graph

    def check(self,updated):
        import warp as wp
        c=current_case(self.case,updated);expected,errors=reference(c,self.phase);checks=[];held=True;values={}
        for mode in ('upstream','cuda'):
            m,d=self.models[mode],self.datas[mode]
            for field in FIELDS:getattr(m,field).assign(c[field])
            for field in ('qpos','mocap_pos','mocap_quat',*OUTPUTS):getattr(d,field).assign(c[field])
            arrays={**{k:getattr(m,k) for k in FIELDS},**{k:getattr(d,k) for k in ('qpos','mocap_pos','mocap_quat','qvel','qacc','qacc_warmstart',*OUTPUTS) if k not in self.keys}}
            before={k:a.numpy().view(np.uint8).copy() for k,a in arrays.items()}
            wp.capture_launch(self.graphs[mode],stream=self.stream);wp.synchronize_stream(self.stream)
            values[mode]={k:getattr(d,k).numpy() for k in self.keys};checks += [dict(mode=mode,**x) for x in check(values[mode],expected,errors)]
            held &= bool(all(np.array_equal(a.numpy().view(np.uint8),before[k]) for k,a in arrays.items()))
            if 'xpos' in self.keys:
                for k in ('xpos','xquat'):held &= bool(np.array_equal(getattr(d,k).numpy()[:,0].view(np.uint8),c[k][:,0].view(np.uint8)))
            if 'geom_xpos' in self.keys:
                static=(c['weld'][c['geom_body']]==0)&(c['mocap'][c['root'][c['geom_body']]]==-1)
                for k in ('geom_xpos','geom_xmat'):held &= bool(np.array_equal(getattr(d,k).numpy()[:,static].view(np.uint8),c[k][:,static].view(np.uint8)))
        return dict(name=c['name'],phase=self.phase,updated=updated,inputs_and_holds=held,checks=checks,
            maximum_difference={k:float(abs(values['cuda'][k]-values['upstream'][k]).max(initial=0.)) for k in self.keys},passed=bool(held and all(x['passed'] for x in checks))),values


def signature(args):
    from libero.gpu.host.backends.mjwarp.build import source_files
    sources=source_files(include_kinematics=True)
    sources={k:v for k,v in sources.items() if k.startswith('upstream/') or 'mjwarp_kinematics' in k or k in ('cmake/MjwarpKinematics.cmake','host/runtime/dynamics/mjwarp_motion.py','data/mjwarp_source_pin.json','project_source_identity')}
    return dict(sources=sources,inputs=input_identity(args),library_sha256=digest(args.library),tool_sha256=digest(__file__),
        reference_sha256=digest(Path(__file__).with_name('mjwarp_kinematics_reference.py')),
        helpers={n:digest(Path(__file__).with_name(n)) for n in ('mjwarp_motion_reference.py','mjwarp_passive_reference.py')})


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
        if not prep.is_file() or json.loads(prep.read_text())['signature']!=sig:raise RuntimeError('Prepare exact kinematics operators offline first')
        cpu=json.loads((args.output/'cpu-oracle.json').read_text())
        if not cpu['passed'] or any(cpu[k]!=sig[k] for k in ('inputs','tool_sha256','reference_sha256')):raise RuntimeError('Prepare matching CPU kinematics oracle first')
    with compilation_scope(args.prepare) as scope,wp.ScopedStream(stream):
        replays=[Replay(c,args,stream,p) for p in PHASES for c in cases(args)]
        if args.prepare:
            wp.synchronize_stream(stream);save_json(prep,dict(signature=sig,compilation_events=scope['events'],physics_integrations=0,newton_solves=0))
            print(json.dumps(dict(prepared=True,captures=len(replays)*2,compilations=len(scope['events']))),flush=True);return 0
        rows=[]
        for replay in replays:
            for updated in (False,True):
                row,values=replay.check(updated);rows.append(row)
                np.savez(args.output/(row['name']+'-'+row['phase']+('-updated' if updated else '')+'-output.npz'),**{mode+'.'+k:v for mode,vv in values.items() for k,v in vv.items()})
                print(json.dumps({k:row[k] for k in ('name','phase','updated','passed')}),flush=True)
                if not row['passed']:break
            if not row['passed']:break
        passed=all(x['passed'] for x in rows)
        save_json(args.output/'summary.json',dict(passed=passed,signature=sig,cases=rows,runtime_compilations=scope['events'],physics_integrations=0,newton_solves=0,performance=None,worlds=2,
            policy='independent FP64 frame algebra with normwise FP32 quaternion/rotation/normalization and trig sensitivity bounds, fixed8x safety; exact input/world/static-geom/owner contracts'))
        print(json.dumps(dict(passed=passed,cases=len(rows))),flush=True);return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())
