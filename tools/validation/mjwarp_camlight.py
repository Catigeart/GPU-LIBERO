"""Frozen N=2 FIXED camera/light leaf checks; no timing, rendering or dynamics."""
import argparse,hashlib,json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from .mjwarp_camlight_reference import reference
from .mjwarp_motion_reference import check_arrays

FIELDS=('cam_pos','cam_quat','cam_poscom0','cam_pos0','cam_mat0','light_pos','light_dir','light_poscom0','light_pos0','light_dir0')
OUTPUTS=('cam_xpos','cam_xmat','light_xpos','light_xdir')
METADATA=('cam_mode','cam_bodyid','cam_targetbodyid','light_mode','light_bodyid','light_targetbodyid')


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def add_arguments(p):
    for name in ('build-root','inputs','output','library'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--cpu-reference',action='store_true')
    return p


def cases(args):
    result=[]
    for p in sorted(args.inputs.glob('*.npz')):
        if digest(p)!=json.loads(p.with_suffix('.json').read_text())['sha256']:raise RuntimeError('Frozen camera/light input changed')
        with np.load(p,allow_pickle=False) as f:c={k:f[k].copy() for k in f.files}
        for k in ('nbody','ncam','nlight'):c[k]=int(c[k])
        c['name']=p.stem;result.append(c)
    if not result:raise ValueError('No frozen camera/light inputs')
    return result


def current_case(original,updated):
    c={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in original.items()}
    if updated:
        c['cam_pos']+=np.array([.007,-.003,.011],np.float32)
        c['light_pos']+=np.array([-.005,.009,.003],np.float32)
        c['cam_quat']*=np.float32(.97)
        c['light_dir']=c['light_dir'][...,[2,0,1]].copy()*np.float32(1.13)
        c['xpos']+=np.array([-.004,.003,.008],np.float32)
        c['xquat']*=np.float32(1.03)
    return c


def input_identity(args):return {p.name:digest(p) for p in sorted(args.inputs.glob('*')) if p.is_file()}


def cpu_reference(args):
    import mujoco
    if mujoco.__version__!='2.3.7':raise RuntimeError('Camera/light CPU oracle requires reference2.3.7')
    rows=[];excluded=[]
    for original in cases(args):
        receipt=json.loads((args.inputs/(original['name']+'.json')).read_text())
        if not receipt['common_cpu']:
            excluded.append(dict(name=original['name'],reason=receipt['reason']));continue
        if 'model_path' in receipt:
            if digest(receipt['model_path'])!=receipt['model_sha256']:raise RuntimeError('Camera/light CPU model changed')
            m=mujoco.MjModel.from_binary_path(receipt['model_path'])
        else:m=mujoco.MjModel.from_xml_string(receipt['xml'])
        d=mujoco.MjData(m)
        for updated in (False,True):
            c=current_case(original,updated)
            # CPU rotVecQuat assumes unit quaternions; pinned rotation is a
            # homogeneous polynomial. Qualify their shared FP64 unit domain.
            # Raw FP32 GPU fixtures and their independent oracle stay unchanged.
            for key in ('xquat','cam_quat'):
                value=c[key].astype(float);length=np.linalg.norm(value,axis=-1,keepdims=True);c[key]=value/length
            expected,_=reference(c,'composed')
            for w in range(2):
                for key in FIELDS:
                    value=c[key][w%len(c[key])];getattr(m,key)[:]=value.reshape(-1,9) if key.endswith('mat0') else value
                d.xpos[:]=c['xpos'][w];d.xquat[:]=c['xquat'][w]
                # CPU mj_local2Global translates via d.xmat, unlike pinned
                # Warp's direct quaternion polynomial; supply the same frame.
                for b in range(m.nbody):mujoco.mju_quat2Mat(d.xmat[b],c['xquat'][w,b])
                mujoco.mj_camlight(m,d)
                for key in OUTPUTS:
                    value=getattr(d,key).copy()
                    if key=='cam_xmat':value=value.reshape(-1,3,3)
                    target=expected[key][w];error=float(abs(value-target).max(initial=0.));limit=4096*np.finfo(float).eps*max(1.,float(abs(target).max(initial=0.)))
                    rows.append(dict(name=c['name'],updated=updated,world=w,field=key,error=error,limit=limit,passed=bool(error<=limit)))
    degenerate=[]
    for value in ([0.,0.,0.],[0.,1.e-18,0.]):
        a=np.array(value);length=mujoco.mju_normalize3(a);degenerate.append(dict(input=value,output=a.tolist(),length=length))
    result=dict(passed=all(r['passed'] for r in rows),version=mujoco.__version__,checks=rows,excluded=excluded,normalize_probes=degenerate,
        inputs=input_identity(args),tool_sha256=digest(__file__),reference_sha256=digest(Path(__file__).with_name('mjwarp_camlight_reference.py')),
        physics_integrations=0,newton_solves=0)
    (args.output/'cpu-oracle.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'],cpu_checks=len(rows))),flush=True);return 0 if result['passed'] else 2


def construct(c,stream):
    import warp as wp
    from mujoco_warp._src import types
    arr=lambda value,dtype:wp.array(np.ascontiguousarray(value),dtype=dtype,device=stream.device)
    m=NS(nbody=c['nbody'],ncam=c['ncam'],nlight=c['nlight'],opt=NS(integrator=types.IntegratorType.EULER,enableflags=0))
    for key in METADATA:setattr(m,key,arr(c[key],int))
    for key in FIELDS:setattr(m,key,arr(c[key],wp.quat if key=='cam_quat' else wp.mat33 if key=='cam_mat0' else wp.vec3))
    d=NS(nworld=2)
    for key in ('xpos','xquat',*OUTPUTS):setattr(d,key,arr(c[key],wp.quat if key=='xquat' else wp.mat33 if key=='cam_xmat' else wp.vec3))
    d.subtree_com=wp.full((2,m.nbody),-17.,dtype=wp.vec3)
    for key in ('qpos','qvel','qacc','qacc_warmstart'):setattr(d,key,wp.full((2,3),31.,dtype=float))
    return m,d


def upstream(m,d,phase):
    import warp as wp
    from mujoco_warp._src import smooth
    if phase in ('cam','composed') and m.ncam:
        wp.launch(smooth._cam_local_to_global,dim=(2,m.ncam),inputs=[m.cam_mode,m.cam_bodyid,m.cam_targetbodyid,m.cam_pos,m.cam_quat,m.cam_poscom0,m.cam_pos0,m.cam_mat0,d.xpos,d.xquat,d.subtree_com],outputs=[d.cam_xpos,d.cam_xmat])
    if phase in ('light','composed') and m.nlight:
        wp.launch(smooth._light_local_to_global,dim=(2,m.nlight),inputs=[m.light_mode,m.light_bodyid,m.light_targetbodyid,m.light_pos,m.light_dir,m.light_poscom0,m.light_pos0,m.light_dir0,d.xpos,d.xquat,d.subtree_com],outputs=[d.light_xpos,d.light_xdir])


class Replay:
    def __init__(self,c,args,stream,phase):
        import warp as wp
        from libero.gpu.host.runtime.dynamics.mjwarp_camlight import CamLightFrames,qualify
        self.case,self.stream,self.phase=c,stream,phase;self.models={};self.datas={};self.graphs={};self.keys=tuple(reference(c,phase)[0])
        for mode in ('upstream','cuda'):
            m,d=construct(c,stream);self.models[mode]=m;self.datas[mode]=d
            if mode=='cuda':self.binding=CamLightFrames(m,d,args.library,stream,admission=qualify(m,d,implementation='cuda'))
            with wp.ScopedCapture(stream=stream) as cap:
                if mode=='cuda':
                    for p in (0,1) if phase=='composed' else (0,) if phase=='cam' else (1,):self.binding.enqueue(p)
                else:upstream(m,d,phase)
            self.graphs[mode]=cap.graph

    def check(self,updated):
        import warp as wp
        c=current_case(self.case,updated);expected,errors=reference(c,self.phase);checks=[];held=True;values={}
        for mode in ('upstream','cuda'):
            m,d=self.models[mode],self.datas[mode]
            for key in FIELDS:getattr(m,key).assign(c[key])
            for key in ('xpos','xquat',*OUTPUTS):getattr(d,key).assign(c[key])
            arrays={**{'model.'+k:v for k,v in vars(m).items() if hasattr(v,'numpy')},
                    **{'data.'+k:v for k,v in vars(d).items() if hasattr(v,'numpy') and k not in self.keys}}
            before={k:a.numpy().view(np.uint8).copy() for k,a in arrays.items()}
            wp.capture_launch(self.graphs[mode],stream=self.stream);wp.synchronize_stream(self.stream)
            values[mode]={k:getattr(d,k).numpy() for k in self.keys}
            checks += [dict(mode=mode,**x) for x in check_arrays({k:v for k,v in values[mode].items() if v.size},expected,errors)]
            held &= all(np.array_equal(a.numpy().view(np.uint8),before[k]) for k,a in arrays.items())
        return dict(name=c['name'],phase=self.phase,updated=updated,inputs_and_holds=bool(held),checks=checks,
            maximum_difference={k:float(abs(values['cuda'][k]-values['upstream'][k]).max(initial=0.)) for k in self.keys},passed=bool(held and all(x['passed'] for x in checks)))


def signature(args):
    from libero.gpu.host.backends.mjwarp.build import source_files
    import warp as wp
    sources=source_files(include_camlight=True)
    sources={k:v for k,v in sources.items() if k.startswith('upstream/') or 'mjwarp_camlight' in k or 'MjwarpCamLight' in k or k in ('host/runtime/dynamics/mjwarp_motion.py','data/mjwarp_source_pin.json','project_source_identity')}
    native=Path(wp.__file__).resolve().parent/'native'
    return dict(sources=sources,warp_native={k:digest(native/k) for k in ('vec.h','builtin.h')},inputs=input_identity(args),library_sha256=digest(args.library),tool_sha256=digest(__file__),
        reference_sha256=digest(Path(__file__).with_name('mjwarp_camlight_reference.py')),
        helpers={n:digest(Path(__file__).with_name(n)) for n in ('mjwarp_motion_reference.py','mjwarp_kinematics_reference.py')})


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
        if not prep.is_file() or json.loads(prep.read_text())['signature']!=sig:raise RuntimeError('Prepare exact camera/light operators offline first')
        cpu=json.loads((args.output/'cpu-oracle.json').read_text())
        if not cpu['passed'] or any(cpu[k]!=sig[k] for k in ('inputs','tool_sha256','reference_sha256')):raise RuntimeError('Prepare matching CPU camera/light oracle first')
    with compilation_scope(args.prepare) as scope,wp.ScopedStream(stream):
        replays=[Replay(c,args,stream,p) for p in ('cam','light','composed') for c in cases(args)]
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
            policy='independent FP64 homogeneous quaternion frames and vector normalization; normwise FP32 arithmetic/sensitivity, fixed8x safety; exact readonly/hold contracts'))
        print(json.dumps(dict(passed=passed,cases=len(rows))),flush=True);return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())
