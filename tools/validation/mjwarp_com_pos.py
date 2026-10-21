"""Frozen N=2 subtree CoM / inertia / motion-axis checks; no timing or dynamics."""
import argparse, hashlib, json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from .mjwarp_com_pos_reference import reference
from .mjwarp_motion_reference import check_arrays

MODEL_FIELDS = ('body_mass','body_subtreemass','body_inertia')
FRAME_FIELDS = ('xipos','ximat','xmat','xanchor','xaxis')
OUTPUTS = ('subtree_com','cinert','cdof')
PHASES = ('subtree','inertia','axes','composed')


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def add_arguments(p):
    for name in ('build-root','inputs','output','library'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--matrix-library',type=Path)
    p.add_argument('--prepare',action='store_true');p.add_argument('--cpu-reference',action='store_true')
    return p


def cases(args):
    result=[]
    for p in sorted(args.inputs.glob('*.npz')):
        if digest(p)!=json.loads(p.with_suffix('.json').read_text())['sha256']:raise RuntimeError('Frozen CoM input changed')
        with np.load(p,allow_pickle=False) as f:c={k:f[k].copy() for k in f.files}
        for k in ('nq','nv','nbody','njnt','ngeom','nsite','nmocap','nC'):c[k]=int(c[k])
        c['name']=p.stem;result.append(c)
    if not result:raise ValueError('No frozen CoM inputs')
    return result


def current_case(original,updated):
    c={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in original.items()}
    if updated:
        c['body_mass']*=np.float32(1.11);c['body_subtreemass']*=np.float32(1.11);c['body_inertia']*=np.float32(.93)
        # Coherent mass scaling keeps supplied subtree masses valid where they
        # were valid originally; explicitly zero supplied masses stay zero.
        c['xipos']+=np.array([.013,-.007,.011],np.float32)
        c['xanchor']+=np.array([-.003,.009,-.005],np.float32)
        c['ximat']=c['ximat'][...,[1,2,0]].copy()
        c['xmat']=c['xmat'][...,[2,0,1]].copy()
        # Proper cyclic frame rotations also exercise current borrowed matrices.
        c['xaxis']*=np.float32(.97)
    return c


def input_identity(args):return {p.name:digest(p) for p in sorted(args.inputs.glob('*')) if p.is_file()}


def cpu_reference(args):
    import mujoco
    if mujoco.__version__!='2.3.7':raise RuntimeError('CoM CPU oracle requires CPU reference 2.3.7')
    rows=[];excluded=[]
    for original in cases(args):
        receipt=json.loads((args.inputs/(original['name']+'.json')).read_text())
        if not receipt['common_cpu']:
            excluded.append(dict(name=original['name'],reason='pinned zero/tiny supplied subtree mass or nonzero-world behavior differs; GPU/formula checks required'));continue
        if 'model_path' in receipt:
            if digest(receipt['model_path'])!=receipt['model_sha256']:raise RuntimeError('CoM CPU model changed')
            m=mujoco.MjModel.from_binary_path(receipt['model_path'])
        else:m=mujoco.MjModel.from_xml_string(receipt['xml'])
        d=mujoco.MjData(m)
        for updated in (False,True):
            c=current_case(original,updated)
            # CPU recomputes subtree mass in FP64; put that exact mathematical
            # denominator into the FP64 oracle, leaving raw FP32 GPU fixtures intact.
            mass=np.vstack([c['body_mass'][w % len(c['body_mass'])].astype(float) for w in range(2)])
            for b in range(m.nbody-1,0,-1):mass[:,c['parent'][b]]+=mass[:,b]
            common=dict(c,body_subtreemass=mass)
            expected,_=reference(common,'composed')
            for w in range(2):
                for field in MODEL_FIELDS:getattr(m,field)[:]=c[field][w % len(c[field])]
                for field in FRAME_FIELDS:
                    value=c[field][w]
                    getattr(d,field)[:]=value.reshape(-1,9) if field.endswith('mat') else value
                mujoco.mj_comPos(m,d)
                for field in OUTPUTS:
                    value=getattr(d,field).copy();target=expected[field][w]
                    if field=='subtree_com':
                        active=mass[w]>=mujoco.mjMINVAL
                        value=value[active];target=target[active]
                        if not np.all(active):
                            excluded.append(dict(name=c['name'],updated=updated,world=w,field=field,bodies=np.flatnonzero(~active).tolist(),
                                reason='CPU substitutes xipos for zero-mass leaves; pinned GPU retains the weighted sum'))
                    if field=='cinert':value=value[1:];target=target[1:] # CPU explicitly zeros world.
                    error=float(abs(value-target).max(initial=0.))
                    # Independent FP64 implementations: CPU's reciprocal-first
                    # subtree division and matrix association differ harmlessly.
                    limit=4096*np.finfo(float).eps*max(1.,float(abs(target).max(initial=0.)))
                    rows.append(dict(name=c['name'],updated=updated,world=w,field=field,error=error,limit=limit,passed=bool(error<=limit)))
    result=dict(passed=all(r['passed'] for r in rows),version=mujoco.__version__,checks=rows,excluded=excluded,
        inputs=input_identity(args),tool_sha256=digest(__file__),reference_sha256=digest(Path(__file__).with_name('mjwarp_com_pos_reference.py')),
        physics_integrations=0,newton_solves=0)
    (args.output/'cpu-oracle.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'],cpu_checks=len(rows))),flush=True)
    return 0 if result['passed'] else 2


def construct(c,stream):
    import warp as wp
    from mujoco_warp._src import types
    arr=lambda value,dtype:wp.array(np.ascontiguousarray(value),dtype=dtype,device=stream.device)
    m=NS(nv=c['nv'],nbody=c['nbody'],njnt=c['njnt'],nC=c['nC'],opt=NS(integrator=types.IntegratorType.EULER,enableflags=0))
    for field,key in (('body_parentid','parent'),('body_rootid','root'),('jnt_type','joint_type'),('jnt_dofadr','dofadr'),('jnt_bodyid','jbody'),
                      ('dof_bodyid','dof_body'),('dof_parentid','dof_parent'),('M_rowadr','row'),('M_rownnz','nnz'),('M_colind','column')):
        setattr(m,field,arr(c[key],int))
    m.body_tree=tuple(arr(np.flatnonzero(c['depth']==i).astype(np.int32),int) for i in range(int(c['depth'].max())+1))
    for field in (*MODEL_FIELDS,'dof_armature'):setattr(m,field,arr(c[field],wp.vec3 if field=='body_inertia' else float))
    d=NS(nworld=2)
    for field in (*FRAME_FIELDS,*OUTPUTS):
        dtype=wp.mat33 if field.endswith('mat') else types.vec10 if field=='cinert' else wp.spatial_vector if field=='cdof' else wp.vec3
        setattr(d,field,arr(c[field],dtype))
    d.crb=wp.full((2,m.nbody),-17.,dtype=types.vec10);d.M=wp.full((2,m.nC),-13.,dtype=float)
    for key in ('qpos','qvel','qacc','qacc_warmstart'):setattr(d,key,wp.full((2,m.nv),31.,dtype=float))
    return m,d


def upstream(m,d,phase):
    import warp as wp
    from mujoco_warp._src import smooth
    if phase in ('composed','mass-chain'):smooth.com_pos(m,d)
    elif phase=='subtree':
        wp.launch(smooth._subtree_com_init,dim=(2,m.nbody),inputs=[m.body_mass,d.xipos],outputs=[d.subtree_com])
        for layer in reversed(m.body_tree):
            wp.launch(smooth._subtree_com_acc,dim=(2,layer.size),inputs=[m.body_parentid,d.subtree_com,layer],outputs=[d.subtree_com])
        wp.launch(smooth._subtree_div,dim=(2,m.nbody),inputs=[m.body_subtreemass,d.subtree_com],outputs=[d.subtree_com])
    elif phase=='inertia':wp.launch(smooth._cinert,dim=(2,m.nbody),inputs=[m.body_rootid,m.body_mass,m.body_inertia,d.xipos,d.ximat,d.subtree_com],outputs=[d.cinert])
    else:wp.launch(smooth._cdof,dim=(2,m.njnt),inputs=[m.body_rootid,m.jnt_type,m.jnt_dofadr,m.jnt_bodyid,d.xmat,d.xanchor,d.xaxis,d.subtree_com],outputs=[d.cdof])
    if phase=='mass-chain':smooth.crb(m,d)


class Replay:
    def __init__(self,c,args,stream,phase):
        import warp as wp
        from libero.gpu.host.runtime.dynamics.mjwarp_com_pos import CenteredFrames
        from libero.gpu.host.runtime.dynamics.mjwarp_mass_matrix import MassMatrix
        self.case,self.stream,self.phase=c,stream,phase;self.models={};self.datas={};self.graphs={}
        phases=range(5) if phase in ('composed','mass-chain') else (0,1,2) if phase=='subtree' else (3,) if phase=='inertia' else (4,)
        self.keys=tuple(reference(c,phase)[0])
        for mode in ('upstream','cuda'):
            m,d=construct(c,stream);self.models[mode]=m;self.datas[mode]=d
            if mode=='cuda':
                self.binding=CenteredFrames(m,d,args.library,stream)
                if phase=='mass-chain':self.matrix=MassMatrix(m,d,args.matrix_library,stream)
            with wp.ScopedCapture(stream=stream) as cap:
                if mode=='cuda':
                    for p in phases:self.binding.enqueue(p)
                    if phase=='mass-chain':self.matrix.enqueue()
                else:upstream(m,d,phase)
            self.graphs[mode]=cap.graph

    def check(self,updated):
        import warp as wp
        c=current_case(self.case,updated);expected,errors=reference(c,self.phase);checks=[];held=True;values={}
        for mode in ('upstream','cuda'):
            m,d=self.models[mode],self.datas[mode]
            for field in (*MODEL_FIELDS,'dof_armature'):getattr(m,field).assign(c[field])
            for field in (*FRAME_FIELDS,*OUTPUTS):getattr(d,field).assign(c[field])
            d.crb.fill_(-17.);d.M.fill_(-13.)
            arrays={**{'model.'+k:v for k,v in vars(m).items() if hasattr(v,'numpy')},
                    **{'data.'+k:v for k,v in vars(d).items() if hasattr(v,'numpy') and k not in self.keys}}
            before={k:a.numpy().view(np.uint8).copy() for k,a in arrays.items()}
            wp.capture_launch(self.graphs[mode],stream=self.stream);wp.synchronize_stream(self.stream)
            values[mode]={k:getattr(d,k).numpy() for k in self.keys}
            checks += [dict(mode=mode,**x) for x in check_arrays(values[mode],expected,errors)]
            held &= all(np.array_equal(a.numpy().view(np.uint8),before[k]) for k,a in arrays.items())
        return dict(name=c['name'],phase=self.phase,updated=updated,inputs_and_holds=bool(held),checks=checks,
            maximum_difference={k:float(abs(values['cuda'][k]-values['upstream'][k]).max(initial=0.)) for k in self.keys},
            passed=bool(held and all(x['passed'] for x in checks)))


def signature(args):
    from libero.gpu.host.backends.mjwarp.build import source_files
    sources=source_files(include_com_pos=True,include_matrix=args.matrix_library is not None)
    sources={k:v for k,v in sources.items() if k.startswith('upstream/') or 'mjwarp_com_pos' in k or 'MjwarpComPos' in k or 'mjwarp_mass_matrix' in k or 'MjwarpMassMatrix' in k or k in ('host/runtime/dynamics/mjwarp_motion.py','data/mjwarp_source_pin.json','project_source_identity')}
    return dict(sources=sources,inputs=input_identity(args),library_sha256=digest(args.library),
        matrix_sha256=digest(args.matrix_library) if args.matrix_library else None,tool_sha256=digest(__file__),
        reference_sha256=digest(Path(__file__).with_name('mjwarp_com_pos_reference.py')),
        helpers={n:digest(Path(__file__).with_name(n)) for n in ('mjwarp_motion_reference.py',)})


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
    require_upstream();init_kernel_cache(args.build_root/'mjwarp/warp-cache');install_warp_guard();wp.init();stream=wp.Stream('cuda:0')
    sig=signature(args);prep=args.output/'preparation.json'
    if not args.prepare:
        if not prep.is_file() or json.loads(prep.read_text())['signature']!=sig:raise RuntimeError('Prepare exact CoM operators offline first')
        cpu=json.loads((args.output/'cpu-oracle.json').read_text())
        if not cpu['passed'] or any(cpu[k]!=sig[k] for k in ('inputs','tool_sha256','reference_sha256')):raise RuntimeError('Prepare matching CPU CoM oracle first')
    with compilation_scope(args.prepare) as scope,wp.ScopedStream(stream):
        replays=[Replay(c,args,stream,p) for p in PHASES for c in cases(args)]
        if args.matrix_library:replays += [Replay(c,args,stream,'mass-chain') for c in cases(args) if c['name'].startswith('libero')]
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
        save_json(args.output/'summary.json',dict(passed=passed,signature=sig,cases=rows,runtime_compilations=scope['events'],
            physics_integrations=0,newton_solves=0,performance=None,worlds=2,
            policy='independent FP64 dense tensor/spatial algebra; FP32 atomic absolute-contribution and propagated forward-error bounds, fixed8x safety; exact readonly/hold contracts'))
        print(json.dumps(dict(passed=passed,cases=len(rows))),flush=True);return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())
