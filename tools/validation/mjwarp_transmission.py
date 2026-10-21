"""Frozen N=2 transmission, generic CSR velocity and actuation-chain qualification."""
import argparse,hashlib,json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from .mjwarp_transmission_reference import transmission,velocity,validate_csr
from .mjwarp_motion_reference import check_arrays
PHASES=('transmission','velocity','composed','actuation-chain')
TASKS=('libero_10-0','libero_10-3','libero_spatial-0')


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def add_arguments(p):
    for name in ('build-root','inputs','output','library','actuation-library'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--cpu-reference',action='store_true')
    return p


def cases(args):
    result=[]
    for p in sorted(args.inputs.glob('*.npz')):
        with np.load(p,allow_pickle=False) as f:c={k:f[k].copy() for k in f.files}
        for key in ('nq','nv','njnt','nu','nactuator','na'):c[key]=int(c[key])
        for key in ('disable_clampctrl','gravity_enabled','velocity_only'):c[key]=bool(c[key])
        c['name']=p.stem;result.append(c)
    if not result:raise ValueError('No frozen transmission inputs')
    return result


def current_case(original,updated):
    c={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in original.items()}
    if updated:
        c['qpos']*=np.float32(.8);c['qpos']+=np.float32(.013)
        c['qvel']*=np.float32(-.7);c['gear']*=np.float32(-1.3)
        c['ctrl']*=np.float32(.6)
        if c['velocity_only']:
            _,used=validate_csr(c);c['moment'][used]*=np.float32(-1.2)
            c['moment_colind'][used]=(c['moment_colind'][used]+1)%c['nv']
            # Permute complete rows without changing any storage owners.
            c['moment_rownnz']=c['moment_rownnz'][:,::-1].copy();c['moment_rowadr']=c['moment_rowadr'][:,::-1].copy()
    return c


def input_identity(args):return {p.name:digest(p) for p in sorted(args.inputs.glob('*')) if p.is_file()}


def cpu_reference(args):
    import mujoco
    checks=[]
    for original in cases(args):
        if original['name'] in TASKS:
            p=args.inputs/(original['name']+'.json');receipt=json.loads(p.read_text())
            if digest(receipt['model_path'])!=receipt['model_sha256']:raise RuntimeError('CPU transmission model changed')
            m=mujoco.MjModel.from_binary_path(receipt['model_path'])
        else:
            # A small real CPU model provides actual mj_transmission, even when
            # actuator/control-slot counts differ in the pinned Warp fixture.
            bodies=''.join('<body><joint name="j%d" type="hinge"/><geom type="sphere" size=".01" mass="1"/></body>'%j for j in range(original['nv']))
            motors=''.join('<motor joint="j%d"/>'%(u%original['nv']) for u in range(original['nactuator']))
            m=mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+bodies+'</worldbody><actuator>'+motors+'</actuator></mujoco>')
        d=mujoco.MjData(m)
        for updated in (False,True):
            c=current_case(original,updated)
            if not c['velocity_only']:
                length,_,dense=transmission(c)
                if (m.nq,m.nv,m.njnt,m.nu)!=(c['nq'],c['nv'],c['njnt'],c['nactuator']):raise ValueError('CPU transmission fixture dimensions differ')
                m.jnt_type[:]=c['joint_type'];m.actuator_trntype[:]=c['trntype'];m.actuator_trnid[:]=c['trnid']
                for w in range(2):
                    m.actuator_gear[:]=c['gear'][w%len(c['gear'])];d.qpos[:]=c['qpos'][w]
                    mujoco.mj_transmission(m,d)
                    for key,actual,target in (('length',d.actuator_length,length[w]),('moment',d.actuator_moment,dense[w])):
                        error=float(abs(actual-target).max());limit=4096*np.finfo(float).eps*max(1.,float(abs(target).max()))
                        checks.append(dict(name=c['name'],updated=updated,world=w,field=key,error=error,limit=limit,passed=bool(error<=limit)))
            dense,_=validate_csr(c);expected,_=velocity(c)
            for w in range(2):
                actual=np.empty(c['nactuator']);mujoco.mju_mulMatVec(actual,dense[w],c['qvel'][w].astype(float))
                error=float(abs(actual-expected['actuator_velocity'][w]).max())
                scale=max(1.,float((abs(dense[w])@abs(c['qvel'][w].astype(float))).max()))
                limit=4096*np.finfo(float).eps*scale
                checks.append(dict(name=c['name'],updated=updated,world=w,field='CSR velocity dense CPU oracle',error=error,limit=limit,passed=bool(error<=limit)))
        del d,m
    result=dict(passed=all(x['passed'] for x in checks),checks=checks,version=mujoco.__version__,inputs=input_identity(args),
        tool_sha256=digest(__file__),reference_sha256=digest(Path(__file__).with_name('mjwarp_transmission_reference.py')),full_physics_substeps=0)
    (args.output/'cpu-oracle.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'],cpu_checks=len(checks))),flush=True);return 0 if result['passed'] else 2


def construct(c,stream):
    import warp as wp
    from .mjwarp_actuation import construct as actuation_model
    m,d=actuation_model(c,stream);m.nq=c['nq'];m.nacttrnbody=0
    m.callback=NS(act_dyn=None,act_gain=None,act_bias=None)
    arr=lambda a,dtype:wp.array(np.ascontiguousarray(a),dtype=dtype,device=stream.device)
    m.jnt_qposadr=arr(c['qposadr'],int);m.jnt_dofadr=arr(c['dofadr'],int);m.actuator_gear=arr(c['gear'],wp.spatial_vector)
    d.qpos=arr(c['qpos'],float);d.qvel=arr(c['qvel'],float)
    # Typed unused inputs for the raw pinned multi-domain comparator.
    for field in ('body_parentid','body_rootid','body_weldid','body_dofnum','body_dofadr','dof_bodyid','dof_parentid','site_bodyid','ten_J_rownnz','ten_J_rowadr','ten_J_colind'):
        setattr(m,field,wp.zeros(1,dtype=int,device=stream.device))
    m.body_isdofancestor=wp.zeros((1,1),dtype=int);m.site_quat=wp.zeros((1,1),dtype=wp.quat)
    m.actuator_cranklength=wp.zeros((1,m.nactuator),dtype=float)
    d.xquat=wp.zeros((2,1),dtype=wp.quat);d.site_xpos=wp.zeros((2,1),dtype=wp.vec3);d.site_xmat=wp.zeros((2,1),dtype=wp.mat33)
    d.subtree_com=wp.zeros((2,1),dtype=wp.vec3);d.cdof=wp.zeros((2,m.nv),dtype=wp.spatial_vector)
    d.ten_J=wp.zeros((2,1),dtype=float);d.ten_length=wp.zeros((2,1),dtype=float)
    return m,d


def upstream_transmission(m,d,counter):
    import warp as wp
    from mujoco_warp._src import smooth
    counter.zero_()
    wp.launch(smooth._transmission,dim=(d.nworld,m.nactuator),inputs=[m.nv,m.body_parentid,m.body_rootid,m.body_weldid,m.body_dofnum,m.body_dofadr,
        m.jnt_type,m.jnt_qposadr,m.jnt_dofadr,m.dof_bodyid,m.dof_parentid,m.site_bodyid,m.site_quat,m.ten_J_rownnz,m.ten_J_rowadr,m.ten_J_colind,
        m.actuator_trntype,m.actuator_trnid,m.actuator_cranklength,m.actuator_gear,m.body_isdofancestor,d.qpos,d.xquat,d.site_xpos,d.site_xmat,d.subtree_com,
        d.cdof,d.ten_J,d.ten_length,counter],outputs=[d.actuator_length,d.moment_rownnz,d.moment_rowadr,d.moment_colind,d.actuator_moment])


class Replay:
    def __init__(self,c,args,stream,phase):
        import warp as wp
        from mujoco_warp._src import forward
        from libero.gpu.host.runtime.dynamics.mjwarp_transmission import TransmissionNumerics,qualify
        from libero.gpu.host.runtime.dynamics.mjwarp_actuation import ActuatorForces
        self.case,self.phase,self.stream=c,phase,stream;self.models={};self.datas={};self.graphs={};self.bindings={};self.force_bindings={};self.counters={}
        for mode in ('upstream','cuda'):
            m,d=construct(c,stream);self.models[mode]=m;self.datas[mode]=d
            self.counters[mode]=wp.empty(2,dtype=int,device=stream.device)
            if mode=='cuda':
                admission=qualify(m,d,transmission='upstream' if phase=='velocity' else 'cuda',actuator_velocity='upstream' if phase=='transmission' else 'cuda')
                self.bindings[mode]=TransmissionNumerics(m,d,args.library,stream,admission=admission)
            if phase=='actuation-chain' and mode=='cuda':self.force_bindings[mode]=ActuatorForces(m,d,args.actuation_library,stream)
            with wp.ScopedCapture(stream=stream) as cap:
                if phase!='velocity':
                    if mode=='cuda':self.bindings[mode].enqueue(0)
                    else:upstream_transmission(m,d,self.counters[mode])
                if phase!='transmission':
                    if mode=='cuda':self.bindings[mode].enqueue(1)
                    else:wp.launch(forward._actuator_velocity,dim=(2,m.nactuator),inputs=[d.qvel,d.moment_rownnz,d.moment_rowadr,d.moment_colind,d.actuator_moment],outputs=[d.actuator_velocity])
                if phase=='actuation-chain':
                    if mode=='upstream':forward.fwd_actuation(m,d)
                    else:
                        for k in (0,1,2):self.force_bindings[mode].enqueue(k)
            self.graphs[mode]=cap.graph

    def check(self,updated):
        import warp as wp
        from .mjwarp_actuation_reference import reference as actuation_reference
        c=current_case(self.case,updated);checks=[];values={};held=True;indices=True
        for mode in ('upstream','cuda'):
            m,d=self.models[mode],self.datas[mode]
            m.actuator_gear.assign(c['gear']);d.qpos.assign(c['qpos']);d.qvel.assign(c['qvel']);d.ctrl.assign(c['ctrl'])
            for field,key in (('moment_rownnz','moment_rownnz'),('moment_rowadr','moment_rowadr'),('moment_colind','moment_colind'),('actuator_moment','moment')):getattr(d,field).assign(c[key])
            d.actuator_length.fill_(37.);d.actuator_velocity.fill_(-31.);d.actuator_force.fill_(23.);d.qfrc_actuator.fill_(-19.)
            roots={k:getattr(d,k).numpy().view(np.uint8).copy() for k in ('qpos','qvel','qacc','qacc_warmstart','ctrl','act','act_dot')}
            gear_before=m.actuator_gear.numpy().view(np.uint8).copy()
            if self.phase=='velocity':
                for k in ('moment_rownnz','moment_rowadr','moment_colind','actuator_moment','actuator_length'):roots[k]=getattr(d,k).numpy().view(np.uint8).copy()
            if self.phase=='transmission':roots['actuator_velocity']=d.actuator_velocity.numpy().view(np.uint8).copy()
            if self.phase!='actuation-chain':
                for k in ('actuator_force','qfrc_actuator'):roots[k]=getattr(d,k).numpy().view(np.uint8).copy()
            wp.capture_launch(self.graphs[mode],stream=self.stream);wp.synchronize_stream(self.stream)
            out={k:getattr(d,f).numpy() for k,f in (('moment_rownnz','moment_rownnz'),('moment_rowadr','moment_rowadr'),('moment_colind','moment_colind'),('moment','actuator_moment'),('length','actuator_length'),('velocity','actuator_velocity'),('actuator_force','actuator_force'),('qfrc_actuator','qfrc_actuator'))}
            actual=c|out;dense,used=validate_csr(actual)
            if self.phase!='velocity':
                length,error,expected_dense=transmission(c)
                checks+= [dict(mode=mode,**x) for x in check_arrays({'actuator_length':out['length']},{'actuator_length':length},{'actuator_length':error})]
                # Atomic allocation order is unspecified upstream. Check exact
                # row identity, ownership, payload copies and unmodified tail.
                indices &= bool(np.all(out['moment_rownnz']==1) and np.all(used[:,:c['nactuator']]) and not np.any(used[:,c['nactuator']:]))
                indices &= bool(np.array_equal(dense,expected_dense))
                for w in range(2):
                    for u in range(c['nactuator']):
                        i=out['moment_rowadr'][w,u];j=c['trnid'][u,0]
                        indices &= bool(out['moment_colind'][w,i]==c['dofadr'][j])
                        indices &= bool(out['moment'][w,i:i+1].view(np.uint8).tobytes()==c['gear'][w%len(c['gear']),u,0:1].view(np.uint8).tobytes())
                indices &= bool(np.array_equal(out['moment'][:,c['nactuator']:].view(np.uint8),c['moment'][:,c['nactuator']:].view(np.uint8)))
                indices &= bool(np.array_equal(out['moment_colind'][:,c['nactuator']:],c['moment_colind'][:,c['nactuator']:]))
                total=self.bindings[mode].arrays['total'].numpy() if mode=='cuda' else self.counters[mode].numpy()
                indices &= bool(np.all(total==c['nactuator']))
            if self.phase!='transmission':
                expected,error=velocity(actual)
                checks += [dict(mode=mode,**x) for x in check_arrays({'actuator_velocity':out['velocity']},expected,error)]
            if self.phase=='actuation-chain':
                expected,error=actuation_reference(actual,'composed')
                checks += [dict(mode=mode,**x) for x in check_arrays({k:out[k] for k in expected},expected,error)]
            held &= bool(all(np.array_equal(getattr(d,k).numpy().view(np.uint8),v) for k,v in roots.items()) and np.array_equal(m.actuator_gear.numpy().view(np.uint8),gear_before))
            values[mode]=out
        keys=['length'] if self.phase=='transmission' else ['velocity'] if self.phase=='velocity' else ['length','velocity']
        if self.phase=='actuation-chain':keys+=['actuator_force','qfrc_actuator']
        return dict(name=c['name'],phase=self.phase,updated=updated,inputs_held=held,csr_contracts=indices,checks=checks,
            max_difference={k:float(abs(values['cuda'][k]-values['upstream'][k]).max()) for k in keys},passed=bool(held and indices and all(x['passed'] for x in checks))),values


def signature(args):
    from libero.gpu.host.backends.mjwarp.build import source_files
    sources=source_files(include_transmission=True,include_actuation=True)
    sources={k:v for k,v in sources.items() if k.startswith('upstream/') or 'mjwarp_transmission' in k or 'mjwarp_actuation' in k or k in ('cmake/MjwarpTransmission.cmake','cmake/MjwarpActuation.cmake','host/runtime/dynamics/mjwarp_motion.py','project_source_identity','data/mjwarp_source_pin.json')}
    helpers={n:digest(Path(__file__).with_name(n)) for n in ('mjwarp_motion_reference.py','mjwarp_actuation.py','mjwarp_actuation_reference.py')}
    return dict(sources=sources,helpers=helpers,inputs=input_identity(args),library_sha256=digest(args.library),actuation_library_sha256=digest(args.actuation_library),
        tool_sha256=digest(__file__),reference_sha256=digest(Path(__file__).with_name('mjwarp_transmission_reference.py')))


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
    sig=signature(args);prepared=args.output/'preparation.json'
    if not args.prepare:
        if not prepared.is_file() or json.loads(prepared.read_text())['signature']!=sig:raise RuntimeError('Prepare exact transmission operators offline first')
        cpu=json.loads((args.output/'cpu-oracle.json').read_text())
        if not cpu['passed'] or any(cpu[k]!=sig[k] for k in ('inputs','tool_sha256','reference_sha256')):raise RuntimeError('Prepare matching CPU transmission oracle first')
    with compilation_scope(args.prepare) as scope,wp.ScopedStream(stream):
        replays=[Replay(c,args,stream,p) for p in PHASES for c in cases(args) if p=='velocity' or not c['velocity_only']]
        if args.prepare:
            wp.synchronize_stream(stream);save_json(prepared,dict(signature=sig,compilation_events=scope['events'],full_physics_substeps=0))
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
        save_json(args.output/'summary.json',dict(passed=passed,signature=sig,cases=rows,runtime_compilations=scope['events'],full_physics_substeps=0,newton_solves=0,performance=None,worlds=2,
            policy='independent FP64 scalar products/ordered dot bounds; actuation qualified at actual length/velocity/moment boundary; fixed8x safety; exact CSR ownership/copy/hold contracts'))
        print(json.dumps(dict(passed=passed,cases=len(rows))),flush=True);return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())
