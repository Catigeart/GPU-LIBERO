"""Frozen N=2 Jacobian, FORCE/TORQUE and rigid gravity qualification."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from .mjwarp_camlight import add_arguments,digest,input_identity
from .mjwarp_queries_reference import reference


def cases(args):
    result=[]
    for path in sorted(args.inputs.glob('*.npz')):
        if digest(path)!=json.loads(path.with_suffix('.json').read_text())['sha256']:raise RuntimeError('Frozen query inputs changed')
        with np.load(path,allow_pickle=False) as f:c={k:f[k].copy() for k in f.files}
        for key in ('nbody','nv','nsite','nsensor','channels','disabled','jac_position','jac_rotation'):c[key]=int(c[key])
        c['name']=path.stem
        pad=((c['nv']+3)//4)*4 if c['nv']<=32 else ((c['nv']+15)//16)*16
        c['ancestor']=np.pad(c['ancestor'],((0,0),(0,pad-c['nv'])))
        result.append(c)
    if not result:raise ValueError('No frozen query inputs')
    for original in tuple(result):
        if original['name']!='deep289':continue
        import copy
        for name in ('position-only','rotation-only','world-body','sensor-cutoff','sensor-disabled','sensor-gaps','broadcast'):
            c=copy.deepcopy(original);c['name']+='-'+name
            if name=='position-only':c['jac_rotation']=0
            if name=='rotation-only':c['jac_position']=0
            if name=='world-body':c['body'][:]=0
            if name=='sensor-cutoff':c['cutoff'][:]=.13;c['sensor_datatype'][:]=[0,1]
            if name=='sensor-disabled':c['disabled']=1
            if name=='sensor-gaps':
                c['channels']+=7;c['sensor_address']+=2;c['sensor_datatype'][:]=[2,3];c['cutoff'][:]=.13;c['initial_sensor']=np.full((2,c['channels']),-77,np.float32)
            if name=='broadcast':
                for k in ('gravity','mass','coefficient'):c[k]=c[k][:1].copy()
            result.append(c)
    return result


def current(c,updated):
    import copy
    c=copy.deepcopy(c)
    if updated:
        c['point']+=np.array([.03,-.01,.02],np.float32);c['com']+=np.array([.01,-.02,.03],np.float32);c['cdof']*=np.float32(.97)
        c['wrench']*=np.float32(-1.13);c['site_position']+=np.array([.01,.02,-.03],np.float32);c['gravity']*=np.float32(.91);c['coefficient']*=np.float32(1.17);c['mass']*=np.float32(.94)
        c['position']+=np.array([.02,-.01,.03],np.float32)
    return c


def construct(c,stream):
    import warp as wp
    from mujoco_warp._src import types
    arr=lambda v,t:wp.array(np.ascontiguousarray(v),dtype=t,device=stream.device)
    m=NS(nbody=c['nbody'],nv=c['nv'],nsite=c['nsite'],nsensor=c['nsensor'],nsensordata=c['channels'],opt=NS(disableflags=int(types.DisableBit.SENSOR) if c['disabled'] else 0))
    for key,field in (('root','body_rootid'),('ancestor','body_isdofancestor'),('body_parent','body_parentid'),('dof_body','dof_bodyid'),('site_body','site_bodyid'),('sensor_type','sensor_type'),('sensor_object','sensor_objid'),('sensor_object_type','sensor_objtype'),('sensor_datatype','sensor_datatype'),('sensor_dimension','sensor_dim'),('sensor_address','sensor_adr'),('sensor_stage_acc','sensor_stage_acc_adr')):setattr(m,field,arr(c[key],int))
    for key,field in (('mass','body_mass'),('coefficient','body_gravcomp'),('cutoff','sensor_cutoff')):setattr(m,field,arr(c[key],float))
    m.opt.gravity=arr(c['gravity'],wp.vec3);d=NS(nworld=2)
    for key,field,t in (('cdof','cdof',wp.spatial_vector),('com','subtree_com',wp.vec3),('site_position','site_xpos',wp.vec3),('site_rotation','site_xmat',wp.mat33),('wrench','cfrc_int',wp.spatial_vector),('position','xipos',wp.vec3),('initial_sensor','sensordata',float),('initial_gravity','qfrc_gravcomp',float)):setattr(d,field,arr(c[key],t))
    q=NS(point=arr(c['point'],wp.vec3),body=arr(c['body'],int),jacp=arr(c['initial_jacp'],float),jacr=arr(c['initial_jacr'],float))
    return m,d,q


def assign(c,m,d,q):
    for key,field in (('mass','body_mass'),('coefficient','body_gravcomp'),('cutoff','sensor_cutoff')):getattr(m,field).assign(c[key])
    m.opt.gravity.assign(c['gravity'])
    for key,field in (('cdof','cdof'),('com','subtree_com'),('site_position','site_xpos'),('site_rotation','site_xmat'),('wrench','cfrc_int'),('position','xipos'),('initial_sensor','sensordata'),('initial_gravity','qfrc_gravcomp')):getattr(d,field).assign(c[key])
    for key in ('point','body'):getattr(q,key).assign(c[key])
    q.jacp.assign(c['initial_jacp']);q.jacr.assign(c['initial_jacr'])


def upstream(c,m,d,q,phase):
    import warp as wp
    from mujoco_warp._src import support,sensor,passive
    if phase=='jacobian':support.jac(m,d,q.jacp if c['jac_position'] else None,q.jacr if c['jac_rotation'] else None,q.point,q.body)
    elif phase=='gravity':
        wp.launch(passive._gravity_force,(2,m.nbody-1,m.nv),inputs=[m.opt.gravity,m.body_parentid,m.body_rootid,m.body_mass,m.body_gravcomp,m.dof_bodyid,m.body_isdofancestor,d.xipos,d.subtree_com,d.cdof],outputs=[d.qfrc_gravcomp])
    elif not c['disabled']:
        dummy_int=wp.empty(0,dtype=int);dummy_vec=wp.empty((0,0),dtype=wp.vec3);dummy_spatial=wp.empty((0,0),dtype=wp.spatial_vector);dummy_float=wp.empty((0,0),dtype=float)
        wp.launch(sensor._sensor_acc,(2,m.nsensor),inputs=[m.body_rootid,dummy_int,dummy_int,m.site_bodyid,dummy_int,m.sensor_type,m.sensor_datatype,m.sensor_objtype,m.sensor_objid,m.sensor_adr,m.sensor_cutoff,m.sensor_stage_acc_adr,dummy_vec,d.xipos,dummy_vec,d.site_xpos,d.site_xmat,dummy_vec,d.subtree_com,dummy_spatial,dummy_float,dummy_float,dummy_spatial,d.cfrc_int],outputs=[d.sensordata])


class Replay:
    def __init__(self,c,args,stream,phase):
        import warp as wp
        from libero.gpu.host.runtime.dynamics.mjwarp_queries import Jacobian,ForceSensors,GravityProjection
        self.case,self.stream,self.phase=c,stream,phase;self.owners={};self.graphs={};self.retained=[]
        for mode in ('upstream','cuda'):
            m,d,q=construct(c,stream);self.owners[mode]=(m,d,q)
            if mode=='cuda':
                if phase=='jacobian':self.binding=Jacobian(m,d,q.point,q.body,q.jacp if c['jac_position'] else None,q.jacr if c['jac_rotation'] else None,args.library,stream)
                elif phase=='sensor':self.binding=ForceSensors(m,d,args.library,stream)
                else:self.binding=GravityProjection(m,d,args.library,stream)
            with wp.ScopedCapture(stream=stream) as cap:
                if mode=='cuda':self.binding.enqueue()
                else:upstream(c,m,d,q,phase)
            self.graphs[mode]=cap.graph
        self.pointers={k:v.ptr for k,v in self.binding.arrays.items()}
    def check(self,updated):
        import warp as wp
        c=current(self.case,updated);expected,bounds=reference(c,self.phase);checks=[];outputs={}
        def add(name,passed,**extra):checks.append(dict(name=name,passed=bool(passed),**extra))
        for mode,(m,d,q) in self.owners.items():
            assign(c,m,d,q);all_arrays={prefix+k:v for prefix,obj in (('m.',m),('opt.',m.opt),('d.',d),('q.',q)) for k,v in vars(obj).items() if hasattr(v,'numpy')};writes={'q.jacp','q.jacr'} if self.phase=='jacobian' else {'d.sensordata'} if self.phase=='sensor' else {'d.qfrc_gravcomp'};readonly={k:v for k,v in all_arrays.items() if k not in writes};before={k:v.numpy().view(np.uint8).copy() for k,v in readonly.items()}
            wp.capture_launch(self.graphs[mode],stream=self.stream);wp.synchronize_stream(self.stream)
            values=dict(jacp=q.jacp.numpy(),jacr=q.jacr.numpy()) if self.phase=='jacobian' else dict(sensor=d.sensordata.numpy()) if self.phase=='sensor' else dict(gravity=d.qfrc_gravcomp.numpy());outputs[mode]=values
            for k,got in values.items():
                error=abs(got.astype(float)-expected[k]);bound=bounds[k];add(mode+' independent '+k,np.all(np.isfinite(got)) and np.all(error<=bound),maximum_error=float(error.max(initial=0.)),maximum_bound=float(bound.max(initial=0.)),maximum_ratio=float(np.max(np.divide(error,bound,out=np.zeros_like(error),where=bound!=0),initial=0.)))
                exact=bound==0;add(mode+' exact hold/zero '+k,np.all(got[exact]==expected[k][exact]))
                if not updated and c['name'] in ('libero_10-0','libero_10-9','libero_spatial-0','deep289','empty'):
                    target=c['cpu_'+{'jacp':'jacp','jacr':'jacr','sensor':'sensor','gravity':'gravity'}[k]]
                    if k=='gravity':target=target+c['initial_gravity']
                    add(mode+' actual CPU2.3.7 '+k,np.all(abs(got-target)<=2*bound+128*2.**-52*(1+abs(target))),maximum_error=float(abs(got-target).max(initial=0.)))
            add(mode+' readonly inputs',all(np.array_equal(v.numpy().view(np.uint8),before[k]) for k,v in readonly.items()))
        add('retained pointers',all(self.binding.arrays[k].ptr==v for k,v in self.pointers.items()))
        return dict(name=c['name'],phase=self.phase,updated=updated,checks=checks,passed=all(v['passed'] for v in checks),maximum_difference={k:float(abs(outputs['cuda'][k]-outputs['upstream'][k]).max(initial=0.)) for k in expected})


def signature(args):
    from libero.gpu.host.backends.mjwarp.build import source_files
    files=source_files(include_queries=True)
    files={k:v for k,v in files.items() if k.startswith('upstream/') or 'mjwarp_queries' in k or 'MjwarpQueries' in k or k in ('cuda/dynamics/mjwarp_spatial.cuh','host/runtime/dynamics/mjwarp_motion.py','data/mjwarp_source_pin.json','project_source_identity')}
    return dict(sources=files,inputs=input_identity(args),library_sha256=digest(args.library),tool_sha256=digest(__file__),reference_sha256=digest(Path(__file__).with_name('mjwarp_queries_reference.py')))


def main(argv=None,*,args=None):
    if args is None:args=add_arguments(argparse.ArgumentParser(description=__doc__)).parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.cpu_reference:raise ValueError('CPU public functions are already captured in the source-bound frozen fixtures')
    from libero.gpu import precision
    precision.select('fp32')
    import warp as wp
    from warp._src.build import init_kernel_cache
    from libero.gpu.host.backends.mjwarp.model import require_upstream
    from libero.gpu.host.build.compilation import compilation_scope,install_warp_guard
    require_upstream();init_kernel_cache(args.build_root/'mjwarp/warp-cache');install_warp_guard();wp.init();stream=wp.Stream('cuda:0');sig=signature(args);prep=args.output/'preparation.json'
    if not args.prepare and (not prep.is_file() or json.loads(prep.read_text())['signature']!=sig):raise RuntimeError('Prepare exact query reference operators offline first')
    with compilation_scope(args.prepare) as scope,wp.ScopedStream(stream):
        replays=[Replay(c,args,stream,p) for c in cases(args) for p in ('jacobian','sensor','gravity')]
        if args.prepare:
            wp.synchronize_stream(stream);prep.write_text(json.dumps(dict(signature=sig,compilation_events=scope['events'],physics_integrations=0),indent=2)+'\n');print(json.dumps(dict(prepared=True,captures=2*len(replays),compilations=len(scope['events']))),flush=True);return 0
        rows=[]
        for replay in replays:
            for updated in (False,True):
                row=replay.check(updated);rows.append(row);print(json.dumps(dict(name=row['name'],phase=row['phase'],updated=updated,passed=row['passed'],failures=[v for v in row['checks'] if not v['passed']])),flush=True)
                if not row['passed']:break
            if not row['passed']:break
        passed=all(v['passed'] for v in rows);(args.output/'summary.json').write_text(json.dumps(dict(passed=passed,signature=sig,cases=rows,runtime_compilations=scope['events'],worlds=2,physics_integrations=0,performance=None),indent=2)+'\n');print(json.dumps(dict(passed=passed,cases=len(rows))),flush=True);return 0 if passed else 2
if __name__=='__main__':raise SystemExit(main())
