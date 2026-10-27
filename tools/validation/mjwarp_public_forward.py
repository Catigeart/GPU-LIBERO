"""Actual N=2 public forward, controller query and complete Graph ownership."""
import argparse,json
from pathlib import Path
from libero.gpu.host.build.task_catalog import load_catalog
from unittest.mock import patch
import numpy as np
from .mjwarp_camlight import digest


def add_arguments(p):
    p.add_argument('--build-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--task',default='libero_10:9');p.add_argument('--prepare',action='store_true');return p


ROOTS=('time','qpos','qvel','qacc','qacc_warmstart','ctrl','qfrc_applied','xfrc_applied')
OUTPUTS=('M','qLD','qLDiagInv','xpos','xquat','xmat','geom_xpos','geom_xmat','site_xpos','site_xmat','subtree_com','cdof','cvel','cdof_dot','qfrc_bias','qfrc_passive','qfrc_smooth','qacc_smooth','qacc','qfrc_constraint','sensordata','actuator_force','actuator_length','actuator_velocity')


def compare(m,a,b):
    rows=[]
    for name in OUTPUTS:
        x,y=getattr(a,name).numpy(),getattr(b,name).numpy();error=float(abs(x.astype(float)-y).max(initial=0.));bound=1024*np.finfo(np.float32).eps*max(1.,float(abs(y).max(initial=0.)));rows.append(dict(field=name,error=error,bound=bound,passed=bool(np.all(np.isfinite(x)) and error<=bound)))
    for name in ('nefc','nacon'):
        x,y=getattr(a,name).numpy(),getattr(b,name).numpy();rows.append(dict(field=name,passed=np.array_equal(x,y),cuda=x.tolist(),upstream=y.tolist()))
    from .precision import balance_residual
    import mujoco
    residuals={}
    for mode,d in (('cuda',a),('upstream',b)):
        matrices=d.M.numpy();acc,smooth,constraint=(getattr(d,k).numpy() for k in ('qacc','qfrc_smooth','qfrc_constraint'))
        for w in range(2):
            matrix=np.empty((m.nv,m.nv));mujoco.mju_sym2dense(matrix,matrices[w].astype(float),m.M_rownnz,m.M_rowadr,m.M_colind);result=balance_residual(matrix,acc[w],smooth[w],constraint[w]);residuals[mode,w]=result;rows.append(dict(field=mode+' force balance',world=w,diagnostic=True,**result))
    for w in range(2):
        x,y=residuals['cuda',w],residuals['upstream',w]
        rows.append(dict(field='force balance conformance',world=w,passed=bool(x['backward_error']<=y['backward_error']+x['limit']),candidate=x['backward_error'],reference=y['backward_error'],arithmetic_allowance=x['limit']))
    return rows


def upstream_numerical_kernel(row):
    # @wp.kernel(module='unique') replaces the module name with a builder/hash
    # name. Retain Python provenance as well: a generated name alone cannot
    # prove that the original numerical operator was replaced.
    return row['module'].startswith('mujoco_warp') or row.get('python_module','').startswith('mujoco_warp') or '/mujoco_warp/' in row.get('source','').replace('\\','/')


def graph_audit(env):
    import warp as wp
    launches=[];original=wp.launch
    def record(*a,**kw):
        kernel=a[0] if a else kw['kernel'];dim=a[1] if len(a)>1 else kw.get('dim',0);size=int(np.prod(dim));module=getattr(getattr(kernel,'module',None),'name','');
        if size:
            function=getattr(kernel,'func',None)
            launches.append(dict(module=module,key=kernel.key,threads=size,
                python_module=getattr(function,'__module__',getattr(kernel,'__module__','')),
                source=getattr(getattr(function,'__code__',None),'co_filename','')))
        return original(*a,**kw)
    # These captures are deliberately fresh; retain the single public context
    # separately from the25 physical contexts. No control Graph replay here.
    env.runtime._forward_graph=None;env.runtime._forward_graph_owners=[]
    with patch.object(wp,'launch',record):
        env.runtime.forward();env.runtime.prepare_control_graph()
    wp.synchronize_stream(env.stream)
    numerical=[v for v in launches if upstream_numerical_kernel(v)]
    return dict(passed=not numerical and len(env.runtime._forward_graph_owners)==1 and len(env.runtime._control_graph_owners)==25,positive_warp_launches=launches,upstream_numerical_launches=numerical,public_newton_contexts=len(env.runtime._forward_graph_owners),control_newton_contexts=len(env.runtime._control_graph_owners),control_graph_replays=0)


def main(argv=None,*,args=None):
    if args is None:args=add_arguments(argparse.ArgumentParser(description=__doc__)).parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=True)
    from libero.gpu import precision
    precision.select('fp32')
    import warp as wp
    import mujoco_warp as mjw
    from libero.libero import get_libero_path
    from libero.libero.envs import GpuBatchEnv
    from libero.gpu.host.build.compilation import compilation_scope
    from libero.gpu.host.backends.mjwarp.build import profile_sources
    tasks=load_catalog(args.build_root);task=next(t for t in tasks if t['suite']+':'+str(t['task_id'])==args.task)
    with compilation_scope(args.prepare) as scope:
        env=GpuBatchEnv(str(Path(get_libero_path('bddl_files'))/task['key']),2,physics_engine='mjwarp',precision='fp32',initialize=False,allow_compile=False,build_root=args.build_root,use_camera_obs=False,ignore_done=True,mjwarp_options=dict(preset='cuda_libero',fast_compile=True))
        try:
            m,d,stream=env.model,env.data,env.stream;sig=dict(identity=env.identity,sources=profile_sources(env),tool=digest(__file__));prep=args.output/'preparation.json'
            if not args.prepare and (not prep.is_file() or json.loads(prep.read_text())['signature']!=sig):raise RuntimeError('Prepare matching original public forward operators offline first')
            with wp.ScopedStream(stream):
                raw=mjw.make_data(env.built.native_model,nworld=2,nccdmax=env.capacities['nconmax'],**env.capacities)
                with wp.ScopedCapture(stream=stream) as capture:mjw.forward(m,raw)
                raw_graph=capture.graph
                env.runtime.forward()
                points=wp.empty_like(env.runtime.controller.points);jp=wp.empty_like(env.runtime.controller.jp);jr=wp.empty_like(env.runtime.controller.jr)
                from libero.gpu.host.backends.mjwarp import kernels
                with wp.ScopedCapture(stream=stream) as capture:
                    wp.launch(kernels.site_points,2,[env.runtime.controller.site,raw.site_xpos,points]);mjw.jac(m,raw,jp,jr,points,env.runtime.controller.body)
                jac_graph=capture.graph
                if args.prepare:
                    wp.synchronize_stream(stream);prep.write_text(json.dumps(dict(signature=sig,compilation_events=scope['events'],physics_integrations=0),indent=2)+'\n');print(json.dumps(dict(prepared=True,compilations=len(scope['events']))),flush=True);return 0
                initial={key:getattr(d,key).numpy() for key in ROOTS};initial['qpos']=env.initial_states[:2,1:1+m.nq].astype(np.float32);initial['qvel']=env.initial_states[:2,1+m.nq:].astype(np.float32);initial['qacc'].fill(0);initial['qacc_warmstart'].fill(.007);initial['time'].fill(.13);rows=[]
                for updated in (False,True):
                    inputs={k:v.copy() for k,v in initial.items()}
                    if updated:
                        inputs['qpos'][:,0]+=[.003,-.002];inputs['qvel']*=np.float32(.93);inputs['qfrc_applied'][:]=.013;inputs['xfrc_applied'][:,1:,:]=.007
                    holds={key:inputs[key].view(np.uint8).copy() for key in ('qpos','qvel','qacc_warmstart','time','ctrl')}
                    for target in (d,raw):
                        for key,value in inputs.items():getattr(target,key).assign(value)
                        target.overflow.zero_()
                    env.runtime.forward();wp.capture_launch(raw_graph,stream=stream);wp.synchronize_stream(stream);checks=compare(env.built.native_model,d,raw)
                    for mode,target in (('cuda',d),('upstream',raw)):
                        for key,value in holds.items():checks.append(dict(field=mode+' hold '+key,passed=np.array_equal(getattr(target,key).numpy().view(np.uint8),value)))
                    env.runtime.controller.gather();wp.capture_launch(jac_graph,stream=stream);wp.synchronize_stream(stream)
                    for name,actual,ref in [('jacp',env.runtime.controller.jp,jp),('jacr',env.runtime.controller.jr,jr)]:
                        x,y=actual.numpy(),ref.numpy();e=float(abs(x-y).max(initial=0.));bound=1024*np.finfo(np.float32).eps*max(1.,float(abs(y).max(initial=0.)));checks.append(dict(field=name,error=e,bound=bound,passed=bool(e<=bound)))
                    row=dict(updated=updated,checks=checks,passed=all(v['passed'] for v in checks if not v.get('diagnostic',False)));rows.append(row);print(json.dumps(dict(task=args.task,updated=updated,passed=row['passed'],failures=[v for v in checks if not v['passed'] and not v.get('diagnostic',False)])),flush=True)
                    if not row['passed']:
                        np.savez(args.output/'failure.npz',**{mode+'.'+key:getattr(target,key).numpy() for mode,target in (('cuda',d),('upstream',raw)) for key in ROOTS+OUTPUTS});break
                audit=graph_audit(env);passed=all(row['passed'] for row in rows) and audit['passed'] and not scope['events'];result=dict(passed=passed,signature=sig,cases=rows,audit=audit,runtime_compilations=scope['events'],worlds=2,physics_integrations=0,performance=None,modern_scene_cache_hit=True);(args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=passed,public_contexts=audit['public_newton_contexts'],control_contexts=audit['control_newton_contexts'],unexpected_upstream=audit['upstream_numerical_launches'])),flush=True);return 0 if passed else 2
        finally:env.close()
if __name__=='__main__':raise SystemExit(main())
