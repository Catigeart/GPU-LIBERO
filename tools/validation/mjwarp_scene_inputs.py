"""Export actual CPU2.3.7 fixed poses, indexing and N=2 scene pose oracles.

This is offline model input preparation, not a GPU correctness/performance run.
Modern MuJoCo never opens the old MJB. Small portable arrays carry the original
prepared fixed-fixture parameters, independently of later sampler versions.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_arguments(p):
    p.add_argument('--build-root',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--tasks',nargs='+',default=['all']);return p


def main(argv=None,*,args=None):
    if args is None:args=add_arguments(argparse.ArgumentParser(description=__doc__)).parse_args(argv)
    import mujoco
    if mujoco.__version__!='2.3.7':raise RuntimeError('Export scene parameters using actual CPU reference2.3.7')
    output=args.output or args.build_root/'mjwarp/scene-inputs';output.mkdir(parents=True,exist_ok=True);tasks=json.loads((args.build_root/'TASK_INSTANCES.json').read_text())['tasks'];library=next(Path(mujoco.__file__).parent.glob('libmujoco.so*'));libhash=digest(library);rows=[]
    for task in tasks:
        if not any(v in ('all',task['suite'],task['suite']+':'+str(task['task_id'])) for v in args.tasks):continue
        directory=Path(task['directory']);model=directory/'model.mjb';bank=directory/'initial_states.npy'
        if digest(model)!=task['files']['model.mjb'] or digest(bank)!=task['files']['initial_states.npy']:raise RuntimeError('Original prepared scene inputs changed')
        m=mujoco.MjModel.from_binary_path(str(model));d=mujoco.MjData(m);states=np.load(bank,allow_pickle=False);names=np.array([mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,i) or '' for i in range(m.nbody)])
        data={k:getattr(m,k).copy() for k in ('body_pos','body_quat','body_parentid','jnt_type','jnt_bodyid','jnt_qposadr','jnt_dofadr','dof_bodyid','geom_type')};data.update(body_names=names,initial_states=states[:2].copy());fields={k:[] for k in ('xpos','xquat','site_xpos','site_xmat','geom_xpos','geom_xmat')}
        for w in range(2):
            mujoco.mj_resetData(m,d);d.qpos[:]=states[w,1:1+m.nq];d.qvel[:]=states[w,1+m.nq:];mujoco.mj_kinematics(m,d)
            for k in fields:fields[k].append(getattr(d,k).copy())
        data.update({k:np.array(v) for k,v in fields.items()});path=output/(directory.name+'.npz');temp=path.with_suffix('.tmp.npz');np.savez(temp,**data);temp.replace(path);receipt=dict(task=task['key'],seed=task['seed'],model_sha256=task['files']['model.mjb'],scene_sha256=task['files']['scene.xml'],init_sha256=task['files']['initial_states.npy'],data_sha256=digest(path),exporter_sha256=digest(__file__),mujoco=mujoco.__version__,library_sha256=libhash,physics_integrations=0,worlds=2);path.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n');rows.append(dict(task=task['key'],path=str(path),passed=True));print(json.dumps(dict(done=len(rows),task=task['key'],passed=True)),flush=True)
    (output/'index.json').write_text(json.dumps(dict(passed=True,rows=rows,version=mujoco.__version__,physics_integrations=0),indent=2)+'\n');return 0
if __name__=='__main__':raise SystemExit(main())


def verify_imported_model(model,data):
    """Common body/site/primitive frames from CPU2.3.7, no physics integration."""
    import mujoco
    d=mujoco.MjData(model);rows=[]
    for w in range(2):
        state=data['initial_states'][w];d.qpos[:]=state[1:1+model.nq];d.qvel[:]=state[1+model.nq:];mujoco.mj_kinematics(model,d)
        for name in ('xpos','xquat','site_xpos','site_xmat','geom_xpos','geom_xmat'):
            actual=getattr(d,name);expected=data[name][w]
            # Compiler-dependent mesh local frames can rotate while representing
            # the same surface. Body/site and primitive frames are common APIs.
            if name.startswith('geom_'):
                mask=data['geom_type']!=7;actual=actual[mask];expected=expected[mask]
            difference=abs(actual-expected)
            if name=='xquat':difference=np.minimum(np.max(difference,axis=-1),np.max(abs(actual+expected),axis=-1))
            error=float(difference.max(initial=0.));bound=float(4096*np.finfo(float).eps*max(1.,float(abs(expected).max(initial=0.))))
            rows.append(dict(world=w,field=name,error=error,bound=bound,passed=error<=bound))
    return dict(passed=all(v['passed'] for v in rows),checks=rows,physics_integrations=0,mesh_local_frames='compiler dependent; excluded from direct frame equality')
