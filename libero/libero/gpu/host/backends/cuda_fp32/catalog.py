"""Prepare portable CPU2.3.7 scene inputs without Warp/native CUDA intake.

Run in a separate CPU MuJoCo2.3.7 environment against this source checkout.
This preserves the original fixture sampler and scene parameters. The modern
FP32 environment subsequently compiles XML with its own MuJoCo3.16 library.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def portable_inputs(model, bank):
    import mujoco
    result={k:getattr(model,k).copy() for k in ('body_pos','body_quat','body_parentid',
        'jnt_type','jnt_bodyid','jnt_qposadr','jnt_dofadr','dof_bodyid','geom_type')}
    result['body_names']=np.array([mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) or '' for i in range(model.nbody)])
    result['initial_states']=bank[np.arange(2)%len(bank)].copy()
    fields={k:[] for k in ('xpos','xquat','site_xpos','site_xmat','geom_xpos','geom_xmat')}
    d=mujoco.MjData(model)
    for state in result['initial_states']:
        mujoco.mj_resetData(model,d)
        d.qpos[:]=state[1:1+model.nq];d.qvel[:]=state[1+model.nq:]
        mujoco.mj_kinematics(model,d)
        for k in fields:fields[k].append(getattr(d,k).copy())
    result.update({k:np.array(v) for k,v in fields.items()})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-root',type=Path,required=True)
    parser.add_argument('--tasks',nargs='+',default=['all'])
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    import mujoco
    if mujoco.__version__!='2.3.7':
        parser.error('Use the separate CPU reference MuJoCo2.3.7 environment')
    from libero.gpu.model import build_task_model
    from libero.gpu.host.adapters.environment.batch_env import _prepare_shared_fixtures
    from libero.gpu.host.build.task_catalog import task_catalog
    from libero.libero import get_libero_path
    from libero.libero.benchmark.init_state_io import load_init_states
    root=args.build_root.resolve();root.mkdir(parents=True,exist_ok=True)
    index=root/'TASK_INSTANCES.json'
    previous=json.loads(index.read_text()) if index.is_file() else {}
    records={row['key']:row for row in previous.get('tasks',[])}
    tasks=[t for t in task_catalog() if any(s in ('all',t['suite'],t['suite']+':'+str(t['task_id'])) for s in args.tasks)]
    if not tasks:parser.error('No tasks match the selectors')
    rows=[]
    for task in tasks:
        cached=records.get(task['key'])
        if cached and cached['seed']==args.seed and cached['bddl_sha256']==digest(task['bddl']):
            directory=root/cached['directory']
            inputs=root/'mjwarp/scene-inputs'/(directory.name+'.npz')
            receipt=inputs.with_suffix('.json')
            exported=json.loads(receipt.read_text()) if receipt.is_file() else {}
            if (inputs.is_file() and exported.get('data_sha256')==digest(inputs) and
                    exported.get('seed')==args.seed and exported.get('model_sha256')==cached['model_sha256'] and
                    exported.get('scene_sha256')==cached['files']['scene.xml'] and
                    exported.get('init_sha256')==cached['files']['initial_states.npy'] and all(
                    (directory/name).is_file() and digest(directory/name)==value for name,value in cached['files'].items())):
                rows.append(cached)
                print(json.dumps(dict(done=len(rows),total=len(tasks),task=task['key'],cache_hit=True,physics_integrations=0)),flush=True)
                continue
        directory=root/'scenes'/(task['suite']+'__'+Path(task['key']).stem+(f'__seed{args.seed}' if args.seed else ''))
        directory.mkdir(parents=True,exist_ok=True)
        built=build_task_model(task['bddl'],use_camera_obs=False,has_renderer=False,
            has_offscreen_renderer=False,hard_reset=False,
            static_model_setup=lambda shell,model:_prepare_shared_fixtures(shell,model,args.seed))
        model=built.native_model
        mujoco.mj_saveModel(model,str(directory/'model.mjb'),None)
        from .scene import relocate_assets
        (directory/'scene.xml').write_text(relocate_assets(built.xml,relative=True))
        source=Path(get_libero_path('init_states'))/task['suite']/(Path(task['key']).stem+'.pruned_init')
        bank=np.asarray(load_init_states(source),np.float64)
        if bank.ndim!=2 or bank.shape[0]<1 or bank.shape[1]!=1+model.nq+model.nv or not np.isfinite(bank).all():
            raise ValueError('Invalid official initial states: '+task['key'])
        np.save(directory/'initial_states.npy',bank,allow_pickle=False)
        files={k:digest(directory/k) for k in ('model.mjb','scene.xml','initial_states.npy')}
        row={k:task[k] for k in ('key','suite','task_id','name')}
        row.update(seed=args.seed,bddl_sha256=digest(task['bddl']),model_sha256=files['model.mjb'],
            dimensions=built.metadata['dimensions'],options=built.metadata['options'],
            static_arrays_sha256=built.metadata['static_arrays_sha256'],directory=str(directory.relative_to(root)),files=files,
            initial_states_shape=list(bank.shape))
        path=root/'mjwarp/scene-inputs'/(directory.name+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
        np.savez(path,**portable_inputs(model,bank))
        receipt=dict(task=task['key'],seed=args.seed,model_sha256=files['model.mjb'],
            scene_sha256=files['scene.xml'],init_sha256=files['initial_states.npy'],data_sha256=digest(path),
            exporter_sha256=digest(__file__),exporter_schema='cuda-fp32-catalog-v1',mujoco='2.3.7',
            library_sha256=digest(next(Path(mujoco.__file__).parent.glob('libmujoco.so*'))),
            physics_integrations=0,worlds=2)
        path.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
        rows.append(row)
        records[row['key']]=row
        print(json.dumps(dict(done=len(rows),total=len(tasks),task=task['key'],physics_integrations=0)),flush=True)
        del built,model,bank
        gc.collect()
    (root/'TASK_INSTANCES.json').write_text(json.dumps(dict(schema='libero-all-task-instances-v1',
        seed=args.seed,count=len(records),tasks=list(records.values())),indent=2)+'\n')


if __name__=='__main__':main()
