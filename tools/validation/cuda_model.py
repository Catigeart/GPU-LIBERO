"""CPU-only independent model comparison; reference export is an explicit mode."""
import hashlib
import json
from pathlib import Path
from libero.gpu.host.build.task_catalog import load_catalog
import sys
import numpy as np


def add_arguments(parser):
    parser.add_argument('--build-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tasks',nargs='+',default=['all'])
    parser.add_argument('--reference-export',action='store_true')
    parser.add_argument('--candidate-scene-cache',type=Path,help='compare separately compiled models from this environment, not the reference MJB')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(args,task,folder):
    import warp as wp
    import mujoco_warp as mjw
    import mujoco
    from libero.libero import get_libero_path
    from libero.gpu.host.backends.mjwarp.model import build_scene,capacity_budget
    from libero.gpu.host.backends.cuda_fp32 import layout_specs as specs
    if (folder/'reference.json').exists():raise RuntimeError('Preserve completed reference export')
    with wp.ScopedDevice('cpu'):
        built,m,*_=build_scene(args.build_root,Path(get_libero_path('bddl_files'))/task['key'],task['seed'],
            dict(ignore_done=True,use_camera_obs=False,has_renderer=False,has_offscreen_renderer=False,hard_reset=False),prepare=False)
        capacity=capacity_budget(built.native_model);d=mjw.make_data(built.native_model,nworld=2,nccdmax=capacity['nconmax'],**capacity)
        arrays={k:getattr(m,k).numpy() for k in specs.MODEL_ARRAYS}
        for k in specs.OPTION_ARRAYS:arrays['opt.'+k]=getattr(m.opt,k).numpy()
        for k in specs.STATISTIC_ARRAYS:arrays['stat.'+k]=getattr(m.stat,k).numpy()
        for i,a in enumerate(m.body_tree):arrays['body_tree.'+str(i)]=a.numpy()
        for i,t in enumerate(m.M_tiles):arrays['M_tiles.'+str(i)+'.adr']=t.adr.numpy();arrays['M_tiles.'+str(i)+'.elemid']=t.elemid.numpy()
        for owner,prefix,names in ((d,'data.',specs.DATA_ARRAYS),(d.contact,'contact.',specs.CONTACT_ARRAYS),(d.efc,'efc.',specs.CONSTRAINT_ARRAYS)):
            for k in names:
                a=getattr(owner,k)
                if isinstance(a,wp.array):arrays[prefix+k]=a.numpy()
        folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/'reference.npz',**arrays)
        mujoco.mj_saveModel(built.native_model,str(folder/'model.mjb'),None)
        scalars={k:getattr(m,k) for k in specs.MODEL_SCALARS}
        record=dict(task=task['key'],model_sha256=digest(folder/'model.mjb'),arrays_sha256=digest(folder/'reference.npz'),
            scalars=scalars,option_scalars={k:getattr(m.opt,k) for k in specs.OPTION_SCALARS},
            tiles=[t.size for t in m.M_tiles],capacity=capacity,mujoco=mujoco.__version__,worlds=2,
            reference_commit='c384ac51df67eb92ab7657cc472ff7494dc036bf')
        (folder/'reference.json').write_text(json.dumps(record,indent=2,default=lambda x:x.item())+'\n')


def check(folder,candidate=None):
    from .cuda_runtime import ForbidWarp
    sys.meta_path.insert(0,ForbidWarp())
    import mujoco
    from libero.gpu.host.backends.cuda_fp32.model import prepare,host_array
    from libero.gpu.host.backends.cuda_fp32 import layout_specs as specs
    record=json.loads((folder/'reference.json').read_text())
    if digest(folder/'model.mjb')!=record['model_sha256'] or digest(folder/'reference.npz')!=record['arrays_sha256']:raise RuntimeError('Changed reference inputs')
    actual=folder/'model.mjb' if candidate is None else candidate
    host=mujoco.MjModel.from_binary_path(str(actual));m=prepare(host);issues=[]
    with np.load(folder/'reference.npz',allow_pickle=False) as arrays:
        def compare(name,actual):
            expected=arrays[name]
            if actual.shape!=expected.shape or actual.dtype!=expected.dtype or not np.array_equal(actual,expected):
                issues.append(dict(field=name,shape=actual.shape,expected_shape=expected.shape,
                    error=None if actual.shape!=expected.shape else float(np.max(abs(actual.astype(float)-expected.astype(float)),initial=0))))
        for k,layout in specs.MODEL_ARRAYS.items():
            try:compare(k,host_array(getattr(m,k,None),layout,m._sizes)[0])
            except Exception as e:issues.append(dict(field=k,error=repr(e)))
        for k,expected in record['scalars'].items():
            if getattr(m,k)!=expected:issues.append(dict(field=k,actual=getattr(m,k),expected=expected))
        for k,layout in specs.OPTION_ARRAYS.items():compare('opt.'+k,host_array(getattr(m.opt,k),layout,{'*':1})[0])
        for k,expected in record['option_scalars'].items():
            if getattr(m.opt,k)!=expected:issues.append(dict(field='opt.'+k,actual=getattr(m.opt,k),expected=expected))
        for i,a in enumerate(m.body_tree):compare('body_tree.'+str(i),a)
        for i,t in enumerate(m.M_tiles):
            if t.size!=record['tiles'][i]:issues.append(dict(field='tile size',index=i))
            compare('M_tiles.'+str(i)+'.adr',t.adr);compare('M_tiles.'+str(i)+'.elemid',t.elemid)
    return dict(task=record['task'],passed=not issues,issues=issues,model_arrays=len(specs.MODEL_ARRAYS),warp_imported=False,
                reference_sha256=digest(folder/'reference.json'),physics_integrations=0,
                actual_model=str(actual),actual_model_sha256=digest(actual))


def main(args):
    from .common import select_tasks
    from libero.gpu import precision
    precision.select('fp32')
    tasks=select_tasks(load_catalog(args.build_root),args.tasks);rows=[]
    candidates={}
    if args.candidate_scene_cache:
        import robosuite
        import hashlib
        site=str(Path(robosuite.__file__).resolve().parent)
        options=dict(use_camera_obs=False,has_renderer=False,has_offscreen_renderer=False,hard_reset=False,ignore_done=True)
        for receipt in args.candidate_scene_cache.glob('*/receipt.json'):
            value=json.loads(receipt.read_text());binding=value['binding']
            if binding['options']!=options or not any(p.startswith(site+'/') for p in binding['assets']):continue
            model=receipt.with_name('model.mjb')
            if digest(model)!=value['model_sha256']:raise RuntimeError('Compiled candidate model changed')
            key=binding['task']
            if key in candidates and candidates[key]!=model:raise RuntimeError('Ambiguous candidate scene: '+key)
            candidates[key]=model
    for task in tasks:
        folder=args.output/(task['suite']+'-'+str(task['task_id']))
        if args.reference_export:export(args,task,folder);row=dict(task=task['key'],exported=True)
        else:
            if args.candidate_scene_cache and task['key'] not in candidates:raise ValueError('Candidate scene is absent: '+task['key'])
            row=check(folder,candidates.get(task['key']))
        rows.append(row);print(json.dumps(row,default=lambda x:x.item()),flush=True)
    name='candidate-model-check.json' if args.candidate_scene_cache else ('export-index.json' if args.reference_export else 'model-check.json')
    (args.output/name).write_text(json.dumps(rows,indent=2,default=lambda x:x.item())+'\n')
    return 0 if args.reference_export or all(r['passed'] for r in rows) else 2
