"""Source-bound portable parameters from the original prepared CPU scene.

Old MJBs are decoded only by the offline CPU2.3.7 exporter. Runtime borrows no
CPU simulator and never resamples fixed-fixture poses from a changed sampler.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from libero.gpu.host.build.compilation import MissingCompiledProduct


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(root,task,*,verify=False):
    path=Path(root)/'mjwarp/scene-inputs'/(Path(task['directory']).name+'.npz');receipt=path.with_suffix('.json')
    if not path.is_file() or not receipt.is_file():raise MissingCompiledProduct('Export original CPU2.3.7 scene parameters offline with tools/validate_gpu.py mjwarp-scene-inputs')
    record=json.loads(receipt.read_text());expected=dict(task=task['key'],seed=task['seed'],model_sha256=task['files']['model.mjb'],scene_sha256=task['files']['scene.xml'],init_sha256=task['files']['initial_states.npy'],mujoco='2.3.7')
    if record.get('exporter_schema') not in (None, 'cuda-fp32-catalog-v1'):
        raise MissingCompiledProduct('Unsupported portable CPU scene format')
    if any(record.get(k)!=v for k,v in expected.items()) or (verify and digest(path)!=record['data_sha256']):raise MissingCompiledProduct('Original prepared scene parameter binding differs; export offline again: '+str(path))
    with np.load(path,allow_pickle=False) as archive:data={k:archive[k].copy() for k in archive.files}
    nb=task['dimensions']['nbody']
    if data['body_names'].shape!=(nb,) or data['body_pos'].shape!=(nb,3) or data['body_quat'].shape!=(nb,4) or not np.all(np.isfinite(data['body_pos'])) or not np.all(np.isfinite(data['body_quat'])):raise ValueError('Original fixed pose layout/values differ')
    return data,record


def apply(shell,model,data):
    import mujoco
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) or '' for i in range(model.nbody)]
    if names!=data['body_names'].tolist():raise ValueError('Modern body names/index space differs from original scene')
    for key in ('body_parentid','jnt_type','jnt_bodyid','jnt_qposadr','jnt_dofadr','dof_bodyid'):
        if not np.array_equal(getattr(model,key),data[key]):raise ValueError('Modern joint/body index mapping differs: '+key)
    for fixture in shell.fixtures_dict.values():
        body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,fixture.root_body)
        if body<0:raise ValueError('Fixture root body is absent: '+fixture.root_body)
        model.body_pos[body]=data['body_pos'][body]
        model.body_quat[body]=data['body_quat'][body]
