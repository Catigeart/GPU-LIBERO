"""Portable CPU catalogue inputs; accepts no exported device model arrays."""
import json
from pathlib import Path
import numpy as np
from libero.gpu.host.build.compilation import MissingCompiledProduct
from .catalog import digest
from ..mjwarp.scene_inputs import apply


def read(root,task,*,verify=False):
    path=Path(root)/'mjwarp/scene-inputs'/(Path(task['directory']).name+'.npz')
    receipt=path.with_suffix('.json')
    if not receipt.is_file():raise MissingCompiledProduct('Prepare portable CPU scene parameters in MuJoCo2.3.7: python -m libero.gpu.host.backends.cuda_fp32.catalog --build-root '+str(root))
    row=json.loads(receipt.read_text())
    expected=dict(task=task['key'],seed=task['seed'],model_sha256=task['files']['model.mjb'],
        scene_sha256=task['files']['scene.xml'],init_sha256=task['files']['initial_states.npy'],
        mujoco='2.3.7')
    if row.get('exporter_schema') not in (None, 'cuda-fp32-catalog-v1'):
        raise MissingCompiledProduct('Unsupported portable CPU scene format')
    if any(row.get(k)!=v for k,v in expected.items()) or (verify and digest(path)!=row['data_sha256']):
        raise MissingCompiledProduct('CPU scene input binding changed; prepare offline again')
    with np.load(path,allow_pickle=False) as a:data={k:a[k].copy() for k in a.files}
    n=task['dimensions']['nbody']
    if data['body_names'].shape!=(n,) or data['body_pos'].shape!=(n,3) or data['body_quat'].shape!=(n,4):
        raise ValueError('Portable CPU body layout differs')
    if not all(np.isfinite(data[k]).all() for k in ('body_pos','body_quat')):
        raise ValueError('Portable CPU poses are not finite')
    return data,row
