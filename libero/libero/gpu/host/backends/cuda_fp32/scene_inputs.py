"""Source-bound CPU catalogue inputs; accepts no exported device model arrays."""
import json
from pathlib import Path
import numpy as np
from libero.gpu.host.build.compilation import MissingCompiledProduct
from .catalog import digest
from ..mjwarp.scene_inputs import apply


def read(root,task):
    path=Path(root)/'mjwarp/scene-inputs'/(Path(task['directory']).name+'.npz')
    receipt=path.with_suffix('.json')
    if not receipt.is_file():raise MissingCompiledProduct('Prepare portable CPU scene parameters offline')
    row=json.loads(receipt.read_text())
    expected=dict(task=task['key'],seed=task['seed'],model_sha256=task['files']['model.mjb'],
        scene_sha256=task['files']['scene.xml'],init_sha256=task['files']['initial_states.npy'],
        mujoco='2.3.7',exporter_schema='cuda-fp32-catalog-v1',
        exporter_sha256=digest(Path(__file__).with_name('catalog.py')))
    if row.get('exporter_schema') is None:
        # The original exporter is a source-tree validation tool. Its pinned
        # digest keeps old CPU inputs admissible in wheels without importing
        # or installing the entire development-tool namespace.
        from libero.gpu.host.paths import GPU_ROOT
        pin=json.loads((GPU_ROOT/'data/cpu_scene_input_source_pin.json').read_text())
        expected.pop('exporter_schema')
        expected['exporter_sha256']=pin['sha256']
    if any(row.get(k)!=v for k,v in expected.items()) or digest(path)!=row['data_sha256']:
        raise MissingCompiledProduct('CPU scene input/source binding changed; prepare offline again')
    with np.load(path,allow_pickle=False) as a:data={k:a[k].copy() for k in a.files}
    n=task['dimensions']['nbody']
    if data['body_names'].shape!=(n,) or data['body_pos'].shape!=(n,3) or data['body_quat'].shape!=(n,4):
        raise ValueError('Portable CPU body layout differs')
    if not all(np.isfinite(data[k]).all() for k in ('body_pos','body_quat')):
        raise ValueError('Portable CPU poses are not finite')
    return data,row
