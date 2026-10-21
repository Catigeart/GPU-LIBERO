"""Modern MuJoCo scene binaries, prepared offline and verified before use.

The CPU2.3.7 MJB catalogue is never consumed by MuJoCo3.16. A hit is bound to
assets, actual imported/processed XML, options, engine, and preparation sources.
"""
import hashlib,json,os
from pathlib import Path
from functools import lru_cache
from libero.gpu.host.build.compilation import MissingCompiledProduct


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def engine_identity():
    import mujoco
    lib=next(Path(mujoco.__file__).parent.glob('libmujoco.so*'))
    return dict(python=mujoco.__version__,native=mujoco.mj_versionString(),library_sha256=digest(lib))


def binding(xml,options,task):
    import xml.etree.ElementTree as ET
    from libero.gpu.host.paths import GPU_ROOT
    assets=sorted({node.get('file') for node in ET.fromstring(xml).iter() if node.get('file')})
    return dict(engine=engine_identity(),task=task['key'],scene=task['files']['scene.xml'],
        imported_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),options=options,
        assets={p:digest(p) for p in assets},sources={name:digest(GPU_ROOT/name) for name in ('model/factory.py','model/cold_assets.py','host/backends/mjwarp/model.py','host/backends/mjwarp/scene_cache.py','host/backends/mjwarp/scene_inputs.py')})


def locate(root,xml,options,task,*,prepare,scene_parameters=None):
    record=binding(xml,options,task);record['scene_parameters']=scene_parameters;key=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest();folder=Path(root)/'mjwarp/models'/key;receipt=folder/'receipt.json';model=folder/'model.mjb'
    if receipt.is_file() and model.is_file():
        row=json.loads(receipt.read_text())
        if row['binding']!=record or digest(model)!=row['model_sha256']:raise MissingCompiledProduct('Modern scene cache changed; prepare offline: '+str(folder))
        return folder,model,row,record
    if not prepare:raise MissingCompiledProduct('Prepare the modern MJWarp scene offline; CPU2.3.7 MJB is incompatible: '+str(folder))
    return folder,None,None,record


def finish(folder,built,row,record):
    processed=hashlib.sha256(built.xml.encode()).hexdigest()
    if row is not None:
        if row['processed_xml_sha256']!=processed or row['model_sha256']!=built.metadata['model_sha256']:raise MissingCompiledProduct('Modern scene processed XML/model differs; prepare offline: '+str(folder))
        return
    import mujoco
    folder.mkdir(parents=True,exist_ok=True);tmp=folder/('model-'+str(os.getpid())+'.tmp.mjb');mujoco.mj_saveModel(built.native_model,str(tmp),None);value=digest(tmp)
    if value!=built.metadata['model_sha256']:
        tmp.unlink();raise RuntimeError('Modern scene serialization differs from prepared metadata')
    row=dict(binding=record,processed_xml_sha256=processed,model_sha256=value)
    os.replace(tmp,folder/'model.mjb');tmp=folder/('receipt-'+str(os.getpid())+'.tmp.json');tmp.write_text(json.dumps(row,indent=2)+'\n');os.replace(tmp,folder/'receipt.json')
