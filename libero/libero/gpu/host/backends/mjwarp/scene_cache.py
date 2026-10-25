"""Fixed modern MuJoCo scenes: direct runtime loads, offline content validation."""
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from libero.gpu.host.build.compilation import MissingCompiledProduct


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def engine_identity():
    import mujoco
    return dict(python=mujoco.__version__, native=mujoco.mj_versionString())


def binding(xml, options, task):
    # Hash the already-loaded XML/options, never asset or library files here.
    return dict(engine=engine_identity(), task=task['key'], scene=task['files']['scene.xml'],
                imported_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(), options=options)


def _data_binding(record):
    record = dict(record)
    record.pop('sources', None)
    record.pop('assets', None)
    engine = dict(record['engine'])
    engine.pop('library_sha256', None)
    record['engine'] = engine
    parameters = record.get('scene_parameters')
    if parameters is not None:
        parameters = dict(parameters)
        for field in ('exporter_sha256', 'exporter_schema', 'library_sha256'):
            parameters.pop(field, None)
        record['scene_parameters'] = parameters
    return record


def offline_inputs(xml):
    import mujoco
    assets = sorted({node.get('file') for node in ET.fromstring(xml).iter() if node.get('file')})
    library = next(Path(mujoco.__file__).parent.glob('libmujoco.so*'))
    return dict(assets={p: digest(p) for p in assets}, library_sha256=digest(library))


def cache_directory(root, xml, options, task, scene_parameters=None):
    record = _data_binding(dict(binding(xml, options, task), scene_parameters=scene_parameters))
    key = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    return Path(root) / 'mjwarp/models' / key, record


def _validate_offline(model, row, inputs):
    expected = row.get('offline_inputs')
    if expected is None:  # Previous receipts embedded content hashes in the key.
        old = row['binding']
        expected = dict(assets=old['assets'], library_sha256=old['engine']['library_sha256'])
    if expected != inputs or digest(model) != row['model_sha256']:
        raise MissingCompiledProduct('Modern scene cache changed; prepare fresh offline: ' + str(model.parent))


def locate(root, xml, options, task, *, prepare, scene_parameters=None):
    folder, record = cache_directory(root, xml, options, task, scene_parameters)
    receipt, model = folder / 'receipt.json', folder / 'model.mjb'
    inputs = offline_inputs(xml) if prepare else None
    if receipt.is_file() and model.is_file():
        row = json.loads(receipt.read_text())
        if row.get('schema') != 'modern-scene-v2' or _data_binding(row['binding']) != record:
            raise MissingCompiledProduct('Modern scene binding changed; prepare offline: ' + str(folder))
        if prepare:
            _validate_offline(model, row, inputs)
        return folder, model, row, record
    if not prepare:
        raise MissingCompiledProduct('Prepare the modern scene offline (including legacy-cache migration); '
                                     'CPU2.3.7 MJB is incompatible: ' + str(folder))
    # Legacy discovery is an offline migration only; never scan during training.
    for old in sorted(folder.parent.glob('*/receipt.json')):
        if old == receipt:
            continue
        row = json.loads(old.read_text())
        if _data_binding(row['binding']) != record:
            continue
        model = old.parent / 'model.mjb'
        _validate_offline(model, row, inputs)
        # The caller imports this binary, then publishes the current format.
        return folder, model, row, record
    return folder, None, None, record


def finish(folder, built, row, record, *, prepare=False, xml=None):
    if not prepare:
        if row is None:
            raise MissingCompiledProduct('Runtime scene publication is disabled; prepare offline')
        return
    processed = hashlib.sha256(built.xml.encode()).hexdigest()
    if row is not None:
        if row['processed_xml_sha256'] != processed or row['model_sha256'] != built.metadata['model_sha256']:
            raise MissingCompiledProduct('Modern scene processed XML/model differs; prepare offline: ' + str(folder))
        if (folder / 'receipt.json').is_file() and (folder / 'model.mjb').is_file():
            return
    import mujoco
    inputs = offline_inputs(xml)
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / ('model-' + str(os.getpid()) + '.tmp.mjb')
    tmp_receipt = folder / ('receipt-' + str(os.getpid()) + '.tmp.json')
    try:
        mujoco.mj_saveModel(built.native_model, str(tmp), None)
        value = digest(tmp)
        if value != built.metadata['model_sha256']:
            raise RuntimeError('Modern scene serialization differs from prepared metadata')
        row = dict(schema='modern-scene-v2', binding=record, offline_inputs=inputs,
                   processed_xml_sha256=processed, model_sha256=value)
        tmp_receipt.write_text(json.dumps(row, indent=2) + '\n')
        os.replace(tmp, folder / 'model.mjb')
        os.replace(tmp_receipt, folder / 'receipt.json')
    finally:
        tmp.unlink(missing_ok=True)
        tmp_receipt.unlink(missing_ok=True)
