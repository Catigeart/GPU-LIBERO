"""Native model and metadata transport in memory; source files stay file-backed."""
import hashlib
from pathlib import Path


def model_bytes(model, expected_sha256=None):
    import mujoco
    import numpy as np
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buffer)
    raw = buffer.tobytes()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('actual native model bytes differ from declared identity')
    return raw


def memory_record(raw):
    if type(raw) is not bytes:
        raise TypeError('immutable bytes required for an in-memory source')
    return dict(data=raw, sha256=hashlib.sha256(raw).hexdigest())


def source_ref(record):
    if 'model' in record:
        import mujoco
        if not isinstance(record['model'], mujoco.MjModel):
            raise TypeError('actual native model owner required')
        return record['model']
    if 'data' in record:
        if type(record['data']) is not bytes:
            raise TypeError('immutable bytes required for an in-memory source')
        return record['data']
    return str(Path(record['path']).resolve())


def source_bytes(reference):
    if type(reference) is bytes:return reference
    if type(reference) is str:return Path(reference).read_bytes()
    return model_bytes(reference)


def checked_records(records):
    result = tuple((source_ref(record), record['sha256']) for record in records)
    for reference, digest in result:
        if hashlib.sha256(source_bytes(reference)).hexdigest() != digest:
            raise ValueError('native/source bytes drift: ' + (reference if type(reference) is str else 'in-memory source'))
    return result


def source_catalog(records):
    """JSON-safe diagnostics; never export live model owners or payloads implicitly."""
    return [dict(path=record['path'], sha256=record['sha256']) if 'path' in record else
            dict(storage='native-model' if 'model' in record else 'bytes', sha256=record['sha256'])
            for record in records]


def load_native_model(*, expected_sha256, model=None, model_path=None):
    """Accept an actual owner or a legacy explicit MJB, with the same identity gate."""
    import mujoco
    if (model is None) == (model_path is None):
        raise ValueError('provide exactly one native model owner or MJB path')
    path = None
    if model_path is not None:
        path = str(Path(model_path).resolve())
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected_sha256:
            raise ValueError('native MJB identity differs from ModelSpec')
        model = mujoco.MjModel.from_binary_path(path)
    if not isinstance(model, mujoco.MjModel):
        raise TypeError('actual native MjModel required')
    model_bytes(model, expected_sha256)
    return model, path
