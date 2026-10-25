"""Native model and metadata transport in memory; source files stay file-backed."""
import hashlib
import os
from pathlib import Path


def model_bytes(model, expected_sha256=None):
    import mujoco
    import numpy as np
    # MuJoCo 2.3.7's file writer is substantially faster on large models than
    # its memory-buffer writer. A private anonymous RAM file uses that writer
    # without creating a disk path, cache, or persistent environment resource.
    # Serialize the live model every time: source-drift checks remain intact.
    fd = None
    if hasattr(os, 'memfd_create') and os.path.isdir('/proc/self/fd'):
        try:
            fd = os.memfd_create('libero-model', os.MFD_CLOEXEC)
        except OSError:
            pass  # Restricted/non-Linux runtimes keep the existing buffer API.
    if fd is None:
        buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
        mujoco.mj_saveModel(model, None, buffer)
        raw = buffer.tobytes()
    else:
        with os.fdopen(fd, 'rb') as source:
            mujoco.mj_saveModel(model, f'/proc/self/fd/{source.fileno()}', None)
            raw = source.read()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('actual native model bytes differ from declared identity')
    return raw


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


def source_catalog(records):
    """JSON-safe diagnostics; never export live model owners or payloads implicitly."""
    return [dict(path=record['path'], sha256=record['sha256']) if 'path' in record else
            dict(storage='native-model' if 'model' in record else 'bytes', sha256=record['sha256'])
            for record in records]


def load_native_model(*, expected_sha256, model=None, model_path=None):
    """Retain a live model or load an explicitly identified external MJB.

    An existing MjModel needs no serialization round trip. Callers validate its
    dimensions, solver options and topology before preparing device workspaces.
    """
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
    return model, path


def actual_decoder_record(built):
    """Describe the existing decoder for diagnostics; do not authenticate its code.

    The model factory already records its source digest. Model dimensions and
    static arrays are checked by intake, independently of this description.
    """
    import inspect
    pin = built.metadata['decoder_pin']
    return {'path': str(Path(inspect.getfile(type(built))).resolve()),
            'sha256': pin['source_sha256']}
