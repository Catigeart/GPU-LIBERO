"""Offline preparation of the public single-environment FP64 entry.

This profile prepares ordinary native products, shares the unified project's
constraint sources/Warp cache, and warms one ordinary control step to prepare
lazy modules. Runtime construction neither signs products nor invokes compilers.
"""
import json
import os
from pathlib import Path
import uuid


def _write_status(folder, result):
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / ('.BUILD-' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x') as stream:
            stream.write(json.dumps(result, indent=2) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, folder / 'BUILD.json')
        fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(root, tasks, *, device='cuda:0', progress=None):
    import numpy as np
    import libero.gpu.precision as scalar
    from libero.libero.envs import OffScreenRenderEnv
    from libero.gpu.host.runtime.core.product_paths import build_project
    from libero.gpu.host.runtime.core.specialization_cache import index_packages
    from .aot import configure_warp_cache
    from .compilation import compilation_scope, install_warp_guard
    from .product_cache import require_product_persistence
    if scalar.bits != 64:
        raise ValueError('The native compatibility build requires precision=fp64')
    root = Path(root).expanduser().resolve()
    rows = []
    result = dict(schema='libero-compatibility-build-v1', status='preparing', real_bits=64,
                  product_storage='native-v1', project=str(root), tasks=rows,
                  physics_executed_for_module_warmup=True)
    folder = root / 'compatibility'
    # Invalidate an old ready marker before preparing a new task selection.
    # Interruption leaves 'preparing', never a false claim of completeness.
    _write_status(folder, result)
    try:
        with build_project(root) as project, compilation_scope(True), require_product_persistence():
            configure_warp_cache(root, seed_cache=True)
            install_warp_guard()
            for task in tasks:
                if progress is not None:
                    progress(dict(phase='compatibility-prepare', task=task['key']))
                start = len(project['events'])
                env = None
                try:
                    env = OffScreenRenderEnv(bddl_file_name=task['bddl'], use_camera_obs=False,
                                             ignore_done=True, device=device)
                    env.step(np.zeros(env.env.action_dim, dtype=np.float64))
                    rows.append(dict(task=task['key'], model_sha256=env.env.built.metadata['model_sha256'],
                                     events=list(project['events'][start:]), warmup_control_steps=1))
                finally:
                    if env is not None:
                        env.close()
            # Both original and placement-generation packages must be load-only
            # discoverable. Publish only after all required products persist.
            index_packages(root / 'specializations')
    except BaseException as error:
        result.update(status='failed', error=dict(type=type(error).__name__, message=str(error)))
        try:
            _write_status(folder, result)
        except OSError as status_error:
            error.add_note('Could not publish failed build status: ' + str(status_error))
        raise
    result['status'] = 'complete'
    _write_status(folder, result)
    return result
