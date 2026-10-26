"""Load the explicit CUDA model instance. Never generates or compiles source."""
import ctypes as ct
import os
from pathlib import Path

from libero.gpu.host.paths import GPU_ROOT
from libero.gpu.host.build.compilation import MissingCompiledProduct
from libero.gpu.host.build.project_paths import project_paths
from .native_product import load_product


def load(config, *, build_root, mathdx_root):
    mode = os.environ.get('LIBERO_GPU_BUILD_MODE', 'dev')
    if mode not in ('dev', 'release'):
        raise ValueError('LIBERO_GPU_BUILD_MODE must be dev or release')
    folder = project_paths(build_root).cmake / f"sm{config['architecture']}-{mode}"
    path = folder / f"native_newton_nv{config['nv']}.so"
    if not path.is_file():
        raise MissingCompiledProduct('Prepare all native CUDA instances with '
            'python -m libero.gpu.build --precision fp64 --build-mode '+mode+
            '; missing '+str(path))
    library, resources = load_product(path, parameter_bytes=config['parameter_bytes'],
        block_threads=config['block_threads'], shared_bytes=config['shared_bytes'],
        architecture=config['architecture'], combined_entry=True)
    query = library.native_newton_configuration
    query.argtypes = [ct.c_int]
    query.restype = ct.c_int
    expected = (config['nv'], config['nM'], config['rows'], config['contacts'],
        config['maxiter'], config['block_threads'], config['precision_bits'],
        config['architecture'], int(config['warmstart']), int(config['use_dx']),
        int(config['use_blas']), int(config['phase_profiling']))
    if tuple(query(i) for i in range(len(expected))) != expected:
        raise ValueError('Loaded native Newton model/configuration differs; rebuild offline')
    return dict(library=library, resources=resources, metadata=dict(
        schema='libero-native-cmake-newton-v1', library_file=str(path),
        source=str(GPU_ROOT/'cuda/newton/native_instances.cu'),
        mathdx_root=str(mathdx_root), build_system='CMake', source_generation=False,
        compile_commands_executed=[]))
