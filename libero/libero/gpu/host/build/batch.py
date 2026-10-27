"""CMake compilation of declared CUDA model instances; no environment imports."""
import subprocess
from pathlib import Path

from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
from .project_paths import project_paths
from libero.gpu.host.runtime.core.product_paths import mathdx_root as locate_sdk, nvcc_path


def build_modes(mode):
    if mode == 'all':
        return ('dev', 'release')
    if mode not in ('dev', 'release'):
        raise ValueError('build mode must be dev, release or all')
    return (mode,)


def compile_project(root=None, *, precision='fp64', modes=('dev',), arch=80,
                    mathdx_root=None, jobs=2, ordered_reductions=False):
    if precision not in ('fp64', 'fp32'):
        raise ValueError('precision must be fp64 or fp32')
    if ordered_reductions and precision != 'fp64':
        raise ValueError('ordered comparison products require FP64')
    paths = project_paths(root)
    sdk = Path(mathdx_root or locate_sdk()).resolve()
    compiler = nvcc_path()
    source = REPO_ROOT if (REPO_ROOT/'CMakeLists.txt').is_file() else GPU_ROOT
    target = ('flash_native_batch_ordered_all' if ordered_reductions else
              'flash_native_batch_all') if precision == 'fp64' else 'flash_cuda_fp32_all'
    result = []
    for mode in modes:
        folder = paths.cmake/f'sm{arch}-{mode}'
        configure = ['cmake', '-S', str(source), '-B', str(folder), '-G', 'Ninja',
            '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CUDA_COMPILER='+str(compiler),
            '-DFLASH_CUDA_ARCH='+str(arch), '-DFLASH_MATHDX_ROOT='+str(sdk),
            '-DFLASH_BUILD_NATIVE='+('ON' if precision == 'fp64' else 'OFF'),
            '-DFLASH_FAST_COMPILE='+('ON' if mode == 'dev' else 'OFF')]
        build = ['cmake', '--build', str(folder), '--target', target, '--parallel', str(jobs)]
        subprocess.run(configure, check=True)
        subprocess.run(build, check=True)
        result.append(dict(mode=mode, directory=str(folder), target=target))
    return dict(status='complete', precision=precision, builds=result,
                physics_integrations=0, environment_constructions=0)
