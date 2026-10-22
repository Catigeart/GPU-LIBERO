"""CMake/Ninja ownership of the maintained substep and vendor CUDA units."""
import libero.gpu.precision as scalar
import hashlib
import json
from pathlib import Path
import subprocess
import time

from .compilation import MissingCompiledProduct, require_compilation
from .project_paths import cmake_root
from libero.gpu.host.runtime.core.product_paths import nvcc_path, record_build


def _write_changed(path, text):
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def prepare(source, root, sdk, arch, *, fast_compile=False, device_lto=False, profile_newton=False,
            nv=39, atomic_reductions=None, single_solve=False):
    if type(single_solve) is not bool or single_solve and scalar.bits != 32:
        raise ValueError('single_solve requires an explicit FP32 candidate')
    from .substep_catalog import select_variant
    variant = select_variant(nv)
    source = Path(source).resolve()
    sdk = Path(sdk).resolve()
    candidates = [p for p in source.parents if (p / 'CMakeLists.txt').is_file()]
    project = next((p for p in candidates if (p / 'setup.py').is_file()), None)
    if project is None:
        # Wheels retain the GPU package's own CMakeLists.txt; a checkout
        # prefers the repository entry above it.
        project = next(iter(candidates), None)
    if project is None:
        raise FileNotFoundError('No maintained CMake project for CUDA source: ' + str(source))
    compiler = Path(nvcc_path()).resolve()
    profile = ('dev' if fast_compile else 'release') + ('-lto' if device_lto else '')
    root = cmake_root(root)
    folder = root / f'sm{arch}-{profile}'
    folder.mkdir(parents=True, exist_ok=True)
    # None follows the CMake option (ON for a new tree). False selects the
    # explicitly ordered comparison binary in the same configured build tree.
    default_atomic = True
    default_cache = True
    default_transpose = 0
    threads = 64
    cache = folder / 'CMakeCache.txt'
    if cache.exists():
        for line in cache.read_text().splitlines():
            if line.startswith('FLASH_NEWTON_ATOMIC_REDUCTIONS:BOOL='):
                default_atomic = line.split('=', 1)[1] not in ('OFF', 'FALSE', '0', '')
            elif line.startswith('FLASH_NEWTON_JACOBIAN_CACHE:BOOL='):
                default_cache = line.split('=', 1)[1] not in ('OFF', 'FALSE', '0', '')
            elif line.startswith('FLASH_NEWTON_TRANSPOSE_METHOD:STRING='):
                default_transpose = int(line.split('=', 1)[1])
            elif line.startswith('FLASH_SUBSTEP_THREADS:STRING='):
                threads = int(line.split('=', 1)[1])
    ordered = atomic_reductions is False
    if atomic_reductions is True:
        default_atomic = True
    # Source contents do not determine the build directory. Ninja's compiler
    # depfiles track each translation unit and its actual included headers.
    config = dict(product='single-physical-substep', project=str(project), compiler=str(compiler), sdk=str(sdk),
                  schedule='step1-osc-step2' if single_solve else 'cpu-three-forwards',
                  solves_per_substep=1 if single_solve else 3, single_solve=single_solve,
                  arch=int(arch), real_bits=scalar.bits, fast_compile=bool(fast_compile), device_lto=bool(device_lto),
                  profile_newton=bool(profile_newton), variant=variant.__dict__,
                  atomic_reductions=False if ordered else default_atomic,
                  default_atomic_reductions=default_atomic,
                  jacobian_cache=default_cache, transpose_method=default_transpose, threads_per_world=threads)
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    configure = ['cmake', '-S', str(project), '-B', str(folder), '-G', 'Ninja',
                 '-DCMAKE_BUILD_TYPE=Release', f'-DCMAKE_CUDA_COMPILER={compiler}',
                 f'-DFLASH_CUDA_ARCH={arch}', f'-DFLASH_MATHDX_ROOT={sdk}',
                 '-DFLASH_FAST_COMPILE=' + ('ON' if fast_compile else 'OFF'),
                 '-DFLASH_GLOBAL_DEVICE_LTO=' + ('ON' if device_lto else 'OFF'),
                 '-DFLASH_NEWTON_ATOMIC_REDUCTIONS=' + ('ON' if default_atomic else 'OFF'),
                 '-DFLASH_NEWTON_JACOBIAN_CACHE=' + ('ON' if default_cache else 'OFF'),
                 f'-DFLASH_NEWTON_TRANSPOSE_METHOD={default_transpose}',
                 f'-DFLASH_SUBSTEP_THREADS={threads}']
    product = variant.product(profile_newton) + ('_fp32' if scalar.bits == 32 else '') + ('_ordered' if ordered else '')
    target = 'flash_' + product
    library = product + '.so'
    # Three compile slots plus one independent vendor/final link slot.
    build = ['cmake', '--build', str(folder), '--target', target, '--parallel', '4']
    result = dict(identity=identity, config=config, source=str(source), folder=str(folder),
                  library=str(folder / library), target=target, commands=[configure, build],
                  compilation_database=str(folder / 'compile_commands.json'),
                  source_generation=False, build_system='CMake/Ninja',
                  translation_units=11 if config['atomic_reductions'] else 8)
    manifest_name = (('manifest-newton-profile' if profile_newton else 'manifest') + variant.suffix +
                     ('-single-solve' if single_solve else '') +
                     ('-ordered' if ordered else '') + ('-fp32' if scalar.bits == 32 else '') + '.json')
    _write_changed(folder / manifest_name,
                   json.dumps(result, indent=2))
    return result


def prepare_staged(source, root, sdk, arch, **options):
    """Regional FP64 product, sharing the existing CMake build tree."""
    if scalar.bits != 64 or options.get('single_solve') or options.get('profile_newton'):
        raise ValueError('staged substeps require FP64, three solves and no fused profiler')
    manifest = prepare(source, root, sdk, arch, **options)
    if manifest['config']['threads_per_world'] != 64:
        raise ValueError('initial staged baseline requires 64-thread native SDK objects')
    manifest['config'] = dict(manifest['config'], execution='staged-motion-ABC',
                              motion_organization='branch8-native-fp64',
                              motion_schedule='entry-first', workspace='regional')
    manifest['translation_units'] += 3  # motion/prepare/Newton/Post replace one fused object
    manifest['target'] = manifest['target'].replace('flash_substep', 'flash_substep_staged', 1)
    manifest['library'] = str(Path(manifest['library']).with_name(
        Path(manifest['library']).name.replace('substep', 'substep_staged', 1)))
    manifest['commands'][1][manifest['commands'][1].index('--target') + 1] = manifest['target']
    manifest['identity'] = hashlib.sha256(json.dumps(manifest['config'], sort_keys=True).encode()).hexdigest()
    name = Path(manifest['library']).stem.replace('substep_staged', 'manifest-staged', 1) + '.json'
    _write_changed(Path(manifest['folder']) / name, json.dumps(manifest, indent=2))
    return manifest


def prepare_motion(source, root, sdk, arch, **options):
    """FP64 motion producer followed by the remaining native fused substep."""
    if scalar.bits != 64 or options.get('single_solve') or options.get('profile_newton'):
        raise ValueError('peeled motion requires FP64, three solves and no fused profiler')
    manifest = prepare(source, root, sdk, arch, **options)
    if manifest['config']['threads_per_world'] != 64:
        raise ValueError('peeled motion retains 64-thread native Newton/OSC objects')
    manifest['config'] = dict(manifest['config'], execution='motion-residual',
                              motion_organization='fp32-warp-branch-fp64', motion_schedule='entry-first')
    manifest['translation_units'] += 1
    manifest['target'] = manifest['target'].replace('flash_substep', 'flash_substep_motion', 1)
    manifest['library'] = str(Path(manifest['library']).with_name(
        Path(manifest['library']).name.replace('substep', 'substep_motion', 1)))
    manifest['commands'][1][manifest['commands'][1].index('--target') + 1] = manifest['target']
    manifest['identity'] = hashlib.sha256(json.dumps(manifest['config'], sort_keys=True).encode()).hexdigest()
    name = Path(manifest['library']).stem.replace('substep_motion', 'manifest-motion', 1) + '.json'
    _write_changed(Path(manifest['folder']) / name, json.dumps(manifest, indent=2))
    return manifest


def prepare_all(source, root, sdk, arch, *, staged=False, **options):
    """One offline target for the entire task catalogue; no per-task builds."""
    from .substep_catalog import variants
    manifest = (prepare_staged if staged else prepare)(source, root, sdk, arch, **options)
    profiled = manifest['config']['profile_newton']
    ordered = options.get('atomic_reductions') is False
    target = ('flash_substeps_ordered_profile_all' if profiled else 'flash_substeps_ordered_all') if ordered else (
        'flash_substeps_profile_all' if profiled else 'flash_substeps_all')
    if scalar.bits == 32:
        target += '_fp32'
    if staged:
        target = 'flash_staged_substeps_ordered_all' if ordered else 'flash_staged_substeps_all'
    manifest['target'] = target
    command = manifest['commands'][1]
    command[command.index('--target')+1] = target
    manifest['libraries'] = [str(Path(manifest['folder'])/(v.product(profiled)+('_fp32' if scalar.bits == 32 else '')+
                            ('_ordered' if ordered else '')+'.so')) for v in variants()]
    if staged:
        manifest['libraries'] = [str(Path(p).with_name(Path(p).name.replace('substep', 'substep_staged', 1)))
                                 for p in manifest['libraries']]
    manifest['translation_units'] = (8 if staged else 4)*len(variants()) + int(staged) + (7 if manifest['config']['atomic_reductions'] else 4)
    manifest['identity'] = hashlib.sha256((manifest['identity']+target).encode()).hexdigest()
    manifest_name = 'manifest-all' + ('-fp32' if scalar.bits == 32 else '') + ('-ordered' if ordered else '') + ('-profile' if profiled else '') + '.json'
    if staged:
        manifest_name = manifest_name.replace('manifest-all', 'manifest-staged-all')
    _write_changed(Path(manifest['folder'])/manifest_name,json.dumps(manifest,indent=2))
    return manifest


def _configured(manifest):
    cache = Path(manifest['folder']) / 'CMakeCache.txt'
    if not cache.exists():
        return False
    entries = {}
    for line in cache.read_text().splitlines():
        if ':' in line and '=' in line and not line.startswith(('#', '//')):
            key, value = line.split('=', 1)
            entries[key.split(':', 1)[0]] = value
    c = manifest['config']
    return all(entries.get(key) == value for key, value in {
        'CMAKE_CUDA_COMPILER': c['compiler'], 'FLASH_MATHDX_ROOT': c['sdk'],
        'CMAKE_BUILD_TYPE': 'Release', 'CMAKE_HOME_DIRECTORY': c['project'],
        'FLASH_CUDA_ARCH': str(c['arch']),
        'FLASH_SUBSTEP_THREADS': str(c['threads_per_world']),
        'FLASH_FAST_COMPILE': 'ON' if c['fast_compile'] else 'OFF',
        'FLASH_GLOBAL_DEVICE_LTO': 'ON' if c['device_lto'] else 'OFF',
        'FLASH_NEWTON_ATOMIC_REDUCTIONS': 'ON' if c['default_atomic_reductions'] else 'OFF',
        'FLASH_NEWTON_JACOBIAN_CACHE': 'ON' if c.get('jacobian_cache', True) else 'OFF',
        'FLASH_NEWTON_TRANSPOSE_METHOD': str(c.get('transpose_method', 0)),
    }.items())


def finish(manifest, *, allow_build=False):
    folder = Path(manifest['folder'])
    if not allow_build:
        if _configured(manifest) and Path(manifest['library']).is_file():
            # Dry-run only: reject stale products without executing configure,
            # compilation, or linking during environment construction.
            check = subprocess.run(['ninja', '-C', str(folder), '-n', manifest.get('target','flash_substep')],
                                   capture_output=True, text=True)
            if check.returncode == 0 and 'ninja: no work to do.' in check.stdout:
                return manifest
        raise MissingCompiledProduct('CMake substep product is missing or stale: ' +
                                     str(folder) + '. Build it offline before creating the environment.')
    require_compilation('CMake substep modules', folder)
    rows = []
    for phase, command in zip(('configure', 'build'), manifest['commands']):
        tick = time.perf_counter()
        log = folder / f'{phase}.log'
        with log.open('w') as stream:
            result = subprocess.run(command, stdout=stream, stderr=stream)
        rows.append(dict(phase=phase, seconds=time.perf_counter() - tick,
                         returncode=result.returncode, log=str(log)))
        (folder / 'completion.json').write_text(json.dumps(rows, indent=2))
        if result.returncode:
            raise RuntimeError(log.read_text()[-12000:])
    record_build('cmake-substep', manifest['identity'], manifest['commands'])
    return manifest


def main():
    import argparse
    from .compilation import compilation_scope
    parser = argparse.ArgumentParser(description='Offline CMake build for every LIBERO substep shape')
    parser.add_argument('--precision',choices=('fp64','fp32'),default=None)
    parser.add_argument('--build-root',type=Path,required=True)
    parser.add_argument('--mathdx-root',type=Path,required=True)
    parser.add_argument('--arch',type=int,default=80)
    parser.add_argument('--fast-compile',action='store_true')
    parser.add_argument('--profile-newton',action='store_true')
    parser.add_argument('--staged',action='store_true',help='Build the optional coarse FP64 decomposition')
    parser.add_argument('--ordered-reductions',action='store_true',
                        help='Build the ordered comparison products alongside the default products')
    args = parser.parse_args()
    scalar.select(args.precision)
    source = Path(__file__).resolve().parents[2]/'cuda/step/fused_substep.cu'
    manifest = prepare_all(source,args.build_root,args.mathdx_root,args.arch,
                           staged=args.staged,
                           fast_compile=args.fast_compile,profile_newton=args.profile_newton,
                           atomic_reductions=False if args.ordered_reductions else None)
    with compilation_scope(True):
        finish(manifest,allow_build=True)
    if not all(Path(path).is_file() for path in manifest['libraries']):
        raise RuntimeError('offline substep catalogue is incomplete')
    print(json.dumps(dict(folder=manifest['folder'],variants=len(manifest['libraries']),tasks=130)))


if __name__=='__main__':
    main()
