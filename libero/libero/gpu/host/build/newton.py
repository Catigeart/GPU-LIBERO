"""Offline Newton source/link preparation and load-only product selection.

The selection describes the in-memory parameter layout and launch schedule.
Only explicit offline preparation reads CUDA sources, SDK files or compiler
dependencies. Products use the existing native cache and its storage budget.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import uuid

from libero.gpu.host.paths import GPU_ROOT
from libero.gpu.host.build.compilation import compilation_allowed, require_compilation
from libero.gpu.host.build.product_cache import current_product_cache, product_cache_scope
from libero.gpu.host.runtime.core.product_paths import nvcc_path, record_build

LINESEARCH = '// @LIBERO_NEWTON_LINESEARCH@'
HEADERS = '// @LIBERO_NEWTON_HEADERS@'


def _query_source(config):
    """Query the loaded kernel and current device, without external tools."""
    return '''
extern "C" int newton_kernel_info(size_t* out, size_t count) {
    if (!out || count != 11) return (int)cudaErrorInvalidValue;
    cudaFuncAttributes attr{};
    cudaError_t err = cudaFuncGetAttributes(&attr, handwritten_newton);
    if (err != cudaSuccess) return (int)err;
    int device = 0, shared_limit = 0, thread_limit = 0;
    err = cudaGetDevice(&device);
    if (err != cudaSuccess) return (int)err;
    err = cudaDeviceGetAttribute(&shared_limit, cudaDevAttrMaxSharedMemoryPerBlock, device);
    if (err != cudaSuccess) return (int)err;
    err = cudaDeviceGetAttribute(&thread_limit, cudaDevAttrMaxThreadsPerBlock, device);
    if (err != cudaSuccess) return (int)err;
    out[0] = 1; out[1] = sizeof(Parameters);
    out[2] = BLOCK_THREADS; out[3] = DYNAMIC_SHARED; out[4] = COMPILED_ARCH;
    out[5] = attr.numRegs; out[6] = attr.localSizeBytes;
    out[7] = attr.sharedSizeBytes; out[8] = attr.maxThreadsPerBlock;
    out[9] = shared_limit; out[10] = thread_limit;
    return (int)cudaSuccess;
}
'''.replace('BLOCK_THREADS', str(config['block_threads'])).replace(
        'DYNAMIC_SHARED', str(config['shared_bytes'])).replace('COMPILED_ARCH', str(config['architecture']))


def _prepare_build(template, headers, config, build_root, mathdx_root):
    require_compilation('Newton source and link preparation', build_root)
    import libero.gpu.precision as scalar
    from .aot import collect_newton_dependency_inputs
    from .build_receipts import file_record
    if config['precision_bits'] != scalar.bits:
        raise ValueError('Newton precision differs from compiler configuration')
    build_root = Path(build_root) / 'offline-newton'
    build_root.mkdir(parents=True, exist_ok=True)
    use_dx, use_blas = config['use_dx'], config['use_blas']
    dependency = None
    if use_dx:
        from libero.gpu.host.runtime.core.product_paths import mathdx_root as locate_sdk
        dependency = Path(mathdx_root or os.environ.get('LIBERO_MATHDX_ROOT') or locate_sdk()).resolve()
        required = ['include/cusolverdx.hpp', 'lib/libcusolverdx.a', 'external/cutlass/include/cutlass/cutlass.h']
        if use_blas:
            required.append('include/cublasdx.hpp')
        for name in required:
            if not (dependency / name).is_file():
                raise FileNotFoundError(dependency / name)
    expanded, stage_sources = _project_header_source([GPU_ROOT / p for p in headers], GPU_ROOT)
    linesearch = GPU_ROOT / 'cuda/newton/linesearch_scalar.cuh'
    native = linesearch.read_text()
    # Match the former header instrumentation order, including the scalar search.
    source = template.replace(LINESEARCH, native).replace(HEADERS, expanded)
    if config['phase_profiling']:
        from libero.gpu.host.runtime.newton import device_phase_profile
        boundary = source.index('\nstruct Parameters {')
        source = device_phase_profile.instrument_headers(source[:boundary], use_dx=use_dx,
            use_blas=use_blas) + source[boundary:]
    source += _query_source(config)
    stage_sources[str(linesearch)] = hashlib.sha256(linesearch.read_bytes()).hexdigest()
    bridge_source = ''
    if use_dx:
        bridge_source = (f"#define NV {config['nv']}\n#define DX_SM {config['architecture']*10}\n"
            f"#define DX_THREADS {config['block_threads']}\n#define DX_MATRIX_BYTES {config['dx_bytes']}\n")
        bridge_files = ['cuda/newton/potrf_dx_bridge.cu']
        if use_blas:
            bridge_files += ['cuda/newton/hessian_blas_dx_bridge.cu', 'cuda/newton/direction_blas_dx_bridge.cu']
        if config['newton_islands'] == 'exact':
            bridge_files.append('cuda/newton/islands_dx_bridge.cu')
        for name in bridge_files:
            if name.endswith('hessian_blas_dx_bridge.cu'):
                bridge_source += f"\n#define DX_HESSIAN_K {config['hessian_k']}\n"
            path = GPU_ROOT / name
            raw = path.read_bytes()
            bridge_source += '\n' + raw.decode()
            stage_sources[str(path)] = hashlib.sha256(raw).hexdigest()
    source_path, bridge = build_root / 'newton.cu', build_root / 'bridge.cu'
    target = build_root / 'newton.so'
    source_path.write_text(source)
    compiler = nvcc_path()
    common = [compiler, *scalar.native_flags(), '-I'+str(GPU_ROOT/'cuda/common'),
        '-Xcompiler', '-fPIC', '-Xcompiler=-fno-gnu-unique', '-O3', '-std=c++17',
        '--expt-relaxed-constexpr', '-arch=sm_'+str(config['architecture'])]
    numeric = ['--fmad=true', '--prec-div=true', '--prec-sqrt=true', '--ftz=false']
    if use_dx:
        bridge.write_text(bridge_source)
        main_object, bridge_object = build_root/'main.o', build_root/'bridge.o'
        commands = [common + ['-dc', *numeric, str(source_path), '-o', str(main_object)],
            common + ['-dc', '-dlto', *numeric, '-I'+str(dependency/'include'),
                      '-I'+str(dependency/'external/cutlass/include'), str(bridge), '-o', str(bridge_object)],
            common + ['-shared', '-dlto', *numeric, '-Xlinker=--exclude-libs=libcusolverdx.a',
                      str(main_object), str(bridge_object), str(dependency/'lib/libcusolverdx.a'), '-o', str(target)]]
        inputs = collect_newton_dependency_inputs(commands, architecture=config['architecture'],
            mathdx_archive=dependency/'lib/libcusolverdx.a')
    else:
        from .build_receipts import compiler_dependencies, toolchain_record
        from .product_cache import compiler_environment_record
        commands = [common + ['-shared', *numeric, str(source_path), '-o', str(target)]]
        flags = common[1:] + numeric
        inputs = dict(units=[file_record(source_path)],
            dependencies=compiler_dependencies(compiler, flags, [source_path]),
            toolchain=toolchain_record(compiler, flags),
            environment=compiler_environment_record()['sha256'])
    # Generator files are offline dependencies too; runtime does not read them.
    inputs['generator'] = [file_record(Path(__file__)),
        file_record(GPU_ROOT/'host/runtime/newton/handwritten_cuda.py')]
    for path, expected in stage_sources.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Newton project source changed during offline preparation')
    inputs['project'] = [file_record(Path(p)) for p in stage_sources]
    metadata = dict(source=str(source_path), source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        source_bytes=len(source.encode()), command=commands[-1], compile_commands=commands,
        device_bridge=str(bridge) if use_dx else None,
        mathdx_root=str(dependency) if dependency is not None else None,
        mathdx_library_sha256=inputs['archive']['sha256'] if use_dx else None,
        stage_sources=stage_sources)
    return target, commands, inputs, metadata


def _validate_inputs(inputs):
    """Offline-only freshness; no runtime source or compiler authentication."""
    from .build_receipts import file_record
    from .product_cache import compiler_environment_record
    records = inputs['units'] + inputs['dependencies'] + inputs['generator'] + inputs['project']
    records += inputs['toolchain']['components']
    if 'archive' in inputs:
        records.append(inputs['archive'])
    for record in records:
        if file_record(Path(record['path'])) != record:
            raise RuntimeError('Newton build input changed: ' + record['path'])
    expected = inputs.get('environment', inputs['toolchain'].get('environment_sha256'))
    if expected != compiler_environment_record()['sha256']:
        raise RuntimeError('Newton compiler environment changed')


def obtain_product(*, template, headers, config, build_root, runtime_root, mathdx_root=None):
    # Only builders share writable source/object paths. Runtime files belong
    # to the environment and do not acquire a compiler-directory lock.
    from contextlib import nullcontext
    from libero.gpu.host.runtime.core.product_paths import build_unit_lock
    scope = build_unit_lock(build_root) if compilation_allowed() else nullcontext()
    cache_scope = (nullcontext() if current_product_cache() is not None else
                   product_cache_scope(Path(build_root)/'products'))
    with scope, cache_scope:
        return _obtain_product(template=template, headers=headers, config=config,
            build_root=build_root, runtime_root=runtime_root, mathdx_root=mathdx_root)


def _obtain_product(*, template, headers, config, build_root, runtime_root, mathdx_root=None):
    """Select by layout/schedule/options; prepare inputs only in offline mode."""
    from libero.gpu.host.runtime.newton.native_product import load_product
    headers = [Path(p).resolve().relative_to(GPU_ROOT).as_posix() for p in headers]
    selection = dict(schema='libero-newton-launch-v1', config=dict(config), headers=headers,
                     schedule=hashlib.sha256(template.encode()).hexdigest())
    cache = current_product_cache()
    offline = compilation_allowed()
    commands, inputs, metadata = [], None, None
    if offline:
        target, commands, inputs, metadata = _prepare_build(template, headers, config, build_root, mathdx_root)
    identity = dict(selection)
    if inputs is not None:
        identity['dependencies'] = inputs
    result = {}
    def validate():
        _validate_inputs(inputs)
    def build():
        require_compilation('Newton CUDA product', build_root)
        for index, command in enumerate(commands):
            output = subprocess.run(command, capture_output=True, text=True)
            with (target.parent/'compile.log').open('w' if index == 0 else 'a') as log:
                log.write(output.stdout + output.stderr)
            if output.returncode:
                raise RuntimeError('Newton CUDA compilation failed: ' + str(target.parent/'compile.log'))
        raw = target.read_bytes()
        description = dict(metadata, selection=selection, library_sha256=hashlib.sha256(raw).hexdigest())
        return {'newton.so': raw, 'product.json': json.dumps(description, sort_keys=True).encode()}
    def exports(artifacts):
        description = json.loads(artifacts['product.json'])
        if description['selection'] != selection:
            raise ValueError('Newton product launch selection differs')
        destination = Path(runtime_root)/('newton-'+uuid.uuid4().hex+'.so')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(artifacts['newton.so'])
        library, resources = load_product(destination, parameter_bytes=config['parameter_bytes'],
            block_threads=config['block_threads'], shared_bytes=config['shared_bytes'],
            architecture=config['architecture'], combined_entry=config['combined_entry'])
        result.update(library=library, resources=resources, metadata=description)
    symbols = ['launch_persistent', 'newton_kernel_info']
    if config['combined_entry']:
        symbols.append('launch_entry')
    _, receipt, status = cache.obtain_artifacts(identity, symbols,
        names=('newton.so', 'product.json'), build=build, validate=validate, exports=exports)
    executed = [] if status['hit'] else commands
    record_build('newton-cuda', status['key'], executed)
    result['metadata'] = dict(result['metadata'], sha256=status['key'],
        compile_commands_executed=executed, product_build=dict(cache=status))
    return result


def _project_header_source(paths, project_root):
    """Inline quoted project headers and return their complete source closure.

    A synthetic guard replaces each file's pragma once. Keeping the guard at
    every inclusion site preserves conditional inclusion: an inactive first
    include must not suppress a later active one. Angle-bracket SDK includes
    remain compiler inputs. Resolved quoted paths must stay inside the project.
    """
    project_root=Path(project_root).resolve()
    quoted=re.compile(r'^[ \t]*#[ \t]*include[ \t]*"([^"\r\n]+)"[^\r\n]*$',re.MULTILINE)
    once=re.compile(r'^[ \t]*#[ \t]*pragma[ \t]+once[ \t]*(?://[^\r\n]*)?\r?$',re.MULTILINE)
    contents={};active=[]

    def expand(path):
        path=Path(path).resolve()
        if not path.is_relative_to(project_root):
            raise ValueError(('quoted CUDA header outside project',str(path),str(project_root)))
        if path not in contents:contents[path]=path.read_bytes()
        text=contents[path].decode('utf-8')
        single=bool(once.search(text))
        if path in active:
            # A pragma-once parent has already defined its guard before this
            # recursive inclusion. An unguarded cycle is not a valid bundle.
            if single:return '// recursive pragma-once include: '+str(path.relative_to(project_root))+'\n'
            raise ValueError(('unguarded CUDA include cycle',tuple(map(str,active+[path]))))
        active.append(path)
        try:
            def include(match):
                child=path.parent/match.group(1)
                if match.group(1) == 'precision.cuh':
                    child=GPU_ROOT/'cuda/common/precision.cuh'
                return expand(child)
            body=quoted.sub(include,once.sub('',text))
        finally:
            active.pop()
        relative=path.relative_to(project_root).as_posix()
        if single:
            guard='LIBERO_INLINE_HEADER_'+hashlib.sha256(relative.encode()).hexdigest().upper()
            body=f'\n#ifndef {guard}\n#define {guard}\n// project source: {relative}\n'+body+f'\n#endif // {guard}\n'
        return body

    source='\n'.join(expand(path) for path in paths)
    hashes={str(path):hashlib.sha256(data).hexdigest() for path,data in contents.items()}
    return source,hashes
