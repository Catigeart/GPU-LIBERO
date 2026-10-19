"""Compile handwritten device functions and compose only their ABI/call order.

NativePlan(name, sources, stages, nworld, build_root, block_size=64,
           defines=None, mathdx=False, argument_specs=None)
Sources are .cuh paths or inline handwritten CUDA strings. Stages are
(qualified_function_name, argument_values, 'lane0'|'block'[, C_type_specs]).
Every device function receives world index first. Opt-in tile stages receive
(world, blockIdx.x, gridDim.x, args...) and use work_tiles=N. Multiple tiles
require exclusively tile stages: legacy world stages are never duplicated.
Cross-tile prepare/worker/commit phases are separate same-stream launches. Arrays are contiguous Warp
buffers passed as raw typed pointers; Python float/int default to double/int32.
Explicit specs, e.g. 'const double*', 'int64_t', override those defaults. Global
argument_specs is a sequence of per-stage spec lists. Binding ownership, shape,
dtype, device and addresses are validated. launch(stream=...) allocates nothing,
uploads no parameter buffer, and is graph-capture compatible. Params are passed
by value, freezing scalars/pointers in each queued kernel or graph node.
No arithmetic translation takes place.

compile_library accepts handwritten CUDA and optional handwritten bridge units.
All compilation uses FMA, precise division/sqrt, and no fast-math. MathDx links
the cuSOLVERDx LTO library; cuBLASDx is header-only.
"""
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
import ctypes as ct
from functools import lru_cache
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from contextvars import ContextVar
from contextlib import contextmanager
from pathlib import Path
from libero.gpu.host.runtime.core.product_paths import selected_mathdx_root, nvcc_path
from .build_receipts import (compiler_dependencies,toolchain_record,file_record,
    build_identity,product_lock,publish_product,materialize_private)

from .product_cache import (product_cache_scope, current_product_authority,
    compiler_environment_record)

import numpy as np
import warp as wp

MATHDX_ROOT = '/srv/libero-split-receive-20261004-r01/dependencies/mathdx-cuda12/nvidia-mathdx-25.12.1-cuda12/nvidia/mathdx/25.12'
_strict_receipts_scope=ContextVar('strict_build_receipts',default=None)
_world_guard_scope=ContextVar('native_plan_world_guard',default=None)


@contextmanager
def world_guard_scope(guard, prefixes):
    """Cold-only guard for explicitly named kernel families, including tiles.

    The generated kernel returns before scratch or public outputs
    are touched. This is stronger than passing an inactive arithmetic mask.
    """
    token=_world_guard_scope.set((guard,tuple(prefixes)))
    try:yield
    finally:_world_guard_scope.reset(token)

@contextmanager
def strict_build_receipts_scope(enabled=True):
    """Explicit prototype cold constructors, without global environment edits."""
    token=_strict_receipts_scope.set(bool(enabled))
    try:yield
    finally:_strict_receipts_scope.reset(token)

_CTYPES = {'double': ct.c_double, 'float': ct.c_float, 'int': ct.c_int32,
           'int32_t': ct.c_int32, 'uint32_t': ct.c_uint32, 'int64_t': ct.c_int64,
           'uint64_t': ct.c_uint64, 'bool': ct.c_bool, 'size_t': ct.c_size_t}


@lru_cache(maxsize=8)
def _large_parameter_support(architecture, nvcc):
    """Cold feature gate for CUDA 12.1's extended kernel argument ABI."""
    result=subprocess.run([nvcc,'--version'],capture_output=True,text=True)
    version=re.search(r'release\s+(\d+)\.(\d+)',result.stdout)
    compiler=tuple(map(int,version.groups())) if result.returncode==0 and version else (0,0)
    driver=ct.c_int(0)
    try:
        lib=ct.CDLL('libcuda.so.1')
        lib.cuDriverGetVersion.argtypes=[ct.POINTER(ct.c_int)]
        lib.cuDriverGetVersion.restype=ct.c_int
        if lib.cuDriverGetVersion(ct.byref(driver))!=0: driver.value=0
    except OSError:
        driver.value=0
    supported=architecture>=70 and compiler>=(12,1) and driver.value>=12010
    return {'supported':supported,'architecture':architecture,
            'compiler_version':list(compiler),'driver_version':driver.value,
            'parameter_limit_bytes':32764 if supported else 4096,
            'documentation':'https://developer.nvidia.com/blog/?p=66058'}


def _compile_library_receipted(name, source, build_root, *, device='cuda:0', mathdx=False,
                    defines=None, bridges=(), include_dirs=(), extra_flags=(), required_symbols=()):
    """Return (ctypes library, manifest); retain source and compiler diagnostics."""
    root = Path(build_root).resolve(); root.mkdir(parents=True, exist_ok=True)
    dev = wp.get_device(device)
    if not dev.is_cuda:
        raise ValueError('native CUDA requires a CUDA device')
    flags = ['-Xcompiler', '-fPIC', '-O3', '-std=c++17', '--fmad=true',
             '--prec-div=true', '--prec-sqrt=true', '--ftz=false', '--expt-relaxed-constexpr',
             '-arch=sm_%d' % dev.arch]
    if any(x.split('=',1)[0] in ('--use_fast_math','-use_fast_math','--fmad','-fmad',
        '--prec-div','-prec-div','--prec-sqrt','-prec-sqrt','--ftz','-ftz') for x in extra_flags):
        raise ValueError('fast-math and disabled FMA are outside the fusion contract')
    flags += list(extra_flags)
    for key, value in sorted((defines or {}).items()):
        if not re.fullmatch(r'[A-Za-z_]\w*', key):
            raise ValueError('invalid CUDA define')
        flags.append('-D%s=%s' % (key, value))
    roots = [str(Path(p).resolve()) for p in include_dirs]
    library = None
    if mathdx:
        dx = selected_mathdx_root(MATHDX_ROOT)
        roots += [str(dx/'include'), str(dx/'external/cutlass/include')]
        library = dx/'lib/libcusolverdx.a'
        if not library.is_file() or not (dx/'include/cublasdx.hpp').is_file():
            raise FileNotFoundError('MathDx installation is incomplete: '+str(dx))
    for inc in roots:
        flags += ['-I', inc]
    bridge_text = [Path(p).read_text() if isinstance(p, Path) or '\n' not in str(p)
                   else str(p) for p in bridges]
    nvcc = nvcc_path()
    authority=current_product_authority()
    def current_toolchain():
        result=toolchain_record(nvcc,flags)
        if authority is not None:
            result=dict(result)
            result['environment_sha256']=compiler_environment_record()['sha256']
            result['environment_schema']='compiler-input-environment-v1'
        return result
    toolchain=current_toolchain()
    with tempfile.TemporaryDirectory(prefix='deps_',dir=root) as scratch:
        units=[]
        for index,text in enumerate([source]+bridge_text):
            unit=Path(scratch)/('unit%d.cu'%index);unit.write_text(text);units.append(unit)
        dependencies=compiler_dependencies(nvcc,flags,units)
    key,identity=build_identity(source=source,bridges=bridge_text,flags=flags,
        toolchain=toolchain,dependencies=dependencies,archive=file_record(library) if library else None,
        architecture=dev.arch)
    # Each plan retains a distinct physical product. Neither repeated plan name
    # nor cache lookup can replace a path that an earlier CDLL still owns.
    stem = re.sub(r'[^A-Za-z0-9_]', '_', name)+'_'+key+'_'+uuid.uuid4().hex
    src = root/(stem+'.cu'); target = root/(stem+'.so')
    src.write_text(source)
    commands = []
    if mathdx or bridge_text:
        objects = []
        for index, text in enumerate([source]+bridge_text):
            unit = src if index == 0 else root/(stem+'_bridge%d.cu' % index)
            unit.write_text(text); obj = unit.with_suffix('.o'); objects.append(str(obj))
            commands.append([nvcc]+flags+['-dc', '-dlto', str(unit), '-o', str(obj)])
        commands.append([nvcc]+flags+['-shared', '-dlto']+objects+
                        ([str(library)] if library else [])+['-o', str(target)])
    else:
        commands.append([nvcc]+flags+['-shared', str(src), '-o', str(target)])
    log = root/(stem+'.log')
    cache_source = None
    cache_root=Path(os.environ.get('LIBERO_CUDA_FUSION_CACHE',str(root/'products'))).resolve()
    cache_folder=cache_root/key
    product_status=None
    if authority is not None:
        # The signed authority is available only through an explicit product
        # scope. Legacy diagnostic strict receipts still always compile below.
        units=[src]
        for index,text in enumerate(bridge_text,1):
            unit=root/(stem+'_bridge%d.cu'%index)
            unit.write_text(text); units.append(unit)
        def validate_current_inputs():
            if any(unit.read_text()!=text for unit,text in zip(units,[source]+bridge_text)):
                raise RuntimeError('CUDA generated source changed during admission')
            if compiler_dependencies(nvcc,flags,units)!=dependencies:
                raise RuntimeError('CUDA dependency selection/content changed during admission')
            if (current_toolchain()!=toolchain or
                    (library and file_record(library)!=identity['archive'])):
                raise RuntimeError('CUDA toolchain/archive changed during admission')
        def build_completed_bytes():
            from libero.gpu.host.build.compilation import require_compilation
            require_compilation('receipted native CUDA library', target)
            output=[]
            for command in commands:
                result=subprocess.run(command,capture_output=True,text=True)
                output.append(result.stdout+result.stderr);log.write_text('\n'.join(output))
                if result.returncode: raise RuntimeError('CUDA compilation failed: '+str(log))
            return target.read_bytes()
        def verify_private_exports(raw):
            # Always copy bytes to this invocation's unique physical path.
            # Cache blobs are never loaded and no CDLL/owner is shared.
            with target.open('wb') as stream: stream.write(raw)
            completed=ct.CDLL(str(target))
            if not all(hasattr(completed,symbol) for symbol in required_symbols):
                raise RuntimeError('native compiled launcher symbol ABI missing')
        raw,receipt,product_status=authority.obtain(identity,required_symbols,
            build=build_completed_bytes,validate=validate_current_inputs,exports=verify_private_exports)
        if product_status['hit']:
            cache_source=str(authority.root/product_status['key']/'product.so')
            log.write_text('Authorized completed product reused; no compiler build commands executed.\n')
    else:
        with product_lock(cache_folder):
            from libero.gpu.host.build.compilation import require_compilation
            require_compilation('receipted native CUDA library', target)
            # This entry issues only a new successful cold compile product.
            # Previously published self-hashed receipts grant no reuse authority.
            output=[]
            for command in commands:
                result=subprocess.run(command,capture_output=True,text=True)
                output.append(result.stdout+result.stderr);log.write_text('\n'.join(output))
                if result.returncode: raise RuntimeError('CUDA compilation failed: '+str(log))
            raw=target.read_bytes()
            # Reject changed dependencies/toolchain during compilation. A new
            # invocation obtains a fresh key; half products never publish.
            if any(file_record(row['path'])!=row for row in dependencies):
                raise RuntimeError('CUDA dependency changed during compilation')
            if toolchain_record(nvcc,flags)!=toolchain or (library and file_record(library)!=identity['archive']):
                raise RuntimeError('CUDA toolchain/archive changed during compilation')
            completed=ct.CDLL(str(target))
            if not all(hasattr(completed,symbol) for symbol in required_symbols):
                raise RuntimeError('native compiled launcher symbol ABI missing')
            receipt=publish_product(cache_folder,key,identity,raw)
    loaded=ct.CDLL(str(target))
    if not all(hasattr(loaded,symbol) for symbol in required_symbols):
        raise RuntimeError('native compiled launcher symbol ABI missing')
    manifest = {'name': name, 'sha256': key, 'source': str(src),
                'library': str(target), 'compile_commands': commands,
                'fma': True, 'fast_math': False, 'architecture': dev.arch,
                'cache_source':cache_source,'build_receipt':receipt,
                'private_product':True,'symbol_abi':'generated source declarations; NativePlan sets exact ctypes signature'}
    if product_status is not None:
        manifest['product_cache']=product_status
        manifest['compile_commands_executed']=[] if product_status['hit'] else commands
    (root/(stem+'.json')).write_text(json.dumps(manifest, indent=2))
    return loaded, manifest


def _compile_library_legacy(name, source, build_root, *, device='cuda:0', mathdx=False,
                    defines=None, bridges=(), include_dirs=(), extra_flags=()):
    """Return (ctypes library, manifest); retain source and compiler diagnostics."""
    root = Path(build_root).resolve(); root.mkdir(parents=True, exist_ok=True)
    dev = wp.get_device(device)
    if not dev.is_cuda:
        raise ValueError('native CUDA requires a CUDA device')
    flags = ['-Xcompiler', '-fPIC', '-O3', '-std=c++17', '--fmad=true',
             '--prec-div=true', '--prec-sqrt=true', '--expt-relaxed-constexpr',
             '-arch=sm_%d' % dev.arch]
    if any(x in ('--use_fast_math', '--fmad=false') for x in extra_flags):
        raise ValueError('fast-math and disabled FMA are outside the fusion contract')
    flags += list(extra_flags)
    for key, value in sorted((defines or {}).items()):
        if not re.fullmatch(r'[A-Za-z_]\w*', key):
            raise ValueError('invalid CUDA define')
        flags.append('-D%s=%s' % (key, value))
    roots = [str(Path(p).resolve()) for p in include_dirs]
    library = None
    if mathdx:
        dx = selected_mathdx_root(MATHDX_ROOT)
        roots += [str(dx/'include'), str(dx/'external/cutlass/include')]
        library = dx/'lib/libcusolverdx.a'
        if not library.is_file() or not (dx/'include/cublasdx.hpp').is_file():
            raise FileNotFoundError('MathDx installation is incomplete: '+str(dx))
    for inc in roots:
        flags += ['-I', inc]
    bridge_text = [Path(p).read_text() if isinstance(p, Path) or '\n' not in str(p)
                   else str(p) for p in bridges]
    digest = hashlib.sha256((source+'\n'+'\n'.join(bridge_text)+repr(flags)).encode())
    # User include dependencies participate in the cache; vendor archive is
    # hashed separately. Never hash the entire installed CUDA header tree.
    # Track actual transitive quoted includes, not every unrelated module.
    dependency_roots = [Path(p).resolve() for p in include_dirs]
    pending = [(source, root)] + [(text, root) for text in bridge_text]
    dependencies = set()
    while pending:
        unit_text, parent = pending.pop()
        for delimiter, spelling in re.findall(r'^\s*#\s*include\s*(["<])([^">\n]+)[">]', unit_text, re.MULTILINE):
            inc = Path(spelling)
            choices = [inc] if inc.is_absolute() else ([parent/inc] if delimiter == '"' else []) + [folder/inc for folder in dependency_roots]
            header = next((p.resolve() for p in choices if p.is_file()), None)
            if header is None or header in dependencies:
                continue  # nvcc diagnoses unresolved includes.
            dependencies.add(header)
            pending.append((header.read_text(), header.parent))
    for header in sorted(dependencies):
        digest.update(str(header).encode());digest.update(header.read_bytes())
    if library:
        digest.update(library.read_bytes())
    key = digest.hexdigest()[:20]
    stem = re.sub(r'[^A-Za-z0-9_]', '_', name)+'_'+key
    src = root/(stem+'.cu'); target = root/(stem+'.so')
    src.write_text(source)
    nvcc = nvcc_path()
    commands = []
    if mathdx or bridge_text:
        objects = []
        for index, text in enumerate([source]+bridge_text):
            unit = src if index == 0 else root/(stem+'_bridge%d.cu' % index)
            unit.write_text(text); obj = unit.with_suffix('.o'); objects.append(str(obj))
            commands.append([nvcc]+flags+['-dc', '-dlto', str(unit), '-o', str(obj)])
        commands.append([nvcc]+flags+['-shared', '-dlto']+objects+
                        ([str(library)] if library else [])+['-o', str(target)])
    else:
        commands.append([nvcc]+flags+['-shared', str(src), '-o', str(target)])
    log = root/(stem+'.log')
    cache_source = None
    reused = target.exists()
    if reused:
        previous = json.loads(target.with_suffix('.json').read_text())
        if previous.get('sha256') != digest.hexdigest():
            raise ValueError('cached native source identity differs')
        binary_sha = previous.get('binary_sha256')
        if binary_sha is not None and hashlib.sha256(target.read_bytes()).hexdigest() != binary_sha:
            raise ValueError('cached native binary differs')
        cache_source = str(target)
    if not target.exists():
        # Names distinguish plans and reports; the generic launch ABI depends
        # only on the complete source/dependency/policy digest. Reuse an
        # already completed identical build under another plan name.
        cache_roots=[root]
        explicit_cache=os.environ.get('LIBERO_CUDA_FUSION_CACHE')
        if explicit_cache: cache_roots.append(Path(explicit_cache).resolve())
        from libero.gpu.host.runtime.core.product_paths import current_build_project, cache_root
        project = current_build_project()
        if project is not None:
            cache_roots.extend((project['root'] / 'cuda', cache_root() / 'builds'))
        candidates=sorted({path for folder in cache_roots for path in folder.rglob('*_'+key+'.so')})
        for candidate in candidates:
            metadata = candidate.with_suffix('.json')
            try:
                identical = json.loads(metadata.read_text()).get('sha256') == digest.hexdigest()
            except (OSError, ValueError):
                continue
            if identical:
                recorded = json.loads(metadata.read_text()).get('binary_sha256')
                if recorded is not None and hashlib.sha256(candidate.read_bytes()).hexdigest() != recorded:
                    continue
                shutil.copyfile(candidate,target)
                cache_source = str(candidate)
                reused = True
                break
    if not target.exists():
        from libero.gpu.host.build.compilation import require_compilation
        require_compilation('native CUDA library', target)
        output = []
        for command in commands:
            result = subprocess.run(command, capture_output=True, text=True)
            output.append(result.stdout+result.stderr)
            log.write_text('\n'.join(output))
            if result.returncode:
                raise RuntimeError('CUDA compilation failed: '+str(log))
    manifest = {'name': name, 'sha256': digest.hexdigest(), 'source': str(src),
                'library': str(target), 'compile_commands': commands,
                'fma': True, 'fast_math': False, 'architecture': dev.arch,
                'cache_source':cache_source,
                'binary_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                'compile_commands_executed':[] if reused else commands}
    (root/(stem+'.json')).write_text(json.dumps(manifest, indent=2))
    return ct.CDLL(str(target)), manifest


def compile_library(name,source,build_root,*,device='cuda:0',mathdx=False,
                    defines=None,bridges=(),include_dirs=(),extra_flags=(),required_symbols=(),
                    strict_build_receipts=None):
    """Receipted build is explicit until actual cold ABBA clears its overhead."""
    scoped=_strict_receipts_scope.get()
    enabled=((scoped if scoped is not None else os.environ.get('LIBERO_BUILD_RECEIPTS')=='1')
             if strict_build_receipts is None else bool(strict_build_receipts))
    function=_compile_library_receipted if enabled else _compile_library_legacy
    options=dict(device=device,mathdx=mathdx,defines=defines,bridges=bridges,
                 include_dirs=include_dirs,extra_flags=extra_flags)
    if enabled:options['required_symbols']=required_symbols
    from libero.gpu.host.runtime.core.product_paths import build_unit_lock
    with build_unit_lock(build_root):
        loaded,manifest=function(name,source,build_root,**options)
    if enabled and not all(hasattr(loaded,symbol) for symbol in required_symbols):
        raise RuntimeError('native compiled launcher symbol ABI missing')
    from libero.gpu.host.runtime.core.product_paths import record_build
    record_build('native-cuda', manifest.get('sha256', name),
                 manifest.get('compile_commands_executed', manifest['compile_commands']))
    return loaded,manifest


class NativePlan:
    def __init__(self, name, sources, stages, nworld, build_root, *, block_size=64,
                 defines=None, mathdx=False, argument_specs=None, device=None,
                 bridges=(), extra_flags=(), track_execution=True, world_guard=None,
                 dynamic_shared_bytes=0, stage_barriers=None, work_tiles=1):
        scoped_guard=_world_guard_scope.get()
        if world_guard is None and scoped_guard is not None and name.startswith(scoped_guard[1]):
            world_guard=scoped_guard[0]
        if not isinstance(nworld, int) or nworld < 1 or block_size not in (32,64,128,256):
            raise ValueError('positive world count and block size 32/64/128/256 required')
        from libero.gpu.host.runtime.core.tiled_execution import validate_tile_stages
        validate_tile_stages(work_tiles, stages)
        self.work_tiles = work_tiles
        self.nworld = nworld; self.block_size = block_size
        if isinstance(dynamic_shared_bytes, bool) or not isinstance(dynamic_shared_bytes, int) or dynamic_shared_bytes < 0:
            raise ValueError('dynamic shared bytes must be a nonnegative integer')
        self.dynamic_shared_bytes = dynamic_shared_bytes
        self.world_guard = world_guard
        if world_guard is not None and (not isinstance(world_guard, wp.array) or world_guard.dtype != wp.int32 or world_guard.shape != (nworld,) or not world_guard.is_contiguous):
            raise ValueError('world_guard must be contiguous resident int32[worlds]')
        if stage_barriers is None:
            barriers = [not (i+1 < len(stages) and stages[i][2] == stages[i+1][2] == 'lane0') for i in range(len(stages))]
        elif isinstance(stage_barriers, bool):
            barriers = [stage_barriers] * len(stages)
        else:
            barriers = list(stage_barriers)
            if len(barriers) != len(stages) or any(not isinstance(b, bool) for b in barriers):
                raise ValueError('stage_barriers must match stages')
        self.stage_barriers = tuple(barriers)
        fields=[]; values=[]; declarations=[]; calls=[]; seen={}; self.owners=[]
        self.scalar_slots={}; self.pointer_slots={}
        self._retired_owners=[]; self._captured_owners=[]
        self._stage_args=[]
        dtype_names = {wp.float64:'double', wp.float32:'float', wp.int32:'int32_t',
                       wp.int64:'int64_t', wp.uint32:'uint32_t', wp.uint64:'uint64_t',
                       wp.uint8:'uint8_t', wp.int8:'int8_t', wp.int16:'int16_t',
                       wp.uint16:'uint16_t', wp.bool:'bool'}
        for si, stage in enumerate(stages):
            fn, args, mode = stage[:3]
            self._stage_args.append(list(args))
            if not re.fullmatch(r'[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*', fn):
                raise ValueError('device stage must be a qualified function name')
            if mode not in ('lane0', 'block', 'tile'):
                raise ValueError('unknown stage execution mode')
            specs = stage[3] if len(stage)>3 else (argument_specs[si] if argument_specs else None)
            if specs is not None and len(specs)!=len(args):
                raise ValueError('stage type specs must match argument count')
            names=[]
            for ai, value in enumerate(args):
                spec = specs[ai] if specs else None
                if isinstance(value, wp.array):
                    if not value.is_contiguous or not value.device.is_cuda:
                        raise ValueError('contiguous CUDA arrays required')
                    ctype = spec or dtype_names.get(value.dtype, '')+'*'
                    if ctype == '*' or '*' not in ctype:
                        raise TypeError('unsupported array type; supply a C pointer spec')
                    # Keep an independent slot per argument: equal cold pointers
                    # may diverge when a public caller supplies dynamic buffers.
                    token = ('ptr', si, ai, ctype)
                    ct_type = ct.c_void_p; raw = value.ptr
                    owner_index=len(self.owners);self.owners.append(value)
                else:
                    ctype = spec or ('double' if isinstance(value, (float,np.floating)) else 'int32_t')
                    if ctype not in _CTYPES:
                        raise TypeError('unsupported scalar C type: '+ctype)
                    ct_type = _CTYPES[ctype]; raw = value
                    token = ('scalar', si, ai)
                if token not in seen:
                    slot = 'a%d' % len(fields); seen[token] = slot
                    fields.append((slot, ct_type)); values.append(raw)
                    declarations.append('    %s %s;' % (ctype, slot))
                if token[0]=='scalar': self.scalar_slots[(si,ai)]=(seen[token],ct_type)
                else: self.pointer_slots[(si,ai)]=(seen[token],owner_index)
                names.append('p->'+seen[token])
            call = '%s(%s%s);' % (fn, 'w, int(blockIdx.x), int(gridDim.x)' if mode == 'tile' else 'w', ', '+', '.join(names) if names else '')
            calls.append(('    if(threadIdx.x==0) { '+call+' }' if mode=='lane0' else '    '+call) + ('\n    __syncthreads();' if barriers[si] else ''))
        if not fields: fields=[('_unused',ct.c_int32)]; values=[0]; declarations=['    int32_t _unused;']
        self.device = wp.get_device(device or (self.owners[0].device if self.owners else 'cuda:0'))
        if world_guard is not None:
            fields.append(('world_guard', ct.c_void_p)); values.append(world_guard.ptr)
            declarations.append('    const int32_t* world_guard;')
            self.owners.append(world_guard)
        if any(a.device != self.device for a in self.owners):
            raise ValueError('all native plan buffers must belong to the same device')
        self.execution_count=wp.zeros(nworld,dtype=wp.int64,device=self.device) if track_execution else None
        if track_execution:
            fields.append(('execution_calls',ct.c_void_p));values.append(self.execution_count.ptr)
            declarations.append('    unsigned long long* execution_calls;')
        self.tokens = tuple((a.ptr, a.shape, a.dtype, a.device) for a in self.owners)
        class Params(ct.Structure): _fields_ = fields
        self.params = Params(*values)
        self._param_type=Params
        parameter_bytes=ct.sizeof(self.params)
        parameter_abi={'parameter_bytes':parameter_bytes,'parameter_limit_bytes':4096,
                       'grid_constant':False,'large_parameter_support':None}
        if parameter_bytes>4096:
            support=_large_parameter_support(self.device.arch,nvcc_path())
            parameter_abi.update(parameter_limit_bytes=support['parameter_limit_bytes'],
                                 grid_constant=support['supported'],large_parameter_support=support)
            if parameter_bytes>support['parameter_limit_bytes']:
                raise ValueError(('native plan kernel argument capacity exceeded',parameter_bytes,support))
        util = (GPU_ROOT / 'cuda/common/math.cuh')
        # Header contents are embedded so dependency changes invalidate the cache.
        warp_util = GPU_ROOT / 'cuda/common/warp.cuh'
        vector_util = GPU_ROOT / 'cuda/common/vector.cuh'
        text = [util.read_text(), warp_util.read_text(), vector_util.read_text()]
        include_dirs=[]
        for item in sources:
            if isinstance(item,Path) or '\n' not in str(item):
                path=Path(item)
                if path.resolve() not in (util.resolve(), warp_util.resolve(), vector_util.resolve()):
                    text.append(path.read_text())
                include_dirs.append(path.parent)
            else: text.append(str(item))
        source = '\n'.join(text)+'\nstruct Params {\n'+'\n'.join(declarations)+'\n};\n'
        kernel_parameter='__grid_constant__ const Params params' if parameter_abi['grid_constant'] else 'Params params'
        if parameter_abi['grid_constant']:
            source+='static_assert(sizeof(Params)==%d,"parameter ABI layout");\n'%parameter_bytes
        source += '__global__ void native_kernel('+kernel_parameter+') {\n    const Params* p=&params;\n    const int w=blockIdx.y;\n'
        if track_execution: source += '    if(threadIdx.x==0 && blockIdx.x==0) atomicAdd(p->execution_calls+w,1ULL);\n'
        if world_guard is not None: source += '    if(p->world_guard[w]==0) return;\n'
        source += '\n'.join(calls)
        source+='\n}\n'
        source += 'extern "C" int launch(const void* p,int worlds,void* stream) {\n'
        source += '    native_kernel<<<dim3(%d,worlds,1),%d,%d,(cudaStream_t)stream>>>(*static_cast<const Params*>(p));\n' % (work_tiles, block_size, dynamic_shared_bytes)
        source += '    return (int)cudaGetLastError();\n}\n'
        source += '''
extern "C" int configure_shared(int bytes) {
    if (bytes <= 49152) return 0;
    int ordinal=0, available=0; cudaFuncAttributes attributes;
    cudaError_t status=cudaGetDevice(&ordinal); if(status!=cudaSuccess) return (int)status;
    status=cudaDeviceGetAttribute(&available,cudaDevAttrMaxSharedMemoryPerBlockOptin,ordinal); if(status!=cudaSuccess) return (int)status;
    status=cudaFuncGetAttributes(&attributes,native_kernel); if(status!=cudaSuccess) return (int)status;
    if ((size_t)bytes+attributes.sharedSizeBytes>(size_t)available) return (int)cudaErrorInvalidValue;
    return (int)cudaFuncSetAttribute(native_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
}
'''
        receipt_scope=_strict_receipts_scope.get()
        receipt_enabled=receipt_scope if receipt_scope is not None else os.environ.get('LIBERO_BUILD_RECEIPTS')=='1'
        if receipt_enabled:
            self.library, self.manifest = compile_library(name,source,build_root,device=self.device,
                    mathdx=mathdx,defines=defines,bridges=bridges,extra_flags=extra_flags,
                    include_dirs=include_dirs,required_symbols=('launch','configure_shared'))
        else:
            self.library, self.manifest = compile_library(name,source,build_root,device=self.device,
                    mathdx=mathdx,defines=defines,bridges=bridges,extra_flags=extra_flags,
                    include_dirs=include_dirs)
        self.library.configure_shared.argtypes=[ct.c_int]
        self.library.configure_shared.restype=ct.c_int
        with wp.ScopedDevice(self.device):
            shared_status=self.library.configure_shared(dynamic_shared_bytes)
        if shared_status:
            raise ValueError(('native plan shared memory not admitted',dynamic_shared_bytes,shared_status))
        self.manifest.update(grid_layout='work_tile_x_world_y', dynamic_shared_bytes=dynamic_shared_bytes, stage_barriers=barriers, world_guard=world_guard is not None)
        self.manifest['work_tiles'] = work_tiles
        self.manifest['stage_execution_modes'] = [stage[2] for stage in stages]
        self.manifest['execution_count_scope'] = 'one_per_world_launch'
        self.manifest['tile_abi'] = 'world,tile,tile_count,args' if any(stage[2]=='tile' for stage in stages) else None
        self.manifest['cross_tile_sync'] = 'separate_same_stream_kernel_boundaries' if work_tiles > 1 else None
        self.manifest['parameter_abi']=parameter_abi
        Path(self.manifest['library']).with_suffix('.json').write_text(json.dumps(self.manifest,indent=2))
        self.library.launch.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p]
        self.library.launch.restype=ct.c_int

    def validate_bindings(self):
        if tuple((a.ptr,a.shape,a.dtype,a.device) for a in self.owners)!=self.tokens:
            raise ValueError('native plan array bindings changed; rebuild the plan')

    def set_scalar(self, stage, index, value, stream=None):
        """Change the host ABI; the next launch copies it into kernel arguments."""
        slot,ctype=self.scalar_slots[(stage,index)]
        self._stage_args[stage][index]=value
        self._set_slot(slot,ctype(value),stream)

    def _set_slot(self, slot, scalar, stream=None):
        stream=stream or wp.get_stream(self.device)
        if stream.device != self.device: raise ValueError('scalar update stream device mismatch')
        setattr(self.params,slot,scalar.value)

    def set_pointer(self, stage, index, array, stream=None):
        """Rebind one argument to an equal-layout array on the same device.

        Old buffers remain owned until queued work completes; graph-captured
        bindings remain owned with the plan. Pointer slots are never coalesced.
        """
        slot,owner_index=self.pointer_slots[(stage,index)]
        old=self.owners[owner_index]
        if not isinstance(array,wp.array) or not array.is_contiguous or \
                array.device!=self.device or array.dtype!=old.dtype or array.shape!=old.shape:
            raise ValueError('pointer replacement must preserve shape/dtype/device/contiguity')
        stream=stream or wp.get_stream(self.device)
        if stream.device!=self.device: raise ValueError('pointer update stream device mismatch')
        if array.ptr==old.ptr: return
        self._set_slot(slot,ct.c_void_p(array.ptr),stream)
        (self._captured_owners if stream.is_capturing else self._retired_owners).extend([old,array])
        self.owners[owner_index]=array; self._stage_args[stage][index]=array
        self.tokens=tuple((a.ptr,a.shape,a.dtype,a.device) for a in self.owners)

    def set_arguments(self, stages, stream=None):
        """Update lists of stage arguments (or full stage tuples) in stream order."""
        if len(stages)!=len(self._stage_args): raise ValueError('stage count changed')
        for si,item in enumerate(stages):
            args=item[1] if len(item)>=3 and isinstance(item[0],str) else item
            if len(args)!=len(self._stage_args[si]): raise ValueError('argument count changed')
            for ai,value in enumerate(args):
                if (si,ai) in self.pointer_slots: self.set_pointer(si,ai,value,stream)
                elif value!=self._stage_args[si][ai]: self.set_scalar(si,ai,value,stream)

    def release_scalar_updates(self):
        """Compatibility name: release retired pointers at a synchronization boundary."""
        wp.synchronize_device(self.device)
        self._retired_owners.clear()

    def reserve_scalar_updates(self, count):
        """Compatibility no-op: by-value kernel arguments require no staging pool."""

    def launch(self, stream=None):
        self.validate_bindings()
        stream = stream or wp.get_stream(self.device)
        if hasattr(stream,'device') and stream.device != self.device:
            raise ValueError('launch stream belongs to another device')
        if hasattr(stream,'is_capturing') and stream.is_capturing:
            self._captured_owners.extend(self.owners)
        raw_stream = stream.cuda_stream if hasattr(stream,'cuda_stream') else int(stream)
        with wp.ScopedDevice(self.device):
            status=self.library.launch(ct.addressof(self.params),self.nworld,raw_stream)
        if status:
            raise RuntimeError('native CUDA launch failed: '+str(status))
