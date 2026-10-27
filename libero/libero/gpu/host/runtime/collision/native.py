"""Private native CUDA detector launcher; no host detector/contact evaluation.

The library is prepared before capture. launch() uses device pointers and the
caller's current Warp CUDA stream; device counts decide which slots enter.
"""
from __future__ import annotations
import libero.gpu.precision as scalar
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
from libero.gpu.host.runtime.core.product_paths import nvcc_path
import ctypes as ct
from libero.gpu.host.build.library_lifetime import load_cdll
import hashlib,json,os,re,subprocess,time
from pathlib import Path
import warp as wp
from libero.gpu.host.build.product_cache import current_product_cache
from libero.gpu.host.build.build_artifacts import receipted_artifact_build
import uuid

MODULE=GPU_ROOT / 'cuda/collision/native'
_CACHE={}


def _store_native_cache(key, obj, target):
    _CACHE[key] = obj
    from libero.gpu.host.runtime.core.product_paths import current_build_lifetime
    owner = current_build_lifetime()
    if owner is not None and Path(target).resolve().is_relative_to(owner.path):
        def retire():
            if _CACHE.get(key) is obj:
                _CACHE.pop(key)
        owner.retire_callbacks.append(retire)

def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_binding_key(path):
    path=Path(path).resolve()
    gpu=GPU_ROOT.resolve();module=MODULE.resolve()
    if path.is_relative_to(gpu):return path.relative_to(gpu).as_posix()
    if path.is_relative_to(module):return 'generated_native/'+path.relative_to(module).as_posix()
    raise ValueError('native source outside current GPU/module source roots: '+str(path))

def _native_source_bindings(vector_header):
    """Include maintained quoted headers in the native detector cache key.

    Only project headers are recorded. System CUDA headers are covered by the
    separately recorded toolchain, and are never recursively enumerated here.
    """
    pending=[MODULE/'native_kernel.cu',MODULE/'native_source.cuh',vector_header]
    seen=set()
    while pending:
        source=Path(pending.pop()).resolve()
        if source in seen:continue
        seen.add(source)
        for spelling in re.findall(r'^\s*#\s*include\s*"([^"\n]+)"',source.read_text(),re.M):
            header=(source.parent/spelling).resolve()
            if header.is_file() and (header.is_relative_to(GPU_ROOT.resolve()) or header.is_relative_to(MODULE.resolve())):
                pending.append(header)
    return {_source_binding_key(p):
            {'sha256':_sha(p),'bytes':p.stat().st_size} for p in sorted(seen)}

def prepare_native(device):
    from libero.gpu.host.runtime.core.product_paths import build_unit_lock, collision_build_root
    root = os.environ.get('P06C_NATIVE_BUILD_ROOT')
    if not root:raise ValueError('P06C_NATIVE_BUILD_ROOT must name private build directory')
    with build_unit_lock(collision_build_root(root, MODULE, GPU_ROOT)):
        return _prepare_native(device)


def _prepare_native(device):
    dev=wp.get_device(device)
    if not dev.is_cuda:raise ValueError('native collision requires CUDA device')
    root=os.environ.get('P06C_NATIVE_BUILD_ROOT')
    if not root:raise ValueError('P06C_NATIVE_BUILD_ROOT must name private build directory')
    from libero.gpu.host.runtime.core.product_paths import collision_build_root, private_product_root
    root = private_product_root(collision_build_root(root, MODULE, GPU_ROOT))
    vector_header=(GPU_ROOT / 'cuda/common/vector.cuh').resolve()
    sources=_native_source_bindings(vector_header)
    key=hashlib.sha256(json.dumps({'sources':sources,'fma':True,'real_bits':scalar.bits,'forced_include':str(vector_header)},sort_keys=True).encode()).hexdigest()
    cache_key=(str(Path(root).resolve()),key,dev.arch,str(dev))
    cache=current_product_cache()
    from libero.gpu.host.runtime.core.product_paths import record_build
    if cache is None and cache_key in _CACHE:
        record_build('collision-native', key)
        return _CACHE[cache_key]
    target=Path(root)/('native-'+key[:24]+'-sm'+str(dev.arch))
    if cache is not None:target=target.with_name(target.name+'_'+uuid.uuid4().hex)
    from libero.gpu.host.build.compilation import compilation_allowed
    offline = compilation_allowed()
    nvcc=Path(os.environ.get('P06C_NVCC') or nvcc_path()).resolve() if offline else None
    gcc=Path(os.environ.get('P06C_CCBIN','/usr/bin/g++')).resolve()
    flags=[*scalar.native_flags(),'-I'+str(GPU_ROOT/'cuda/common'),'-std=c++17','-O2','-arch=sm_'+str(dev.arch),'--fmad=true',
           '--ftz=false','--prec-div=true','--prec-sqrt=true','-rdc=true',
           '-include',str(vector_header),
           '-ccbin',str(gcc)]
    argv=[str(nvcc) if nvcc else 'nvcc',*flags,'-shared','-Xcompiler','-fPIC',
          '-Xlinker','-rpath','-Xlinker',str(nvcc.parent.parent/'lib64') if nvcc else '',
          str((MODULE / 'native_kernel.cu')),'-o',str(target/'libcollision.so')]
    ptx_argv=[str(nvcc) if nvcc else 'nvcc',*flags,'--ptx',str((MODULE / 'native_kernel.cu')),
              '-o',str(target/'collision.ptx')]
    from libero.gpu.host.build.compilation import compilation_allowed
    offline = compilation_allowed()
    toolchain = ({str(p):{'sha256':_sha(p),'bytes':p.stat().st_size} for p in [nvcc,gcc]}
                 if offline else {})
    if cache is not None:
        target.mkdir(parents=True,exist_ok=False)
        source_paths={str(p):(p.relative_to(GPU_ROOT).as_posix() if p.is_relative_to(GPU_ROOT.resolve()) else 'generated_native/'+p.relative_to(MODULE.resolve()).as_posix())
            for p in [GPU_ROOT/key if not key.startswith('generated_native/') else MODULE/key.removeprefix('generated_native/') for key in sources]}
        closure={spelling:sources[logical]['sha256'] for spelling,logical in source_paths.items()}
        record={'schema':'p06c-authorized-native-build-v1','sources':sources,
            'toolchain_files':toolchain,'argv':[argv,ptx_argv],'cwd':str(target),'exit_code':0,
            'scope':'private raw contact adapter layout; ordinary cached compiler product'}
        loaded={}
        def log_compile(index,result):
            (target/f'compile_{index}.stdout.txt').write_text(result.stdout)
            (target/f'compile_{index}.stderr.txt').write_text(result.stderr)
        def verify_private_product():
            # Runtime needs the loaded ABI and CUDA function, not compiler,
            # ELF symbol-table, process-map or SDK source authentication.
            loaded['object']=_bind_native_product(target,dev)
            if offline:
                record['artifacts']={p.name:{'sha256':_sha(p),'bytes':p.stat().st_size}
                    for p in target.iterdir() if p.is_file()}
        product_result=receipted_artifact_build(commands=[argv,ptx_argv],
            units=[MODULE/'native_kernel.cu'],outputs={'libcollision.so':target/'libcollision.so','collision.ptx':target/'collision.ptx'},
            architecture=dev.arch,source_closure=closure,source_aliases=source_paths,
            verify=verify_private_product,cwd=target,logs=log_compile,
            required_symbols=('native_private_abi','launch_native','native_kernel_identity'))
        record['product_build']=product_result
        if offline:(target/'BUILD.json').write_text(json.dumps(record,indent=2))
        obj=loaded['object'];obj['product_build']=product_result
        _store_native_cache(cache_key, obj, target)
        record_build('collision-native', key, product_result['compile_commands_executed'])
        return obj
    reused = target.exists()
    if target.exists():
        # Select the matching build recipe. Full compiler/artifact provenance
        # is checked only during offline preparation.
        record=json.loads((target/'BUILD.json').read_text())
        if (record['sources']!=sources or record['exit_code']!=0 or
            (offline and (record['argv'] != [argv,ptx_argv] or record['cwd'] != str(target) or
                          record['toolchain_files'] != toolchain))):
            raise ValueError('native build source/status mismatch')
        for name in ('libcollision.so','collision.ptx'):
            if not (target/name).is_file():
                raise FileNotFoundError('native build missing product: '+name)
        if offline:
            expected_files={'PRE_BUILD.json','compile_0.stdout.txt','compile_0.stderr.txt',
                            'compile_1.stdout.txt','compile_1.stderr.txt',
                            'libcollision.so','collision.ptx'}
            if set(record['artifacts']) != expected_files:
                raise ValueError('native build incomplete artifact set')
            for name,binding in record['artifacts'].items():
                p=target/name
                if _sha(p)!=binding['sha256'] or p.stat().st_size!=binding['bytes']:
                    raise ValueError('native build artifact mismatch: '+name)
    else:
        from libero.gpu.host.build.compilation import require_compilation
        require_compilation('collision CUDA library', target)
        target.mkdir(parents=True,exist_ok=False)
        record={'schema':'p06c-native-build-v1','sources':sources,
                'toolchain_files':toolchain,
                'versions':{},'argv':[argv,ptx_argv],'cwd':str(target),
                'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                'arithmetic':'binary64, source associations, fmad=true, ftz=false',
                'actual_native_mjContact_ABI':False,
                'scope':'private raw contact adapter layout only'}
        for name,tool in [('nvcc',nvcc),('g++',gcc)]:
            from libero.gpu.host.build.child_environment import child_env
            child=child_env(target);child['LD_LIBRARY_PATH']='/usr/lib/wsl/lib'
            result=subprocess.run([str(tool),'--version'],cwd=target,env=child,capture_output=True,text=True)
            record['versions'][name]={'exit_code':result.returncode,
                                     'stdout':result.stdout,'stderr':result.stderr}
        record['child_env']={k:v for k,v in child.items() if k.startswith('XDG_') or k in (
            'PWD','TMPDIR','TMP','TEMP','CUDA_CACHE_PATH','WARP_CACHE_PATH','WP_CACHE_DIR',
            'LIBERO_CONFIG_PATH','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS','LD_LIBRARY_PATH')}
        (target/'PRE_BUILD.json').write_text(json.dumps(record,indent=2))
        for idx,args in enumerate([argv,ptx_argv]):
            result=subprocess.run(args,cwd=target,env=child,capture_output=True,text=True)
            (target/f'compile_{idx}.stdout.txt').write_text(result.stdout)
            (target/f'compile_{idx}.stderr.txt').write_text(result.stderr)
            if result.returncode:
                record['exit_code']=result.returncode
                (target/'BUILD.json').write_text(json.dumps(record,indent=2))
                raise RuntimeError(f'native compile failed: {target}: {result.stderr}')
        record['exit_code']=0
        record['artifacts']={p.name:{'sha256':_sha(p),'bytes':p.stat().st_size}
                             for p in sorted(target.iterdir()) if p.is_file()}
        (target/'BUILD.json').write_text(json.dumps(record,indent=2))
    obj=_bind_native_product(target,dev)
    record_build('collision-native', key, () if reused else (argv, ptx_argv))
    _store_native_cache(cache_key, obj, target)
    return obj

def _bind_native_product(target,dev):
    lib=load_cdll(target/'libcollision.so')
    lib.native_private_abi.argtypes=[ct.POINTER(ct.c_int)]
    lib.native_private_abi.restype=ct.c_int
    sizes=(ct.c_int*7)()
    if lib.native_private_abi(sizes)!=7 or list(sizes)!=[scalar.itemsize,4,4,13*scalar.itemsize,0,scalar.itemsize,4*scalar.itemsize]:
        raise ValueError('private native contact adapter ABI mismatch')
    ptr=ct.c_void_p;ci=ct.c_int;cd=scalar.ct_real
    lib.launch_native.argtypes=[ptr]*17+[ci]*7+[cd,ci]+[ptr]*7
    lib.launch_native.restype=ci
    # Resolve the exact registered CUDA kernel while this image and Warp context
    # are current. This module preparation occurs before any case workspace.
    lib.native_kernel_identity.argtypes=[ct.POINTER(ct.c_ulonglong),ct.POINTER(ct.c_int)]
    lib.native_kernel_identity.restype=ci
    values=(ct.c_ulonglong*5)();attrs=(ct.c_int*8)()
    with wp.ScopedDevice(dev):
        identity_status=lib.native_kernel_identity(values,attrs)
    if identity_status or not values[0] or not values[1]:
        raise RuntimeError('actual registered CUDA kernel identity failed '+str(identity_status))
    execution={'schema':'native-cuda-resources-v1',
               'library_file':str(target/'libcollision.so'),
               'library_handle':int(lib._handle),'cuda_function':int(values[1]),
               'device':str(dev),
               'kernel_attributes':dict(zip(
                   ['binary_version','ptx_version','num_regs','max_threads_per_block','cache_mode_ca',
                    'runtime_version','driver_version','attribute_struct_bytes'],map(int,attrs))),
               'local_bytes':int(values[2]),'shared_bytes':int(values[3]),'constant_bytes':int(values[4])}
    return {'library':lib,'build_path':str(target),
            'adapter_abi':list(sizes),'device':str(dev),'execution_binding':execution}

def launch_native(ws,stream=None):
    stream=stream or wp.get_stream(ws.device)
    if wp.get_device(stream.device)!=wp.get_device(ws.device):
        raise ValueError('native detector stream device mismatch')
    def ptr(x):return ct.c_void_p(int(x.ptr or 0))
    inputs=[ws.call_g1,ws.call_g2,ws.call_type1,ws.call_type2,ws.call_margin,
            ws.call_count,ws.active,ws.xpos,ws.xmat,ws.geom_size,ws.geom_rbound,
            ws.geom_dataid,ws.mesh_vert,ws.mesh_vertadr,ws.mesh_vertnum,
            ws.mesh_graphadr,ws.mesh_graph]
    outputs=[ws.native_raw,ws.native_count,ws.native_status,ws.native_diag,
             ws.native_frame0,ws.native_framewritten]
    # ScopedDevice makes the Warp context current for CUDA Runtime interop.
    with wp.ScopedDevice(ws.device):
        status=ws.native['library'].launch_native(
            *[ptr(x) for x in inputs],ws.nworld,ws.pairs,ws.ngeom,ws.nmesh,
            ws.nvert,ws.ngraph,ws.mpr_iterations,ws.mpr_tolerance,ws.multiccd,
            *[ptr(x) for x in outputs],ct.c_void_p(int(stream.cuda_stream)))
    if status:raise RuntimeError('native CUDA launch error '+str(status))
