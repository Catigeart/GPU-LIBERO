"""Private native CUDA detector launcher; no host detector/contact evaluation.

The library is prepared before capture. launch() uses device pointers and the
caller's current Warp CUDA stream; device counts decide which slots enter.
"""
from __future__ import annotations
import libero.gpu.precision as scalar
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
from libero.gpu.host.runtime.core.product_paths import nvcc_path
import ctypes as ct
import hashlib,json,os,re,subprocess,time
from pathlib import Path
import warp as wp
from libero.gpu.host.build.product_cache import current_product_authority
from libero.gpu.host.build.build_artifacts import receipted_artifact_build
import uuid

MODULE=GPU_ROOT / 'cuda/collision/native'
_CACHE={}

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

def _host_function_symbol(image,address_offset):
    import struct
    data=Path(image).read_bytes()
    if data[:7]!=b'\x7fELF\x02\x01\x01' or struct.unpack_from('<H',data,18)[0]!=62:
        raise ValueError('actual host image ELF ABI')
    shoff=struct.unpack_from('<Q',data,40)[0];size,count,index=struct.unpack_from('<HHH',data,58)
    if size!=64 or shoff+size*count>len(data):raise ValueError('actual host image section bounds')
    sections=[struct.unpack_from('<IIQQQQIIQQ',data,shoff+i*size) for i in range(count)]
    found=[]
    for q in sections:
        if q[1]!=2:continue
        strings=sections[q[6]];names=data[strings[4]:strings[4]+strings[5]]
        if q[9]!=24:raise ValueError('actual host symbol ABI')
        for off in range(q[4],q[4]+q[5],q[9]):
            name,info,other,section,value,nbytes=struct.unpack_from('<IBBHQQ',data,off)
            if info&15==2 and value==address_offset:
                found.append({'name':names[name:].split(b'\0',1)[0].decode(),
                              'info':info,'other':other,'section_index':section,
                              'value':value,'bytes':nbytes})
    if len(found)!=1 or not found[0]['name'].startswith('_Z13native_kernel'):
        raise ValueError('exact hidden production host stub symbol/offset')
    return found[0]

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
    from libero.gpu.host.runtime.core.product_paths import collision_build_root
    root = collision_build_root(root, MODULE, GPU_ROOT)
    vector_header=(GPU_ROOT / 'cuda/common/vector.cuh').resolve()
    sources=_native_source_bindings(vector_header)
    key=hashlib.sha256(json.dumps({'sources':sources,'fma':True,'real_bits':scalar.bits,'forced_include':str(vector_header)},sort_keys=True).encode()).hexdigest()
    cache_key=(str(Path(root).resolve()),key,dev.arch,str(dev))
    authority=current_product_authority()
    from libero.gpu.host.runtime.core.product_paths import record_build
    if authority is None and cache_key in _CACHE:
        record_build('collision-native', key)
        return _CACHE[cache_key]
    target=Path(root)/('native-'+key[:24]+'-sm'+str(dev.arch))
    if authority is not None:target=target.with_name(target.name+'_'+uuid.uuid4().hex)
    nvcc=Path(os.environ.get('P06C_NVCC') or nvcc_path()).resolve()
    gcc=Path(os.environ.get('P06C_CCBIN','/usr/bin/g++')).resolve()
    flags=[*scalar.native_flags(),'-I'+str(GPU_ROOT/'cuda/common'),'-std=c++17','-O2','-arch=sm_'+str(dev.arch),'--fmad=true',
           '--ftz=false','--prec-div=true','--prec-sqrt=true','-rdc=true',
           '-include',str(vector_header),
           '-ccbin',str(gcc)]
    argv=[str(nvcc),*flags,'-shared','-Xcompiler','-fPIC',
          '-Xlinker','-rpath','-Xlinker',str(nvcc.parent.parent/'lib64'),
          str((MODULE / 'native_kernel.cu')),'-o',str(target/'libcollision.so')]
    ptx_argv=[str(nvcc),*flags,'--ptx',str((MODULE / 'native_kernel.cu')),
              '-o',str(target/'collision.ptx')]
    toolchain={str(p):{'sha256':_sha(p),'bytes':p.stat().st_size} for p in [nvcc,gcc]}
    if authority is not None:
        target.mkdir(parents=True,exist_ok=False)
        source_paths={str(p):(p.relative_to(GPU_ROOT).as_posix() if p.is_relative_to(GPU_ROOT.resolve()) else 'generated_native/'+p.relative_to(MODULE.resolve()).as_posix())
            for p in [GPU_ROOT/key if not key.startswith('generated_native/') else MODULE/key.removeprefix('generated_native/') for key in sources]}
        closure={spelling:sources[logical]['sha256'] for spelling,logical in source_paths.items()}
        record={'schema':'p06c-authorized-native-build-v1','sources':sources,
            'toolchain_files':toolchain,'argv':[argv,ptx_argv],'cwd':str(target),'exit_code':0,
            'scope':'private raw contact adapter layout only; actual signed product authority'}
        loaded={}
        def log_compile(index,result):
            (target/f'compile_{index}.stdout.txt').write_text(result.stdout)
            (target/f'compile_{index}.stderr.txt').write_text(result.stderr)
        def verify_private_product():
            record['artifacts']={p.name:{'sha256':_sha(p),'bytes':p.stat().st_size}
                for p in target.iterdir() if p.is_file()}
            (target/'BUILD.json').write_text(json.dumps(record,indent=2))
            # Complete original ABI, CUDA function/host stub, dladdr, map/image
            # and SDK checks happen before the authority can sign completion.
            loaded['object']=_bind_native_product(target,record,nvcc,dev,sources)
        product_result=receipted_artifact_build(commands=[argv,ptx_argv],
            units=[MODULE/'native_kernel.cu'],outputs={'libcollision.so':target/'libcollision.so','collision.ptx':target/'collision.ptx'},
            architecture=dev.arch,source_closure=closure,source_aliases=source_paths,
            verify=verify_private_product,cwd=target,logs=log_compile,
            required_symbols=('native_private_abi','launch_native','native_kernel_identity'))
        record['product_build']=product_result
        (target/'BUILD.json').write_text(json.dumps(record,indent=2))
        obj=loaded['object'];obj['build_sha256']=_sha(target/'BUILD.json');obj['product_build']=product_result
        _CACHE[cache_key]=obj
        record_build('collision-native', key, product_result['compile_commands_executed'])
        return obj
    reused = target.exists()
    if target.exists():
        # A build from another process may only be reused after all source,
        # compiler identity, flags and output bindings are reverified.
        record=json.loads((target/'BUILD.json').read_text())
        if (record['sources']!=sources or record['exit_code']!=0 or
            record['argv'] != [argv,ptx_argv] or record['cwd'] != str(target) or
            record['toolchain_files'] != toolchain):
            raise ValueError('native build source/status mismatch')
        expected_files={'PRE_BUILD.json','compile_0.stdout.txt','compile_0.stderr.txt',
                        'compile_1.stdout.txt','compile_1.stderr.txt',
                        'libcollision.so','collision.ptx'}
        if set(record['artifacts']) != expected_files:
            raise ValueError('native build incomplete artifact set')
        for name,binding in record['artifacts'].items():
            p=target/name
            if _sha(p)!=binding['sha256'] or p.stat().st_size!=binding['bytes']:
                raise ValueError('native build artifact mismatch: '+name)
        for path,binding in record['toolchain_files'].items():
            if _sha(path)!=binding['sha256']:
                raise ValueError('native toolchain identity mismatch')
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
    obj=_bind_native_product(target,record,nvcc,dev,sources)
    record_build('collision-native', key, () if reused else (argv, ptx_argv))
    _CACHE[cache_key]=obj
    return obj

def _bind_native_product(target,record,nvcc,dev,sources):
    lib=ct.CDLL(str(target/'libcollision.so'))
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
    class DlInfo(ct.Structure):
        _fields_=[('name',ct.c_char_p),('base',ct.c_void_p),
                  ('symbol',ct.c_char_p),('address',ct.c_void_p)]
    dl=ct.CDLL(None);dl.dladdr.argtypes=[ct.c_void_p,ct.POINTER(DlInfo)]
    dl.dladdr.restype=ci
    addresses={'kernel_stub':int(values[0]),'launch_export':int(ct.cast(lib.launch_native,ptr).value),
               'identity_export':int(ct.cast(lib.native_kernel_identity,ptr).value)}
    resolved={}
    for name,address in addresses.items():
        info=DlInfo()
        if dl.dladdr(ptr(address),ct.byref(info))!=1 or not info.name:
            raise ValueError('actual native dladdr failed '+name)
        image=Path(info.name.decode()).resolve()
        if image!=(target/'libcollision.so').resolve():
            raise ValueError('actual native function belongs to another image '+name)
        resolved[name]={'file':str(image),'base':int(info.base),
                        'symbol':info.symbol.decode() if info.symbol else None,
                        'symbol_address':int(info.address or 0),'address':address}
    host_stub=_host_function_symbol(target/'libcollision.so',resolved['kernel_stub']['address']-resolved['kernel_stub']['base'])
    image=target/'libcollision.so'
    mapped=[]
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields=line.split()
        if len(fields)>5 and fields[-1]==str(image):
            begin,end=[int(v,16) for v in fields[0].split('-')]
            mapped.append({'start':begin,'end':end,'permissions':fields[1],
                           'file_offset':int(fields[2],16),'file':fields[-1]})
    if not mapped or _sha(image)!=record['artifacts']['libcollision.so']['sha256']:
        raise ValueError('actual native image mapping/hash drift')
    execution={'schema':'p06c-actual-native-module-binding-v1','identity_status':identity_status,
               'library_file':str(image),'library_sha256':_sha(image),
               'library_handle':int(lib._handle),'cuda_function':int(values[1]),
               'device':str(dev),'dladdr':resolved,'maps':mapped,'host_stub_symbol':host_stub,
               'kernel_attributes':dict(zip(
                   ['binary_version','ptx_version','num_regs','max_threads_per_block','cache_mode_ca',
                    'runtime_version','driver_version','attribute_struct_bytes'],map(int,attrs))),
               'local_bytes':int(values[2]),'shared_bytes':int(values[3]),'constant_bytes':int(values[4]),
               'sdk_headers':{str(p):{'sha256':_sha(p),'bytes':p.stat().st_size} for p in
                   [nvcc.parent.parent/'include'/n for n in ('cuda_runtime.h','cuda_runtime_api.h','driver_types.h')]},
               'api_source_sha256':sources[_source_binding_key(MODULE/'native_kernel.cu')]['sha256'],
               'binding_program_sha256':_sha(__file__),
               'scope':'actual cudaGetFuncBySymbol/native stub/image preparation; no CUDA API argument values captured by Nsight'}
    obj={'library':lib,'build_path':str(target),'build_sha256':_sha(target/'BUILD.json'),
         'adapter_abi':[scalar.itemsize,4,4,13*scalar.itemsize,0,scalar.itemsize,4*scalar.itemsize],'device':str(dev),'execution_binding':execution}
    from libero.gpu.host.runtime.core.product_paths import record_module_binding
    obj['module_identity_file'] = record_module_binding(target, execution)
    return obj

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
