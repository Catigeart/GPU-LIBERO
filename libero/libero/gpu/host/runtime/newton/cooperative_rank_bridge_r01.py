"""Isolated diagnostic CUDA rank helper; not a Newton runtime adapter.

Compilation/loading happen cold under the sole GPU runner. Cold bind retains
owners and native argument boxes; hot launch changes no pointers/owners.
"""
import libero.gpu.precision as scalar
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
from pathlib import Path
import ctypes as ct
import hashlib
import json
import re
import traceback
from types import SimpleNamespace
import warp as wp
from .s07_bridge import default_nvrtc, FIXED_OPTIONS, S07DeviceBridge

SOURCE_SHA='213ae9b7aeff85604b0afb20340667a006edd27af493cf6db7ab1657a333eccd'
SHIM=b'// standalone device-only compilation; NVRTC builtin CUDA intrinsics\n#pragma once\n'
OPTIONS=(b'--gpu-architecture=sm_120',*FIXED_OPTIONS)


def dimensions(nv,nworld):
    if type(nv) is not int or type(nworld) is not int or not 1<=nv<=57 or nworld<1:
        raise ValueError('rank diagnostic dimensions outside source prior')
    if nworld*nv*nv>=2**31:raise ValueError('rank diagnostic int32 offset overflow')
    return (nworld,1,1),(64,1,1)


class CooperativeRankModule:
    def __init__(self,build_root,device='cuda:0'):
        self.device=wp.get_device(device)
        if not self.device.is_cuda or self.device.arch!=120:raise ValueError('only actual sm120 CUDA scope')
        source=(GPU_ROOT / 'host/runtime/newton/cooperative_rank_update_r01.cu').read_bytes()
        if hashlib.sha256(source).hexdigest()!=SOURCE_SHA:raise ValueError('rank source pin drift')
        self.root=Path(build_root).resolve();self.root.mkdir(parents=True,exist_ok=False)
        (self.root/'source.cu').write_bytes(source)
        self.record=dict(scope='diagnostic helper only; independent GPU/compiled qualification pending',source_sha256=SOURCE_SHA,
            loader_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),options=[x.decode() for x in OPTIONS],
            shim_sha256=hashlib.sha256(SHIM).hexdigest(),nvrtc_path=default_nvrtc(),block=[64,1,1],accepted=False)
        self._cuda=None
        try:
            self.ptx=self._compile(source)
            self._load()
        except BaseException:
            self.record['failure']=traceback.format_exc();raise
        finally:
            (self.root/'record.json').write_text(json.dumps(self.record,indent=2)+'\n')

    def _compile(self,source):
        lib=ct.CDLL(self.record['nvrtc_path'])
        signatures={'nvrtcCreateProgram':[ct.POINTER(ct.c_void_p),ct.c_char_p,ct.c_char_p,ct.c_int,ct.POINTER(ct.c_char_p),ct.POINTER(ct.c_char_p)],'nvrtcCompileProgram':[ct.c_void_p,ct.c_int,ct.POINTER(ct.c_char_p)],'nvrtcGetProgramLogSize':[ct.c_void_p,ct.POINTER(ct.c_size_t)],'nvrtcGetProgramLog':[ct.c_void_p,ct.c_char_p],'nvrtcGetPTXSize':[ct.c_void_p,ct.POINTER(ct.c_size_t)],'nvrtcGetPTX':[ct.c_void_p,ct.c_char_p],'nvrtcDestroyProgram':[ct.POINTER(ct.c_void_p)]}
        for name,args in signatures.items():getattr(lib,name).argtypes=args;getattr(lib,name).restype=ct.c_int
        def checked(name,*args):
            rc=getattr(lib,name)(*args);self.record.setdefault('return_codes',[]).append([name,rc])
            if rc:raise RuntimeError((name,rc))
        prog=ct.c_void_p();headers=(ct.c_char_p*1)(SHIM);names=(ct.c_char_p*1)(b'cuda_runtime.h')
        checked('nvrtcCreateProgram',ct.byref(prog),source,b'rank.cu',1,headers,names)
        failed=False
        try:
            options=(ct.c_char_p*len(OPTIONS))(*OPTIONS)
            rc=lib.nvrtcCompileProgram(prog,len(OPTIONS),options);self.record['compile_return']=rc
            size=ct.c_size_t();checked('nvrtcGetProgramLogSize',prog,ct.byref(size))
            log=ct.create_string_buffer(max(size.value,1));checked('nvrtcGetProgramLog',prog,log)
            (self.root/'compiler_log.raw').write_bytes(log.raw[:size.value])
            if rc:raise RuntimeError(('NVRTC rank compile',rc,log.value.decode(errors='replace')))
            checked('nvrtcGetPTXSize',prog,ct.byref(size))
            if size.value<2:raise RuntimeError('empty rank PTX')
            out=ct.create_string_buffer(size.value);checked('nvrtcGetPTX',prog,out)
            if out.raw[-1:]!=b'\x00':raise RuntimeError('rank PTX NUL missing')
            ptx=out.raw[:-1];(self.root/'actual_product.ptx').write_bytes(ptx)
            counts={key:len(re.findall(pattern,ptx.decode())) for key,pattern in {'fma':r'\bfma(?:\.rn)?\.f64\b','div_approx':r'\bdiv\.approx\.f64\b','sqrt_approx':r'\bsqrt\.approx\.f64\b','div_rn':r'\bdiv\.rn\.f64\b','sqrt_rn':r'\bsqrt\.rn\.f64\b','barriers':r'\bbar\.sync\b'}.items()}
            self.record.update(ptx_sha256=hashlib.sha256(ptx).hexdigest(),instruction_counts=counts)
            if counts['fma'] or counts['div_approx'] or counts['sqrt_approx'] or not counts['div_rn'] or not counts['sqrt_rn'] or not counts['barriers']:
                raise ValueError(('rank compiled arithmetic/barrier scope rejected',counts))
            return ptx
        except BaseException:failed=True;raise
        finally:
            rc=lib.nvrtcDestroyProgram(ct.byref(prog));self.record['destroy_return']=rc
            if rc and not failed:raise RuntimeError(('NVRTC destroy',rc))

    def _load(self):
        self._cuda=S07DeviceBridge._driver(self)
        self.context=ct.c_void_p();rc=self._cuda.cuCtxGetCurrent(ct.byref(self.context))
        if rc or not self.context.value:raise RuntimeError(('rank context',rc))
        self._module=ct.c_void_p();self._ptx_buffer=ct.create_string_buffer(self.ptx)
        rc=self._cuda.cuModuleLoadData(ct.byref(self._module),ct.cast(self._ptx_buffer,ct.c_void_p))
        if rc:raise RuntimeError(('rank module',rc))
        self.function=ct.c_void_p();rc=self._cuda.cuModuleGetFunction(ct.byref(self.function),self._module,b'n_cooperative_rank_update_r01')
        if rc:raise RuntimeError(('rank function',rc))

    def bind(self,*,data,vec,plus,active,status,rank,nv,nworld,stream):
        grid,block=dimensions(nv,nworld)
        arrays=(data,vec,plus,active,status,rank)
        counts=(nworld*nv*nv,nworld*nv,nworld,nworld,nworld,nworld)
        for i,(array,count) in enumerate(zip(arrays,counts)):
            dtype=scalar.wp_real if i<2 else wp.int32
            if array.dtype!=dtype or array.device!=self.device or array.ndim!=1 or array.size!=count or not array.is_contiguous:
                raise ValueError('rank bound dtype/device/flat shape/contiguity')
        spans=[(int(x.ptr),int(x.ptr)+x.size*(8 if i<2 else 4)) for i,x in enumerate(arrays)]
        for i,(a,b) in enumerate(spans):
            if any(a<d and c<b for c,d in spans[i+1:]):raise ValueError('rank owners overlap')
        if stream.device!=self.device:raise ValueError('rank stream ownership')
        boxes=[ct.c_void_p(int(x.ptr)) for x in arrays]+[ct.c_int(nv),ct.c_int(nworld)]
        parameters=(ct.c_void_p*8)(*(ct.addressof(x) for x in boxes))
        return BoundRank(self,arrays,stream,grid,block,boxes,parameters)


class BoundRank:
    def __init__(self,module,arrays,stream,grid,block,boxes,parameters):
        self.module=module;self.arrays=arrays;self.stream=stream;self.grid=grid;self.block=block
        self.boxes=boxes;self.parameters=parameters
    def launch(self):
        # Caller holds immutable plus/active/status for this same-stream call;
        # no host value read, status steering, allocation, compilation or sync.
        rc=self.module._cuda.cuLaunchKernel(self.module.function,*self.grid,*self.block,0,
                    ct.c_void_p(self.stream.cuda_stream),self.parameters,None)
        if rc:raise RuntimeError(('rank launch',rc))
