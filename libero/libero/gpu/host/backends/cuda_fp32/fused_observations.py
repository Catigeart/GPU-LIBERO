"""Load-only, world-local low-dimensional observation packing."""
import ctypes as ct
import hashlib
from pathlib import Path
from .bridges import Observe

FIELDS=('qpos','qvel','body_xpos','body_xquat','site_xpos')


class RawPackInputs(ct.Structure):
    _fields_=[('worlds',ct.c_int),('stride',ct.c_int),('width',ct.c_int*5),
              ('offset',ct.c_int*5),('source',ct.c_void_p*5),('target',ct.c_void_p)]


def source_id():
    from libero.gpu.host.paths import GPU_ROOT
    names=('cuda/fp32/fused_observations.cu','cuda/fp32/fused_observations.cuh','cmake/fp32/FP32Observations.cmake')
    return hashlib.sha256(':'.join(hashlib.sha256((GPU_ROOT/name).read_bytes()).hexdigest()
                                 for name in names).encode()).hexdigest()


class FusedRawPack:
    def __init__(self,packer,library):
        from libero.gpu.host.build.compilation import MissingCompiledProduct
        self.output,self.stream=packer.buffer,packer.buffer.stream
        self.path=Path(library)
        if not self.path.is_file():raise MissingCompiledProduct('Build flash_fp32_fused_observations offline')
        self.library=lib=ct.CDLL(str(self.path));lib.flash_fp32_observations_source_id.restype=ct.c_char_p
        lib.flash_fp32_observations_pack_bytes.restype=ct.c_size_t
        if lib.flash_fp32_observations_source_id().decode()!=source_id():
            raise MissingCompiledProduct('Stale fused observations; rebuild offline')
        if lib.flash_fp32_observations_pack_bytes()!=ct.sizeof(RawPackInputs):raise ValueError('Fused packing ABI differs')
        if lib.flash_fp32_observations_arch()!=self.stream.device.arch:raise ValueError('Fused packing architecture differs')
        if set(packer.layout)!=set(FIELDS):raise ValueError('Fused raw packing requires the five default fields')
        if packer.nworld*packer.stride>2**31-256:raise ValueError('Fused packing dimensions exceed indexing')
        self.arguments=RawPackInputs(worlds=packer.nworld,stride=packer.stride,target=packer.buffer.ptr,
            width=(ct.c_int*5)(*(packer.layout[name][1] for name in FIELDS)),
            offset=(ct.c_int*5)(*(packer.layout[name][0] for name in FIELDS)))
        self.launch=lib.flash_fp32_observations_pack
        self.launch.argtypes=[ct.POINTER(RawPackInputs),ct.c_void_p];self.launch.restype=ct.c_int
        size=lib.flash_fp32_observations_sample_bytes;size.restype=ct.c_size_t
        if size()!=ct.sizeof(Observe):raise ValueError('Fused sampling ABI differs')
        self.sample_launch=lib.flash_fp32_observations_sample
        self.sample_launch.argtypes=[ct.POINTER(RawPackInputs),ct.POINTER(Observe),ct.c_float,ct.c_int,ct.c_void_p]
        self.sample_launch.restype=ct.c_int

    def enqueue(self,fields):
        self.sources=tuple(fields[name] for name in FIELDS)
        self.arguments.source=(ct.c_void_p*5)(*(source.ptr for source in self.sources))
        error=self.launch(ct.byref(self.arguments),self.stream.cuda_stream)
        if error:raise RuntimeError('Fused raw packing failed: CUDA error '+str(error))

    def sample(self,lowdim,fields,timestep,force=False,world_mask=None,stream=None):
        self.sources=tuple(fields[name] for name in FIELDS)
        self.arguments.source=(ct.c_void_p*5)(*(source.ptr for source in self.sources))
        args=lowdim.sample_arguments(self.output,timestep,world_mask,stream)
        error=self.sample_launch(ct.byref(self.arguments),ct.byref(args),float(timestep),int(force),self.stream.cuda_stream)
        if error:raise RuntimeError('Fused observation sampling failed: CUDA error '+str(error))
