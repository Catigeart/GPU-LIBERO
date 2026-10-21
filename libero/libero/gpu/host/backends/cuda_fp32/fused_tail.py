"""Load-only world-local FP32 advancement, sharing the qualified Euler ABI."""
import ctypes as ct
import hashlib
from pathlib import Path

from libero.gpu.host.runtime.dynamics.mjwarp_euler import EulerInputs
from libero.gpu.host.runtime.dynamics.mjwarp_publication import PublicationInputs


class TailInputs(ct.Structure):
    _fields_=[('integration',EulerInputs),('publication',PublicationInputs),
              ('solved_acceleration',ct.c_void_p),('warmstart',ct.c_void_p)]


def source_id():
    from libero.gpu.host.paths import GPU_ROOT
    sources=('cuda/fp32/fused_tail.cu','cuda/fp32/fused_tail.cuh','cuda/dynamics/mjwarp_euler.cuh',
             'cuda/dynamics/mjwarp_publication.cuh','cmake/FP32Tail.cmake')
    return hashlib.sha256(':'.join(hashlib.sha256((GPU_ROOT/name).read_bytes()).hexdigest()
                                 for name in sources).encode()).hexdigest()


class FusedAdvance:
    def __init__(self,euler,library,*,publication=None,qacc=None,warmstart=None):
        from libero.gpu.host.build.compilation import MissingCompiledProduct
        self.euler,self.stream=euler,euler.stream
        self.path=Path(library)
        if not self.path.is_file():
            raise MissingCompiledProduct('Build flash_fp32_fused_tail offline')
        self.library=lib=ct.CDLL(str(self.path))
        lib.flash_fp32_tail_source_id.restype=ct.c_char_p
        lib.flash_fp32_tail_arguments_bytes.restype=ct.c_size_t
        if lib.flash_fp32_tail_source_id().decode()!=source_id():
            raise MissingCompiledProduct('Stale FP32 fused tail; rebuild flash_fp32_fused_tail offline')
        if lib.flash_fp32_tail_arguments_bytes()!=ct.sizeof(EulerInputs):
            raise ValueError('FP32 fused tail ABI differs')
        if lib.flash_fp32_tail_arch()!=self.stream.device.arch:
            raise ValueError('FP32 fused tail architecture differs')
        self.launch=lib.flash_fp32_tail_advance
        self.launch.argtypes=[ct.POINTER(EulerInputs),ct.c_void_p]
        self.launch.restype=ct.c_int
        self.arguments=euler.arguments
        if publication is not None:
            from . import device as cuda
            from libero.gpu.host.runtime.dynamics.mjwarp_motion import bind_array,reject_output_overlap
            n,nv=euler.data.nworld,euler.model.nv
            self.owners=(euler,publication,qacc,warmstart)
            arrays=dict(euler.arrays,**publication.arrays)
            bind_array(arrays,'solved_acceleration',qacc,cuda.float32,(n,nv),self.stream)
            bind_array(arrays,'warmstart',warmstart,cuda.float32,(n,nv),self.stream)
            reject_output_overlap(arrays,('warmstart',))
            size=lib.flash_fp32_tail_finish_bytes;size.restype=ct.c_size_t
            if size()!=ct.sizeof(TailInputs):raise ValueError('FP32 fused publication ABI differs')
            self.arguments=TailInputs(euler.arguments,publication.arguments,qacc.ptr,warmstart.ptr)
            self.launch=lib.flash_fp32_tail_finish
            self.launch.argtypes=[ct.POINTER(TailInputs),ct.c_void_p];self.launch.restype=ct.c_int

    def enqueue(self):
        error=self.launch(ct.byref(self.arguments),self.stream.cuda_stream)
        if error:raise RuntimeError('FP32 fused advancement failed: CUDA error '+str(error))
