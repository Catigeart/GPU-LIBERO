"""Load-only world-local FP32 advancement, sharing the qualified Euler ABI."""
import ctypes as ct
import hashlib
from pathlib import Path

from libero.gpu.host.runtime.dynamics.mjwarp_euler import EulerInputs


def source_id():
    from libero.gpu.host.paths import GPU_ROOT
    sources=('cuda/fp32/fused_tail.cu','cuda/dynamics/mjwarp_euler.cuh','cmake/FP32Fusion.cmake')
    return hashlib.sha256(':'.join(hashlib.sha256((GPU_ROOT/name).read_bytes()).hexdigest()
                                 for name in sources).encode()).hexdigest()


class FusedAdvance:
    def __init__(self,euler,library):
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

    def enqueue(self):
        error=self.launch(ct.byref(self.euler.arguments),self.stream.cuda_stream)
        if error:raise RuntimeError('FP32 fused advancement failed: CUDA error '+str(error))
