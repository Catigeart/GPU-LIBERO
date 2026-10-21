"""Admit one current-input world velocity/bias region, with component fallback."""
import ctypes as ct
import hashlib
from pathlib import Path
from libero.gpu.host.runtime.dynamics.mjwarp_velocity import VelocityInputs
from libero.gpu.host.runtime.dynamics.mjwarp_bias import BiasInputs
from libero.gpu.host.runtime.dynamics.mjwarp_motion import reject_output_overlap


def source_id():
    from libero.gpu.host.paths import GPU_ROOT
    names=('cuda/fp32/fused_motion.cu','cuda/dynamics/mjwarp_velocity.cuh',
           'cuda/dynamics/mjwarp_bias.cuh','cmake/FP32Motion.cmake')
    return hashlib.sha256(':'.join(hashlib.sha256((GPU_ROOT/name).read_bytes()).hexdigest()
                                 for name in names).encode()).hexdigest()


class FusedMotion:
    def __init__(self,velocity,bias,library):
        from libero.gpu.host.build.compilation import MissingCompiledProduct
        if (velocity.model is not bias.model or velocity.data is not bias.data or
                velocity.stream is not bias.stream or velocity.topology is not bias.topology):
            raise ValueError('Fused motion owners differ')
        for name in ('qvel','cdof','cvel','cdof_dot'):
            if velocity.arrays[name].ptr!=bias.arrays[name].ptr:
                raise ValueError('Fused motion bridge differs: '+name)
        arrays={'velocity_'+k:v for k,v in velocity.arrays.items()}
        arrays.update({'bias_'+k:v for k,v in bias.arrays.items() if k not in ('qvel','cdof','cvel','cdof_dot')})
        reject_output_overlap(arrays,('velocity_cvel','velocity_cdof_dot','bias_cacc','bias_cfrc_int','bias_qfrc_bias'))
        bias.prepare_fusion()
        self.components=velocity,bias;self.stream=velocity.stream;self.interval=bias.interval
        self.eligible=(self.interval is not None and velocity.model.nbody<=64 and velocity.model.nbranch<=32)
        self.path=Path(library);self.library=None
        if not self.eligible:return
        if not self.path.is_file():raise MissingCompiledProduct('Build flash_fp32_fused_motion offline')
        self.library=lib=ct.CDLL(str(self.path));lib.flash_fp32_motion_source_id.restype=ct.c_char_p
        if lib.flash_fp32_motion_source_id().decode()!=source_id():raise MissingCompiledProduct('Stale fused motion; rebuild offline')
        if lib.flash_fp32_motion_arch()!=self.stream.device.arch:raise ValueError('Fused motion architecture differs')
        for name,kind in (('velocity',VelocityInputs),('bias',BiasInputs)):
            size=getattr(lib,'flash_fp32_motion_'+name+'_bytes');size.restype=ct.c_size_t
            if size()!=ct.sizeof(kind):raise ValueError('Fused motion ABI differs: '+name)
        self.launch=lib.flash_fp32_motion_launch
        self.launch.argtypes=[ct.POINTER(VelocityInputs),ct.POINTER(BiasInputs),ct.c_void_p,ct.c_int,ct.c_void_p]
        self.launch.restype=ct.c_int

    @property
    def identity(self):
        return dict(selected='world-motion-bias' if self.eligible else 'component-fallback',source_id=source_id(),
            library_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest() if self.library is not None else None,
            numerical_workspace_bytes=0,additional_topology_bytes=0,values='borrow-current-input-no-cache')

    def enqueue(self):
        v,b=self.components
        if not self.eligible:
            v.enqueue_fused();b.enqueue_fused();return
        error=self.launch(ct.byref(v.arguments),ct.byref(b.arguments),self.interval.ptr,
                          int(b.gravity_disabled),self.stream.cuda_stream)
        if error:raise RuntimeError('Fused motion launch failed: CUDA error '+str(error))
