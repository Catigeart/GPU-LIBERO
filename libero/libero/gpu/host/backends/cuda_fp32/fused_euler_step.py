"""Load-only damped Euler factor/advance and accepted component fallback."""
import ctypes as ct
import hashlib
from pathlib import Path
from .fused_tail import TailInputs
from libero.gpu.host.runtime.dynamics.mjwarp_mass_factor import MassFactorInputs
from libero.gpu.host.runtime.dynamics.mjwarp_motion import reject_output_overlap


def source_id():
    from libero.gpu.host.paths import GPU_ROOT
    names=('cuda/fp32/fused_euler_step.cu','cuda/fp32/fused_tail.cuh',
           'cuda/dynamics/mjwarp_euler.cuh','cuda/dynamics/mjwarp_publication.cuh',
           'cuda/dynamics/mjwarp_mass_factor.cuh','cmake/FP32EulerStep.cmake')
    return hashlib.sha256(':'.join(hashlib.sha256((GPU_ROOT/n).read_bytes()).hexdigest()
                                  for n in names).encode()).hexdigest()


class FusedEulerStep:
    def __init__(self,factor,advance,library):
        from libero.gpu.host.build.compilation import MissingCompiledProduct
        e=factor.euler
        if e is not advance.euler or factor.plan.stream is not advance.stream:
            raise ValueError('Euler factor/advance owners differ')
        self.factor,self.advance,self.stream=factor,advance,advance.stream
        self.eligible=(factor.preparation_eligible and isinstance(advance.arguments,TailInputs)
            and advance.arguments.publication.warnings==0
            and factor.arrays['solution'].ptr==e.arrays['acceleration'].ptr
            and factor.arrays['solution'].shape==(e.data.nworld,e.model.nv))
        self.path=Path(library);self.library=None
        if not self.eligible:return
        arrays=dict(factor.arrays)
        if arrays['solution'].ptr==arrays['rhs'].ptr:arrays.pop('rhs')
        arrays.update({'euler_'+k:v for k,v in e.arrays.items() if k not in ('matrix','acceleration')})
        pub=advance.owners[1]
        arrays.update({'publication_'+k:v for k,v in pub.arrays.items()})
        arrays['solved_acceleration']=advance.owners[2];arrays['warmstart']=advance.owners[3]
        reject_output_overlap(arrays,('matrix','factor','diagonal','solution','euler_derivative',
            'euler_qpos','euler_qvel','publication_time','publication_overflow','warmstart'))
        if not self.path.is_file():raise MissingCompiledProduct('Build flash_fp32_fused_euler_step offline')
        self.library=lib=ct.CDLL(str(self.path));lib.flash_fp32_euler_step_source_id.restype=ct.c_char_p
        if lib.flash_fp32_euler_step_source_id().decode()!=source_id():raise MissingCompiledProduct('Stale Euler step; rebuild offline')
        if lib.flash_fp32_euler_step_arch()!=self.stream.device.arch:raise ValueError('Euler step architecture differs')
        for name,kind in (('mass',MassFactorInputs),('tail',TailInputs)):
            size=getattr(lib,'flash_fp32_euler_step_'+name+'_bytes');size.restype=ct.c_size_t
            if size()!=ct.sizeof(kind):raise ValueError('Euler step ABI differs: '+name)
        self.launch=lib.flash_fp32_euler_step_launch
        self.launch.argtypes=[ct.POINTER(MassFactorInputs)]*3+[ct.POINTER(TailInputs),ct.c_void_p,ct.c_void_p];self.launch.restype=ct.c_int

    def enqueue(self):
        if not self.eligible:self.factor.enqueue_prepared();self.advance.enqueue();return
        error=self.launch(*(ct.byref(a) for a in self.factor.group_arguments),
            ct.byref(self.advance.arguments),self.factor.euler.source_matrix.ptr,self.stream.cuda_stream)
        if error:raise RuntimeError('Euler step failed: CUDA error '+str(error))
