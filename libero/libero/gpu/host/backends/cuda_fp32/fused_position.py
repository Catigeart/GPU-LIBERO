"""Load-only composition of admitted centered-frame and mass bindings."""
import ctypes as ct
import hashlib
from pathlib import Path
from libero.gpu.host.runtime.dynamics.mjwarp_com_pos import ComPosInputs
from libero.gpu.host.runtime.dynamics.mjwarp_mass_matrix import MassMatrixInputs
from libero.gpu.host.runtime.dynamics.mjwarp_motion import reject_output_overlap


def source_id():
    from libero.gpu.host.paths import GPU_ROOT
    names=('cuda/fp32/fused_position.cu','cuda/dynamics/mjwarp_com_pos.cuh',
           'cuda/dynamics/mjwarp_mass_matrix.cuh','cmake/fp32/FP32Position.cmake')
    return hashlib.sha256(':'.join(hashlib.sha256((GPU_ROOT/name).read_bytes()).hexdigest()
                                 for name in names).encode()).hexdigest()


class FusedCenteredMass:
    def __init__(self,centered,mass,library):
        from libero.gpu.host.build.compilation import MissingCompiledProduct
        if (centered.model is not mass.model or centered.data is not mass.data or
                centered.stream is not mass.stream or centered.tree is not mass.tree):
            raise ValueError('Fused position owners differ')
        for name in ('cinert','cdof'):
            if centered.arrays[name].ptr!=mass.arrays[name].ptr:
                raise ValueError('Fused position producer/consumer differs: '+name)
        arrays={'centered_'+k:v for k,v in centered.arrays.items()}
        arrays.update({'mass_'+k:v for k,v in mass.arrays.items() if k not in ('cinert','cdof')})
        reject_output_overlap(arrays,('centered_subtree','centered_cinert','centered_cdof','mass_crb','mass_matrix'))
        self.components=centered,mass
        self.tree,self.stream=centered.tree,centered.stream
        self.interval=self.tree.prepare_intervals()
        self.path=Path(library)
        if not self.path.is_file():raise MissingCompiledProduct('Build flash_fp32_fused_position offline')
        self.library=lib=ct.CDLL(str(self.path));lib.flash_fp32_position_source_id.restype=ct.c_char_p
        if lib.flash_fp32_position_source_id().decode()!=source_id():raise MissingCompiledProduct('Stale fused position; rebuild offline')
        if lib.flash_fp32_position_arch()!=self.stream.device.arch:raise ValueError('Fused position architecture differs')
        for name,kind in (('centered',ComPosInputs),('mass',MassMatrixInputs)):
            size=getattr(lib,'flash_fp32_position_'+name+'_bytes');size.restype=ct.c_size_t
            if size()!=ct.sizeof(kind):raise ValueError('Fused position ABI differs: '+name)
        self.launch=lib.flash_fp32_position_launch
        self.launch.argtypes=[ct.POINTER(ComPosInputs),ct.POINTER(MassMatrixInputs),ct.c_void_p,
            ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_void_p];self.launch.restype=ct.c_int

    def enqueue(self):
        c,m=self.components;t=self.tree
        error=self.launch(ct.byref(c.arguments),ct.byref(m.arguments),t.bodies.ptr,t.starts.ptr,
                          t.levels,t.child_start.ptr,t.child_id.ptr,
                          self.interval.ptr if self.interval is not None else None,self.stream.cuda_stream)
        if error:raise RuntimeError('Fused centered/mass launch failed: CUDA error '+str(error))
