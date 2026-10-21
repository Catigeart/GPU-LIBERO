"""Load-only preparation/OSC/publication with the accepted large-NV fallback."""
import ctypes as ct
import hashlib
from pathlib import Path
from .fused_control_prepare import ControlPrepareInputs
from libero.gpu.host.runtime.step.substep_types import ControllerSolveData
from libero.gpu.host.runtime.dynamics.mjwarp_motion import bind_array,reject_output_overlap


def source_id():
    from libero.gpu.host.paths import GPU_ROOT
    names=('cuda/fp32/fused_controller.cu', 'cuda/fp32/fused_control_prepare.cuh', 'cuda/runtime/bridge_types.cuh', 'cuda/dynamics/mjwarp_queries.cuh', 'cuda/control/osc_solver.cu', 'cuda/control/osc_solver.cuh', 'cuda/control/osc_profile.cuh', 'cuda/step/substep_shared_layout.cuh', 'cuda/step/substep_shape.cuh', 'cuda/step/substep_parameters.cuh', 'cuda/step/substep_types.cuh', 'cuda/common/precision.cuh', 'cuda/common/vector.cuh', 'cuda/step/instantiations/osc_mathdx.cu', 'cmake/FP32Controller.cmake')
    return hashlib.sha256(':'.join(hashlib.sha256((GPU_ROOT/n).read_bytes()).hexdigest()
                                  for n in names).encode()).hexdigest()


class FusedController:
    def __init__(self,c,library):
        import numpy as np
        from libero.gpu.host.runtime.cuda_api import array_api
        from libero.gpu.host.build.compilation import MissingCompiledProduct
        wp=array_api(c.stream)
        self.preparation=c.preparation;self.stream=c.stream
        self.args,self.ids,self.target=c.args,c.ids,c.data.ctrl
        self.worlds,self.stride=c.n,c.model.nu
        self.publish=c.publish
        self.owners=tuple(c.buffers.values()),tuple(c.policy_values)
        arrays=dict(c.preparation.arrays)
        for key,a,t,shape in (('control_blob',c.args,wp.uint8,(ct.sizeof(ControllerSolveData),)),
                ('actuator_ids',c.ids,wp.int32,(9,)),('target',c.data.ctrl,wp.float32,(c.n,c.model.nu))):
            bind_array(arrays,key,a,t,shape,c.stream)
        ids=c.ids.numpy()
        if np.any(ids<0) or np.any(ids>=c.model.nu) or len(np.unique(ids))!=9:
            raise ValueError('Fused controller actuator publication indices differ')
        arrays.update({'control_'+k:a for k,a in c.buffers.items()})
        reject_output_overlap(arrays,('target',))
        self.eligible=c.model.nv<=64
        self.path=Path(library);self.library=None
        if not self.eligible:return
        if not self.path.is_file():raise MissingCompiledProduct('Build flash_fp32_fused_controller offline')
        self.library=lib=ct.CDLL(str(self.path));lib.flash_fp32_controller_source_id.restype=ct.c_char_p
        if lib.flash_fp32_controller_source_id().decode()!=source_id():raise MissingCompiledProduct('Stale fused controller; rebuild offline')
        if lib.flash_fp32_controller_arch()!=c.stream.device.arch:raise ValueError('Fused controller architecture differs')
        for entry,kind in (('prepare',ControlPrepareInputs),('solve',ControllerSolveData)):
            size=getattr(lib,'flash_fp32_controller_'+entry+'_bytes');size.restype=ct.c_size_t
            if size()!=ct.sizeof(kind):raise ValueError('Fused controller ABI differs: '+entry)
        self.launch=lib.flash_fp32_controller_launch
        self.launch.argtypes=[ct.POINTER(ControlPrepareInputs),ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_void_p]
        self.launch.restype=ct.c_int

    def enqueue(self,policy_source):
        from libero.gpu.host.runtime.cuda_api import array_api
        wp=array_api(self.stream)
        if (policy_source.device!=self.stream.device or policy_source.dtype!=wp.int32 or
                not policy_source.is_contiguous or policy_source.shape!=(self.worlds,)):
            raise ValueError('Fused controller policy source differs')
        if not self.eligible:
            self.preparation.enqueue(policy_source)
            error=self.publish(self.args.ptr,self.worlds,self.ids.ptr,self.target.ptr,self.stride,self.stream.cuda_stream)
        else:
            error=self.launch(ct.byref(self.preparation.arguments),policy_source.ptr,self.args.ptr,self.ids.ptr,
                              self.target.ptr,self.stride,self.stream.cuda_stream)
        if error:raise RuntimeError('Fused controller failed: CUDA error '+str(error))
