"""Load-only control/unconstrained region with separate accepted components."""
import ctypes as ct
import hashlib
from pathlib import Path
from .fused_control_prepare import ControlPrepareInputs
from libero.gpu.host.runtime.step.substep_types import ControllerSolveData
from libero.gpu.host.runtime.dynamics.mjwarp_actuation import ActuationInputs
from libero.gpu.host.runtime.dynamics.mjwarp_smooth_force import SmoothForceInputs
from libero.gpu.host.runtime.dynamics.mjwarp_mass_solve import MassSolveInputs
from libero.gpu.host.runtime.dynamics.mjwarp_motion import reject_output_overlap


def source_id():
    from libero.gpu.host.paths import GPU_ROOT
    names=('cuda/fp32/fused_control_drive.cu','cuda/fp32/fused_control_prepare.cuh',
        'cuda/runtime/bridge_types.cuh','cuda/dynamics/mjwarp_queries.cuh',
        'cuda/dynamics/mjwarp_actuation.cuh','cuda/dynamics/mjwarp_smooth_force.cuh',
        'cuda/dynamics/mjwarp_mass_solve.cuh','cuda/control/osc_solver.cu',
        'cuda/control/osc_solver.cuh','cuda/control/osc_profile.cuh',
        'cuda/step/substep_shared_layout.cuh','cuda/step/substep_shared.cuh',
        'cuda/step/forward_workspace.cuh','cuda/step/substep_shape.cuh',
        'cuda/step/substep_parameters.cuh','cuda/step/substep_types.cuh',
        'cuda/common/precision.cuh','cuda/common/vector.cuh',
        'cuda/step/instantiations/osc_mathdx.cu','cmake/FP32ControlDrive.cmake')
    return hashlib.sha256(':'.join(hashlib.sha256((GPU_ROOT/n).read_bytes()).hexdigest()
                                  for n in names).encode()).hexdigest()


class FusedControlDrive:
    def __init__(self,controller,unconstrained,library):
        from libero.gpu.host.build.compilation import MissingCompiledProduct
        c,a=controller,unconstrained.actuation
        if c.model is not a.model or c.data is not a.data or c.stream is not a.stream:
            raise ValueError('Control/actuation owners differ')
        if a.arrays['ctrl'].ptr!=c.data.ctrl.ptr or a.arguments.controls!=c.model.nu:
            raise ValueError('Control/actuation publication bridge differs')
        self.controller,self.unconstrained,self.stream=c,unconstrained,c.stream
        self.eligible=(c.fused.eligible and unconstrained.eligible
                       and a.arguments.actuators==9 and a.arguments.controls==9)
        self.path=Path(library);self.library=None
        if not self.eligible:return
        outputs={'force':a.arrays['force'],'qfrc':a.arrays['qfrc'],
                 'smooth':a.smooth.arrays['smooth']}
        if unconstrained.solve.solution.ptr!=unconstrained.solve.rhs.ptr:
            outputs['solution']=unconstrained.solve.solution
        cross={'prepare_'+k:v for k,v in c.preparation.arrays.items()}
        cross.update({'control_'+k:v for k,v in c.buffers.items()});cross['target']=c.data.ctrl
        cross.update(outputs);reject_output_overlap(cross,tuple(outputs))
        drive={k:v for k,v in a.arrays.items() if k!='ctrl'}
        drive.update({'smooth_'+k:v for k,v in a.smooth.arrays.items()})
        drive.update(factor=unconstrained.solve.plan.data.qLD,
                     diagonal=unconstrained.solve.plan.data.qLDiagInv,
                     solution=unconstrained.solve.solution)
        for value in list(c.buffers.values())+[c.data.ctrl,c.points,c.jp,c.jr]:
            if not value.capacity:continue
            for key,other in drive.items():
                if other.capacity and max(value.ptr,other.ptr)<min(value.ptr+value.capacity,other.ptr+other.capacity):
                    raise ValueError('Control output overlaps actuation/mass storage: '+key)
        if not self.path.is_file():raise MissingCompiledProduct('Build flash_fp32_fused_control_drive offline')
        self.library=lib=ct.CDLL(str(self.path));lib.flash_fp32_control_drive_source_id.restype=ct.c_char_p
        if lib.flash_fp32_control_drive_source_id().decode()!=source_id():raise MissingCompiledProduct('Stale control/drive; rebuild offline')
        if lib.flash_fp32_control_drive_arch()!=self.stream.device.arch:raise ValueError('Control/drive architecture differs')
        for name,kind in (('prepare',ControlPrepareInputs),('control',ControllerSolveData),
                          ('actuation',ActuationInputs),('smooth',SmoothForceInputs),('mass',MassSolveInputs)):
            size=getattr(lib,'flash_fp32_control_drive_'+name+'_bytes');size.restype=ct.c_size_t
            if size()!=ct.sizeof(kind):raise ValueError('Control/drive ABI differs: '+name)
        self.launch=lib.flash_fp32_control_drive_launch
        self.launch.argtypes=[ct.POINTER(ControlPrepareInputs),ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int,ct.POINTER(ActuationInputs),ct.POINTER(SmoothForceInputs)]+[ct.POINTER(MassSolveInputs)]*3+[ct.c_void_p];self.launch.restype=ct.c_int

    def enqueue(self,policy_source):
        from . import device as cuda,constants as types
        c=self.controller
        if not self.eligible or c.model.opt.disableflags&types.DisableBit.ACTUATION:
            c.fused.enqueue(policy_source);return False
        if (policy_source.device!=self.stream.device or policy_source.dtype!=cuda.int32 or
                not policy_source.is_contiguous or policy_source.shape!=(c.n,)):
            raise ValueError('Control/drive policy source differs')
        a=self.unconstrained.actuation
        error=self.launch(ct.byref(c.preparation.arguments),policy_source.ptr,c.args.ptr,c.ids.ptr,
            c.data.ctrl.ptr,c.model.nu,ct.byref(a.arguments),ct.byref(a.smooth.arguments),
            *(ct.byref(v) for v in self.unconstrained.group_arguments),self.stream.cuda_stream)
        if error:raise RuntimeError('Control/drive failed: CUDA error '+str(error))
        return True
