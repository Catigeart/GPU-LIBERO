"""Load-only current actuation/RHS/backsubstitution and accepted fallback."""
import ctypes as ct
import hashlib
from pathlib import Path
from libero.gpu.host.runtime.dynamics.mjwarp_actuation import ActuationInputs
from libero.gpu.host.runtime.dynamics.mjwarp_smooth_force import SmoothForceInputs
from libero.gpu.host.runtime.dynamics.mjwarp_mass_solve import MassSolveInputs
from libero.gpu.host.runtime.dynamics.mjwarp_mass_factor import world_groups
from libero.gpu.host.runtime.dynamics.mjwarp_motion import reject_output_overlap


def source_id():
    from libero.gpu.host.paths import GPU_ROOT
    names=('cuda/fp32/fused_actuation_solve.cu','cuda/dynamics/mjwarp_actuation.cuh',
           'cuda/dynamics/mjwarp_smooth_force.cuh','cuda/dynamics/mjwarp_mass_solve.cuh',
           'cmake/fp32/FP32ActuationSolve.cmake')
    return hashlib.sha256(':'.join(hashlib.sha256((GPU_ROOT/n).read_bytes()).hexdigest()
                                  for n in names).encode()).hexdigest()


class FusedActuationSolve:
    def __init__(self,actuation,solve,library):
        from libero.gpu.host.build.compilation import MissingCompiledProduct
        a,s,p=actuation,actuation.smooth,solve.plan
        if a.model is not p.model or a.data is not p.data or a.stream is not p.stream:
            raise ValueError('Actuation/mass solve owners differ')
        if s.arrays['smooth'].ptr!=solve.rhs.ptr or solve.rhs.shape[1]!=a.model.nv:
            raise ValueError('Actuation/mass solve RHS bridge differs')
        arrays={'act_'+k:v for k,v in a.arrays.items()}
        arrays.update({'smooth_'+k:v for k,v in s.arrays.items() if k!='actuator'})
        arrays.update(factor=p.data.qLD,diagonal=p.data.qLDiagInv,
                      address=p.model.qLD_block_adr)
        for i,g in enumerate(p.groups):arrays['block'+str(i)]=g['starts']
        outputs=['act_force','act_qfrc','smooth_smooth']
        if solve.solution.ptr!=solve.rhs.ptr:
            arrays['solution']=solve.solution;outputs.append('solution')
        reject_output_overlap(arrays,outputs)
        self.actuation,self.solve,self.stream=a,solve,a.stream
        self.group_arguments=world_groups(solve.arguments,MassSolveInputs,allow_single=True)
        self.eligible=self.group_arguments is not None
        self.path=Path(library);self.library=None
        if not self.eligible:return
        if not self.path.is_file():raise MissingCompiledProduct('Build flash_fp32_fused_actuation_solve offline')
        self.library=lib=ct.CDLL(str(self.path))
        if lib.flash_fp32_actuation_solve_arch()!=self.stream.device.arch:raise ValueError('Actuation/mass architecture differs')
        for name,kind in (('actuation',ActuationInputs),('smooth',SmoothForceInputs),('mass',MassSolveInputs)):
            size=getattr(lib,'flash_fp32_actuation_solve_'+name+'_bytes');size.restype=ct.c_size_t
            if size()!=ct.sizeof(kind):raise ValueError('Actuation/mass ABI differs: '+name)
        self.launch=lib.flash_fp32_actuation_solve_launch
        self.launch.argtypes=[ct.POINTER(ActuationInputs),ct.POINTER(SmoothForceInputs)]+[ct.POINTER(MassSolveInputs)]*3+[ct.c_void_p]
        self.launch.restype=ct.c_int

    def enqueue(self):
        a=self.actuation
        if not self.eligible:a.enqueue_smooth();self.solve.enqueue_fused();return
        error=self.launch(ct.byref(a.arguments),ct.byref(a.smooth.arguments),
                          *(ct.byref(v) for v in self.group_arguments),self.stream.cuda_stream)
        if error:raise RuntimeError('Actuation/mass solve failed: CUDA error '+str(error))
