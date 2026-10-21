"""Load-only resident initialization/stopping for the pinned FP32 Newton solver."""
import ctypes as ct
import hashlib
from pathlib import Path


class InitInputs(ct.Structure):
    _fields_ = [(k,ct.c_int) for k in ('worlds','nv','pad','rows','mass_entries','warmstart')]
    _fields_ += [(k,ct.c_void_p) for k in ('nefc','mass_row','mass_col','mass_address','mass',
        'jacobian','aref','warm_acceleration','smooth_acceleration','qacc','ma','jaref','search_dot','niter','done')]


class StopInputs(ct.Structure):
    _fields_ = [(k,ct.c_int) for k in ('worlds','nv','iterations','tolerance_count','inertia_count','warn_overflow')]
    _fields_ += [(k,ct.c_void_p) for k in ('tolerance','inertia','grad_dot','decrement','improvement',
                                        'alpha','niter','overflow','nsolving','done')]


def source_id():
    from libero.gpu.host.paths import GPU_ROOT
    names=('cuda/newton/mjwarp_solver_state.cu','cuda/newton/mjwarp_solver_state.cuh',
           'cuda/newton/mjwarp_products.cuh','cmake/MjwarpSolverState.cmake')
    return hashlib.sha256(':'.join(hashlib.sha256((GPU_ROOT/n).read_bytes()).hexdigest()
                                  for n in names).encode()).hexdigest()


class SolverStatePlan:
    """Qualify the binary outside capture; numerical arrays stay device resident."""

    def __init__(self,model,data,library,stream):
        from libero.gpu.host.runtime.cuda_api import type_api
        types = type_api(model)
        from libero.gpu.host.build.compilation import MissingCompiledProduct
        if (model.is_sparse or model.opt.cone!=types.ConeType.ELLIPTIC or
                model.opt.solver!=types.SolverType.NEWTON or
                model.opt.integrator==types.IntegratorType.DISCRETE or
                model.opt.enableflags & types.EnableBit.SLEEP):
            raise ValueError('Solver state requires dense elliptic full-DOF Newton')
        if model.nv<1 or data.nworld<1 or data.njmax<1 or model.opt.iterations<0:
            raise ValueError('invalid solver state dimensions or iteration count')
        self.library_path=Path(library).resolve()
        if not self.library_path.is_file():
            raise MissingCompiledProduct('Build flash_mjwarp_solver_state_fp32 offline')
        self.library=ct.CDLL(str(self.library_path))
        self.library.flash_mjwarp_solver_state_source_id.restype=ct.c_char_p
        if self.library.flash_mjwarp_solver_state_source_id().decode()!=source_id():
            raise MissingCompiledProduct('Solver state binary is stale; rebuild its CMake target offline')
        for name,structure in (('init',InitInputs),('stop',StopInputs)):
            method=getattr(self.library,'flash_mjwarp_solver_state_'+name+'_bytes');method.restype=ct.c_size_t
            if method()!=ct.sizeof(structure):raise ValueError('Solver state C/Python ABI differs: '+name)
        if not stream.device.is_cuda or self.library.flash_mjwarp_solver_state_arch()!=stream.device.arch:
            raise MissingCompiledProduct('Solver state binary architecture differs')
        self.model,self.data,self.stream=model,data,stream

    def bind(self,context):
        return DenseSolverState(self,context)


class DenseSolverState:
    """Host pointer descriptions only, retaining per-solve and shared data owners."""

    def __init__(self,plan,context):
        from libero.gpu.host.runtime.cuda_api import array_api
        wp = array_api(plan.stream)
        from libero.gpu.host.runtime.cuda_api import type_api
        types = type_api(plan.model)
        if getattr(context,'compact_d_full',None) is not None:
            raise ValueError('compact contexts are outside the solver state domain')
        self.plan=plan
        m,d,s=plan.model,plan.data,plan.stream
        n,nv,pad,rows=int(d.nworld),int(m.nv),int(m.nv_pad),int(d.njmax)
        init=dict(nefc=d.nefc,mass_row=m.M_mulm_rowadr,mass_col=m.M_mulm_col,mass_address=m.M_mulm_madr,
            mass=d.M,jacobian=d.efc.J,aref=d.efc.aref,warm_acceleration=d.qacc_warmstart,
            smooth_acceleration=d.qacc_smooth,qacc=d.qacc,ma=d.efc.Ma,jaref=context.Jaref,
            search_dot=context.search_dot,niter=d.solver_niter,done=context.done)
        stop=dict(tolerance=m.opt.tolerance,inertia=m.stat.meaninertia,grad_dot=context.grad_dot,
            decrement=context.newton_decrement,improvement=context.improvement,alpha=context.alpha,
            niter=d.solver_niter,overflow=d.overflow,done=context.done)
        shapes={k:(n,) for k in ('nefc','search_dot','niter','done','grad_dot','decrement','improvement','alpha','overflow')}
        shapes.update({k:(n,nv) for k in ('warm_acceleration','smooth_acceleration','qacc','ma')})
        shapes.update(mass_row=(nv+1,),jacobian=(n,rows,pad),aref=(n,rows),jaref=(n,rows))
        for name,a in dict(init,**stop).items():
            if a.device!=s.device or not a.is_contiguous:
                raise ValueError('Solver state array device/layout differs: '+name)
            if name in shapes and tuple(a.shape)!=shapes[name]:
                raise ValueError('Solver state array shape differs: '+name)
            expected=wp.int32 if name in ('nefc','mass_row','mass_col','mass_address','niter','overflow') else (
                wp.bool if name=='done' else wp.float32)
            if a.dtype!=expected:raise ValueError('Solver state array dtype differs: '+name)
        if d.M.ndim!=2 or d.M.shape[0]!=n or init['mass_col'].ndim!=1 or (
                init['mass_col'].shape!=init['mass_address'].shape):
            raise ValueError('Solver state sparse symmetric mass layout differs')
        for name in ('tolerance','inertia'):
            if tuple(stop[name].shape) not in ((1,),(n,)):
                raise ValueError('Solver state scalar must broadcast or be per-world: '+name)
        self.initial_arrays,self.stop_arrays,self.stream=init,stop,s
        self.init_arguments=InitInputs(worlds=n,nv=nv,pad=pad,rows=rows,mass_entries=d.M.shape[1],
            warmstart=int(not(m.opt.disableflags & types.DisableBit.WARMSTART)),**{k:a.ptr for k,a in init.items()})
        self.stop_arguments=StopInputs(worlds=n,nv=nv,iterations=m.opt.iterations,
            tolerance_count=stop['tolerance'].size,inertia_count=stop['inertia'].size,
            warn_overflow=int(m.opt.warn_overflow),**{k:a.ptr for k,a in stop.items()})
        self.initialize_launch=plan.library.flash_mjwarp_solver_state_initialize
        self.initialize_launch.argtypes=[ct.POINTER(InitInputs),ct.c_void_p];self.initialize_launch.restype=ct.c_int
        self.stop_launch=plan.library.flash_mjwarp_solver_state_stop
        self.stop_launch.argtypes=[ct.POINTER(StopInputs),ct.c_void_p];self.stop_launch.restype=ct.c_int

    def bind_stopping(self,nsolving):
        from libero.gpu.host.runtime.cuda_api import array_api
        wp = array_api(self.stream)
        if nsolving.device!=self.stream.device or not nsolving.is_contiguous or (
                nsolving.dtype!=wp.int32 or tuple(nsolving.shape)!=(1,)):
            raise ValueError('Solver state conditional counter layout differs')
        if 'nsolving' in self.stop_arrays and self.stop_arrays['nsolving'].ptr!=nsolving.ptr:
            raise ValueError('a solver state binding cannot change its conditional counter')
        self.stop_arrays['nsolving']=nsolving
        self.stop_arguments.nsolving=nsolving.ptr

    def initialize(self):
        error=self.initialize_launch(ct.byref(self.init_arguments),self.stream.cuda_stream)
        if error:raise RuntimeError('Handwritten Newton initialization failed: CUDA error '+str(error))

    def stop(self):
        if not self.stop_arguments.nsolving:raise RuntimeError('bind the per-solve conditional counter first')
        error=self.stop_launch(ct.byref(self.stop_arguments),self.stream.cuda_stream)
        if error:raise RuntimeError('Handwritten Newton stopping failed: CUDA error '+str(error))
