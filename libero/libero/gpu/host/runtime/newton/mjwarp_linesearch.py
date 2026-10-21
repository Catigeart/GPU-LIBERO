"""Load-only resident binding for the pinned dense FP32 Newton line search."""
import ctypes as ct
import hashlib
from pathlib import Path


class LinesearchInputs(ct.Structure):
    _fields_ = [(name, ct.c_int) for name in
        ('worlds','nv','pad','rows','mass_entries','capacity','tolerance_count',
         'ls_tolerance_count','impratio_count','inertia_count','iterations','warn_overflow')]
    _fields_ += [(name, ct.c_void_p) for name in
        ('done','ne','nf','nefc','contact_count','mass_row','mass_col','mass_address',
         'mass','jacobian','search','search_dot','smooth','tolerance','ls_tolerance',
         'impratio','inertia','row_type','row_contact','contact_dimension','contact_address',
         'contact_friction','diagonal','frictionloss','mv','jv','quad','qacc','ma','jaref',
         'improvement','alpha','overflow')]


def source_id():
    from libero.gpu.host.paths import GPU_ROOT
    names = ('cuda/newton/mjwarp_linesearch.cu','cuda/newton/mjwarp_linesearch.cuh',
             'cuda/newton/mjwarp_products.cuh', 'cmake/MjwarpLinesearch.cmake')
    return hashlib.sha256(':'.join(hashlib.sha256((GPU_ROOT/n).read_bytes()).hexdigest()
                                  for n in names).encode()).hexdigest()


def unsupported_reason(model):
    if model.block_dim.linesearch_iterative != 32:
        return 'unqualified linesearch block dimension'
    return None


class LinesearchPlan:
    """Admit source, ABI, device and domain before any graph capture."""

    def __init__(self, model, data, library, stream):
        from libero.gpu.host.runtime.cuda_api import type_api
        types = type_api(model)
        from libero.gpu.host.build.compilation import MissingCompiledProduct
        if (model.is_sparse or model.opt.cone != types.ConeType.ELLIPTIC or
                model.opt.solver != types.SolverType.NEWTON or
                model.opt.integrator == types.IntegratorType.DISCRETE or
                model.opt.enableflags & types.EnableBit.SLEEP or unsupported_reason(model)):
            raise ValueError('Linesearch requires dense elliptic full-DOF Newton with block dimension 32')
        if model.opt.ls_iterations < 0 or model.nv < 1 or data.njmax < 1 or data.nworld < 1:
            raise ValueError('invalid linesearch dimensions or iteration count')
        self.library_path = Path(library).resolve()
        if not self.library_path.is_file():
            raise MissingCompiledProduct('Build flash_mjwarp_linesearch_fp32 offline')
        self.library = ct.CDLL(str(self.library_path))
        self.library.flash_mjwarp_linesearch_source_id.restype = ct.c_char_p
        self.library.flash_mjwarp_linesearch_arguments_bytes.restype = ct.c_size_t
        if self.library.flash_mjwarp_linesearch_source_id().decode() != source_id():
            raise MissingCompiledProduct('Linesearch binary is stale; rebuild its CMake target offline')
        if self.library.flash_mjwarp_linesearch_arguments_bytes() != ct.sizeof(LinesearchInputs):
            raise ValueError('Linesearch C/Python ABI differs')
        if not stream.device.is_cuda or self.library.flash_mjwarp_linesearch_arch() != stream.device.arch:
            raise MissingCompiledProduct('Linesearch binary architecture differs')
        self.model, self.data, self.stream = model, data, stream

    def bind(self, context):
        return DenseLinesearch(self, context)


class DenseLinesearch:
    """Borrow pointers and retain their owners; no capture-time transfer/allocation."""

    def __init__(self, plan, context):
        from libero.gpu.host.runtime.cuda_api import array_api
        wp = array_api(plan.stream)
        from libero.gpu.host.runtime.cuda_api import type_api
        types = type_api(plan.model)
        if getattr(context,'compact_d_full',None) is not None:
            raise ValueError('compact solver contexts are outside the linesearch domain')
        self.plan = plan
        m,d,s = plan.model,plan.data,plan.stream
        n,nv,pad,rows,cap = int(d.nworld),int(m.nv),int(m.nv_pad),int(d.njmax),int(d.naconmax)
        arrays = dict(done=context.done,ne=d.ne,nf=d.nf,nefc=d.nefc,contact_count=d.nacon,
            mass_row=m.M_mulm_rowadr,mass_col=m.M_mulm_col,mass_address=m.M_mulm_madr,
            mass=d.M,jacobian=d.efc.J,search=context.search,search_dot=context.search_dot,
            smooth=d.qfrc_smooth,tolerance=m.opt.tolerance,ls_tolerance=m.opt.ls_tolerance,
            impratio=m.opt.impratio_invsqrt,inertia=m.stat.meaninertia,row_type=d.efc.type,
            row_contact=d.efc.id,contact_dimension=d.contact.dim,contact_address=d.contact.efc_address,
            contact_friction=d.contact.friction,diagonal=d.efc.D,frictionloss=d.efc.frictionloss,
            mv=context.mv,jv=context.jv,quad=context.quad,qacc=d.qacc,ma=d.efc.Ma,
            jaref=context.Jaref,improvement=context.improvement,alpha=context.alpha,overflow=d.overflow)
        shapes = {k:(n,) for k in ('done','ne','nf','nefc','search_dot','improvement','alpha','overflow')}
        shapes.update({k:(n,nv) for k in ('search','smooth','mv','qacc','ma')})
        shapes.update({k:(n,rows) for k in ('row_type','row_contact','diagonal','frictionloss','jv','jaref')})
        shapes.update(contact_count=(1,),mass_row=(nv+1,),jacobian=(n,rows,pad),
            contact_dimension=(cap,),contact_address=(cap,6),quad=(n,rows))
        integer = {'ne','nf','nefc','contact_count','mass_row','mass_col','mass_address',
                   'row_type','row_contact','contact_dimension','contact_address','overflow'}
        for name,a in arrays.items():
            if a.device != s.device or not a.is_contiguous:
                raise ValueError('Linesearch array device/layout differs: ' + name)
            if name in shapes and tuple(a.shape) != shapes[name]:
                raise ValueError('Linesearch array shape differs: ' + name)
            expected = types.vec5 if name=='contact_friction' else wp.vec3 if name=='quad' else (
                wp.int32 if name in integer else wp.bool if name=='done' else wp.float32)
            if a.dtype != expected:
                raise ValueError('Linesearch array dtype differs: ' + name)
        if d.M.ndim != 2 or d.M.shape[0] != n:
            raise ValueError('Linesearch requires sparse symmetric mass storage')
        for name in ('tolerance','ls_tolerance','impratio','inertia'):
            if tuple(arrays[name].shape) not in ((1,),(n,)):
                raise ValueError('Linesearch scalar must broadcast or be per-world: ' + name)
        self.arrays,self.stream = arrays,s
        self.arguments = LinesearchInputs(worlds=n,nv=nv,pad=pad,rows=rows,
            mass_entries=d.M.shape[1],capacity=cap,tolerance_count=arrays['tolerance'].size,
            ls_tolerance_count=arrays['ls_tolerance'].size,impratio_count=arrays['impratio'].size,
            inertia_count=arrays['inertia'].size,iterations=m.opt.ls_iterations,
            warn_overflow=int(m.opt.warn_overflow),**{k:a.ptr for k,a in arrays.items()})
        self.launch = plan.library.flash_mjwarp_linesearch_launch
        self.launch.argtypes = [ct.POINTER(LinesearchInputs),ct.c_int,ct.c_void_p]
        self.launch.restype = ct.c_int

    def enqueue(self, stage=0):
        error = self.launch(ct.byref(self.arguments),stage,self.stream.cuda_stream)
        if error:
            raise RuntimeError('Handwritten linesearch launch failed: CUDA error ' + str(error))
