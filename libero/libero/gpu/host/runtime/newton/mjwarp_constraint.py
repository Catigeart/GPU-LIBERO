"""Load-only binding for dense nonincremental MJWarp FP32 constraint updates."""
import ctypes as ct
import hashlib
from pathlib import Path


class ConstraintInputs(ct.Structure):
    _fields_ = [(name, ct.c_int) for name in
                ('worlds', 'nv', 'pad', 'rows', 'capacity', 'impratio_count')]
    _fields_ += [(name, ct.c_void_p) for name in
                ('done', 'ne', 'nf', 'nefc', 'contact_count', 'contact_dimension', 'contact_address',
                 'impratio', 'contact_friction', 'row_type', 'row_contact', 'diagonal',
                 'frictionloss', 'jaref', 'jacobian', 'force', 'state', 'qfrc_constraint')]


def source_id():
    from libero.gpu.host.paths import GPU_ROOT
    names = ('cuda/newton/mjwarp_constraint.cu', 'cuda/newton/mjwarp_constraint.cuh',
             'cmake/MjwarpConstraint.cmake')
    hashes = ':'.join(hashlib.sha256((GPU_ROOT / n).read_bytes()).hexdigest() for n in names)
    return hashlib.sha256(hashes.encode()).hexdigest()


class ConstraintPlan:
    """Admit the product before capture, then bind existing per-solve arrays."""

    def __init__(self, model, data, library, stream):
        from mujoco_warp._src import types
        from libero.gpu.host.build.compilation import MissingCompiledProduct
        if (model.is_sparse or model.opt.cone != types.ConeType.ELLIPTIC or
                model.opt.solver != types.SolverType.NEWTON or
                model.opt.integrator == types.IntegratorType.DISCRETE or
                model.opt.enableflags & types.EnableBit.SLEEP):
            raise ValueError('Constraint leaf requires dense elliptic full-DOF Newton')
        if not 0 < model.nv <= model.nv_pad or data.nworld < 1 or data.njmax < 1:
            raise ValueError('invalid constraint dimensions')
        self.library_path = Path(library).resolve()
        if not self.library_path.is_file():
            raise MissingCompiledProduct('Build flash_mjwarp_constraint_fp32 offline')
        self.library = ct.CDLL(str(self.library_path))
        self.library.flash_mjwarp_constraint_source_id.restype = ct.c_char_p
        self.library.flash_mjwarp_constraint_arguments_bytes.restype = ct.c_size_t
        if self.library.flash_mjwarp_constraint_source_id().decode() != source_id():
            raise MissingCompiledProduct('Constraint binary is stale; rebuild its CMake target offline')
        if self.library.flash_mjwarp_constraint_arguments_bytes() != ct.sizeof(ConstraintInputs):
            raise ValueError('Constraint C/Python ABI differs')
        if not stream.device.is_cuda or self.library.flash_mjwarp_constraint_arch() != stream.device.arch:
            raise MissingCompiledProduct('Constraint binary architecture differs')
        self.model, self.data, self.stream = model, data, stream

    def bind(self, context):
        return DenseConstraint(self, context)


class DenseConstraint:
    """Metadata/pointers only; no capture-time allocation or data transfer."""

    def __init__(self, plan, context):
        import warp as wp
        from mujoco_warp._src import types
        self.plan = plan
        m, d, s = plan.model, plan.data, plan.stream
        n, nv, pad, rows, cap = int(d.nworld), int(m.nv), int(m.nv_pad), int(d.njmax), int(d.naconmax)
        arrays = dict(done=context.done, ne=d.ne, nf=d.nf, nefc=d.nefc, contact_count=d.nacon,
            contact_dimension=d.contact.dim, contact_address=d.contact.efc_address,
            impratio=m.opt.impratio_invsqrt, contact_friction=d.contact.friction,
            row_type=d.efc.type, row_contact=d.efc.id, diagonal=d.efc.D,
            frictionloss=d.efc.frictionloss, jaref=context.Jaref, jacobian=d.efc.J,
            force=d.efc.force, state=d.efc.state, qfrc_constraint=d.qfrc_constraint)
        shapes = dict(done=(n,), ne=(n,), nf=(n,), nefc=(n,), contact_count=(1,),
            contact_dimension=(cap,), contact_address=(cap,6), row_type=(n,rows),
            row_contact=(n,rows), diagonal=(n,rows), frictionloss=(n,rows),
            jaref=(n,rows), jacobian=(n,rows,pad), force=(n,rows), state=(n,rows),
            qfrc_constraint=(n,nv))
        integer = {'ne','nf','nefc','contact_count','contact_dimension','contact_address','row_type','row_contact','state'}
        for name, a in arrays.items():
            if a.device != s.device or not a.is_contiguous:
                raise ValueError('Constraint array device/layout differs: ' + name)
            if name in shapes and tuple(a.shape) != shapes[name]:
                raise ValueError('Constraint shape differs: ' + name)
            if name == 'contact_friction':
                if not ((a.dtype == types.vec5 and tuple(a.shape) == (cap,)) or
                        (a.dtype == wp.float32 and tuple(a.shape) == (cap,5))):
                    raise ValueError('Constraint friction layout differs')
            elif a.dtype != (wp.int32 if name in integer else wp.bool if name == 'done' else wp.float32):
                raise ValueError('Constraint dtype differs: ' + name)
        if tuple(arrays['impratio'].shape) not in ((1,), (n,)):
            raise ValueError('Constraint impratio must broadcast or be per-world')
        self.arrays, self.stream = arrays, s
        self.arguments = ConstraintInputs(worlds=n, nv=nv, pad=pad, rows=rows, capacity=cap,
            impratio_count=arrays['impratio'].size, **{name:a.ptr for name,a in arrays.items()})
        self.launch = plan.library.flash_mjwarp_constraint_launch
        self.launch.argtypes = [ct.POINTER(ConstraintInputs),ct.c_int,ct.c_void_p]
        self.launch.restype = ct.c_int

    def enqueue(self, stage=0):
        error = self.launch(ct.byref(self.arguments),stage,self.stream.cuda_stream)
        if error:
            raise RuntimeError('Handwritten constraint launch failed: CUDA error ' + str(error))
