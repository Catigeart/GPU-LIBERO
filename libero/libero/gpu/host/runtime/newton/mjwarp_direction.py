"""Load-only pointer binding for the handwritten MJWarp gradient/direction stage."""
import ctypes as ct
import hashlib
from pathlib import Path

FAMILIES = (21, 22, 24, 27, 30, 33, 36, 37, 39, 43, 45, 51, 57)


class DirectionInputs(ct.Structure):
    _fields_ = [(name, ct.c_int) for name in
                ('worlds', 'nv', 'pad', 'ma_stride', 'smooth_stride', 'force_stride')]
    _fields_ += [(name, ct.c_void_p) for name in
                ('done', 'ma', 'smooth', 'force', 'hessian', 'grad', 'grad_dot',
                 'decrement', 'grad_scale', 'search_unchanged', 'factor', 'search', 'search_dot')]


def source_id():
    from libero.gpu.host.paths import GPU_ROOT
    names = ('cuda/newton/mjwarp_direction.cu', 'cuda/newton/mjwarp_direction.cuh',
             'cmake/MjwarpDirection.cmake')
    hashes = ':'.join(hashlib.sha256((GPU_ROOT / n).read_bytes()).hexdigest() for n in names)
    return hashlib.sha256(hashes.encode()).hexdigest()


def unsupported_reason(model):
    if model.nv not in FAMILIES:
        return 'unqualified direction NV'
    expected = 64 if model.nv <= 32 else 32
    field = 'update_gradient_cholesky' if model.nv <= 32 else 'update_gradient_cholesky_blocked'
    if getattr(model.block_dim, field) != expected:
        return 'unqualified direction block dimension'
    return None


class DirectionPlan:
    """Own the admitted binary outside capture; bind fresh scratch per solve."""

    def __init__(self, model, data, library, stream):
        from libero.gpu.host.build.compilation import MissingCompiledProduct
        reason = unsupported_reason(model)
        if reason:
            raise ValueError(reason)
        self.library_path = Path(library).resolve()
        if not self.library_path.is_file():
            raise MissingCompiledProduct('Build flash_mjwarp_direction_fp32 offline')
        self.library = ct.CDLL(str(self.library_path))
        self.library.flash_mjwarp_direction_source_id.restype = ct.c_char_p
        self.library.flash_mjwarp_direction_arguments_bytes.restype = ct.c_size_t
        if self.library.flash_mjwarp_direction_source_id().decode() != source_id():
            raise MissingCompiledProduct('Direction binary is stale; rebuild its CMake target offline')
        if self.library.flash_mjwarp_direction_arguments_bytes() != ct.sizeof(DirectionInputs):
            raise ValueError('Direction C/Python ABI differs')
        if not stream.device.is_cuda or self.library.flash_mjwarp_direction_arch() != stream.device.arch:
            raise MissingCompiledProduct('Direction binary architecture differs')
        self.model, self.data, self.stream = model, data, stream

    def bind(self, context):
        return DenseDirection(self, context)


class DenseDirection:
    """Metadata/pointers only: no allocation, readback or compilation in capture."""

    def __init__(self, plan, context):
        from libero.gpu.host.runtime.cuda_api import array_api
        wp = array_api(plan.stream)
        self.plan = plan
        m, d, s = plan.model, plan.data, plan.stream
        n, nv, pad = int(d.nworld), int(m.nv), int(m.nv_pad)
        arrays = dict(done=context.done, ma=d.efc.Ma, smooth=d.qfrc_smooth,
            force=d.qfrc_constraint, hessian=context.h, grad=context.grad,
            grad_dot=context.grad_dot, decrement=context.newton_decrement,
            grad_scale=context.grad_scale, search_unchanged=context.search_unchanged,
            factor=context.hfactor, search=context.search, search_dot=context.search_dot)
        shapes = dict(done=(n,), hessian=(n,pad,pad), grad=(n,pad), grad_dot=(n,),
            decrement=(n,), grad_scale=(n,), search_unchanged=(n,), search=(n,nv),
            search_dot=(n,), factor=(n,pad,pad) if nv > 32 else (n,0,0))
        for name, a in arrays.items():
            expected = wp.bool if name in ('done','search_unchanged') else wp.float32
            if a.device != s.device or not a.is_contiguous or a.dtype != expected:
                raise ValueError('Direction array device/layout/dtype differs: ' + name)
            if name in shapes and tuple(a.shape) != shapes[name]:
                raise ValueError('Direction shape differs: ' + name)
            if name in ('ma','smooth','force') and (a.ndim != 2 or a.shape[0] != n or a.shape[1] < nv):
                raise ValueError('Direction input shape differs: ' + name)
        self.arrays, self.stream = arrays, s
        self.arguments = DirectionInputs(worlds=n, nv=nv, pad=pad,
            **{name+'_stride':arrays[name].shape[1] for name in ('ma','smooth','force')},
            **{name:a.ptr for name,a in arrays.items()})
        self.launch = plan.library.flash_mjwarp_direction_launch
        self.launch.argtypes = [ct.POINTER(DirectionInputs),ct.c_int,ct.c_void_p]
        self.launch.restype = ct.c_int

    def enqueue(self, stage=0):
        error = self.launch(ct.byref(self.arguments),stage,self.stream.cuda_stream)
        if error:
            raise RuntimeError('Handwritten direction launch failed: CUDA error ' + str(error))
