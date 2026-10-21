"""Load-only binding for the independent handwritten MJWarp FP32 Hessian leaf.

HessianPlan prepares model constants before graph capture. DenseHessian binds
an existing SolverContext using metadata and pointers only, including inside a
conditional graph. Both the local replay and optional Newton dispatcher use it.
"""
import ctypes as ct
import hashlib
from pathlib import Path

import numpy as np


class HessianInputs(ct.Structure):
    _fields_ = [(name, ct.c_int) for name in (
        'worlds', 'nv', 'padded_nv', 'rows', 'mass_entries', 'contact_capacity',
        'impratio_count', 'cone_launch_groups', 'cone_contacts_per_group')]
    _fields_ += [(name, ct.c_void_p) for name in (
        'mass_upper_address', 'nefc', 'mass', 'jacobian', 'diagonal', 'state',
        'jaref', 'done', 'impratio_invsqrt', 'triangle_row', 'triangle_column',
        'contact_count', 'contact_world', 'contact_dimension', 'contact_friction',
        'contact_distance', 'contact_margin', 'contact_address', 'hessian')]


def source_id():
    from libero.gpu.host.paths import GPU_ROOT
    files = ('cuda/newton/mjwarp_hessian.cu', 'cuda/newton/mjwarp_hessian.cuh',
             'cmake/MjwarpHessian.cmake')
    hashes = ':'.join(hashlib.sha256((GPU_ROOT / name).read_bytes()).hexdigest() for name in files)
    return hashlib.sha256(hashes.encode()).hexdigest()


class HessianPlan:
    """Static preparation outside capture; owns the library and CSR lookup."""

    def __init__(self, model, data, library, stream):
        from libero.gpu.host.runtime.cuda_api import array_api
        wp = array_api(stream)
        from libero.gpu.host.runtime.cuda_api import type_api
        types = type_api(model)
        from libero.gpu.host.build.compilation import MissingCompiledProduct

        if (model.is_sparse or model.opt.cone != types.ConeType.ELLIPTIC or
                model.opt.solver != types.SolverType.NEWTON or
                model.opt.integrator == types.IntegratorType.DISCRETE or
                model.opt.enableflags & types.EnableBit.SLEEP):
            raise ValueError('Hessian leaf requires dense elliptic Newton, non-discrete, full DOFs')
        nv, pad = int(model.nv), int(model.nv_pad)
        expected_pad = ((nv + 3) // 4) * 4 if nv <= 32 else ((nv + 15) // 16) * 16
        if pad != expected_pad or pad not in (24, 28, 32, 48, 64):
            raise ValueError('unsupported/noncanonical Hessian padding')
        device = stream.device
        if not device.is_cuda:
            raise ValueError('Hessian leaf requires a CUDA stream')
        self.library_path = Path(library).resolve()
        if not self.library_path.is_file():
            raise MissingCompiledProduct('Build CMake target flash_mjwarp_hessian_fp32 offline')
        self.library = ct.CDLL(str(self.library_path))
        self.library.flash_mjwarp_hessian_source_id.restype = ct.c_char_p
        self.library.flash_mjwarp_hessian_arguments_bytes.restype = ct.c_size_t
        if self.library.flash_mjwarp_hessian_source_id().decode() != source_id():
            raise MissingCompiledProduct('Handwritten Hessian binary is stale; rebuild its CMake target offline')
        if self.library.flash_mjwarp_hessian_arguments_bytes() != ct.sizeof(HessianInputs):
            raise ValueError('Hessian C/Python ABI mismatch')
        if self.library.flash_mjwarp_hessian_arch() != device.arch:
            raise MissingCompiledProduct('Handwritten Hessian binary architecture differs')

        n, rows, capacity = int(data.nworld), int(data.njmax), int(data.naconmax)
        mr, mc = model.M_hinit_i.numpy().reshape(-1), model.M_colind.numpy().reshape(-1)
        if (mr.shape != mc.shape or not mr.size or np.any(mc < 0) or
                np.any(mc > mr) or np.any(mr >= nv)):
            raise ValueError('expected lower-triangle CSR mass indices')
        positions = mc * pad + mr
        if np.unique(positions).size != positions.size:
            raise ValueError('duplicate CSR mass indices require explicit coalescing')
        mapping = np.full((pad, pad), -1, np.int32)
        mapping.reshape(-1)[positions] = np.arange(mr.size, dtype=np.int32)
        with wp.ScopedStream(stream):
            self.mass_upper_address = wp.array(mapping, dtype=wp.int32, device=device)
        self.model, self.data, self.stream = model, data, stream
        self.mass_entries = mr.size

    def bind(self, context):
        return DenseHessian(self.model, self.data, context, self.library_path,
                            self.stream, plan=self)


class DenseHessian:
    """A context-specific launch binding; no GPU allocation or readback.

    Without a supplied plan this retains the standalone leaf API. The output
    follows upstream storage: upper M/cone, full quadratic JTDJ and padded unit
    diagonals for blocked Cholesky. Array owners live as long as this binding.
    """

    def __init__(self, model, data, context, library, stream, *, plan=None):
        from libero.gpu.host.runtime.cuda_api import array_api
        wp = array_api(stream)
        from libero.gpu.host.runtime.cuda_api import type_api
        types = type_api(model)
        self.plan = plan if plan is not None else HessianPlan(model, data, library, stream)
        if self.plan.model is not model or self.plan.data is not data or self.plan.stream is not stream:
            raise ValueError('Hessian plan belongs to another model, data or stream')
        self.library_path, self.library = self.plan.library_path, self.plan.library
        self.mass_upper_address = self.plan.mass_upper_address
        n, rows, capacity = int(data.nworld), int(data.njmax), int(data.naconmax)
        nv, pad, mass_entries = int(model.nv), int(model.nv_pad), self.plan.mass_entries
        device = stream.device
        # Keep array owners alive for captured graphs as well as direct calls.
        arrays = dict(mass_upper_address=self.mass_upper_address, nefc=data.nefc,
            mass=data.M, jacobian=data.efc.J, diagonal=data.efc.D, state=data.efc.state,
            jaref=context.Jaref, done=context.done, impratio_invsqrt=model.opt.impratio_invsqrt,
            triangle_row=model.dof_tri_row, triangle_column=model.dof_tri_col,
            contact_count=data.nacon, contact_world=data.contact.worldid,
            contact_dimension=data.contact.dim, contact_friction=data.contact.friction,
            contact_distance=data.contact.dist, contact_margin=data.contact.includemargin,
            contact_address=data.contact.efc_address, hessian=context.h)
        triangle = nv * (nv + 1) // 2
        shapes = dict(mass_upper_address=(pad, pad), nefc=(n,), mass=(n, mass_entries),
            jacobian=(n, rows, pad), diagonal=(n, rows), state=(n, rows), jaref=(n, rows),
            done=(n,), triangle_row=(triangle,), triangle_column=(triangle,),
            contact_count=(1,), contact_world=(capacity,), contact_dimension=(capacity,),
            contact_distance=(capacity,), contact_margin=(capacity,),
            contact_address=(capacity, 6), hessian=(n, pad, pad))
        integer = {'mass_upper_address', 'nefc', 'state', 'triangle_row', 'triangle_column',
                   'contact_count', 'contact_world', 'contact_dimension', 'contact_address'}
        for name, a in arrays.items():
            if a.device != device or not a.is_contiguous:
                raise ValueError('Hessian arrays must be contiguous on the supplied CUDA device: ' + name)
            if name in shapes and tuple(a.shape) != shapes[name]:
                raise ValueError('Hessian shape differs: ' + name)
            expected = wp.int32 if name in integer else wp.bool if name == 'done' else wp.float32
            if name == 'contact_friction':
                valid = ((a.dtype == types.vec5 and tuple(a.shape) == (capacity,)) or
                         (a.dtype == wp.float32 and tuple(a.shape) == (capacity, 5)))
                if not valid:
                    raise ValueError('Hessian friction layout differs')
            elif a.dtype != expected:
                raise ValueError('Hessian dtype differs: ' + name)
        if tuple(arrays['impratio_invsqrt'].shape) not in ((1,), (n,)):
            raise ValueError('impratio must broadcast or be per world')
        groups = (device.sm_count * 6 * 256 + triangle - 1) // triangle
        self.arguments = HessianInputs(worlds=n, nv=nv, padded_nv=pad, rows=rows,
            mass_entries=mass_entries, contact_capacity=capacity,
            impratio_count=arrays['impratio_invsqrt'].size, cone_launch_groups=groups,
            cone_contacts_per_group=(capacity + groups - 1) // groups,
            **{name: a.ptr for name, a in arrays.items()})
        self.arrays, self.stream = arrays, stream
        self.launch = self.library.flash_mjwarp_hessian_launch
        self.launch.argtypes = [ct.POINTER(HessianInputs), ct.c_int, ct.c_void_p]
        self.launch.restype = ct.c_int

    def enqueue(self, stage=0):
        """0 = complete assembly, 1 = mass/quadratic/padding, 2 = cone add."""
        error = self.launch(ct.byref(self.arguments), stage, self.stream.cuda_stream)
        if error:
            raise RuntimeError('Handwritten Hessian launch failed: CUDA error ' + str(error))
