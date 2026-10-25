"""Load-only CRB / mass assembly on borrowed resident MJWarp arrays."""
import ctypes as ct
import hashlib
from .mjwarp_motion import ResidentTree
from pathlib import Path


class MassMatrixInputs(ct.Structure):
    _fields_ = [(k, ct.c_int) for k in ('worlds', 'nv', 'bodies', 'matrix_width', 'armature_rows')]
    _fields_ += [(k, ct.c_void_p) for k in ('body_parent', 'dof_body', 'dof_parent', 'matrix_row',
                                          'matrix_nnz', 'armature', 'cdof', 'crb', 'matrix')]


def source_id():
    from libero.gpu.host.paths import GPU_ROOT
    names = ('cuda/dynamics/mjwarp_mass_matrix.cu', 'cuda/dynamics/mjwarp_mass_matrix.cuh',
             'cmake/MjwarpMassMatrix.cmake')
    hashes = ':'.join(hashlib.sha256((GPU_ROOT / n).read_bytes()).hexdigest() for n in names)
    return hashlib.sha256(hashes.encode()).hexdigest()


def validate_topology(parent, dof_body, dof_parent, row, nnz, column, layers):
    """Prove layer dependencies and exact CSR row ownership before capture.

    No physical values are inspected/cached. Invalid metadata must never make a
    device ancestor traversal walk out of its row or race a different depth.
    """
    import numpy as np
    nb, nv, nc = len(parent), len(dof_body), len(column)
    if nb < 1 or nv < 1 or parent[0] != 0 or len(dof_parent) != nv:
        raise ValueError('Mass matrix topology dimensions differ')
    if np.any(parent[1:] < 0) or np.any(parent[1:] >= np.arange(1, nb)):
        raise ValueError('Mass matrix body parents must precede children')
    if np.any(dof_body < 0) or np.any(dof_body >= nb):
        raise ValueError('Mass matrix DOF body exceeds storage')
    if np.any(dof_parent < -1) or np.any(dof_parent >= np.arange(nv)):
        raise ValueError('Mass matrix DOF parents must precede children')
    if (len(row) != nv or len(nnz) != nv or np.any(nnz < 1) or
            not np.array_equal(row, np.r_[0, np.cumsum(nnz)[:-1]]) or int(nnz.sum()) != nc):
        raise ValueError('Mass matrix CSR row storage differs')
    depth = np.zeros(nb, np.int32)
    for b in range(1, nb):
        depth[b] = depth[parent[b]] + 1
    coverage = np.zeros(nb, np.int32)
    if len(layers) != int(depth.max()) + 1:
        raise ValueError('Mass matrix tree layer count differs')
    for level, bodies in enumerate(layers):
        if np.any(bodies < 0) or np.any(bodies >= nb) or np.any(depth[bodies] != level):
            raise ValueError('Mass matrix tree layer dependency differs')
        np.add.at(coverage, bodies, 1)
    if not np.all(coverage == 1):
        raise ValueError('Mass matrix tree layers overlap or omit bodies')
    for i in range(nv):
        ancestors = []; j = i
        while j >= 0:
            ancestors.append(j); j = int(dof_parent[j])
        if not np.array_equal(column[row[i]:row[i] + nnz[i]], ancestors[::-1]):
            raise ValueError('Mass matrix CSR row differs from DOF ancestor chain')


class MassMatrix:
    """Copies and layer launches follow upstream; no numerical cache/workspace."""
    def __init__(self, model, data, library, stream):
        from libero.gpu.host.runtime.cuda_api import array_api
        wp = array_api(stream)
        from libero.gpu.host.runtime.cuda_api import type_api
        types = type_api(model)
        from libero.gpu.host.build.compilation import MissingCompiledProduct
        self.model, self.data, self.stream = model, data, stream
        self.path = Path(library).resolve()
        if not self.path.is_file():
            raise MissingCompiledProduct('Build flash_mjwarp_mass_matrix_fp32 offline')
        self.library = ct.CDLL(str(self.path))
        self.library.flash_mjwarp_mass_matrix_arguments_bytes.restype = ct.c_size_t
        if self.library.flash_mjwarp_mass_matrix_arguments_bytes() != ct.sizeof(MassMatrixInputs):
            raise ValueError('Mass matrix C/Python ABI differs')
        if not stream.device.is_cuda or self.library.flash_mjwarp_mass_matrix_arch() != stream.device.arch:
            raise MissingCompiledProduct('Mass matrix architecture differs')
        n, nv, nb, nc = data.nworld, model.nv, model.nbody, model.nC
        if min(n, nv, nb, nc) < 1 or max(n*nv*6, n*nb*10, n*nc) > 2**31 - 256:
            raise ValueError('Mass matrix dimensions exceed launch/index storage')
        arrays = {}

        def bind(key, array, dtype, shape):
            if (array.device != stream.device or array.dtype != dtype or not array.is_contiguous or
                    tuple(array.shape) != shape):
                raise ValueError('Mass matrix array layout differs: ' + key)
            arrays[key] = array
            return array

        static = {}
        for key, field, shape in (('body_parent', 'body_parentid', (nb,)), ('dof_body', 'dof_bodyid', (nv,)),
                ('dof_parent', 'dof_parentid', (nv,)), ('matrix_row', 'M_rowadr', (nv,)),
                ('matrix_nnz', 'M_rownnz', (nv,)), ('matrix_column', 'M_colind', (nc,))):
            static[key] = bind(key, getattr(model, field), wp.int32, shape).numpy()
        self.layers = tuple(model.body_tree)
        layers = [bind('layer' + str(i), a, wp.int32, (a.size,)).numpy() for i, a in enumerate(self.layers)]
        validate_topology(static['body_parent'], static['dof_body'], static['dof_parent'],
                          static['matrix_row'], static['matrix_nnz'], static['matrix_column'], layers)
        self.host_tree=static['body_parent'],layers
        armature = model.dof_armature
        if armature.ndim != 2 or armature.shape[0] < 1 or armature.shape[0]*nv > 2**31-1:
            raise ValueError('Mass matrix armature broadcast shape differs')
        bind('armature', armature, wp.float32, (armature.shape[0], nv))
        for key, field, dtype, shape in (('cinert', 'cinert', types.vec10, (n, nb)),
                ('cdof', 'cdof', wp.spatial_vector, (n, nv)), ('crb', 'crb', types.vec10, (n, nb)),
                ('matrix', 'M', wp.float32, (n, nc))):
            bind(key, getattr(data, field), dtype, shape)
        for key in ('crb', 'matrix'):
            a = arrays[key]
            for other, b in arrays.items():
                if other != key and a.capacity and b.capacity and max(a.ptr, b.ptr) < min(a.ptr+a.capacity, b.ptr+b.capacity):
                    raise ValueError('Mass matrix output overlaps another array: ' + key + '/' + other)
        self.arrays = arrays
        self.arguments = MassMatrixInputs(worlds=n, nv=nv, bodies=nb, matrix_width=nc,
            armature_rows=armature.shape[0], **{k: arrays[k].ptr for k, _ in MassMatrixInputs._fields_[5:]})
        self.accumulate = self.library.flash_mjwarp_mass_matrix_accumulate
        self.accumulate.argtypes = [ct.POINTER(MassMatrixInputs), ct.c_void_p, ct.c_int, ct.c_void_p]
        self.accumulate.restype = ct.c_int
        self.assemble = self.library.flash_mjwarp_mass_matrix_assemble
        self.assemble.argtypes = [ct.POINTER(MassMatrixInputs), ct.c_void_p]
        self.assemble.restype = ct.c_int

    @property
    def identity(self):
        return dict(source_id=source_id(), library_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
                    layers=[a.size for a in self.layers], values='borrow-current-input-no-cache', workspace_bytes=0,
                    additional_topology_bytes=self.tree.nbytes if hasattr(self,'tree') and self.owns_tree else 0)

    def enqueue(self):
        from libero.gpu.host.runtime.cuda_api import array_api
        wp = array_api(self.stream)
        with wp.ScopedStream(self.stream):
            wp.copy(self.arrays['crb'], self.arrays['cinert'])
            for layer in reversed(self.layers):
                if layer.size:
                    error = self.accumulate(ct.byref(self.arguments), layer.ptr, layer.size, self.stream.cuda_stream)
                    if error:
                        raise RuntimeError('Mass matrix accumulation launch failed: CUDA error ' + str(error))
            self.arrays['matrix'].zero_()
            error = self.assemble(ct.byref(self.arguments), self.stream.cuda_stream)
            if error:
                raise RuntimeError('Mass matrix assembly launch failed: CUDA error ' + str(error))

    def prepare_fusion(self,topology=None):
        """Borrow the independently admitted immutable layer/child indices."""
        if topology is not None and (topology.model is not self.model or topology.data is not self.data or topology.stream is not self.stream):
            raise ValueError('Mass fused tree owner differs')
        if hasattr(self,'tree'):return
        self.owns_tree=topology is None
        self.tree=(ResidentTree(self.model,self.data,self.stream,*self.host_tree)
                   if topology is None else topology.prepare_layer_tree())
        self.fused=self.library.flash_mjwarp_mass_matrix_fused
        self.fused.argtypes=[ct.POINTER(MassMatrixInputs),ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_void_p]
        self.fused.restype=ct.c_int

    def enqueue_fused(self):
        tree=self.tree
        error=self.fused(ct.byref(self.arguments),self.arrays['cinert'].ptr,tree.bodies.ptr,tree.starts.ptr,tree.levels,tree.child_start.ptr,tree.child_id.ptr,self.stream.cuda_stream)
        if error:raise RuntimeError('Fused mass matrix failed: CUDA error '+str(error))
