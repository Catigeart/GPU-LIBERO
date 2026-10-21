"""Borrow current CoM inputs/outputs and immutable layers; no numerical workspace."""
import ctypes as ct
import hashlib
from .mjwarp_motion import source_id, load_leaf, bind_array, reject_output_overlap, ResidentTree


def caller_unsupported_reason(model, data):
    from libero.gpu.host.runtime.cuda_api import type_api
    types = type_api(model)
    if model.opt.integrator != types.IntegratorType.EULER:
        return 'com_pos caller requires Euler'
    if model.opt.enableflags & types.EnableBit.SLEEP:
        return 'active-body compaction'
    if min(data.nworld, model.nv, model.njnt) < 1 or model.nbody < 2:
        return 'empty rigid com_pos domain'
    return None


def validate_indices(nv, parent, root, kind, dofadr, jbody, layers):
    """Prove layer dependencies, root references and disjoint joint DOF writes."""
    import numpy as np
    nb, nj = len(parent), len(kind)
    if nb < 2 or nv < 1 or nj < 1 or parent[0] != 0 or len(root) != nb or len(dofadr) != nj or len(jbody) != nj:
        raise ValueError('CoM index dimensions differ')
    if np.any(parent[1:] < 0) or np.any(parent[1:] >= np.arange(1, nb)):
        raise ValueError('CoM body parents must precede children')
    depth = np.zeros(nb, int); expected_root = np.zeros(nb, int)
    for b in range(1, nb):
        depth[b] = depth[parent[b]]+1
        expected_root[b] = b if parent[b] == 0 else expected_root[parent[b]]
    if not np.array_equal(root, expected_root):
        raise ValueError('CoM root differs from the world child ancestry')
    coverage = np.zeros(nb, int)
    if len(layers) != int(depth.max())+1:
        raise ValueError('CoM tree layer count differs')
    for level, bodies in enumerate(layers):
        if np.any(bodies < 0) or np.any(bodies >= nb) or np.any(depth[bodies] != level):
            raise ValueError('CoM tree layer dependency differs')
        np.add.at(coverage, bodies, 1)
    if not np.all(coverage == 1):
        raise ValueError('CoM tree layers overlap or omit bodies')
    if np.any(kind < 0) or np.any(kind > 3) or np.any(jbody < 1) or np.any(jbody >= nb):
        raise ValueError('CoM joint type/body differs')
    dofs = np.zeros(nv, int)
    for k, at in zip(kind, dofadr):
        width = (6, 3, 1, 1)[int(k)]
        if at < 0 or at+width > nv:
            raise ValueError('CoM joint DOF span exceeds storage')
        dofs[at:at+width] += 1
    if not np.all(dofs == 1):
        raise ValueError('CoM joint DOF spans overlap or omit coordinates')


class ComPosInputs(ct.Structure):
    _fields_ = [(k, ct.c_int) for k in ('worlds','bodies','joints','nv','mass_rows','subtree_rows','inertia_rows')]
    _fields_ += [(k, ct.c_void_p) for k in ('parent','root','kind','dofadr','jbody','mass','subtreemass','inertia',
                                          'xipos','ximat','xmat','anchor','axis','subtree','cinert','cdof')]


class CenteredFrames:
    def __init__(self, model, data, library, stream):
        from libero.gpu.host.runtime.cuda_api import array_api
        wp = array_api(stream)
        from libero.gpu.host.runtime.cuda_api import type_api
        types = type_api(model)
        reason = caller_unsupported_reason(model, data)
        if reason:
            raise ValueError('Unsupported CUDA com_pos: '+reason)
        if tuple(int(x) for x in (types.JointType.FREE, types.JointType.BALL, types.JointType.SLIDE, types.JointType.HINGE)) != (0,1,2,3):
            raise ValueError('Pinned JointType representation differs')
        n, nb, nj, nv = data.nworld, model.nbody, model.njnt, model.nv
        if max(n*nb*10, n*nv*6, n*nj*3) > 2**31-256:
            raise ValueError('CoM dimensions exceed index storage')
        self.model, self.data, self.stream = model, data, stream
        self.path, self.library = load_leaf('com_pos', ComPosInputs, library, stream)
        a = self.arrays = {}; values = {}
        for key, field, width in (('parent','body_parentid',nb),('root','body_rootid',nb),('kind','jnt_type',nj),
                                  ('dofadr','jnt_dofadr',nj),('jbody','jnt_bodyid',nj)):
            values[key] = bind_array(a, key, getattr(model,field), wp.int32, (width,), stream).numpy()
        self.layers = tuple(model.body_tree)
        levels = [bind_array(a, 'layer'+str(i), layer, wp.int32, (layer.size,), stream).numpy() for i,layer in enumerate(self.layers)]
        validate_indices(nv, *(values[k] for k in ('parent','root','kind','dofadr','jbody')), levels)
        self.host_tree=values['parent'],levels
        rows = {}
        for key, field, dtype, components in (('mass','body_mass',wp.float32,1),('subtreemass','body_subtreemass',wp.float32,1),
                                             ('inertia','body_inertia',wp.vec3,3)):
            value = getattr(model,field)
            if value.ndim != 2 or value.shape[0] < 1 or value.shape[0]*nb*components > 2**31-256:
                raise ValueError('CoM broadcast storage differs: '+key)
            bind_array(a, key, value, dtype, (value.shape[0],nb), stream)
            rows[{'mass':'mass_rows','subtreemass':'subtree_rows','inertia':'inertia_rows'}[key]] = value.shape[0]
        for key, field, width, dtype in (('xipos','xipos',nb,wp.vec3),('ximat','ximat',nb,wp.mat33),('xmat','xmat',nb,wp.mat33),
                ('anchor','xanchor',nj,wp.vec3),('axis','xaxis',nj,wp.vec3),('subtree','subtree_com',nb,wp.vec3),
                ('cinert','cinert',nb,types.vec10),('cdof','cdof',nv,wp.spatial_vector)):
            bind_array(a, key, getattr(data,field), dtype, (n,width), stream)
        reject_output_overlap(a, ('subtree','cinert','cdof'))
        self.arguments = ComPosInputs(worlds=n,bodies=nb,joints=nj,nv=nv,**rows,
            **{k:a[k].ptr for k,_ in ComPosInputs._fields_[7:]})
        self.launch = self.library.flash_mjwarp_com_pos_launch
        self.launch.argtypes = [ct.POINTER(ComPosInputs),ct.c_int,ct.c_void_p,ct.c_int,ct.c_void_p]
        self.launch.restype = ct.c_int

    @property
    def identity(self):
        return dict(source_id=source_id('com_pos'),library_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            numerical_workspace_bytes=0,additional_topology_bytes=self.tree.nbytes if hasattr(self,'tree') and self.owns_tree else 0,layers=[a.size for a in self.layers],
            reduction='FP32 atomics including world, reverse depth',values='borrow-current-input-no-cache')

    def enqueue(self, phase=None):
        if phase is None:
            for p in range(5):
                self.enqueue(p)
            return
        if phase not in range(5):
            raise ValueError('CoM phase must be 0..4')
        for layer in reversed(self.layers) if phase == 1 else (None,):
            error = self.launch(ct.byref(self.arguments),phase,layer.ptr if layer is not None else None,
                                layer.size if layer is not None else 0,self.stream.cuda_stream)
            if error:
                raise RuntimeError('CoM launch failed: CUDA error '+str(error))

    def prepare_fusion(self,topology=None):
        """Share immutable layer/child indices after independent admission."""
        if hasattr(self,'tree'):return
        self.owns_tree=topology is None
        if topology is None:self.tree=ResidentTree(self.model,self.data,self.stream,*self.host_tree)
        else:
            if topology.model is not self.model or topology.data is not self.data or topology.stream is not self.stream:
                raise ValueError('CoM fused tree owner differs')
            self.tree=topology.prepare_layer_tree()
        self.fused=self.library.flash_mjwarp_com_pos_fused
        self.fused.argtypes=[ct.POINTER(ComPosInputs),ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_void_p]
        self.fused.restype=ct.c_int

    def enqueue_fused(self):
        tree=self.tree
        error=self.fused(ct.byref(self.arguments),tree.bodies.ptr,tree.starts.ptr,tree.levels,tree.child_start.ptr,tree.child_id.ptr,self.stream.cuda_stream)
        if error:raise RuntimeError('Fused CoM launch failed: CUDA error '+str(error))
