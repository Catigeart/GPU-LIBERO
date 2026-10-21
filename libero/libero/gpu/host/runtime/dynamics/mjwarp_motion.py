"""Cold topology admission shared by com_vel, bias and postconstraint RNE."""
import ctypes as ct
import hashlib
from pathlib import Path


def source_id(component):
    from libero.gpu.host.paths import GPU_ROOT
    recipe = {'velocity':'Velocity','bias':'Bias','postconstraint':'Postconstraint','passive':'Passive','actuation':'Actuation','euler':'Euler','transmission':'Transmission','kinematics':'Kinematics','com_pos':'ComPos','camlight':'CamLight','fluid':'Fluid','external_wrench':'ExternalWrench','constraint_build':'ConstraintBuild','broadphase':'Broadphase','plane':'Plane','box':'Box','convex':'Convex','queries':'Queries','publication':'Publication'}[component]
    folder='collision' if component in ('broadphase','plane','box','convex') else 'dynamics'
    names = (f'cuda/{folder}/mjwarp_{component}.cu', f'cuda/{folder}/mjwarp_{component}.cuh')
    if component=='kinematics':names+=('cuda/dynamics/mjwarp_camlight.cuh',)
    if component not in ('passive','actuation','euler','transmission','kinematics','com_pos','camlight','fluid','external_wrench','constraint_build','broadphase','plane','box','convex','publication'):names+=('cuda/dynamics/mjwarp_spatial.cuh',)
    if component=='convex':names+=('cuda/collision/mjwarp_convex_geometry.cuh','cuda/collision/mjwarp_convex_simplex.cuh','cuda/collision/mjwarp_convex_gjk.cuh','cuda/collision/mjwarp_convex_epa.cuh','cuda/collision/mjwarp_box_manifold.cuh','cuda/collision/mjwarp_convex_collision.cuh')
    if component in ('plane','box','convex'):names+=('cuda/collision/mjwarp_contact.cuh',)
    names+=(f'cmake/Mjwarp{recipe}.cmake',)
    return hashlib.sha256(':'.join(hashlib.sha256((GPU_ROOT/n).read_bytes()).hexdigest() for n in names).encode()).hexdigest()


def load_leaf(component, arguments, library, stream, *, entry=None):
    from libero.gpu.host.build.compilation import MissingCompiledProduct
    path = Path(library).resolve()
    if not path.is_file():
        raise MissingCompiledProduct(f'Build flash_mjwarp_{component}_fp32 offline')
    lib = ct.CDLL(str(path)); prefix = f'flash_mjwarp_{component}_'
    identity, arch = (getattr(lib, prefix+n) for n in ('source_id', 'arch'))
    identity.restype = ct.c_char_p
    if identity().decode() != source_id(component):
        raise MissingCompiledProduct(f'{component} binary is stale; rebuild offline')
    symbol = 'arguments_bytes' if entry is None else entry+'_arguments_bytes'
    try:
        size = getattr(lib, prefix+symbol)
    except AttributeError:
        raise MissingCompiledProduct(f'{component} binary lacks {entry} entry; rebuild offline') from None
    size.restype = ct.c_size_t
    if size() != ct.sizeof(arguments):
        raise ValueError(f'{component} C/Python ABI differs')
    if not stream.device.is_cuda or arch() != stream.device.arch:
        raise MissingCompiledProduct(f'{component} binary architecture differs')
    return path, lib


def validate_motion(parent, jnum, jadr, dnum, dadr, joint, joint_dof, dof_body, branches, starts, layers):
    """Check complete leaf paths; return the unique publisher for each body."""
    import numpy as np
    nb, nv, nj = len(parent), len(dof_body), len(joint)
    if nb < 2 or nv < 1 or parent[0] != 0 or jnum[0] or dnum[0]:
        raise ValueError('Motion topology dimensions/world differ')
    if np.any(parent[1:] < 0) or np.any(parent[1:] >= np.arange(1, nb)):
        raise ValueError('Motion body parent order differs')
    # Checked against pinned JointType below in MotionTopology, not CPU enums.
    widths = (6, 3, 1, 1)
    if np.any(joint < 0) or np.any(joint > 3) or np.any(jnum < 0) or np.any(dnum < 0):
        raise ValueError('Motion joint type/count differs')
    jc, dc = np.zeros(nj, int), np.zeros(nv, int)
    for b in range(1, nb):
        count, first, dof = int(jnum[b]), int(jadr[b]), int(dadr[b])
        if not count:
            if dnum[b]:
                raise ValueError('Motion fixed body has DOFs')
            continue
        if first < 0 or first+count > nj or dof < 0:
            raise ValueError('Motion joint address exceeds storage')
        expected = dof
        for j in range(first, first+count):
            width = widths[int(joint[j])]
            if joint_dof[j] != expected or expected+width > nv:
                raise ValueError('Motion joint DOF span differs')
            jc[j] += 1; dc[expected:expected+width] += 1
            if not np.all(dof_body[expected:expected+width] == b):
                raise ValueError('Motion DOF body differs')
            expected += width
        if expected-dof != dnum[b]:
            raise ValueError('Motion body DOF count differs')
    if not np.all(jc == 1) or not np.all(dc == 1):
        raise ValueError('Motion joints/DOFs overlap or are omitted')
    if len(starts) < 2 or starts[0] != 0 or starts[-1] != len(branches) or np.any(np.diff(starts) < 1):
        raise ValueError('Motion branch offsets differ')
    child_counts = np.bincount(parent[1:], minlength=nb)
    leaves = np.flatnonzero(child_counts[1:] == 0)+1
    actual = []; owner = np.full(nb, -1, np.int32)
    for branch in range(len(starts)-1):
        path = branches[starts[branch]:starts[branch+1]]
        if np.any(path < 1) or np.any(path >= nb):
            raise ValueError('Motion branch body exceeds storage')
        chain = []; b = int(path[-1])
        while b:
            chain.append(b); b = int(parent[b])
        if not np.array_equal(path, chain[::-1]):
            raise ValueError('Motion branch is not a complete world-to-leaf path')
        actual.append(int(path[-1]))
        for b in path:
            if owner[b] < 0:
                owner[b] = branch
    if sorted(actual) != list(leaves) or np.any(owner[1:] < 0):
        raise ValueError('Motion leaf paths overlap or omit leaves')
    depth = np.zeros(nb, int); coverage = np.zeros(nb, int)
    for b in range(1, nb):
        depth[b] = depth[parent[b]]+1
    if len(layers) != int(depth.max())+1:
        raise ValueError('Motion depth layer count differs')
    for level, bodies in enumerate(layers):
        if np.any(bodies < 0) or np.any(bodies >= nb) or np.any(depth[bodies] != level):
            raise ValueError('Motion depth dependency differs')
        np.add.at(coverage, bodies, 1)
    if not np.all(coverage == 1):
        raise ValueError('Motion depth layers overlap or omit bodies')
    return owner


def bind_array(arrays, key, a, dtype, shape, stream):
    if a.device != stream.device or a.dtype != dtype or not a.is_contiguous or tuple(a.shape) != shape:
        raise ValueError('Motion array layout differs: ' + key)
    arrays[key] = a
    return a


def reject_output_overlap(arrays, outputs):
    for key in outputs:
        a = arrays[key]
        for name, b in arrays.items():
            if name != key and a.capacity and b.capacity and max(a.ptr, b.ptr) < min(a.ptr+a.capacity, b.ptr+b.capacity):
                raise ValueError('Motion output overlaps another array: ' + key + '/' + name)


class ResidentTree:
    """Owned immutable indices from a caller-admitted body tree; no physics cache."""
    def __init__(self,model,data,stream,parent,layers):
        import numpy as np
        from libero.gpu.host.runtime.cuda_api import array_api
        wp=array_api(stream)
        self.model,self.data,self.stream=model,data,stream
        self.host_parent=parent
        self.levels=len(layers);self.max_width=max(len(a) for a in layers)
        values=dict(bodies=np.concatenate(layers),starts=np.r_[0,np.cumsum([len(a) for a in layers])],
            child_start=np.r_[0,np.cumsum(np.bincount(parent[1:],minlength=len(parent)))],
            child_id=np.argsort(parent[1:],kind='stable')+1)
        for key,value in values.items():setattr(self,key,wp.array(np.asarray(value,np.int32),dtype=wp.int32,device=stream.device))

    @property
    def nbytes(self):
        return (sum(getattr(self,k).capacity for k in ('bodies','starts','child_start','child_id'))
                + (self.interval_size.capacity if getattr(self,'interval_size',None) is not None else 0))

    def prepare_intervals(self):
        """Bound direct subtree work; non-preorder or large trees stay staged."""
        if hasattr(self,'interval_size'):return self.interval_size
        import numpy as np
        from libero.gpu.host.runtime.cuda_api import array_api
        parent=self.host_parent;nb=len(parent);size=np.ones(nb,np.int64)
        for b in range(nb-1,0,-1):size[parent[b]]+=size[b]
        # Nested child intervals plus the exact descendant counts prove that
        # each subtree occupies [body, body+size), rather than assuming preorder.
        preorder=np.all(np.arange(1,nb)+size[1:]<=parent[1:]+size[parent[1:]])
        self.interval_work=int(size.sum())
        self.interval_size=None
        if nb<=64 and self.interval_work<=1024 and preorder:
            wp=array_api(self.stream)
            self.interval_size=wp.array(size.astype(np.int32),dtype=wp.int32,device=self.stream.device)
        return self.interval_size


class MotionTopology:
    def __init__(self, model, data, stream):
        from libero.gpu.host.runtime.cuda_api import array_api
        wp = array_api(stream)
        from libero.gpu.host.runtime.cuda_api import type_api
        types = type_api(model)
        if tuple(int(x) for x in (types.JointType.FREE, types.JointType.BALL, types.JointType.SLIDE, types.JointType.HINGE)) != (0, 1, 2, 3):
            raise ValueError('Pinned JointType representation differs')
        n, nb, nv, nj, branches = data.nworld, model.nbody, model.nv, model.njnt, model.nbranch
        if min(n, nv, branches) < 1 or nb < 2 or max(n*nv*6, n*nb*10, n*branches) > 2**31-256:
            raise ValueError('Motion dimensions exceed index storage')
        self.model, self.data, self.stream = model, data, stream
        self.arrays = arrays = {}; values = {}
        fields = (('body_parent', 'body_parentid', nb), ('body_jntnum', 'body_jntnum', nb),
            ('body_jntadr', 'body_jntadr', nb), ('body_dofnum', 'body_dofnum', nb),
            ('body_dofadr', 'body_dofadr', nb), ('jnt_type', 'jnt_type', nj),
            ('joint_dof', 'jnt_dofadr', nj), ('dof_body', 'dof_bodyid', nv),
            ('branch_bodies', 'body_branches', model.body_branches.size),
            ('branch_start', 'body_branch_start', branches+1))
        for key, field, width in fields:
            values[key] = bind_array(arrays, key, getattr(model, field), wp.int32, (width,), stream).numpy()
        self.layers = tuple(model.body_tree)
        levels = [bind_array(arrays, 'layer'+str(i), a, wp.int32, (a.size,), stream).numpy() for i,a in enumerate(self.layers)]
        owner = validate_motion(*(values[k] for k in ('body_parent','body_jntnum','body_jntadr','body_dofnum',
            'body_dofadr','jnt_type','joint_dof','dof_body','branch_bodies','branch_start')), levels)
        arrays['body_owner'] = wp.array(owner, dtype=int, device=stream.device)
        self.host_tree=values['body_parent'],levels
        self.duplicate_body_visits = int(model.body_branches.size-(nb-1))

    def prepare_layer_tree(self):
        if not hasattr(self,'resident_tree'):
            self.resident_tree=ResidentTree(self.model,self.data,self.stream,*self.host_tree)
        return self.resident_tree

    @property
    def identity(self):
        return dict(branches=self.model.nbranch, layers=[a.size for a in self.layers],
            duplicate_body_visits=self.duplicate_body_visits, owner_policy='first-full-leaf-path',
            topology_bytes=self.arrays['body_owner'].capacity+(self.resident_tree.nbytes if hasattr(self,'resident_tree') else 0), numerical_workspace_bytes=0)
