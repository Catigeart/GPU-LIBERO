"""Load-only spatial velocity/DOF derivative binding."""
import ctypes as ct
import hashlib
from .mjwarp_motion import source_id, load_leaf, bind_array, reject_output_overlap


class VelocityInputs(ct.Structure):
    _fields_ = [(k, ct.c_int) for k in ('worlds','nv','bodies','branches')]
    _fields_ += [(k, ct.c_void_p) for k in ('body_jntnum','body_jntadr','body_dofadr','jnt_type',
        'branch_bodies','branch_start','body_owner','qvel','cdof','cvel','cdof_dot')]


class ComVelocity:
    def __init__(self, model, data, library, stream, topology):
        from libero.gpu.host.runtime.cuda_api import array_api
        wp = array_api(stream)
        self.model, self.data, self.stream, self.topology = model, data, stream, topology
        self.path, self.library = load_leaf('velocity', VelocityInputs, library, stream)
        self.arrays = a = dict(topology.arrays)
        n, nv, nb = data.nworld, model.nv, model.nbody
        for key, dtype, shape in (('qvel', wp.float32, (n,nv)), ('cdof', wp.spatial_vector, (n,nv)),
                ('cvel', wp.spatial_vector, (n,nb)), ('cdof_dot', wp.spatial_vector, (n,nv))):
            bind_array(a, key, getattr(data,key), dtype, shape, stream)
        reject_output_overlap(a, ('cvel','cdof_dot'))
        self.arguments = VelocityInputs(worlds=n, nv=nv, bodies=nb, branches=model.nbranch,
            **{k:a[k].ptr for k,_ in VelocityInputs._fields_[4:]})
        self.launch = self.library.flash_mjwarp_velocity_launch
        self.launch.argtypes = [ct.POINTER(VelocityInputs), ct.c_int, ct.c_void_p]; self.launch.restype = ct.c_int

    @property
    def identity(self):
        return dict(source_id=source_id('velocity'), library_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
                    values='borrow-current-input-no-cache', **self.topology.identity)

    def enqueue(self):
        for phase in (0,1):
            error = self.launch(ct.byref(self.arguments), phase, self.stream.cuda_stream)
            if error:
                raise RuntimeError('com_vel launch failed: CUDA error '+str(error))
