"""Load-only force assembly/projection, borrowing current resident inputs."""
import ctypes as ct
import hashlib
from pathlib import Path


class SmoothForceInputs(ct.Structure):
    _fields_=[(k,ct.c_int) for k in ('worlds','nv','bodies','vector_stride')]
    _fields_ += [(k,ct.c_void_p) for k in ('body_root','dof_body','subtree_size','passive','bias',
        'actuator','applied','body_com','subtree_com','cdof','external','smooth')]


def source_id():
    from libero.gpu.host.paths import GPU_ROOT
    names=('cuda/dynamics/mjwarp_smooth_force.cu','cuda/dynamics/mjwarp_smooth_force.cuh','cmake/MjwarpSmoothForce.cmake')
    return hashlib.sha256(':'.join(hashlib.sha256((GPU_ROOT/n).read_bytes()).hexdigest() for n in names).encode()).hexdigest()


class SmoothForce:
    """No numerical caching, allocation, download or block synchronization per call."""
    def __init__(self,model,data,library,stream):
        import numpy as np
        from libero.gpu.host.runtime.cuda_api import array_api
        wp = array_api(stream)
        from libero.gpu.host.build.compilation import MissingCompiledProduct
        self.model,self.data,self.stream=model,data,stream
        self.path=Path(library).resolve()
        if not self.path.is_file():raise MissingCompiledProduct('Build flash_mjwarp_smooth_force_fp32 offline')
        self.library=ct.CDLL(str(self.path))
        self.library.flash_mjwarp_smooth_force_arguments_bytes.restype=ct.c_size_t
        if self.library.flash_mjwarp_smooth_force_arguments_bytes()!=ct.sizeof(SmoothForceInputs):
            raise ValueError('Smooth force C/Python ABI differs')
        if not stream.device.is_cuda or self.library.flash_mjwarp_smooth_force_arch()!=stream.device.arch:
            raise MissingCompiledProduct('Smooth force binary architecture differs')
        n,nv,nb=data.nworld,model.nv,model.nbody
        arrays={}
        def bind(key,array,dtype,shape):
            if (array.device!=stream.device or array.dtype!=dtype or not array.is_contiguous or
                    tuple(array.shape)!=shape):raise ValueError('Smooth force array layout differs: '+key)
            arrays[key]=array
        for key,array,shape in (('body_root',model.body_rootid,(nb,)),('dof_body',model.dof_bodyid,(nv,)),
                                ('subtree_size',model.body_subtreenum,(nb,))):bind(key,array,wp.int32,shape)
        stride=data.qfrc_smooth.shape[1]
        if n<1 or nv<1 or nb<1 or stride<nv:raise ValueError('Smooth force dimensions differ')
        for key,field in (('passive','qfrc_passive'),('bias','qfrc_bias'),('actuator','qfrc_actuator'),
                          ('applied','qfrc_applied'),('smooth','qfrc_smooth')):
            bind(key,getattr(data,field),wp.float32,(n,stride))
        for key,field,dtype,shape in (('body_com','xipos',wp.vec3,(n,nb)),
                ('subtree_com','subtree_com',wp.vec3,(n,nb)),('cdof','cdof',wp.spatial_vector,(n,nv)),
                ('external','xfrc_applied',wp.spatial_vector,(n,nb))):bind(key,getattr(data,field),dtype,shape)
        # Static topology admission outside capture; physical arrays stay resident.
        body=arrays['dof_body'].numpy();root=arrays['body_root'].numpy();size=arrays['subtree_size'].numpy()
        if (np.any(body<0) or np.any(body>=nb) or np.any(root<0) or np.any(root>=nb) or
                np.any(size<1) or np.any(np.arange(nb)+size>nb)):
            raise ValueError('Smooth force topology exceeds body storage')
        # Inputs and output have distinct ownership in upstream Data. Partial
        # overlaps would invalidate fused loads; reject them before capture.
        out=arrays['smooth'];out_end=out.ptr+out.capacity
        for key,a in arrays.items():
            if key!='smooth' and max(out.ptr,a.ptr)<min(out_end,a.ptr+a.capacity):
                raise ValueError('Smooth force output overlaps input: '+key)
        self.arrays=arrays
        self.arguments=SmoothForceInputs(worlds=n,nv=nv,bodies=nb,vector_stride=stride,
                                        **{k:a.ptr for k,a in arrays.items()})
        self.launch=self.library.flash_mjwarp_smooth_force_launch
        self.launch.argtypes=[ct.POINTER(SmoothForceInputs),ct.c_int,ct.c_void_p];self.launch.restype=ct.c_int

    @property
    def identity(self):
        return dict(force_source_id=source_id(),force_library_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
                    force_values='borrow-current-data-no-cache')

    def enqueue(self,*,split=False):
        for mode in ((0,1) if split else (2,)):
            error=self.launch(ct.byref(self.arguments),mode,self.stream.cuda_stream)
            if error:raise RuntimeError('Smooth force launch failed: CUDA error '+str(error))
