"""Load-only current-point/Jacobian/control gather, with live GPU indices."""
import ctypes as ct
import hashlib
from pathlib import Path
from .bridges import Gather
from libero.gpu.host.runtime.dynamics.mjwarp_queries import JacobianInputs
from libero.gpu.host.runtime.dynamics.mjwarp_motion import bind_array,reject_output_overlap


class ControlPrepareInputs(ct.Structure):
    _fields_=[('gather',Gather),('jacobian',JacobianInputs),('points',ct.c_void_p),('policy',ct.c_void_p)]


def source_id():
    from libero.gpu.host.paths import GPU_ROOT
    names=('cuda/fp32/fused_control_prepare.cu','cuda/fp32/fused_control_prepare.cuh',
           'cuda/runtime/bridge_types.cuh','cuda/dynamics/mjwarp_queries.cuh','cmake/fp32/FP32ControlPrepare.cmake')
    return hashlib.sha256(':'.join(hashlib.sha256((GPU_ROOT/name).read_bytes()).hexdigest() for name in names).encode()).hexdigest()


class FusedControlPrepare:
    def __init__(self,controller,library):
        import numpy as np
        from libero.gpu.host.runtime.cuda_api import array_api
        from libero.gpu.host.build.compilation import MissingCompiledProduct
        c=controller;m,d,stream=c.model,c.data,c.stream;wp=array_api(stream)
        n,nv,nq,nm,ns=c.n,m.nv,m.nq,m.nC,m.nsite
        if min(n,nm,ns)<1 or min(nv,nq)<7 or not 0<=c.site<ns:
            raise ValueError('Control preparation dimensions/site differ')
        j=c.jacobian
        if j.stream is not stream or j.arguments.geometry.worlds!=n or j.arguments.geometry.dofs!=nv:
            raise ValueError('Control query owner/dimensions differ')
        for key,a in (('point',c.points),('position',c.jp),('rotation',c.jr)):
            if j.arrays[key].ptr!=a.ptr:raise ValueError('Control query bridge differs: '+key)
        if (j.arrays['com'].ptr!=d.subtree_com.ptr or j.arrays['cdof'].ptr!=d.cdof.ptr):
            raise ValueError('Control query physical owner differs')
        self.stream=stream;self.arrays=a={'query_'+k:v for k,v in j.arrays.items()}
        for key,value,t,shape in (('qi',c.qi,wp.int32,(7,)),('vi',c.vi,wp.int32,(7,)),('mi',c.mi,wp.int32,(49,)),
            ('qpos',d.qpos,wp.float32,(n,nq)),('qvel',d.qvel,wp.float32,(n,nv)),('bias',d.qfrc_bias,wp.float32,(n,nv)),
            ('mass',d.M,wp.float32,(n,nm)),('site_pos',d.site_xpos,wp.vec3,(n,ns)),('site_rot',d.site_xmat,wp.mat33,(n,ns)),
            ('policy',c.buffers['policy'],wp.int32,(n,))):
            bind_array(a,key,value,t,shape,stream)
        for key,width in (('pm',49),('pj',42),('pq',7),('pv',7),('bias',7),('pp',3),('pr',9),('pe',6)):
            bind_array(a,'out_'+key,c.buffers[key],wp.float32,(n,width),stream)
        qi,vi,mi=(a[k].numpy() for k in ('qi','vi','mi'))
        if (np.any(qi<0) or np.any(qi>=nq) or np.any(vi<0) or np.any(vi>=nv) or np.any(mi>=nm)):
            raise ValueError('Control gather indices exceed storage')
        reject_output_overlap(a,('query_point','query_position','query_rotation','policy',
            'out_pm','out_pj','out_pq','out_pv','out_bias','out_pp','out_pr','out_pe'))
        # Retain allocations and the query, without a Controller ownership cycle.
        self.owners=j,tuple(c.policy_values)
        self.arguments=ControlPrepareInputs(c._gather_arguments(),j.arguments,c.points.ptr,c.buffers['policy'].ptr)
        self.n=n;self.path=Path(library)
        if not self.path.is_file():raise MissingCompiledProduct('Build flash_fp32_fused_control_prepare offline')
        self.library=lib=ct.CDLL(str(self.path))
        if lib.flash_fp32_control_prepare_arch()!=stream.device.arch:raise ValueError('Control preparation architecture differs')
        size=lib.flash_fp32_control_prepare_arguments_bytes;size.restype=ct.c_size_t
        if size()!=ct.sizeof(ControlPrepareInputs):raise ValueError('Control preparation ABI differs')
        self.launch=lib.flash_fp32_control_prepare_launch
        self.launch.argtypes=[ct.POINTER(ControlPrepareInputs),ct.c_void_p,ct.c_void_p];self.launch.restype=ct.c_int

    def enqueue(self,policy_source=None):
        if policy_source is not None:
            from libero.gpu.host.runtime.cuda_api import array_api
            wp=array_api(self.stream)
            if (policy_source.device!=self.stream.device or policy_source.dtype!=wp.int32 or
                    not policy_source.is_contiguous or tuple(policy_source.shape)!=(self.n,)):
                raise ValueError('Control policy source layout differs')
        err=self.launch(ct.byref(self.arguments),policy_source.ptr if policy_source is not None else None,self.stream.cuda_stream)
        if err:raise RuntimeError('Control preparation failed: CUDA error '+str(err))
