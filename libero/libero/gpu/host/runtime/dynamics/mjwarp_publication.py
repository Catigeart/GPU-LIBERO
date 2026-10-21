"""Borrowed publication state; no numerical scratch or per-call allocations."""
import ctypes as ct
import hashlib
from .mjwarp_motion import bind_array,load_leaf,reject_output_overlap,source_id


class PublicationInputs(ct.Structure):
    _fields_=[(k,ct.c_int) for k in ('worlds','timestep_rows','sparse','contacts','rows','nonzeros','warnings')]+[(k,ct.c_void_p) for k in ('timestep','count','rownnz','rowadr','contact_count','candidate_count','time','overflow')]


class PublicationState:
    def __init__(self,m,d,library,stream):
        from libero.gpu.host.runtime.cuda_api import array_api
        wp = array_api(stream)
        from libero.gpu.host.runtime.cuda_api import type_api
        types = type_api(m)
        if tuple(map(int,(types.OverflowType.NEFC,types.OverflowType.NJMAX_NNZ,types.OverflowType.BROADPHASE,types.OverflowType.NARROWPHASE)))!=(1,2,4,8):raise ValueError('Pinned publication flags differ')
        n,rows=d.nworld,d.njmax
        if n<1 or min(rows,d.naconmax,d.njmax_nnz)<0 or max(n*max(1,rows),m.opt.timestep.size)>2**31-256:raise ValueError('Publication index dimensions exceed storage')
        if m.opt.timestep.ndim!=1 or m.opt.timestep.size<1:raise ValueError('Publication timestep broadcast differs')
        self.arrays=a={};self.stream=stream
        for key,value,dtype,shape in (('timestep',m.opt.timestep,wp.float32,(m.opt.timestep.size,)),('count',d.nefc,wp.int32,(n,)),('rownnz',d.efc.J_rownnz,wp.int32,(n,rows) if m.is_sparse else d.efc.J_rownnz.shape),('rowadr',d.efc.J_rowadr,wp.int32,(n,rows) if m.is_sparse else d.efc.J_rowadr.shape),('contact_count',d.nacon,wp.int32,(1,)),('candidate_count',d.ncollision,wp.int32,(1,)),('time',d.time,wp.float32,(n,)),('overflow',d.overflow,wp.int32,(n,))):bind_array(a,key,value,dtype,shape,stream)
        reject_output_overlap(a,('time','overflow'))
        self.path,self.library=load_leaf('publication',PublicationInputs,library,stream)
        self.arguments=PublicationInputs(worlds=n,timestep_rows=m.opt.timestep.size,sparse=int(m.is_sparse),contacts=d.naconmax,rows=rows,nonzeros=d.njmax_nnz,warnings=int(m.opt.warn_overflow),**{k:a[k].ptr for k,_ in PublicationInputs._fields_[7:]})
        self.launch=self.library.flash_mjwarp_publication_launch;self.launch.argtypes=[ct.POINTER(PublicationInputs),ct.c_void_p];self.launch.restype=ct.c_int
    @property
    def identity(self):return dict(source_id=source_id('publication'),library_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),numerical_workspace_bytes=0,shared_bytes=0)
    def enqueue(self):
        error=self.launch(ct.byref(self.arguments),self.stream.cuda_stream)
        if error:raise RuntimeError('Publication launch failed: CUDA error '+str(error))
