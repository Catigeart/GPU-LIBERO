"""Load-only Euler leaves; borrow current inputs and one private factor workspace."""
import ctypes as ct
import hashlib
from .mjwarp_motion import source_id,load_leaf,bind_array,reject_output_overlap
from .mjwarp_passive import validate_joints


def caller_unsupported_reason(model,data):
    from libero.gpu.host.runtime.cuda_api import type_api
    types = type_api(model)
    if model.opt.integrator!=types.IntegratorType.EULER:return 'Euler numerical caller requires Euler'
    if model.opt.enableflags & types.EnableBit.SLEEP:return 'active-DOF compaction'
    if model.na:return 'stateful activation retained upstream'
    if min(model.nq,model.nv,model.njnt,model.nC,data.nworld)<1:return 'empty Euler domain'
    return None


class EulerInputs(ct.Structure):
    _fields_=[(k,ct.c_int) for k in ('worlds','nv','nq','joints','matrix_width','timestep_rows','damping_rows','dampingpoly_rows')]
    _fields_ += [(k,ct.c_void_p) for k in ('joint_type','qposadr','dofadr','rowadr','rownnz',
        'timestep','damping','dampingpoly','acceleration','derivative','matrix','qvel','qpos')]


class EulerNumerics:
    def __init__(self,model,data,library,stream,*,workspace=None):
        import numpy as np
        from libero.gpu.host.runtime.cuda_api import array_api
        wp = array_api(stream)
        from libero.gpu.host.runtime.cuda_api import type_api
        types = type_api(model)
        reason=caller_unsupported_reason(model,data)
        if reason:raise ValueError('Unsupported CUDA Euler: '+reason)
        n,nv,nq,nj,nc=data.nworld,model.nv,model.nq,model.njnt,model.nC
        if max(n*nq,n*nv,n*nc,n*nj)>2**31-256:raise ValueError('Euler dimensions exceed index storage')
        if tuple(int(x) for x in (types.JointType.FREE,types.JointType.BALL,types.JointType.SLIDE,types.JointType.HINGE))!=(0,1,2,3):
            raise ValueError('Euler pinned joint types differ')
        self.model,self.data,self.stream=model,data,stream
        self.path,self.library=load_leaf('euler',EulerInputs,library,stream)
        self.arrays=a={};metadata={}
        for key,field,width in (('joint_type','jnt_type',nj),('qposadr','jnt_qposadr',nj),('dofadr','jnt_dofadr',nj),
                ('dof_joint','dof_jntid',nv),('rowadr','M_rowadr',nv),('rownnz','M_rownnz',nv)):
            metadata[key]=bind_array(a,key,getattr(model,field),wp.int32,(width,),stream).numpy()
        validate_joints(nq,nv,*(metadata[k] for k in ('joint_type','qposadr','dofadr','dof_joint')))
        rows,nnz=metadata['rowadr'],metadata['rownnz'];columns=model.M_colind.numpy()
        if columns.shape!=(nc,) or np.any(rows<0) or np.any(nnz<1) or np.any(rows+nnz>nc):
            raise ValueError('Euler mass row exceeds storage')
        diagonal=rows+nnz-1
        if len(np.unique(diagonal))!=nv or not np.array_equal(columns[diagonal],np.arange(nv)):
            raise ValueError('Euler mass row diagonal ownership differs')
        broadcasts={}
        for key,field,dtype,width in (('timestep','timestep',wp.float32,None),('damping','dof_damping',wp.float32,nv),
                ('dampingpoly','dof_dampingpoly',wp.vec2,nv)):
            value=getattr(model.opt if key=='timestep' else model,field)
            shape=(value.shape[0],) if width is None else (value.shape[0],width)
            if value.shape[0]<1 or value.size*2>2**31-256:raise ValueError('Euler broadcast storage differs: '+key)
            bind_array(a,key,value,dtype,shape,stream);broadcasts[key+'_rows']=value.shape[0]
        for key,width in (('qpos',nq),('qvel',nv)):
            bind_array(a,key,getattr(data,key),wp.float32,(n,width),stream)
        bind_array(a,'acceleration',data.qacc if workspace is None else workspace.acceleration,wp.float32,(n,nv),stream)
        if workspace is not None:
            bind_array(a,'derivative',workspace.damping,wp.float32,(n,nv),stream)
            bind_array(a,'matrix',workspace.matrix,wp.float32,(n,nc),stream)
        # In-place state updates are intentional. Their outputs must still be
        # disjoint from acceleration, model metadata and the private workspace.
        reject_output_overlap(a,('qpos','qvel')+(('derivative','matrix') if workspace is not None else ()))
        self.arguments=EulerInputs(worlds=n,nv=nv,nq=nq,joints=nj,matrix_width=nc,**broadcasts,
            **{k:a[k].ptr if k in a else 0 for k,_ in EulerInputs._fields_[8:]})
        self.has_damping_workspace=workspace is not None
        self.preparation_eligible=bool(np.array_equal(rows,np.r_[0,np.cumsum(nnz)[:-1]]) and int(nnz.sum())==nc)
        self.source_matrix=None
        self.launch=self.library.flash_mjwarp_euler_launch
        self.launch.argtypes=[ct.POINTER(EulerInputs),ct.c_int,ct.c_void_p];self.launch.restype=ct.c_int

    def bind_preparation(self,source):
        from libero.gpu.host.runtime.cuda_api import array_api
        wp=array_api(self.stream)
        if not self.has_damping_workspace:raise ValueError('Euler damping workspace absent')
        bind_array(self.arrays,'source_matrix',source,wp.float32,
                   (self.data.nworld,self.model.nC),self.stream)
        reject_output_overlap(self.arrays,('derivative','matrix','qvel','qpos'))
        self.source_matrix=source
        self.prepare_launch=self.library.flash_mjwarp_euler_prepare
        self.prepare_launch.argtypes=[ct.POINTER(EulerInputs),ct.c_void_p,ct.c_void_p]
        self.prepare_launch.restype=ct.c_int
        return self

    def prepare_damping(self):
        if self.source_matrix is None:raise ValueError('Euler preparation source is unbound')
        if not self.preparation_eligible:
            from libero.gpu.host.runtime.cuda_api import array_api
            self.enqueue(0);array_api(self.stream).copy(self.arrays['matrix'],self.source_matrix)
            self.enqueue(1);return
        error=self.prepare_launch(ct.byref(self.arguments),self.source_matrix.ptr,self.stream.cuda_stream)
        if error:raise RuntimeError('Euler preparation failed: CUDA error '+str(error))

    @property
    def identity(self):
        return dict(source_id=source_id('euler'),library_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            values='borrow-current-input-no-cache',additional_numerical_workspace_bytes=0,topology_bytes=0,
            domain='non-sleeping stateless Euler; FREE/BALL/SLIDE/HINGE')

    def enqueue(self,phase):
        if phase in (0,1) and not self.has_damping_workspace:raise ValueError('Euler damping workspace absent')
        error=self.launch(ct.byref(self.arguments),phase,self.stream.cuda_stream)
        if error:raise RuntimeError('Euler launch failed: CUDA error '+str(error))
