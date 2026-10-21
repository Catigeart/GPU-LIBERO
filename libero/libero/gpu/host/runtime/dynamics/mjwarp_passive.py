"""Load-only passive leaves with exact joint/DOF output ownership."""
import ctypes as ct
import hashlib
from .mjwarp_motion import source_id,load_leaf,bind_array,reject_output_overlap


def validate_joints(nq,nv,kinds,qadr,dadr,dof_joint):
    import numpy as np
    nj=len(kinds)
    if min(nq,nv,nj)<1 or any(len(a)!=nj for a in (qadr,dadr)) or len(dof_joint)!=nv:
        raise ValueError('Passive joint dimensions differ')
    if np.any(kinds<0) or np.any(kinds>3):raise ValueError('Passive joint type differs')
    qc=np.zeros(nq,int);dc=np.zeros(nv,int)
    for j,kind in enumerate(kinds):
        nqj=(7,4,1,1)[kind];nvj=(6,3,1,1)[kind];q=int(qadr[j]);d=int(dadr[j])
        if q<0 or d<0 or q+nqj>nq or d+nvj>nv:
            raise ValueError('Passive joint span exceeds storage')
        if not np.all(dof_joint[d:d+nvj]==j):raise ValueError('Passive DOF joint owner differs')
        qc[q:q+nqj]+=1;dc[d:d+nvj]+=1
    if not np.all(qc==1) or not np.all(dc==1):raise ValueError('Passive joints overlap or omit coordinates')


class PassiveInputs(ct.Structure):
    _fields_=[(k,ct.c_int) for k in ('worlds','nv','nq','joints','spring_rows','stiffness_rows',
        'stiffnesspoly_rows','damping_rows','dampingpoly_rows','disable_spring','disable_damper',
        'gravity_enabled','has_fluid','has_adhesion')]
    _fields_ += [(k,ct.c_void_p) for k in ('joint_type','qposadr','dofadr','dof_joint','actgravcomp',
        'qpos_spring','stiffness','stiffnesspoly','damping','dampingpoly','qpos','qvel',
        'gravcomp','fluid','adhesion','spring','damper','passive')]


class PassiveForces:
    def __init__(self,model,data,library,stream):
        from libero.gpu.host.runtime.cuda_api import array_api
        wp = array_api(stream)
        from libero.gpu.host.runtime.cuda_api import type_api
        types = type_api(model)
        n,nv,nq,nj=data.nworld,model.nv,model.nq,model.njnt
        if min(n,nv,nq,nj)<1 or max(n*nq,n*nv,n*nj)>2**31-256:
            raise ValueError('Passive dimensions exceed index storage')
        if tuple(int(k) for k in (types.JointType.FREE,types.JointType.BALL,types.JointType.SLIDE,types.JointType.HINGE))!=(0,1,2,3):
            raise ValueError('Passive pinned joint types differ')
        self.model,self.data,self.stream=model,data,stream
        self.path,self.library=load_leaf('passive',PassiveInputs,library,stream)
        self.arrays=a={};metadata={}
        for key,field,width in (('joint_type','jnt_type',nj),('qposadr','jnt_qposadr',nj),
                ('dofadr','jnt_dofadr',nj),('dof_joint','dof_jntid',nv),('actgravcomp','jnt_actgravcomp',nj)):
            metadata[key]=bind_array(a,key,getattr(model,field),wp.int32,(width,),stream).numpy()
        validate_joints(nq,nv,*(metadata[k] for k in ('joint_type','qposadr','dofadr','dof_joint')))
        rows={}
        for key,field,width,dtype,rowkey in (('qpos_spring','qpos_spring',nq,wp.float32,'spring_rows'),
                ('stiffness','jnt_stiffness',nj,wp.float32,'stiffness_rows'),
                ('stiffnesspoly','jnt_stiffnesspoly',nj,wp.vec2,'stiffnesspoly_rows'),
                ('damping','dof_damping',nv,wp.float32,'damping_rows'),
                ('dampingpoly','dof_dampingpoly',nv,wp.vec2,'dampingpoly_rows')):
            value=getattr(model,field)
            if value.ndim!=2 or value.shape[0]<1 or value.shape[0]*width*2>2**31-256:
                raise ValueError('Passive broadcast storage differs: '+key)
            bind_array(a,key,value,dtype,(value.shape[0],width),stream);rows[rowkey]=value.shape[0]
        for key,field,width in (('qpos','qpos',nq),('qvel','qvel',nv),('gravcomp','qfrc_gravcomp',nv),
                ('fluid','qfrc_fluid',nv),('adhesion','qfrc_adhesion',nv),('spring','qfrc_spring',nv),
                ('damper','qfrc_damper',nv),('passive','qfrc_passive',nv)):
            bind_array(a,key,getattr(data,field),wp.float32,(n,width),stream)
        reject_output_overlap(a,('spring','damper','passive'))
        self.flags=dict(disable_spring=int(bool(model.opt.disableflags & types.DisableBit.SPRING)),
            disable_damper=int(bool(model.opt.disableflags & types.DisableBit.DAMPER)),
            gravity_enabled=int(not (model.opt.disableflags & types.DisableBit.GRAVITY)),
            has_fluid=int(bool(model.has_fluid)),has_adhesion=int(bool(model.flg_adhesion)))
        self.arguments=PassiveInputs(worlds=n,nv=nv,nq=nq,joints=nj,**rows,**self.flags,
            **{k:a[k].ptr for k,_ in PassiveInputs._fields_[14:]})
        self.launch=self.library.flash_mjwarp_passive_launch
        self.launch.argtypes=[ct.POINTER(PassiveInputs),ct.c_int,ct.c_void_p];self.launch.restype=ct.c_int

    @property
    def identity(self):
        return dict(source_id=source_id('passive'),library_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            flags=self.flags,values='borrow-current-input-no-cache',numerical_workspace_bytes=0,topology_bytes=0)

    def enqueue(self,phase):
        error=self.launch(ct.byref(self.arguments),phase,self.stream.cuda_stream)
        if error:raise RuntimeError('Passive force launch failed: CUDA error '+str(error))
