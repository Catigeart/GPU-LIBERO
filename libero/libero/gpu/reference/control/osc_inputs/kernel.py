"""Given full physical fields, prepare the fixed robosuite OSC input arrays.

The velocity operation is the original binding_utils get_site_xvelp/r:
full mj_jacSite 3×nv times full qvel. Mass uses the original MuJoCo 2.3.7
engine_support.c:mj_fullM sparse ancestor layout; only the selected 7×7
block is written. No controller refresh/cache state or forward is changed.
"""
import libero.gpu.precision as scalar

import numpy as np
import warp as wp

wp.set_module_options({'enable_backward':False,'fuse_fp':True})


def _integer_scalar(value,name):
    """Check the original scalar schema before any Python integer conversion."""
    raw=np.asarray(value)
    if raw.shape!=() or raw.dtype.kind not in ('i','u'):
        raise ValueError(name+' must be an integer scalar')
    return int(raw)


def _integer_vector(value,name,length):
    """Keep the original integer width until its domain has been checked."""
    raw=np.asarray(value)
    if raw.shape!=(length,) or raw.dtype.kind not in ('i','u'):
        raise ValueError(name+' must be an integer vector of length '+str(length))
    return raw


@wp.kernel(enable_backward=False)
def _osc_inputs(qpos:wp.array(dtype=scalar.wp_real,ndim=2),
                qvel:wp.array(dtype=scalar.wp_real,ndim=2),
                site_pos:wp.array(dtype=scalar.wp_real,ndim=2),
                site_mat:wp.array(dtype=scalar.wp_real,ndim=2),
                jacp:wp.array(dtype=scalar.wp_real,ndim=3),
                jacr:wp.array(dtype=scalar.wp_real,ndim=3),
                qM:wp.array(dtype=scalar.wp_real,ndim=2),
                qfrc_bias:wp.array(dtype=scalar.wp_real,ndim=2),
                active:wp.array(dtype=wp.int32),
                qpos_index:wp.array(dtype=wp.int32),
                qvel_index:wp.array(dtype=wp.int32),
                sparse_map:wp.array(dtype=wp.int32,ndim=2),
                ee_pos:wp.array(dtype=scalar.wp_real,ndim=2),
                ee_mat:wp.array(dtype=scalar.wp_real,ndim=2),
                xvelp:wp.array(dtype=scalar.wp_real,ndim=2),
                xvelr:wp.array(dtype=scalar.wp_real,ndim=2),
                joint_pos:wp.array(dtype=scalar.wp_real,ndim=2),
                joint_vel:wp.array(dtype=scalar.wp_real,ndim=2),
                J:wp.array(dtype=scalar.wp_real,ndim=3),
                M:wp.array(dtype=scalar.wp_real,ndim=3),
                bias:wp.array(dtype=scalar.wp_real,ndim=2),
                status:wp.array(dtype=wp.int32)):
    w=wp.tid()
    status[w]=0
    for k in range(3):
        ee_pos[w,k]=scalar.wp_real(0.0)
        xvelp[w,k]=scalar.wp_real(0.0)
        xvelr[w,k]=scalar.wp_real(0.0)
    for k in range(9): ee_mat[w,k]=scalar.wp_real(0.0)
    for k in range(7):
        joint_pos[w,k]=scalar.wp_real(0.0)
        joint_vel[w,k]=scalar.wp_real(0.0)
        bias[w,k]=scalar.wp_real(0.0)
        for r in range(6): J[w,r,k]=scalar.wp_real(0.0)
        for c in range(7): M[w,k,c]=scalar.wp_real(0.0)
    if active[w]==0: return
    if active[w]!=1:
        status[w]=1
        return
    for k in range(48):
        if not wp.isfinite(qpos[w,k]):
            status[w]=2
            return
    for k in range(43):
        if not wp.isfinite(qvel[w,k]) or not wp.isfinite(qfrc_bias[w,k]):
            status[w]=2
            return
        for r in range(3):
            if not wp.isfinite(jacp[w,r,k]) or not wp.isfinite(jacr[w,r,k]):
                status[w]=2
                return
    for k in range(153):
        if not wp.isfinite(qM[w,k]):
            status[w]=2
            return
    for k in range(3):
        if not wp.isfinite(site_pos[w,k]):
            status[w]=2
            return
    for k in range(9):
        if not wp.isfinite(site_mat[w,k]):
            status[w]=2
            return
    for k in range(3): ee_pos[w,k]=site_pos[w,k]
    for k in range(9): ee_mat[w,k]=site_mat[w,k]
    for r in range(3):
        vp=scalar.wp_real(0.0)
        vr=scalar.wp_real(0.0)
        for k in range(43):
            vp=vp+jacp[w,r,k]*qvel[w,k]
            vr=vr+jacr[w,r,k]*qvel[w,k]
        xvelp[w,r]=vp
        xvelr[w,r]=vr
    for i in range(7):
        pi=qpos_index[i]
        vi=qvel_index[i]
        joint_pos[w,i]=qpos[w,pi]
        joint_vel[w,i]=qvel[w,vi]
        bias[w,i]=qfrc_bias[w,vi]
        for r in range(3):
            J[w,r,i]=jacp[w,r,vi]
            J[w,r+3,i]=jacr[w,r,vi]
        for j in range(7):
            mj=sparse_map[vi,qvel_index[j]]
            if mj>=0:
                M[w,i,j]=qM[w,mj]


class OSCInputsWorkspace:
    def __init__(self,model,capacity,device='cuda:0'):
        dimensions=tuple(_integer_scalar(getattr(model,name),name) for name in
                         ('nq','nv','nM','nsite','eef_site_id'))
        if dimensions!=(48,43,153,31,13):
            raise ValueError('unsupported fixed OSC model')
        capacity=_integer_scalar(capacity,'capacity')
        if capacity<=0: raise ValueError('capacity')
        qi=_integer_vector(model.qpos_index,'qpos_index',7)
        vi=_integer_vector(model.qvel_index,'qvel_index',7)
        qi_values=tuple(int(x) for x in qi)
        vi_values=tuple(int(x) for x in vi)
        if len(set(qi_values))!=7 or len(set(vi_values))!=7 or \
           any(x<0 or x>=48 for x in qi_values) or \
           any(x<0 or x>=43 for x in vi_values):
            raise ValueError('joint index domain')
        parent=_integer_vector(model.dof_parentid,'dof_parentid',43)
        madr=_integer_vector(model.dof_Madr,'dof_Madr',43)
        parent_values=tuple(int(x) for x in parent)
        madr_values=tuple(int(x) for x in madr)
        if any(x< -1 or x>=43 for x in parent_values):
            raise ValueError('sparse parent domain')
        if any(x<0 or x>=153 for x in madr_values):
            raise ValueError('dof_Madr domain')
        sparse=np.full((43,43),-1,np.int32)
        adr=0
        for i in range(43):
            if madr_values[i]!=adr: raise ValueError('dof_Madr inconsistent')
            j=i
            seen=set()
            while j>=0:
                if j>=43 or j in seen: raise ValueError('sparse parent cycle')
                seen.add(j)
                sparse[i,j]=adr
                sparse[j,i]=adr
                adr+=1
                j=parent_values[j]
        if adr!=153: raise ValueError('sparse count')
        self.capacity,self.device=capacity,device
        for name,tail in (('qpos',(48,)),('qvel',(43,)),
             ('site_pos',(3,)),('site_mat',(9,)),
             ('jacp',(3,43)),('jacr',(3,43)),
             ('qM',(153,)),('qfrc_bias',(43,)),
             ('ee_pos',(3,)),('ee_mat',(9,)),
             ('xvelp',(3,)),('xvelr',(3,)),
             ('joint_pos',(7,)),('joint_vel',(7,)),
             ('J',(6,7)),('M',(7,7)),('bias',(7,))):
            setattr(self,name,wp.zeros((capacity,)+tail,
                                      dtype=scalar.wp_real,device=device))
        self.active=wp.zeros(capacity,dtype=wp.int32,device=device)
        self.status=wp.zeros(capacity,dtype=wp.int32,device=device)
        self.qpos_index=wp.array(np.ascontiguousarray(qi,np.int32),
                                 dtype=wp.int32,device=device)
        self.qvel_index=wp.array(np.ascontiguousarray(vi,np.int32),
                                 dtype=wp.int32,device=device)
        self.sparse_map=wp.array(sparse,dtype=wp.int32,device=device)

    def launch(self,stream=None):
        wp.launch(_osc_inputs,dim=self.capacity,inputs=[self.qpos,self.qvel,
            self.site_pos,self.site_mat,self.jacp,self.jacr,self.qM,
            self.qfrc_bias,self.active,self.qpos_index,self.qvel_index,
            self.sparse_map,self.ee_pos,self.ee_mat,self.xvelp,self.xvelr,
            self.joint_pos,self.joint_vel,self.J,self.M,self.bias,self.status],
            device=self.device,stream=stream)
