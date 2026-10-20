"""Independent full-angle hinge kinematics diagnostic candidate.

Original bounded kernel and its qualification remain unchanged. Only finite
hinge half-angle >4 uses CUDA binary64 sin/cos. Original small-angle graph,
quaternion normalization, arithmetic ordering and atomic primitive commit stay.
"""
import libero.gpu.precision as scalar
import warp as wp
from .kernel import Model, Data, Diagnostic, _v3, _q4, _v3row, _q4row, _put3, _put4, _identity, _norm, _normalize, _mul, _rot, _rotmat, _mat, _commit
from .certified_trig import sin45, cos44
from .workspace import BodyKinematicsWorkspace
wp.set_module_options({'enable_backward':False,'fuse_fp':True,'fast_math':False})

@wp.kernel(enable_backward=False)
def _compute_wide(m:Model,d:Data,g:Diagnostic):
    w=wp.tid()
    # Clear every owned output and sideband on every replay, including inactive/error worlds.
    for k in range(m.nb*3): d.xpos[w,k]=scalar.wp_real(0.);g.pre_pos[w,k]=scalar.wp_real(0.)
    for k in range(m.nb*4): d.xquat[w,k]=scalar.wp_real(0.);g.pre_quat[w,k]=scalar.wp_real(0.)
    for k in range(m.nb*9): d.xmat[w,k]=scalar.wp_real(0.)
    for k in range(m.nq):d.qtmp[w,k]=scalar.wp_real(0.)
    for k in range(m.nm*4):d.mqtmp[w,k]=scalar.wp_real(0.)
    for k in range(m.nj*3): d.xanchor[w,k]=scalar.wp_real(0.);d.xaxis[w,k]=scalar.wp_real(0.);g.joint_pos[w,k]=scalar.wp_real(0.)
    for k in range(m.nj*4): g.joint_quat[w,k]=scalar.wp_real(0.)
    for k in range(3):d.world_xipos[w,k]=scalar.wp_real(0.)
    for k in range(9):d.world_ximat[w,k]=scalar.wp_real(0.)
    for k in range(m.nj):
        g.norm_before[w,k]=scalar.wp_real(0.);g.joint_route[w,k]=0;g.joint_rot_identity[w,k]=0;g.hinge_zero[w,k]=0;g.offcenter_rot_identity[w,k]=0
        g.trig_arg[w,k]=scalar.wp_real(0.);g.trig_sin[w,k]=scalar.wp_real(0.);g.trig_cos[w,k]=scalar.wp_real(0.);g.norm_branch[w,k]=0
    for k in range(m.nm):g.mocap_before[w,k]=scalar.wp_real(0.);g.mocap_branch[w,k]=0
    for k in range(m.nb):g.body_norm_before[w,k]=scalar.wp_real(0.);g.body_route[w,k]=0;g.body_mat_identity[w,k]=0;g.body_norm_branch[w,k]=0
    d.status[w]=0
    if d.active[w]==0:return
    if d.active[w]!=1:d.status[w]=1;return
    for k in range(m.nq):
        if not wp.isfinite(d.qpos[w,k]):d.status[w]=4;return
    for k in range(m.nm*3):
        if not wp.isfinite(d.mocap_pos[w,k]):d.status[w]=4;return
    for k in range(m.nm*4):
        if not wp.isfinite(d.mocap_quat[w,k]):d.status[w]=4;return
    # Hinge scalars are untouched by quaternion normalization. Reject the
    # bounded trig domain before any owned geometry or caller state is written.
    for j in range(m.nj):
        if m.jtype[j]==3:
            qa=m.qadr[j]
            angle=d.qpos[w,qa]-m.qpos0[qa]
            if angle!=scalar.wp_real(0.):
                z=angle*scalar.wp_real(0.5)
                if not wp.isfinite(z):d.status[w]=2;return
    # Original source writes five world values before any quaternion normalization.
    d.xquat[w,0]=scalar.wp_real(1.)
    _mat(scalar.vec4_real(scalar.wp_real(1.),scalar.wp_real(0.),scalar.wp_real(0.),scalar.wp_real(0.)),d.xmat,w,0)
    d.world_ximat[w,0]=scalar.wp_real(1.);d.world_ximat[w,4]=scalar.wp_real(1.);d.world_ximat[w,8]=scalar.wp_real(1.)
    for k in range(m.nq):d.qtmp[w,k]=d.qpos[w,k]
    for k in range(m.nm*4):d.mqtmp[w,k]=d.mocap_quat[w,k]
    for j in range(m.nj):
        t=m.jtype[j]
        if t==0 or t==1:
            a=m.qadr[j]
            if t==0:a+=3
            q=scalar.vec4_real(d.qtmp[w,a],d.qtmp[w,a+1],d.qtmp[w,a+2],d.qtmp[w,a+3])
            n=_norm(q)
            if not wp.isfinite(n):d.status[w]=5;return
            g.norm_before[w,j]=n
            if n<scalar.wp_real(1.e-15):g.norm_branch[w,j]=1
            q=_normalize(q,n)
            for k in range(4):d.qtmp[w,a+k]=q[k]
    for k in range(m.nm):
        q=_q4row(d.mqtmp,w,k);n=_norm(q)
        if not wp.isfinite(n):d.status[w]=5;return
        g.mocap_before[w,k]=n
        if n<scalar.wp_real(1.e-15):g.mocap_branch[w,k]=1
        _put4(d.mqtmp,w,k,_normalize(q,n))
    for b in range(1,m.nb):
        pos=scalar.vec3_real();quat=scalar.vec4_real()
        adr=m.jadr[b]
        if m.jnum[b]==1 and m.jtype[adr]==0:
            qa=m.qadr[adr]
            pos=scalar.vec3_real(d.qtmp[w,qa],d.qtmp[w,qa+1],d.qtmp[w,qa+2])
            quat=scalar.vec4_real(d.qtmp[w,qa+3],d.qtmp[w,qa+4],d.qtmp[w,qa+5],d.qtmp[w,qa+6])
            _put3(g.pre_pos,w,b,pos);_put4(g.pre_quat,w,b,quat);g.body_route[w,b]=1
            _put3(d.xanchor,w,adr,pos)
            _put3(d.xaxis,w,adr,scalar.vec3_real(scalar.wp_real(0.),scalar.wp_real(0.),scalar.wp_real(1.)))
            _put3(g.joint_pos,w,adr,pos);_put4(g.joint_quat,w,adr,quat);g.joint_route[w,adr]=0
        else:
            pid=m.parent[b]
            mid=m.mocapid[b]
            bodypos=_v3(m.bpos,b);bodyquat=_q4(m.bquat,b)
            if mid>=0:
                bodypos=_v3row(d.mocap_pos,w,mid);bodyquat=_q4row(d.mqtmp,w,mid);g.body_route[w,b]=2
            vec=_rotmat(bodypos,d.xmat,w,pid)
            ppos=_v3row(d.xpos,w,pid)
            pos=scalar.vec3_real(ppos[0]+vec[0],ppos[1]+vec[1],ppos[2]+vec[2])
            quat=_mul(_q4row(d.xquat,w,pid),bodyquat)
            _put3(g.pre_pos,w,b,pos);_put4(g.pre_quat,w,b,quat)
            for j in range(adr,adr+m.jnum[b]):
                qa=m.qadr[j];t=m.jtype[j]
                axis=_v3(m.jaxis,j);jp=_v3(m.jpos,j)
                if _identity(quat):g.joint_rot_identity[w,j]=1
                xaxis=_rot(axis,quat)
                xanchor=_rot(jp,quat)
                xanchor=scalar.vec3_real(xanchor[0]+pos[0],xanchor[1]+pos[1],xanchor[2]+pos[2])
                for k in range(3):
                    if not wp.isfinite(xaxis[k]) or not wp.isfinite(xanchor[k]):d.status[w]=5;return
                if t==2:
                    delta=d.qtmp[w,qa]-m.qpos0[qa]
                    pos=scalar.vec3_real(pos[0]+xaxis[0]*delta,pos[1]+xaxis[1]*delta,pos[2]+xaxis[2]*delta)
                elif t==1 or t==3:
                    qloc=scalar.vec4_real()
                    if t==1:qloc=scalar.vec4_real(d.qtmp[w,qa],d.qtmp[w,qa+1],d.qtmp[w,qa+2],d.qtmp[w,qa+3])
                    else:
                        angle=d.qtmp[w,qa]-m.qpos0[qa]
                        if angle==scalar.wp_real(0.):
                            g.hinge_zero[w,j]=1
                            qloc=scalar.vec4_real(scalar.wp_real(1.),scalar.wp_real(0.),scalar.wp_real(0.),scalar.wp_real(0.))
                        else:
                            z=angle*scalar.wp_real(0.5)
                            if not wp.isfinite(z):d.status[w]=2;return
                            if wp.abs(z)<=scalar.wp_real(4.):
                                sn=sin45(z);cs=cos44(z)
                            else:
                                sn=wp.sin(z);cs=wp.cos(z)
                            if not wp.isfinite(sn) or not wp.isfinite(cs):d.status[w]=5;return
                            g.trig_arg[w,j]=z;g.trig_sin[w,j]=sn;g.trig_cos[w,j]=cs
                            qloc=scalar.vec4_real(cs,axis[0]*sn,axis[1]*sn,axis[2]*sn)
                    quat=_mul(quat,qloc)
                    vec=scalar.vec3_real(xanchor[0]-pos[0],xanchor[1]-pos[1],xanchor[2]-pos[2])
                    if _identity(quat):g.offcenter_rot_identity[w,j]=1
                    vec1=_rot(jp,quat)
                    pos=scalar.vec3_real(pos[0]+(vec[0]-vec1[0]),pos[1]+(vec[1]-vec1[1]),pos[2]+(vec[2]-vec1[2]))
                else:d.status[w]=3;return
                for k in range(3):
                    if not wp.isfinite(pos[k]):d.status[w]=5;return
                for k in range(4):
                    if not wp.isfinite(quat[k]):d.status[w]=5;return
                _put3(d.xanchor,w,j,xanchor);_put3(d.xaxis,w,j,xaxis)
                _put3(g.joint_pos,w,j,pos);_put4(g.joint_quat,w,j,quat);g.joint_route[w,j]=t
        for k in range(3):
            if not wp.isfinite(pos[k]):d.status[w]=5;return
        n=_norm(quat)
        if not wp.isfinite(n):d.status[w]=5;return
        g.body_norm_before[w,b]=n
        if n<scalar.wp_real(1.e-15):g.body_norm_branch[w,b]=1
        quat=_normalize(quat,n)
        _put4(d.xquat,w,b,quat);_put3(d.xpos,w,b,pos)
        if _identity(quat):g.body_mat_identity[w,b]=1
        _mat(quat,d.xmat,w,b)
        for k in range(9):
            if not wp.isfinite(d.xmat[w,9*b+k]):d.status[w]=5;return


class WideAngleBodyKinematicsWorkspace(BodyKinematicsWorkspace):
    """Fresh prepared owners; caller must register this distinct source bundle."""
    def __init__(self,model,capacity,device='cuda:0',*,mode='diagnostic'):
        if mode!='diagnostic':
            raise ValueError('full-angle BodyK has no production runtime receipt')
        super().__init__(model,capacity,device)
    def launch(self,stream=None):
        self.model.validate()
        wp.launch(_compute_wide,dim=self.capacity,inputs=[self.static,self.data,self.diag],device=self.device,stream=stream)
        wp.launch(_commit,dim=self.capacity,inputs=[self.static,self.data,self.diag],device=self.device,stream=stream)
