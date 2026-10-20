"""MuJoCo 2.3.7 mj_kinematics world/body/joint prefix, one thread per world."""
from __future__ import annotations
import libero.gpu.precision as scalar
import hashlib
import numpy as np
import warp as wp
from .certified_trig import sin45, cos44
wp.set_module_options({'enable_backward':False,'fuse_fp':True,'fast_math':False})

@wp.struct
class Model:
    qpos0: wp.array(dtype=scalar.wp_real)
    parent: wp.array(dtype=wp.int32)
    jnum: wp.array(dtype=wp.int32)
    jadr: wp.array(dtype=wp.int32)
    mocapid: wp.array(dtype=wp.int32)
    bpos: wp.array(dtype=scalar.wp_real)
    bquat: wp.array(dtype=scalar.wp_real)
    jtype: wp.array(dtype=wp.int32)
    qadr: wp.array(dtype=wp.int32)
    jaxis: wp.array(dtype=scalar.wp_real)
    jpos: wp.array(dtype=scalar.wp_real)
    nq: int
    nb: int
    nj: int
    nm: int

@wp.struct
class Data:
    active: wp.array(dtype=wp.int32)
    qpos: wp.array2d(dtype=scalar.wp_real)
    mocap_pos: wp.array2d(dtype=scalar.wp_real)
    mocap_quat: wp.array2d(dtype=scalar.wp_real)
    qtmp: wp.array2d(dtype=scalar.wp_real)
    mqtmp: wp.array2d(dtype=scalar.wp_real)
    xpos: wp.array2d(dtype=scalar.wp_real)
    xquat: wp.array2d(dtype=scalar.wp_real)
    xmat: wp.array2d(dtype=scalar.wp_real)
    xanchor: wp.array2d(dtype=scalar.wp_real)
    xaxis: wp.array2d(dtype=scalar.wp_real)
    world_xipos: wp.array2d(dtype=scalar.wp_real)
    world_ximat: wp.array2d(dtype=scalar.wp_real)
    status: wp.array(dtype=wp.int32)

@wp.struct
class Diagnostic:
    norm_before: wp.array2d(dtype=scalar.wp_real)
    mocap_before: wp.array2d(dtype=scalar.wp_real)
    pre_pos: wp.array2d(dtype=scalar.wp_real)
    pre_quat: wp.array2d(dtype=scalar.wp_real)
    joint_pos: wp.array2d(dtype=scalar.wp_real)
    joint_quat: wp.array2d(dtype=scalar.wp_real)
    body_norm_before: wp.array2d(dtype=scalar.wp_real)
    body_route: wp.array2d(dtype=wp.int32)
    joint_route: wp.array2d(dtype=wp.int32)
    body_mat_identity: wp.array2d(dtype=wp.int32)
    joint_rot_identity: wp.array2d(dtype=wp.int32)
    hinge_zero: wp.array2d(dtype=wp.int32)
    offcenter_rot_identity: wp.array2d(dtype=wp.int32)
    trig_arg: wp.array2d(dtype=scalar.wp_real)
    trig_sin: wp.array2d(dtype=scalar.wp_real)
    trig_cos: wp.array2d(dtype=scalar.wp_real)
    norm_branch: wp.array2d(dtype=wp.int32)
    mocap_branch: wp.array2d(dtype=wp.int32)
    body_norm_branch: wp.array2d(dtype=wp.int32)

@wp.func
def _v3(a: wp.array(dtype=scalar.wp_real), k: int) -> scalar.vec3_real:
    return scalar.vec3_real(a[3*k],a[3*k+1],a[3*k+2])

@wp.func
def _q4(a: wp.array(dtype=scalar.wp_real), k: int) -> scalar.vec4_real:
    return scalar.vec4_real(a[4*k],a[4*k+1],a[4*k+2],a[4*k+3])

@wp.func
def _v3row(a: wp.array2d(dtype=scalar.wp_real), w: int, k: int) -> scalar.vec3_real:
    return scalar.vec3_real(a[w,3*k],a[w,3*k+1],a[w,3*k+2])

@wp.func
def _q4row(a: wp.array2d(dtype=scalar.wp_real), w: int, k: int) -> scalar.vec4_real:
    return scalar.vec4_real(a[w,4*k],a[w,4*k+1],a[w,4*k+2],a[w,4*k+3])

@wp.func
def _put3(a: wp.array2d(dtype=scalar.wp_real), w: int, k: int, v: scalar.vec3_real):
    a[w,3*k]=v[0];a[w,3*k+1]=v[1];a[w,3*k+2]=v[2]

@wp.func
def _put4(a: wp.array2d(dtype=scalar.wp_real), w: int, k: int, v: scalar.vec4_real):
    a[w,4*k]=v[0];a[w,4*k+1]=v[1];a[w,4*k+2]=v[2];a[w,4*k+3]=v[3]

@wp.func
def _identity(q: scalar.vec4_real) -> bool:
    return q[0]==scalar.wp_real(1.0) and q[1]==scalar.wp_real(0.0) and q[2]==scalar.wp_real(0.0) and q[3]==scalar.wp_real(0.0)

@wp.func
def _norm(q: scalar.vec4_real) -> scalar.wp_real:
    return wp.sqrt(((q[0]*q[0]+q[1]*q[1])+q[2]*q[2])+q[3]*q[3])

@wp.func
def _normalize(q: scalar.vec4_real,n: scalar.wp_real) -> scalar.vec4_real:
    if n<scalar.wp_real(1.e-15):
        return scalar.vec4_real(scalar.wp_real(1.),scalar.wp_real(0.),scalar.wp_real(0.),scalar.wp_real(0.))
    inv=scalar.wp_real(1.)/n
    return scalar.vec4_real(q[0]*inv,q[1]*inv,q[2]*inv,q[3]*inv)

@wp.func
def _mul(a: scalar.vec4_real,b: scalar.vec4_real) -> scalar.vec4_real:
    return scalar.vec4_real(((a[0]*b[0]-a[1]*b[1])-a[2]*b[2])-a[3]*b[3],
      ((a[0]*b[1]+a[1]*b[0])+a[2]*b[3])-a[3]*b[2],
      ((a[0]*b[2]-a[1]*b[3])+a[2]*b[0])+a[3]*b[1],
      ((a[0]*b[3]+a[1]*b[2])-a[2]*b[1])+a[3]*b[0])

@wp.func
def _rot(v: scalar.vec3_real,q: scalar.vec4_real) -> scalar.vec3_real:
    if _identity(q): return v
    t0=(q[0]*v[0]+q[2]*v[2])-q[3]*v[1]
    t1=(q[0]*v[1]+q[3]*v[0])-q[1]*v[2]
    t2=(q[0]*v[2]+q[1]*v[1])-q[2]*v[0]
    return scalar.vec3_real(v[0]+scalar.wp_real(2.)*(q[2]*t2-q[3]*t1),
      v[1]+scalar.wp_real(2.)*(q[3]*t0-q[1]*t2),
      v[2]+scalar.wp_real(2.)*(q[1]*t1-q[2]*t0))

@wp.func
def _rotmat(v: scalar.vec3_real,mat: wp.array2d(dtype=scalar.wp_real),w: int,b: int) -> scalar.vec3_real:
    k=9*b
    return scalar.vec3_real((mat[w,k]*v[0]+mat[w,k+1]*v[1])+mat[w,k+2]*v[2],
      (mat[w,k+3]*v[0]+mat[w,k+4]*v[1])+mat[w,k+5]*v[2],
      (mat[w,k+6]*v[0]+mat[w,k+7]*v[1])+mat[w,k+8]*v[2])

@wp.func
def _mat(q: scalar.vec4_real,mat: wp.array2d(dtype=scalar.wp_real),w: int,b: int):
    k=9*b
    if _identity(q):
        mat[w,k]=scalar.wp_real(1.);mat[w,k+1]=scalar.wp_real(0.);mat[w,k+2]=scalar.wp_real(0.)
        mat[w,k+3]=scalar.wp_real(0.);mat[w,k+4]=scalar.wp_real(1.);mat[w,k+5]=scalar.wp_real(0.)
        mat[w,k+6]=scalar.wp_real(0.);mat[w,k+7]=scalar.wp_real(0.);mat[w,k+8]=scalar.wp_real(1.)
    else:
        q00=q[0]*q[0];q01=q[0]*q[1];q02=q[0]*q[2];q03=q[0]*q[3]
        q11=q[1]*q[1];q12=q[1]*q[2];q13=q[1]*q[3]
        q22=q[2]*q[2];q23=q[2]*q[3];q33=q[3]*q[3]
        mat[w,k]=((q00+q11)-q22)-q33
        mat[w,k+4]=((q00-q11)+q22)-q33
        mat[w,k+8]=((q00-q11)-q22)+q33
        mat[w,k+1]=scalar.wp_real(2.)*(q12-q03)
        mat[w,k+2]=scalar.wp_real(2.)*(q13+q02)
        mat[w,k+3]=scalar.wp_real(2.)*(q12+q03)
        mat[w,k+5]=scalar.wp_real(2.)*(q23-q01)
        mat[w,k+6]=scalar.wp_real(2.)*(q13-q02)
        mat[w,k+7]=scalar.wp_real(2.)*(q23+q01)

@wp.kernel(enable_backward=False)
def _compute(m:Model,d:Data,g:Diagnostic):
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
                if not wp.isfinite(z) or wp.abs(z)>scalar.wp_real(4.):d.status[w]=2;return
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
                            if wp.abs(z)>scalar.wp_real(4.) or not wp.isfinite(z):d.status[w]=2;return
                            sn=sin45(z);cs=cos44(z)
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

@wp.kernel(enable_backward=False)
def _commit(m:Model,d:Data,g:Diagnostic):
    w=wp.tid()
    if d.status[w]!=0:
        # A late arithmetic failure is rolled back in all module-owned outputs.
        # Caller qpos/mocap remain untouched on every failure path.
        for k in range(m.nb*3):d.xpos[w,k]=scalar.wp_real(0.);g.pre_pos[w,k]=scalar.wp_real(0.)
        for k in range(m.nb*4):d.xquat[w,k]=scalar.wp_real(0.);g.pre_quat[w,k]=scalar.wp_real(0.)
        for k in range(m.nb*9):d.xmat[w,k]=scalar.wp_real(0.)
        for k in range(m.nq):d.qtmp[w,k]=scalar.wp_real(0.)
        for k in range(m.nm*4):d.mqtmp[w,k]=scalar.wp_real(0.)
        for k in range(m.nj*3):d.xanchor[w,k]=scalar.wp_real(0.);d.xaxis[w,k]=scalar.wp_real(0.);g.joint_pos[w,k]=scalar.wp_real(0.)
        for k in range(m.nj*4):g.joint_quat[w,k]=scalar.wp_real(0.)
        for k in range(3):d.world_xipos[w,k]=scalar.wp_real(0.)
        for k in range(9):d.world_ximat[w,k]=scalar.wp_real(0.)
        for k in range(m.nj):
            g.norm_before[w,k]=scalar.wp_real(0.);g.joint_route[w,k]=0;g.joint_rot_identity[w,k]=0;g.hinge_zero[w,k]=0;g.offcenter_rot_identity[w,k]=0
            g.trig_arg[w,k]=scalar.wp_real(0.);g.trig_sin[w,k]=scalar.wp_real(0.);g.trig_cos[w,k]=scalar.wp_real(0.);g.norm_branch[w,k]=0
        for k in range(m.nm):g.mocap_before[w,k]=scalar.wp_real(0.);g.mocap_branch[w,k]=0
        for k in range(m.nb):g.body_norm_before[w,k]=scalar.wp_real(0.);g.body_route[w,k]=0;g.body_mat_identity[w,k]=0;g.body_norm_branch[w,k]=0
        return
    if d.active[w]!=1:return
    for k in range(m.nq):d.qpos[w,k]=d.qtmp[w,k]
    for k in range(m.nm*4):d.mocap_quat[w,k]=d.mqtmp[w,k]
