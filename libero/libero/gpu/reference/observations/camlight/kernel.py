"""MuJoCo 2.3.7 mj_camlight fixed route (Apache-2.0), device implementation.

The position/quaternion source order follows the accepted 2K1 implementation;
this private copy admits D01's world-outer two-dimensional ABI.
"""
from __future__ import annotations
import libero.gpu.precision as scalar

import warp as wp

wp.set_module_options({'enable_backward':False,'fuse_fp':True,'fast_math':False})

@wp.struct
class Model:
    cam_bodyid: wp.array(dtype=wp.int32)
    cam_pos: wp.array(dtype=scalar.wp_real)
    cam_quat: wp.array(dtype=scalar.wp_real)
    light_bodyid: wp.array(dtype=wp.int32)
    light_pos: wp.array(dtype=scalar.wp_real)
    light_dir: wp.array(dtype=scalar.wp_real)
    nbody: int
    ncam: int
    nlight: int

@wp.struct
class Data:
    active: wp.array(dtype=wp.int32)
    body_xpos: wp.array(dtype=scalar.wp_real,ndim=2)
    body_xquat: wp.array(dtype=scalar.wp_real,ndim=2)
    body_xmat: wp.array(dtype=scalar.wp_real,ndim=2)
    cam_xpos: wp.array(dtype=scalar.wp_real,ndim=2)
    cam_xmat: wp.array(dtype=scalar.wp_real,ndim=2)
    light_xpos: wp.array(dtype=scalar.wp_real,ndim=2)
    light_xdir: wp.array(dtype=scalar.wp_real,ndim=2)
    cam_tmpquat: wp.array(dtype=scalar.wp_real,ndim=2)
    light_raw_dir: wp.array(dtype=scalar.wp_real,ndim=2)
    light_norm: wp.array(dtype=scalar.wp_real,ndim=2)
    cam_identity: wp.array(dtype=wp.int32,ndim=2)
    light_identity: wp.array(dtype=wp.int32,ndim=2)
    light_norm_branch: wp.array(dtype=wp.int32,ndim=2)
    status: wp.array(dtype=wp.int32)

@wp.func
def _position(body_xpos:wp.array(dtype=scalar.wp_real,ndim=2),
              body_xmat:wp.array(dtype=scalar.wp_real,ndim=2),
              local:wp.array(dtype=scalar.wp_real),out:wp.array(dtype=scalar.wp_real,ndim=2),
              w:int,b:int,i:int):
    for row in range(3):
        a=9*b+3*row;c=3*i
        v=body_xmat[w,a]*local[c]
        v=v+body_xmat[w,a+1]*local[c+1]
        v=v+body_xmat[w,a+2]*local[c+2]
        out[w,c+row]=v+body_xpos[w,3*b+row]

@wp.func
def _matrix(body_xquat:wp.array(dtype=scalar.wp_real,ndim=2),
            local:wp.array(dtype=scalar.wp_real),out:wp.array(dtype=scalar.wp_real,ndim=2),
            tmp:wp.array(dtype=scalar.wp_real,ndim=2),w:int,b:int,i:int):
    a=4*b;c=4*i
    q0=body_xquat[w,a]*local[c]
    q0=q0-body_xquat[w,a+1]*local[c+1]
    q0=q0-body_xquat[w,a+2]*local[c+2]
    q0=q0-body_xquat[w,a+3]*local[c+3]
    q1=body_xquat[w,a]*local[c+1]
    q1=q1+body_xquat[w,a+1]*local[c]
    q1=q1+body_xquat[w,a+2]*local[c+3]
    q1=q1-body_xquat[w,a+3]*local[c+2]
    q2=body_xquat[w,a]*local[c+2]
    q2=q2-body_xquat[w,a+1]*local[c+3]
    q2=q2+body_xquat[w,a+2]*local[c]
    q2=q2+body_xquat[w,a+3]*local[c+1]
    q3=body_xquat[w,a]*local[c+3]
    q3=q3+body_xquat[w,a+1]*local[c+2]
    q3=q3-body_xquat[w,a+2]*local[c+1]
    q3=q3+body_xquat[w,a+3]*local[c]
    tmp[w,c]=q0;tmp[w,c+1]=q1;tmp[w,c+2]=q2;tmp[w,c+3]=q3
    m=9*i
    if q0==scalar.wp_real(1.0) and q1==scalar.wp_real(0.0) and q2==scalar.wp_real(0.0) and q3==scalar.wp_real(0.0):
        out[w,m]=scalar.wp_real(1.0);out[w,m+1]=scalar.wp_real(0.0);out[w,m+2]=scalar.wp_real(0.0)
        out[w,m+3]=scalar.wp_real(0.0);out[w,m+4]=scalar.wp_real(1.0);out[w,m+5]=scalar.wp_real(0.0)
        out[w,m+6]=scalar.wp_real(0.0);out[w,m+7]=scalar.wp_real(0.0);out[w,m+8]=scalar.wp_real(1.0)
    else:
        q00=q0*q0;q01=q0*q1;q02=q0*q2;q03=q0*q3
        q11=q1*q1;q12=q1*q2;q13=q1*q3
        q22=q2*q2;q23=q2*q3;q33=q3*q3
        out[w,m]=((q00+q11)-q22)-q33
        out[w,m+4]=((q00-q11)+q22)-q33
        out[w,m+8]=((q00-q11)-q22)+q33
        out[w,m+1]=scalar.wp_real(2.0)*(q12-q03)
        out[w,m+2]=scalar.wp_real(2.0)*(q13+q02)
        out[w,m+3]=scalar.wp_real(2.0)*(q12+q03)
        out[w,m+5]=scalar.wp_real(2.0)*(q23-q01)
        out[w,m+6]=scalar.wp_real(2.0)*(q13-q02)
        out[w,m+7]=scalar.wp_real(2.0)*(q23+q01)

@wp.func
def _light_direction(d:Data,m:Model,w:int,i:int,b:int):
    a=4*b
    v=3*i
    q0=d.body_xquat[w,a]
    q1=d.body_xquat[w,a+1]
    q2=d.body_xquat[w,a+2]
    q3=d.body_xquat[w,a+3]
    x=m.light_dir[v];y=m.light_dir[v+1];z=m.light_dir[v+2]
    if q0==scalar.wp_real(1.0) and q1==scalar.wp_real(0.0) and q2==scalar.wp_real(0.0) and q3==scalar.wp_real(0.0):
        d.light_identity[w,i]=1
        rx=x;ry=y;rz=z
    else:
        d.light_identity[w,i]=0
        t0=q0*x+q2*z-q3*y
        t1=q0*y+q3*x-q1*z
        t2=q0*z+q1*y-q2*x
        rx=x+scalar.wp_real(2.0)*(q2*t2-q3*t1)
        ry=y+scalar.wp_real(2.0)*(q3*t0-q1*t2)
        rz=z+scalar.wp_real(2.0)*(q1*t1-q2*t0)
    d.light_raw_dir[w,v]=rx
    d.light_raw_dir[w,v+1]=ry
    d.light_raw_dir[w,v+2]=rz
    norm=wp.sqrt((rx*rx+ry*ry)+rz*rz)
    d.light_norm[w,i]=norm
    if norm<scalar.wp_real(1.0e-15):
        d.light_norm_branch[w,i]=1
        d.light_xdir[w,v]=scalar.wp_real(1.0)
        d.light_xdir[w,v+1]=scalar.wp_real(0.0)
        d.light_xdir[w,v+2]=scalar.wp_real(0.0)
    else:
        d.light_norm_branch[w,i]=0
        inv=scalar.wp_real(1.0)/norm
        d.light_xdir[w,v]=rx*inv
        d.light_xdir[w,v+1]=ry*inv
        d.light_xdir[w,v+2]=rz*inv

@wp.func
def _clear(d:Data,m:Model,w:int):
    for i in range(m.ncam*3):d.cam_xpos[w,i]=scalar.wp_real(0.0)
    for i in range(m.ncam*9):d.cam_xmat[w,i]=scalar.wp_real(0.0)
    for i in range(m.ncam*4):d.cam_tmpquat[w,i]=scalar.wp_real(0.0)
    for i in range(m.nlight*3):
        d.light_xpos[w,i]=scalar.wp_real(0.0)
        d.light_xdir[w,i]=scalar.wp_real(0.0)
        d.light_raw_dir[w,i]=scalar.wp_real(0.0)
    for i in range(m.nlight):
        d.light_norm[w,i]=scalar.wp_real(0.0)
        d.light_identity[w,i]=0
        d.light_norm_branch[w,i]=0
    for i in range(m.ncam):d.cam_identity[w,i]=0

@wp.kernel
def compute(m:Model,d:Data):
    w=wp.tid()
    d.status[w]=0
    if d.active[w]!=0 and d.active[w]!=1:
        d.status[w]=1
        _clear(d,m,w)
        return
    if d.active[w]==0:
        _clear(d,m,w)
        return
    if m.ncam==0 and m.nlight==0:
        # The original empty loops read no body field.
        return
    for i in range(m.nbody*3):
        if not wp.isfinite(d.body_xpos[w,i]):
            d.status[w]=2;_clear(d,m,w);return
    for i in range(m.nbody*4):
        if not wp.isfinite(d.body_xquat[w,i]):
            d.status[w]=2;_clear(d,m,w);return
    for i in range(m.nbody*9):
        if not wp.isfinite(d.body_xmat[w,i]):
            d.status[w]=2;_clear(d,m,w);return
    for i in range(m.ncam):
        b=m.cam_bodyid[i]
        _position(d.body_xpos,d.body_xmat,m.cam_pos,d.cam_xpos,w,b,i)
        _matrix(d.body_xquat,m.cam_quat,d.cam_xmat,d.cam_tmpquat,w,b,i)
        q=4*i
        # _matrix's temporary is the exact branch input of mju_quat2Mat.
        d.cam_identity[w,i]=wp.where(d.cam_tmpquat[w,q]==scalar.wp_real(1.0) and
          d.cam_tmpquat[w,q+1]==scalar.wp_real(0.0) and
          d.cam_tmpquat[w,q+2]==scalar.wp_real(0.0) and
          d.cam_tmpquat[w,q+3]==scalar.wp_real(0.0),1,0)
    for i in range(m.nlight):
        b=m.light_bodyid[i]
        _position(d.body_xpos,d.body_xmat,m.light_pos,d.light_xpos,w,b,i)
        _light_direction(d,m,w,i,b)
    for i in range(m.ncam*3):
        if not wp.isfinite(d.cam_xpos[w,i]):d.status[w]=3
    for i in range(m.ncam*9):
        if not wp.isfinite(d.cam_xmat[w,i]):d.status[w]=3
    for i in range(m.nlight*3):
        if not wp.isfinite(d.light_xpos[w,i]) or not wp.isfinite(d.light_xdir[w,i]):d.status[w]=3
    for i in range(m.nlight):
        if not wp.isfinite(d.light_norm[w,i]):d.status[w]=3
    if d.status[w]!=0:_clear(d,m,w)
