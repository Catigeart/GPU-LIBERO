"""Given OBB pairs, reproduce mj_collideOBB(product=NULL), not BVH traversal."""
from __future__ import annotations
import libero.gpu.precision as scalar
import numpy as np
import warp as wp

wp.set_module_options({'enable_backward':False,'fuse_fp':True,'fast_math':False})


@wp.func
def _dot(a0:scalar.wp_real,a1:scalar.wp_real,a2:scalar.wp_real,b0:scalar.wp_real,b1:scalar.wp_real,b2:scalar.wp_real)->scalar.wp_real:
    return a0*b0+a1*b1+a2*b2


@wp.kernel(enable_backward=False)
def _obb(aabb:wp.array(dtype=scalar.wp_real),pos:wp.array(dtype=scalar.wp_real),mat:wp.array(dtype=scalar.wp_real),
         active:wp.array(dtype=wp.int32),counts:wp.array(dtype=wp.int32),
         out:wp.array(dtype=wp.int32),axis:wp.array(dtype=wp.int32),status:wp.array(dtype=wp.int32),slots:int):
    tid=wp.tid();w=tid//slots;q=tid-w*slots
    out[tid]=0;axis[tid]=-1;status[tid]=0
    if active[w]==0:return
    if counts[w]<0 or counts[w]>slots:
        status[tid]=1;return
    if q>=counts[w]:return
    b=tid*12;p=tid*6;m=tid*18
    inf10=aabb[b+3]>=scalar.wp_real(1.0e10);inf11=aabb[b+4]>=scalar.wp_real(1.0e10);inf12=aabb[b+5]>=scalar.wp_real(1.0e10)
    inf20=aabb[b+9]>=scalar.wp_real(1.0e10);inf21=aabb[b+10]>=scalar.wp_real(1.0e10);inf22=aabb[b+11]>=scalar.wp_real(1.0e10)
    if (inf10 and inf11 and inf12) or (inf20 and inf21 and inf22):
        out[tid]=1;return
    c0=scalar.vec3_real();c1=scalar.vec3_real()
    for i in range(2):
        bb=b+6*i;mm=m+9*i;pp=p+3*i
        center=scalar.vec3_real(
            _dot(aabb[bb],aabb[bb+1],aabb[bb+2],mat[mm],mat[mm+1],mat[mm+2])+pos[pp],
            _dot(aabb[bb],aabb[bb+1],aabb[bb+2],mat[mm+3],mat[mm+4],mat[mm+5])+pos[pp+1],
            _dot(aabb[bb],aabb[bb+1],aabb[bb+2],mat[mm+6],mat[mm+7],mat[mm+8])+pos[pp+2])
        if i==0:c0=center
        else:c1=center
    for j in range(2):
        if (j==0 and (inf20 or inf21 or inf22)) or (j==1 and (inf10 or inf11 or inf12)):
            continue
        for k in range(3):
            jm=m+9*j
            n0=mat[jm+k];n1=mat[jm+3+k];n2=mat[jm+6+k]
            proj0=_dot(c0[0],c0[1],c0[2],n0,n1,n2)
            proj1=_dot(c1[0],c1[1],c1[2],n0,n1,n2)
            r0=scalar.wp_real(0.0);r1=scalar.wp_real(0.0)
            for i in range(2):
                bb=b+6*i;mm=m+9*i
                d0=_dot(mat[mm],mat[mm+3],mat[mm+6],n0,n1,n2)
                d1=_dot(mat[mm+1],mat[mm+4],mat[mm+7],n0,n1,n2)
                d2=_dot(mat[mm+2],mat[mm+5],mat[mm+8],n0,n1,n2)
                radius=wp.abs(aabb[bb+3]*d0)+wp.abs(aabb[bb+4]*d1)+wp.abs(aabb[bb+5]*d2)
                if i==0:r0=radius
                else:r1=radius
            if r0+r1<wp.abs(proj1-proj0):
                out[tid]=0;axis[tid]=3*j+k;return
    out[tid]=1


class OBBWorkspace:
    def __init__(self,nworld,slots,device='cuda:0'):
        if nworld<1 or slots<1:raise ValueError('positive dimensions required')
        self.nworld=nworld;self.slots=slots;self.device=device
        self.aabb=wp.zeros(nworld*slots*12,dtype=scalar.wp_real,device=device)
        self.pos=wp.zeros(nworld*slots*6,dtype=scalar.wp_real,device=device)
        self.mat=wp.zeros(nworld*slots*18,dtype=scalar.wp_real,device=device)
        self.active=wp.zeros(nworld,dtype=wp.int32,device=device)
        self.counts=wp.zeros(nworld,dtype=wp.int32,device=device)
        self.out=wp.zeros(nworld*slots,dtype=wp.int32,device=device)
        self.axis=wp.zeros(nworld*slots,dtype=wp.int32,device=device)
        self.status=wp.zeros(nworld*slots,dtype=wp.int32,device=device)

    def upload(self,aabb,pos,mat,active,counts):
        specs=((aabb,(self.nworld,self.slots,12),np.float64,self.aabb),
               (pos,(self.nworld,self.slots,6),np.float64,self.pos),
               (mat,(self.nworld,self.slots,18),np.float64,self.mat),
               (active,(self.nworld,),np.int32,self.active),
               (counts,(self.nworld,),np.int32,self.counts))
        for value,shape,dtype,dst in specs:
            arr=np.asarray(value)
            if arr.shape!=shape or arr.dtype!=dtype:raise ValueError('input dtype/shape')
            if value is active and not np.all((arr==0)|(arr==1)):raise ValueError('active mask')
            dst.assign(np.ascontiguousarray(arr).ravel())

    def launch(self,stream=None):
        wp.launch(_obb,dim=self.nworld*self.slots,inputs=[self.aabb,self.pos,self.mat,self.active,self.counts,
                  self.out,self.axis,self.status,self.slots],device=self.device,stream=stream)
