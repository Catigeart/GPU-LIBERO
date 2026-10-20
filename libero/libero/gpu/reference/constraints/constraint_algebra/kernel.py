"""Independent N09 A/B MuJoCo constraint algebra kernels for the fixed dense model."""
from __future__ import annotations
import libero.gpu.precision as scalar
import hashlib
import numpy as np
import warp as wp

wp.set_module_options({'enable_backward':False,'fuse_fp':True,'fast_math':False})


@wp.kernel(enable_backward=False)
def _diag(count:wp.array(dtype=wp.int32),ncon:wp.array(dtype=wp.int32),active:wp.array(dtype=wp.int32),
          efc_type:wp.array(dtype=wp.int32),efc_id:wp.array(dtype=wp.int32),
          cgeom:wp.array(dtype=wp.int32),geom_body:wp.array(dtype=wp.int32),
          body_inv:wp.array(dtype=scalar.wp_real),dof_inv:wp.array(dtype=scalar.wp_real),jnt_dof:wp.array(dtype=wp.int32),
          out:wp.array(dtype=scalar.wp_real),status:wp.array(dtype=wp.int32),
          rows:int,contacts:int,ngeom:int,nbody:int,nv:int,njnt:int):
    tid=wp.tid();w=tid//rows;r=tid-w*rows
    out[tid]=scalar.wp_real(0.0);status[tid]=0
    if active[w]==0:return
    if active[w]!=1:status[tid]=7;return
    if count[w]<0 or count[w]>rows or ncon[w]<0 or ncon[w]>contacts:
        status[tid]=1;return
    if r>=count[w]:return
    typ=efc_type[tid];id=efc_id[tid]
    if typ==1:
        if id<0 or id>=nv:status[tid]=2;return
        out[tid]=dof_inv[id];return
    if typ==3:
        if id<0 or id>=njnt:status[tid]=2;return
        adr=jnt_dof[id]
        if adr<0 or adr>=nv:status[tid]=2;return
        out[tid]=dof_inv[adr];return
    if typ==5 or typ==7:
        if id<0 or id>=ncon[w]:status[tid]=2;return
        base=(w*contacts+id)*3
        g1=cgeom[base];g2=cgeom[base+1];dim=cgeom[base+2]
        if g1<0 or g2<0 or g1>=ngeom or g2>=ngeom or (dim!=1 and dim!=3 and dim!=4 and dim!=6):
            status[tid]=3;return
        if typ==7:
            start=r
            while start>0 and efc_type[w*rows+start-1]==7 and efc_id[w*rows+start-1]==id:
                start-=1
            if r-start>=dim or start+dim>count[w]:status[tid]=6;return
            for j in range(dim):
                if efc_type[w*rows+start+j]!=7 or efc_id[w*rows+start+j]!=id:
                    status[tid]=6;return
            if start+dim<count[w] and efc_type[w*rows+start+dim]==7 and efc_id[w*rows+start+dim]==id:
                status[tid]=6;return
        b1=geom_body[g1];b2=geom_body[g2]
        if b1<0 or b2<0 or b1>=nbody or b2>=nbody:status[tid]=3;return
        tran=body_inv[2*b1]+body_inv[2*b2]
        rot=body_inv[2*b1+1]+body_inv[2*b2+1]
        if typ==5:
            out[tid]=tran;return
        local=r-start
        if local<3:out[tid]=tran
        else:out[tid]=rot
        return
    status[tid]=5


@wp.kernel(enable_backward=False)
def _reference(count:wp.array(dtype=wp.int32),active:wp.array(dtype=wp.int32),
               rownnz:wp.array(dtype=wp.int32),rowadr:wp.array(dtype=wp.int32),
               colind:wp.array(dtype=wp.int32),J:wp.array(dtype=scalar.wp_real),
               qvel:wp.array(dtype=scalar.wp_real),kbip:wp.array(dtype=scalar.wp_real),
               epos:wp.array(dtype=scalar.wp_real),margin:wp.array(dtype=scalar.wp_real),
               vel:wp.array(dtype=scalar.wp_real),aref:wp.array(dtype=scalar.wp_real),status:wp.array(dtype=wp.int32),
               rows:int,nnzcap:int,nv:int):
    tid=wp.tid();w=tid//rows;r=tid-w*rows
    vel[tid]=scalar.wp_real(0.0);aref[tid]=scalar.wp_real(0.0);status[tid]=0
    if active[w]==0:return
    if active[w]!=1:status[tid]=4;return
    if count[w]<0 or count[w]>rows:status[tid]=1;return
    if r>=count[w]:return
    nnz=rownnz[tid];adr=rowadr[tid]
    if nnz!=43 or adr!=r*43 or adr+43>nnzcap:status[tid]=2;return
    base=w*nnzcap
    acc0=scalar.wp_real(0.0);acc1=scalar.wp_real(0.0);acc2=scalar.wp_real(0.0);acc3=scalar.wp_real(0.0)
    if nnz>=4:
        c0=colind[base+adr];c1=colind[base+adr+1];c2=colind[base+adr+2];c3=colind[base+adr+3]
        if c0!=0 or c1!=1 or c2!=2 or c3!=3:
            status[tid]=3;return
        acc0=J[base+adr]*qvel[w*nv+c0]
        acc1=J[base+adr+1]*qvel[w*nv+c1]
        acc2=J[base+adr+2]*qvel[w*nv+c2]
        acc3=J[base+adr+3]*qvel[w*nv+c3]
        for block in range(1,nnz//4):
            i=4*block
            c0=colind[base+adr+i];c1=colind[base+adr+i+1];c2=colind[base+adr+i+2];c3=colind[base+adr+i+3]
            if c0!=i or c1!=i+1 or c2!=i+2 or c3!=i+3:
                status[tid]=3;return
            acc0=acc0+J[base+adr+i]*qvel[w*nv+c0]
            acc1=acc1+J[base+adr+i+1]*qvel[w*nv+c1]
            acc2=acc2+J[base+adr+i+2]*qvel[w*nv+c2]
            acc3=acc3+J[base+adr+i+3]*qvel[w*nv+c3]
    v=(acc0+acc2)+(acc1+acc3)
    tail=nnz%4
    first=nnz-tail
    if tail>=1:
        c=colind[base+adr+first]
        if c!=first:status[tid]=3;return
        t0=J[base+adr+first]*qvel[w*nv+c]
        if tail==1:
            v=t0+v
        else:
            c=colind[base+adr+first+1]
            if c!=first+1:status[tid]=3;return
            t1=J[base+adr+first+1]*qvel[w*nv+c]
            partial=t0+t1
            if tail==3:
                c=colind[base+adr+first+2]
                if c!=first+2:status[tid]=3;return
                t2=J[base+adr+first+2]*qvel[w*nv+c]
                partial=t2+partial
            v=v+partial
    vel[tid]=v
    aref[tid]=-kbip[tid*4+1]*v-kbip[tid*4]*kbip[tid*4+2]*(epos[tid]-margin[tid])


def _assign(value,shape,dtype,dst):
    arr=np.asarray(value)
    if arr.shape!=shape or arr.dtype!=dtype:raise ValueError(f'input schema {arr.shape}/{arr.dtype} != {shape}/{dtype}')
    dst.assign(np.ascontiguousarray(arr).ravel())


# SHA-256 of the four model arrays extracted from the fixed, hash-checked MJB.
_STATIC_SCHEMA={
    'geom_bodyid':((270,),np.dtype('int32'),'070eb94ada17eacb0594540416f6bd8f58daa167c0158727e3b7bfbfbf206189'),
    'body_invweight0':((38,2),np.dtype('float64'),'acdcd90bdc312dbb12207478253c863d9e3eb8f94f7a3a8c8361c2409d5290f6'),
    'dof_invweight0':((43,),np.dtype('float64'),'948ba8f2b7f47fcb111d8fbe72a24c7a6dce8a63e9adadbce1154d0451b21568'),
    'jnt_dofadr':((18,),np.dtype('int32'),'13c9448fa632713f92cc44c1c525c473e12ce7105bc7b23c30d8fcdf8b90a02b'),
}


def validate_fixed_static(static):
    """Validate A's production static input against the frozen MJB adapter."""
    if set(static)!=set(_STATIC_SCHEMA):raise ValueError('fixed model static keys')
    for key,(shape,dtype,digest) in _STATIC_SCHEMA.items():
        value=np.asarray(static[key])
        if value.shape!=shape or value.dtype!=dtype:raise ValueError(f'fixed model {key} schema')
        if hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()!=digest:
            raise ValueError(f'fixed model {key} identity')


class DiagApproxWorkspace:
    def __init__(self,nworld,rows,contacts,static,device='cuda:0'):
        if nworld<1 or rows<1 or contacts<1:raise ValueError('positive capacity required')
        validate_fixed_static(static)
        self.nworld=nworld;self.rows=rows;self.contacts=contacts;self.device=device
        self.ngeom=len(static['geom_bodyid']);self.nbody=np.asarray(static['body_invweight0']).size//2
        self.nv=len(static['dof_invweight0']);self.njnt=len(static['jnt_dofadr'])
        self.static={}
        for key,array in static.items():
            a=np.asarray(array)
            if a.dtype not in (np.dtype('int32'),np.dtype('float64')):raise ValueError('static dtype')
            self.static[key]=wp.array(np.ascontiguousarray(a).ravel(),dtype=wp.int32 if a.dtype==np.int32 else scalar.wp_real,device=device)
        self.count=wp.zeros(nworld,dtype=wp.int32,device=device);self.ncon=wp.zeros(nworld,dtype=wp.int32,device=device)
        self.active=wp.zeros(nworld,dtype=wp.int32,device=device)
        self.efc_type=wp.zeros(nworld*rows,dtype=wp.int32,device=device);self.efc_id=wp.zeros(nworld*rows,dtype=wp.int32,device=device)
        self.cgeom=wp.zeros(nworld*contacts*3,dtype=wp.int32,device=device)
        self.out=wp.zeros(nworld*rows,dtype=scalar.wp_real,device=device);self.status=wp.zeros(nworld*rows,dtype=wp.int32,device=device)

    def upload(self,count,ncon,active,efc_type,efc_id,cgeom):
        if not np.all((np.asarray(active)==0)|(np.asarray(active)==1)):raise ValueError('active mask')
        for a,s,t,d in ((count,(self.nworld,),np.int32,self.count),(ncon,(self.nworld,),np.int32,self.ncon),
            (active,(self.nworld,),np.int32,self.active),(efc_type,(self.nworld,self.rows),np.int32,self.efc_type),
            (efc_id,(self.nworld,self.rows),np.int32,self.efc_id),(cgeom,(self.nworld,self.contacts,3),np.int32,self.cgeom)):
            _assign(a,s,t,d)

    def launch(self,stream=None):
        s=self.static
        wp.launch(_diag,dim=self.nworld*self.rows,inputs=[self.count,self.ncon,self.active,self.efc_type,self.efc_id,
            self.cgeom,s['geom_bodyid'],s['body_invweight0'],s['dof_invweight0'],s['jnt_dofadr'],
            self.out,self.status,self.rows,self.contacts,self.ngeom,self.nbody,self.nv,self.njnt],device=self.device,stream=stream)


class ReferenceWorkspace:
    def __init__(self,nworld,rows,nnzcap,nv=43,device='cuda:0'):
        if nworld<1 or rows<1 or nnzcap<rows*43 or nv!=43:raise ValueError('fixed dense43 capacity required')
        self.nworld=nworld;self.rows=rows;self.nnzcap=nnzcap;self.nv=nv;self.device=device
        self.count=wp.zeros(nworld,dtype=wp.int32,device=device);self.active=wp.zeros(nworld,dtype=wp.int32,device=device)
        self.rownnz=wp.zeros(nworld*rows,dtype=wp.int32,device=device);self.rowadr=wp.zeros(nworld*rows,dtype=wp.int32,device=device)
        self.colind=wp.zeros(nworld*nnzcap,dtype=wp.int32,device=device);self.J=wp.zeros(nworld*nnzcap,dtype=scalar.wp_real,device=device)
        self.qvel=wp.zeros(nworld*nv,dtype=scalar.wp_real,device=device);self.kbip=wp.zeros(nworld*rows*4,dtype=scalar.wp_real,device=device)
        self.epos=wp.zeros(nworld*rows,dtype=scalar.wp_real,device=device);self.margin=wp.zeros(nworld*rows,dtype=scalar.wp_real,device=device)
        self.vel=wp.zeros(nworld*rows,dtype=scalar.wp_real,device=device);self.aref=wp.zeros(nworld*rows,dtype=scalar.wp_real,device=device)
        self.status=wp.zeros(nworld*rows,dtype=wp.int32,device=device)

    def upload(self,count,active,rownnz,rowadr,colind,J,qvel,kbip,epos,margin):
        if not np.all((np.asarray(active)==0)|(np.asarray(active)==1)):raise ValueError('active mask')
        specs=((count,(self.nworld,),np.int32,self.count),(active,(self.nworld,),np.int32,self.active),
            (rownnz,(self.nworld,self.rows),np.int32,self.rownnz),(rowadr,(self.nworld,self.rows),np.int32,self.rowadr),
            (colind,(self.nworld,self.nnzcap),np.int32,self.colind),(J,(self.nworld,self.nnzcap),np.float64,self.J),
            (qvel,(self.nworld,self.nv),np.float64,self.qvel),(kbip,(self.nworld,self.rows,4),np.float64,self.kbip),
            (epos,(self.nworld,self.rows),np.float64,self.epos),(margin,(self.nworld,self.rows),np.float64,self.margin))
        for a,s,t,d in specs:_assign(a,s,t,d)

    def launch(self,stream=None):
        wp.launch(_reference,dim=self.nworld*self.rows,inputs=[self.count,self.active,self.rownnz,self.rowadr,self.colind,
            self.J,self.qvel,self.kbip,self.epos,self.margin,self.vel,self.aref,self.status,self.rows,self.nnzcap,self.nv],
            device=self.device,stream=stream)
