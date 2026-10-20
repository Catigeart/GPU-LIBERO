"""Source-ordered MuJoCo 2.3.7 mj_rne(flg_acc=0), Apache-2.0."""
import libero.gpu.precision as scalar
from dataclasses import dataclass
import numpy as np
import warp as wp

wp.set_module_options({'enable_backward':False,'fuse_fp':True,'fast_math':False})

@dataclass(frozen=True)
class RNEBiasModel:
    nbody:int
    nv:int
    body_parent:object
    body_dofadr:object
    body_dofnum:object
    dof_body:object
    gravity:object
    gravity_disabled:int

    @classmethod
    def prepare(cls,m,device='cuda'):
        nb,nv=int(m.nbody),int(m.nv)
        if nb<1 or nv<0:raise ValueError('invalid dimensions')
        def integer(name,size):
            a=np.asarray(getattr(m,name))
            if a.shape!=(size,) or not np.issubdtype(a.dtype,np.integer) or np.any(a<-(1<<31)) or np.any(a>=1<<31):
                raise ValueError(name)
            return a.astype(np.int32)
        bp=integer('body_parentid',nb);ba=integer('body_dofadr',nb);bn=integer('body_dofnum',nb);db=integer('dof_bodyid',nv)
        gravity=np.asarray(m.opt_gravity)
        disabled=integer('gravity_disabled',1)
        if gravity.shape!=(3,) or gravity.dtype!=np.float64 or not np.all(np.isfinite(gravity)) or int(disabled[0]) not in (0,1):
            raise ValueError('gravity schema')
        if bp[0]!=0 or bn[0]!=0 or np.any(bp[1:]<0) or np.any(bp[1:]>=np.arange(1,nb)):
            raise ValueError('body topology')
        if (np.any(bn<0) or np.any((bn==0)&(ba!=-1)) or
                np.any((bn>0)&((ba<0)|(ba+bn>nv))) or
                np.any(db<=0) or np.any(db>=nb)):
            raise ValueError('body dof range')
        covered=np.zeros(nv,dtype=np.int32)
        for i in range(1,nb):
            a,c=int(ba[i]),int(bn[i])
            if c:
                covered[a:a+c]+=1
                if np.any(db[a:a+c]!=i):raise ValueError('dof body mismatch')
        if np.any(covered!=1):raise ValueError('dof coverage')
        return cls(nb,nv,*(wp.array(x,dtype=wp.int32,device=device) for x in (bp,ba,bn,db)),
                   wp.array(gravity,dtype=scalar.wp_real,device=device),int(disabled[0]))

@wp.func
def _dot6(dof:wp.array2d(dtype=scalar.wp_real),frc:wp.array2d(dtype=scalar.wp_real),w:int,d:int,b:int)->scalar.wp_real:
    p0=dof[w,6*d]*frc[w,6*b]
    p1=dof[w,6*d+1]*frc[w,6*b+1]
    p2=dof[w,6*d+2]*frc[w,6*b+2]
    p3=dof[w,6*d+3]*frc[w,6*b+3]
    p4=dof[w,6*d+4]*frc[w,6*b+4]
    p5=dof[w,6*d+5]*frc[w,6*b+5]
    return ((p0+p2)+(p1+p3))+(p4+p5)

@wp.kernel
def _rne_bias(active:wp.array(dtype=wp.int32),cdof_dot:wp.array2d(dtype=scalar.wp_real),
              qvel:wp.array2d(dtype=scalar.wp_real),cinert:wp.array2d(dtype=scalar.wp_real),
              cvel:wp.array2d(dtype=scalar.wp_real),cdof:wp.array2d(dtype=scalar.wp_real),
              bp:wp.array(dtype=wp.int32),ba:wp.array(dtype=wp.int32),bn:wp.array(dtype=wp.int32),
              db:wp.array(dtype=wp.int32),gravity:wp.array(dtype=scalar.wp_real),disabled:int,
              nb:int,nv:int,cacc:wp.array2d(dtype=scalar.wp_real),frc:wp.array2d(dtype=scalar.wp_real),
              result:wp.array2d(dtype=scalar.wp_real),status:wp.array(dtype=wp.int32)):
    w=wp.tid();status[w]=0
    for k in range(nb*6):
        cacc[w,k]=scalar.wp_real(0.0);frc[w,k]=scalar.wp_real(0.0)
    for k in range(nv):result[w,k]=scalar.wp_real(0.0)
    if active[w]==0:return
    if active[w]!=1:status[w]=1;return
    for k in range(nv*6):
        if not wp.isfinite(cdof_dot[w,k]) or not wp.isfinite(cdof[w,k]):status[w]=2;return
    for k in range(nv):
        if not wp.isfinite(qvel[w,k]):status[w]=2;return
    for k in range(nb*10):
        if not wp.isfinite(cinert[w,k]):status[w]=2;return
    for k in range(nb*6):
        if not wp.isfinite(cvel[w,k]):status[w]=2;return
    if disabled==0:
        for k in range(3):cacc[w,3+k]=gravity[k]*scalar.wp_real(-1.0)
    for i in range(1,nb):
        adr=ba[i];count=bn[i]
        # mju_mulDofVec: n=0 zero, n=1 scale, n>1 row-wise zero-skip accumulation.
        t0,t1,t2=scalar.wp_real(0.0),scalar.wp_real(0.0),scalar.wp_real(0.0)
        t3,t4,t5=scalar.wp_real(0.0),scalar.wp_real(0.0),scalar.wp_real(0.0)
        if count==1:
            q=qvel[w,adr];d=6*adr
            t0=cdof_dot[w,d]*q;t1=cdof_dot[w,d+1]*q;t2=cdof_dot[w,d+2]*q
            t3=cdof_dot[w,d+3]*q;t4=cdof_dot[w,d+4]*q;t5=cdof_dot[w,d+5]*q
        elif count>1:
            for j in range(count):
                q=qvel[w,adr+j]
                if q!=scalar.wp_real(0.0):
                    d=6*(adr+j)
                    t0+=cdof_dot[w,d]*q;t1+=cdof_dot[w,d+1]*q;t2+=cdof_dot[w,d+2]*q
                    t3+=cdof_dot[w,d+3]*q;t4+=cdof_dot[w,d+4]*q;t5+=cdof_dot[w,d+5]*q
        p=6*bp[i];a=6*i;b=10*i
        cacc[w,a]=cacc[w,p]+t0;cacc[w,a+1]=cacc[w,p+1]+t1;cacc[w,a+2]=cacc[w,p+2]+t2
        cacc[w,a+3]=cacc[w,p+3]+t3;cacc[w,a+4]=cacc[w,p+4]+t4;cacc[w,a+5]=cacc[w,p+5]+t5
        v0,v1,v2=cacc[w,a],cacc[w,a+1],cacc[w,a+2]
        v3,v4,v5=cacc[w,a+3],cacc[w,a+4],cacc[w,a+5]
        f0=cinert[w,b]*v0+cinert[w,b+3]*v1+cinert[w,b+4]*v2-cinert[w,b+8]*v4+cinert[w,b+7]*v5
        f1=cinert[w,b+3]*v0+cinert[w,b+1]*v1+cinert[w,b+5]*v2+cinert[w,b+8]*v3-cinert[w,b+6]*v5
        f2=cinert[w,b+4]*v0+cinert[w,b+5]*v1+cinert[w,b+2]*v2-cinert[w,b+7]*v3+cinert[w,b+6]*v4
        f3=cinert[w,b+8]*v1-cinert[w,b+7]*v2+cinert[w,b+9]*v3
        f4=cinert[w,b+6]*v2-cinert[w,b+8]*v0+cinert[w,b+9]*v4
        f5=cinert[w,b+7]*v0-cinert[w,b+6]*v1+cinert[w,b+9]*v5
        cv0,cv1,cv2=cvel[w,a],cvel[w,a+1],cvel[w,a+2]
        cv3,cv4,cv5=cvel[w,a+3],cvel[w,a+4],cvel[w,a+5]
        t0=cinert[w,b]*cv0+cinert[w,b+3]*cv1+cinert[w,b+4]*cv2-cinert[w,b+8]*cv4+cinert[w,b+7]*cv5
        t1=cinert[w,b+3]*cv0+cinert[w,b+1]*cv1+cinert[w,b+5]*cv2+cinert[w,b+8]*cv3-cinert[w,b+6]*cv5
        t2=cinert[w,b+4]*cv0+cinert[w,b+5]*cv1+cinert[w,b+2]*cv2-cinert[w,b+7]*cv3+cinert[w,b+6]*cv4
        t3=cinert[w,b+8]*cv1-cinert[w,b+7]*cv2+cinert[w,b+9]*cv3
        t4=cinert[w,b+6]*cv2-cinert[w,b+8]*cv0+cinert[w,b+9]*cv4
        t5=cinert[w,b+7]*cv0-cinert[w,b+6]*cv1+cinert[w,b+9]*cv5
        x0=-cv2*t1+cv1*t2;x1=cv2*t0-cv0*t2;x2=-cv1*t0+cv0*t1
        x3=-cv2*t4+cv1*t5;x4=cv2*t3-cv0*t5;x5=-cv1*t3+cv0*t4
        x0+=-cv5*t4+cv4*t5;x1+=cv5*t3-cv3*t5;x2+=-cv4*t3+cv3*t4
        frc[w,a]=f0+x0;frc[w,a+1]=f1+x1;frc[w,a+2]=f2+x2
        frc[w,a+3]=f3+x3;frc[w,a+4]=f4+x4;frc[w,a+5]=f5+x5
    for ii in range(nb-1):
        i=nb-1-ii;p=bp[i]
        if p>0:
            for k in range(6):frc[w,6*p+k]+=frc[w,6*i+k]
    for i in range(nv):result[w,i]=_dot6(cdof,frc,w,i,db[i])
    for k in range(nb*6):
        if not wp.isfinite(cacc[w,k]) or not wp.isfinite(frc[w,k]):status[w]=3
    for k in range(nv):
        if not wp.isfinite(result[w,k]):status[w]=3
    if status[w]==3:
        for k in range(nb*6):
            cacc[w,k]=scalar.wp_real(0.0);frc[w,k]=scalar.wp_real(0.0)
        for k in range(nv):result[w,k]=scalar.wp_real(0.0)

def launch(model,active,cdof_dot,qvel,cinert,cvel,cdof,cacc,frc,result,status,stream=None):
    n=active.shape[0]
    schema=((active,(n,),wp.int32),(cdof_dot,(n,model.nv*6),scalar.wp_real),
            (qvel,(n,model.nv),scalar.wp_real),(cinert,(n,model.nbody*10),scalar.wp_real),
            (cvel,(n,model.nbody*6),scalar.wp_real),(cdof,(n,model.nv*6),scalar.wp_real),
            (cacc,(n,model.nbody*6),scalar.wp_real),(frc,(n,model.nbody*6),scalar.wp_real),
            (result,(n,model.nv),scalar.wp_real),(status,(n,),wp.int32))
    for a,shape,dtype in schema:
        if a.shape!=shape or a.dtype!=dtype or not a.device.is_cuda:raise ValueError(f'invalid device array: {shape}/{dtype}')
    wp.launch(_rne_bias,dim=n,inputs=[active,cdof_dot,qvel,cinert,cvel,cdof,model.body_parent,
              model.body_dofadr,model.body_dofnum,model.dof_body,model.gravity,model.gravity_disabled,
              model.nbody,model.nv],outputs=[cacc,frc,result,status],stream=stream)
