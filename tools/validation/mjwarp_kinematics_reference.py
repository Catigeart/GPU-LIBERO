"""Independent FP64 frames and normwise FP32 forward-sensitivity bounds.

Quaternions and vectors carry an L2 error radius internally. Quaternion
multiplication preserves product norms; rotation has norm ||q||^2 and
normalization's radial error is projected away. This avoids repeatedly treating
one orientation uncertainty as four independent coordinate errors on deep trees.
Output radii bound each component; the existing fixed8x safety is unchanged.
"""
import numpy as np
from .mjwarp_motion_reference import U,G1


def size(x):return float(np.linalg.norm(np.asarray(x).ravel()))
def exact(x):return np.asarray(x,np.float64),0.
def gamma(n):return n*U/(1-n*U)
def add(a,b):
    x,e=a;y,f=b
    return x+y,e+f+G1*(size(x)+size(y)+e+f)
def sub(a,b):return add(a,(-b[0],b[1]))
def multiply(a,b):
    x,e=a;y,f=b
    nx=size(x);ny=size(y)
    return x*y,nx*f+ny*e+e*f+G1*(nx+e)*(ny+f)


def normalize(q):
    value,error=q;length=size(value)
    if length==0 and error==0:return exact(np.array([0.,0.,0.,1.]))
    if length==0:return np.array([0.,0.,0.,1.]),2.+gamma(16)
    # ||x/rx-y/ry||^2=(||x-y||^2-(rx-ry)^2)/(rx*ry).
    # Norm/divide/rescale rounding contributes at most gamma16 on the sphere.
    sensitivity=error/np.sqrt(length*(length-error)) if error<length else 2.
    return value/length,sensitivity+gamma(16)


def mul(a,b):
    x,e=a;y,f=b;out=np.array([
        x[0]*y[0]-x[1]*y[1]-x[2]*y[2]-x[3]*y[3],
        x[0]*y[1]+x[1]*y[0]+x[2]*y[3]-x[3]*y[2],
        x[0]*y[2]-x[1]*y[3]+x[2]*y[0]+x[3]*y[1],
        x[0]*y[3]+x[1]*y[2]-x[2]*y[1]+x[3]*y[0]])
    # Four products/three sums per component. Their absolute majorant's L2
    # norm is <= 2||x||||y||; quaternion multiplication is norm multiplicative.
    nx,ny=size(x),size(y)
    return out,nx*f+ny*e+e*f+2*gamma(7)*(nx+e)*(ny+f)


def rotate(v,q):
    value,ve=v;quat,qe=q;s=quat[0];u=quat[1:]
    out=2*(np.dot(u,value)*u)+(s*s-np.dot(u,u))*value+2*s*np.cross(u,value)
    r,n=size(quat),size(value)
    inherited=(r+qe)**2*ve+(2*r*qe+qe*qe)*n
    # The three pinned polynomial terms have total L2 magnitude <= 8r^2 n;
    # each coordinate path uses fewer than32 rounded arithmetic operations.
    return out,inherited+8*gamma(32)*(r+qe)**2*(n+ve)


def axis_angle(axis,angle):
    v,e=axis;theta,te=angle;half=theta*.5
    ae=te*.5+G1*(abs(theta)+te)*.5
    sine,cosine=np.sin(half),np.cos(half)
    se=ae+4*U*(1+abs(sine));ce=ae+4*U*(1+abs(cosine))
    ve=abs(sine)*e+size(v)*se+e*se+G1*(size(v)+e)*(abs(sine)+se)
    return np.r_[cosine,v*sine],float(np.hypot(ce,ve))


def matrix(q):
    v,e=q;a,b,c,d=v[0]*v[0],v[0]*v[1],v[0]*v[2],v[0]*v[3]
    h,f,k,j,l,z=v[1]*v[1],v[1]*v[2],v[1]*v[3],v[2]*v[2],v[2]*v[3],v[3]*v[3]
    out=np.array([[a+h-j-z,2*(f-d),2*(k+c)],[2*(f+d),a-h+j-z,2*(l-b)],[2*(k-c),2*(l+b),a-h-j+z]])
    r=size(v)
    # Frobenius norm <= sqrt3 times the quaternion rotation operator norm;
    # a six-times-r^2 majorant bounds the nine local polynomial expressions.
    return out,np.sqrt(3.)*(2*r*e+e*e)+6*gamma(8)*(r+e)**2


def reference(c,phase):
    n,nb,nj,ng,ns=2,c['nbody'],c['njnt'],c['ngeom'],c['nsite'];values={};errors={}
    get=lambda key,w,i:exact(c[key][w%len(c[key]),i])
    pos=c['xpos'].astype(float).copy();quat=c['xquat'].astype(float).copy();pe=np.zeros((2,nb));qe=np.zeros((2,nb))
    anchor=c['xanchor'].astype(float).copy();ae=np.zeros((2,nj));axis=c['xaxis'].astype(float).copy();axe=np.zeros((2,nj))
    if phase in ('body','composed'):
        for w in range(n):
            for b in range(1,nb):
                first,count=int(c['body_jntadr'][b]),int(c['body_jntnum'][b]);parent=int(c['parent'][b])
                if count==1 and c['joint_type'][first]==0:
                    q=int(c['qposadr'][first]);p=exact(c['qpos'][w,q:q+3]);qq=normalize(exact(c['qpos'][w,q+3:q+7]))
                    anchor[w,first],ae[w,first]=p;axis[w,first],axe[w,first]=get('jnt_axis',w,first)
                else:
                    mc=int(c['mocap'][b])
                    p=get('mocap_pos',w,mc) if mc>=0 else get('body_pos',w,b)
                    qq=get('mocap_quat',w,mc) if mc>=0 else get('body_quat',w,b)
                    p=add(rotate(p,(quat[w,parent],qe[w,parent])),(pos[w,parent],pe[w,parent]));qq=mul((quat[w,parent],qe[w,parent]),qq)
                    for j in range(first,first+count):
                        q=int(c['qposadr'][j]);kind=int(c['joint_type'][j]);local=get('jnt_pos',w,j);ax=get('jnt_axis',w,j)
                        anc=add(rotate(local,qq),p);axisj=rotate(ax,qq)
                        if kind==1:qq=mul(qq,normalize(exact(c['qpos'][w,q:q+4])));p=sub(anc,rotate(local,qq))
                        elif kind==2:p=add(p,multiply(axisj,sub(exact(c['qpos'][w,q]),get('qpos0',w,q))))
                        elif kind==3:qq=mul(qq,axis_angle(ax,sub(exact(c['qpos'][w,q]),get('qpos0',w,q))));p=sub(anc,rotate(local,qq))
                        anchor[w,j],ae[w,j]=anc;axis[w,j],axe[w,j]=axisj
                    qq=normalize(qq)
                pos[w,b],pe[w,b]=p;quat[w,b],qe[w,b]=qq
        values.update(xpos=pos,xquat=quat,xanchor=anchor,xaxis=axis);errors.update(xpos=np.broadcast_to(pe[:,:,None],pos.shape),xquat=np.broadcast_to(qe[:,:,None],quat.shape),xanchor=np.broadcast_to(ae[:,:,None],anchor.shape),xaxis=np.broadcast_to(axe[:,:,None],axis.shape))
    for target in ('matrices','inertial','geom','site'):
        if phase not in (target,'composed'):continue
        if target=='matrices':
            out=np.empty((2,nb,3,3));err=np.empty_like(out)
            for w in range(2):
                for b in range(nb):out[w,b],err[w,b]=matrix((quat[w,b],qe[w,b]))
            values['xmat']=out;errors['xmat']=err;continue
        count=nb if target=='inertial' else ng if target=='geom' else ns
        pk,mk=('xipos','ximat') if target=='inertial' else ('geom_xpos','geom_xmat') if target=='geom' else ('site_xpos','site_xmat')
        pp,mm=c[pk].astype(float).copy(),c[mk].astype(float).copy();ep,em=np.zeros_like(pp),np.zeros_like(mm)
        for w in range(2):
            for item in range(count):
                b=item if target=='inertial' else int(c['geom_body'][item]) if target=='geom' else int(c['site_body'][item])
                if target=='geom' and c['weld'][b]==0 and c['mocap'][c['root'][b]]==-1:continue
                lp,lq=('body_ipos','body_iquat') if target=='inertial' else ('geom_pos','geom_quat') if target=='geom' else ('site_pos','site_quat')
                qq=(quat[w,b],qe[w,b]);pp[w,item],ep[w,item]=add((pos[w,b],pe[w,b]),rotate(get(lp,w,item),qq))
                mm[w,item],em[w,item]=matrix(mul(qq,get(lq,w,item)))
        values[pk]=pp;values[mk]=mm;errors[pk]=ep;errors[mk]=em
    return values,errors
