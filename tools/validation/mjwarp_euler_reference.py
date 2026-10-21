"""Independent FP64 semi-implicit Euler and propagated FP32 roundoff bounds."""
import numpy as np
from .mjwarp_motion_reference import U,G1,exact,add,multiply
from .mjwarp_passive_reference import norm,divide


def normalize(q,quaternion=False):
    length,error=norm(q)
    if length==0:
        return exact(np.array([0.,0.,0.,1.]) if quaternion else np.zeros(3))
    return divide(q,(length,error))


def integrate_quaternion(q,v,dt):
    length,error=norm(v);axis,ae=normalize(v)
    angle,angle_error=multiply(exact(dt*.5),(length,error))
    # sin/cos have unit absolute derivative. Include four FP32 ulps of
    # libdevice error in addition to argument uncertainty (no fast-math).
    sine=np.sin(angle);cosine=np.cos(angle)
    se=angle_error+4*U*(1+abs(sine));ce=angle_error+4*U*(1+abs(cosine))
    xyz,xe=multiply((axis,ae),(sine,se));b=np.r_[cosine,xyz];be=np.r_[ce,xe]
    q,qe=normalize(q,True);out=np.empty(4);oe=np.empty(4)
    terms=(((0,0,1),(1,1,-1),(2,2,-1),(3,3,-1)),
           ((0,1,1),(1,0,1),(2,3,1),(3,2,-1)),
           ((0,2,1),(1,3,-1),(2,0,1),(3,1,1)),
           ((0,3,1),(1,2,1),(2,1,-1),(3,0,1)))
    for k,row in enumerate(terms):
        total=None
        for i,j,sign in row:
            value,err=multiply((q[i],qe[i]),(b[j],be[j]));term=(sign*value,err)
            total=term if total is None else add(total,term)
        out[k],oe[k]=total
    return normalize((out,oe),True)


def reference(c,phase):
    n,nv=2,c['nv'];dt=c['timestep'][np.arange(n)%len(c['timestep'])].astype(float)
    a=c['damping'][np.arange(n)%len(c['damping'])].astype(float)
    p=c['dampingpoly'][np.arange(n)%len(c['dampingpoly'])].astype(float)
    derivative=add(add(exact(a),multiply(multiply(exact(2.),exact(p[:,:,0])),exact(abs(c['qvel'])))),
                   multiply(multiply(multiply(exact(3.),exact(p[:,:,1])),exact(abs(c['qvel']))),exact(abs(c['qvel']))))
    matrix=c['matrix'].astype(float).copy();me=np.zeros_like(matrix)
    if phase in ('diagonal','composed'):
        der=derivative if phase=='composed' else exact(c['derivative'])
        diag=c['row']+c['nnz']-1
        matrix[:,diag],me[:,diag]=add(exact(matrix[:,diag]),multiply(exact(dt[:,None]),der))
    velocity=add(exact(c['qvel']),multiply(exact(c['acceleration']),exact(dt[:,None])))
    qv,ve=velocity if phase=='composed' else exact(c['qvel'])
    position=c['qpos'].astype(float).copy();pe=np.zeros_like(position)
    for w in range(n):
        for kind,q,d in zip(c['joint_type'],c['qposadr'],c['dofadr']):
            q,d=int(q),int(d)
            if kind==0:
                position[w,q:q+3],pe[w,q:q+3]=add(exact(position[w,q:q+3]),multiply(exact(dt[w]),(qv[w,d:d+3],ve[w,d:d+3])))
                q+=3;d+=3
            if kind<2:
                position[w,q:q+4],pe[w,q:q+4]=integrate_quaternion(exact(position[w,q:q+4]),(qv[w,d:d+3],ve[w,d:d+3]),dt[w])
            else:position[w,q],pe[w,q]=add(exact(position[w,q]),multiply(exact(dt[w]),(qv[w,d],ve[w,d])))
    values,errors={},{}
    for key,(value,error) in dict(derivative=derivative,matrix=(matrix,me),qvel=velocity,qpos=(position,pe)).items():
        if phase=='composed' or key==dict(derivative='derivative',diagonal='matrix',velocity='qvel',position='qpos')[phase]:
            values[key]=value;errors[key]=error
    return values,errors
