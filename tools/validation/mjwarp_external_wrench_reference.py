"""Independent FP64 contact/spatial algebra and order-independent FP32 bounds."""
import numpy as np
from .mjwarp_motion_reference import U,exact,add,multiply,cross


def decode(c,i,*,adhesion=True):
    """Force:torque in contact axes, with the pinned capacity/address rules."""
    value=np.zeros(6);error=np.zeros(6)
    address=c['contact_address'][i];dim=c['contact_dim'][i];rows=c['njmax']
    if address[0]<0:return value,error
    efc=c['constraint_force'][c['contact_world'][i]]
    if c['cone']==0:
        if dim==1:value[0]=efc[address[0]]
        else:
            normal=exact(0.)
            for j in range(dim-1):
                row=int(address[0])+2*j
                p=exact(efc[row] if row<rows else 0.)
                n=exact(efc[row+1] if row+1<rows else 0.)
                normal=add(normal,add(p,n))
                value[j+1],error[j+1]=multiply(add(p,(-n[0],n[1])),exact(c['contact_friction'][i,j]))
            value[0],error[0]=normal
    else:
        for j in range(dim):
            if address[j]<rows:value[j]=efc[address[j]]
    if adhesion:
        value[0],error[0]=add((value[0],error[0]),exact(-float(c['contact_adhesion'][i])))
    return value,error


def rotate(rotation,vector):
    """Transpose-matrix action, with all coefficients taken as raw FP32 input."""
    result=[];bounds=[]
    for k in range(3):
        total=exact(0.)
        for j in range(3):total=add(total,multiply(exact(rotation[j,k]),(vector[0][j],vector[1][j])))
        result.append(total[0]);bounds.append(total[1])
    return np.array(result),np.array(bounds)


def translate(force,torque,com,point):
    offset=add(exact(com),exact(-point.astype(np.float64)))
    correction=cross(offset,force)
    shifted=add(torque,(-correction[0],correction[1]))
    return np.r_[shifted[0],force[0]],np.r_[shifted[1],force[1]]


def reference(c,phase='composed'):
    nb=c['nbody'];value=np.zeros((2,nb,6));error=np.zeros_like(value)
    if phase=='contact':value[:]=c['external']
    else:
        for w in range(2):
            for b in range(1,nb):
                applied=c['applied'][w,b]
                value[w,b],error[w,b]=translate(exact(applied[:3]),exact(applied[3:]),c['subtree_com'][w,c['root'][b]],c['xipos'][w,b])
    if phase=='applied':return {'cfrc_ext':value},{'cfrc_ext':error}
    count=np.zeros((2,nb),int);major=abs(value)+error
    for i in range(min(int(c['contact_count'][0]),c['naconmax'])):
        b1,b2=c['geom_body'][c['contact_geom'][i]]
        if b1==0 and b2==0:continue
        w=c['contact_world'][i];local,local_error=decode(c,i)
        force=rotate(c['contact_frame'][i],(local[:3],local_error[:3]))
        torque=rotate(c['contact_frame'][i],(local[3:],local_error[3:]))
        for b,sign in ((b1,-1),(b2,1)):
            if b==0:continue
            contribution,roundoff=translate(force,torque,c['subtree_com'][w,c['root'][b]],c['contact_pos'][i])
            value[w,b]+=sign*contribution;error[w,b]+=roundoff
            major[w,b]+=abs(contribution)+roundoff;count[w,b]+=1
    # Any legal FP32 atomic ordering, including cancellation, is covered by the
    # sum of absolute contributions, not the possibly tiny final result.
    gamma=count*U/(1-count*U)
    error+=gamma[:,:,None]*major
    return {'cfrc_ext':value},{'cfrc_ext':error}
