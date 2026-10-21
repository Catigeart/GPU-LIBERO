"""Independent FP64 scalar transmission and ordered CSR dot roundoff bounds."""
import numpy as np
from .mjwarp_motion_reference import exact,add,multiply


def transmission(c):
    n,na,nv=2,c['nactuator'],c['nv'];length=np.zeros((n,na));error=np.zeros_like(length);dense=np.zeros((n,na,nv))
    for w in range(n):
        for u in range(na):
            j=c['trnid'][u,0];g=c['gear'][w%len(c['gear']),u,0]
            length[w,u],error[w,u]=multiply(exact(c['qpos'][w,c['qposadr'][j]]),exact(g))
            dense[w,u,c['dofadr'][j]]=g
    return length,error,dense


def velocity(c):
    values=np.zeros((2,c['nactuator']));errors=np.zeros_like(values)
    for w in range(2):
        for u in range(c['nactuator']):
            start=c['moment_rowadr'][w,u];count=c['moment_rownnz'][w,u];total=exact(0.)
            for i in range(start,start+count):
                total=add(total,multiply(exact(c['moment'][w,i]),exact(c['qvel'][w,c['moment_colind'][w,i]])))
            values[w,u],errors[w,u]=total
    return {'actuator_velocity':values},{'actuator_velocity':errors}


def validate_csr(c):
    """Exact row ownership/column rules; physical row offsets may vary legally."""
    na,nv,nm=c['nactuator'],c['nv'],c['moment'].shape[1]
    dense=np.zeros((2,na,nv));used=np.zeros((2,nm),bool)
    for w in range(2):
        for u in range(na):
            start,count=int(c['moment_rowadr'][w,u]),int(c['moment_rownnz'][w,u])
            if count<0 or start<0 or start+count>nm:raise ValueError('CSR row exceeds storage')
            for i in range(start,start+count):
                col=int(c['moment_colind'][w,i])
                if col<0 or col>=nv or used[w,i]:raise ValueError('CSR column/row ownership differs')
                used[w,i]=True;dense[w,u,col]+=float(c['moment'][w,i])
    return dense,used
