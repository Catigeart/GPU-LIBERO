"""Independent FP64 postconstraint RNE and input-derived FP32 error bounds."""
import numpy as np
from .mjwarp_motion_reference import U, exact, add, multiply, matvec, spatial, force_cross


def reference(c):
    nb=c['nbody']
    external_error=c.get('cfrc_ext_error',np.zeros_like(c['cfrc_ext'],dtype=np.float64))
    ca,cae,cf,cfe=(np.zeros((2,nb,6)) for _ in range(4))
    for w in range(2):
        if not c['gravity_disabled']:
            ca[w,0,3:]=-c['gravity'][w%len(c['gravity'])].astype(np.float64)
        for b in range(1,nb):
            p=c['parent'][b];acc=(ca[w,p].copy(),cae[w,p].copy())
            for d in range(int(c['dadr'][b]),int(c['dadr'][b]+c['dnum'][b])):
                acc=add(acc,multiply(exact(c['cdof_dot'][w,d]),exact(c['qvel'][w,d])))
                acc=add(acc,multiply(exact(c['cdof'][w,d]),exact(c['qacc'][w,d])))
            ca[w,b],cae[w,b]=acc
            inertia=spatial(c['cinert'][w,b]);v=exact(c['cvel'][w,b])
            force=add(matvec(inertia,acc),force_cross(v,matvec(inertia,v)))
            cf[w,b],cfe[w,b]=add(force,(-c['cfrc_ext'][w,b].astype(np.float64),external_error[w,b]))
        # Bound any legal sibling atomic order using absolute contributions.
        major=abs(cf[w])+cfe[w];count=np.ones(nb,int)
        for b in reversed(range(1,nb)):
            p=c['parent'][b];cf[w,p]+=cf[w,b];major[p]+=major[b];count[p]+=count[b]
            cfe[w,p]+=cfe[w,b]
        for b in range(nb):
            gamma=(count[b]-1)*U/(1-(count[b]-1)*U)
            cfe[w,b]+=gamma*major[b]
    return {'cacc':ca,'cfrc_int':cf},{'cacc':cae,'cfrc_int':cfe}
