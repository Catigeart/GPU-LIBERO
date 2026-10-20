"""Independent FP64 actuation and componentwise input-derived FP32 bounds."""
import numpy as np
from .mjwarp_motion_reference import U,exact,add,multiply


def clamp(value,limits):
    x,e=value;lo,hi=limits
    # Clamp is 1-Lipschitz; exact FP32 endpoints add no arithmetic error.
    out=float(np.fmin(np.fmax(lo,x),hi))
    low=float(np.fmin(np.fmax(lo,x-e),hi));high=float(np.fmin(np.fmax(lo,x+e),hi))
    return out,max(out-low,high-out)


def force(c):
    out=np.zeros((2,c['nactuator']));err=np.zeros_like(out)
    for w in range(2):
        for u in range(c['nactuator']):
            control=exact(0.)
            if c['ctrlnum'][u]>0:
                at=c['ctrladr'][u];control=exact(c['ctrl'][w,at])
                if c['ctrllimited'][at] and not c['disable_clampctrl']:
                    control=clamp(control,c['ctrlrange'][w%len(c['ctrlrange']),at])
            gp=c['gainprm'][w%len(c['gainprm']),u];bp=c['biasprm'][w%len(c['biasprm']),u]
            gain=exact(gp[0]);bias=exact(0.)
            if c['gain_type'][u]==1:
                gain=add(add(gain,multiply(exact(gp[1]),exact(c['length'][w,u]))),multiply(exact(gp[2]),exact(c['velocity'][w,u])))
            if c['bias_type'][u]==1:
                bias=add(add(exact(bp[0]),multiply(exact(bp[1]),exact(c['length'][w,u]))),multiply(exact(bp[2]),exact(c['velocity'][w,u])))
            result=add(multiply(gain,control),bias)
            if c['forcelimited'][u]:result=clamp(result,c['forcerange'][w%len(c['forcerange']),u])
            out[w,u],err[w,u]=result
    return {'actuator_force':out},{'actuator_force':err}


def dense_moment(c):
    m=np.zeros((2,c['nactuator'],c['nv']))
    for w in range(2):
        for u in range(c['nactuator']):
            for i in range(c['moment_rownnz'][w,u]):
                at=c['moment_rowadr'][w,u]+i;m[w,u,c['moment_colind'][w,at]]+=c['moment'][w,at]
    return m


def project(c,values=None,errors=None):
    f=c['actuator_force'].astype(float) if values is None else values['actuator_force']
    fe=np.zeros_like(f) if errors is None else errors['actuator_force']
    out=np.zeros((2,c['nv']));err=np.zeros_like(out);major=np.zeros_like(out);counts=np.zeros_like(out,int)
    for w in range(2):
        for u in range(c['nactuator']):
            for i in range(c['moment_rownnz'][w,u]):
                at=c['moment_rowadr'][w,u]+i;d=c['moment_colind'][w,at]
                v,e=multiply(exact(c['moment'][w,at]),(f[w,u],fe[w,u]))
                out[w,d]+=v;err[w,d]+=e;major[w,d]+=abs(v)+e;counts[w,d]+=1
    # Any legal atomic order is bounded by absolute rounded contributions.
    gamma=np.maximum(counts-1,0)*U/(1-np.maximum(counts-1,0)*U)
    err+=gamma*major
    return {'qfrc_actuator':out},{'qfrc_actuator':err}


def limits(c,values=None,errors=None):
    q=c['raw_qfrc'].astype(float) if values is None else values['qfrc_actuator']
    qe=np.zeros_like(q) if errors is None else errors['qfrc_actuator']
    out=np.empty_like(q);err=np.empty_like(q)
    for w in range(2):
        for d in range(c['nv']):
            j=c['dof_joint'][d];v=(q[w,d],qe[w,d])
            if c['gravity_enabled'] and c['actgravcomp'][j]:v=add(v,exact(c['gravcomp'][w,d]))
            if c['jointlimited'][j]:v=clamp(v,c['jointrange'][w%len(c['jointrange']),j])
            out[w,d],err[w,d]=v
    return {'qfrc_actuator':out},{'qfrc_actuator':err}


def reference(c,phase):
    if phase=='force':return force(c)
    if phase=='projection':return project(c)
    if phase=='limits':return limits(c)
    v,e=force(c);p,pe=project(c,v,e);q,qe=limits(c,p,pe)
    return v|q,e|qe
