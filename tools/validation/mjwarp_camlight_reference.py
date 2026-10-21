"""Independent FP64 FIXED frames; normwise FP32 arithmetic/sensitivity bounds."""
import numpy as np
from .mjwarp_kinematics_reference import exact,add,mul,rotate,matrix,size,gamma


def normalize_direction(vector):
    value,error=vector;length=size(value)
    if length==0 and error==0:return exact(np.zeros(3))
    if length==0:return np.zeros(3),2.+gamma(16)
    sensitivity=error/np.sqrt(length*(length-error)) if error<length else 2.
    return value/length,sensitivity+gamma(16)


def reference(c,phase,*,frame_errors=None):
    errors_in=frame_errors or {};nb,nc,nl=c['nbody'],c['ncam'],c['nlight'];values={};errors={}
    # Frozen standalone arrays are exact inputs. Composition carries the prior
    # kinematics allowance as a conservative L2 radius for each vector/quaternion.
    pe=errors_in.get('xpos',np.zeros_like(c['xpos'],dtype=float))
    qe=errors_in.get('xquat',np.zeros_like(c['xquat'],dtype=float))
    for prefix,count in (('cam',nc),('light',nl)):
        if phase not in (prefix,'composed'):continue
        pos=np.empty((2,count,3));ep=np.empty_like(pos)
        other=np.empty((2,count,3,3) if prefix=='cam' else (2,count,3));eo=np.empty_like(other)
        for w in range(2):
            for i in range(count):
                b=int(c[prefix+'_bodyid'][i]);p=c['xpos'][w,b].astype(float),size(pe[w,b]);q=c['xquat'][w,b].astype(float),size(qe[w,b])
                local=exact(c[prefix+'_pos'][w%len(c[prefix+'_pos']),i])
                pos[w,i],ep[w,i]=add(p,rotate(local,q))
                if prefix=='cam':
                    cq=exact(c['cam_quat'][w%len(c['cam_quat']),i]);other[w,i],eo[w,i]=matrix(mul(q,cq))
                else:
                    direction=exact(c['light_dir'][w%len(c['light_dir']),i]);other[w,i],eo[w,i]=normalize_direction(rotate(direction,q))
        values[prefix+'_xpos']=pos;errors[prefix+'_xpos']=ep
        key='cam_xmat' if prefix=='cam' else 'light_xdir';values[key]=other;errors[key]=eo
    return values,errors
