"""Independent double rigid queries, with absolute-operation FP32 bounds."""
import numpy as np

U=2.**-24

def cross_size(a,b):
    return np.abs(a)[...,[1,2,0]]*np.abs(b)[...,[2,0,1]]+np.abs(a)[...,[2,0,1]]*np.abs(b)[...,[1,2,0]]

def jacobian(c,w,b,p):
    cd=c['cdof'][w].astype(float);com=c['com'][w,int(c['root'][b])].astype(float)
    off=p.astype(float)-com;linear=cd[:,3:]+np.cross(cd[:,:3],off)
    scale=np.abs(cd[:,3:])+cross_size(cd[:,:3],off) if len(cd) else np.empty((0,3))
    # Include cancellation-sensitive point/COM subtraction and input conversion.
    scale+=cross_size(cd[:,:3],np.abs(p)+np.abs(com))
    mask=c['ancestor'][b,:c['nv']].astype(bool);linear[~mask]=0;angular=cd[:,:3].copy();angular[~mask]=0;scale[~mask]=0
    return linear.T,angular.T,(64*U*scale).T,64*U*np.abs(angular.T)

def reference(c,phase):
    n,nb,nv=2,c['nbody'],c['nv'];values={};errors={}
    if phase=='jacobian':
        p=c['initial_jacp'].astype(float).copy();r=c['initial_jacr'].astype(float).copy();ep=np.zeros_like(p);er=np.zeros_like(r)
        for w in range(n):
            jp,jr,bp,br=jacobian(c,w,int(c['body'][w]),c['point'][w])
            if c['jac_position']:p[w]=jp;ep[w]=bp
            if c['jac_rotation']:r[w]=jr;er[w]=br
        values.update(jacp=p,jacr=r);errors.update(jacp=ep,jacr=er)
    elif phase=='sensor':
        out=c['initial_sensor'].astype(float).copy();error=np.zeros_like(out)
        if not c['disabled']:
            for w in range(n):
                for i,t in enumerate(c['sensor_type']):
                    site=int(c['sensor_object'][i]);body=int(c['site_body'][site]);root=int(c['root'][body]);f=c['wrench'][w,body,3:].astype(float);v=c['wrench'][w,body,:3].astype(float);R=c['site_rotation'][w,site].astype(float)
                    scale=abs(f)
                    if t==4:v=f
                    else:
                        pos=c['site_position'][w,site].astype(float);com=c['com'][w,root].astype(float);v-=np.cross(pos-com,f);scale=abs(c['wrench'][w,body,:3])+cross_size(abs(pos)+abs(com),f)
                    e=128*U*(abs(R).T@scale);v=R.T@v;cut=float(c['cutoff'][i]);kind=int(c['sensor_datatype'][i])
                    if cut>0:
                        if kind==0:v=np.clip(v,-cut,cut)
                        elif kind==1:v=np.minimum(v,cut)
                    first=int(c['sensor_address'][i]);out[w,first:first+3]=v;error[w,first:first+3]=e
        values['sensor']=out;errors['sensor']=error
    else:
        out=c['initial_gravity'].astype(float).copy();scale=abs(out);error=np.zeros_like(out)
        for w in range(n):
            for b in range(1,nb):
                jp,_,ep,_=jacobian(c,w,b,c['position'][w,b]);g=c['gravity'][w%len(c['gravity'])].astype(float);mass=float(c['mass'][w%len(c['mass']),b]);coef=float(c['coefficient'][w%len(c['coefficient']),b]);f=-g*mass*coef
                out[w]+=jp.T@f;scale[w]+=abs(jp.T)@abs(f);error[w]+=ep.T@abs(f)
        error+=8*((nb+16)*U/(1-(nb+16)*U))*scale
        values['gravity']=out;errors['gravity']=error
    return values,errors
