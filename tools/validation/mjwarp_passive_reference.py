"""Independent FP64 passive algebra and input-derived FP32 error allowances."""
import numpy as np
from .mjwarp_motion_reference import U,G1,exact,add,multiply


def neg(a):return -a[0],a[1]


def norm(a):
    x,e=a;v=float(np.linalg.norm(x))
    # Euclidean norm is 1-Lipschitz; five rounded multiply/add/sqrt stages.
    return v,float(np.linalg.norm(e))+8*U*(v+float(np.linalg.norm(e)))


def divide(a,b):
    x,e=a;y,f=b;margin=abs(y)-f
    if margin<=0:raise ValueError('Reference denominator uncertainty includes zero')
    value=x/y
    return value,(e+abs(value)*f)/margin+G1*(abs(x)+e)/margin


def quaternion_difference(q,ref):
    length,le=norm(exact(q))
    if length>0:
        normalized=divide(exact(q),(length,le))
    else:normalized=exact(np.array([0.,0.,0.,1.]))
    x,xe=normalized;r=np.array([ref[0],-ref[1],-ref[2],-ref[3]],float)
    products=((0,[(0,0,1),(1,1,-1),(2,2,-1),(3,3,-1)]),
              (1,[(0,1,1),(1,0,1),(2,3,1),(3,2,-1)]),
              (2,[(0,2,1),(1,3,-1),(2,0,1),(3,1,1)]),
              (3,[(0,3,1),(1,2,1),(2,1,-1),(3,0,1)]))
    v=np.zeros(4);e=np.zeros(4)
    for k,terms in products:
        total=None
        for i,j,s in terms:
            term=multiply(exact(r[i]),(x[j],xe[j]));term=(s*term[0],term[1])
            total=term if total is None else add(total,term)
        v[k],e[k]=total
    sine,se=norm((v[1:],e[1:]));w,we=v[0],e[0]
    if sine==0:
        # The log map near either identity is bounded by axis perturbation.
        return np.zeros(3),np.full(3,4*np.linalg.norm(e[1:])/max(abs(w)-we,np.finfo(float).tiny))
    angle=2*np.arctan2(sine,w);wrapped=angle>np.pi
    speed=angle-2*np.pi if wrapped else angle
    radius2=sine*sine+w*w
    radius_margin=radius2-2*(abs(sine)*se+abs(w)*we)-se*se-we*we
    if radius_margin<=0:raise ValueError('Quaternion reference is not locally conditioned')
    ae=2*(abs(w)*se+abs(sine)*we)/radius_margin
    # Fixed transcendental allowance and rounded pi; independent of outputs.
    ae+=16*U*max(abs(angle),1.)
    if wrapped:ae+=4*U*np.pi
    expected=v[1:]*speed/sine
    if sine<=se:
        error=np.full(3,4*np.linalg.norm(e[1:])/max(abs(w)-we,.25)+ae)
    else:
        _,error=divide(multiply((v[1:],e[1:]),(speed,ae)),(sine,se))
    # The shortest-arc torque is discontinuous at pi. Include both branches
    # only if input roundoff itself straddles that boundary.
    if abs(angle-np.pi)<=ae:error+=2*np.pi
    return expected,error


def poly(linear,coeff,x):
    total=add(exact(linear),multiply(exact(coeff[0]),x))
    return add(total,multiply(multiply(exact(coeff[1]),x),x))


def joint(c):
    spring=np.zeros((2,c['nv']));damper=np.zeros_like(spring);se=np.zeros_like(spring);de=np.zeros_like(spring)
    for w in range(2):
        for j,kind in enumerate(c['joint_type']):
            d=int(c['dofadr'][j]);p=int(c['qposadr'][j]);width=(6,3,1,1)[kind]
            if not c['disable_damper']:
                for at in range(d,d+width):
                    gain=c['damping'][w%len(c['damping']),at];co=c['dampingpoly'][w%len(c['dampingpoly']),at]
                    if gain or np.any(co):
                        vel=exact(c['qvel'][w,at]);value=multiply(neg(vel),poly(gain,co,exact(abs(vel[0]))))
                        damper[w,at],de[w,at]=value
            gain=c['stiffness'][w%len(c['stiffness']),j];co=c['stiffnesspoly'][w%len(c['stiffnesspoly']),j]
            if c['disable_spring'] or not (gain or np.any(co)):continue
            ref=c['qpos_spring'][w%len(c['qpos_spring'])]
            if kind==0:
                delta=add(exact(c['qpos'][w,p:p+3]),neg(exact(ref[p:p+3])))
                value=multiply(neg(poly(gain,co,norm(delta))),delta)
                spring[w,d:d+3],se[w,d:d+3]=value;d+=3;p+=3
            if kind in (0,1):
                delta=quaternion_difference(c['qpos'][w,p:p+4],ref[p:p+4])
                value=multiply(neg(poly(gain,co,norm(delta))),delta)
                spring[w,d:d+3],se[w,d:d+3]=value
            else:
                delta=add(exact(c['qpos'][w,p]),neg(exact(ref[p])))
                spring[w,d],se[w,d]=multiply(neg(delta),poly(gain,co,delta))
    return {'qfrc_spring':spring,'qfrc_damper':damper},{'qfrc_spring':se,'qfrc_damper':de}


def combine(c,values=None,errors=None):
    if values is None:values={k:c[k] for k in ('qfrc_spring','qfrc_damper')}
    if errors is None:errors={k:np.zeros_like(v) for k,v in values.items()}
    v,e=add((values['qfrc_spring'],errors['qfrc_spring']),(values['qfrc_damper'],errors['qfrc_damper']))
    for w in range(2):
        for d in range(c['nv']):
            total=(v[w,d],e[w,d])
            if c['gravity_enabled'] and not c['actgravcomp'][c['dof_joint'][d]]:
                total=add(total,exact(c['qfrc_gravcomp'][w,d]))
            if c['has_fluid']:total=add(total,exact(c['qfrc_fluid'][w,d]))
            if c['has_adhesion']:total=add(total,exact(c['qfrc_adhesion'][w,d]))
            v[w,d],e[w,d]=total
    return {'qfrc_passive':v},{'qfrc_passive':e}


def reference(c,phase):
    if phase=='joint':return joint(c)
    if phase=='combine':return combine(c)
    v,e=joint(c);s,se=combine(c,v,e)
    return v|s,e|se
