"""FP64 inertia-box fluid / Jacobian projection with propagated FP32 allowances."""
import numpy as np
from .mjwarp_motion_reference import U,G1,exact,add,multiply,cross
from .mjwarp_passive_reference import divide,joint
MINVAL=float(np.float32(1.e-15))


def neg(a):return -a[0],a[1]
def absolute(a):return abs(a[0]),a[1]
def dot(a,b):
    total=exact(0.)
    for k in range(len(a[0])):total=add(total,multiply((a[0][k],a[1][k]),(b[0][k],b[1][k])))
    return total

def matvec(mat,vector):
    out=np.empty(3);err=np.empty(3)
    for k in range(3):out[k],err[k]=dot(exact(mat[k]),vector)
    return out,err

def clamp_min(a):
    # Compare shared real mathematics to CPU's FP64 literal, and include the
    # independently known FP32 conversion of that literal in GPU allowances.
    floor=1.e-15;low=min(floor,MINVAL);high=max(floor,MINVAL)
    x,e=a;v=max(floor,x)
    return v,max(v-max(low,x-e),max(high,x+e)-v)

def sqrt(a):
    x,e=a;v=np.sqrt(x);lo=np.sqrt(max(0.,x-e));hi=np.sqrt(max(0.,x+e))
    return v,max(v-lo,hi-v)+G1*hi

def power(a,n):
    x,e=a;v=x**n;lo=max(0.,x-e)**n;hi=(x+e)**n
    # 32u covers rounding plus the documented CUDA powf <=8-ULP allowance
    # for normal finite results; see NVIDIA mathematical-functions appendix.
    return v,max(v-lo,hi-v)+32*U*hi


def body_forces(c):
    nb=c['nbody'];out=np.zeros((2,nb,6));oe=np.zeros_like(out)
    pi=(np.pi,G1*np.pi)
    for w in range(2):
        density=float(c['density'][w%len(c['density'])]);viscosity=float(c['viscosity'][w%len(c['viscosity'])])
        wind=c['wind'][w%len(c['wind'])].astype(float)
        for b in range(1,nb):
            mass=float(c['body_mass'][w%len(c['body_mass']),b])
            if mass<MINVAL:continue
            rotation=c['ximat'][w,b].astype(float);velocity=c['cvel'][w,b].astype(float)
            delta=add(exact(c['xipos'][w,b]),neg(exact(c['subtree_com'][w,c['root'][b]])))
            angular=matvec(rotation.T,exact(velocity[:3]))
            linear=matvec(rotation.T,add(exact(velocity[3:]),neg(cross(delta,exact(velocity[:3])))))
            if np.any(wind):linear=add(linear,neg(matvec(rotation.T,exact(wind))))
            box=[]
            if viscosity>0 or density>0:
                inertia=c['body_inertia'][w%len(c['body_inertia']),b].astype(float);scale=divide(exact(6.),exact(mass))
                for k in range(3):
                    i,j=((1,2),(0,2),(0,1))[k]
                    combination=add(add(exact(inertia[i]),exact(inertia[j])),exact(-inertia[k]))
                    box.append(sqrt(multiply(clamp_min(combination),scale)))
            force=[exact(0.) for _ in range(3)];torque=[exact(0.) for _ in range(3)]
            if viscosity>0:
                diameter=divide(add(add(box[0],box[1]),box[2]),exact(3.));diam3=power(diameter,3)
                for k in range(3):
                    av=(angular[0][k],angular[1][k]);lv=(linear[0][k],linear[1][k])
                    torque[k]=multiply(multiply(multiply(neg(av),diam3),pi),exact(viscosity))
                    force[k]=multiply(multiply(multiply(multiply(exact(-3.),lv),diameter),pi),exact(viscosity))
            if density>0:
                p4=[power(x,4) for x in box];scale=divide(exact(density),exact(64.))
                for k in range(3):
                    i,j=((1,2),(0,2),(0,1))[k];lv=(linear[0][k],linear[1][k]);av=(angular[0][k],angular[1][k])
                    drag=multiply(multiply(multiply(multiply(multiply(exact(.5*density),box[i]),box[j]),absolute(lv)),lv),exact(1.))
                    force[k]=add(force[k],neg(drag))
                    drag=multiply(multiply(multiply(multiply(box[k],add(p4[i],p4[j])),absolute(av)),av),scale)
                    torque[k]=add(torque[k],neg(drag))
            pair=lambda terms:(np.array([x[0] for x in terms]),np.array([x[1] for x in terms]))
            out[w,b,:3],oe[w,b,:3]=matvec(rotation,pair(force))
            out[w,b,3:],oe[w,b,3:]=matvec(rotation,pair(torque))
    return out,oe


def projection(c,force,error):
    out=np.zeros((2,c['nv']));oe=np.zeros_like(out)
    for w in range(2):
        for d in range(c['nv']):
            motion=c['cdof'][w,d].astype(float);body=int(c['dof_body'][d]);com=c['subtree_com'][w,c['root'][body]];total=exact(0.)
            for b in range(body,body+int(c['subtree_size'][body])):
                f,fe=force[w,b],error[w,b]
                if not np.any(f) and not np.any(fe):continue
                offset=add(exact(c['xipos'][w,b]),neg(exact(com)))
                shift=cross(exact(motion[:3]),offset)
                jac=exact(np.r_[motion[3:],motion[:3]])
                term=add(dot(jac,(f,fe)),dot(shift,(f[:3],fe[:3])))
                total=add(total,term)
            out[w,d],oe[w,d]=total
    return out,oe


def reference(c,phase):
    values={};errors={}
    if phase in ('force','composed','passive'):
        force,error=body_forces(c);values['body_force']=force;errors['body_force']=error
    else:force=c['body_force'].astype(float);error=np.zeros_like(force)
    if phase!='force':
        values['qfrc_fluid'],errors['qfrc_fluid']=projection(c,force,error)
    if phase=='passive':
        jv,je=joint(c);total,te=add((jv['qfrc_spring'],je['qfrc_spring']),(jv['qfrc_damper'],je['qfrc_damper']))
        # This isolated composition disables gravity/adhesion. Their caller
        # ordering/holds are separately covered by host and production checks.
        total,te=add((total,te),(values['qfrc_fluid'],errors['qfrc_fluid']))
        values.update(jv,qfrc_passive=total);errors.update(je,qfrc_passive=te)
    return values,errors
