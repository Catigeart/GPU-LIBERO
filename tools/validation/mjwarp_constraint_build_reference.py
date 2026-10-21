"""Independent row/Jacobian algebra and componentwise FP32 roundoff bounds."""
import numpy as np
from .mjwarp_motion_reference import U,G1,exact,add,multiply,cross


def neg(a):return -a[0],a[1]
def constant(value):return np.float64(value),abs(float(np.float32(value))-value)
def interval(a):return a[0]-a[1],a[0]+a[1]
def enclosing(value,lo,hi):return value,np.maximum(abs(value-lo),abs(hi-value))
def maximum(a,b):
    lo1,hi1=interval(a);lo2,hi2=interval(b)
    return enclosing(np.maximum(a[0],b[0]),np.maximum(lo1,lo2),np.maximum(hi1,hi2))
def minimum(a,b):return neg(maximum(neg(a),neg(b)))
def clamp(a,lo,hi):return minimum(maximum(a,lo),hi)
def divide(a,b):
    x,ex=a;y,ey=b;lower=np.maximum(abs(y)-ey,np.finfo(float).tiny)
    value=x/y;error=(abs(y)*ex+abs(x)*ey)/(abs(y)*lower)+G1*(abs(x)+ex)/lower
    return value,error
def power(a,b):
    lo,hi=interval(a);plo,phi=interval(b);lo=max(0.,float(lo));hi=max(lo,float(hi))
    values=[lo**plo,lo**phi,hi**plo,hi**phi]
    value=float(a[0])**float(b[0]);lower=min(values);upper=max(values)
    # Same conservative32u local powf allowance as the accepted fluid oracle.
    return value,max(abs(value-lower),abs(upper-value))+32*U*max(abs(lower),abs(upper))
def broadcast(c,key,w):return c[key][w%len(c[key])]


def kbi(c,w,ref,imp,position):
    time=exact(float(ref[0]));damping=exact(float(ref[1]))
    if not c['disable_refsafe']:time=maximum(time,multiply(exact(2.),exact(broadcast(c,'timestep',w))))
    lo=clamp(exact(imp[0]),constant(.0001),constant(.9999));hi=clamp(exact(imp[1]),constant(.0001),constant(.9999))
    width=maximum(constant(1.e-15),exact(imp[2]));mid=clamp(exact(imp[3]),constant(.0001),constant(.9999));p=maximum(exact(1.),exact(imp[4]))
    if ref[0]<=0:k=divide(exact(-float(ref[0])),maximum(constant(1.e-15),multiply(hi,hi)))
    else:k=divide(exact(1.),maximum(constant(1.e-15),multiply(multiply(multiply(multiply(multiply(hi,hi),time),time),damping),damping)))
    if ref[1]<=0:b=divide(exact(-float(ref[1])),maximum(constant(1.e-15),hi))
    else:b=divide(exact(2.),maximum(constant(1.e-15),multiply(hi,time)))
    x=divide((abs(position[0]),position[1]),width)
    if lo[0]==hi[0] or imp[2]<=1.e-15:result=multiply(exact(.5),add(lo,hi))
    elif x[0]<=0:result=lo
    elif x[0]>=1:result=hi
    elif p[0]==1:result=add(lo,multiply(x,add(hi,neg(lo))))
    elif x[0]<=mid[0]:
        poly=multiply(divide(exact(1.),power(mid,add(p,exact(-1.)))),power(x,p))
        result=clamp(add(lo,multiply(poly,add(hi,neg(lo)))),lo,hi)
    else:
        poly=add(exact(1.),neg(multiply(divide(exact(1.),power(add(exact(1.),neg(mid)),add(p,exact(-1.)))),power(add(exact(1.),neg(x)),p))))
        result=clamp(add(lo,multiply(poly,add(hi,neg(lo)))),lo,hi)
    # Crossing polynomial pieces can only move impedance within its clamped
    # range. This input-derived branch allowance is used only at uncertain
    # endpoints/midpoint; ordinary cases retain the tighter propagated bound.
    xl,xh=interval(x);ml,mh=interval(mid)
    if (xl<0<xh) or (xl<1<xh) or (xl<=mh and xh>=ml and abs(x[0]-mid[0])<=x[1]+mid[1]):
        result=(result[0],max(result[1],abs(result[0]-interval(lo)[0]),abs(interval(hi)[1]-result[0])))
    return k,b,result


def row(c,w,position,imp_position,weight,ref,imp,margin,velocity,friction,kind):
    k,b,i=kbi(c,w,ref,imp,imp_position)
    if kind==1:k=exact(0.)
    D=divide(exact(1.),maximum(multiply(divide(weight,i),add(exact(1.),neg(i))),constant(1.e-15)))
    # Grouping follows the real equation; individual FP32 associations are
    # covered by the componentwise bounds and the fixed8x accepted safety.
    aref=add(neg(multiply(multiply(k,i),position)),neg(multiply(b,velocity)))
    return dict(pos=add(position,exact(margin)),margin=exact(margin),D=D,vel=velocity,aref=aref,frictionloss=exact(friction))


def contact_jacobian(c,w,contact,component):
    g=c['contact_geom'][contact];body1,body2=c['geom_body'][g]
    point=exact(c['contact_pos'][contact]);frame=c['contact_frame'][contact]
    value=np.zeros(c['padded_nv']);error=np.zeros_like(value)
    for d in range(c['nv']):
        vectors=[]
        for body in (body1,body2):
            if not c['ancestor'][body,d]:vectors.append((exact(np.zeros(3)),exact(np.zeros(3))));continue
            angular=exact(c['cdof'][w,d,:3]);linear=exact(c['cdof'][w,d,3:])
            offset=add(point,neg(exact(c['subtree_com'][w,c['root'][body]])))
            vectors.append((add(linear,cross(angular,offset)),angular))
        linear=add(vectors[1][0],neg(vectors[0][0]));angular=add(vectors[1][1],neg(vectors[0][1]))
        def dot(vector,direction):
            total=exact(0.)
            for j in range(3):total=add(total,multiply((vector[0][j],vector[1][j]),exact(frame[direction,j])))
            return total
        if c['cone']==1:result=dot(linear if component<3 else angular,component%3)
        else:
            result=dot(linear,0)
            if c['contact_dim'][contact]>1:
                direction=component//2+1;f=float(c['contact_friction'][contact,direction-1])*(1.-2.*float(component&1))
                result=add(result,multiply(dot(linear if direction<3 else angular,direction%3),exact(f)))
        value[d],error[d]=result
    velocity=float(np.dot(value[:c['nv']],c['qvel'][w].astype(float)))
    products=abs(value[:c['nv']]*c['qvel'][w])+error[:c['nv']]*abs(c['qvel'][w])
    # Covers CTA/warp trees and pinned tiled partial atomic reductions alike.
    gamma=(c['nv']+64)*U/(1-(c['nv']+64)*U)
    ve=float(np.dot(error[:c['nv']],abs(c['qvel'][w])))+gamma*float(products.sum())
    return (value,error),(velocity,ve)


def build_reference(c):
    rows=[{},{}];counts=np.zeros((2,4),int);contacts={}
    if c['disable_constraint']:return rows,counts,contacts
    for w in range(2):
        if not c['disable_friction']:
            for d,friction in enumerate(broadcast(c,'dof_friction',w)):
                if friction<=0:continue
                J=np.zeros(c['padded_nv']);J[d]=1.;key=(1,d,0)
                rows[w][key]=dict(J=(J,np.zeros_like(J)),**row(c,w,exact(0.),exact(0.),exact(broadcast(c,'dof_weight',w)[d]),
                    broadcast(c,'dof_ref',w)[d],broadcast(c,'dof_imp',w)[d],0.,exact(c['qvel'][w,d]),float(friction),1))
                counts[w,1]+=1;counts[w,3]+=1
        if not c['disable_limit']:
            for j in c['limited_joint']:
                bounds=broadcast(c,'joint_range',w)[j];q=c['qpos'][w,c['joint_qpos'][j]];margin=broadcast(c,'joint_margin',w)[j]
                lower=add(exact(q),exact(-float(bounds[0])));upper=add(exact(bounds[1]),exact(-float(q)))
                # Discrete activation and side selection follow the actual
                # FP32 input expression, independently of numerical budgets.
                l32=np.float32(q-bounds[0]);u32=np.float32(bounds[1]-q)
                if np.float32(min(l32,u32)-margin)>=0:continue
                position=add(lower if l32<u32 else upper,exact(-float(margin)));sign=1. if l32<u32 else -1.;d=c['joint_dof'][j]
                J=np.zeros(c['padded_nv']);J[d]=sign
                rows[w][(3,int(j),0)]=dict(J=(J,np.zeros_like(J)),**row(c,w,position,position,exact(broadcast(c,'dof_weight',w)[d]),
                    broadcast(c,'joint_ref',w)[j],broadcast(c,'joint_imp',w)[j],float(margin),exact(sign*float(c['qvel'][w,d])),0.,3))
                counts[w,2]+=1;counts[w,3]+=1
    if c['disable_contact']:return rows,counts,contacts
    for contact in range(min(c['naconmax'],int(c['contact_count'][0]))):
        if not c['contact_type'][contact]&1:continue
        position=add(exact(c['contact_dist'][contact]),exact(-float(c['contact_margin'][contact])))
        if not (np.float32(c['contact_dist'][contact]-c['contact_margin'][contact])<0 or (c['adhesion_enabled'] and c['contact_adhesion'][contact]!=0)):continue
        w=int(c['contact_world'][contact]);dim=int(c['contact_dim'][contact]);n=dim if c['cone']==1 or dim==1 else 2*(dim-1)
        contacts[contact]=(w,n);counts[w,3]+=n
        b1,b2=c['geom_body'][c['contact_geom'][contact]];weights=broadcast(c,'body_weight',w)
        weight=add(exact(weights[b1,0]),exact(weights[b2,0]));ratio=exact(broadcast(c,'impratio_invsqrt',w));friction=c['contact_friction'][contact]
        for component in range(n):
            J,velocity=contact_jacobian(c,w,contact,component);scale=exact(1.);iw=weight
            pos=position
            if c['cone']==1 and component>0:
                scale=multiply(ratio,ratio)
                if component>1:scale=multiply(scale,divide(multiply(exact(friction[0]),exact(friction[0])),multiply(exact(friction[component-1]),exact(friction[component-1]))))
                pos=exact(0.)
            elif c['cone']==0 and dim>1:
                iw=add(weight,multiply(multiply(exact(friction[0]),exact(friction[0])),weight))
                for factor in (2.,friction[0],friction[0],ratio[0],ratio[0]):iw=multiply(iw,exact(factor))
            kind=5 if dim==1 else 7 if c['cone']==1 else 6
            result=row(c,w,pos,position,iw,c['contact_ref'][contact],c['contact_imp'][contact],float(c['contact_margin'][contact]),velocity,0.,kind)
            if c['cone']==1 and component>0:
                result['D']=divide(result['D'],scale)
                ref=c['contact_ref_friction'][contact]
                if not np.any(ref):ref=c['contact_ref'][contact]
                b=kbi(c,w,ref,c['contact_imp'][contact],position)[1]
                result['aref']=neg(multiply(b,velocity))
            if c['adhesion_enabled'] and c['contact_adhesion'][contact]!=0 and (component==0 or c['cone']==0):
                adhesion=exact(c['contact_adhesion'][contact])
                if c['cone']==0 and dim>1:adhesion=divide(adhesion,exact(float(2*(dim-1))))
                result['aref']=add(result['aref'],multiply(divide(exact(1.),result['D']),adhesion))
            rows[w][(kind,contact,component)]=dict(J=J,Jqvel=velocity,**result)
    return rows,counts,contacts
