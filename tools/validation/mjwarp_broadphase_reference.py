"""Vectorized independent FP64 bounds for broadphase inequality decisions."""
import numpy as np
from .mjwarp_motion_reference import exact,add,multiply


def neg(a):return -a[0],a[1]
def absolute(a):return abs(a[0]),a[1]
def component(a,k):return a[0][...,k],a[1][...,k]
def dot(a,b):
    total=exact(np.zeros(a[0].shape[:-1]))
    for k in range(3):total=add(total,multiply(component(a,k),component(b,k)))
    return total
def rotate(matrix,vector):
    result=[];errors=[]
    for k in range(3):
        value,error=dot(exact(matrix[:,k,:]),vector);result.append(value);errors.append(error)
    return np.stack(result,axis=1),np.stack(errors,axis=1)
def bound_extreme(a,b,maximum):
    operation=np.maximum if maximum else np.minimum
    value=operation(a[0],b[0]);lo=operation(a[0]-a[1],b[0]-b[1]);hi=operation(a[0]+a[1],b[0]+b[1])
    return value,np.maximum(abs(value-lo),abs(hi-value))
def conjunction(residuals):
    values=np.stack([x[0] for x in residuals],axis=1);errors=np.stack([x[1] for x in residuals],axis=1)
    return np.all(values-8*errors>=0,axis=1),np.all(values+8*errors>=0,axis=1)
def row(c,key,w):return c[key][w%len(c[key])]


def geometry(c,w):
    p=c['pairs_in'];g1,g2=p[:,0],p[:,1]
    x1,x2=c['xpos'][w,g1].astype(float),c['xpos'][w,g2].astype(float)
    R1,R2=c['xmat'][w,g1].astype(float),c['xmat'][w,g2].astype(float)
    box=row(c,'aabb',w);b1,b2=box[g1].astype(float),box[g2].astype(float)
    r1,r2=row(c,'rbound',w)[g1].astype(float),row(c,'rbound',w)[g2].astype(float)
    m1=add(exact(row(c,'margin',w)[g1]),exact(row(c,'gap',w)[g1]));m2=add(exact(row(c,'margin',w)[g2]),exact(row(c,'gap',w)[g2]))
    explicit=c['pair_ids_in'][:,0]>=0
    if np.any(explicit):
        id=c['pair_ids_in'][explicit,0];pm=add(exact(row(c,'pair_margin',w)[id]),exact(row(c,'pair_gap',w)[id]))
        m1[0][explicit]=pm[0];m1[1][explicit]=pm[1];m2[0][explicit]=0.;m2[1][explicit]=0.
    return g1,g2,x1,x2,R1,R2,b1,b2,r1,r2,m1,m2


def classify(c,w):
    n=len(c['pairs_in']);must=np.ones(n,bool);possible=np.ones(n,bool);mask=c['filter']
    if n==0:return must,possible
    g1,g2,x1,x2,R1,R2,b1,b2,r1,r2,m1,m2=geometry(c,w)
    plane=(r1==0)|(r2==0);margin=add(m1,m2)
    if mask&1:
        first=r1==0;delta=add(exact(np.where(first[:,None],x2,x1)),neg(exact(np.where(first[:,None],x1,x2))))
        axis=np.where(first[:,None],R1[:,:,2],R2[:,:,2]);radius=exact(np.where(first,r2,r1))
        sure,maybe=conjunction([add(add(radius,margin),neg(dot(delta,exact(axis))))]);must[plane]&=sure[plane];possible[plane]&=maybe[plane]
    if mask&2:
        delta=add(exact(x2),neg(exact(x1)));bound=add(add(add(exact(r1),exact(r2)),m1),m2)
        sure,maybe=conjunction([add(multiply(bound,bound),neg(dot(delta,delta)))]);must[~plane]&=sure[~plane];possible[~plane]&=maybe[~plane]
    center1=add(rotate(R1,exact(b1[:,0])),exact(x1));center2=add(rotate(R2,exact(b2[:,0])),exact(x2))
    if mask&4:
        count=len(g1);minimum1=exact(np.full((count,3),1.e10));minimum2=exact(np.full((count,3),1.e10));maximum1=exact(np.full((count,3),-1.e10));maximum2=exact(np.full((count,3),-1.e10))
        for x in (-1.,1.):
            for y in (-1.,1.):
                for z in (-1.,1.):
                    sign=exact(np.array([x,y,z]));corner1=rotate(R1,multiply(exact(b1[:,1]),sign));corner2=rotate(R2,multiply(exact(b2[:,1]),sign))
                    minimum1=bound_extreme(minimum1,corner1,False);maximum1=bound_extreme(maximum1,corner1,True)
                    minimum2=bound_extreme(minimum2,corner2,False);maximum2=bound_extreme(maximum2,corner2,True)
        residuals=[]
        for axis in range(3):
            residuals.append(add(add(add(component(center1,axis),component(maximum1,axis)),margin),neg(add(component(center2,axis),component(minimum2,axis)))))
            residuals.append(add(add(add(component(center2,axis),component(maximum2,axis)),margin),neg(add(component(center1,axis),component(minimum1,axis)))))
        sure,maybe=conjunction(residuals);must[~plane]&=sure[~plane];possible[~plane]&=maybe[~plane]
    if mask&8:
        normals=[R1[:,:,k] for k in range(3)]+[R2[:,:,k] for k in range(3)];residuals=[]
        for normal in normals:
            projection1=dot(center1,exact(normal));projection2=dot(center2,exact(normal));radii=[]
            for matrix,size in ((R1,b1[:,1]),(R2,b2[:,1])):
                radius=exact(np.zeros(n))
                for axis in range(3):radius=add(radius,absolute(multiply(exact(size[:,axis]),dot(exact(matrix[:,:,axis]),exact(normal)))))
                radii.append(radius)
            residuals.append(add(add(add(radii[0],radii[1]),margin),neg(absolute(add(projection2,neg(projection1))))))
        sure,maybe=conjunction(residuals);must[~plane]&=sure[~plane];possible[~plane]&=maybe[~plane]
    sensor=c['pair_ids_in'][:,1]>=0;must[sensor]=True;possible[sensor]=True
    if len(c['geom_dataid'])>1:
        dataid=row(c,'geom_dataid',w);missing=((c['geom_type'][g1]==7)&(dataid[g1]<0))|((c['geom_type'][g2]==7)&(dataid[g2]<0))
        must[missing]=False;possible[missing]=False
    return must,possible


def reference(c):
    must=[];possible=[];ambiguous=0
    for w in range(2):
        yes,maybe=classify(c,w);ambiguous+=int(np.count_nonzero(yes!=maybe))
        for i,(g1,g2) in enumerate(c['pairs_in']):
            if c['geom_type'][g1]>c['geom_type'][g2]:g1,g2=g2,g1
            value=(w,int(g1),int(g2),int(c['pair_ids_in'][i,0]),int(c['pair_ids_in'][i,1]))
            if yes[i]:must.append(value)
            if maybe[i]:possible.append(value)
    return set(must),set(possible),ambiguous
