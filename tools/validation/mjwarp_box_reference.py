"""Independent convex distance/SAT and contact witness residuals, FP64."""
import numpy as np
from .mjwarp_plane_reference import material
U=np.finfo(np.float32).eps/2


def box_separation(p1, p2, R1, R2, size1, size2):
    """Independent SAT on actual transformed corners, including rounded R.

    CPU2.3.7 mjc_BoxBox supplies the face/cross-axis geometry. Cross the
    actual edge columns for face normals instead of assuming an exact
    orthogonal inverse for an FP32 rotation matrix.
    """
    p1, p2, R1, R2, size1, size2 = (np.asarray(v, np.float64)
                                   for v in (p1, p2, R1, R2, size1, size2))
    axes = [np.cross(R[:, i], R[:, (i + 1) % 3]) for R in (R1, R2) for i in range(3)]
    axes += [np.cross(R1[:, i], R2[:, j]) for i in range(3) for j in range(3)]
    axes = np.array([v / np.linalg.norm(v) for v in axes if np.linalg.norm(v) > 1.e-14])
    if not len(axes):
        raise ValueError('Degenerate box edge basis')
    gaps = abs(axes @ (p2 - p1)) - abs(axes @ R1) @ size1 - abs(axes @ R2) @ size2
    return float(max(gaps))


def segment_box_distance(pos,half,size):
    """Exact piecewise-quadratic segment-to-AABB closest distance."""
    cuts=[-1.,1.]
    for k in range(3):
        if half[k]:cuts.extend(float((s*size[k]-pos[k])/half[k]) for s in (-1,1) if -1<(s*size[k]-pos[k])/half[k]<1)
    cuts=sorted(set(cuts));best=np.inf
    for left,right in zip(cuts[:-1],cuts[1:]):
        middle=pos+half*((left+right)*.5);active=abs(middle)>size
        direction=half[active];offset=pos[active]-np.sign(middle[active])*size[active];denom=np.dot(direction,direction)
        t=np.clip(-np.dot(direction,offset)/denom,left,right) if denom else left
        for x in (left,float(t),right):
            delta=np.maximum(abs(pos+half*x)-size,0.);best=min(best,float(np.dot(delta,delta)))
    return np.sqrt(best)


def signed_point_box(point,size):
    delta=abs(point)-size
    return np.linalg.norm(np.maximum(delta,0)) if np.any(delta>0) else np.max(delta)


def pair_math(c,w,g1,g2):
    p1,p2=c['xpos'][w,[g1,g2]].astype(float);R1,R2=c['xmat'][w,[g1,g2]].astype(float);size1=c['size'][w%len(c['size']),g1].astype(float);size2=c['size'][w%len(c['size']),g2].astype(float);kind=int(c['geom_type'][g1]);scale=1.+np.max(np.abs([p1,p2]))+np.max(np.abs([size1,size2]));gamma=512*U/(1-512*U);budget=8*gamma*scale*(1+np.max(np.abs([R1,R2])))**3
    info=dict(kind=kind,p1=p1,p2=p2,R1=R1,R2=R2,size1=size1,size2=size2,budget=budget)
    if kind in (2,3):
        local=R2.T@(p1-p2);half=R2.T@R1[:,2]*size1[1] if kind==3 else np.zeros(3)
        closest=segment_box_distance(local,half,size2) if kind==3 else max(0.,signed_point_box(local,size2))
        info.update(closest=closest-size1[0],inside=closest<=budget,axis=R1[:,2],half_length=size1[1] if kind==3 else 0.)
    else:
        axes=[*R1.T,*R2.T]
        axes += [np.cross(a,b) for a in R1.T for b in R2.T if np.dot(np.cross(a,b),np.cross(a,b))>=1.e-7]
        axes=np.array([v/np.linalg.norm(v) for v in axes]);separation=abs(axes@(p2-p1))-abs(axes@R1)@size1-abs(axes@R2)@size2
        info.update(axes=axes,separation=float(max(separation)))
    return info


def geometry_checks(info,dist,pos,frame,id):
    """Check geometry of both engines, independently of their manifolds/order."""
    kind=info['kind'];budget=info['budget'];normal=np.asarray(frame,float)[0];pos=np.asarray(pos,float);dist=float(dist);checks=[]
    if dist>=1.e9:return [dict(name='absent capsule component sentinel',passed=kind==3 and id==1 and dist==1.e10)]
    norm=np.linalg.norm(normal);checks.append(dict(name='unit contact normal',passed=bool(abs(norm-1)<=budget)))
    if kind in (2,3):
        radius=info['size1'][0];center=pos-normal*(radius+.5*dist);axis=info['axis'];offset=center-info['p1'];t=np.dot(offset,axis)/max(np.dot(axis,axis),1.e-30);transverse=offset-t*axis
        checks.append(dict(name='sphere center on finite capsule centerline',passed=bool(np.linalg.norm(transverse)<=budget and abs(t)<=info['half_length']+budget)))
        local=info['R2'].T@(center-info['p2']);expected=signed_point_box(local,info['size2'])-radius
        checks.append(dict(name='signed point-box distance minus sphere radius',passed=bool(abs(expected-dist)<=budget)))
        if id==0 and not info['inside']:checks.append(dict(name='first contact attains independent segment-box minimum',passed=bool(abs(info['closest']-dist)<=budget)))
    else:
        align=np.max(abs(info['axes']@normal));checks.append(dict(name='normal is a box face or edge-cross SAT direction',passed=bool(abs(align-1)<=budget)))
        separation=np.dot(normal,info['p2']-info['p1'])-np.dot(abs(info['R1'].T@normal),info['size1'])-np.dot(abs(info['R2'].T@normal),info['size2'])
        checks.append(dict(name='contact depth bounded by supporting planes',passed=bool(dist>=separation-budget)))
    # For a midpoint contact, reversing half the signed distance along n gives
    # witnesses on the two surfaces. Capsule surface checked via reconstructed
    # center; both boxes checked by their local closed slabs and nearest face.
    for box,sgn in ((1,-1),(2,1)) if kind==6 else ((2,1),):
        point=pos+sgn*.5*dist*normal;local=info['R'+str(box)].T@(point-info['p'+str(box)]);delta=abs(local)-info['size'+str(box)]
        checks.append(dict(name='box'+str(box)+' surface witness',passed=bool(np.max(delta)<=budget and np.min(abs(delta))<=budget)))
    return checks


def materials(c,w,g1,g2,pair):return material(c,w,g1,g2,pair)
