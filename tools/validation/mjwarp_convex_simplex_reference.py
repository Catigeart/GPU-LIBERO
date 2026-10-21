"""Independent FP64 convex-QP enumeration and projection/KKT residuals."""
import itertools
import numpy as np
U=np.finfo(np.float32).eps/2


def closest(vertices):
    """Enumerate affinely independent faces; no signed-volume port reused."""
    vertices=np.asarray(vertices,float);best=np.inf;point=None;weights=None
    for size in range(1,len(vertices)+1):
        for subset in itertools.combinations(range(len(vertices)),size):
            v=vertices[list(subset)];base=v[-1]
            if size==1:w=np.ones(1)
            else:
                A=(v[:-1]-base).T;solution,residual,rank,singular=np.linalg.lstsq(A,-base,rcond=None)
                if rank<size-1:continue
                w=np.r_[solution,1-sum(solution)]
                if min(w)<-128*np.finfo(float).eps*(1+np.max(abs(w))):continue
            x=w@v;distance=float(np.dot(x,x))
            if distance<best:
                best=distance;point=x;weights=np.zeros(len(vertices));weights[list(subset)]=w
    if point is None:raise RuntimeError('Convex QP enumeration has no vertex candidate')
    return dict(point=point,distance=np.sqrt(best),weights=weights)


def check(simplex,count,got):
    s=np.asarray(simplex,float);n=int(count);w=np.asarray(got['weights'],float);p=np.asarray(got['point'],float);distance=float(got['distance']);ref=closest(s[:n]);mass=1+np.sum(abs(w));scale=1+np.max(abs(s[:n]));gamma=512*U/(1-512*U);budget=8*gamma*mass*scale
    checks=[dict(name='finite weights/point/distance',passed=bool(np.all(np.isfinite(w)) and np.all(np.isfinite(p)) and np.isfinite(distance))),
        dict(name='barycentric feasibility',passed=bool(np.all(w[:n]>=-8*gamma*mass) and abs(sum(w[:n])-1)<=8*gamma*mass and np.all(w[n:]==0))),
        dict(name='exact kept mask from candidate zero pattern',passed=int(got['mask'])==sum((1<<i) for i in range(4) if w[i]!=0))]
    point_error=float(np.max(abs(p-w[:n]@s[:n])));norm_error=abs(distance-np.linalg.norm(p));optimal_error=abs(distance-ref['distance']);kkt=np.min((s[:n]-p)@p)
    checks += [dict(name='point matches candidate barycentric combination',passed=point_error<=budget,maximum_error=point_error),dict(name='distance matches candidate point norm',passed=norm_error<=budget,maximum_error=norm_error),
        dict(name='independent convex QP minimum distance',passed=optimal_error<=budget,maximum_error=float(optimal_error),budget=float(budget)),dict(name='projection variational inequality',passed=bool(kkt>=-budget*(1+np.linalg.norm(p)+np.max(abs(s[:n])))),minimum_kkt=float(kkt))]
    for c in checks:c['passed']=bool(c['passed'])
    return checks
