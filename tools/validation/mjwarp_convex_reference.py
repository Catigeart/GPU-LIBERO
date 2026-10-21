"""FP64 rigid support equations, exhaustive mesh oracle and exact cache mapping."""
import numpy as np
U=np.finfo(np.float32).eps/2


def support(c,i):
    w,g=int(c['world'][i]),int(c['geom'][i]);kind=int(c['geom_type'][g]);size=c['size'][w%len(c['size']),g].astype(float);R=c['xmat'][w,g].astype(float);p=c['xpos'][w,g].astype(float);direction=c['direction'][i].astype(float);local=R.T@direction;margin=float(c['request_margin'][i]);res=np.zeros(3);projection=None;gap=None
    if kind==2:point=p+(size[0]+.5*margin)*direction
    elif kind==3:
        res=local*size[0];res[2]+=np.sign(local[2])*size[1];point=R@res+p
    elif kind==4:
        res=local*size;norm=np.linalg.norm(res);res=res/norm if norm else np.zeros(3);point=R@(res*size)+p
    elif kind==5:
        length=np.linalg.norm(local[:2]);res[:2]=local[:2]*(size[0]/length) if length>1.e-15 else 0.;res[2]=size[1] if local[2]>=0 else -size[1];point=R@res+p
    elif kind==6:point=R@(np.sign(local)*size)+p
    else:
        mesh=int(c['geom_dataid'][w%len(c['geom_dataid']),g]);adr,num=int(c['mesh_vertadr'][mesh]),int(c['mesh_vertnum'][mesh]);values=c['mesh_vert'][adr:adr+num].astype(float)@local;index=int(np.argmax(values));point=R@c['mesh_vert'][adr+index].astype(float)+p
        # The support gap distinguishes unique from nearly tied exposed
        # vertices; no ranking decision is imposed inside the FP32 interval.
        top=np.partition(values,-2)[-2:] if num>=2 else np.array([-np.inf,values[0]]);gap=float(max(top)-min(top));projection=float(np.dot(p,direction)+values[index])
    if kind!=2 and margin>0:point+=direction*(.5*margin)
    if projection is None:projection=float(np.dot(direction,point))
    elif margin>0:projection+=np.dot(direction,direction)*(.5*margin)
    scale=1.+np.max(abs(p))+np.max(abs(size))+abs(margin)+np.max(abs(point));gamma=128*U/(1-128*U);budget=8*gamma*scale*(1+np.linalg.norm(direction))*(1+np.max(abs(R)))**2
    return dict(point=point,projection=projection,budget=budget,kind=kind,size=size,R=R,pos=p,direction=direction,local=local,margin=margin,gap=gap)


def check(c,i,point,cached,vertex,ref,*,require_indexed_corner=False):
    kind=ref['kind'];budget=ref['budget'];point=np.asarray(point,float);checks=[];value=float(np.dot(point,ref['direction']));projection_error=abs(value-ref['projection']);checks.append(dict(name='independent support projection',passed=projection_error<=budget,maximum_error=projection_error))
    if kind in (2,4):
        error=float(np.max(abs(point-ref['point'])));checks.append(dict(name='smooth support point',passed=error<=budget,maximum_error=error));checks.append(dict(name='unindexed primitive markers',passed=cached==-1 and vertex==-1))
    elif kind==3:
        base=ref['pos']+ref['R']@(ref['local']*ref['size'][0])
        if ref['margin']>0:base+=ref['direction']*(.5*ref['margin'])
        endpoint=ref['R'].T@(point-base)
        checks.append(dict(name='capsule exposed segment endpoint',passed=bool(np.linalg.norm(endpoint[:2])<=budget and abs(endpoint[2])<=ref['size'][1]+budget)))
        checks.append(dict(name='unindexed primitive markers',passed=cached==-1 and vertex==-1))
    elif kind==5:
        # At an exactly axial/tiny radial direction the chosen cap/point is
        # nonunique; still preserve legal cap index and absent mesh cache.
        checks.append(dict(name='cylinder cap/cache metadata',passed=cached==-1 and vertex in (0,1)))
        shifted=point-ref['pos']-(ref['direction']*(.5*ref['margin']) if ref['margin']>0 else 0);local=ref['R'].T@shifted
        checks.append(dict(name='cylinder surface point',passed=bool(np.linalg.norm(local[:2])<=ref['size'][0]+budget and abs(abs(local[2])-ref['size'][1])<=budget)))
        if ref['size'][1]>budget:checks.append(dict(name='cylinder cap id matches point',passed=bool((local[2]>=0 and vertex==0) or (local[2]<0 and vertex==1))))
    elif kind==6:
        shifted=point-ref['pos']-(ref['direction']*(.5*ref['margin']) if ref['margin']>0 else 0);local=ref['R'].T@shifted
        checks.append(dict(name='box support within closed slabs',passed=bool(np.all(abs(local)<=ref['size']+budget))))
        bits=True
        for k in range(3):
            if local[k]>budget:bits &=bool(vertex&(1<<k))
            if local[k]<-budget:bits &=not bool(vertex&(1<<k))
        checks.append(dict(name='box vertex bits match exposed signs',passed=bool(bits and cached==-1 and 0<=vertex<=7)))
        if require_indexed_corner:
            # A zero support component permits any point on the exposed face,
            # but EPA's feature recovery requires the *indexed corner*. The
            # pinned sign(0) point is not that corner; projection alone cannot
            # detect this point/feature metadata inconsistency.
            corner=np.array([ref['size'][k] if vertex&(1<<k) else -ref['size'][k] for k in range(3)])
            expected=ref['R']@corner+ref['pos']
            if ref['margin']>0:expected+=ref['direction']*(.5*ref['margin'])
            error=float(np.max(abs(point-expected)))
            checks.append(dict(name='box support equals its indexed corner',passed=bool(0<=vertex<=7 and error<=budget),maximum_error=error))
    else:
        w,g=int(c['world'][i]),int(c['geom'][i]);mesh=int(c['geom_dataid'][w%len(c['geom_dataid']),g]);adr,num=int(c['mesh_vertadr'][mesh]),int(c['mesh_vertnum'][mesh]);graph=int(c['mesh_graphadr'][mesh]);valid=0<=vertex<num
        if graph<0 or num<10:valid &=cached==adr+vertex
        else:
            n=int(c['mesh_graph'][graph]);valid &=0<=cached<n
            if 0<=cached<n:valid &=int(c['mesh_graph'][graph+2+n+cached])==vertex
        checks.append(dict(name='mesh local vertex and mode-specific cache relation',passed=bool(valid)))
        if 0<=vertex<num:
            expected=ref['R']@c['mesh_vert'][adr+vertex].astype(float)+ref['pos']
            if ref['margin']>0:expected+=ref['direction']*(.5*ref['margin'])
            error=float(np.max(abs(expected-point)));checks.append(dict(name='mesh point matches indexed vertex',passed=error<=budget,maximum_error=error))
    for item in checks:item['passed']=bool(item['passed'])
    return checks
