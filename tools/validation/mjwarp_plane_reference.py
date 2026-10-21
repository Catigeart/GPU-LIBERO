"""Independent FP64 plane manifolds, material equations and FP32 budgets."""
import numpy as np
U=np.finfo(np.float32).eps/2


def normalize(v):
    n=np.linalg.norm(v);return v/n if n else np.zeros(3)
def frame(n):
    n=normalize(n);b=np.array([0.,1.,0.]) if -.5<n[1]<.5 else np.array([0.,0.,1.]);b=normalize(b-n*np.dot(n,b))
    if np.linalg.norm(n)==0:b[:]=0
    return np.array([n,b,np.cross(n,b)])
def parameter(c,key,w,index):return c[key][w%len(c[key]),index].astype(np.float64)


def material(c,w,g1,g2,pair):
    if pair>=0:
        p={key:parameter(c,'pair_'+key,w,pair) for key in ('solref','solreffriction','solimp','friction','margin','gap','adhesion')};p['dim']=int(c['pair_dim'][pair])
    else:
        p={key:parameter(c,key,w,g1)+parameter(c,key,w,g2) for key in ('margin','gap')};p['solreffriction']=np.zeros(2)
        a,b=(parameter(c,'solmix',w,g) for g in (g1,g2));p1,p2=c['priority'][[g1,g2]];f1,f2=(parameter(c,'friction',w,g) for g in (g1,g2));a1,a2=(parameter(c,'adhesion',w,g) for g in (g1,g2))
        if p1!=p2:
            pick=0 if p1>p2 else 1;mix=1.-pick;p['dim']=int(c['condim'][[g1,g2][pick]]);f=(f1,f2)[pick];p['adhesion']=(a1,a2)[pick]
        else:
            if a<1.e-15 and b<1.e-15:mix=.5
            elif a<1.e-15:mix=0.
            elif b<1.e-15:mix=1.
            else:mix=a/(a+b)
            p['dim']=int(max(c['condim'][g1],c['condim'][g2]));f=np.maximum(f1,f2);p['adhesion']=a1+a2
        p['friction']=f[[0,0,1,2,2]];r1,r2=(parameter(c,'solref',w,g) for g in (g1,g2));i1,i2=(parameter(c,'solimp',w,g) for g in (g1,g2))
        p['solref']=mix*r1+(1-mix)*r2 if r1[0]>0 and r2[0]>0 else np.minimum(r1,r2);p['solimp']=mix*i1+(1-mix)*i2
    p['friction']=np.maximum(p['friction'],1.e-5);return p


def mesh_manifold(c,mesh,n,origin,center,R,margin):
    if mesh<0:return []
    adr,num=int(c['mesh_vertadr'][mesh]),int(c['mesh_vertnum'][mesh]);v=c['mesh_vert'][adr:adr+num].astype(float);local=R.T@n
    support=-v@local;best=int(np.argmax(support));world=center+R@v[best];dist=np.dot(n,world-origin);out=[(dist,world-.5*dist*n)]
    polyadr,polynum=int(c['mesh_polyadr'][mesh]),int(c['mesh_polynum'][mesh])
    if polynum<=0 or dist>margin:return out
    begin,count=int(c['mesh_polymapadr'][adr+best]),int(c['mesh_polymapnum'][adr+best]);faces=c['mesh_polymap'][begin:begin+count]
    dots=c['mesh_polynormal'][polyadr+faces].astype(float)@local;valid=dots<1.
    if not np.any(valid):return out
    face=polyadr+int(faces[int(np.argmin(dots))]);begin,count=int(c['mesh_polyvertadr'][face]),int(c['mesh_polyvertnum'][face]);indices=c['mesh_polyvert'][begin:begin+count];fv=v[indices]
    matches=np.flatnonzero(indices==best);a=int(matches[0]) if len(matches) else 0;b,cid,d=(a+1)%count,(a+2)%count,(a+3)%count
    def area(b,cid,d):return .5*np.linalg.norm(np.cross(fv[a]-fv[cid],fv[b]-fv[d]))
    if count>4:
        current=area(b,cid,d)
        while area(b,cid,(d+1)%count)>current:
            d=(d+1)%count;current=area(b,cid,d)
            while area(b,(cid+1)%count,d)>current:cid=(cid+1)%count;current=area(b,cid,d)
            while area((b+1)%count,cid,d)>current:b=(b+1)%count;current=area(b,cid,d)
    selected=[a,b,cid,d][:min(count,4)]
    for i in selected[1:]:
        world=center+R@fv[i]
        if np.dot(n,world-center)>0:continue
        dist=np.dot(n,world-origin);out.append((dist,world-.5*dist*n))
    return out


def manifold(c,w,g1,g2,p):
    n=c['xmat'][w,g1,:,2].astype(float);origin=c['xpos'][w,g1].astype(float);center=c['xpos'][w,g2].astype(float);R=c['xmat'][w,g2].astype(float);size=parameter(c,'size',w,g2);kind=c['geom_type'][g2];f=frame(n)
    def sphere(center):
        dist=np.dot(center-origin,n)-size[0];return dist,center-n*(size[0]+.5*dist)
    if kind==2:out=[sphere(center)]
    elif kind==3:
        axis=R[:,2];b=axis-n*np.dot(n,axis);length=np.linalg.norm(b);b=b/length if length else b
        if length<.5:b=np.array([0.,1.,0.]) if -.5<n[1]<.5 else np.array([0.,0.,1.])
        f=np.array([n,b,np.cross(n,b)]);out=[sphere(center+axis*size[1]),sphere(center-axis*size[1])]
    elif kind==4:
        point=center+R@(-normalize((R.T@n)*size)*size);dist=np.dot(n,point-origin);out=[(dist,point-n*dist*.5)]
    elif kind==5:
        axis=R[:,2].copy();prj=np.dot(n,axis)
        if prj>0:axis=-axis;prj=-prj
        dist0=np.dot(center-origin,n);radial=axis*prj-n;length2=np.dot(radial,radial);radial=radial*(size[0]/np.sqrt(length2)) if length2>=1.e-12 else np.array([size[0],0.,0.]);rp=np.dot(radial,n);axis*=size[1];prj*=size[1]
        d1,d2,d3=dist0+prj+rp,dist0-prj+rp,dist0+prj-rp*.5;lateral=normalize(np.cross(radial,axis))*(size[0]*np.sqrt(3)*.5)
        out=[(d1,center+radial+axis-n*d1*.5),(d2,center+radial-axis-n*d2*.5),(d3,center+lateral+axis-radial*.5-n*d3*.5),(d3,center-lateral+axis-radial*.5-n*d3*.5)]
    elif kind==6:
        out=[];cd=np.dot(center-origin,n)
        for k in range(8):
            corner=R@(size*np.array([1. if k&(1<<j) else -1. for j in range(3)]));d=cd+np.dot(n,corner);out.append((d,corner+center-.5*n*d))
    elif kind==7:out=mesh_manifold(c,int(c['geom_dataid'][w%len(c['geom_dataid']),g2]),n,origin,center,R,float(p['margin']+p['gap']))
    else:return []
    count={2:1,3:2,4:1,5:4,6:8,7:4}[kind];out+= [(1.e10,np.zeros(3))]*(count-len(out));return [(d,x,f) for d,x in out]


def reference(c):
    rows={};must=set();possible=set();ng=c['ngeom'];cap=c['capacity'];n=int(c['candidate_count'][0]);initial=int(c['initial_count'][0])
    # 256 elementary FP32 operations cover a geometry/frame output chain;
    # fixed8x accounts for library sqrt/normalization and composed transforms.
    # Discrete support/face choices are checked separately against the pinned
    # implementation; these fixtures choose their nondegenerate active domain.
    gamma=256*U/(1-256*U)
    for t in range(min(n,cap)):
        g1,g2=map(int,c['candidate_geom'][t]);w=int(c['candidate_world'][t]);pair,sensor=map(int,c['candidate_ids'][t])
        if c['geom_type'][g1]!=0 or not 2<=c['geom_type'][g2]<=7:continue
        p=material(c,w,g1,g2,pair);scale=1.+np.max(abs(c['xpos'][w,[g1,g2]]))+np.max(abs(parameter(c,'size',w,g2)))
        if c['geom_type'][g2]==7 and c['geom_dataid'][w%len(c['geom_dataid']),g2]>=0:scale+=np.max(abs(c['mesh_vert'])) if c['mesh_vert'].size else 0.
        geom_budget=8*gamma*scale*(1+np.max(abs(c['xmat'][w,[g1,g2]])))**3
        for id,(dist,pos,f) in enumerate(manifold(c,w,g1,g2,p)):
            key=(w,g1,g2,id);detected=dist<float(p['margin']+p['gap']);uncertain=abs(dist-float(p['margin']+p['gap']))<=geom_budget
            emitted=(pair!=-2 and detected) or sensor>=0
            if emitted and (not uncertain or sensor>=0):must.add(key)
            if emitted or (pair!=-2 and uncertain):possible.add(key)
            dim=1 if p['adhesion']!=0 and dist>=p['margin'] else p['dim'];kind=(1 if pair>=-1 and detected else 0)|(2 if sensor>=0 else 0)
            row=dict(p,dist=dist,pos=pos,frame=f,dim=dim,geom=np.array([g1,g2]),worldid=w,type=kind,geomcollisionid=id,includemargin=p['margin']);row.pop('margin');row.pop('gap');row['efc_address']=np.full(c['columns'],-1,np.int32)
            rows[key]=(row,geom_budget,uncertain)
    return dict(rows=rows,must=must,possible=possible,initial=initial)
