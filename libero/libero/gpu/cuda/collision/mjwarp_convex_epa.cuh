// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Pinned collision_gjk.py: rigid GJK phase, polytope construction and EPA.
// Every invocation borrows a disjoint global scratch slice. No block barriers.
#pragma once
#include "mjwarp_convex_gjk.cuh"
#include <cstdio>
namespace flash::mjwarp {
struct ConvexPhase {
  bool needs_epa;
  float distance;
  CV x1,x2;
  ConvexGJKResult result;
  ConvexGeom geom1,geom2;
  bool require_enclosing=false;
};
__device__ inline bool convex_discrete(int a,int b) {return (a==6 || a==7) && (b==6 || b==7);}
__device__ inline ConvexPhase gjk_phase(float tolerance,float cutoff,int iterations,ConvexGeom a,ConvexGeom b,int type1,int type2,CV initial1,CV initial2) {
  CV size1=a.size,size2=b.size;float margin1=a.margin,margin2=b.margin,full1=0,full2=0,radius1=0,radius2=0;
  bool discrete=convex_discrete(type1,type2) && margin1==0 && margin2==0;
  if (type1==2 || type1==3) {radius1=a.size[0];full1=radius1+.5f*margin1;a.margin=0;a.size[0]=0;}
  if (type2==2 || type2==3) {radius2=b.size[0];full2=radius2+.5f*margin2;b.margin=0;b.size[0]=0;}
  if (radius1+radius2>0) {
    cutoff+=full1+full2;
    ConvexGJKResult r=gjk_distance(tolerance,iterations,a,b,initial1,initial2,type1,type2,cutoff,discrete);
    a.cached_index=r.index1;b.cached_index=r.index2;
    if (r.dist>tolerance) {
      // The original cutoff return leaves the shrunken geoms in its unused
      // output. Keep this state transition distinct from the inflated return.
      if (r.dist==convex_float_max) return {false,r.dist,r.x1,r.x2,{},a,b};
      CV normal=cnormalize(r.x2-r.x1),x1=r.x1,x2=r.x2;
      if (full1>0) x1=x1+full1*normal;
      if (full2>0) x2=x2-full2*normal;
      a.size=size1;a.margin=margin1;b.size=size2;b.margin=margin2;
      return {false,r.dist-(full1+full2),x1,x2,r,a,b};
    }
    // Deep penetration retries the original shapes and retains support caches.
    a.size=size1;a.margin=margin1;b.size=size2;b.margin=margin2;cutoff-=full1+full2;
  }
  ConvexGJKResult r=gjk_distance(tolerance,iterations,a,b,initial1,initial2,type1,type2,cutoff,discrete);
  a.cached_index=r.index1;b.cached_index=r.index2;
  // A closest point on the current simplex is an upper distance bound, not
  // a separation certificate. Discrete GJK can stagnate above tolerance for
  // shallow overlapping boxes. Let EPA resolve that inconclusive result,
  // but require its initial hull to enclose the origin before expansion.
  bool retry_box=type1==6 && type2==6 && discrete && r.dist>tolerance
      && r.dim>=2 && !r.separated;
  if (retry_box) {
    // Start one bounded refinement from the current witnesses instead of
    // repeating the original center direction. A smaller simplex-distance
    // bound or a separating support plane is useful new information. This
    // also repairs stagnation whose first simplex cannot seed an EPA hull.
    ConvexGJKResult refined=gjk_distance(tolerance,iterations,a,b,r.x1,r.x2,
        type1,type2,cutoff,discrete);
    if (refined.separated || (refined.dim>=2 && refined.dist<r.dist)) r=refined;
    a.cached_index=r.index1;b.cached_index=r.index2;
  }
  if ((r.dist>tolerance && !retry_box) || r.dim<2 || r.separated)
    return {false,r.dist,r.x1,r.x2,{},a,b};
  return {true,r.dist,r.x1,r.x2,r,a,b,retry_box};
}
struct ConvexPolytope {
  float *vertices,*projection,*norm2;
  int *vertex_index,*face,*horizon;
  int face_capacity,horizon_capacity,status,nvert,nface,nhorizon;
  CV center;
  __device__ CV vertex(int i) const {return cload(vertices,2*i)-cload(vertices,2*i+1);}
};
struct ConvexEPAResult {float distance;int ncontact;CV x1,x2;int face;};
__device__ inline CV triangle_affine(CV a,CV b,CV c,CV p) {
  float m14=simplex_minor(a,b,c,1,2),m24=simplex_minor(a,b,c,0,2),m34=simplex_minor(a,b,c,0,1),maximum;int i,j;
  if (fabsf(m14)>=fabsf(m24) && fabsf(m14)>=fabsf(m34)) {maximum=m14;i=1;j=2;}
  else if (fabsf(m24)>=fabsf(m34)) {maximum=m24;i=0;j=2;} else {maximum=m34;i=0;j=1;}
  return cv(simplex_area(p,b,c,i,j)/maximum,simplex_area(p,c,a,i,j)/maximum,simplex_area(p,a,b,i,j)/maximum);
}
__device__ inline bool triangle_point_intersect(CV a,CV b,CV c,CV p) {
  CV w=triangle_affine(a,b,c,p);
  if (w[0]<0 || w[1]<0 || w[2]<0) return false;
  CV projected=w[0]*a+w[1]*b+w[2]*c,d=projected-p;return sqrtf(cdot(d,d))<convex_minval;
}
__device__ inline bool tetra_same_side(CV a,CV b,CV c,CV d) {
  CV n=ccross(b-a,c-a);float one=cdot(n,d-a),two=cdot(n,-a);return (one>0 && two>0) || (one<0 && two<0);
}
__device__ inline bool tetra_contains_origin(CV a,CV b,CV c,CV d) {
  return tetra_same_side(a,b,c,d) && tetra_same_side(b,c,d,a) && tetra_same_side(c,d,a,b) && tetra_same_side(d,a,b,c);
}
__device__ inline void convex_store(float* dst,int i,CV value) {for (int k=0;k<3;++k) dst[3*i+k]=value[k];}
__device__ inline float epa_attach_face(ConvexPolytope& p,int index,int a,int b,int c) {
  if (p.nface==p.face_capacity) return 0;
  CV one=p.vertex(a),two=p.vertex(b),three=p.vertex(c);bool bad;CV r=project_origin_plane(three,two,one,bad);
  if (bad) return 0;
  if (cdot(r,one-p.center)<0) r=-r;
  p.face[index]=a+(b<<10)+(c<<20);convex_store(p.projection,index,r);p.norm2[index]=cdot(r,r);return p.norm2[index];
}
__device__ inline void epa_support(ConvexPolytope& p,int index,ConvexGeom a,ConvexGeom b,int type1,int type2,CV direction,int& cache1,int& cache2) {
  ConvexSupport one=convex_support(a,type1,direction),two=convex_support(b,type2,-direction);
  convex_store(p.vertices,2*index,one.point);convex_store(p.vertices,2*index+1,two.point);
  p.vertex_index[2*index]=one.vertex_index;p.vertex_index[2*index+1]=two.vertex_index;cache1=one.cached_index;cache2=two.cached_index;
}
__device__ inline void epa_initial_vertices(ConvexPolytope& p,const ConvexGJKResult& r,int count) {
  for (int i=0;i<count;++i) {
    convex_store(p.vertices,2*i,r.simplex1[i]);convex_store(p.vertices,2*i+1,r.simplex2[i]);
    p.vertex_index[2*i]=r.vertex1[i];p.vertex_index[2*i+1]=r.vertex2[i];
  }
}
__device__ inline void epa_replace_triangle(ConvexPolytope& p,ConvexGJKResult& r,int a,int b,int c) {
  int index[3]={a,b,c};
  for (int i=0;i<3;++i) {
    int j=index[i];r.simplex1[i]=cload(p.vertices,2*j);r.simplex2[i]=cload(p.vertices,2*j+1);r.simplex[i]=r.simplex1[i]-r.simplex2[i];r.vertex1[i]=p.vertex_index[2*j];r.vertex2[i]=p.vertex_index[2*j+1];
  }
  r.dim=3;
}
__device__ inline void polytope_segment(ConvexPolytope& p,ConvexGJKResult& r,ConvexGeom a,ConvexGeom b,int type1,int type2) {
  CV diff=r.simplex[1]-r.simplex[0];p.center=.5f*(r.simplex[0]+r.simplex[1]);int index=0;float value=convex_float_max;
  for (int i=0;i<3;++i) if (fabsf(diff[i])<value) {value=fabsf(diff[i]);index=i;}
  CV axis=cv(0,0,0);axis[index]=1;CV d1=ccross(axis,diff),u=convex_div(diff,sqrtf(cdot(diff,diff)));float rot[9];
  // The pinned 120-degree rotation, evaluated without trigonometry.
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) rot[3*i+j]=(i==j ? -.5f : 0)+1.5f*u[i]*u[j];
  rot[1]-=u[2]*.86602540378f;rot[2]+=u[1]*.86602540378f;rot[3]+=u[2]*.86602540378f;rot[5]-=u[0]*.86602540378f;rot[6]-=u[1]*.86602540378f;rot[7]+=u[0]*.86602540378f;
  CV directions[3]={d1,crot(rot,d1),{}};directions[2]=crot(rot,directions[1]);epa_initial_vertices(p,r,2);int cache1,cache2;
  for (int i=0;i<3;++i) epa_support(p,i+2,a,b,type1,type2,convex_div(directions[i],sqrtf(cdot(directions[i],directions[i]))),cache1,cache2);
  const int faces[6][3]={{0,2,3},{0,4,2},{0,3,4},{1,3,2},{1,2,4},{1,4,3}};
  for (int i=0;i<6;++i) if (epa_attach_face(p,i,faces[i][0],faces[i][1],faces[i][2])<1.e-10f) {p.status=-1;epa_replace_triangle(p,r,faces[i][0],faces[i][1],faces[i][2]);return;}
  CV v2=p.vertex(2),v3=p.vertex(3),v4=p.vertex(4),v1=r.simplex[0],base=r.simplex[1]-v1;
  float one=simplex_det(v2-v1,v3-v1,base),two=simplex_det(v3-v1,v4-v1,base),three=simplex_det(v4-v1,v2-v1,base);
  if (!((one>=0 && two>=0 && three>=0) || (one<=0 && two<=0 && three<=0))) {p.status=1;return;}
  p.nvert=5;p.nface=6;p.status=0;
}
__device__ inline void polytope_triangle(ConvexPolytope& p,ConvexGJKResult& r,ConvexGeom a,ConvexGeom b,int type1,int type2,bool& require_enclosing) {
  p.center=(r.simplex[0]+r.simplex[1]+r.simplex[2])*(1.0f/3.0f);CV n=ccross(r.simplex[1]-r.simplex[0],r.simplex[2]-r.simplex[0]);float norm=sqrtf(cdot(n,n));
  if (norm<convex_minval) {p.status=2;return;}n=convex_div(n,norm);epa_initial_vertices(p,r,3);int cache1,cache2;
  epa_support(p,3,a,b,type1,type2,-n,cache1,cache2);epa_support(p,4,a,b,type1,type2,n,cache1,cache2);
  CV v1=r.simplex[0],v2=r.simplex[1],v3=r.simplex[2],v4=p.vertex(3),v5=p.vertex(4);
  if (triangle_point_intersect(v1,v2,v3,v4)) {p.status=3;return;}
  if (triangle_point_intersect(v1,v2,v3,v5)) {p.status=4;return;}
  if (r.dist>1.e-5f && !tetra_contains_origin(v1,v2,v3,v4) && !tetra_contains_origin(v1,v2,v3,v5)) {p.status=5;return;}
  const int faces[6][3]={{4,0,1},{4,2,0},{4,1,2},{3,1,0},{3,0,2},{3,2,1}};
  // The inherited 1e-10 squared face-distance cutoff rejects valid hulls
  // with less than 10um penetration. This also affects ordinary below-tolerance
  // box queries, not only inconclusive GJK retries. Retain positive faces for
  // zero-margin boxes, then certify ALL half-spaces before EPA if a face used
  // this exception. Ordinary hulls above the cutoff need no extra certificate.
  bool eligible_box=type1==6 && type2==6 && a.margin==0 && b.margin==0;
  for (int i=0;i<6;++i) {
    float distance2=epa_attach_face(p,i,faces[i][0],faces[i][1],faces[i][2]);
    if (require_enclosing || eligible_box) {
      if (!(distance2>0)) {p.status=6+i;return;}
      if (distance2<1.e-10f) require_enclosing=true;
    } else if (distance2<1.e-10f) {p.status=6+i;return;}
  }
  p.nvert=5;p.nface=6;p.status=0;
}
__device__ inline void polytope_tetrahedron(ConvexPolytope& p,ConvexGJKResult& r) {
  p.center=.25f*(r.simplex[0]+r.simplex[1]+r.simplex[2]+r.simplex[3]);epa_initial_vertices(p,r,4);float dist[4];int closest=0;
  const int faces[4][3]={{0,1,2},{0,3,1},{0,2,3},{3,2,1}};
  for (int i=0;i<4;++i) {
    dist[i]=epa_attach_face(p,i,faces[i][0],faces[i][1],faces[i][2]);
    if (dist[i]<1.e-17f) {p.status=-1;epa_replace_triangle(p,r,faces[i][0],faces[i][1],faces[i][2]);return;}
    if (i==1) closest=dist[0]<dist[1] ? 0 : 1;
    else if (i>1 && dist[i]<dist[closest]) closest=i;
  }
  if (!tetra_contains_origin(r.simplex[0],r.simplex[1],r.simplex[2],r.simplex[3])) {
    if (dist[closest]>convex_minval) {p.status=12;return;}
    p.status=-1;epa_replace_triangle(p,r,faces[closest][0],faces[closest][1],faces[closest][2]);return;
  }
  p.nvert=4;p.nface=4;p.status=0;
}
__device__ inline int epa_add_edge(ConvexPolytope& p,int a,int b) {
  int n=p.nhorizon;if (n<0) return -1;int edge=(min(a,b)<<10)|max(a,b);
  for (int i=0;i<n;++i) if (edge==p.horizon[i]) {p.horizon[i]=p.horizon[n-1];return n-1;}
  if (n==p.horizon_capacity) return -1;p.horizon[n]=edge;return n+1;
}
__device__ inline void epa_face_edges(ConvexPolytope& p,int face) {
  int a=face&0x3ff,b=(face>>10)&0x3ff,c=(face>>20)&0x3ff;
  p.nhorizon=epa_add_edge(p,a,b);p.nhorizon=epa_add_edge(p,b,c);p.nhorizon=epa_add_edge(p,c,a);
}
__device__ inline bool epa_deleted(int face) {return (unsigned(face)&0x80000000u)!=0;}
__device__ inline bool epa_invalid(int face) {return (unsigned(face)&0xc0000000u)!=0;}
__device__ inline ConvexEPAResult epa_witness(const ConvexPolytope& p,int face) {
  int packed=p.face[face],a=packed&0x3ff,b=(packed>>10)&0x3ff,c=(packed>>20)&0x3ff;
  CV w=triangle_affine(p.vertex(a),p.vertex(b),p.vertex(c),cload(p.projection,face));
  CV x1=w[0]*cload(p.vertices,2*a)+w[1]*cload(p.vertices,2*b)+w[2]*cload(p.vertices,2*c);
  CV x2=w[0]*cload(p.vertices,2*a+1)+w[1]*cload(p.vertices,2*b+1)+w[2]*cload(p.vertices,2*c+1);
  return {-sqrtf(p.norm2[face]),1,x1,x2,face};
}
__device__ inline void epa_horizon_overflow(const ConvexPolytope& p,bool warn,int world,int* overflow) {
  if (warn) printf("Warning: EPA horizon = %d isn't large enough.\nTo disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.EPA_HORIZON (or = 0 for all)\n",p.horizon_capacity);
  atomicOr(overflow+world,1<<8);
}
__device__ inline ConvexEPAResult epa_expand(float tolerance,int iterations,ConvexPolytope& p,ConvexGeom a,ConvexGeom b,int type1,int type2,bool discrete,bool warn,int world,int* overflow) {
  float upper=convex_float_max,upper2=convex_float_max,epsilon=discrete ? 1.e-7f : tolerance;int index=-1,nvalid=p.nface;
  iterations=min(iterations,1000);
  for (int iteration=0;iteration<iterations;++iteration) {
    int previous=index;index=-1;float lower2=convex_float_max;
    for (int i=0;i<p.nface;++i) if (!epa_invalid(p.face[i]) && p.norm2[i]<lower2) {index=i;lower2=p.norm2[i];}
    if (lower2>upper2 || index<0) {index=previous;break;}
    if (lower2<=0) break;
    float lower=sqrtf(lower2);int wi=p.nvert,cache1,cache2;CV pr=cload(p.projection,index);
    epa_support(p,wi,a,b,type1,type2,convex_div(pr,lower),cache1,cache2);CV w=p.vertex(wi);a.cached_index=cache1;b.cached_index=cache2;++p.nvert;
    float next=cdot(pr,w)/lower;if (next<upper) {upper=next;upper2=upper*upper;}
    if (upper-lower<epsilon) break;
    if (discrete) {
      bool repeated=false;
      for (int i=0;i<p.nvert-1;++i) if (p.vertex_index[2*i]==p.vertex_index[2*wi] && p.vertex_index[2*i+1]==p.vertex_index[2*wi+1]) {repeated=true;break;}
      if (repeated) break;
    }
    --nvalid;p.face[index]=int(unsigned(p.face[index])|0x80000000u);epa_face_edges(p,p.face[index]);
    if (p.nhorizon==-1) {epa_horizon_overflow(p,warn,world,overflow);index=-1;break;}
    for (int i=0;i<p.nface;++i) {
      if (epa_deleted(p.face[i])) continue;
      if (cdot(cload(p.projection,i),w)-p.norm2[i]>1.e-10f) {
        if (!epa_invalid(p.face[i])) --nvalid;
        p.face[i]=int(unsigned(p.face[i])|0x80000000u);epa_face_edges(p,p.face[i]);
        if (p.nhorizon==-1) {epa_horizon_overflow(p,warn,world,overflow);index=-1;break;}
      }
    }
    for (int i=0;i<p.nhorizon;++i) {
      int edge=p.horizon[i];float dist2=epa_attach_face(p,p.nface,wi,edge&0x3ff,(edge>>10)&0x3ff);
      if (dist2==0) {index=-1;break;}++p.nface;
      if (dist2>=lower2 && dist2<=upper2) ++nvalid;
      else p.face[p.nface-1]=int(unsigned(p.face[p.nface-1])|0x40000000u);
    }
    if (nvalid==0 || index==-1) break;p.nhorizon=0;
  }
  if (index>=0) return epa_witness(p,index);
  return {convex_float_max,0,cv(0,0,0),cv(0,0,0),-1};
}
__device__ inline bool convex_multicontact_supported(const ConvexGeom& a,const ConvexGeom& b,int type1,int type2) {
  if (a.margin!=0 || b.margin!=0) return false;
  bool face1=type1==6 || type1==7 || type1==5,face2=type2==6 || type2==7 || type2==5;
  return (face1 && face2) || (type1==3 && face2) || (face1 && type2==3);
}
// Only inconclusive box retries need this certificate. Evaluate half-space
// signs in FP64 on the actual FP32 hull vertices: no distance clamp, enlarged
// collision tolerance, shared storage or additional support queries.
__device__ inline bool epa_encloses_origin(const ConvexPolytope& p) {
  if (p.nvert<4 || p.nface<4) return false;
  for (int i=0;i<p.nface;++i) {
    int packed=p.face[i];
    CV a=p.vertex(packed&1023),b=p.vertex((packed>>10)&1023),c=p.vertex((packed>>20)&1023);
    double u[3],v[3],n[3];
    for (int k=0;k<3;++k) {u[k]=double(b[k])-double(a[k]);v[k]=double(c[k])-double(a[k]);}
    n[0]=u[1]*v[2]-u[2]*v[1];n[1]=u[2]*v[0]-u[0]*v[2];n[2]=u[0]*v[1]-u[1]*v[0];
    double origin=0,center=0;
    for (int k=0;k<3;++k) {origin-=n[k]*double(a[k]);center+=n[k]*(double(p.center[k])-double(a[k]));}
    if (center==0 || (origin>0 && center<0) || (origin<0 && center>0)) return false;
  }
  return true;
}
__device__ inline ConvexEPAResult epa_phase(float tolerance,int iterations,ConvexGJKResult r,ConvexGeom a,ConvexGeom b,int type1,int type2,ConvexPolytope& p,bool warn,int world,int* overflow,bool require_enclosing=false) {
  p.nface=0;p.nvert=0;p.nhorizon=0;p.status=0;
  if (r.dim==2) polytope_segment(p,r,a,b,type1,type2);
  else if (r.dim==4) polytope_tetrahedron(p,r);
  if (r.dim==3) polytope_triangle(p,r,a,b,type1,type2,require_enclosing);
  if (p.status) return {r.dist,1,r.x1,r.x2,-1};
  if (require_enclosing && !epa_encloses_origin(p)) return {r.dist,1,r.x1,r.x2,-1};
  bool discrete=convex_discrete(type1,type2) && a.margin==0 && b.margin==0;
  ConvexEPAResult result=epa_expand(tolerance,iterations,p,a,b,type1,type2,discrete,warn,world,overflow);
  if (!convex_multicontact_supported(a,b,type1,type2)) result.face=-1;
  return result;
}
__device__ inline int gjk_simplex_to_epa(const ConvexGJKResult& r,ConvexPolytope& p) {
  for (int i=0;i<3;++i) {
    int k=r.dim>i ? i : 0;convex_store(p.vertices,2*i,r.simplex1[k]);convex_store(p.vertices,2*i+1,r.simplex2[k]);p.vertex_index[2*i]=r.vertex1[k];p.vertex_index[2*i+1]=r.vertex2[k];
  }
  p.face[0]=(1<<10)+(2<<20);return 0;
}
}
