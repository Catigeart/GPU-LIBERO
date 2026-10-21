// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Pinned convex box-box multicontact. General MULTICCD is a separate domain.
#pragma once
#include "mjwarp_convex_epa.cuh"
namespace flash::mjwarp {
struct ConvexManifold {int count;CV one[4],two[4];float distance[4];};
struct BoxFeature {int dim,index[3];CV point[3];};
struct BoxNormals {int count,index[3];CV normal[3];};
__device__ inline BoxFeature box_feature(const ConvexPolytope& p,int face,int side) {
  int ids[3]={face&1023,(face>>10)&1023,(face>>20)&1023};BoxFeature f{};
  for (int i=0;i<3;++i) {f.index[i]=p.vertex_index[2*ids[i]+side];f.point[i]=cload(p.vertices,2*ids[i]+side);}
  if (f.index[0]!=f.index[1]) f.dim=(f.index[2]==f.index[0] || f.index[2]==f.index[1]) ? 2 : 3;
  else {f.index[1]=f.index[2];f.point[1]=f.point[2];f.dim=f.index[0]!=f.index[2] ? 2 : 1;}
  return f;
}
__device__ inline CV box_face_normal(int index) {CV n=cv(0,0,0);n[index/2]=(index%2)==0 ? 1 : -1;return n;}
__device__ inline BoxNormals box_normals_near(const ConvexGeom& g,CV direction) {
  BoxNormals n{};CV local=cnormalize(crotT(g.rot,direction));
  for (int i=0;i<6;++i) if (cdot(local,box_face_normal(i))>0.99605101346969604f) {n.count=1;n.normal[0]=crot(g.rot,box_face_normal(i));n.index[0]=i;return n;}
  return n;
}
__device__ inline BoxNormals box_normals(const ConvexGeom& g,const BoxFeature& f,CV direction) {
  BoxNormals n{};
  if (f.dim==3) {
    CV common=cv(0,0,0);int count=0;
    for (int k=0;k<3;++k) {
      int bit=1<<k;float value=float((f.index[0]&bit) && (f.index[1]&bit) && (f.index[2]&bit))-float(!(f.index[0]&bit) && !(f.index[1]&bit) && !(f.index[2]&bit));common[k]=value;
      if (value!=0) n.index[count++]=2*k;
    }
    if (common[0]+common[1]+common[2]==-1) ++n.index[0];
    if (count==1) {n.count=1;n.normal[0]=crot(g.rot,common);return n;}
    return box_normals_near(g,direction);
  }
  if (f.dim==2) {
    for (int k=0;k<3;++k) {
      int bit=1<<k;float value=float((f.index[0]&bit) && (f.index[1]&bit))-float(!(f.index[0]&bit) && !(f.index[1]&bit));
      if (value!=0) {int i=n.count++;CV local=cv(0,0,0);local[k]=value;n.normal[i]=crot(g.rot,local);n.index[i]=2*k+(value>0 ? 0 : 1);}
    }
    if (n.count==1 || n.count==2) return n;
    return box_normals_near(g,direction);
  }
  if (f.dim==1) {
    n.count=3;
    for (int k=0;k<3;++k) {int i=2*k+((f.index[0]&(1<<k)) ? 0 : 1);n.index[k]=i;n.normal[k]=crot(g.rot,box_face_normal(i));}
  }
  return n;
}
__device__ inline BoxNormals box_edge_normals(const ConvexGeom& g,const BoxFeature& f,CV* end) {
  BoxNormals n{};
  if (f.dim==2) {end[0]=f.point[1];n.normal[0]=cnormalize(f.point[1]-f.point[0]);n.count=1;}
  if (f.dim==1) {
    CV local;
    for (int k=0;k<3;++k) local[k]=(f.index[0]&(1<<k)) ? g.size[k] : -g.size[k];
    for (int k=0;k<3;++k) {CV point=local;point[k]=-point[k];end[k]=crot(g.rot,point)+g.pos;n.normal[k]=cnormalize(end[k]-f.point[0]);}n.count=3;
  }
  return n;
}
__device__ inline int box_face(const ConvexGeom& g,int face,CV* vertices) {
  const int indices[6][4]={{7,3,1,5},{2,6,4,0},{2,3,7,6},{4,5,1,0},{6,7,5,4},{3,2,0,1}};
  for (int i=0;i<4;++i) {CV point;for (int k=0;k<3;++k) point[k]=(indices[face][i]&(1<<k)) ? g.size[k] : -g.size[k];vertices[i]=crot(g.rot,point)+g.pos;}
  return 4;
}
__device__ inline float convex_quad_area(CV a,CV b,CV c,CV d) {CV n=ccross(a-c,b-d);return .5f*sqrtf(cdot(n,n));}
__device__ inline void convex_polygon_quad(const CV* p,int count,int* result) {
  int b=1,c=2,d=3;result[0]=0;result[1]=b;result[2]=c;result[3]=d;float area=convex_quad_area(p[0],p[b],p[c],p[d]);
  for (int a=0;a<count;++a) {
    while (true) {
      float next=convex_quad_area(p[a],p[b],p[c],p[(d+1)%count]);if (next<=area) break;
      area=next;d=(d+1)%count;result[0]=a;result[1]=b;result[2]=c;result[3]=d;
      while (true) {next=convex_quad_area(p[a],p[b],p[(c+1)%count],p[d]);if (next<=area) break;area=next;c=(c+1)%count;result[0]=a;result[1]=b;result[2]=c;result[3]=d;}
      while (true) {next=convex_quad_area(p[a],p[(b+1)%count],p[c],p[d]);if (next<=area) break;area=next;b=(b+1)%count;result[0]=a;result[1]=b;result[2]=c;result[3]=d;}
    }
    if (b==a) {b=(b+1)%count;if (c==b) {c=(c+1)%count;if (d==c) d=(d+1)%count;}}
  }
}
__device__ inline ConvexManifold box_polygon_clip(const CV* face1,int n1,const CV* face2,int n2,CV normal,CV direction) {
  ConvexManifold result{};if (n1<3) return result;
  CV normals[4],storage1[8],storage2[8];float plane[4];CV *polygon=storage1,*clipped=storage2;
  for (int i=0;i<n1;++i) {normals[i]=ccross(face1[(i+1)%n1]-face1[i],(face1[i]+normal)-face1[i]);plane[i]=cdot(normals[i],face1[i]);}
  int count=n2;for (int i=0;i<n2;++i) polygon[i]=face2[i];
  for (int edge=0;edge<n1;++edge) {
    int nclipped=0;
    for (int i=0;i<count;++i) {
      CV p=polygon[i],q=polygon[(i+1)%count];bool inside1=cdot(p-face1[edge],normals[edge])>-1.e-10f,inside2=cdot(q-face1[edge],normals[edge])>-1.e-10f;
      if (!inside1 && !inside2) continue;
      if (inside1 && inside2) {clipped[nclipped++]=q;continue;}
      float denominator=cdot(normals[edge],q-p),t=fabsf(denominator)<1.e-10f ? convex_float_max : (plane[edge]-cdot(normals[edge],p))/denominator;
      if (t>-3.e-7f && t<1+3.e-7f) {t=fminf(1,fmaxf(0,t));clipped[nclipped++]=p+t*(q-p);}
      if (inside2) clipped[nclipped++]=q;
    }
    CV* tmp=polygon;polygon=clipped;clipped=tmp;count=nclipped;
  }
  int kept=0;for (int i=0;i<count;++i) if (cdot(polygon[i]-face1[0],normal)<=0) polygon[kept++]=polygon[i];count=kept;
  if (count<1) return result;
  int ids[4]={0,1,2,3};
  if (n2==2 && count>2) {
    float maximum=0;ids[0]=0;ids[1]=1;
    for (int i=0;i<count;++i) for (int j=i+1;j<count;++j) {CV d=polygon[j]-polygon[i];float length=cdot(d,d);if (length>maximum) {maximum=length;ids[0]=i;ids[1]=j;}}
    result.count=2;
  } else if (count>4) {convex_polygon_quad(polygon,count,ids);result.count=4;} else result.count=count;
  for (int i=0;i<result.count;++i) {CV v=polygon[ids[i]];float distance=cdot(v-face1[0],normal);result.one[i]=v-direction*fabsf(distance);result.two[i]=v;result.distance[i]=distance;}
  return result;
}
__device__ inline ConvexManifold box_multicontact(const ConvexPolytope& p,int face,CV x1,CV x2,const ConvexGeom& a,const ConvexGeom& b) {
  ConvexManifold fallback{};fallback.count=1;fallback.one[0]=x1;fallback.two[0]=x2;
  BoxFeature one=box_feature(p,p.face[face],0),two=box_feature(p,p.face[face],1);CV direction=x2-x1,end[3];BoxNormals n1=box_normals(a,one,-direction),n2=box_normals(b,two,direction);
  int i=0,j=0,edge1=0,edge2=0;bool aligned=false;
  for (int k=0;k<n1.count && !aligned;++k) for (int l=0;l<n2.count;++l) if (cdot(n1.normal[k],n2.normal[l])<-0.99605101346969604f) {i=k;j=l;aligned=true;break;}
  if (!aligned) {
    if (one.dim<3 && n2.count && (one.dim<=two.dim || !n1.count)) {
      n1=box_edge_normals(a,one,end);
      for (int k=0;k<n2.count && !aligned;++k) {
        if (cdot(n2.normal[k],direction)<=convex_minval) continue;
        for (int l=0;l<n1.count;++l) if (fabsf(cdot(n1.normal[l],n2.normal[k]))<0.088782943785190582f) {i=l;j=k;aligned=true;break;}
      }
      if (!aligned) return fallback;edge1=1;
    } else if (two.dim<3) {
      n2=box_edge_normals(b,two,end);
      for (int k=0;k<n1.count && !aligned;++k) {
        if (cdot(n1.normal[k],-direction)<=convex_minval) continue;
        for (int l=0;l<n2.count;++l) if (fabsf(cdot(n2.normal[l],n1.normal[k]))<0.088782943785190582f) {i=l;j=k;aligned=true;break;}
      }
      if (!aligned) return fallback;edge2=1;
    } else return fallback;
  }
  CV face1[4],face2[4];int count1,count2;
  if (edge1) {face1[0]=one.point[0];face1[1]=end[i];count1=2;} else count1=box_face(a,n1.index[edge2 ? j : i],face1);
  if (edge2) {face2[0]=two.point[0];face2[1]=end[i];count2=2;} else count2=box_face(b,n2.index[j],face2);
  if (edge1) {ConvexManifold result=box_polygon_clip(face2,count2,face1,count1,n2.normal[j],-n2.normal[j]);for (int k=0;k<result.count;++k) {CV v=result.one[k];result.one[k]=result.two[k];result.two[k]=v;}return result;}
  if (edge2) return box_polygon_clip(face1,count1,face2,count2,n1.normal[j],-n1.normal[j]);
  return box_polygon_clip(face1,count1,face2,count2,n1.normal[i],n2.normal[j]);
}
}
