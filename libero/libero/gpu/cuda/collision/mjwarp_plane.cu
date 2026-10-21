// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Pinned collision_primitive.py/_core.py plane colliders, no Warp native headers.
#include "mjwarp_plane.cuh"
namespace flash::mjwarp {
namespace {
constexpr float absent=1.e10f;
struct PlaneManifold {
  int count;
  float dist[8];
  CV pos[8],frame[3];
};
__device__ PlaneManifold empty_manifold(CV normal,int count) {
  PlaneManifold out{};out.count=count;
  for (int i=0;i<8;++i) out.dist[i]=absent;
  contact_frame(normal,out.frame);return out;
}
__device__ void plane_sphere(PlaneManifold& out,int i,CV n,CV origin,CV center,float radius) {
  float d=cdot(center-origin,n)-radius;out.dist[i]=d;out.pos[i]=center-n*(radius+.5f*d);
}
__device__ PlaneManifold plane_capsule(CV n,CV origin,CV center,CV axis,CV size) {
  PlaneManifold out=empty_manifold(n,2);
  CV tangent=axis-n*cdot(n,axis);float length=sqrtf(cdot(tangent,tangent));
  // Deliberately preserve the pinned capsule frame, which differs from the
  // generic frame constructor when its projected axis is nearly degenerate.
  CV b=length==0 ? tangent : tangent*(1.0f/length);
  if (length<.5f) b=(-.5f<n[1] && n[1]<.5f) ? cv(0,1,0) : cv(0,0,1);
  out.frame[0]=n;out.frame[1]=b;out.frame[2]=ccross(n,b);
  plane_sphere(out,0,n,origin,center+axis*size[1],size[0]);
  plane_sphere(out,1,n,origin,center-axis*size[1],size[0]);return out;
}
__device__ PlaneManifold plane_box(CV n,CV origin,CV center,const float* R,CV size) {
  PlaneManifold out=empty_manifold(n,8);float center_dist=cdot(center-origin,n);
  for (int i=0;i<8;++i) {
    CV corner=crot(R,cv(i&1 ? size[0] : -size[0],i&2 ? size[1] : -size[1],i&4 ? size[2] : -size[2]));
    float d=center_dist+cdot(n,corner);out.dist[i]=d;out.pos[i]=corner+center-(.5f*n)*d;
  }
  return out;
}
__device__ PlaneManifold plane_cylinder(CV n,CV origin,CV center,CV axis,CV size) {
  PlaneManifold out=empty_manifold(n,4);float projection=cdot(n,axis);
  if (projection>0) {axis=-axis;projection=-projection;}
  float dist0=cdot(center-origin,n);CV radial=axis*projection-n;float length2=cdot(radial,radial);
  radial=length2>=1.e-12f ? radial*(size[0]/sqrtf(length2)) : cv(size[0],0,0);
  float radial_projection=cdot(radial,n);axis=axis*size[1];projection*=size[1];
  out.dist[0]=dist0+projection+radial_projection;out.pos[0]=center+radial+axis-n*(out.dist[0]*.5f);
  out.dist[1]=dist0-projection+radial_projection;out.pos[1]=center+radial-axis-n*(out.dist[1]*.5f);
  out.dist[2]=dist0+projection-radial_projection*.5f;out.dist[3]=out.dist[2];
  CV lateral=cnormalize(ccross(radial,axis))*(size[0]*sqrtf(3.0f)*.5f);
  out.pos[2]=center+lateral+axis-radial*.5f-n*(out.dist[2]*.5f);
  out.pos[3]=center-lateral+axis-radial*.5f-n*(out.dist[3]*.5f);return out;
}
__device__ float quadrilateral_area(const ContactInputs& a,int vertadr,int face,int i,int j,int k,int l) {
  CV v1=cload(a.mesh_vert,vertadr+a.mesh_polyvert[face+i]),v2=cload(a.mesh_vert,vertadr+a.mesh_polyvert[face+j]);
  CV v3=cload(a.mesh_vert,vertadr+a.mesh_polyvert[face+k]),v4=cload(a.mesh_vert,vertadr+a.mesh_polyvert[face+l]);
  CV cross=ccross(v1-v3,v2-v4);return .5f*sqrtf(cdot(cross,cross));
}
__device__ int hull4(const ContactInputs& a,int vertadr,int face,int count,int anchor,int* indices) {
  int b=(anchor+1)%count,c=(anchor+2)%count,d=(anchor+3)%count;
  indices[0]=anchor;indices[1]=b;indices[2]=c;indices[3]=d;
  if (count<=4) return count;
  float area=quadrilateral_area(a,vertadr,face,anchor,b,c,d);
  while (true) {
    int next=(d+1)%count;float candidate=quadrilateral_area(a,vertadr,face,anchor,b,c,next);
    if (candidate<=area) break;
    d=next;area=candidate;indices[1]=b;indices[2]=c;indices[3]=d;
    while (true) {
      next=(c+1)%count;candidate=quadrilateral_area(a,vertadr,face,anchor,b,next,d);
      if (candidate<=area) break;
      c=next;area=candidate;indices[1]=b;indices[2]=c;indices[3]=d;
    }
    while (true) {
      next=(b+1)%count;candidate=quadrilateral_area(a,vertadr,face,anchor,next,c,d);
      if (candidate<=area) break;
      b=next;area=candidate;indices[1]=b;indices[2]=c;indices[3]=d;
    }
  }
  return 4;
}
__device__ PlaneManifold plane_mesh(const ContactInputs& a,int w,int geom,CV n,CV origin,CV center,const float* R,float margin) {
  PlaneManifold out=empty_manifold(n,4);int mesh=a.geom_dataid[(w%a.dataid_rows)*a.geoms+geom];
  int adr=mesh>=0 ? a.mesh_vertadr[mesh] : -1,num=mesh>=0 ? a.mesh_vertnum[mesh] : -1;
  int graph=mesh>=0 ? a.mesh_graphadr[mesh] : -1,index=-1;CV local_normal=crotT(R,n),direction=-local_normal;
  float maximum=-absent;
  if (graph==-1 || num<10) {
    for (int i=0;i<num;++i) {float support=cdot(direction,cload(a.mesh_vert,adr+i));if (support>maximum) {maximum=support;index=i;}}
  } else {
    int vertices=a.mesh_graph[graph],edgeadr=graph+2,globaladr=graph+2+vertices,localadr=graph+2+2*vertices;
    int previous=-1,best=0;maximum=cdot(direction,cload(a.mesh_vert,adr+a.mesh_graph[globaladr]));
    while (best!=previous) {
      previous=best;int i=a.mesh_graph[edgeadr+best];
      while (a.mesh_graph[localadr+i]>=0) {
        int neighbor=a.mesh_graph[localadr+i],vertex=a.mesh_graph[globaladr+neighbor];float support=cdot(direction,cload(a.mesh_vert,adr+vertex));
        if (support>maximum) {maximum=support;best=neighbor;}
        ++i;
      }
    }
    index=a.mesh_graph[globaladr+best];
  }
  if (index<0) return out;
  CV v0=center+crot(R,cload(a.mesh_vert,adr+index));float dist=cdot(n,v0-origin);out.dist[0]=dist;out.pos[0]=v0-(.5f*dist)*n;
  int polygons=a.mesh_polynum[mesh],polyadr=a.mesh_polyadr[mesh];
  if (polygons<=0 || dist>margin) return out;
  int mapadr=a.mesh_polymapadr[adr+index],mapnum=a.mesh_polymapnum[adr+index],best_poly=-1;float best_dot=1;
  for (int i=0;i<mapnum;++i) {
    int id=a.mesh_polymap[mapadr+i];float dot=cdot(cload(a.mesh_polynormal,polyadr+id),local_normal);
    if (dot<best_dot) {best_dot=dot;best_poly=id;}
  }
  if (best_poly<0) return out;
  int face=a.mesh_polyvertadr[polyadr+best_poly],count=a.mesh_polyvertnum[polyadr+best_poly],anchor=0;
  for (int i=0;i<count;++i) if (a.mesh_polyvert[face+i]==index) {anchor=i;break;}
  int indices[4],hull=hull4(a,adr,face,count,anchor,indices),contact=1;
  for (int i=1;i<hull;++i) {
    CV point=center+crot(R,cload(a.mesh_vert,adr+a.mesh_polyvert[face+indices[i]]));
    if (cdot(n,point-center)>0) continue;
    float d=cdot(n,point-origin);out.dist[contact]=d;out.pos[contact]=point-(.5f*d)*n;++contact;
  }
  return out;
}
__global__ void plane_contacts(ContactInputs a) {
  for (int i=blockIdx.x*blockDim.x+threadIdx.x;i<a.capacity;i+=gridDim.x*blockDim.x) {
    if (i>=a.candidate_count[0]) continue;
    int g1=a.candidate_geom[2*i],g2=a.candidate_geom[2*i+1],type=a.geom_type[g2];
    if (a.geom_type[g1]!=0 || type<2 || type>7) continue;
    int w=a.candidate_world[i],pair=a.candidate_ids[2*i],sensor=a.candidate_ids[2*i+1];
    ContactMaterial p=contact_material(a,w,g1,g2,pair);
    CV origin=cload(a.xpos,w*a.geoms+g1),center=cload(a.xpos,w*a.geoms+g2),n=ccolumn(a.xmat+9*(w*a.geoms+g1),2);
    const float* R=a.xmat+9*(w*a.geoms+g2);CV size=cload(contact_parameter(a.size,w,a.geoms,g2,3));
    PlaneManifold out=empty_manifold(n,1);
    if (type==2) plane_sphere(out,0,n,origin,center,size[0]);
    else if (type==3) out=plane_capsule(n,origin,center,ccolumn(R,2),size);
    else if (type==4) {
      CV direction=crotT(R,n);for (int k=0;k<3;++k) direction[k]*=size[k];direction=-cnormalize(direction);
      for (int k=0;k<3;++k) direction[k]*=size[k];CV point=center+crot(R,direction);float d=cdot(n,point-origin);
      out.dist[0]=d;out.pos[0]=point-(n*d)*.5f;
    } else if (type==5) out=plane_cylinder(n,origin,center,ccolumn(R,2),size);
    else if (type==6) out=plane_box(n,origin,center,R,size);
    else out=plane_mesh(a,w,g2,n,origin,center,R,p.margin+p.gap);
    for (int k=0;k<out.count;++k) publish_contact(a,p,w,g1,g2,pair,sensor,k,out.dist[k],out.pos[k],out.frame);
  }
}
}
}
extern "C" {
const char* flash_mjwarp_plane_source_id() {return FLASH_MJW_PLANE_SOURCE_ID;}
std::size_t flash_mjwarp_plane_arguments_bytes() {return sizeof(flash::mjwarp::ContactInputs);}
int flash_mjwarp_plane_arch() {return FLASH_MJW_PLANE_ARCH;}
int flash_mjwarp_plane_launch(const flash::mjwarp::ContactInputs* a,cudaStream_t stream) {
  if (!a || a->worlds<1 || a->geoms<1 || a->capacity<0) return int(cudaErrorInvalidValue);
  if (a->capacity) {int blocks=(a->capacity+127)/128;if (blocks>4096) blocks=4096;flash::mjwarp::plane_contacts<<<blocks,128,0,stream>>>(*a);}
  return int(cudaGetLastError());
}
}
