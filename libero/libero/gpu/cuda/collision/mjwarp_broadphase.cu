// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Pinned MJWarp collision_driver.py NXN broadphase and bounding filters.
#include "mjwarp_broadphase.cuh"
#include <cmath>
namespace flash::mjwarp {
namespace {
struct V {
  float x,y,z;
  __device__ float operator[](int i) const {return i==0 ? x : i==1 ? y : z;}
};
__device__ inline V load(const float* p,int i) {p+=3*i;return {p[0],p[1],p[2]};}
__device__ inline V operator+(V a,V b) {return {a.x+b.x,a.y+b.y,a.z+b.z};}
__device__ inline V operator-(V a,V b) {return {a.x-b.x,a.y-b.y,a.z-b.z};}
__device__ inline float dot(V a,V b) {return a.x*b.x+a.y*b.y+a.z*b.z;}
__device__ inline V rotate(const float* r,V v) {
  return {r[0]*v.x+r[1]*v.y+r[2]*v.z,r[3]*v.x+r[4]*v.y+r[5]*v.z,r[6]*v.x+r[7]*v.y+r[8]*v.z};
}
__device__ inline V column(const float* r,int k) {return {r[k],r[3+k],r[6+k]};}

__device__ bool plane_filter(float r1,float r2,float m1,float m2,V p1,V p2,const float* R1,const float* R2) {
  if (r1==0.0f) return dot(p2-p1,column(R1,2))<=r2+m1+m2;
  if (r2==0.0f) return dot(p1-p2,column(R2,2))<=r1+m1+m2;
  return true;
}

__device__ bool sphere_filter(float r1,float r2,float m1,float m2,V p1,V p2) {
  float bound=r1+r2+m1+m2;V difference=p2-p1;
  return dot(difference,difference)<=bound*bound;
}

__device__ bool aabb_filter(V center1,V center2,V size1,V size2,float m1,float m2,V p1,V p2,const float* R1,const float* R2) {
  center1=rotate(R1,center1)+p1;center2=rotate(R2,center2)+p2;
  constexpr float maxval=1.e10f;
  float lo1[3]={maxval,maxval,maxval},hi1[3]={-maxval,-maxval,-maxval};
  float lo2[3]={maxval,maxval,maxval},hi2[3]={-maxval,-maxval,-maxval};
  // Preserve the original corner construction and max/min comparisons rather
  // than replacing them by an algebraically equivalent absolute-matrix bound.
  for (int i=0;i<2;++i) for (int j=0;j<2;++j) for (int k=0;k<2;++k) {
    float x=float(2*i-1),y=float(2*j-1),z=float(2*k-1);
    V corner1=rotate(R1,{x*size1.x,y*size1.y,z*size1.z});
    V corner2=rotate(R2,{x*size2.x,y*size2.y,z*size2.z});
    for (int axis=0;axis<3;++axis) {
      if (corner1[axis]<lo1[axis]) lo1[axis]=corner1[axis];
      if (corner1[axis]>hi1[axis]) hi1[axis]=corner1[axis];
      if (corner2[axis]<lo2[axis]) lo2[axis]=corner2[axis];
      if (corner2[axis]>hi2[axis]) hi2[axis]=corner2[axis];
    }
  }
  float margin=m1+m2;
  for (int axis=0;axis<3;++axis)
    if (center1[axis]+hi1[axis]+margin<center2[axis]+lo2[axis] ||
        center2[axis]+hi2[axis]+margin<center1[axis]+lo1[axis]) return false;
  return true;
}

__device__ bool obb_filter(V center1,V center2,V size1,V size2,float m1,float m2,V p1,V p2,const float* R1,const float* R2) {
  V centers[2]={rotate(R1,center1)+p1,rotate(R2,center2)+p2};
  V normals[6]={column(R1,0),column(R1,1),column(R1,2),column(R2,0),column(R2,1),column(R2,2)};
  V sizes[2]={size1,size2};float margin=m1+m2;
  // The pinned predicate uses these six face axes. Do not silently add the
  // nine edge cross axes of a full SAT, which changes its candidate set.
  for (int j=0;j<2;++j) for (int k=0;k<3;++k) {
    float projection[2],radius[2];
    for (int i=0;i<2;++i) {
      projection[i]=dot(centers[i],normals[3*j+k]);
      radius[i]=fabsf(sizes[i].x*dot(normals[3*i],normals[3*j+k]))+
                fabsf(sizes[i].y*dot(normals[3*i+1],normals[3*j+k]))+
                fabsf(sizes[i].z*dot(normals[3*i+2],normals[3*j+k]));
    }
    if (radius[0]+radius[1]+margin<fabsf(projection[1]-projection[0])) return false;
  }
  return true;
}

__device__ bool bounding_filter(BroadphaseInputs a,int w,int g1,int g2,int pairid) {
  const float* box=a.aabb+((w%a.aabb_rows)*a.geoms)*6;
  V center1=load(box,2*g1),center2=load(box,2*g2),size1=load(box,2*g1+1),size2=load(box,2*g2+1);
  const float* bound=a.rbound+(w%a.rbound_rows)*a.geoms;
  float r1=bound[g1],r2=bound[g2],m1=0.0f,m2=0.0f;
  if (pairid>=0) {
    m1=a.pair_margin[(w%a.pair_margin_rows)*a.explicit_pairs+pairid]+
       a.pair_gap[(w%a.pair_gap_rows)*a.explicit_pairs+pairid];
  } else {
    const float *margin=a.margin+(w%a.margin_rows)*a.geoms,*gap=a.gap+(w%a.gap_rows)*a.geoms;
    m1=margin[g1]+gap[g1];m2=margin[g2]+gap[g2];
  }
  V p1=load(a.xpos,w*a.geoms+g1),p2=load(a.xpos,w*a.geoms+g2);
  const float *R1=a.xmat+9*(w*a.geoms+g1),*R2=a.xmat+9*(w*a.geoms+g2);
  if (r1==0.0f || r2==0.0f) {
    if (a.filter&1) return plane_filter(r1,r2,m1,m2,p1,p2,R1,R2);
  } else {
    if ((a.filter&2) && !sphere_filter(r1,r2,m1,m2,p1,p2)) return false;
    if ((a.filter&4) && !aabb_filter(center1,center2,size1,size2,m1,m2,p1,p2,R1,R2)) return false;
    if ((a.filter&8) && !obb_filter(center1,center2,size1,size2,m1,m2,p1,p2,R1,R2)) return false;
  }
  return true;
}

__global__ void candidate_pairs(BroadphaseInputs a) {
  for (long long t=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;t<static_cast<long long>(a.worlds)*a.pairs;t+=static_cast<long long>(gridDim.x)*blockDim.x) {
    int index=int(t),w=index/a.pairs,i=index%a.pairs,g1=a.pairs_in[2*i],g2=a.pairs_in[2*i+1];
    int type1=a.geom_type[g1],type2=a.geom_type[g2];
    if (a.dataid_rows>1) {
      const int* dataid=a.geom_dataid+(w%a.dataid_rows)*a.geoms;
      if ((type1==7 && dataid[g1]<0) || (type2==7 && dataid[g2]<0)) continue;
    }
    int contact_pair=a.pair_ids_in[2*i],sensor_pair=a.pair_ids_in[2*i+1];
    if (!bounding_filter(a,w,g1,g2,contact_pair) && sensor_pair<0) continue;
    int slot=atomicAdd(a.count,1);
    if (slot>=a.capacity) continue;
    a.pairs_out[2*slot]=type1>type2 ? g2 : g1;
    a.pairs_out[2*slot+1]=type1>type2 ? g1 : g2;
    a.pair_ids_out[2*slot]=contact_pair;a.pair_ids_out[2*slot+1]=sensor_pair;
    a.world_out[slot]=w;
  }
}
}
}
extern "C" {
const char* flash_mjwarp_broadphase_source_id() {return FLASH_MJW_BROADPHASE_SOURCE_ID;}
std::size_t flash_mjwarp_broadphase_arguments_bytes() {return sizeof(flash::mjwarp::BroadphaseInputs);}
int flash_mjwarp_broadphase_arch() {return FLASH_MJW_BROADPHASE_ARCH;}
int flash_mjwarp_broadphase_launch(const flash::mjwarp::BroadphaseInputs* a,cudaStream_t stream) {
  if (!a || a->worlds<1 || a->geoms<1 || a->pairs<0 || a->capacity<0) return int(cudaErrorInvalidValue);
  if (a->pairs) {
    int blocks=(a->worlds*a->pairs+255)/256;
    if (blocks>4096) blocks=4096;
    flash::mjwarp::candidate_pairs<<<blocks,256,0,stream>>>(*a);
  }
  return int(cudaGetLastError());
}
}
