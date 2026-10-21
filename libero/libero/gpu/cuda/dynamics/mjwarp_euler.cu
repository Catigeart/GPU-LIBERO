// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Pinned forward.py Euler numerical leaves and math.py quaternion integration.
#include "mjwarp_euler.cuh"
#include <cmath>
#ifndef FLASH_MJW_EULER_SOURCE_ID
#error "Build through flash_mjwarp_euler_fp32"
#endif
namespace flash::mjwarp {
namespace {
struct Quaternion { float w,x,y,z; };
__device__ inline Quaternion normalize(Quaternion q) {
  float length=sqrtf(q.w*q.w+q.x*q.x+q.y*q.y+q.z*q.z);
  if (length>0.0f) {
    float r=1.0f/length;
    return {q.w*r,q.x*r,q.y*r,q.z*r};
  }
  // Pinned Warp quat storage fallback is [0,0,0,1], even though MJWarp
  // interprets its components as wxyz. Preserve that degenerate-input rule.
  return {0.0f,0.0f,0.0f,1.0f};
}
__device__ inline Quaternion integrate(Quaternion q,const float* v,float dt) {
  float length=sqrtf(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
  float x=0.0f,y=0.0f,z=0.0f;
  if (length>0.0f) {x=v[0]/length;y=v[1]/length;z=v[2]/length;}
  float angle=dt*length;
  float s=sinf(angle*0.5f),c=cosf(angle*0.5f);
  Quaternion b={c,x*s,y*s,z*s};q=normalize(q);
  return normalize({q.w*b.w-q.x*b.x-q.y*b.y-q.z*b.z,
    q.w*b.x+q.x*b.w+q.y*b.z-q.z*b.y,
    q.w*b.y-q.x*b.z+q.y*b.w+q.z*b.x,
    q.w*b.z+q.x*b.y-q.y*b.x+q.z*b.w});
}
__global__ void damping_derivative(EulerInputs a) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if (t>=a.worlds*a.nv) return;
  int w=t/a.nv,d=t%a.nv;
  const float* p=a.dampingpoly+((w%a.dampingpoly_rows)*a.nv+d)*2;
  float v=fabsf(a.qvel[t]);
  float result=a.damping[(w%a.damping_rows)*a.nv+d];
  result+=2.0f*p[0]*v;
  result+=3.0f*p[1]*v*v;
  a.derivative[t]=result;
}
__global__ void damping_diagonal(EulerInputs a) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if (t>=a.worlds*a.nv) return;
  int w=t/a.nv,d=t%a.nv,at=w*a.matrix_width+a.rowadr[d]+a.rownnz[d]-1;
  a.matrix[at]+=a.timestep[w%a.timestep_rows]*a.derivative[t];
}
// Admission proves that the CSR rows partition the complete input matrix and
// that each row ends with its unique diagonal. Eight lanes cooperate on a row
// copy; lane zero publishes its derivative and adjusted diagonal. All lanes
// compute the same current derivative privately, with no shared state/barrier.
__global__ void damping_prepare(EulerInputs a,const float* source) {
  const int w=blockIdx.y,lane=threadIdx.x&7,
      d=blockIdx.x*(blockDim.x/8)+threadIdx.x/8;
  if (d>=a.nv) return;
  const int t=w*a.nv+d,first=a.rowadr[d],count=a.rownnz[d];
  const float* p=a.dampingpoly+((w%a.dampingpoly_rows)*a.nv+d)*2;
  const float v=fabsf(a.qvel[t]);
  float derivative=a.damping[(w%a.damping_rows)*a.nv+d];
  derivative+=2.0f*p[0]*v;
  derivative+=3.0f*p[1]*v*v;
  if (lane==0) a.derivative[t]=derivative;
  const int base=w*a.matrix_width+first;
  for (int k=lane;k<count-1;k+=8) a.matrix[base+k]=source[base+k];
  if (lane==0) a.matrix[base+count-1]=source[base+count-1]
      +a.timestep[w%a.timestep_rows]*derivative;
}
__global__ void next_velocity(EulerInputs a) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if (t>=a.worlds*a.nv) return;
  int w=t/a.nv;
  a.qvel[t]=a.qvel[t]+a.acceleration[t]*a.timestep[w%a.timestep_rows];
}
__global__ void next_position(EulerInputs a) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if (t>=a.worlds*a.joints) return;
  int w=t/a.joints,j=t%a.joints,kind=a.joint_type[j];
  float dt=a.timestep[w%a.timestep_rows];
  float* q=a.qpos+w*a.nq+a.qposadr[j];
  const float* v=a.qvel+w*a.nv+a.dofadr[j];
  if (kind==0) {
    q[0]+=dt*v[0];q[1]+=dt*v[1];q[2]+=dt*v[2];q+=3;v+=3;
  }
  if (kind<2) {
    Quaternion r=integrate({q[0],q[1],q[2],q[3]},v,dt);
    q[0]=r.w;q[1]=r.x;q[2]=r.y;q[3]=r.z;
  } else q[0]+=dt*v[0];
}
}
}
extern "C" {
const char* flash_mjwarp_euler_source_id() { return FLASH_MJW_EULER_SOURCE_ID; }
std::size_t flash_mjwarp_euler_arguments_bytes() { return sizeof(flash::mjwarp::EulerInputs); }
int flash_mjwarp_euler_arch() { return FLASH_MJW_EULER_ARCH; }
int flash_mjwarp_euler_prepare(const flash::mjwarp::EulerInputs* a,
                             const float* source,cudaStream_t stream) {
  if (!a || !source || !a->matrix || !a->derivative || a->worlds<1 ||
      a->nv<1 || a->matrix_width<1) return int(cudaErrorInvalidValue);
  flash::mjwarp::damping_prepare<<<dim3((a->nv+15)/16,a->worlds),128,0,stream>>>(*a,source);
  return int(cudaGetLastError());
}
int flash_mjwarp_euler_launch(const flash::mjwarp::EulerInputs* a,int phase,cudaStream_t stream) {
  if (!a || a->worlds<1 || a->nv<1 || a->nq<1 || a->joints<1 || a->matrix_width<1 ||
      a->timestep_rows<1 || a->damping_rows<1 || a->dampingpoly_rows<1) return int(cudaErrorInvalidValue);
  using namespace flash::mjwarp;
  int dofs=(a->worlds*a->nv+255)/256,joints=(a->worlds*a->joints+255)/256;
  if (phase==0) damping_derivative<<<dofs,256,0,stream>>>(*a);
  else if (phase==1) damping_diagonal<<<dofs,256,0,stream>>>(*a);
  else if (phase==2) next_velocity<<<dofs,256,0,stream>>>(*a);
  else if (phase==3) next_position<<<joints,256,0,stream>>>(*a);
  else return int(cudaErrorInvalidValue);
  return int(cudaGetLastError());
}
}
