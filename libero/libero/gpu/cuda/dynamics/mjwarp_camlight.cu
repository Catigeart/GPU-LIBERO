// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Pinned smooth.py::camlight FIXED paths; no rendering or tracking modes.
#include "mjwarp_camlight.cuh"
#include <cmath>

namespace flash::mjwarp {
namespace {
struct Vec3 {float x,y,z;};
struct Quat {float w,x,y,z;};
__device__ inline Vec3 load3(const float* p,int at) {p+=3*at;return {p[0],p[1],p[2]};}
__device__ inline Quat load4(const float* p,int at) {p+=4*at;return {p[0],p[1],p[2],p[3]};}
__device__ inline void store3(float* p,int at,Vec3 v) {p+=3*at;p[0]=v.x;p[1]=v.y;p[2]=v.z;}
__device__ inline Vec3 operator+(Vec3 a,Vec3 b) {return {a.x+b.x,a.y+b.y,a.z+b.z};}
__device__ inline Vec3 operator*(float a,Vec3 b) {return {a*b.x,a*b.y,a*b.z};}
__device__ inline float dot(Vec3 a,Vec3 b) {return a.x*b.x+a.y*b.y+a.z*b.z;}
__device__ inline Vec3 cross(Vec3 a,Vec3 b) {return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
__device__ inline Vec3 rotate(Vec3 v,Quat q) {
  Vec3 u={q.x,q.y,q.z};
  Vec3 r=2.0f*(dot(u,v)*u)+(q.w*q.w-dot(u,u))*v;
  return r+(2.0f*q.w)*cross(u,v);
}
__device__ inline Quat multiply(Quat a,Quat b) {
  return {a.w*b.w-a.x*b.x-a.y*b.y-a.z*b.z,
    a.w*b.x+a.x*b.w+a.y*b.z-a.z*b.y,
    a.w*b.y-a.x*b.z+a.y*b.w+a.z*b.x,
    a.w*b.z+a.x*b.y-a.y*b.x+a.z*b.w};
}
__device__ inline void matrix(float* p,int at,Quat q) {
  p+=9*at;
  float a=q.w*q.w,b=q.w*q.x,c=q.w*q.y,d=q.w*q.z;
  float e=q.x*q.x,f=q.x*q.y,g=q.x*q.z,h=q.y*q.y,i=q.y*q.z,j=q.z*q.z;
  p[0]=a+e-h-j;p[1]=2.0f*(f-d);p[2]=2.0f*(g+c);
  p[3]=2.0f*(f+d);p[4]=a-e+h-j;p[5]=2.0f*(i-b);
  p[6]=2.0f*(g-c);p[7]=2.0f*(i+b);p[8]=a-e-h+j;
}

__global__ void camera_frames(CamLightInputs a) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if (t>=a.worlds*a.cameras) return;
  int w=t/a.cameras,c=t%a.cameras,body=w*a.bodies+a.cam_body[c];
  Quat q=load4(a.xquat,body);
  Vec3 local=load3(a.cam_pos,(w%a.cam_pos_rows)*a.cameras+c);
  store3(a.cam_xpos,t,load3(a.xpos,body)+rotate(local,q));
  matrix(a.cam_xmat,t,multiply(q,load4(a.cam_quat,(w%a.cam_quat_rows)*a.cameras+c)));
}

__global__ void light_frames(CamLightInputs a) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if (t>=a.worlds*a.lights) return;
  int w=t/a.lights,l=t%a.lights,body=w*a.bodies+a.light_body[l];
  Quat q=load4(a.xquat,body);
  Vec3 local=load3(a.light_pos,(w%a.light_pos_rows)*a.lights+l);
  store3(a.light_xpos,t,load3(a.xpos,body)+rotate(local,q));
  Vec3 dir=rotate(load3(a.light_dir,(w%a.light_dir_rows)*a.lights+l),q);
  // Warp1.18 vec3 normalize: sqrt(dot), component division if length>0;
  // otherwise zero. CPU's mjMINVAL/axis fallback does not apply here.
  float length=sqrtf(dir.x*dir.x+dir.y*dir.y+dir.z*dir.z);
  if (length>0.0f) dir={dir.x/length,dir.y/length,dir.z/length};
  else dir={0.0f,0.0f,0.0f};
  store3(a.light_xdir,t,dir);
}
}
}

extern "C" {
const char* flash_mjwarp_camlight_source_id() {return FLASH_MJW_CAMLIGHT_SOURCE_ID;}
std::size_t flash_mjwarp_camlight_arguments_bytes() {return sizeof(flash::mjwarp::CamLightInputs);}
int flash_mjwarp_camlight_arch() {return FLASH_MJW_CAMLIGHT_ARCH;}
int flash_mjwarp_camlight_launch(const flash::mjwarp::CamLightInputs* a,int phase,cudaStream_t stream) {
  if (!a || a->worlds<1 || a->bodies<1 || a->cameras<0 || a->lights<0 ||
      a->cam_pos_rows<1 || a->cam_quat_rows<1 || a->light_pos_rows<1 || a->light_dir_rows<1)
    return int(cudaErrorInvalidValue);
  using namespace flash::mjwarp;
  if (phase==0) {
    if (!a->cameras) return 0;
    camera_frames<<<(a->worlds*a->cameras+255)/256,256,0,stream>>>(*a);
  } else if (phase==1) {
    if (!a->lights) return 0;
    light_frames<<<(a->worlds*a->lights+255)/256,256,0,stream>>>(*a);
  } else return int(cudaErrorInvalidValue);
  return int(cudaGetLastError());
}
}
