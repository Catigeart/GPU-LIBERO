// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Pinned passive.py joint forces/final sum and math.py quaternion operations.
#include "mjwarp_passive.cuh"
#include <cmath>
#ifndef FLASH_MJW_PASSIVE_SOURCE_ID
#error "Build through flash_mjwarp_passive_fp32"
#endif
namespace flash::mjwarp {
namespace {
__device__ inline float polynomial(float linear, const float* poly, float x) {
  float value=linear;
  value+=poly[0]*x;
  value+=poly[1]*x*x;
  return value;
}

__device__ inline void rotation_difference(const float* q, const float* ref, float* out) {
  float norm=sqrtf(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
  float v[4];
  // Warp1.18 quat::normalize has kEps=0 and this exact fallback layout.
  if (norm>0.0f) { float inv=1.0f/norm; for (int k=0;k<4;++k) v[k]=q[k]*inv; }
  else { v[0]=0.0f; v[1]=0.0f; v[2]=0.0f; v[3]=1.0f; }
  float rw=ref[0], rx=-ref[1], ry=-ref[2], rz=-ref[3];
  float w=rw*v[0]-rx*v[1]-ry*v[2]-rz*v[3];
  float x=rw*v[1]+rx*v[0]+ry*v[3]-rz*v[2];
  float y=rw*v[2]-rx*v[3]+ry*v[0]+rz*v[1];
  float z=rw*v[3]+rx*v[2]-ry*v[1]+rz*v[0];
  float sine=sqrtf(x*x+y*y+z*z);
  if (sine==0.0f) { out[0]=out[1]=out[2]=0.0f; return; }
  constexpr float pi=3.14159265358979323846f;
  float speed=2.0f*atan2f(sine,w);
  if (speed>pi) speed-=2.0f*pi;
  // Keep axis*speed/sine order instead of moving the division before multiply.
  out[0]=x*speed/sine;out[1]=y*speed/sine;out[2]=z*speed/sine;
}

__global__ void joint_forces(PassiveInputs a) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if (t>=a.worlds*a.joints) return;
  int w=t/a.joints,j=t%a.joints,d=a.dofadr[j],p=a.qposadr[j],kind=a.joint_type[j];
  int width=kind==0?6:(kind==1?3:1);
  const float* q=a.qpos+w*a.nq;
  const float* ref=a.qpos_spring+(w%a.spring_rows)*a.nq;
  float k=a.stiffness[(w%a.stiffness_rows)*a.joints+j];
  const float* sp=a.stiffnesspoly+((w%a.stiffnesspoly_rows)*a.joints+j)*2;
  bool has_stiffness=!a.disable_spring && (k!=0.0f || sp[0]!=0.0f || sp[1]!=0.0f);
  for (int i=0;i<width;++i) {
    float damping=a.damping[(w%a.damping_rows)*a.nv+d+i];
    const float* dp=a.dampingpoly+((w%a.dampingpoly_rows)*a.nv+d+i)*2;
    float v=a.qvel[w*a.nv+d+i],force=0.0f;
    if (!a.disable_damper && (damping!=0.0f || dp[0]!=0.0f || dp[1]!=0.0f))
      force=-v*polynomial(damping,dp,fabsf(v));
    a.damper[w*a.nv+d+i]=force;
    a.spring[w*a.nv+d+i]=0.0f;
  }
  if (!has_stiffness) return;
  float* spring=a.spring+w*a.nv+d;
  if (kind==0) {
    float x=q[p]-ref[p],y=q[p+1]-ref[p+1],z=q[p+2]-ref[p+2];
    float gain=polynomial(k,sp,sqrtf(x*x+y*y+z*z));
    spring[0]=-gain*x;spring[1]=-gain*y;spring[2]=-gain*z;
    spring+=3;p+=3;
  }
  if (kind==0 || kind==1) {
    float delta[3];rotation_difference(q+p,ref+p,delta);
    float r=sqrtf(delta[0]*delta[0]+delta[1]*delta[1]+delta[2]*delta[2]);
    float gain=polynomial(k,sp,r);
    for (int i=0;i<3;++i) spring[i]=-gain*delta[i];
  } else {
    float delta=q[p]-ref[p];spring[0]=-delta*polynomial(k,sp,delta);
  }
}

__global__ void combine_forces(PassiveInputs a) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if (t>=a.worlds*a.nv) return;
  int dof=t%a.nv;
  float force=a.spring[t];force+=a.damper[t];
  if (a.gravity_enabled && !a.actgravcomp[a.dof_joint[dof]]) force+=a.gravcomp[t];
  if (a.has_fluid) force+=a.fluid[t];
  if (a.has_adhesion) force+=a.adhesion[t];
  a.passive[t]=force;
}
}
}
extern "C" {
const char* flash_mjwarp_passive_source_id() { return FLASH_MJW_PASSIVE_SOURCE_ID; }
std::size_t flash_mjwarp_passive_arguments_bytes() { return sizeof(flash::mjwarp::PassiveInputs); }
int flash_mjwarp_passive_arch() { return FLASH_MJW_PASSIVE_ARCH; }
int flash_mjwarp_passive_launch(const flash::mjwarp::PassiveInputs* a,int phase,cudaStream_t stream) {
  if (!a || a->worlds<1 || a->nv<1 || a->nq<1 || a->joints<1 || a->spring_rows<1 ||
      a->stiffness_rows<1 || a->stiffnesspoly_rows<1 || a->damping_rows<1 || a->dampingpoly_rows<1)
    return int(cudaErrorInvalidValue);
  using namespace flash::mjwarp;
  if (phase==0) joint_forces<<<(a->worlds*a->joints+255)/256,256,0,stream>>>(*a);
  else if (phase==1) combine_forces<<<(a->worlds*a->nv+255)/256,256,0,stream>>>(*a);
  else return int(cudaErrorInvalidValue);
  return int(cudaGetLastError());
}
}
