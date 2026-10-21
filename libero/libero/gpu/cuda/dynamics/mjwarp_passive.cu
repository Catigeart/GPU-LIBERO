// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Pinned passive.py joint forces/final sum and math.py quaternion operations.
#include "mjwarp_passive.cuh"
#include "mjwarp_passive_math.cuh"
#include <cmath>
#ifndef FLASH_MJW_PASSIVE_SOURCE_ID
#error "Build through flash_mjwarp_passive_fp32"
#endif
namespace flash::mjwarp {
namespace {
using passive_math::polynomial;
using passive_math::rotation_difference;

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
