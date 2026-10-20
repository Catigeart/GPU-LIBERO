// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Pinned forward.py stateless fixed/affine scalar-joint actuation path.
#include "mjwarp_actuation.cuh"
#include <cmath>
#ifndef FLASH_MJW_ACTUATION_SOURCE_ID
#error "Build through flash_mjwarp_actuation_fp32"
#endif
namespace flash::mjwarp {
namespace {
__device__ inline float clip(float x,const float* bounds) {
  // Warp floating clamp uses native fmin/fmax, including its NaN behavior.
  return fminf(fmaxf(bounds[0],x),bounds[1]);
}

__global__ void actuator_force(ActuationInputs a) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if (t>=a.worlds*a.actuators) return;
  int w=t/a.actuators,u=t%a.actuators;
  float ctrl=0.0f;
  if (a.ctrlnum[u]>0) {
    int adr=a.ctrladr[u];ctrl=a.ctrl[w*a.controls+adr];
    if (a.ctrllimited[adr] && !a.disable_clampctrl)
      ctrl=clip(ctrl,a.ctrlrange+((w%a.ctrlrange_rows)*a.controls+adr)*2);
  }
  const float* gp=a.gainprm+((w%a.gain_rows)*a.actuators+u)*10;
  const float* bp=a.biasprm+((w%a.bias_rows)*a.actuators+u)*10;
  float gain=gp[0];
  if (a.gain_type[u]==1) gain=gp[0]+gp[1]*a.length[t]+gp[2]*a.velocity[t];
  float bias=0.0f;
  if (a.bias_type[u]==1) bias=bp[0]+bp[1]*a.length[t]+bp[2]*a.velocity[t];
  float force=gain*ctrl+bias;
  if (a.forcelimited[u]) force=clip(force,a.forcerange+((w%a.forcerange_rows)*a.actuators+u)*2);
  a.force[t]=force;
}

__global__ void project_force(ActuationInputs a) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if (t>=a.worlds*a.actuators) return;
  int w=t/a.actuators,at=a.moment_rowadr[t],count=a.moment_rownnz[t];
  for (int i=0;i<count;++i) {
    int slot=w*a.moments+at+i,dof=a.moment_colind[slot];
    // Keep pinned per-actuator CSR order and global FP32 atomics.
    atomicAdd(a.qfrc+w*a.nv+dof,a.moment[slot]*a.force[t]);
  }
}

__global__ void gravcomp_limits(ActuationInputs a) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if (t>=a.worlds*a.nv) return;
  int w=t/a.nv,j=a.dof_joint[t%a.nv];
  float force=a.qfrc[t];
  if (a.gravity_enabled && a.actgravcomp[j]) force+=a.gravcomp[t];
  if (a.jointlimited[j]) force=clip(force,a.jointrange+((w%a.jointrange_rows)*a.joints+j)*2);
  a.qfrc[t]=force;
}
}
}
extern "C" {
const char* flash_mjwarp_actuation_source_id() { return FLASH_MJW_ACTUATION_SOURCE_ID; }
std::size_t flash_mjwarp_actuation_arguments_bytes() { return sizeof(flash::mjwarp::ActuationInputs); }
int flash_mjwarp_actuation_arch() { return FLASH_MJW_ACTUATION_ARCH; }
int flash_mjwarp_actuation_launch(const flash::mjwarp::ActuationInputs* a,int phase,cudaStream_t stream) {
  if (!a || a->worlds<1 || a->nv<1 || a->controls<1 || a->actuators<1 || a->joints<1 || a->moments<1 ||
      a->gain_rows<1 || a->bias_rows<1 || a->ctrlrange_rows<1 || a->forcerange_rows<1 || a->jointrange_rows<1)
    return int(cudaErrorInvalidValue);
  using namespace flash::mjwarp;
  if (phase==0) actuator_force<<<(a->worlds*a->actuators+255)/256,256,0,stream>>>(*a);
  else if (phase==1) project_force<<<(a->worlds*a->actuators+255)/256,256,0,stream>>>(*a);
  else if (phase==2) gravcomp_limits<<<(a->worlds*a->nv+255)/256,256,0,stream>>>(*a);
  else return int(cudaErrorInvalidValue);
  return int(cudaGetLastError());
}
}
