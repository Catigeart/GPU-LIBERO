// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Pinned forward.py stateless fixed/affine scalar-joint actuation path.
#include "mjwarp_actuation.cuh"
#include "mjwarp_smooth_force.cuh"
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
// One CTA owns a whole world. Force production and zeroing are independent;
// the two barriers replace the original completion boundaries around CSR adds.
template<int Threads,bool Compose=false,bool Masked=false>
__global__ void world_actuation(ActuationInputs a,SmoothForceInputs s) {
  const int w=blockIdx.x,x=threadIdx.x;
  __shared__ unsigned present[Masked ? 4 : 1];
  if constexpr (Masked) {
    bool active=false;
    if(x<s.bodies) {
      const float* ft=s.external+6*(w*s.bodies+x);
      active=ft[0]!=0 || ft[1]!=0 || ft[2]!=0 || ft[3]!=0 || ft[4]!=0 || ft[5]!=0;
    }
    const unsigned word=__ballot_sync(0xffffffffu,active);
    if(!(x%32)) present[x/32]=word;
  }
  for (int d=x;d<a.nv;d+=Threads) a.qfrc[w*a.nv+d]=0.f;
  for (int u=x;u<a.actuators;u+=Threads) {
    const int t=w*a.actuators+u;
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
  __syncthreads();
  for (int u=x;u<a.actuators;u+=Threads) {
    const int t=w*a.actuators+u,at=a.moment_rowadr[t],count=a.moment_rownnz[t];
    const float force=a.force[t];
    for (int i=0;i<count;++i) {
      const int slot=w*a.moments+at+i,dof=a.moment_colind[slot];
      atomicAdd(a.qfrc+w*a.nv+dof,a.moment[slot]*force);
    }
  }
  __syncthreads();
  for (int d=x;d<a.nv;d+=Threads) {
    const int t=w*a.nv+d,j=a.dof_joint[d];
    float force=a.qfrc[t];
    if (a.gravity_enabled && a.actgravcomp[j]) force+=a.gravcomp[t];
    if (a.jointlimited[j]) force=clip(force,a.jointrange+((w%a.jointrange_rows)*a.joints+j)*2);
    a.qfrc[t]=force;
    if constexpr (Compose) {
      // The current DOF owns both publications; no force reload or barrier.
      float value=s.passive[w*s.vector_stride+d]-s.bias[w*s.vector_stride+d]+force+s.applied[w*s.vector_stride+d];
      float sum=0.f;
      const bool any=Masked ? bool(present[0]|present[1]|present[2]|present[3]) : true;
      if(any) {
        const float* cd = s.cdof + (w * s.nv + d) * 6;
        // cd is (angular, linear); applied wrench is (force, torque), in world axes.
        float jac[6] = {cd[3], cd[4], cd[5], cd[0], cd[1], cd[2]};
        int body = s.dof_body[d];
        const float* com = s.subtree_com + (w * s.bodies + s.body_root[body]) * 3;
        // Model's depth-first order makes each subtree a contiguous body interval.
        for (int b = body; b < body + s.subtree_size[body]; ++b) {
          const float* ft = s.external + (w * s.bodies + b) * 6;
          if constexpr (Masked) {if (!(present[b/32] & (1u<<(b%32)))) continue;}
          else if (ft[0] == 0 && ft[1] == 0 && ft[2] == 0 && ft[3] == 0 && ft[4] == 0 && ft[5] == 0) continue;
          const float* point = s.body_com + (w * s.bodies + b) * 3;
          float dx = point[0] - com[0], dy = point[1] - com[1], dz = point[2] - com[2];
          float cx = cd[1] * dz - cd[2] * dy;
          float cy = cd[2] * dx - cd[0] * dz;
          float cz = cd[0] * dy - cd[1] * dx;
          float dot6 = 0.0f;
          for (int k = 0; k < 6; ++k) dot6 += jac[k] * ft[k];
          float dot3 = cx * ft[0] + cy * ft[1] + cz * ft[2];
          sum += dot6 + dot3;
        }
      }
      s.smooth[w*s.vector_stride+d]=value+sum;
    }
  }
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
int flash_mjwarp_actuation_fused(const flash::mjwarp::ActuationInputs* a,cudaStream_t stream) {
  if (!a || a->worlds<1 || a->nv<1 || a->controls<1 || a->actuators<1 || a->joints<1 || a->moments<1 ||
      a->gain_rows<1 || a->bias_rows<1 || a->ctrlrange_rows<1 || a->forcerange_rows<1 || a->jointrange_rows<1)
    return cudaErrorInvalidValue;
  flash::mjwarp::world_actuation<128><<<a->worlds,128,0,stream>>>(*a,{});
  return cudaGetLastError();
}

std::size_t flash_mjwarp_actuation_smooth_bytes() { return sizeof(flash::mjwarp::SmoothForceInputs); }
int flash_mjwarp_actuation_smooth(const flash::mjwarp::ActuationInputs* a,
                                const flash::mjwarp::SmoothForceInputs* s,cudaStream_t stream) {
  if (!a || !s || a->worlds<1 || a->nv<1 || a->controls<1 || a->actuators<1 || a->joints<1 || a->moments<1 ||
      a->gain_rows<1 || a->bias_rows<1 || a->ctrlrange_rows<1 || a->forcerange_rows<1 || a->jointrange_rows<1 ||
      s->worlds!=a->worlds || s->nv!=a->nv || s->vector_stride!=a->nv || s->bodies<1 || s->actuator!=a->qfrc)
    return cudaErrorInvalidValue;
  if(s->bodies<=128) flash::mjwarp::world_actuation<128,true,true><<<a->worlds,128,0,stream>>>(*a,*s);
  else flash::mjwarp::world_actuation<128,true,false><<<a->worlds,128,0,stream>>>(*a,*s);
  return cudaGetLastError();
}

}
