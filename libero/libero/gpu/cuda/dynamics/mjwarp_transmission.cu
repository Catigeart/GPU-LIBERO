// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Pinned smooth.py scalar-joint transmission and forward.py actuator velocity.
#include "mjwarp_transmission.cuh"
#ifndef FLASH_MJW_TRANSMISSION_SOURCE_ID
#error "Build through flash_mjwarp_transmission_fp32"
#endif
namespace flash::mjwarp {
namespace {
__global__ void scalar_transmission(TransmissionInputs a) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if (t>=a.worlds*a.actuators) return;
  int w=t/a.actuators,u=t%a.actuators;
  // actuator_joint borrows packed vec2i; the second id is unused for joints.
  int joint=a.actuator_joint[2*u];
  float gear=a.gear[((w%a.gear_rows)*a.actuators+u)*6];
  a.length[t]=a.qpos[w*a.nq+a.joint_qposadr[joint]]*gear;
  a.rownnz[t]=1;
  // Preserve pinned per-world atomic row allocation, including zero gear.
  // Admission guarantees capacity >= actuators and the counter starts at zero.
  int row=atomicAdd(a.total+w,1);
  a.rowadr[t]=row;
  a.column[w*a.moments+row]=a.joint_dofadr[joint];
  a.moment[w*a.moments+row]=gear;
}
__global__ void actuator_velocity(TransmissionInputs a) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if (t>=a.worlds*a.actuators) return;
  int w=t/a.actuators,at=a.rowadr[t],count=a.rownnz[t];
  float velocity=0.0f;
  for (int i=0;i<count;++i) {
    int slot=w*a.moments+at+i;
    velocity+=a.moment[slot]*a.qvel[w*a.nv+a.column[slot]];
  }
  a.velocity[t]=velocity;
}

// A single world owns the append counter and all scalar rows. Length, moment
// and optional velocity share the producer's private gear/DOF values. No lane
// reads another lane's newly allocated row.
template<bool Velocity,int Threads,bool Step1=false>
__global__ void world_transmission(TransmissionInputs a,int sensors=0,
                                   float* sensor_data=nullptr,float* energy=nullptr) {
  const int w=blockIdx.x,lane=threadIdx.x;
  // The caller's substep-entry clears have world-local ownership. They are
  // independent of the actuator rows and reuse this launch without a barrier.
  // Ordinary transmission callers retain their sensor and energy buffers.
  if constexpr (Step1) {
    for (int i=lane;i<sensors;i+=Threads) sensor_data[w*sensors+i]=0.0f;
    if (lane<2) energy[2*w+lane]=0.0f;
  }
  if (lane==0) a.total[w]=0;
  if constexpr (Threads==32) __syncwarp();
  else __syncthreads();
  for (int u=lane;u<a.actuators;u+=Threads) {
    const int t=w*a.actuators+u,joint=a.actuator_joint[2*u],
              dof=a.joint_dofadr[joint];
    const float gear=a.gear[((w%a.gear_rows)*a.actuators+u)*6];
    a.length[t]=a.qpos[w*a.nq+a.joint_qposadr[joint]]*gear;
    a.rownnz[t]=1;
    const int row=atomicAdd(a.total+w,1);
    a.rowadr[t]=row;
    a.column[w*a.moments+row]=dof;
    a.moment[w*a.moments+row]=gear;
    if constexpr (Velocity) {
      float value=0.0f;
      value+=gear*a.qvel[w*a.nv+dof];
      a.velocity[t]=value;
    }
  }
}
}
}
extern "C" {
const char* flash_mjwarp_transmission_source_id() { return FLASH_MJW_TRANSMISSION_SOURCE_ID; }
std::size_t flash_mjwarp_transmission_arguments_bytes() { return sizeof(flash::mjwarp::TransmissionInputs); }
int flash_mjwarp_transmission_arch() { return FLASH_MJW_TRANSMISSION_ARCH; }
int flash_mjwarp_transmission_step1(const flash::mjwarp::TransmissionInputs* a,
                                   int sensors,float* sensor_data,float* energy,
                                   cudaStream_t stream) {
  if (!a || a->worlds<1 || a->actuators<1 || a->nv<1 || a->nq<1 ||
      a->moments<a->actuators || a->gear_rows<1 || !a->total ||
      sensors<0 || (sensors && !sensor_data) || !energy)
    return int(cudaErrorInvalidValue);
  using namespace flash::mjwarp;
  if (a->actuators<=32)
    world_transmission<true,32,true><<<a->worlds,32,0,stream>>>(*a,sensors,sensor_data,energy);
  else
    world_transmission<true,128,true><<<a->worlds,128,0,stream>>>(*a,sensors,sensor_data,energy);
  return int(cudaGetLastError());
}
int flash_mjwarp_transmission_fused(const flash::mjwarp::TransmissionInputs* a,
                                   int velocity,cudaStream_t stream) {
  if (!a || a->worlds<1 || a->actuators<1 || a->nv<1 || a->nq<1 ||
      a->moments<a->actuators || a->gear_rows<1 || !a->total ||
      (velocity!=0 && velocity!=1)) return int(cudaErrorInvalidValue);
  using namespace flash::mjwarp;
  if (a->actuators<=32) {
    if (velocity) world_transmission<true,32><<<a->worlds,32,0,stream>>>(*a);
    else world_transmission<false,32><<<a->worlds,32,0,stream>>>(*a);
  } else {
    if (velocity) world_transmission<true,128><<<a->worlds,128,0,stream>>>(*a);
    else world_transmission<false,128><<<a->worlds,128,0,stream>>>(*a);
  }
  return int(cudaGetLastError());
}
int flash_mjwarp_transmission_launch(const flash::mjwarp::TransmissionInputs* a,int phase,cudaStream_t stream) {
  if (!a || a->worlds<1 || a->nv<1 || a->nq<1 || a->actuators<1 || a->moments<1 || a->gear_rows<1)
    return int(cudaErrorInvalidValue);
  using namespace flash::mjwarp;
  int blocks=(a->worlds*a->actuators+255)/256;
  if (phase==0) scalar_transmission<<<blocks,256,0,stream>>>(*a);
  else if (phase==1) actuator_velocity<<<blocks,256,0,stream>>>(*a);
  else return int(cudaErrorInvalidValue);
  return int(cudaGetLastError());
}
}
