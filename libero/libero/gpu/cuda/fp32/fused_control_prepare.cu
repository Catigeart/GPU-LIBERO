// Copyright 2025 The Newton Developers
// Copyright 2026 Flash-LIBERO contributors
// SPDX-License-Identifier: Apache-2.0
// Current site point -> full Jacobian -> Panda OSC gathering, per world.
#include "fused_control_prepare.cuh"
#ifndef FLASH_FP32_CONTROL_PREPARE_SOURCE_ID
#error "Build through flash_fp32_fused_control_prepare"
#endif
namespace flash::fp32 {
template<int Threads>
__global__ void control_prepare(ControlPrepareInputs input,const int* policy_source) {
  const auto& a=input.gather;
  const auto& j=input.jacobian;
  const auto& g=j.geometry;
  const int w=blockIdx.x,x=threadIdx.x;
  const float* point=a.position+3*(w*a.nsites+a.site);
  if (x<3) input.points[3*w+x]=point[x];
  if (x==0 && policy_source) input.policy[w]=policy_source[w];
  // Query jobs read the live site position directly, so publishing points
  // does not need a producer barrier. Keep the full public JP/JR arrays.
  const int body=j.body[w];
  for (int dof=x;dof<g.dofs;dof+=Threads) {
    float linear[3]={},angular[3]={};
    if (g.ancestor[body*g.ancestor_stride+dof]) {
      const float* motion=g.cdof+6*(w*g.dofs+dof);
      const float* center=g.com+3*(w*g.bodies+g.root[body]);
      const float dx=point[0]-center[0],dy=point[1]-center[1],dz=point[2]-center[2];
      const float cross[3]={motion[1]*dz-motion[2]*dy,motion[2]*dx-motion[0]*dz,motion[0]*dy-motion[1]*dx};
      for (int k=0;k<3;++k) {angular[k]=motion[k];linear[k]=motion[k+3]+cross[k];}
    }
    for (int k=0;k<3;++k) {
      const int at=(3*w+k)*g.dofs+dof;
      j.position[at]=linear[k];j.rotation[at]=angular[k];
    }
  }
  __syncthreads(); // all query rows are ready for selected gathers/EE velocity
  if (x<49) { int at=a.mi[x];a.out_mass[49*w+x]=at>=0 ? a.mass[w*a.nmass+at] : 0.f; }
  if (x<42) {
    int axis=x/7,dof=a.vi[x%7];
    a.out_jac[42*w+x]=(axis<3 ? a.jp[(3*w+axis)*a.nv+dof] : a.jr[(3*w+axis-3)*a.nv+dof]);
  }
  if (x<7) {
    a.out_q[7*w+x]=a.qpos[w*a.nq+a.qi[x]];a.out_v[7*w+x]=a.qvel[w*a.nv+a.vi[x]];
    a.out_bias[7*w+x]=a.bias[w*a.nv+a.vi[x]];
  }
  if (x<3) a.out_pos[3*w+x]=a.position[3*(w*a.nsites+a.site)+x];
  if (x<9) a.out_rot[9*w+x]=a.rotation[9*(w*a.nsites+a.site)+x];
  // Each warp owns a spatial component; lane-local sums then one shuffle
  // reduction replace six serial full-NV loops. The seven selected robot
  // columns above are not sufficient: object DOFs also enter EE velocity.
  const int lane=x%32,warp=x/32;
  for (int axis=warp;axis<6;axis+=Threads/32) {
    const float* row=axis<3 ? a.jp+(3*w+axis)*a.nv : a.jr+(3*w+axis-3)*a.nv;
    float value=0;
    for (int k=lane;k<a.nv;k+=32) value+=row[k]*a.qvel[w*a.nv+k];
    for (int shift=16;shift;shift/=2) value+=__shfl_down_sync(0xffffffffu,value,shift);
    if (!lane) a.out_vel[6*w+axis]=value;
  }
}
}
extern "C" {
const char* flash_fp32_control_prepare_source_id() {return FLASH_FP32_CONTROL_PREPARE_SOURCE_ID;}
int flash_fp32_control_prepare_arch() {return FLASH_CUDA_ARCH;}
std::size_t flash_fp32_control_prepare_arguments_bytes() {return sizeof(flash::fp32::ControlPrepareInputs);}
int flash_fp32_control_prepare_launch(const flash::fp32::ControlPrepareInputs* a,const int* policy_source,cudaStream_t stream) {
  if (!a || a->gather.worlds<1 || a->gather.nv<7 || a->gather.nq<7 || a->gather.nmass<1 ||
      a->gather.nsites<1 || a->gather.site<0 || a->gather.site>=a->gather.nsites ||
      a->gather.worlds!=a->jacobian.geometry.worlds || a->gather.nv!=a->jacobian.geometry.dofs ||
      !a->points || !a->policy || !a->jacobian.position || !a->jacobian.rotation) return cudaErrorInvalidValue;
  flash::fp32::control_prepare<256><<<a->gather.worlds,256,0,stream>>>(*a,policy_source);
  return cudaGetLastError();
}
}
