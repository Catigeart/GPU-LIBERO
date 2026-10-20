// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Pinned MJWarp smooth.py::rne(flg_acc=False) and its five numerical phases.
#include "mjwarp_bias.cuh"
#include "mjwarp_spatial.cuh"
#ifndef FLASH_MJW_BIAS_SOURCE_ID
#error "Build through flash_mjwarp_bias_fp32"
#endif
namespace flash::mjwarp {
namespace {
__global__ void acceleration_root(BiasInputs a) {
  int w=blockIdx.x*blockDim.x+threadIdx.x;
  if (w>=a.worlds) return;
  float* acc=a.cacc+w*a.bodies*6;
  for (int k=0; k<3; ++k) { acc[k]=0.0f; acc[k+3]=-a.gravity[(w%a.gravity_rows)*3+k]; }
}

__global__ void acceleration_branch(BiasInputs a) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if (t>=a.worlds*a.branches) return;
  int w=t/a.branches, branch=t%a.branches;
  float acc[6]; for (int k=0; k<6; ++k) acc[k]=a.cacc[w*a.bodies*6+k];
  for (int at=a.branch_start[branch]; at<a.branch_start[branch+1]; ++at) {
    int b=a.branch_bodies[at], first=a.body_dofadr[b];
    for (int j=0; j<a.body_dofnum[b]; ++j)
      for (int k=0; k<6; ++k) acc[k] += a.cdof_dot[(w*a.nv+first+j)*6+k]*a.qvel[w*a.nv+first+j];
    if (a.body_owner[b]==branch)
      for (int k=0; k<6; ++k) a.cacc[(w*a.bodies+b)*6+k]=acc[k];
  }
}

__global__ void body_force(BiasInputs a) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if (t>=a.worlds*a.bodies) return;
  float* out=a.cfrc_int+t*6;
  if (t%a.bodies==0) { for (int k=0; k<6; ++k) out[k]=0.0f; return; }
  const float *inertia=a.cinert+t*10, *velocity=a.cvel+t*6;
  float force[6], momentum[6], cross[6];
  inert_vec(inertia,a.cacc+t*6,force);
  inert_vec(inertia,velocity,momentum);
  force_cross(velocity,momentum,cross);
  for (int k=0; k<6; ++k) out[k]=force[k]+cross[k];
}

__global__ void force_backward(BiasInputs a, const int* layer, int count) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if (t>=a.worlds*count) return;
  int w=t/count, b=layer[t%count];
  if (b==0) return;
  // Unlike CRB, pinned RNE also publishes root forces into world (parent=0).
  const float* child=a.cfrc_int+(w*a.bodies+b)*6;
  float* parent=a.cfrc_int+(w*a.bodies+a.body_parent[b])*6;
  for (int k=0; k<6; ++k) atomicAdd(parent+k,child[k]);
}

__global__ void project_bias(BiasInputs a) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if (t>=a.worlds*a.nv) return;
  int w=t/a.nv, dof=t%a.nv;
  const float* force=a.cfrc_int+(w*a.bodies+a.dof_body[dof])*6;
  float value=0.0f;
  for (int k=0; k<6; ++k) value+=a.cdof[t*6+k]*force[k];
  a.qfrc_bias[t]=value;
}
}
}
extern "C" {
const char* flash_mjwarp_bias_source_id() { return FLASH_MJW_BIAS_SOURCE_ID; }
std::size_t flash_mjwarp_bias_arguments_bytes() { return sizeof(flash::mjwarp::BiasInputs); }
int flash_mjwarp_bias_arch() { return FLASH_MJW_BIAS_ARCH; }
int flash_mjwarp_bias_launch(const flash::mjwarp::BiasInputs* a, int phase,
                            const int* layer, int count, cudaStream_t stream) {
  if (!a || a->worlds<1 || a->nv<1 || a->bodies<2 || a->branches<1 || a->gravity_rows<1)
    return int(cudaErrorInvalidValue);
  using namespace flash::mjwarp;
  if (phase==0) acceleration_root<<<(a->worlds+255)/256,256,0,stream>>>(*a);
  else if (phase==1) acceleration_branch<<<(a->worlds*a->branches+255)/256,256,0,stream>>>(*a);
  else if (phase==2) body_force<<<(a->worlds*a->bodies+255)/256,256,0,stream>>>(*a);
  else if (phase==3) {
    if (!layer || count<1) return int(cudaErrorInvalidValue);
    force_backward<<<(a->worlds*count+255)/256,256,0,stream>>>(*a,layer,count);
  } else if (phase==4) project_bias<<<(a->worlds*a->nv+255)/256,256,0,stream>>>(*a);
  else return int(cudaErrorInvalidValue);
  return int(cudaGetLastError());
}
}
