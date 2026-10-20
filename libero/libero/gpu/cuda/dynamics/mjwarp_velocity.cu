// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Adapted from pinned MJWarp smooth.py::_comvel_root/_comvel_branch.
#include "mjwarp_velocity.cuh"
#include "mjwarp_spatial.cuh"
#ifndef FLASH_MJW_VELOCITY_SOURCE_ID
#error "Build through flash_mjwarp_velocity_fp32"
#endif
namespace flash::mjwarp {
namespace {
__global__ void velocity_root(VelocityInputs a) {
  int t = blockIdx.x*blockDim.x + threadIdx.x;
  if (t < a.worlds*6) a.cvel[(t/6*a.bodies)*6 + t%6] = 0.0f;
}

// Every branch is a complete world-to-leaf path. A private prefix eliminates
// inter-thread reads/writes of common ancestors; only their canonical owner
// publishes outputs. Joint/DOF arithmetic order follows the pinned source.
__global__ void velocity_branch(VelocityInputs a) {
  int t = blockIdx.x*blockDim.x + threadIdx.x;
  if (t >= a.worlds*a.branches) return;
  int w=t/a.branches, branch=t%a.branches;
  const float* qvel = a.qvel+w*a.nv;
  const float* cdof = a.cdof+w*a.nv*6;
  float v[6] = {};
  for (int at=a.branch_start[branch]; at<a.branch_start[branch+1]; ++at) {
    int body=a.branch_bodies[at], dof=a.body_dofadr[body];
    bool publish = a.body_owner[body] == branch;
    int first=a.body_jntadr[body], count=a.body_jntnum[body];
    for (int joint=first; joint<first+count; ++joint) {
      int type=a.jnt_type[joint];
      // Pinned JointType: FREE=0, BALL=1, SLIDE=2, HINGE=3, admitted by host.
      if (type == 0) {
        for (int j=0; j<3; ++j) {
          for (int k=0; k<6; ++k) v[k] += cdof[(dof+j)*6+k]*qvel[dof+j];
          if (publish) for (int k=0; k<6; ++k) a.cdof_dot[(w*a.nv+dof+j)*6+k] = 0.0f;
        }
        dof += 3;
      }
      int width = type==0 || type==1 ? 3 : 1;
      // Ball/free rotations all use the same pre-rotation velocity.
      if (publish) for (int j=0; j<width; ++j) {
        float dot[6]; motion_cross(v, cdof+(dof+j)*6, dot);
        for (int k=0; k<6; ++k) a.cdof_dot[(w*a.nv+dof+j)*6+k] = dot[k];
      }
      for (int j=0; j<width; ++j)
        for (int k=0; k<6; ++k) v[k] += cdof[(dof+j)*6+k]*qvel[dof+j];
      dof += width;
    }
    if (publish) for (int k=0; k<6; ++k) a.cvel[(w*a.bodies+body)*6+k] = v[k];
  }
}
}
}
extern "C" {
const char* flash_mjwarp_velocity_source_id() { return FLASH_MJW_VELOCITY_SOURCE_ID; }
std::size_t flash_mjwarp_velocity_arguments_bytes() { return sizeof(flash::mjwarp::VelocityInputs); }
int flash_mjwarp_velocity_arch() { return FLASH_MJW_VELOCITY_ARCH; }
int flash_mjwarp_velocity_launch(const flash::mjwarp::VelocityInputs* a, int phase, cudaStream_t stream) {
  if (!a || a->worlds<1 || a->nv<1 || a->branches<1) return int(cudaErrorInvalidValue);
  if (phase==0) flash::mjwarp::velocity_root<<<(a->worlds*6+255)/256,256,0,stream>>>(*a);
  else if (phase==1) flash::mjwarp::velocity_branch<<<(a->worlds*a->branches+255)/256,256,0,stream>>>(*a);
  else return int(cudaErrorInvalidValue);
  return int(cudaGetLastError());
}
}
