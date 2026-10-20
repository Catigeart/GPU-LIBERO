// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
namespace flash::mjwarp {
struct SmoothForceInputs {
  int worlds, nv, bodies, vector_stride;
  const int *body_root, *dof_body, *subtree_size;
  const float *passive, *bias, *actuator, *applied;
  const float *body_com, *subtree_com, *cdof, *external;
  float* smooth;
};
}
extern "C" {
const char* flash_mjwarp_smooth_force_source_id();
std::size_t flash_mjwarp_smooth_force_arguments_bytes();
int flash_mjwarp_smooth_force_arch();
// 0: assemble, 1: add external projection, 2: both in one kernel.
int flash_mjwarp_smooth_force_launch(const flash::mjwarp::SmoothForceInputs*, int, cudaStream_t);
}
