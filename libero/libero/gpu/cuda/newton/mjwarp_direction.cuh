// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstddef>

namespace flash::mjwarp {
// Host-only launch descriptor. Every array is resident and caller-owned.
struct DirectionInputs {
  int worlds, nv, pad, ma_stride, smooth_stride, force_stride;
  const bool* done;
  const float *ma, *smooth, *force, *hessian;
  float *grad, *grad_dot, *decrement, *grad_scale;
  bool* search_unchanged;
  float *factor, *search, *search_dot;
};
}
extern "C" {
const char* flash_mjwarp_direction_source_id();
std::size_t flash_mjwarp_direction_arguments_bytes();
int flash_mjwarp_direction_arch();
// 0: gradient then direction; 1: gradient only; 2: direction only.
int flash_mjwarp_direction_launch(const flash::mjwarp::DirectionInputs*, int,
                                 cudaStream_t);
}
