// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Pinned MJWarp math.py: motion_cross, motion_cross_force, inert_vec.
#pragma once
#include <cuda_runtime.h>

namespace flash::mjwarp {
__device__ inline void cross3(const float* a, const float* b, float* out) {
  out[0] = a[1]*b[2] - a[2]*b[1];
  out[1] = a[2]*b[0] - a[0]*b[2];
  out[2] = a[0]*b[1] - a[1]*b[0];
}
__device__ inline void motion_cross(const float* a, const float* b, float* out) {
  float x[3], y[3];
  cross3(a, b, out);
  cross3(a+3, b, x);
  cross3(a, b+3, y);
  for (int k=0; k<3; ++k) out[k+3] = x[k] + y[k];
}
__device__ inline void force_cross(const float* v, const float* f, float* out) {
  float x[3], y[3];
  cross3(v, f, x);
  cross3(v+3, f+3, y);
  for (int k=0; k<3; ++k) out[k] = x[k] + y[k];
  cross3(v, f+3, out+3);
}
__device__ inline void inert_vec(const float* i, const float* v, float* out) {
  out[0] = i[0]*v[0] + i[3]*v[1] + i[4]*v[2] - i[8]*v[4] + i[7]*v[5];
  out[1] = i[3]*v[0] + i[1]*v[1] + i[5]*v[2] + i[8]*v[3] - i[6]*v[5];
  out[2] = i[4]*v[0] + i[5]*v[1] + i[2]*v[2] - i[7]*v[3] + i[6]*v[4];
  out[3] = i[8]*v[1] - i[7]*v[2] + i[9]*v[3];
  out[4] = i[6]*v[2] - i[8]*v[0] + i[9]*v[4];
  out[5] = i[7]*v[0] - i[6]*v[1] + i[9]*v[5];
}
}
