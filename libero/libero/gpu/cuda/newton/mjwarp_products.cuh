// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Common FP32 dot products from pinned MJWarp support.py::mul_m_kernel and
// solver.py::_linesearch_jv_fused_kernel/_solve_init_jaref_kernel.
#pragma once

namespace flash::mjwarp {
// Ordinary device functions: shared source for initialization and line search.
// Callers retain their distinct row scheduling, skip masks and commit rules.
static __device__ float mass_row_dot(int row, const int* row_address,
    const int* column, const int* mass_address, const float* mass, const float* vector) {
  float sum = 0.f;
  for (int k = row_address[row]; k < row_address[row + 1]; ++k)
    sum += mass[mass_address[k]] * vector[column[k]];
  return sum;
}

static __device__ float jacobian_partial_dot(int nv, int start, int count,
                                             const float* row, const float* vector) {
  float sum = 0.f;
  for (int k = 0; k < count; ++k) {
    int i = start + k;
    if (i < nv) sum += row[i] * vector[i];
  }
  return sum;
}
}
