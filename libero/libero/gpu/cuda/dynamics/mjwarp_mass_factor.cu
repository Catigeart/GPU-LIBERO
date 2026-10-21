// Copyright 2025 The Newton Developers
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0 Readable adaptation
// of pinned MJWarp c384ac51df67eb92ab7657cc472ff7494dc036bf: smooth.py
// small/tile Cholesky factorize and factorize_solve blocks. Opposite-triangle
// clearing follows Warp 1.18 native/tile_cholesky.h.
#include "mjwarp_mass_factor.cuh"
#include <cusolverdx.hpp>
#ifndef FLASH_MJW_MASS_FACTOR_SOURCE_ID
#error "Build through flash_mjwarp_mass_factor_fp32"
#endif
namespace flash::mjwarp {
namespace {
// Each thread owns an independent block. Solve=true retains the upstream
// interleaving of factorization and forward solve, including
// reciprocal-multiply.
template <int Size, bool Solve>
__global__ void scalar_factor(MassFactorInputs a) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.worlds * a.blocks)
    return;
  int w = t / a.blocks, start = a.block_start[t % a.blocks];
  int adr = a.factor_address[start], base = a.matrix_row[start];
  const float *matrix = a.matrix + w * a.matrix_width + base;
  const float *y = Solve ? a.rhs + w * a.rhs_width + start : nullptr;
  float *x = Solve ? a.solution + w * a.solution_width + start : nullptr;
  if (adr == -2) {
    for (int i = 0; i < Size; ++i) {
      float inverse = 1.0f / matrix[i];
      a.diagonal_inverse[w * a.nv + start + i] = inverse;
      if constexpr (Solve)
        x[i] = inverse * y[i];
    }
    return;
  }
  float *upper = a.factor + w * a.factor_width + adr;
  for (int i = 0; i < Size; ++i) {
    float value = matrix[i * (i + 1) / 2 + i];
    float rhs_value = Solve ? y[i] : 0.0f;
    for (int k = 0; k < i; ++k) {
      float u = upper[k * Size + i];
      value -= u * u;
      if constexpr (Solve)
        rhs_value -= u * x[k];
    }
    float diagonal = sqrtf(value);
    upper[i * Size + i] = diagonal;
    float inverse = 1.0f / diagonal;
    if constexpr (Solve)
      x[i] = rhs_value * inverse;
    for (int j = i + 1; j < Size; ++j) {
      float entry = matrix[j * (j + 1) / 2 + i];
      for (int k = 0; k < i; ++k)
        entry -= upper[k * Size + i] * upper[k * Size + j];
      upper[i * Size + j] = entry * inverse;
    }
  }
  if constexpr (Solve) {
    for (int reverse = 0; reverse < Size; ++reverse) {
      int i = Size - 1 - reverse;
      float value = x[i];
      for (int k = i + 1; k < Size; ++k)
        value -= upper[i * Size + k] * x[k];
      x[i] = value / upper[i * Size + i];
    }
  }
}

namespace dx = cusolverdx;
using Potrf9 =
    decltype(dx::Size<9>() + dx::Precision<float>() +
             dx::Type<dx::type::real>() + dx::Function<dx::function::potrf>() +
             dx::FillMode<dx::fill_mode::upper>() +
             dx::Arrangement<dx::row_major>() + dx::SM<800>() +
             dx::BlockDim<32>() + dx::Block());
using Potrs9 =
    decltype(dx::Size<9, 9, 1>() + dx::Precision<float>() +
             dx::Type<dx::type::real>() + dx::Function<dx::function::potrs>() +
             dx::FillMode<dx::fill_mode::upper>() +
             dx::Arrangement<dx::row_major, dx::col_major>() + dx::SM<800>() +
             dx::BlockDim<32>() + dx::Block());

template <bool Solve> __global__ void tile_factor9(MassFactorInputs a) {
  int w = blockIdx.x, block = blockIdx.y, tid = threadIdx.x;
  int start = a.block_start[block],
      adr = w * a.factor_width + a.factor_address[start];
  __shared__ __align__(16) float upper[81], rhs[9];
  __shared__ int info;
  for (int k = tid; k < 81; k += 32) {
    int index = a.gather[block * 81 + k];
    // Warp tile_load_indexed predicates either out-of-range side to zero.
    upper[k] = index >= 0 && index < a.matrix_width
                   ? a.matrix[w * a.matrix_width + index]
                   : 0.0f;
  }
  if constexpr (Solve)
    for (int k = tid; k < 9; k += 32)
      rhs[k] = a.rhs[w * a.rhs_width + start + k];
  if (tid == 0)
    info = 0;
  __syncthreads();
  Potrf9().execute(upper, 9, &info);
  __syncthreads();
  for (int k = tid; k < 81; k += 32) {
    if (k / 9 > k % 9)
      upper[k] = 0.0f;
    a.factor[adr + k] = upper[k];
  }
  if constexpr (Solve) {
    __syncthreads();
    Potrs9().execute(upper, 9, rhs, 9);
    __syncthreads();
    for (int k = tid; k < 9; k += 32)
      a.solution[w * a.solution_width + start + k] = rhs[k];
  }
}
using Potrf9World =
    decltype(dx::Size<9>() + dx::Precision<float>() +
             dx::Type<dx::type::real>() + dx::Function<dx::function::potrf>() +
             dx::FillMode<dx::fill_mode::upper>() +
             dx::Arrangement<dx::row_major>() + dx::SM<800>() +
             dx::BlockDim<64>() + dx::Block());
using Potrs9World =
    decltype(dx::Size<9, 9, 1>() + dx::Precision<float>() +
             dx::Type<dx::type::real>() + dx::Function<dx::function::potrs>() +
             dx::FillMode<dx::fill_mode::upper>() +
             dx::Arrangement<dx::row_major, dx::col_major>() + dx::SM<800>() +
             dx::BlockDim<64>() + dx::Block());

// Literal scalar1/6 factor/solve loops followed by a whole-CTA9-DOF SDK
// operation. Scalar groups are independent; bounded admission limits work
// serialized before the SDK, whose block barriers require all64threads.
template <bool Solve>
__global__ void world_factor(MassFactorInputs one, MassFactorInputs six,
                             MassFactorInputs nine) {
  const int w = blockIdx.x, tid = threadIdx.x;
#pragma unroll
  for (int group = 0; group < 2; ++group) {
    const MassFactorInputs a = group ? six : one;
    const int size = group ? 6 : 1;
    for (int block = tid; block < a.blocks; block += 64) {
      const int start = a.block_start[block];
      int adr = a.factor_address[start], base = a.matrix_row[start];
      const float *matrix = a.matrix + w * a.matrix_width + base;
      const float *y = Solve ? a.rhs + w * a.rhs_width + start : nullptr;
      float *x = Solve ? a.solution + w * a.solution_width + start : nullptr;
      if (adr == -2) {
        for (int i = 0; i < size; ++i) {
          float inverse = 1.0f / matrix[i];
          a.diagonal_inverse[w * a.nv + start + i] = inverse;
          if constexpr (Solve)
            x[i] = inverse * y[i];
        }
        continue;
      }
      float *upper = a.factor + w * a.factor_width + adr;
      for (int i = 0; i < size; ++i) {
        float value = matrix[i * (i + 1) / 2 + i];
        float rhs_value = Solve ? y[i] : 0.0f;
        for (int k = 0; k < i; ++k) {
          float u = upper[k * size + i];
          value -= u * u;
          if constexpr (Solve)
            rhs_value -= u * x[k];
        }
        float diagonal = sqrtf(value);
        upper[i * size + i] = diagonal;
        float inverse = 1.0f / diagonal;
        if constexpr (Solve)
          x[i] = rhs_value * inverse;
        for (int j = i + 1; j < size; ++j) {
          float entry = matrix[j * (j + 1) / 2 + i];
          for (int k = 0; k < i; ++k)
            entry -= upper[k * size + i] * upper[k * size + j];
          upper[i * size + j] = entry * inverse;
        }
      }
      if constexpr (Solve) {
        for (int reverse = 0; reverse < size; ++reverse) {
          int i = size - 1 - reverse;
          float value = x[i];
          for (int k = i + 1; k < size; ++k)
            value -= upper[i * size + k] * x[k];
          x[i] = value / upper[i * size + i];
        }
      }
    }
  }
  if (!nine.blocks)
    return; // block-uniform; no cooperative work exists
  const MassFactorInputs a = nine;
  int start = a.block_start[0],
      adr = w * a.factor_width + a.factor_address[start];
  __shared__ __align__(16) float upper[81], rhs[9];
  __shared__ int info;
  for (int k = tid; k < 81; k += 64) {
    int index = a.gather[0 * 81 + k];
    // Warp tile_load_indexed predicates either out-of-range side to zero.
    upper[k] = index >= 0 && index < a.matrix_width
                   ? a.matrix[w * a.matrix_width + index]
                   : 0.0f;
  }
  if constexpr (Solve)
    for (int k = tid; k < 9; k += 64)
      rhs[k] = a.rhs[w * a.rhs_width + start + k];
  if (tid == 0)
    info = 0;
  __syncthreads();
  Potrf9World().execute(upper, 9, &info);
  __syncthreads();
  for (int k = tid; k < 81; k += 64) {
    if (k / 9 > k % 9)
      upper[k] = 0.0f;
    a.factor[adr + k] = upper[k];
  }
  if constexpr (Solve) {
    __syncthreads();
    Potrs9World().execute(upper, 9, rhs, 9);
    __syncthreads();
    for (int k = tid; k < 9; k += 64)
      a.solution[w * a.solution_width + start + k] = rhs[k];
  }
}

} // namespace
} // namespace flash::mjwarp
extern "C" {
const char *flash_mjwarp_mass_factor_source_id() {
  return FLASH_MJW_MASS_FACTOR_SOURCE_ID;
}
std::size_t flash_mjwarp_mass_factor_arguments_bytes() {
  return sizeof(flash::mjwarp::MassFactorInputs);
}
int flash_mjwarp_mass_factor_arch() { return FLASH_MJW_MASS_FACTOR_ARCH; }

int flash_mjwarp_mass_factor_groups(const flash::mjwarp::MassFactorInputs *one,
                                    const flash::mjwarp::MassFactorInputs *six,
                                    const flash::mjwarp::MassFactorInputs *nine,
                                    int solve, cudaStream_t stream) {
  if (!one || !six || !nine || one->worlds < 1 || one->worlds != six->worlds ||
      one->worlds != nine->worlds || one->blocks < 0 || six->blocks < 0 ||
      nine->blocks < 0 || nine->blocks > 1 || one->blocks + six->blocks > 32 ||
      (solve != 0 && solve != 1))
    return int(cudaErrorInvalidValue);
  if (solve)
    flash::mjwarp::world_factor<true>
        <<<one->worlds, 64, 0, stream>>>(*one, *six, *nine);
  else
    flash::mjwarp::world_factor<false>
        <<<one->worlds, 64, 0, stream>>>(*one, *six, *nine);
  return int(cudaGetLastError());
}
int flash_mjwarp_mass_factor_launch(const flash::mjwarp::MassFactorInputs *a,
                                    int kind, int size, int solve,
                                    cudaStream_t stream) {
  using namespace flash::mjwarp;
  if (!a || a->worlds < 1 || a->blocks < 1 || (solve != 0 && solve != 1))
    return int(cudaErrorInvalidValue);
  int grid = (a->worlds * a->blocks + 63) / 64;
  if (kind == 0 && size == 1) {
    if (solve)
      scalar_factor<1, true><<<grid, 64, 0, stream>>>(*a);
    else
      scalar_factor<1, false><<<grid, 64, 0, stream>>>(*a);
  } else if (kind == 0 && size == 6) {
    if (solve)
      scalar_factor<6, true><<<grid, 64, 0, stream>>>(*a);
    else
      scalar_factor<6, false><<<grid, 64, 0, stream>>>(*a);
  } else if (kind == 1 && size == 9) {
    if (solve)
      tile_factor9<true><<<dim3(a->worlds, a->blocks), 32, 0, stream>>>(*a);
    else
      tile_factor9<false><<<dim3(a->worlds, a->blocks), 32, 0, stream>>>(*a);
  } else
    return int(cudaErrorInvalidValue);
  return int(cudaGetLastError());
}
}
