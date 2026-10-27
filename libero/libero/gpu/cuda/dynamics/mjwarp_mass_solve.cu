// Copyright 2025 The Newton Developers
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0 Readable CUDA port
// of pinned MJWarp c384ac51df67eb92ab7657cc472ff7494dc036bf:
// smooth.py::_small_cholesky_solve_block/_small_cholesky_solve and
// _tile_cholesky_solve_block. The tile path uses the same cuSolverDx POTRS
// as Warp 1.18's tile_cholesky_solve, without Warp/generated CUDA headers.
#include "mjwarp_mass_solve.cuh"
#include <cusolverdx.hpp>

#ifndef FLASH_MJW_MASS_SOLVE_SOURCE_ID
#error "Build through flash_mjwarp_mass_solve_fp32"
#endif

namespace flash::mjwarp {
namespace {
constexpr int Compact = -2;

// One thread per (world, independent mass block), matching the pinned scalar
// path. Read only the upper factor, including when x and y alias exactly.
template <int Size> __global__ void scalar_solve(MassSolveInputs a) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.worlds * a.blocks)
    return;
  int w = t / a.blocks, start = a.block_start[t % a.blocks];
  int address = a.factor_address[start];
  const float *y = a.rhs + w * a.rhs_stride + start;
  float *x = a.solution + w * a.solution_stride + start;
  if (address == Compact) {
    for (int i = 0; i < Size; ++i)
      x[i] = a.diagonal_inverse[w * a.diagonal_stride + start + i] * y[i];
    return;
  }
  const float *upper = a.factor + w * a.factor_stride + address;
  for (int i = 0; i < Size; ++i) {
    float value = y[i];
    for (int k = 0; k < i; ++k)
      value -= upper[k * Size + i] * x[k];
    x[i] = value / upper[i * Size + i];
  }
  for (int reverse = 0; reverse < Size; ++reverse) {
    int i = Size - 1 - reverse;
    float value = x[i];
    for (int k = i + 1; k < Size; ++k)
      value -= upper[i * Size + k] * x[k];
    x[i] = value / upper[i * Size + i];
  }
}

namespace dx = cusolverdx;
using Potrs9 =
    decltype(dx::Size<9, 9, 1>() + dx::Precision<float>() +
             dx::Type<dx::type::real>() + dx::Function<dx::function::potrs>() +
             dx::FillMode<dx::fill_mode::upper>() +
             dx::Arrangement<dx::row_major, dx::col_major>() + dx::SM<(FLASH_CUDA_ARCH * 10)>() +
             dx::BlockDim<64>() + dx::Block());

// One 64-thread block per (world, 9-DOF mass block), with bounded scratch.
// The factor is already Cholesky; no refactorization or densification occurs.
__global__ void tile_solve9(MassSolveInputs a) {
  int w = blockIdx.x, start = a.block_start[blockIdx.y], tid = threadIdx.x;
  int address = w * a.factor_stride + a.factor_address[start];
  __shared__ __align__(16) float upper[81], rhs[9];
  for (int k = tid; k < 81; k += 64)
    upper[k] = a.factor[address + k];
  for (int k = tid; k < 9; k += 64)
    rhs[k] = a.rhs[w * a.rhs_stride + start + k];
  __syncthreads();
  Potrs9().execute(upper, 9, rhs, 9);
  __syncthreads();
  for (int k = tid; k < 9; k += 64)
    a.solution[w * a.solution_stride + start + k] = rhs[k];
}
__global__ void world_solve(MassSolveInputs one, MassSolveInputs six,
                            MassSolveInputs nine) {
  const int w = blockIdx.x, tid = threadIdx.x;
#pragma unroll
  for (int group = 0; group < 2; ++group) {
    const MassSolveInputs a = group ? six : one;
    const int size = group ? 6 : 1;
    for (int block = tid; block < a.blocks; block += 64) {
      const int start = a.block_start[block];
      int address = a.factor_address[start];
      const float *y = a.rhs + w * a.rhs_stride + start;
      float *x = a.solution + w * a.solution_stride + start;
      if (address == Compact) {
        for (int i = 0; i < size; ++i)
          x[i] = a.diagonal_inverse[w * a.diagonal_stride + start + i] * y[i];
        continue;
      }
      const float *upper = a.factor + w * a.factor_stride + address;
      for (int i = 0; i < size; ++i) {
        float value = y[i];
        for (int k = 0; k < i; ++k)
          value -= upper[k * size + i] * x[k];
        x[i] = value / upper[i * size + i];
      }
      for (int reverse = 0; reverse < size; ++reverse) {
        int i = size - 1 - reverse;
        float value = x[i];
        for (int k = i + 1; k < size; ++k)
          value -= upper[i * size + k] * x[k];
        x[i] = value / upper[i * size + i];
      }
    }
  }
  if (!nine.blocks)
    return;
  const MassSolveInputs a = nine;
  const int start = a.block_start[0];
  int address = w * a.factor_stride + a.factor_address[start];
  __shared__ __align__(16) float upper[81], rhs[9];
  for (int k = tid; k < 81; k += 64)
    upper[k] = a.factor[address + k];
  for (int k = tid; k < 9; k += 64)
    rhs[k] = a.rhs[w * a.rhs_stride + start + k];
  __syncthreads();
  Potrs9().execute(upper, 9, rhs, 9);
  __syncthreads();
  for (int k = tid; k < 9; k += 64)
    a.solution[w * a.solution_stride + start + k] = rhs[k];
}

} // namespace
} // namespace flash::mjwarp

extern "C" {
const char *flash_mjwarp_mass_solve_source_id() {
  return FLASH_MJW_MASS_SOLVE_SOURCE_ID;
}
std::size_t flash_mjwarp_mass_solve_arguments_bytes() {
  return sizeof(flash::mjwarp::MassSolveInputs);
}
int flash_mjwarp_mass_solve_arch() { return FLASH_MJW_MASS_SOLVE_ARCH; }

int flash_mjwarp_mass_solve_groups(const flash::mjwarp::MassSolveInputs *one,
                                   const flash::mjwarp::MassSolveInputs *six,
                                   const flash::mjwarp::MassSolveInputs *nine,
                                   cudaStream_t stream) {
  if (!one || !six || !nine || one->worlds < 1 || one->worlds != six->worlds ||
      one->worlds != nine->worlds || one->blocks < 0 || six->blocks < 0 ||
      nine->blocks < 0 || nine->blocks > 1 || one->blocks + six->blocks > 32)
    return int(cudaErrorInvalidValue);
  flash::mjwarp::world_solve<<<one->worlds, 64, 0, stream>>>(*one, *six, *nine);
  return int(cudaGetLastError());
}
int flash_mjwarp_mass_solve_launch(const flash::mjwarp::MassSolveInputs *input,
                                   int kind, int size, cudaStream_t stream) {
  using namespace flash::mjwarp;
  if (!input || input->worlds < 1 || input->blocks < 1 || input->threads != 64)
    return int(cudaErrorInvalidValue);
  const auto &a = *input;
  if (kind == 0 && size == 1)
    scalar_solve<1><<<(a.worlds * a.blocks + 63) / 64, 64, 0, stream>>>(a);
  else if (kind == 0 && size == 6)
    scalar_solve<6><<<(a.worlds * a.blocks + 63) / 64, 64, 0, stream>>>(a);
  else if (kind == 1 && size == 9)
    tile_solve9<<<dim3(a.worlds, a.blocks), 64, 0, stream>>>(a);
  else
    return int(cudaErrorInvalidValue);
  return int(cudaGetLastError());
}
}
