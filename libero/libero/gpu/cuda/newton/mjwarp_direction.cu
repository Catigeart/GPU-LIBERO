// Copyright 2025 The Newton Developers
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0 Handwritten CUDA
// adaptation of pinned MJWarp c384ac51df67eb92ab7657cc472ff7494dc036bf:
// solver.py::_update_gradient_zero_grad_dot/_update_gradient_grad,
// _update_gradient_cholesky/_update_gradient_cholesky_blocked, and
// block_cholesky.py::_create_blocked_cholesky_augmented_factorize_solve_func.
// Uses the same cuSolverDx POTRF/POTRS/TRSM operations as Warp's MathDx path.
// Register subtiles follow Warp 1.18 native/tile_matmul.h::scalar_matmul.
// No Warp headers, generated sources, hidden solver calls or scalar fallback.
#include "mjwarp_direction.cuh"
#include "mjwarp_solver_state.cuh"
#include <cusolverdx.hpp>

#ifndef FLASH_MJW_DIRECTION_SOURCE_ID
#error "Build through flash_mjwarp_direction_fp32"
#endif

namespace flash::mjwarp {
namespace {
namespace dx = cusolverdx;

__global__ void gradient_reset(int worlds, const bool *done, float *grad_dot,
                               float *decrement, float *scale,
                               bool *unchanged) {
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= worlds)
    return;
  unchanged[w] = false; // upstream also updates this flag for done worlds
  if (done[w])
    return;
  grad_dot[w] = decrement[w] = 0.f;
  scale[w] = 1.f;
}

__global__ void gradient_values(DirectionInputs a) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.worlds * a.nv)
    return;
  int w = t / a.nv, i = t % a.nv;
  if (a.done[w])
    return;
  float g = a.ma[w * a.ma_stride + i] - a.smooth[w * a.smooth_stride + i] -
            a.force[w * a.force_stride + i];
  a.grad[w * a.pad + i] = g;
  atomicAdd(a.grad_dot + w, g * g); // same DOF-level accumulation as upstream
}

template <int N, int Threads>
using Potrf =
    decltype(dx::Size<N>() + dx::Precision<float>() +
             dx::Type<dx::type::real>() + dx::Function<dx::function::potrf>() +
             dx::FillMode<dx::fill_mode::upper>() +
             dx::Arrangement<dx::row_major>() + dx::SM<800>() +
             dx::BlockDim<Threads>() + dx::Block());

template <int N>
using Potrs =
    decltype(dx::Size<N, N, 1>() + dx::Precision<float>() +
             dx::Type<dx::type::real>() + dx::Function<dx::function::potrs>() +
             dx::FillMode<dx::fill_mode::upper>() +
             dx::Arrangement<dx::row_major, dx::col_major>() + dx::SM<800>() +
             dx::BlockDim<64>() + dx::Block());

template <int Columns, bool Transposed>
using Trsm =
    decltype(dx::Size<16, Columns>() + dx::Precision<float>() +
                         dx::Type<dx::type::real>() +
                         dx::Function<dx::function::trsm>() +
                         dx::Side<dx::side::left>() +
                         dx::Diag<dx::diag::non_unit>() + dx::FillMode <
                     Transposed
                 ? dx::fill_mode::lower
             : dx::fill_mode::upper > () + dx::Arrangement < Transposed
                 ? dx::col_major
                 : dx::row_major,
             dx::row_major >
                 () + dx::SM<800>() + dx::BlockDim<32>() + dx::Block());

template <int N, int Threads>
__device__ void factor_upper(float *tile, int *info) {
  if (threadIdx.x == 0)
    *info = 0;
  __syncthreads();
  Potrf<N, Threads>().execute(tile, N, info);
  __syncthreads();
  for (int e = threadIdx.x; e < N * N; e += Threads)
    if (e / N > e % N)
      tile[e] = 0.f;
  __syncthreads();
}

template <int Threads>
__device__ void publish(DirectionInputs a, const float *solution) {
  int w = blockIdx.x;
  float square = 0.f, decrement = 0.f;
  for (int i = threadIdx.x; i < a.nv; i += Threads) {
    float x = solution[i];
    a.search[w * a.nv + i] = -x;
    square += x * x;
    decrement += a.grad[w * a.pad + i] * x;
  }
  // Warp tiled reduction: each lane first sums its strided elements.
  for (int offset = 16; offset; offset /= 2) {
    square += __shfl_down_sync(0xffffffff, square, offset);
    decrement += __shfl_down_sync(0xffffffff, decrement, offset);
  }
  __shared__ float partial[4];
  if ((threadIdx.x & 31) == 0) {
    partial[threadIdx.x / 32 * 2] = square;
    partial[threadIdx.x / 32 * 2 + 1] = decrement;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    a.search_dot[w] = partial[0] + (Threads == 64 ? partial[2] : 0.f);
    a.decrement[w] = partial[1] + (Threads == 64 ? partial[3] : 0.f);
  }
}

template <int N, bool Stopping = false>
__global__ void direction_small(DirectionInputs a, StopInputs stop) {
  int w = blockIdx.x, t = threadIdx.x;
  if (a.done[w])
    return;
  __shared__ float matrix[N * N], rhs[N];
  __shared__ int info;
  for (int e = t; e < N * N; e += 64)
    matrix[e] = a.hessian[(std::size_t(w) * a.pad + e / N) * a.pad + e % N];
  for (int i = t; i < N; i += 64)
    rhs[i] = a.grad[w * a.pad + i];
  __syncthreads();
  factor_upper<N, 64>(matrix, &info);
  Potrs<N>().execute(matrix, N, rhs, N);
  __syncthreads();
  publish<64>(a, rhs);
  // The publication owner has just produced decrement. Its stopping decision
  // needs no new block barrier; all other predicates were ready on entry.
  if constexpr (Stopping)
    if (t == 0) {
      int iterations = ++stop.niter[w];
      float tolerance = stop.tolerance[w % stop.tolerance_count];
      float scale = stop.inertia[w % stop.inertia_count] * float(stop.nv);
      float improvement = stop.improvement[w] / scale;
      float gradient = sqrtf(stop.grad_dot[w]) / scale;
      float model_improvement = (0.5f * stop.decrement[w]) / scale;
      bool done = stop.alpha[w] == 0.f ||
                  (improvement > 0.f && improvement < tolerance) ||
                  gradient < tolerance || model_improvement < tolerance;
      if (done || iterations == stop.iterations) {
        if (!done && iterations == stop.iterations) {
          constexpr int IterationsOverflow = 1 << 9;
          stop.overflow[w] |= IterationsOverflow;
        }
        stop.done[w] = true;
        atomicAdd(stop.nsolving, -1);
      }
    }
}

template <int Pad>
__device__ void load_block(const float *source, int row, int column,
                           float *tile) {
  // Pinned aligned tile_load uses 16-byte transfers for these dense blocks.
#pragma unroll
  for (int q = threadIdx.x; q < 64; q += 32)
    reinterpret_cast<float4 *>(tile)[q] = *reinterpret_cast<const float4 *>(
        source + (row + q / 4) * Pad + column + (q % 4) * 4);
  __syncthreads();
}

template <int Pad>
__device__ void store_block(float *target, int row, int column,
                            const float *tile) {
#pragma unroll
  for (int q = threadIdx.x; q < 64; q += 32)
    *reinterpret_cast<float4 *>(target + (row + q / 4) * Pad + column +
                                (q % 4) * 4) =
        reinterpret_cast<const float4 *>(tile)[q];
  __syncthreads();
}

// Same block-level Schur update as tile_matmul(transpose(left), right,
// tile,-1).
__device__ void subtract_gram(float *tile, const float *left,
                              const float *right) {
  // Warp chooses a 4x2 register subtile for M=N=16 and a 32-thread block.
  const int row = (threadIdx.x / 8) * 4, column = (threadIdx.x % 8) * 2;
  float product[4][2] = {};
#pragma unroll
  for (int k = 0; k < 16; ++k) {
    float l[4], r[2];
#pragma unroll
    for (int i = 0; i < 4; ++i)
      l[i] = left[k * 16 + row + i];
#pragma unroll
    for (int j = 0; j < 2; ++j)
      r[j] = right[k * 16 + column + j];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j)
        product[i][j] = fmaf(l[i], r[j], product[i][j]);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j)
      tile[(row + i) * 16 + column + j] -= product[i][j];
  __syncthreads();
}

template <int Pad, bool Stopping = false>
__global__ void direction_blocked(DirectionInputs a, StopInputs stop) {
  int w = blockIdx.x, t = threadIdx.x;
  if (a.done[w])
    return;
  const float *h = a.hessian + std::size_t(w) * Pad * Pad;
  float *u = a.factor + std::size_t(w) * Pad * Pad;
  __shared__ __align__(16) float diagonal[256], panel[256], left[256],
      right[256], rhs[Pad];
  __shared__ int info;
  for (int i = t; i < Pad; i += 32)
    rhs[i] = 0.f;
  __syncthreads();

  // The final unused padding column carries b. A 1e30 diagonal suppresses
  // the augmented rank-one correction, exactly as in pinned block_cholesky.py.
  // Catalogue NV is strictly less than Pad, so no physical DOF is overwritten.
  for (int k = 0; k < Pad; k += 16) {
    load_block<Pad>(h, k, k, diagonal);
    if (k + 16 == Pad) {
      for (int i = t; i < 15; i += 32)
        diagonal[i * 16 + 15] = a.grad[w * Pad + k + i];
      if (t == 0)
        diagonal[255] = 1.e30f;
      __syncthreads();
    }
    for (int j = 0; j < k; j += 16) {
      load_block<Pad>(u, j, k, left);
      subtract_gram(diagonal, left, left);
    }
    factor_upper<16, 32>(diagonal, &info);
    if (k + 16 == Pad)
      for (int i = t; i < 15; i += 32)
        rhs[k + i] = diagonal[i * 16 + 15];
    store_block<Pad>(u, k, k, diagonal);

    for (int i = k + 16; i < Pad; i += 16) {
      load_block<Pad>(h, k, i, panel);
      if (i + 16 == Pad) {
        for (int r = t; r < 16; r += 32)
          panel[r * 16 + 15] = a.grad[w * Pad + k + r];
        __syncthreads();
      }
      for (int j = 0; j < k; j += 16) {
        load_block<Pad>(u, j, k, left);
        load_block<Pad>(u, j, i, right);
        subtract_gram(panel, left, right);
      }
      Trsm<16, true>().execute(diagonal, 16, panel, 16);
      __syncthreads();
      if (i + 16 == Pad)
        for (int r = t; r < 16; r += 32)
          rhs[k + r] = panel[r * 16 + 15];
      store_block<Pad>(u, k, i, panel);
    }
  }

  for (int i = Pad - 16; i >= 0; i -= 16) {
    for (int j = i + 16; j < Pad; j += 16) {
      load_block<Pad>(u, i, j, left);
      if (t < 16) {
        float product = 0.f;
#pragma unroll
        for (int k = 0; k < 16; ++k)
          product = fmaf(left[t * 16 + k], rhs[j + k], product);
        rhs[i + t] -= product;
      }
      __syncthreads();
    }
    load_block<Pad>(u, i, i, diagonal);
    Trsm<1, false>().execute(diagonal, 16, rhs + i, 1);
    __syncthreads();
  }
  publish<32>(a, rhs);
  // The publication owner has just produced decrement. Its stopping decision
  // needs no new block barrier; all other predicates were ready on entry.
  if constexpr (Stopping)
    if (t == 0) {
      int iterations = ++stop.niter[w];
      float tolerance = stop.tolerance[w % stop.tolerance_count];
      float scale = stop.inertia[w % stop.inertia_count] * float(stop.nv);
      float improvement = stop.improvement[w] / scale;
      float gradient = sqrtf(stop.grad_dot[w]) / scale;
      float model_improvement = (0.5f * stop.decrement[w]) / scale;
      bool done = stop.alpha[w] == 0.f ||
                  (improvement > 0.f && improvement < tolerance) ||
                  gradient < tolerance || model_improvement < tolerance;
      if (done || iterations == stop.iterations) {
        if (!done && iterations == stop.iterations) {
          constexpr int IterationsOverflow = 1 << 9;
          stop.overflow[w] |= IterationsOverflow;
        }
        stop.done[w] = true;
        atomicAdd(stop.nsolving, -1);
      }
    }
}
} // namespace
} // namespace flash::mjwarp

extern "C" const char *flash_mjwarp_direction_source_id() {
  return FLASH_MJW_DIRECTION_SOURCE_ID;
}
extern "C" std::size_t flash_mjwarp_direction_arguments_bytes() {
  return sizeof(flash::mjwarp::DirectionInputs);
}
extern "C" int flash_mjwarp_direction_arch() { return 80; }
extern "C" int
flash_mjwarp_direction_launch(const flash::mjwarp::DirectionInputs *input,
                              int stage, cudaStream_t stream) {
  using namespace flash::mjwarp;
  if (!input || stage < 0 || stage > 2)
    return int(cudaErrorInvalidValue);
  const auto &a = *input;
  if (a.worlds < 1)
    return int(cudaErrorInvalidValue);
  if (stage != 2) {
    gradient_reset<<<(a.worlds + 255) / 256, 256, 0, stream>>>(
        a.worlds, a.done, a.grad_dot, a.decrement, a.grad_scale,
        a.search_unchanged);
    gradient_values<<<(a.worlds * a.nv + 255) / 256, 256, 0, stream>>>(a);
  }
  if (stage != 1) {
    switch (a.nv) {
#define SMALL(N)                                                               \
  case N:                                                                      \
    direction_small<N><<<a.worlds, 64, 0, stream>>>(a, {});                    \
    break
      SMALL(21);
      SMALL(22);
      SMALL(24);
      SMALL(27);
      SMALL(30);
#undef SMALL
    case 33:
    case 36:
    case 37:
    case 39:
    case 43:
    case 45:
      direction_blocked<48><<<a.worlds, 32, 0, stream>>>(a, {});
      break;
    case 51:
    case 57:
      direction_blocked<64><<<a.worlds, 32, 0, stream>>>(a, {});
      break;
    default:
      return int(cudaErrorInvalidValue);
    }
  }
  return int(cudaGetLastError());
}

extern "C" std::size_t flash_mjwarp_direction_stop_bytes() {
  return sizeof(flash::mjwarp::StopInputs);
}
extern "C" int
flash_mjwarp_direction_stopping(const flash::mjwarp::DirectionInputs *input,
                                const flash::mjwarp::StopInputs *stop,
                                cudaStream_t stream) {
  using namespace flash::mjwarp;
  if (!input || !stop || input->worlds < 1 || input->worlds != stop->worlds ||
      input->nv != stop->nv || input->done != stop->done ||
      input->grad_dot != stop->grad_dot ||
      input->decrement != stop->decrement || !stop->niter || !stop->nsolving ||
      stop->tolerance_count < 1 || stop->inertia_count < 1 ||
      stop->iterations < 1 || (stop->warn_overflow & (1 << 9)))
    return int(cudaErrorInvalidValue);
  const auto &a = *input;
  switch (a.nv) {
#define SMALL(N)                                                               \
  case N:                                                                      \
    direction_small<N, true><<<a.worlds, 64, 0, stream>>>(a, *stop);           \
    break
    SMALL(21);
    SMALL(22);
    SMALL(24);
    SMALL(27);
    SMALL(30);
#undef SMALL
  case 33:
  case 36:
  case 37:
  case 39:
  case 43:
  case 45:
    direction_blocked<48, true><<<a.worlds, 32, 0, stream>>>(a, *stop);
    break;
  case 51:
  case 57:
    direction_blocked<64, true><<<a.worlds, 32, 0, stream>>>(a, *stop);
    break;
  default:
    return int(cudaErrorInvalidValue);
  }
  return int(cudaGetLastError());
}
