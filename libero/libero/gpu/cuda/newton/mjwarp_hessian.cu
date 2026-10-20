// Copyright 2025 The Newton Developers
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Handwritten CUDA adaptation of mujoco_warp/_src/solver.py, commit
// c384ac51df67eb92ab7657cc472ff7494dc036bf:
//   _update_gradient_JTDAJ_dense_tiled, _padding_h,
//   _elliptic_hessian_entry_from_projections, _update_gradient_JTCJ_dense.
// Register sub-tile selection follows Warp 1.18.0 native/tile_matmul.h,
// partitioned_gemm::scalar_matmul (Apache-2.0, NVIDIA copyright above).
// Changes: plain device pointers, explicit shared storage and CUDA threads,
// persistent CSR-address lookup, and padding folded into the first kernel.
// No Warp headers, generated CUDA, MathDx link, factorization or Newton loop.
#include "mjwarp_hessian.cuh"
#include <cmath>

#ifndef FLASH_MJW_HESSIAN_SOURCE_ID
#error "Build this translation unit through its maintained CMake target"
#endif

namespace flash::mjwarp {
namespace {
constexpr int kThreads = 128;
constexpr int kRowsPerTile = 16;
constexpr int kQuadratic = 1;
constexpr int kCone = 4;
constexpr float kMinValue = 1.e-15f;  // pinned MuJoCo mjMINVAL

// One cooperative CTA per world, matching the baseline's world ownership.
// As in Warp's partitioned_gemm::scalar_matmul, each thread owns a register
// submatrix. J and D*J are prepared once per row tile, then reused by all H
// entries. The largest specialization uses 8 KiB, independent of world/rows.
template <int Pad>
__global__ void mass_quadratic(int nv, int rows, int mass_entries,
                               const int* __restrict__ address,
                               const int* __restrict__ nefc,
                               const float* __restrict__ mass,
                               const float* __restrict__ jacobian,
                               const float* __restrict__ diagonal,
                               const int* __restrict__ state,
                               const bool* __restrict__ done,
                               float* __restrict__ hessian) {
  const int world = blockIdx.x;
  if (done[world]) return;  // block-uniform; also protects publication/holds
  const int tid = threadIdx.x;
  // Same sub-tile choice as the pinned Warp scalar GEMM (128-thread block).
  constexpr int kMinimumTiles = (3 * kThreads + 3) / 4;
  constexpr int kRows = ((Pad + 7) / 8) * ((Pad + 3) / 4) >= kMinimumTiles ? 8
      : ((Pad + 3) / 4) * ((Pad + 3) / 4) >= kMinimumTiles ? 4
      : ((Pad + 3) / 4) * ((Pad + 1) / 2) >= kMinimumTiles ? 4
      : ((Pad + 1) / 2) * ((Pad + 1) / 2) >= kThreads ? 2 : 1;
  constexpr int kColumns = kRows == 8 ? 4
      : kRows == 4 && ((Pad + 3) / 4) * ((Pad + 3) / 4) >= kMinimumTiles ? 4
      : kRows == 4 ? 2 : kRows;
  constexpr int kTileColumns = (Pad + kColumns - 1) / kColumns;
  constexpr int kTiles = ((Pad + kRows - 1) / kRows) * kTileColumns;
  constexpr int kOwnedTiles = (kTiles + kThreads - 1) / kThreads;
  float value[kOwnedTiles][kRows][kColumns];
  __shared__ float j[kRowsPerTile][Pad];
  __shared__ float weighted_j[kRowsPerTile][Pad];
  const std::size_t row_base = std::size_t(world) * rows;
  const std::size_t h_base = std::size_t(world) * Pad * Pad;

#pragma unroll
  for (int slot = 0; slot < kOwnedTiles; ++slot) {
    const int tile = tid + slot * kThreads;
    const int first_row = tile / kTileColumns * kRows;
    const int first_column = tile % kTileColumns * kColumns;
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
#pragma unroll
      for (int c = 0; c < kColumns; ++c) {
        const int i = first_row + r, column = first_column + c;
        const int entry = i < Pad && column < Pad ? address[i * Pad + column] : -1;
        value[slot][r][c] = entry >= 0 ? mass[std::size_t(world) * mass_entries + entry] : 0.f;
      }
    }
  }

  for (int start = 0; start < nefc[world]; start += kRowsPerTile) {
    for (int element = tid; element < kRowsPerTile * Pad; element += kThreads) {
      const int row = start + element / Pad;
      const float x = row < nefc[world]
          ? jacobian[(row_base + row) * Pad + element % Pad] : 0.f;
      const float weight = row < nefc[world] && state[row_base + row] == kQuadratic
          ? diagonal[row_base + row] : 0.f;
      j[element / Pad][element % Pad] = x;
      weighted_j[element / Pad][element % Pad] = x * weight;
    }
    __syncthreads();

#pragma unroll
    for (int slot = 0; slot < kOwnedTiles; ++slot) {
      const int tile_id = tid + slot * kThreads;
      if (tile_id < kTiles) {
        const int first_row = tile_id / kTileColumns * kRows;
        const int first_column = tile_id % kTileColumns * kColumns;
        float product[kRows][kColumns] = {};
#pragma unroll
        for (int k = 0; k < kRowsPerTile; ++k) {
          float left[kRows], right[kColumns];
#pragma unroll
          for (int r = 0; r < kRows; ++r)
            left[r] = first_row + r < Pad ? weighted_j[k][first_row + r] : 0.f;
#pragma unroll
          for (int c = 0; c < kColumns; ++c)
            right[c] = first_column + c < Pad ? j[k][first_column + c] : 0.f;
#pragma unroll
          for (int r = 0; r < kRows; ++r) {
#pragma unroll
            for (int c = 0; c < kColumns; ++c)
              product[r][c] = fmaf(left[r], right[c], product[r][c]);
          }
        }
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
#pragma unroll
          for (int c = 0; c < kColumns; ++c)
            value[slot][r][c] += product[r][c];
        }
      }
    }
    __syncthreads();  // shared J/D lifetime ends before the next tile load
  }

#pragma unroll
  for (int slot = 0; slot < kOwnedTiles; ++slot) {
    const int tile = tid + slot * kThreads;
    const int first_row = tile / kTileColumns * kRows;
    const int first_column = tile % kTileColumns * kColumns;
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
#pragma unroll
      for (int c = 0; c < kColumns; ++c) {
        const int i = first_row + r, column = first_column + c;
        if (i < Pad && column < Pad) {
          const bool padding_diagonal = nv > 32 && i >= nv && i == column;
          hessian[h_base + i * Pad + column] = padding_diagonal ? 1.f : value[slot][r][c];
        }
      }
    }
  }
}

// The baseline limits launch groups using the SM count, then loops over pooled
// contacts. Keep the same mapping and atomic accumulation in this first port.
__global__ void cone_curvature(int nv, int pad, int rows, int capacity,
                              int impratio_count, int groups, int contacts_per_group,
                              const float* __restrict__ impratio,
                              const int* __restrict__ triangle_row,
                              const int* __restrict__ triangle_column,
                              const int* __restrict__ contact_count,
                              const int* __restrict__ contact_world,
                              const int* __restrict__ dimension,
                              const float* __restrict__ friction,
                              const float* __restrict__ distance,
                              const float* __restrict__ margin,
                              const int* __restrict__ address,
                              const float* __restrict__ jacobian,
                              const float* __restrict__ diagonal,
                              const int* __restrict__ state,
                              const float* __restrict__ jaref,
                              const bool* __restrict__ done,
                              float* __restrict__ hessian) {
  const int triangle = nv * (nv + 1) / 2;
  const std::size_t thread = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (thread >= std::size_t(groups) * triangle) return;
  const int group = thread / triangle, element = thread % triangle;
  const int dof1 = triangle_row[element], dof2 = triangle_column[element];
  const int count = min(contact_count[0], capacity);

  for (int batch = 0; batch < contacts_per_group; ++batch) {
    const int contact = group + batch * groups;
    if (contact >= count) return;
    const int world = contact_world[contact], dim = dimension[contact];
    if (done[world] || dim == 1 || distance[contact] - margin[contact] >= 0.f) continue;
    const int head = address[6 * contact];
    const std::size_t base = std::size_t(world) * rows;
    if (head < 0 || state[base + head] != kCone) continue;

    const float* fri = friction + 5 * contact;
    const float mu = fri[0] * impratio[world % impratio_count];
    const float mu2 = mu * mu;
    const float denominator = mu2 * (1.f + mu2);
    const float dm = diagonal[base + head] / (denominator != 0.f ? denominator : kMinValue);
    if (dm == 0.f) continue;
    const float n = jaref[base + head] * mu;
    const float z01 = mu * jacobian[(base + head) * pad + dof1];
    const float z02 = mu * jacobian[(base + head) * pad + dof2];
    float tt = 0.f, projection1 = 0.f, projection2 = 0.f, tangent_dot = 0.f;
    for (int d = 1; d < dim; ++d) {
      const int row = address[6 * contact + d];
      if (row >= 0) {
        const float scale = fri[d - 1];
        const float u = jaref[base + row] * scale;
        const float z1 = scale * jacobian[(base + row) * pad + dof1];
        const float z2 = scale * jacobian[(base + row) * pad + dof2];
        tt += u * u;
        projection1 += u * z1;
        projection2 += u * z2;
        tangent_dot += z1 * z2;
      }
    }
    const float t = fmaxf(sqrtf(tt), kMinValue);
    const float inverse_t = 1.f / t;
    const float mu_n_over_t = mu * n * inverse_t;
    const float p1 = projection1 * inverse_t, p2 = projection2 * inverse_t;
    const float curvature = dm * (z01 * z02 - mu * (z01 * p2 + z02 * p1)
        + mu_n_over_t * p1 * p2 + (mu2 - mu_n_over_t) * tangent_dot);
    atomicAdd(hessian + (std::size_t(world) * pad + dof1) * pad + dof2, curvature);
  }
}

template <int Pad>
void launch_quadratic(const HessianInputs& a, cudaStream_t stream) {
  mass_quadratic<Pad><<<a.worlds, kThreads, 0, stream>>>(a.nv, a.rows,
      a.mass_entries, a.mass_upper_address, a.nefc, a.mass, a.jacobian,
      a.diagonal, a.state, a.done, a.hessian);
}
}  // namespace
}  // namespace flash::mjwarp

extern "C" std::size_t flash_mjwarp_hessian_arguments_bytes() {
  return sizeof(flash::mjwarp::HessianInputs);
}
extern "C" const char* flash_mjwarp_hessian_source_id() { return FLASH_MJW_HESSIAN_SOURCE_ID; }
extern "C" int flash_mjwarp_hessian_arch() { return FLASH_MJW_HESSIAN_ARCH; }

extern "C" int flash_mjwarp_hessian_launch(const flash::mjwarp::HessianInputs* inputs,
                                           int stage, cudaStream_t stream) {
  using namespace flash::mjwarp;
  if (!inputs || stage < 0 || stage > 2) return cudaErrorInvalidValue;
  const auto& a = *inputs;
  if (a.worlds <= 0 || a.rows <= 0 || a.nv <= 0 || a.nv > a.padded_nv ||
      a.mass_entries <= 0 || a.contact_capacity <= 0 ||
      (a.impratio_count != 1 && a.impratio_count != a.worlds) ||
      a.cone_launch_groups <= 0 || a.cone_contacts_per_group <= 0)
    return cudaErrorInvalidValue;
  if (stage != 2) {
    switch (a.padded_nv) {
      case 24: launch_quadratic<24>(a, stream); break;
      case 28: launch_quadratic<28>(a, stream); break;
      case 32: launch_quadratic<32>(a, stream); break;
      case 48: launch_quadratic<48>(a, stream); break;
      case 64: launch_quadratic<64>(a, stream); break;
      default: return cudaErrorInvalidValue;
    }
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) return error;
  }
  if (stage != 1) {
    const std::size_t threads = std::size_t(a.cone_launch_groups) * a.nv * (a.nv + 1) / 2;
    cone_curvature<<<(threads + 255) / 256, 256, 0, stream>>>(a.nv, a.padded_nv,
        a.rows, a.contact_capacity, a.impratio_count, a.cone_launch_groups,
        a.cone_contacts_per_group, a.impratio_invsqrt, a.triangle_row,
        a.triangle_column, a.contact_count, a.contact_world, a.contact_dimension,
        a.contact_friction, a.contact_distance, a.contact_margin, a.contact_address,
        a.jacobian, a.diagonal, a.state, a.jaref, a.done, a.hessian);
    return cudaGetLastError();
  }
  return cudaSuccess;
}
