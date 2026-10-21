// Copyright 2025 The Newton Developers
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// Copyright 2026 Flash-LIBERO contributors
// SPDX-License-Identifier: Apache-2.0
// One readable world-local damped Euler step; public Newton qacc stays intact.
#include "../dynamics/mjwarp_mass_factor.cuh"
#include "fused_tail.cuh"
#include <cusolverdx.hpp>
#ifndef FLASH_FP32_EULER_STEP_SOURCE_ID
#error "Build through flash_fp32_fused_euler_step"
#endif
namespace flash::fp32 {
using flash::mjwarp::EulerInputs;
using flash::mjwarp::MassFactorInputs;
namespace dx = cusolverdx;
using Potrf9 =
    decltype(dx::Size<9>() + dx::Precision<float>() +
             dx::Type<dx::type::real>() + dx::Function<dx::function::potrf>() +
             dx::FillMode<dx::fill_mode::upper>() +
             dx::Arrangement<dx::row_major>() + dx::SM<800>() +
             dx::BlockDim<64>() + dx::Block());
using Potrs9 =
    decltype(dx::Size<9, 9, 1>() + dx::Precision<float>() +
             dx::Type<dx::type::real>() + dx::Function<dx::function::potrs>() +
             dx::FillMode<dx::fill_mode::upper>() +
             dx::Arrangement<dx::row_major, dx::col_major>() + dx::SM<800>() +
             dx::BlockDim<64>() + dx::Block());

using Potrf9World = Potrf9;
using Potrs9World = Potrs9;
__global__ void euler_step(MassFactorInputs one, MassFactorInputs six,
                           MassFactorInputs nine, TailInputs inputs,
                           const float *original) {

  const int w = blockIdx.x, tid = threadIdx.x;
  const EulerInputs e = inputs.integration;
  __shared__ __align__(16) float upper[81], rhs[9];
  __shared__ int info;
  const int first9 = nine.blocks ? nine.block_start[0] : -1;
  {
    // Each DOF owns its current derivative, full CSR row and modified diagonal.
    // The one producer barrier is also the input boundary for every factor
    // block.
    for (int dof = tid; dof < e.nv; dof += 64) {
      const int t = w * e.nv + dof;
      const float *polynomial =
          e.dampingpoly + 2 * ((w % e.dampingpoly_rows) * e.nv + dof);
      const float speed = fabsf(e.qvel[t]);
      float damping = e.damping[(w % e.damping_rows) * e.nv + dof];
      damping += 2.f * polynomial[0] * speed;
      damping += 3.f * polynomial[1] * speed * speed;
      e.derivative[t] = damping;
      const int first = w * e.matrix_width + e.rowadr[dof],
                count = e.rownnz[dof];
      for (int k = 0; k < count - 1; ++k)
        e.matrix[first + k] = original[first + k];
      e.matrix[first + count - 1] = original[first + count - 1] +
                                    e.timestep[w % e.timestep_rows] * damping;
    }
    __syncthreads(); // all current rows are ready for scalar/SDK consumers
  }
#pragma unroll
  for (int group = 0; group < 2; ++group) {
    const MassFactorInputs a = group ? six : one;
    const int size = group ? 6 : 1;
    for (int block = tid; block < a.blocks; block += 64) {
      const int start = a.block_start[block];
      int adr = a.factor_address[start], base = a.matrix_row[start];
      const float *matrix = a.matrix + w * a.matrix_width + base;
      const float *y = a.rhs + w * a.rhs_width + start;
      float *x = a.solution + w * a.solution_width + start;
      if (adr == -2) {
        for (int i = 0; i < size; ++i) {
          float inverse = 1.0f / matrix[i];
          a.diagonal_inverse[w * a.nv + start + i] = inverse;

          x[i] = inverse * y[i];
        }
        continue;
      }
      float *upper = a.factor + w * a.factor_width + adr;
      for (int i = 0; i < size; ++i) {
        float value = matrix[i * (i + 1) / 2 + i];
        float rhs_value = y[i];
        for (int k = 0; k < i; ++k) {
          float u = upper[k * size + i];
          value -= u * u;

          rhs_value -= u * x[k];
        }
        float diagonal = sqrtf(value);
        upper[i * size + i] = diagonal;
        float inverse = 1.0f / diagonal;

        x[i] = rhs_value * inverse;
        for (int j = i + 1; j < size; ++j) {
          float entry = matrix[j * (j + 1) / 2 + i];
          for (int k = 0; k < i; ++k)
            entry -= upper[k * size + i] * upper[k * size + j];
          upper[i * size + j] = entry * inverse;
        }
      }
      {
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
  if (nine.blocks) {
    const MassFactorInputs a = nine;
    int start = a.block_start[0],
        adr = w * a.factor_width + a.factor_address[start];
    for (int k = tid; k < 81; k += 64) {
      int index = a.gather[0 * 81 + k];
      // Warp tile_load_indexed predicates either out-of-range side to zero.
      upper[k] = index >= 0 && index < a.matrix_width
                     ? a.matrix[w * a.matrix_width + index]
                     : 0.0f;
    }

    for (int k = tid; k < 9; k += 64)
      rhs[k] = a.rhs[w * a.rhs_width + start + k];
    if (tid == 0)
      info = 0;
    __syncthreads();
    Potrf9World().execute(upper, 9, &info);
    __syncthreads();
    // POTRS consumes the selected upper triangle through a const pointer.
    // Defer lower clearing/publication until it has finished, keeping the
    // whole factor immutable during the SDK read and removing the old clear
    // -> solve barrier. No consumer observes these private factors mid-kernel.
    Potrs9World().execute(upper, 9, rhs, 9);
    __syncthreads();
    for (int k = tid; k < 81; k += 64) {
      if (k / 9 > k % 9)
        upper[k] = 0.0f;
      a.factor[adr + k] = upper[k];
    }
    for (int k = tid; k < 9; k += 64)
      a.solution[w * a.solution_width + start + k] = rhs[k];
  }
  // A cooperative solve's final barrier already orders scalar block stores.
  // For scalar-only layouts, this is their required completion boundary.
  if (!nine.blocks)
    __syncthreads();

  {

    constexpr int Threads = 64;
    const auto &a = inputs.integration;
    const int world = w, lane = tid;
    const float dt = a.timestep[world % a.timestep_rows];
    float *velocity = a.qvel + world * a.nv;
    for (int dof = lane; dof < a.nv; dof += Threads) {
      // The cooperative9 result is already complete in shared memory. Scalar
      // blocks were published before that barrier; no extra boundary is needed.
      const float acceleration =
          nine.blocks && dof >= first9 && dof < first9 + 9
              ? rhs[dof - first9]
              : a.acceleration[world * a.nv + dof];
      velocity[dof] = velocity[dof] + acceleration * dt;
    }
    __syncthreads();

    for (int joint = lane; joint < a.joints; joint += Threads) {
      const int kind = a.joint_type[joint];
      float *q = a.qpos + world * a.nq + a.qposadr[joint];
      const float *v = velocity + a.dofadr[joint];
      if (kind == 0) {
        q[0] += dt * v[0];
        q[1] += dt * v[1];
        q[2] += dt * v[2];
        q += 3;
        v += 3;
      }
      if (kind < 2) {
        const float speed = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        float x = 0.f, y = 0.f, z = 0.f;
        if (speed > 0.f) {
          x = v[0] / speed;
          y = v[1] / speed;
          z = v[2] / speed;
        }
        const float angle = dt * speed;
        const float s = sinf(angle * .5f), c = cosf(angle * .5f);
        const float bw = c, bx = x * s, by = y * s, bz = z * s;
        float qw = q[0], qx = q[1], qy = q[2], qz = q[3];
        float length = sqrtf(qw * qw + qx * qx + qy * qy + qz * qz);
        if (length > 0.f) {
          const float r = 1.f / length;
          qw *= r;
          qx *= r;
          qy *= r;
          qz *= r;
        } else {
          qw = qx = qy = 0.f;
          qz = 1.f;
        } // pinned degenerate quaternion rule
        const float rw = qw * bw - qx * bx - qy * by - qz * bz;
        const float rx = qw * bx + qx * bw + qy * bz - qz * by;
        const float ry = qw * by - qx * bz + qy * bw + qz * bx;
        const float rz = qw * bz + qx * by - qy * bx + qz * bw;
        length = sqrtf(rw * rw + rx * rx + ry * ry + rz * rz);
        if (length > 0.f) {
          const float r = 1.f / length;
          q[0] = rw * r;
          q[1] = rx * r;
          q[2] = ry * r;
          q[3] = rz * r;
        } else {
          q[0] = q[1] = q[2] = 0.f;
          q[3] = 1.f;
        }
      } else
        q[0] += dt * v[0];
    }

    {
      // Warmstart retains NEWTON qacc, not the damping-corrected acceleration
      // used for integration. Each lane owns disjoint DOF stores.
      if (lane >= 32)
        for (int dof = lane - 32; dof < a.nv; dof += 32)
          inputs.warmstart[world * a.nv + dof] =
              inputs.solved_acceleration[world * a.nv + dof];
      // The second warp can publish while the first integrates joint poses.
      if (lane == 32) {
        const auto &p = inputs.publication;
        const int count = p.count[world];
        int bits = p.overflow[world];
        p.time[world] += p.timestep[world % p.timestep_rows];
        if (count > p.rows) {
          bits |= 1;
        } else if (count > 0 && p.sparse) {
          const int last = world * p.rows + count - 1;
          if (p.rowadr[last] + p.rownnz[last] > p.nonzeros) {
            bits |= 2;
          }
        }
        // These pooled counters are immutable throughout this world-owned tail.
        if (p.candidate_count[0] > p.contacts) {
          bits |= 4;
        }
        if (p.contact_count[0] > p.contacts) {
          bits |= 8;
        }
        p.overflow[world] = bits;
      }
    }
  }
}
} // namespace flash::fp32

extern "C" {
const char *flash_fp32_euler_step_source_id() {
  return FLASH_FP32_EULER_STEP_SOURCE_ID;
}
int flash_fp32_euler_step_arch() { return FLASH_CUDA_ARCH; }
std::size_t flash_fp32_euler_step_mass_bytes() {
  return sizeof(flash::mjwarp::MassFactorInputs);
}
std::size_t flash_fp32_euler_step_tail_bytes() {
  return sizeof(flash::fp32::TailInputs);
}
int flash_fp32_euler_step_launch(const flash::mjwarp::MassFactorInputs *one,
                                 const flash::mjwarp::MassFactorInputs *six,
                                 const flash::mjwarp::MassFactorInputs *nine,
                                 const flash::fp32::TailInputs *inputs,
                                 const float *original, cudaStream_t stream) {
  if (!one || !six || !nine || !inputs || !original)
    return int(cudaErrorInvalidValue);
  const auto &e = inputs->integration;
  if (e.worlds < 1 || e.nv < 1 || e.joints < 1 ||
      inputs->publication.worlds != e.worlds || inputs->publication.warnings ||
      one->worlds != e.worlds || six->worlds != e.worlds ||
      nine->worlds != e.worlds || one->nv != e.nv || six->nv != e.nv ||
      nine->nv != e.nv || one->matrix != e.matrix || six->matrix != e.matrix ||
      nine->matrix != e.matrix || one->solution != e.acceleration ||
      six->solution != e.acceleration || nine->solution != e.acceleration ||
      one->solution_width != e.nv || one->matrix_width != e.matrix_width ||
      one->blocks < 0 || six->blocks < 0 || nine->blocks < 0 ||
      nine->blocks > 1 || one->blocks + six->blocks > 32)
    return int(cudaErrorInvalidValue);
  flash::fp32::euler_step<<<e.worlds, 64, 0, stream>>>(*one, *six, *nine,
                                                       *inputs, original);
  return int(cudaGetLastError());
}
}
