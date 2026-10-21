// Copyright 2025 The Newton Developers
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// Copyright 2026 Flash-LIBERO contributors
// SPDX-License-Identifier: Apache-2.0
// World-local actuation -> smooth RHS -> current mass-factor backsubstitution.
#include "../dynamics/mjwarp_actuation.cuh"
#include "../dynamics/mjwarp_mass_solve.cuh"
#include "../dynamics/mjwarp_smooth_force.cuh"
#include <cusolverdx.hpp>
#ifndef FLASH_FP32_ACTUATION_SOLVE_SOURCE_ID
#error "Build through flash_fp32_fused_actuation_solve"
#endif
namespace flash::fp32 {
using flash::mjwarp::ActuationInputs;
using flash::mjwarp::MassSolveInputs;
using flash::mjwarp::SmoothForceInputs;
constexpr int Compact = -2;
__device__ inline float clip(float x, const float *bounds) {
  return fminf(fmaxf(bounds[0], x), bounds[1]);
}
namespace dx = cusolverdx;
using Potrs9 =
    decltype(dx::Size<9, 9, 1>() + dx::Precision<float>() +
             dx::Type<dx::type::real>() + dx::Function<dx::function::potrs>() +
             dx::FillMode<dx::fill_mode::upper>() +
             dx::Arrangement<dx::row_major, dx::col_major>() + dx::SM<800>() +
             dx::BlockDim<64>() + dx::Block());

template <int MaskWords>
__global__ void actuation_solve(ActuationInputs a, SmoothForceInputs s,
                                MassSolveInputs one, MassSolveInputs six,
                                MassSolveInputs nine) {

  const int w = blockIdx.x, x = threadIdx.x;
  constexpr int Threads = 64;
  __shared__ unsigned present[MaskWords ? MaskWords : 1];
  __shared__ __align__(16) float upper9[81], rhs9[9];
  const int first9 = nine.blocks ? nine.block_start[0] : -1;
  if (nine.blocks) {
    const int address = w * nine.factor_stride + nine.factor_address[first9];
    for (int k = x; k < 81; k += 64)
      upper9[k] = nine.factor[address + k];
  }
  if constexpr (MaskWords > 0) {
    // Two warps cover the admitted activity words, including zero tails.
    for (int base = 0; base < MaskWords * 32; base += 64) {
      const int b = base + x;
      bool active = false;
      if (b < s.bodies) {
        const float *ft = s.external + 6 * (w * s.bodies + b);
        active = ft[0] != 0 || ft[1] != 0 || ft[2] != 0 || ft[3] != 0 ||
                 ft[4] != 0 || ft[5] != 0;
      }
      const unsigned word = __ballot_sync(0xffffffffu, active);
      if (!(x % 32))
        present[b / 32] = word;
    }
  }
  for (int d = x; d < a.nv; d += Threads)
    a.qfrc[w * a.nv + d] = 0.f;
  for (int u = x; u < a.actuators; u += Threads) {
    const int t = w * a.actuators + u;
    float ctrl = 0.0f;
    if (a.ctrlnum[u] > 0) {
      int adr = a.ctrladr[u];
      ctrl = a.ctrl[w * a.controls + adr];
      if (a.ctrllimited[adr] && !a.disable_clampctrl)
        ctrl = clip(ctrl, a.ctrlrange +
                              ((w % a.ctrlrange_rows) * a.controls + adr) * 2);
    }
    const float *gp = a.gainprm + ((w % a.gain_rows) * a.actuators + u) * 10;
    const float *bp = a.biasprm + ((w % a.bias_rows) * a.actuators + u) * 10;
    float gain = gp[0];
    if (a.gain_type[u] == 1)
      gain = gp[0] + gp[1] * a.length[t] + gp[2] * a.velocity[t];
    float bias = 0.0f;
    if (a.bias_type[u] == 1)
      bias = bp[0] + bp[1] * a.length[t] + bp[2] * a.velocity[t];
    float force = gain * ctrl + bias;
    if (a.forcelimited[u])
      force = clip(force, a.forcerange +
                              ((w % a.forcerange_rows) * a.actuators + u) * 2);
    a.force[t] = force;
  }
  __syncthreads();
  for (int u = x; u < a.actuators; u += Threads) {
    const int t = w * a.actuators + u, at = a.moment_rowadr[t],
              count = a.moment_rownnz[t];
    const float force = a.force[t];
    for (int i = 0; i < count; ++i) {
      const int slot = w * a.moments + at + i, dof = a.moment_colind[slot];
      atomicAdd(a.qfrc + w * a.nv + dof, a.moment[slot] * force);
    }
  }
  __syncthreads();
  for (int d = x; d < a.nv; d += Threads) {
    const int t = w * a.nv + d, j = a.dof_joint[d];
    float force = a.qfrc[t];
    if (a.gravity_enabled && a.actgravcomp[j])
      force += a.gravcomp[t];
    if (a.jointlimited[j])
      force = clip(force,
                   a.jointrange + ((w % a.jointrange_rows) * a.joints + j) * 2);
    a.qfrc[t] = force;
    {
      // The current DOF owns both publications; no force reload or barrier.
      float value = s.passive[w * s.vector_stride + d] -
                    s.bias[w * s.vector_stride + d] + force +
                    s.applied[w * s.vector_stride + d];
      float sum = 0.f;
      bool any = true;
      if constexpr (MaskWords == 2)
        any = bool(present[0] | present[1]);
      else if constexpr (MaskWords == 4)
        any = bool(present[0] | present[1] | present[2] | present[3]);
      if (any) {
        const float *cd = s.cdof + (w * s.nv + d) * 6;
        // cd is (angular, linear); applied wrench is (force, torque), in world
        // axes.
        float jac[6] = {cd[3], cd[4], cd[5], cd[0], cd[1], cd[2]};
        int body = s.dof_body[d];
        const float *com =
            s.subtree_com + (w * s.bodies + s.body_root[body]) * 3;
        // Model's depth-first order makes each subtree a contiguous body
        // interval.
        for (int b = body; b < body + s.subtree_size[body]; ++b) {
          const float *ft = s.external + (w * s.bodies + b) * 6;
          if constexpr (MaskWords > 0) {
            if (!(present[b / 32] & (1u << (b % 32))))
              continue;
          } else if (ft[0] == 0 && ft[1] == 0 && ft[2] == 0 && ft[3] == 0 &&
                     ft[4] == 0 && ft[5] == 0)
            continue;
          const float *point = s.body_com + (w * s.bodies + b) * 3;
          float dx = point[0] - com[0], dy = point[1] - com[1],
                dz = point[2] - com[2];
          float cx = cd[1] * dz - cd[2] * dy;
          float cy = cd[2] * dx - cd[0] * dz;
          float cz = cd[0] * dy - cd[1] * dx;
          float dot6 = 0.0f;
          for (int k = 0; k < 6; ++k)
            dot6 += jac[k] * ft[k];
          float dot3 = cx * ft[0] + cy * ft[1] + cz * ft[2];
          sum += dot6 + dot3;
        }
      }
      const float rhs = value + sum;
      s.smooth[w * s.vector_stride + d] = rhs;
      // The actual DOF producer also stages its 9-DOF RHS. The following
      // completion barrier serves both scalar consumers and cuSolverDx;
      // there is no separate RHS read/copy or boundary barrier.
      if (nine.blocks && d >= first9 && d < first9 + 9)
        rhs9[d - first9] = rhs;
    }
  }

  __syncthreads(); // all current RHS values, including shared9, are ready
#pragma unroll
  for (int group = 0; group < 2; ++group) {
    const MassSolveInputs a = group ? six : one;
    const int size = group ? 6 : 1;
    for (int block = x; block < a.blocks; block += 64) {
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

  if (nine.blocks) {
    Potrs9().execute(upper9, 9, rhs9, 9);
    __syncthreads();
    for (int k = x; k < 9; k += 64)
      nine.solution[w * nine.solution_stride + first9 + k] = rhs9[k];
  }
}
} // namespace flash::fp32
extern "C" {
const char *flash_fp32_actuation_solve_source_id() {
  return FLASH_FP32_ACTUATION_SOLVE_SOURCE_ID;
}
int flash_fp32_actuation_solve_arch() { return FLASH_CUDA_ARCH; }
std::size_t flash_fp32_actuation_solve_actuation_bytes() {
  return sizeof(flash::mjwarp::ActuationInputs);
}
std::size_t flash_fp32_actuation_solve_smooth_bytes() {
  return sizeof(flash::mjwarp::SmoothForceInputs);
}
std::size_t flash_fp32_actuation_solve_mass_bytes() {
  return sizeof(flash::mjwarp::MassSolveInputs);
}
int flash_fp32_actuation_solve_launch(
    const flash::mjwarp::ActuationInputs *a,
    const flash::mjwarp::SmoothForceInputs *s,
    const flash::mjwarp::MassSolveInputs *one,
    const flash::mjwarp::MassSolveInputs *six,
    const flash::mjwarp::MassSolveInputs *nine, cudaStream_t stream) {
  if (!a || !s || !one || !six || !nine || a->worlds < 1 || a->nv < 1 ||
      a->actuators < 1 || a->controls < 1 || a->joints < 1 || a->moments < 1 ||
      s->worlds != a->worlds || s->nv != a->nv || s->vector_stride != a->nv ||
      s->bodies < 1 || s->actuator != a->qfrc || one->worlds != a->worlds ||
      six->worlds != a->worlds || nine->worlds != a->worlds ||
      one->nv != a->nv || six->nv != a->nv || nine->nv != a->nv ||
      one->blocks < 0 || six->blocks < 0 || one->blocks + six->blocks > 32 ||
      nine->blocks < 0 || nine->blocks > 1 || one->rhs != s->smooth ||
      six->rhs != s->smooth || nine->rhs != s->smooth)
    return int(cudaErrorInvalidValue);
  if (s->bodies <= 64)
    flash::fp32::actuation_solve<2>
        <<<a->worlds, 64, 0, stream>>>(*a, *s, *one, *six, *nine);
  else if (s->bodies <= 128)
    flash::fp32::actuation_solve<4>
        <<<a->worlds, 64, 0, stream>>>(*a, *s, *one, *six, *nine);
  else
    flash::fp32::actuation_solve<0>
        <<<a->worlds, 64, 0, stream>>>(*a, *s, *one, *six, *nine);
  return int(cudaGetLastError());
}
}
