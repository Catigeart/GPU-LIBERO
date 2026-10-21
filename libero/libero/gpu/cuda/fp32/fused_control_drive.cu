// Copyright 2025 The Newton Developers
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// Copyright 2026 Flash-LIBERO contributors
// SPDX-License-Identifier: Apache-2.0
// Continuous world-local control preparation -> OSC -> unconstrained solve.
#include "../control/osc_solver.cuh"
#include "../dynamics/mjwarp_actuation.cuh"
#include "../dynamics/mjwarp_mass_solve.cuh"
#include "../dynamics/mjwarp_smooth_force.cuh"
#include "../step/substep_shared_layout.cuh"
#include "../step/substep_types.cuh"
#include "fused_control_prepare.cuh"
#include <cusolverdx.hpp>
#ifndef FLASH_FP32_CONTROL_DRIVE_SOURCE_ID
#error "Build through flash_fp32_fused_control_drive"
#endif
static_assert(sizeof(real_t) == 4);
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
__global__ void
control_drive(ControlPrepareInputs input, const int *policy_source,
              const flash_libero::ControllerSolveData *control, const int *ids,
              float *target, int stride, ActuationInputs act,
              SmoothForceInputs smooth, MassSolveInputs one,
              MassSolveInputs six, MassSolveInputs nine) {
  extern __shared__ __align__(16) unsigned char shared[];
  {

    constexpr int Threads = flash_libero::SubstepShape::threads;
    const auto &a = input.gather;
    const auto &j = input.jacobian;
    const auto &g = j.geometry;
    const int w = blockIdx.x, x = threadIdx.x;
    const float *point = a.position + 3 * (w * a.nsites + a.site);
    if (x < 3)
      input.points[3 * w + x] = point[x];
    if (x == 0 && policy_source)
      input.policy[w] = policy_source[w];
    // Query jobs read the live site position directly, so publishing points
    // does not need a producer barrier. Keep the full public JP/JR arrays.
    const int body = j.body[w];
    for (int dof = x; dof < g.dofs; dof += Threads) {
      float linear[3] = {}, angular[3] = {};
      if (g.ancestor[body * g.ancestor_stride + dof]) {
        const float *motion = g.cdof + 6 * (w * g.dofs + dof);
        const float *center = g.com + 3 * (w * g.bodies + g.root[body]);
        const float dx = point[0] - center[0], dy = point[1] - center[1],
                    dz = point[2] - center[2];
        const float cross[3] = {motion[1] * dz - motion[2] * dy,
                                motion[2] * dx - motion[0] * dz,
                                motion[0] * dy - motion[1] * dx};
        for (int k = 0; k < 3; ++k) {
          angular[k] = motion[k];
          linear[k] = motion[k + 3] + cross[k];
        }
      }
      for (int k = 0; k < 3; ++k) {
        const int at = (3 * w + k) * g.dofs + dof;
        j.position[at] = linear[k];
        j.rotation[at] = angular[k];
      }
    }
    __syncthreads(); // all query rows are ready for selected gathers/EE
                     // velocity
    if (x < 49) {
      int at = a.mi[x];
      a.out_mass[49 * w + x] = at >= 0 ? a.mass[w * a.nmass + at] : 0.f;
    }
    if (x < 42) {
      int axis = x / 7, dof = a.vi[x % 7];
      a.out_jac[42 * w + x] =
          (axis < 3 ? a.jp[(3 * w + axis) * a.nv + dof]
                    : a.jr[(3 * w + axis - 3) * a.nv + dof]);
    }
    if (x < 7) {
      a.out_q[7 * w + x] = a.qpos[w * a.nq + a.qi[x]];
      a.out_v[7 * w + x] = a.qvel[w * a.nv + a.vi[x]];
      a.out_bias[7 * w + x] = a.bias[w * a.nv + a.vi[x]];
    }
    if (x < 3)
      a.out_pos[3 * w + x] = a.position[3 * (w * a.nsites + a.site) + x];
    if (x < 9)
      a.out_rot[9 * w + x] = a.rotation[9 * (w * a.nsites + a.site) + x];
    // Each warp owns a spatial component; lane-local sums then one shuffle
    // reduction replace six serial full-NV loops. The seven selected robot
    // columns above are not sufficient: object DOFs also enter EE velocity.
    // Six8-lane groups compute spatial velocities concurrently inside the
    // existing64-thread OSC block, instead of two warps serializing three axes.
    const int lane = x & 7, axis = x / 8;
    if (axis < 6) {
      const float *row = axis < 3 ? a.jp + (3 * w + axis) * a.nv
                                  : a.jr + (3 * w + axis - 3) * a.nv;
      float value = 0.f;
      for (int k = lane; k < a.nv; k += 8)
        value += row[k] * a.qvel[w * a.nv + k];
      const unsigned mask = 0xffu << (x & 24);
      for (int shift = 4; shift; shift /= 2)
        value += __shfl_down_sync(mask, value, shift, 8);
      if (!lane)
        a.out_vel[6 * w + axis] = value;
    }
    // control_fast first reads only needs/active/error, then its existing block
    // barrier orders all prepared physical fields and policy before any
    // consumer. No additional boundary barrier is necessary here. OSC remains a
    // maintained separate numerical function, not a textual expansion of its
    // matrix solver.
    physics_controller::solve(*control, w, shared);
    if (x < 9)
      target[w * stride + ids[x]] = control->ctrl[9 * w + x];
  }
  // All nine Panda controls are published and consumed by warp0. Other
  // warps only clear disjoint force elements until the existing actuation
  // barrier, which also ends every OSC shared read before arena reuse.
  __syncwarp();
  const ActuationInputs &a = act;
  const SmoothForceInputs &s = smooth;

  const int w = blockIdx.x, x = threadIdx.x;
  constexpr int Threads = 64;
  // OSC is finished. Reuse its arena rather than adding another allocation.
  float *upper9 = reinterpret_cast<float *>(shared);
  float *rhs9 = upper9 + 84;
  unsigned *present = reinterpret_cast<unsigned *>(shared + 384);
  static_assert(flash_libero::OscLayout::bytes >= 400);
  const int first9 = nine.blocks ? nine.block_start[0] : -1;
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
  // Existing force completion also frees the OSC arena. Stage mass/masks
  // alongside CSR projection; the next existing barrier orders both.
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
const char *flash_fp32_control_drive_source_id() {
  return FLASH_FP32_CONTROL_DRIVE_SOURCE_ID;
}
int flash_fp32_control_drive_arch() { return FLASH_CUDA_ARCH; }
std::size_t flash_fp32_control_drive_prepare_bytes() {
  return sizeof(flash::fp32::ControlPrepareInputs);
}
std::size_t flash_fp32_control_drive_control_bytes() {
  return sizeof(flash_libero::ControllerSolveData);
}
std::size_t flash_fp32_control_drive_actuation_bytes() {
  return sizeof(flash::mjwarp::ActuationInputs);
}
std::size_t flash_fp32_control_drive_smooth_bytes() {
  return sizeof(flash::mjwarp::SmoothForceInputs);
}
std::size_t flash_fp32_control_drive_mass_bytes() {
  return sizeof(flash::mjwarp::MassSolveInputs);
}
int flash_fp32_control_drive_launch(
    const flash::fp32::ControlPrepareInputs *input, const int *policy,
    const flash_libero::ControllerSolveData *control, const int *ids,
    float *target, int stride, const flash::mjwarp::ActuationInputs *act,
    const flash::mjwarp::SmoothForceInputs *smooth,
    const flash::mjwarp::MassSolveInputs *one,
    const flash::mjwarp::MassSolveInputs *six,
    const flash::mjwarp::MassSolveInputs *nine, cudaStream_t stream) {
  if (!input || !control || !ids || !target || !act || !smooth || !one ||
      !six || !nine || input->gather.worlds < 1 || input->gather.nv < 7 ||
      input->gather.nv > 64 || stride != 9 || act->actuators != 9 ||
      act->ctrl != target || act->controls != stride ||
      act->worlds != input->gather.worlds || act->nv != input->gather.nv ||
      smooth->worlds != act->worlds || smooth->nv != act->nv ||
      smooth->vector_stride != act->nv || smooth->actuator != act->qfrc ||
      one->worlds != act->worlds || six->worlds != act->worlds ||
      nine->worlds != act->worlds || one->blocks < 0 || six->blocks < 0 ||
      nine->blocks < 0 || nine->blocks > 1 || one->blocks + six->blocks > 32 ||
      one->rhs != smooth->smooth || six->rhs != smooth->smooth ||
      nine->rhs != smooth->smooth)
    return int(cudaErrorInvalidValue);
  if (smooth->bodies <= 64)
    flash::fp32::control_drive<2>
        <<<act->worlds, 64, flash_libero::OscLayout::bytes, stream>>>(
            *input, policy, control, ids, target, stride, *act, *smooth, *one,
            *six, *nine);
  else if (smooth->bodies <= 128)
    flash::fp32::control_drive<4>
        <<<act->worlds, 64, flash_libero::OscLayout::bytes, stream>>>(
            *input, policy, control, ids, target, stride, *act, *smooth, *one,
            *six, *nine);
  else
    flash::fp32::control_drive<0>
        <<<act->worlds, 64, flash_libero::OscLayout::bytes, stream>>>(
            *input, policy, control, ids, target, stride, *act, *smooth, *one,
            *six, *nine);
  return int(cudaGetLastError());
}
}
