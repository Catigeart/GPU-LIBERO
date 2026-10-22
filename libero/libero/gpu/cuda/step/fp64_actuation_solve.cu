// Copyright 2025 The Newton Developers
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// Copyright 2026 Flash-LIBERO contributors
// SPDX-License-Identifier: Apache-2.0
// FP64 adaptation of fp32/fused_actuation_solve.cu. Actuators and DOFs own
// independent work; the CPU2.3.7 packed LDL factor remains the mass owner.
// Public admission/publication stays at the original residual forward point.
#include "fp64_actuation_solve.cuh"
#include "../common/vector.cuh"
#include "forward_mass.cuh"
#include "../common/warp.cuh"
#include <cmath>

namespace flash_libero {
__device__ bool prepare_fp64_force(const SubstepArguments &a, int w,
    const int *motion_ready, double *scalar_force, double *force, double *acceleration) {
  const int lane = threadIdx.x;
  const auto &u = a.actuation;
  const auto &s = a.smooth_acceleration;
  const int nv = s.nv, nu = u.nu, nb = s.nbody;
  // Exceptional inputs retain the original checked implementation. A failed
  // private candidate does not change warning counters or the factor owner.
  if (!motion_ready || motion_ready[w] != 1 || a.state.requested_mask[w] != 1 ||
      a.state.first_error[w] || nv > 128 || nu > 32 || nb > 64 || (u.disable_flags & 1024) ||
      a.control_gate.valid[w] != 1 || a.transmission.position_status[w] != 0 ||
      s.count[w] != 1 || s.body_count[w] != nb - 1 ||
      s.position_valid[w] != 1 || s.position_written[w] != s.forward_epoch[w] ||
      s.source_active[w] != 1 ||
      (s.source_status[w] != 0 && s.source_status[w] != 5)) return false;

  __shared__ unsigned present[2];
  const size_t v = size_t(w) * nv, m = size_t(w) * s.nm, act = size_t(w) * nu;
  const double *control = a.control_copy.source + act;
  const double *qvel = a.transmission.qvel + v;
  const double *moment = u.moment + act * nv;
  const double *passive = s.passive + v, *bias = a.bias_forces.result + v;
  bool bad = false;
  for (int k = lane; k < nb - 1; k += blockDim.x)
    bad |= s.body_ids[size_t(w) * (nb - 1) + k] != k + 1;
  // Both warps vote, including padded inactive bodies. The mask avoids a
  // complete body walk for every DOF when no external wrench is present.
  if (lane < 64) {
    const int body = lane;
    bool active = false;
    if (body > 0 && body < nb) {
      const double *ft = s.xfrc + (size_t(w) * nb + body) * 6;
      for (int k = 0; k < 6; ++k) {
        bad |= !isfinite(ft[k]);
        active |= ft[k] != 0.0;
      }
    }
    const unsigned mask = __ballot_sync(0xffffffffu, active);
    if (!(lane & 31)) present[lane / 32] = mask;
  }
  int bad_ctrl = libero_warp::no_index;
  double ctrl = lane < nu ? control[lane] : 0.0;
  int ctrl_route = 0;
  if (lane < nu && !(u.disable_flags & 128)) {
    if (ctrl < u.ctrl_lo[lane]) { ctrl = u.ctrl_lo[lane]; ctrl_route = -1; }
    else if (ctrl > u.ctrl_hi[lane]) { ctrl = u.ctrl_hi[lane]; ctrl_route = 1; }
  }
  if (lane < 32) {
    bad_ctrl = libero_warp::first_index(lane < nu &&
        (ctrl != ctrl || ctrl > 1.e10 || ctrl < -1.e10), lane);
    // BADCTRL must update the original persistent warning exactly once, at
    // its original invocation. Leave that case to the residual fallback.
    bad |= bad_ctrl != libero_warp::no_index;
  }
  double velocity = 0.0, value = 0.0, before_limit = 0.0;
  int force_route = 0;
  if (lane < nu) {
    // Transmission still runs at its native admission point, where its
    // status also qualifies consumption of this private candidate. Only an
    // affine velocity term needs a duplicate dot product here. Fixed gains
    // and zero velocity coefficients consume no actuator velocity.
    if (u.bias_type[lane] == 1 && u.bias2[lane] != 0.0)
      for (int d = 0; d < nv; ++d)
        velocity += moment[size_t(lane) * nv + d] * qvel[d];
    value = u.gain[lane] * ctrl;
    if (u.bias_type[lane] == 1)
      value += u.bias0[lane] + u.bias1[lane] * u.length[act + lane] +
               u.bias2[lane] * velocity;
    before_limit = value;
    if (u.force_limited[lane]) {
      if (value < u.force_lo[lane]) { value = u.force_lo[lane]; force_route = -1; }
      else if (value > u.force_hi[lane]) { value = u.force_hi[lane]; force_route = 1; }
    }
    if (u.disable_flags & 1024) value = before_limit = ctrl = 0.0;
    scalar_force[lane] = value;
    bad |= !isfinite(value) || !isfinite(velocity) || !isfinite(before_limit);
  }
  if (__syncthreads_or(bad)) return false;

  // The producer owns its actuator sum, smooth RHS and projected external
  // wrench. Body order and force-then-torque additions follow mj_xfrcAccumulate.
  for (int d = lane; d < nv; d += blockDim.x) {
    double actuator = 0.0;
    for (int j = 0; j < nu; ++j)
      actuator += moment[size_t(j) * nv + d] * scalar_force[j];
    double rhs = ((passive[d] - bias[d]) + s.applied[v + d]) + actuator;
    const double initial = rhs;
    const double *cd = s.cdof + (v + d) * 6;
    if (present[0] | present[1]) {
      for (int body = 1; body < nb; ++body) {
        if (!(present[body / 32] & (1u << (body % 32)))) continue;
        if (!s.ancestor[size_t(body) * nv + d]) continue;
        const size_t b = size_t(w) * nb + body;
        const double *ft = s.xfrc + b * 6;
        const double *point = s.xipos + b * 3;
        const double *com = s.subtree_com + (size_t(w) * nb + s.body_rootid[body]) * 3;
        const double dx = point[0] - com[0], dy = point[1] - com[1], dz = point[2] - com[2];
        const double jac[3] = {cd[3] + cd[1]*dz - cd[2]*dy,
                               cd[4] + cd[2]*dx - cd[0]*dz,
                               cd[5] + cd[0]*dy - cd[1]*dx};
        double f = 0.0, t = 0.0;
        for (int k = 0; k < 3; ++k) {
          if (ft[k] != 0.0) f += jac[k] * ft[k];
          if (ft[k + 3] != 0.0) t += cd[k] * ft[k + 3];
        }
        rhs = (rhs + f) + t;
      }
    }
    bad |= !isfinite(actuator) || !isfinite(initial) || !isfinite(rhs);
    u.qfrc_actuator[v + d] = actuator;
    s.smooth[v + d] = s.seed[v + d] = initial;
    force[d] = rhs;
  }
  if (__syncthreads_or(bad)) return false;
  if (!forward_mass::solve(force, acceleration, s.source_ld + m,
      s.source_inv + v, s.madr, s.parent, a.forward_workspace.mass_plan)) return false;

  // Retain established private work views. Only the residual's original gate
  // may publish smooth/acceleration or increment the admitted reuse counter.
  for (int j = lane; j < nu; j += blockDim.x) {
    u.force[act + j] = scalar_force[j];
    u.force_pre[act + j] = before_limit;
    u.ctrl_used[act + j] = ctrl;
    u.ctrl_route[act + j] = ctrl_route;
    u.force_route[act + j] = force_route;
  }
  for (int d = lane; d < nv; d += blockDim.x) {
    s.scratch[v + d] = s.qfrc[v + d] = force[d];
    s.qacc_smooth[v + d] = acceleration[d];
    s.rhs[size_t(w)*4*nv + d] = force[d];
    s.solution[size_t(w)*4*nv + d] = acceleration[d];
    s.inv[v + d] = s.source_inv[v + d];
    s.sqrt_inv[v + d] = s.source_sqrt_inv[v + d];
    s.pivot[v + d] = s.source_pivot[v + d];
  }
  for (int d = lane + nv; d < 4*nv; d += blockDim.x)
    s.rhs[size_t(w)*4*nv + d] = s.solution[size_t(w)*4*nv + d] = 0.0;
  for (int k = lane; k < s.nm; k += blockDim.x) s.ld[m + k] = s.source_ld[m + k];
  if (lane == 0) {
    u.status[w] = u.warn_badctrl_count[w] = 0;
    u.warn_badctrl_lastinfo[w] = -1;
    s.entry_status[w] = s.d02_status[w] = 0;
    s.solve_status[w] = s.source_warning_count[w] ? 5 : 0;
    s.warning_count[w] = s.source_warning_count[w];
    s.warning_last[w] = s.source_warning_last[w];
    s.status[w] = s.source_warning_count[w] ? 25 : 0;
  }
  __syncthreads(); // close private stores before an in-kernel consumer
  return true;
}

__global__ __launch_bounds__(256) void fp64_actuation_solve_kernel(
    const SubstepArguments *arguments, const int *motion_ready, int *ready,
    SubstepEntry entry) {
  const int w = blockIdx.y;
  if (threadIdx.x == 0) ready[w] = 0;
  if (entry != SubstepEntry::Start && arguments->continuation[w] != int(entry)) return;
  __shared__ double scalar_force[32], force[128], acceleration[128];
  const bool produced = prepare_fp64_force(*arguments,w,motion_ready,
                                         scalar_force,force,acceleration);
  if (threadIdx.x == 0) ready[w] = produced ? 1 : 0;
}
} // namespace flash_libero
