// Copyright 2026 Flash-LIBERO contributors
// SPDX-License-Identifier: Apache-2.0
// Actual independent FP64 initialization and remainder, retaining accepted
// FP32-organized motion/force producers and the CPU three-solve schedule.
#include "precision.cuh"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <type_traits>
#include <cublasdx.hpp>
#include "aligned_substep.cuh"
#include "fp64_actuation_solve.cuh"
#include "substep_shared.cuh"
#include "substep_profile.cuh"
#include "newton_solver.cuh"
#include "newton_workspace.cuh"
#include "../common/math.cuh"
#include "../common/vector.cuh"
#include "../common/warp.cuh"
#include "../common/packed_mass.cuh"
#include "forward_mass.cuh"
#include "../runtime/forward.cuh"
using flash_libero::SubstepShape;
#include "substep_forward_math.cuh"

namespace flash_libero {
namespace {
#include "staged_phase.cuh"
#include "substep_forward_prepare.cuh"
}

__global__ void aligned_prepare_kernel(const AlignedSubstepArguments *request,
    SubstepEntry entry, int phase, bool motion_enabled, bool force_enabled) {
  const auto &s = request->stages;
  const auto &args = *s.arguments;
  const int w = blockIdx.y;
  if (!staged_enter(s, args, w, entry, phase)) return;
  auto *shared = substep_shared_data();
  const int first = entry == SubstepEntry::Start ? 0 :
      entry == SubstepEntry::ControllerPosition ? 1 : 2;
  const int *motion = motion_enabled ? s.motion_reuse : nullptr;
  int *force = force_enabled ? request->force_ready : nullptr;
  // Controller/checks/begin_forward have already run in the preceding Post
  // tail. Its shared arena is dead. Reproduce the accepted resident producer
  // here before the third forward; a prefix continuation has its entry producer.
  if (phase == 2 && first != 2 && force) {
    const bool produced = prepare_fp64_force(args, w, motion,
        shared_at<real_t>(shared, SmoothLayout::scalar_force),
        shared_at<real_t>(shared, SmoothLayout::force),
        shared_at<real_t>(shared, SmoothLayout::acceleration));
    if (threadIdx.x == 0) force[w] = produced ? 2 : 0;
  }
  const bool reuse = prepare_dynamics(args, w, nullptr, phase == first ? motion : nullptr);
  const int reuse_status = phase == 1 ? s.smooth_reuse_status[w] : -1;
  if (phase == 1 && first != 1 && reuse_status < 0) force = nullptr;
  prepare_solver_inputs(args, w, shared, reuse, reuse_status, force);
}

// Smooth public owners are complete at A's exit. Init has its own compact
// arena; it never interprets those global inputs as a shared allocation.
__global__ void aligned_newton_initial_kernel(const AlignedSubstepArguments *request, int phase) {
  const auto &s = request->stages;
  const int w = blockIdx.y;
  if (s.phase[w] != phase) return;
  const auto &args = *s.arguments;
  const auto &newton = args.newton;
  const auto &parameters = *newton.parameters;
  physics_newton::NewtonOptions options{};
  options.stop_policy = newton.stop_policy;
  options.audit = newton.audit;
  options.first_disagreement = newton.first_disagreement;
  const auto &smooth = args.smooth_commit;
  const bool resident = args.state.requested_mask[w] != 0 && smooth.valid[w] == 1 &&
      !smooth.error[w] && smooth.written[w] == smooth.epoch[w];
  if (resident) {
    options.smooth_force = smooth.pf + size_t(w) * SubstepShape::nv;
    options.smooth_acceleration = smooth.pa + size_t(w) * SubstepShape::nv;
  }
  physics_newton::solve_newton<physics_newton::NewtonPhase::Initialize>(substep_shared_data(),
      parameters.model, parameters.worlds[w], newton.upstream, newton.enabled,
      newton.publication, newton.expected_epoch, options,
      request->newton_initial + size_t(w) * physics_newton::NewtonInitialLayout::bytes);
}

__global__ void aligned_newton_resume_kernel(const AlignedSubstepArguments *request, int phase) {
  const auto &s = request->stages;
  const int w = blockIdx.y;
  if (s.phase[w] != phase) return;
  const auto &args = *s.arguments;
  const auto &newton = args.newton;
  const auto &parameters = *newton.parameters;
  auto *shared = substep_shared_data();
  physics_newton::NewtonOptions options{};
  options.stop_policy = newton.stop_policy;
  options.audit = newton.audit;
  options.first_disagreement = newton.first_disagreement;
  const auto &smooth = args.smooth_commit;
  const bool resident = args.state.requested_mask[w] != 0 && smooth.valid[w] == 1 &&
      !smooth.error[w] && smooth.written[w] == smooth.epoch[w];
  if (resident) {
    options.smooth_force = shared_at<real_t>(shared, SmoothLayout::force);
    options.smooth_acceleration = shared_at<real_t>(shared, SmoothLayout::acceleration);
    for (int j = threadIdx.x; j < SubstepShape::nv; j += blockDim.x) {
      options.smooth_force[j] = smooth.pf[size_t(w) * SubstepShape::nv + j];
      options.smooth_acceleration[j] = smooth.pa[size_t(w) * SubstepShape::nv + j];
    }
  }
  // Resume's control/vector restore barrier also publishes the smooth inputs.
  physics_newton::solve_newton<physics_newton::NewtonPhase::Resume>(shared,
      parameters.model, parameters.worlds[w], newton.upstream, newton.enabled,
      newton.publication, newton.expected_epoch, options,
      request->newton_initial + size_t(w) * physics_newton::NewtonInitialLayout::bytes);
  const real_t *qacc = shared_at<real_t>(shared, NewtonLayout::qacc);
  for (int j = threadIdx.x; j < SubstepShape::nv; j += blockDim.x)
    s.qacc[size_t(w) * SubstepShape::nv + j] = qacc[j];
}
} // namespace flash_libero
