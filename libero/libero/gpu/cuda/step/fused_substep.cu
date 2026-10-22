#include "precision.cuh"
// Maintained single-physical-substep CUDA implementation (catalogue dimensions).
// The ordinary kernel retains the CPU application's three complete forwards.
// A separate FP32 candidate follows step1 -> OSC -> step2 with one solve.
// Newton numerical bodies remain separately compiled in newton_solver.cu.
// CMake compiles this file directly. There is no build/runtime source generation.

// One block owns one world's physical substep; no loop over control substeps.
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <type_traits>
#include <cublasdx.hpp>
#include "substep_kernel.cuh"
#if defined(FLASH_FP64_MOTION_PASSIVE)
#include "fp64_motion_passive.cuh"
#endif
#include "substep_shared_layout.cuh"
#include "newton_solver.cuh"
#include "newton_shared_layout.cuh"
#include "../postprocess/post_solver.cuh"
#include "substep_shared.cuh"
#include "substep_profile.cuh"
#include "../common/math.cuh"
#include "../common/vector.cuh"
#include "../common/warp.cuh"
#include "../common/packed_mass.cuh"
#include "forward_mass.cuh"
#include "../control/osc_runtime.cuh"
#include "../control/controller_pipeline.cuh"
#include "../runtime/forward.cuh"
#include "../runtime/position_reuse.cuh"
#include "../runtime/recovery.cuh"

using flash_libero::SubstepShape;


#include "substep_forward_math.cuh"

namespace flash_libero {
namespace {

#include "substep_forward_prepare.cuh"
#include "substep_forward_newton.cuh"
#include "substep_forward_post.cuh"

// Complete collective suffix, retained for the original fused entry.
__device__ __forceinline__ void solve_dynamics(const SubstepArguments &args, int world,
    unsigned char *shared, bool post_outputs, bool reuse, int smooth_reuse_status = -1) {
  CF_STEP_PROFILE_STAGES(CF_STEP_PROFILE_PHASE(Actuation));
  prepare_solver_inputs(args, world, shared, reuse, smooth_reuse_status
#if defined(FLASH_SUBSTEP_PROFILE)
      , step_profile_stages
#endif
  );
  CF_STEP_PROFILE_NEXT(CF_STEP_PROFILE_PHASE(Newton));
  run_forward_newton(args, world, shared);
  CF_STEP_PROFILE_NEXT(CF_STEP_PROFILE_PHASE(Post));
  publish_forward(args, world, shared, post_outputs
#if defined(FLASH_SUBSTEP_PROFILE)
      , step_profile_stages
#endif
  );
}

// Complete CPU-compatible forward. Keep the FP64 order and every warmstart
// write; only the FP32 substep entry separates these two physical stages.
__device__ __forceinline__ void forward_dynamics(const SubstepArguments &args, int world,
                                                 unsigned char *shared, bool post_outputs,
                                                 int smooth_reuse_status = -1, const int *motion_prepared = nullptr) {
  CF_STEP_PROFILE_SCOPE(Forward);
  const bool reuse = prepare_dynamics(args, world, nullptr, motion_prepared);
  solve_dynamics(args, world, shared, post_outputs, reuse, smooth_reuse_status);
}

#include "substep_tail.cuh"
} // namespace

// This is the complete world-local substep body. Normal execution enters once.
// Continuations exist only across the original parallel-prefix/reset boundaries.
// A block-wide return always occurs after a barrier and a world-uniform predicate.
#if defined(FLASH_FP64_MOTION_PASSIVE)
__global__ void motion_residual_kernel(const SubstepArguments *__restrict__ arguments, bool policy_step,
                                       SubstepEntry entry, bool publish_post, const int *motion_prepared) {
#else
__global__ void fused_substep_kernel(const SubstepArguments *__restrict__ arguments, bool policy_step,
                                     SubstepEntry entry, bool publish_post) {
  const int *motion_prepared = nullptr;
#endif
  const auto &args = *arguments;
  auto *shared = substep_shared_data();
  const int world = blockIdx.y;
  if (entry != SubstepEntry::Start && args.continuation[world] != int(entry))
    return;
  CF_STEP_PROFILE_BEGIN(entry, policy_step);
  int smooth_reuse_status = -1;

  if (entry == SubstepEntry::Start) {
    if (threadIdx.x == 0) {
      args.continuation[world] = int(SubstepEntry::Complete);
      args.forward_workspace.metadata[size_t(world) * ForwardCacheWords + ForwardCacheValid] = 0;
    }
    __syncthreads();

    // RobotEnv.step: sim.forward() precedes controller.set_goal/run_controller.
    // Preserve both intermediate force snapshots on policy steps (CPU robot
    // force/torque history); other substeps have no force consumer before the
    // final forward. Newton and warmstart still execute at all three points.
    // Admission and the parallel position prefix were launched by the caller.
    forward_dynamics(args, world, shared, policy_step, -1, motion_prepared);
    // This register value dies at a position suspension. The next entry starts
    // with -1; it cannot reuse a payload produced before a prefix refresh.
    const auto &smooth = args.smooth_commit;
    if (args.state.forward_valid[world] == 1 && !args.state.first_error[world] &&
        smooth.valid[world] == 1 && !smooth.error[world] &&
        smooth.written[world] == smooth.epoch[world])
      smooth_reuse_status = smooth.repaired[world] ? 25 : 0;
    if (threadIdx.x == 0) {
      const auto &in = args.controller_plan;
      osc_cuda::plan(world, in.active, in.first, in.needs, in.refresh, in.call, in.force);
    }
    // begin_forward consumes the plan on the same leader, then publishes once.
    begin_forward(args, world, args.controller_plan.refresh);
    if (args.position.work[world]) {
      if (threadIdx.x == 0)
        args.continuation[world] = int(SubstepEntry::ControllerPosition);
      return;
    }
  }

  if (entry == SubstepEntry::Start || entry == SubstepEntry::ControllerPosition) {
    // Controller.update requests its own forward, including its Newton/warmstart.
    forward_dynamics(args, world, shared, policy_step, smooth_reuse_status,
        entry == SubstepEntry::ControllerPosition ? motion_prepared : nullptr);
    run_controller(args, world, policy_step);

    // MuJoCo mj_step: checkPos -> checkVel -> forward -> checkAcc -> Euler.
    if (threadIdx.x == 0) {
      cf_recovery::admit(world, args.state.intent, args.state.first_error, args.recovery.active);
    }
    __syncthreads();
    check_and_reset(args, world, args.state.qpos, args.state.nq, 4);
    check_and_reset(args, world, args.state.qvel, args.state.nv, 5);
    begin_forward(args, world, args.state.intent);
    if (args.position.work[world]) {
      if (threadIdx.x == 0)
        args.continuation[world] = int(SubstepEntry::PhysicsPosition);
      return;
    }
  }

  if (entry != SubstepEntry::AccelerationReset) {
    // A qualified control graph may defer force/sensor work in intermediate
    // substeps. The final substep keeps Post here, before Euler, exactly as in
    // mj_forward/mj_step. Every forward still solves Newton and publishes qacc.
    forward_dynamics(args, world, shared, publish_post, -1,
        entry == SubstepEntry::PhysicsPosition ? motion_prepared : nullptr);
  }
  finish_substep(args, world, entry);
}

#if FLASH_REAL_BITS == 32 && !defined(FLASH_FP64_MOTION_PASSIVE)
// Experimental FP32 schedule. It is a distinct kernel, so the accepted
// three-forward kernel keeps its own register and shared-memory allocation.
__global__ void single_solve_substep_kernel(const SubstepArguments *__restrict__ arguments,
                                           bool policy_step, SubstepEntry entry, bool publish_post) {
  const auto &args = *arguments;
  auto *shared = substep_shared_data();
  const int world = blockIdx.y;
  if (entry != SubstepEntry::Start && entry != SubstepEntry::PositionChecks &&
      args.continuation[world] != int(entry))
    return;
  CF_STEP_PROFILE_BEGIN(entry, policy_step);
  // step1 checks must precede the multi-CTA position prefix. This small entry
  // runs before the host enqueues that prefix; it performs no forward/solve.
  if (entry == SubstepEntry::PositionChecks) {
    if (threadIdx.x == 0) {
      args.continuation[world] = int(SubstepEntry::Complete);
      args.forward_workspace.metadata[size_t(world) * ForwardCacheWords + ForwardCacheValid] = 0;
      cf_recovery::admit(world, args.state.intent, args.state.first_error, args.recovery.active);
    }
    __syncthreads();
    check_and_reset(args, world, args.state.qpos, args.state.nq, 4);
    check_and_reset(args, world, args.state.qvel, args.state.nv, 5);
    return;
  }
  if (entry == SubstepEntry::Start) {
    CF_STEP_PROFILE_SCOPE(Forward);
    prepare_dynamics(args, world);
    if (threadIdx.x == 0) {
      const auto &in = args.controller_plan;
      // MJWarp consumes freshly prepared physical inputs even on the first
      // substep after reset; it never starts from the constructor's cache.
      osc_cuda::plan(world, in.active, in.first, in.needs, in.refresh, in.call, 1);
    }
    __syncthreads();
    run_controller(args, world, policy_step);
    solve_dynamics(args, world, shared, publish_post, false);
  }
  finish_substep(args, world, entry);
}
#endif

#if !defined(FLASH_FP64_MOTION_PASSIVE)
// Select exactly the suspended worlds. This small kernel performs no physics.
__global__ void select_substep_continuation(const SubstepArguments *__restrict__ arguments,
                                            SubstepEntry entry, int worlds, int *predicate) {
  if (threadIdx.x != 0)
    return;
  const auto &args = *arguments;
  int any = 0;
  for (int world = 0; world < worlds; ++world) {
    const int selected = args.continuation[world] == int(entry);
    args.position.work[world] = selected;
    args.recovery.reset_mask[world] = selected;
    any |= selected;
  }
  predicate[0] = any;
}
#endif // original selector is shared by the peeled host dispatcher
} // namespace flash_libero
