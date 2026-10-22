#pragma once
// Included in flash_libero's private namespace. Common control/reset/Euler tail.
// Start another forward without rebuilding position on a valid same-epoch hit.
// A miss is returned to the existing multi-CTA prefix, never computed here.
__device__ void begin_forward(const SubstepArguments &args, int world, const int *mask) {
  CF_STEP_PROFILE_SCOPE(BeginForward);
  // All admission operations belong to the same leader. Only their final
  // results are consumed by other lanes; intermediate barriers add no ordering.
  if (threadIdx.x == 0) {
    args.state.forward_mask[world] = mask[world];
    const auto &in = args.foundation_gate;
    forward_cuda::gate_copy(world, in.admission, in.count, in.epoch, in.expected, in.producer_valid, in.event,
                            in.errors, in.stage_out, in.local_out, in.item_out, in.event_out, in.active,
                            in.valid, in.capacity, in.stage, in.admitted, in.public_error, in.public_stage,
                            in.public_local, in.public_item, in.public_event, in.world);
    args.state.requested_mask[world] = args.begin.mask[world];
    const auto &begin = args.begin;
    forward_cuda::begin(world, begin.mask, begin.error, begin.state, begin.source, begin.epoch, begin.active,
                        begin.position, begin.velocity, begin.bias, begin.smooth, begin.expected);
    const auto &cache = args.position;
    position_reuse::check(world, args.begin.active, args.state.first_error, args.state.epoch, cache.saved,
                          cache.cached, cache.constraint_status, cache.constraint_committed,
                          cache.collision_error, cache.mass_warning, cache.collision_warning, cache.work,
                          cache.hit, args.begin.epoch, args.begin.position, cache.written,
                          cache.collision_mask, cache.transport_active, cache.consumed);
  }
  __syncthreads();
}

// The field inventory and defaults are the existing MuJoCo data-reset owners.
// checkPos and checkVel each have their own reset and epoch update, in that order.
__device__ void check_and_reset(const SubstepArguments &args, int world, const real_t *values, int width,
                                int warning) {
  CF_STEP_PROFILE_SCOPE(Checks);
  const auto &reset = args.recovery;
  cf_recovery::check(world, values, reset.active, reset.reset_mask, reset.reason, reset.bad_index,
                     reset.source_error, width, warning);
  __syncthreads();
  // First publish the check's writes, then finish every lane's mask read.
  // The next check reuses this mask: a fast warp must not overwrite it while
  // another warp still reads the previous decision. The vote returns a stable
  // register predicate and closes that read lifetime before either warp leaves.
  const bool needs_reset = __syncthreads_or(reset.reset_mask[world] == 1);
  if (!needs_reset)
    return;
  for (int field = 0; field < reset.field_count; ++field) {
    const auto &view = reset.fields[field];
    cf_recovery::reset_f64(world, view.values, view.defaults, reset.reset_mask, view.width);
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    cf_recovery::warnings(world, reset.warning_number, reset.warning_lastinfo, reset.reset_mask, reset.reason,
                          reset.bad_index);
    cf_recovery::protocol(world, reset.reset_mask, args.state.first_error, args.state.epoch,
                          args.state.expected_epoch, args.state.forward_valid, args.state.step_valid,
                          reset.reset_events);
  }
  __syncthreads();
}

// The pipeline owns gather, seed, solve and publication in an independent TU.
__device__ void run_controller(const SubstepArguments &args, int world, bool policy_step) {
  CF_STEP_PROFILE_SCOPE(Controller);
  physics_controller::run_controller(args, world, policy_step, substep_shared_data());
}

#include "substep_integrate.cuh"

__device__ bool prepare_integrate(const SubstepArguments &args, int world, SubstepEntry entry) {
  if (entry != SubstepEntry::AccelerationReset) {
    if (threadIdx.x == 0) {
      cf_recovery::acceleration(world, args.state.intent, args.state.first_error, args.state.forward_valid,
                                args.state.forward_written, args.state.epoch, args.newton.valid,
                                args.newton.written, args.newton.expected_epoch, args.recovery.active);
    }
    __syncthreads();
    check_and_reset(args, world, args.state.qacc, args.state.nv, 6);
    if (args.recovery.reset_mask[world] == 1) {
      if (threadIdx.x == 0)
        args.continuation[world] = int(SubstepEntry::AccelerationReset);
      return false;
    }
  }

  return true;
}

__device__ void finish_substep(const SubstepArguments &args, int world, SubstepEntry entry) {
  if (!prepare_integrate(args, world, entry)) return;
  // An acceleration-reset continuation has already completed its full forward.
  integrate(args, world);
  if (threadIdx.x == 0)
    args.continuation[world] = int(SubstepEntry::Complete);
}
