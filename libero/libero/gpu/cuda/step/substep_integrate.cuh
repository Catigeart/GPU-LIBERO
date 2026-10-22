#pragma once
// Shared Euler gate, mathematical body and publication for native world entries.
__device__ void integrate(const SubstepArguments &args, int world) {
  CF_STEP_PROFILE_SCOPE(Integrate);
  const auto &state = args.state;
  physics_euler::cf_euler::gate_core_block(world, state.intent, state.epoch, state.forward_valid,
                                           state.forward_written, state.forward_error, state.first_error,
                                           args.euler.active);
  __syncthreads();
  {
    const auto &in = args.euler;
    physics_euler::cf_euler::evaluate_core(
        world, in.active, in.qp, in.qv, in.time, in.qm, in.smooth, in.constraint, in.acc, in.damping, in.madr,
        in.parent, in.jt, in.padr, in.vadr, in.qh, in.qhi, in.count, in.ma, in.damp, in.status, in.ms, in.wc,
        in.wl, in.dt, in.nq, in.nv, in.nm, in.nj, in.qacc_stride, in.joint_quat, in.joint_type_offsets,
        in.joint_type_indices);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    physics_euler::cf_euler::publish(world, args.euler.active, args.euler.status, state.epoch,
                                     state.step_written, state.step_valid, state.first_error,
                                     state.forward_valid);
    physics_euler::cf_euler::expected(world, state.intent, state.expected_epoch);
    physics_euler::cf_euler::count(world, state.step_valid, state.integration_count);
  }
  __syncthreads();
}
