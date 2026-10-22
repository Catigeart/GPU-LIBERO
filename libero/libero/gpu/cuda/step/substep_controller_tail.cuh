#pragma once
// Included after substep_tail.cuh in a private namespace. One source for
// controller publication, position/velocity checks and the next forward gate.
__device__ void controller_tail(const StagedSubstepArguments &s, int w, bool policy_step) {
  const auto &args = *s.arguments;
  run_controller(args, w, policy_step);
  if (threadIdx.x == 0)
    cf_recovery::admit(w, args.state.intent, args.state.first_error, args.recovery.active);
  __syncthreads();
  check_and_reset(args, w, args.state.qpos, args.state.nq, 4);
  check_and_reset(args, w, args.state.qvel, args.state.nv, 5);
  begin_forward(args, w, args.state.intent);
  if (threadIdx.x == 0) {
    if (args.position.work[w]) {
      args.continuation[w] = int(SubstepEntry::PhysicsPosition);
      s.phase[w] = -1;
    } else s.phase[w] = 2;
  }
}
