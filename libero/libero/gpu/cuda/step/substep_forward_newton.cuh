#pragma once
// The unchanged complete Newton call; smooth shared inputs are already ready.
__device__ __forceinline__ void run_forward_newton(const SubstepArguments &args,
    int world, unsigned char *shared) {
  const auto &newton = args.newton;
  // This invocation owns the force workspace until Newton has finished.
  // Skipped forwards and prefix/reset continuations keep canonical inputs.
  const auto &smooth = args.smooth_commit;
  const bool resident = args.state.requested_mask[world] != 0 && smooth.valid[world] == 1 &&
                        !smooth.error[world] && smooth.written[world] == smooth.epoch[world];
  physics_newton::NewtonOptions options{};
  options.stop_policy = newton.stop_policy;
  options.audit = newton.audit;
  options.first_disagreement = newton.first_disagreement;
  options.smooth_acceleration = resident ?
      flash_libero::shared_at<real_t>(shared, flash_libero::SmoothLayout::acceleration) : nullptr;
  options.smooth_force = resident ?
      flash_libero::shared_at<real_t>(shared, flash_libero::SmoothLayout::force) : nullptr;
  const auto &request = *newton.parameters;
  physics_newton::solve_newton(shared, request.model, request.worlds[world], newton.upstream, newton.enabled,
                               newton.publication, newton.expected_epoch, options);
}
