#pragma once
// Shared native velocity/passive sequence; BranchMotion changes only the
// spatial producer, with the same status, owner and publication contracts.
struct NativeMotion {
  __device__ void operator()(const BodyVelocityData &in, int world) const {
    physics_velocity::d_comvel_bias_cached(
        world, in.active, in.cdof, in.qvel, in.bp, in.ba, in.bn, in.dj, in.jt, in.nb, in.nv, in.cvel,
        in.cdot, in.status, in.cinert, in.gravity, in.disabled, in.cache_acc, in.cache_force,
        in.cache_status, in.cache_valid, in.cache_state, in.cache_forward, in.state_epoch, in.forward_epoch,
        in.level_offsets, in.level_bodies, in.nlevels);
  }
};
__device__ inline bool forward_reuse(const SubstepArguments &args, int world) {
  const auto &workspace = args.forward_workspace;
  const auto *cache = workspace.metadata + size_t(world) * ForwardCacheWords;
  return args.state.requested_mask[world] == 1 && args.begin.active[world] == 1 &&
                     !args.state.first_error[world] && !args.position.work[world] &&
                     cache[ForwardCacheValid] == 1 &&
                     cache[ForwardCacheState] == uint64_t(args.state.epoch[world]) &&
                     cache[ForwardCachePosition] == workspace.position_generation[world];
}
template<class Motion = NativeMotion>
__device__ inline void prepare_velocity(const SubstepArguments &args, int world,
                                       bool reuse, Motion motion = {}, bool prepared = false) {
  const auto &workspace = args.forward_workspace;
  // Velocity and passive forces consume the current, already prepared position.
  if (args.state.requested_mask[world] != 0) {
    if (threadIdx.x == 0) {
      const auto &in = args.velocity_gate;
      physics_velocity::d_gate(world, in.active, in.error, in.state, in.source, in.epoch, in.valid,
                               in.written, in.gate, in.require);
    }
    __syncthreads();
    if (!reuse) {
      {
        const auto &in = args.transmission;
        physics_velocity::d_transmission_velocity_block(
            world, in.moment, in.qvel, in.active, in.position_status, in.nv, in.nu, in.velocity, in.status);
      }
      // The collective transmission helper publishes its outputs on every exit.
      if (!prepared) {
        motion(args.body_velocity, world);
        // d_comvel_bias_cached owns its exit synchronization on every path.
        {
          const auto &in = args.joint_passive;
          physics_velocity::d_joint_passive_count(world, in.qpos, in.qvel, in.active, in.jtype, in.qadr, in.dadr,
                                                  in.stiffness, in.spring, in.damping, in.out, in.status, in.nv,
                                                  in.nq, in.njnt, in.joint_disjoint, in.joint_type_offsets,
                                                  in.joint_type_indices);
        }
        __syncthreads();
        {
          const auto &in = args.fluid_passive;
          physics_velocity::d_fluid_parallel(world, in, workspace.fluid);
        }
        __syncthreads();
      } // complete motion/passive values may come from the peeled producer
    }
    if (threadIdx.x == 0) {
      const auto &in = args.velocity_status;
      if (!reuse)
        physics_velocity::d_velocity_status(world, in.a, in.b, in.c, in.out);
      else
        // Post still consumes the immutable RNE forward payload. Its numerical
        // contents stay live; only the current-forward publication changes.
        args.body_velocity.cache_forward[world] = args.body_velocity.forward_epoch[world];
      const auto &finish = args.velocity_finish;
      physics_velocity::d_finish(world, finish.gate, finish.status, finish.epoch, finish.valid,
                                 finish.written, finish.error, finish.repair, finish.code);
    }
  }
}
