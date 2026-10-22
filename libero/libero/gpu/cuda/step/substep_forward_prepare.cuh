#pragma once
// Included in flash_libero's private device namespace; numerical leaves are
// shared with the retained fused entry. No complete kernel source inclusion.
// Collective preparation for the current position and velocity. No actuation,
// Newton, warmstart update, or forward-complete publication occurs here.
// The returned flag describes same-state numerical reuse for the FP64 path.
__device__ __forceinline__ bool prepare_dynamics(const SubstepArguments &args, int world) {
  const auto &workspace = args.forward_workspace;
  uint64_t *cache = workspace.metadata + size_t(world) * ForwardCacheWords;
  // The Start entry clears validity. Within this physical substep only the
  // controller (ctrl) and Newton/Post (qacc/history) run between forwards.
  // Model/mocap are immutable in this native owner. A reset changes state;
  // a position rebuild changes generation even when state itself is unchanged.
  const bool reuse = args.state.requested_mask[world] == 1 && args.begin.active[world] == 1 &&
                     !args.state.first_error[world] && !args.position.work[world] &&
                     cache[ForwardCacheValid] == 1 &&
                     cache[ForwardCacheState] == uint64_t(args.state.epoch[world]) &&
                     cache[ForwardCachePosition] == workspace.position_generation[world];
  CF_STEP_PROFILE_STAGES(reuse ? CF_STEP_PROFILE_PHASE(VelocityReuse) : CF_STEP_PROFILE_PHASE(VelocityCold));
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
      {
        const auto &in = args.body_velocity;
        physics_velocity::d_comvel_bias_cached(
            world, in.active, in.cdof, in.qvel, in.bp, in.ba, in.bn, in.dj, in.jt, in.nb, in.nv, in.cvel,
            in.cdot, in.status, in.cinert, in.gravity, in.disabled, in.cache_acc, in.cache_force,
            in.cache_status, in.cache_valid, in.cache_state, in.cache_forward, in.state_epoch, in.forward_epoch,
            in.level_offsets, in.level_bodies, in.nlevels);
      }
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
  // Constraint velocity/reference acceleration.
  CF_STEP_PROFILE_NEXT(reuse ? CF_STEP_PROFILE_PHASE(ReferenceReuse) : CF_STEP_PROFILE_PHASE(ReferenceCold));
  if (threadIdx.x == 0) {
    const auto &in = args.reference_admission;
    forward_cuda::reference_mask(world, in.valid, in.written, in.expected, in.first, in.active);
  }
  __syncthreads();
  if (!reuse) {
    {
      const auto &in = args.reference_gate;
      physics_constraint_reference::cf_s::reference_gate(world, in.active, in.status, in.committed, in.counts,
                                                         in.qvel, in.epoch, in.expected, in.ready, in.rows,
                                                         in.nv);
    }
    __syncthreads();
    {
      const auto &in = args.constraint_reference;
      physics_constraint_reference::cf_s::reference(world, in.committed, in.counts, in.efc_J, in.kbip,
                                                    in.efc_pos, in.efc_margin, in.qvel, in.qvel_epoch, in.vel,
                                                    in.aref, in.ref_epoch, in.rows, in.nv);
    }
    // reference() owns the publication of every numerical tile; no additional
    // barrier is needed before commit_reference consumes those same rows.
    {
      const auto &in = args.reference_commit;
      physics_constraint_reference::cf_s::commit_reference(world, in.ready, in.status, in.committed, in.counts,
                                                           in.sv, in.sa, in.epoch, in.pv, in.pa, in.ref_epoch,
                                                           in.rows);
    }
  } else if (threadIdx.x == 0) {
    // S-position and qvel have not changed; keep vel/aref in their retained
    // global owners and republish the new forward epoch without copying rows.
    // A masked controller forward still clears reference_ready in its gate.
    // Readiness is publication, not numerical validity: restore it on a hit.
    args.reference_commit.ready[world] = 1;
    args.reference_commit.ref_epoch[world] = args.reference_commit.epoch[world];
  }
  __syncthreads();
  // Bias, actuation, and unconstrained acceleration.
  CF_STEP_PROFILE_NEXT(reuse ? CF_STEP_PROFILE_PHASE(BiasReuse) : CF_STEP_PROFILE_PHASE(BiasCold));
  if (args.state.requested_mask[world] != 0) {
    if (threadIdx.x == 0) {
      const auto &in = args.bias_gate;
      physics_bias::d_gate(world, in.active, in.error, in.state, in.source, in.epoch, in.valid, in.written,
                           in.gate, in.require);
      const auto &reference = args.bias_reference;
      physics_bias::d_reference_gate(world, reference.done, reference.err, reference.written,
                                     reference.expected, reference.epoch, reference.gate, reference.first);
    }
    __syncthreads();
    if (!reuse) {
      {
        const auto &in = args.bias_forces;
        physics_bias::d_rne_bias_cached(world, in.active, in.cdot, in.qvel, in.cinert, in.cvel, in.cdof, in.bp,
                                        in.ba, in.bn, in.db, in.gravity, in.disabled, in.nb, in.nv, in.cacc,
                                        in.frc, in.result, in.status, in.cache_acc, in.cache_force,
                                        in.cache_status, in.cache_valid, in.cache_state, in.cache_forward,
                                        in.state_epoch, in.forward_epoch, in.level_offsets, in.level_bodies,
                                        in.nlevels, in.children_offsets, in.children_indices);
      }
      // Normal RNE exits publish their outputs. On an exceptional fallback,
      // status stays leader-owned and commit's entry barrier closes all writes.
      {
        const auto &in = args.bias_commit;
        physics_bias::d_commit_bias_block(world, in.active, in.gate, in.status, in.epoch, in.error, in.work,
                                          in.pub, in.valid, in.written, in.width);
      }
    } else if (threadIdx.x == 0) {
      const auto &in = args.bias_commit;
      physics_velocity::d_finish(world, in.gate, in.status, in.epoch, in.valid, in.written,
                                 in.error, 0, 211);
    }
    if (threadIdx.x == 0) {
      if (reuse)
        ++cache[ForwardCacheReused];
      else {
        ++cache[ForwardCacheComputed];
        if (cache[ForwardCacheValid])
          ++cache[ForwardCacheInvalidated];
        const auto &velocity = args.velocity_finish;
        const auto &bias = args.bias_commit;
        const auto &reference = args.reference_commit;
        cache[ForwardCacheValid] = velocity.valid[world] == 1 && !velocity.error[world] &&
                                  velocity.written[world] == velocity.epoch[world] &&
                                  bias.valid[world] == 1 && !bias.error[world] &&
                                  bias.written[world] == bias.epoch[world] &&
                                  reference.ready[world] == 1 && !reference.status[world] &&
                                  reference.committed[world] == 1 &&
                                  reference.ref_epoch[world] == reference.epoch[world] &&
                                  args.body_velocity.cache_valid[world] == 1 &&
                                  !args.body_velocity.cache_status[world];
        cache[ForwardCacheState] = uint64_t(args.state.epoch[world]);
        cache[ForwardCachePosition] = workspace.position_generation[world];
      }
    }
  }
  return reuse;
}

__device__ __forceinline__ void prepare_solver_inputs(const SubstepArguments &args,
    int world, unsigned char *shared, bool reuse, int smooth_reuse_status
#if defined(FLASH_SUBSTEP_PROFILE)
    , substep_profile::Stages &step_profile_stages
#endif
) {
  const auto &workspace = args.forward_workspace;
  if (args.state.requested_mask[world] != 0) {
    if (threadIdx.x == 0) {
      const auto &in = args.actuation_gate;
      physics_force::d_gate(world, in.active, in.error, in.state, in.source, in.epoch, in.valid, in.written,
                            in.gate, in.require);
      const auto &control = args.control_gate;
      physics_force::d_force_gate(world, control.valid, control.gate);
    }
    __syncthreads();
    {
      const auto &in = args.control_copy;
      physics_force::d_copy_admitted_block(world, in.gate, in.source, in.target, in.width);
    }
    __syncthreads();
    {
      const auto &in = args.actuation;
      physics_force::d_actuation_count(
          world, in.length, in.moment, in.velocity, in.ctrl, in.active, in.vel_status, in.gain, in.bias0,
          in.bias1, in.bias2, in.bias_type, in.ctrl_lo, in.ctrl_hi, in.force_lo, in.force_hi,
          in.force_limited, in.disable_flags, in.force, in.force_pre, in.ctrl_used, in.ctrl_route,
          in.force_route, in.qfrc_actuator, in.status, in.warn_badctrl_count, in.warn_badctrl_lastinfo,
          in.warning_number, in.warning_lastinfo, in.nv, in.nu);
    }
    // d_actuation_count ends with a collective publication barrier. Preserve
    // that barrier once; the next operations are owned by the same leader.
    if (threadIdx.x == 0) {
      const auto &in = args.actuation_finish;
      physics_force::d_actuation_finish(world, in.gate, in.status, in.first);
      const auto &smooth = args.smooth_gate;
      physics_force::d_gate(world, smooth.active, smooth.error, smooth.state, smooth.source, smooth.epoch,
                            smooth.valid, smooth.written, smooth.gate, smooth.require);
      const auto &force = args.smooth_force_gate;
      physics_force::d_force_gate(world, force.valid, force.gate);
    }
    __syncthreads();
    {
      CF_STEP_PROFILE_NEXT(CF_STEP_PROFILE_PHASE(Smooth));
      const auto &in = args.smooth_acceleration;
      if (reuse && smooth_reuse_status >= 0 && in.active[world] == 1 &&
          args.smooth_gate.gate[world] == 1) {
        real_t *force = flash_libero::shared_at<real_t>(shared, flash_libero::SmoothLayout::force);
        real_t *acc = flash_libero::shared_at<real_t>(shared, flash_libero::SmoothLayout::acceleration);
        for (int i = threadIdx.x; i < in.nv; i += blockDim.x) {
          force[i] = args.smooth_commit.pf[size_t(world) * in.nv + i];
          acc[i] = args.smooth_commit.pa[size_t(world) * in.nv + i];
        }
        if (threadIdx.x == 0) in.status[world] = smooth_reuse_status;
      } else {
        physics_force::d_smooth_chain(
            world, in.passive, in.bias, in.applied, in.actuator, in.active, in.entry_status, in.smooth, in.seed,
            in.nv, in.xfrc, in.xipos, in.subtree_com, in.cdof, in.body_ids, in.body_count, in.body_rootid,
            in.ancestor, in.scratch, in.stages, in.qfrc, in.d02_status, in.nbody, in.qm, in.count, in.madr,
            in.parent, in.ld, in.inv, in.sqrt_inv, in.solution, in.pivot, in.solve_status, in.warning_count,
            in.warning_last, in.nm, in.rhs, in.qacc_smooth, in.status, in.source_ld, in.source_inv,
            in.source_sqrt_inv, in.source_pivot, in.source_status, in.source_warning_count,
            in.source_warning_last, in.source_active, in.position_valid, in.position_written, in.forward_epoch,
            in.factor_reuse_count, in.body_dof_offsets, in.body_dof_indices, workspace.mass_plan);
      }
    }
    __syncthreads();
    {
      const auto &in = args.smooth_commit;
      physics_force::d_commit_smooth_shared_block(world, in.active, in.gate, in.status, in.epoch, in.error,
                                                  in.pf, in.pa, in.valid, in.written, in.repaired, in.width);
    }
    __syncthreads();
  }
  // Assemble the existing S-to-Newton inputs; keep every solve and warmstart write.
  CF_STEP_PROFILE_NEXT(CF_STEP_PROFILE_PHASE(NewtonBridge));
  if (threadIdx.x == 0) {
    const auto &in = args.newton_admission;
    physics_constraint_projection::cf_s::n_mask(world, in.active, in.first, in.smooth, in.reference,
                                                in.sstatus, in.admitted, in.error);
    const auto &counts = args.constraint_counts;
    physics_constraint_projection::cf_s::project_counts(
        world, counts.admitted, counts.source_counts, counts.source_ncon, counts.nefc, counts.ne, counts.nf,
        counts.ncon, counts.count, counts.empty, counts.newton, counts.active, counts.status, counts.item,
        counts.rows, counts.contacts, counts.nv);
  }
  __syncthreads();
  {
    const auto &in = args.cone_parameters;
    physics_constraint_projection::cf_s::project_cone_heads_verified(
        world, in.admitted, in.nefc, in.ne, in.nf, in.ncon, in.typ, in.id, in.dim, in.mu, in.friction,
        in.out_dim, in.out_mu, in.out_friction, in.status, in.item, in.rows, in.contacts, in.compact_heads,
        in.compact_count, in.compact_valid, in.head_dim, in.head_mu, in.head_friction, in.head_cone_valid);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const auto &in = args.newton_epoch;
    physics_constraint_projection::cf_s::increment_plan(world, in.mask, in.first, in.epoch);
    // Three scalar channels per world, qualified by bind(). Their producing
    // epoch increment and consuming copies have the same thread owner.
    for (const auto &copy : args.post_epochs)
      copy.destination[world] = copy.source[world];
  }

  // Post observes current contact addresses and the three input epochs.
  const auto &contacts = args.contact_addresses;
  const int count = contacts.count[world];
  if (contacts.active[world] == 1 && contacts.status[world] == 0 && contacts.committed[world] == 1 &&
      count >= 0 && count <= contacts.capacity) {
    for (int contact = threadIdx.x; contact < count; contact += blockDim.x) {
      const int at = world * contacts.capacity + contact;
      contacts.current[at] = contacts.excluded[at] ? -1 : contacts.address[at];
    }
  }
  // Contact addresses and the leader's scalar snapshots are independent;
  // one barrier publishes both before Newton/Post consume them.
  __syncthreads();
}
