"""Boundary callbacks for the first exact raw-model diagnostic environment."""
from ..core.model_bytes import source_catalog
import numpy as np
import warp as wp
from .recovery_status import RecoveryBoundaryStatus


@wp.kernel(enable_backward=False)
def _model_commit(expected:wp.array(dtype=wp.int64), actual:wp.array(dtype=wp.int64),
                  ready:wp.array(dtype=wp.int32), forward_valid:wp.array(dtype=wp.int32),
                  step_valid:wp.array(dtype=wp.int32)):
    w=wp.tid()
    actual[w]=expected[w]
    ready[w]=1
    forward_valid[w]=0
    step_valid[w]=0


@wp.kernel(enable_backward=False)
def _written(mask:wp.array(dtype=wp.int32), actual:wp.array(dtype=wp.int64),
             expected:wp.array(dtype=wp.int64), state_valid:wp.array(dtype=wp.int32),
             forward_valid:wp.array(dtype=wp.int32), step_valid:wp.array(dtype=wp.int32)):
    w=wp.tid()
    if mask[w]==1:
        actual[w]=actual[w]+wp.int64(1)
        expected[w]=expected[w]+wp.int64(1)
        state_valid[w]=1
        forward_valid[w]=0
        step_valid[w]=0


def make_boundary_callbacks(bundle, built):
    """Return real resident callbacks; unsupported static mutations fail explicitly."""
    runtime, stream = bundle.runtime, bundle.stream
    reset_ctrl = bundle.reset_ctrl
    boundary_status = RecoveryBoundaryStatus(bundle)
    replacement_handler = [None]
    def require(world, current_stream):
        if world!=0 or current_stream is not stream or getattr(bundle,'closed',False):
            raise ValueError('foreign or closed raw runtime boundary')
    def synchronize_and_check():
        boundary_status.synchronize_and_check()
    def close_runtime():
        nonlocal runtime,reset_ctrl
        bundle.close()
        boundary_status.release()
        runtime=None;reset_ctrl=None
    def state_written(fields,world,current_stream):
        require(world,current_stream)
        if not set(fields).issubset({'qpos','qvel','time','ctrl','qacc_warmstart','qfrc_applied','xfrc_applied'}):
            raise ValueError('unsupported integration-state boundary write')
        wp.launch(_written,dim=1,inputs=[runtime.intent,runtime.state_epoch,bundle.expected_state,
            bundle.foundation.state_valid,runtime.forward.valid,runtime.valid],
            device=stream.device,stream=stream)
    def reset(world,current_stream):
        require(world,current_stream)
        synchronize_and_check()
        # Explicit native reset inputs, not dynamic CPU physics outputs.
        with wp.ScopedStream(current_stream):
            for key, array in [('qpos',runtime.euler.qpos),('qvel',runtime.euler.qvel),
                               ('time',runtime.euler.time),('qacc_warmstart',runtime.euler.qacc_warmstart),
                               ('qfrc_applied',bundle.dynamics.qfrc_applied),
                               ('xfrc_applied',bundle.dynamics.xfrc_applied)]:
                array.assign(bundle.initial_state[key])
            if bundle.reset_ctrl is not reset_ctrl:
                raise ValueError('retained reset ctrl root replaced')
            reset_ctrl.assign(bundle.initial_state['ctrl'])
            bundle.controller.initialize_ctrl_from_reset(runtime.intent,reset_ctrl,stream=stream)
            state_written({'qpos','qvel','time','ctrl','qacc_warmstart'},world,current_stream)
    def step1(world,current_stream):
        require(world,current_stream)
        runtime.forward.enqueue_step1(runtime.intent,stream)
    def update_parameters(changes,world,current_stream):
        require(world,current_stream)
        raise NotImplementedError('raw model parameter upload needs exact generation/readset transfer')
    def update_visual(changes,world,current_stream):
        require(world,current_stream)
        # The factory performs the one actual observer-owned GPU upload. This
        # callback validates the physics boundary only, without a second upload.
        if set(changes)-{'site_rgba','geom_rgba'}:
            raise ValueError('visual-only boundary contains a physics parameter')
    def bind_parameter_replacement(handler):
        if not callable(handler):raise TypeError('cold replacement handler must be callable')
        replacement_handler[0]=handler
    def update_parameters_transaction(changes,world,current_stream,token,model_generation):
        require(world,current_stream)
        if replacement_handler[0] is None:
            raise RuntimeError('task has not installed the atomic cold replacement handler')
        from .cold_model import prepare_parameter_model
        from .continuation import continuation_snapshot, restore_continuation
        from .recovery_final_event_factory import prepare_final_event_recovery_physics as prepare_shape_physics
        from .recovery_final_event_environment_factory import assemble_prepared
        from libero.gpu.host.runtime.core.build_lifetime import cold_build_scope
        @cold_build_scope(project_root=lambda: getattr(bundle, 'build_project_root', None))
        def build_new(old_components):
            nonlocal runtime,reset_ctrl
            # Host copies transport actual GPU bytes; no physics is evaluated.
            snapshot=continuation_snapshot(bundle)
            observations=old_components.observations.snapshot_continuation(stream=stream)
            preparation=prepare_parameter_model(bundle,built,changes,token,model_generation)
            initial={key:np.array(value,copy=True) for key,value in bundle.initial_state.items()}
            old_components.release()
            # Old boundary callbacks share these cells. They must not retain
            # the old physical owners during allocation of the fresh arena.
            runtime=None;reset_ctrl=None
            import gc
            gc.collect()
            fresh=None
            try:
                fresh=prepare_shape_physics(preparation.built,state=initial,device=stream.device,
                    stream=stream,mode='diagnostic',with_controller=True,
                    model_spec=preparation.spec,constraint_package=preparation.constraint_package)
                restore=restore_continuation(fresh,snapshot)
                # All fresh static owners have passed their actual native/source
                # binding validation. Publish the device generation only now.
                fresh.runtime.forward.validate()
                with wp.ScopedStream(stream):
                    fresh.expected_model_epoch.assign(np.full(1,token.next_generation,np.int64))
                    fresh.model_ready.assign(np.zeros(1,np.int32))
                wp.launch(_model_commit,dim=1,inputs=[fresh.expected_model_epoch,fresh.device_model_epoch,
                    fresh.model_ready,fresh.runtime.forward.valid,fresh.runtime.valid],
                    device=stream.device,stream=stream)
                wp.synchronize_stream(stream)
                if (fresh.model_ready.numpy()[0]!=1 or
                    fresh.device_model_epoch.numpy()[0]!=token.next_generation):
                    raise RuntimeError('actual cold model publication failed')
                fresh.continuation_evidence=restore
                components=assemble_prepared(preparation.built,fresh,observation_continuation=observations)
                return components,preparation.descriptor_plan
            except BaseException as error:
                # Preserve actual pre-release observations even when no fresh
                # arena reaches publication. Headers bind the full byte packet;
                # small clocks/status fields remain directly reviewable.
                small={key:value.tolist() for key,value in snapshot['data'].items()
                       if value.size<=16 and value.dtype.kind in 'iufb'}
                error.gpu_cold_failure_evidence=dict(
                    schema='gpu-cold-replacement-failure-v1',
                    phase='fresh_prepare_or_restore_before_external_publication',
                    actual_before_failure=True,restore_accepted=False,
                    source_model_sha256=snapshot['source_model_sha256'],
                    prepared_model_sha256=preparation.spec.model_sha256,
                    snapshot_headers=snapshot['headers'],small_actual_fields=small,
                    CPU_physics=False)
                close=getattr(fresh,"close",None)
                if callable(close):
                    try:close()
                    except BaseException as cleanup:
                        error.gpu_cleanup_errors=list(getattr(error,"gpu_cleanup_errors",()))+[str(cleanup)]
                raise
        return replacement_handler[0](build_new,token,model_generation)
    def evidence(world):
        require(world,stream)
        wp.synchronize_stream(stream)
        return dict(N_final_candidate_attempt=bundle.newton.final_event.candidate_attempt.numpy().tolist(),
            N_final_candidate_completed=bundle.newton.final_event.candidate_completed.numpy().tolist(),
            N_final_source_stage=bundle.newton.final_event.source_stage.numpy().tolist(),
            N_rejected_final_entry=bundle.newton.final_event.rejected_final_entry.numpy().tolist(),
            warning_number=runtime.recovery.warning_number.numpy().tolist(),
            warning_lastinfo=runtime.recovery.warning_lastinfo.numpy().tolist(),
            recovery_reset_events=runtime.reset_events.numpy().tolist(),
            recovery_reason=runtime.recovery.reason.numpy().tolist(),
            recovery_bad_index=runtime.recovery.bad_index.numpy().tolist(),
            mode='diagnostic',physics_backend='gpu',model_sha256=bundle.model_spec.model_sha256,
            model_identity=bundle.model_spec.identity_key,
            device_model_epoch=bundle.device_model_epoch.numpy().tolist(),
            independent_model_expected=bundle.expected_model_epoch.numpy().tolist(),
            device_model_ready=bundle.model_ready.numpy().tolist(),
            device_time=runtime.euler.time.numpy().tolist(),
            actual_integration_count=runtime.integration_count.numpy().tolist(),
            integration_count_scope='lifetime successful Euler commits; boundary state writes excluded',
            actual_native_timestep=float(built.native_model.opt.timestep),
            planned_substeps_per_action=runtime.substeps,
            no_CPU_hot_backfill=True,source_phase='real resident boundary and stage publications',
            production_numeric_acceptance=False,
            source_records=source_catalog(bundle.intake['kwargs']['source_records']),
            state_epoch=runtime.state_epoch.numpy().tolist(),
            independent_state_expected=bundle.expected_state.numpy().tolist(),
            actual_forward_epoch=bundle.dynamics.forward_epoch.numpy().tolist(),
            independent_forward_expected=runtime.forward.expected_forward.numpy().tolist(),
            independent_newton_expected=runtime.forward.expected_newton.numpy().tolist(),
            actual_newton_attempt=bundle.newton.n_attempt.numpy().tolist(),
            actual_newton_written=bundle.newton.n_written.numpy().tolist(),
            post_written=bundle.post.written.numpy().tolist(),
            state_status=runtime.valid.numpy().tolist(),
            forward_status=runtime.forward.valid.numpy().tolist(),
            first_error=runtime.first_error.numpy().tolist(),
            foundation_first_error=bundle.foundation.errors.numpy().tolist())
    return dict(reset=reset,step1=step1,state_written=state_written,
        synchronize_and_check=synchronize_and_check,close_runtime=close_runtime,
        update_parameters=update_parameters,update_visual=update_visual,evidence=evidence,
        bind_parameter_replacement=bind_parameter_replacement,
        update_parameters_transaction=update_parameters_transaction)
