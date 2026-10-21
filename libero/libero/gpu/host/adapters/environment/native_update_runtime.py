"""Actual cold native-model replacement wiring for the stable public GPU facade.

Native preflight leaves the old arena live. Device replacement copies actual
continuation bytes, releases the old owner before allocation, restores without
physics, and publishes a fresh source/model generation. No acceptance is aliased.
"""
from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import numpy as np

from libero.gpu.host.runtime.core.contracts import ModelGeneration
from libero.gpu.host.runtime.core.native_model_update import prepare_native_model_update
from libero.gpu.host.runtime.core.build_lifetime import cold_build_scope


def _snapshot(bundle):
    from libero.gpu.host.runtime.step.continuation import continuation_snapshot
    return continuation_snapshot(bundle)


def _restore(bundle, snapshot, plan):
    from libero.gpu.host.runtime.step.continuation import restore_native_model_continuation
    return restore_native_model_continuation(bundle, snapshot, plan)


def _initial_state(model):
    from libero.gpu.host.runtime.step.shape_factory import default_integration_state
    return default_integration_state(model)


def commit_native_model_epoch(bundle, generation):
    """Publish the actual new device owner and invalidate old derived freshness."""
    import warp as wp
    from libero.gpu.host.runtime.step.recovery_final_event_environment import _model_commit
    bundle.runtime.forward.validate()
    expected = bundle.expected_model_epoch
    worlds = expected.shape[0]
    with wp.ScopedStream(bundle.stream):
        expected.assign(np.full(worlds, generation, np.int64))
        bundle.model_ready.assign(np.zeros(worlds, np.int32))
    wp.launch(_model_commit, dim=worlds, inputs=[expected, bundle.device_model_epoch,
        bundle.model_ready, bundle.runtime.forward.valid, bundle.runtime.valid],
        device=bundle.stream.device, stream=bundle.stream)
    wp.synchronize_stream(bundle.stream)
    if (not np.array_equal(bundle.model_ready.numpy(), np.ones(worlds, np.int32)) or
            not np.array_equal(bundle.device_model_epoch.numpy(), np.full(worlds, generation, np.int64))):
        raise RuntimeError('actual native replacement device model publication failed')
    if np.any(bundle.runtime.first_error.numpy()):
        raise RuntimeError('actual native replacement retains a device error')
    return {'schema': 'native-model-device-epoch-publication-v1',
            'generation': generation, 'actual_ready_epoch_checked': True,
            'old_forward_step_freshness_invalidated': True}


@dataclass(frozen=True)
class NativeUpdateOperations:
    """Internal call boundaries permit CPU call-order tests; defaults are actual."""
    snapshot: object = _snapshot
    restore: object = _restore
    commit: object = commit_native_model_epoch
    initial_state: object = _initial_state
    prepare_plan: object = prepare_native_model_update


def _failure_packet(snapshot, plan, phase, *, old_released):
    if snapshot is None:
        return {'schema': 'native-model-replacement-failure-v1', 'phase': phase,
                'old_arena_released': old_released, 'actual_snapshot_captured': False,
                'CPU_physics': False}
    small = {name: value.tolist() for name, value in snapshot['data'].items()
             if value.size <= 16 and value.dtype.kind in 'iufb'}
    return {'schema': 'native-model-replacement-failure-v1', 'phase': phase,
            'old_arena_released': old_released, 'actual_snapshot_captured': True,
            'source_model_sha256': snapshot['source_model_sha256'],
            'new_model_sha256': plan.spec.model_sha256,
            'snapshot_headers': snapshot['headers'], 'small_actual_fields': small,
            'restored_owner_published': False, 'CPU_physics': False}


def bind_native_update_runtime(components, bundle, *, prepare, assemble,
                               cache_dir=None, _operations=None):
    """Install the native-write binder before the real task object is available.

    ``prepare`` receives the actual built/state/device/stream/ModelSpec/S package;
    ``assemble`` binds its actual observer/physics owners. Product entry supplies
    these current factories, and each fresh assembly is bound recursively here.
    Task _install_shell calls the installed binder with replace_native_components.
    """
    if not callable(prepare) or not callable(assemble):
        raise TypeError('actual native replacement prepare/assemble factories required')
    ops = NativeUpdateOperations() if _operations is None else _operations
    for name in ('snapshot', 'restore', 'commit', 'initial_state', 'prepare_plan'):
        if not callable(getattr(ops, name, None)):
            raise TypeError('native replacement operation must be callable: ' + name)
    if bundle.model_pack.model is not components.built.native_model:
        raise ValueError('native update binding model owner differs')
    holder = [bundle]
    handler = [None]
    bound_sim = [components.sim]

    def install(task_replace):
        if not callable(task_replace):
            raise TypeError('actual task native replacement handler required')
        handler[0] = task_replace
        current = components.sim
        bound_sim[0] = current
        current.bind_native_model_updates(update, components.built.metadata['native_write_schema'])

    @cold_build_scope(project_root=lambda: getattr(holder[0], 'build_project_root', None))
    def update(array_changes, option_changes):
        if handler[0] is None:
            raise RuntimeError('native model writes require the installed task owner')
        stable = bound_sim[0]
        current_bundle = holder[0]
        if (current_bundle is None or getattr(current_bundle, 'closed', False) or
                components.released or stable is not components.sim):
            raise RuntimeError('native model update belongs to a closed or foreign owner')
        old_generation = stable._model_generation
        plan = ops.prepare_plan(components.built, array_changes, option_changes,
                                cache_dir=cache_dir, source_generation=old_generation.generation)
        # This entire native validation occurs before asking the task to replace
        # anything, so illegal/unsupported parameters keep the old owner live.
        plan.validate_base()
        next_generation = old_generation.generation + 1
        packet = [None]
        phase = ['snapshot_before_release']
        fresh_components = [None]
        prepared = [None]
        release_attempted = [False]

        def build_new(old):
            if old is not components or old.sim is not stable:
                raise ValueError('foreign task owner requested native replacement')
            old_bundle = holder[0]
            stream = old_bundle.stream
            snapshot = ops.snapshot(old_bundle)
            packet[0] = snapshot
            observations = old.observations.snapshot_continuation(stream=stream)
            plan.validate_base()
            initial = ops.initial_state(plan.built.native_model)
            phase[0] = 'release_old_before_allocation'
            release_attempted[0] = True
            old.release()
            if not old.released:
                raise RuntimeError('old arena did not finish release')
            holder[0] = None
            old_bundle = None
            gc.collect()
            phase[0] = 'prepare_new_actual_native_owner'
            fresh_bundle = prepare(plan.built, state=initial, device=stream.device,
                stream=stream, mode='diagnostic', with_controller=True,
                model_spec=plan.spec, constraint_package=plan.constraint_package)
            prepared[0] = fresh_bundle
            if fresh_bundle.model_pack.model is not plan.built.native_model:
                raise ValueError('prepared native replacement uses another actual model owner')
            phase[0] = 'restore_actual_dynamic_continuation'
            restore = ops.restore(fresh_bundle, snapshot, plan)
            phase[0] = 'publish_actual_model_epoch'
            publication = ops.commit(fresh_bundle, next_generation)
            fresh_bundle.continuation_evidence = restore
            fresh_bundle.native_model_publication = publication
            phase[0] = 'assemble_actual_observer_and_facade'
            fresh = assemble(plan.built, fresh_bundle, observation_continuation=observations)
            fresh_components[0] = fresh
            fresh.sim._model_generation = ModelGeneration(plan.spec, generation=next_generation)
            if fresh.sim._world_state is not None:
                fresh.sim._world_state.model = fresh.sim._model_generation
            bind_native_update_runtime(fresh, fresh_bundle, prepare=prepare, assemble=assemble,
                                       cache_dir=cache_dir, _operations=_operations)
            fresh.validate()
            phase[0] = 'task_stable_facade_adoption'
            return fresh

        try:
            handler[0](build_new, stable)
            if not components.released or stable._closed:
                raise RuntimeError('task did not publish the actual restored stable facade')
            if (stable._model_generation.spec.identity_key != plan.spec.identity_key or
                    stable._model_generation.generation != next_generation):
                raise RuntimeError('published native model descriptor/generation differs')
            phase[0] = 'commit_native_owner_binding'
            plan.mark_committed()
            stable.native_model_update_evidence = {
                'schema': 'native-model-runtime-replacement-v1',
                'source_model_identity': plan.base_spec.identity_key,
                'new_model_identity': plan.spec.identity_key,
                'generation': next_generation, 'same_public_sim': True,
                'actual_dynamic_restore': True, 'CPU_physics': False,
                'generation_bridge_granted': False}
        except BaseException as error:
            error.gpu_cold_failure_evidence = _failure_packet(packet[0], plan, phase[0],
                                                            old_released=release_attempted[0])
            # Retain actual before-packet even after cleanup drops all GPU roots.
            if packet[0] is not None:
                error.gpu_native_before_packet = packet[0]
            if plan.state == 'prepared':
                try: plan.discard('native replacement failed: ' + phase[0])
                except BaseException as cleanup:
                    error.gpu_cleanup_errors = list(getattr(error, 'gpu_cleanup_errors', ())) + [repr(cleanup)]
            if release_attempted[0]:
                stable._quarantine('native_model_replacement')
                fresh = fresh_components[0]
                cleanup = fresh.release if fresh is not None else getattr(prepared[0], 'close', None)
                if callable(cleanup):
                    try: cleanup()
                    except BaseException as cleanup_error:
                        error.gpu_cleanup_errors = list(getattr(error, 'gpu_cleanup_errors', ())) + [repr(cleanup_error)]
            raise

    components.native_model_replacement_binder = install
    return components
