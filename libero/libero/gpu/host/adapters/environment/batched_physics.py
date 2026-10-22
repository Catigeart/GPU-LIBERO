"""Explicit same-model native batch entry; no per-world owner dispatch.

Each bundle has one shared static model and W world-major dynamic states.
Different static models must be grouped into separate batches; matching NV alone
is insufficient. This is a physics driver, not a production task-factory grant.
"""
from dataclasses import dataclass
from contextlib import nullcontext
import hashlib
import json
import numpy as np


def stack_reset_states(states):
    """Cold/action-boundary data preparation from complete singleton states."""
    states = tuple(states)
    if not states:
        raise ValueError("nonempty reset states required")
    keys = set(states[0])
    result = {}
    for state in states:
        if set(state) != keys:
            raise ValueError("all worlds require the same complete reset fields")
    for key in keys:
        values = [np.asarray(state[key]) for state in states]
        if any(value.dtype != np.float64 or not value.ndim or value.shape[0] != 1 or
               value.shape != values[0].shape for value in values):
            raise ValueError("float64 singleton world-major reset field required: " + key)
        result[key] = np.ascontiguousarray(np.concatenate(values, axis=0))
    return result


def descriptor_key(built):
    """Exact static identity for dispatch grouping, not only shape compatibility."""
    metadata = built.metadata
    # Metadata contains source/decoder/name/topology/read-set identity; reject
    # unserializable metadata instead of replacing it with process-local IDs.
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def group_model_descriptors(built_models):
    """Stable groups of input indices sharing an exact static descriptor."""
    groups = {}
    for index, built in enumerate(built_models):
        groups.setdefault(descriptor_key(built), []).append(index)
    return tuple(tuple(indices) for indices in groups.values())


@dataclass
class NativeBatchedPhysics:
    """One enqueue sequence with native grid.y world ownership."""
    bundle: object
    debug: bool = False

    def __post_init__(self):
        self.nworld = self.bundle.runtime.euler.capacity
        if self.nworld < 1 or self.bundle.runtime.intent.shape != (self.nworld,):
            raise ValueError("actual native batch owner required")
        self._control_graph = None
        self._substep_fusion = None
        self._observer = None

    def bind_observer(self, observer):
        """Bind sampling policy before capture; None releases the retained borrow."""
        if observer is not None:
            if observer.nworld != self.nworld:
                raise ValueError('observer and physics world counts differ')
            if type(observer.graph_eligible) is not bool:
                raise TypeError('observer graph_eligible must be a bool property')
            if not callable(observer.notify_control_graph_replay):
                raise TypeError('observer graph replay notification required')
        self.discard_control_graph()
        self._observer = observer

    def discard_control_graph(self):
        """Release graph borrows before replacing observer or arena owners."""
        self._control_graph = None
        self._substep_fusion = None

    def prepare_fused_substep_graph(self, build_root, *, allow_compile=False,
                                    fast_compile=False, device_lto=False, profile_newton=False,
                                    atomic_reductions=None, defer_post=True, single_solve=False, staged=False):
        """Capture one readable fusion kernel per physical substep.

        Offline callers may permit compilation; ordinary execution loads only
        completed objects. The selected graph retains all argument owners and
        is invalidated by the same observer/arena changes as the ordinary graph.
        """
        if self.debug or not self._graph_eligible():
            raise ValueError('substep fusion requires device-only sampling')
        from libero.gpu.host.runtime.step.fused_substep import FusedSubstep
        if type(staged) is not bool:
            raise TypeError('staged must be a bool')
        if staged:
            from libero.gpu.host.runtime.step.staged_substep import StagedSubstep
            FusedSubstep = StagedSubstep
        import warp as wp
        fusion = FusedSubstep(self.bundle.runtime, build_root,
                              allow_build=allow_compile,
                              fast_compile=fast_compile, device_lto=device_lto,
                              profile_newton=profile_newton, atomic_reductions=atomic_reductions,
                              single_solve=single_solve)
        # Only the built-in device sampler has a proved read set (poses and
        # velocities, never intermediate cfrc/sensordata). Unknown observers
        # retain complete Post even if they permit graph capture.
        from libero.gpu.host.adapters.observations.adapter import GpuObservationAdapter
        defer_post = bool(defer_post and type(self._observer) is GpuObservationAdapter
                          and self._observer.graph_eligible)
        with wp.ScopedCapture(stream=self.bundle.stream) as capture:
            for substep in range(self.bundle.runtime.substeps):
                fusion.enqueue_substep(policy_step=(substep == 0),
                    publish_post=not defer_post or substep == self.bundle.runtime.substeps-1)
        self._control_graph = capture.graph
        self._substep_fusion = fusion
        return fusion

    def _graph_eligible(self):
        if self._observer is None:
            return True
        eligible = self._observer.graph_eligible
        if type(eligible) is not bool:
            raise TypeError('observer graph_eligible must be a bool property')
        return eligible

    def stage_action(self, action):
        if self.bundle.controller is None:
            raise ValueError("controller was not prepared")
        if self.debug:
            self.bundle.controller.stage_action(action)
        else:
            import warp as wp
            with wp.ScopedStream(self.bundle.stream):
                self.bundle.controller.controller.action.assign(action)

    def prepare_control_graph(self):
        """Capture device-only sampling; host sampling keeps the substep loop."""
        if self.debug or not self._graph_eligible():
            self.discard_control_graph()
            return
        if self._control_graph is not None:
            return
        import warp as wp
        # Warp selects cache-backed lazy module loading on modern drivers and
        # preloads on older drivers. The compilation guard remains active.
        with wp.ScopedCapture(stream=self.bundle.stream) as capture:
            for substep in range(self.bundle.runtime.substeps):
                self.enqueue_substep(policy_step=(substep == 0))
        self._control_graph = capture.graph

    def prepare_substep_modules(self):
        """Offline-only recording of both controller branches, without replay.

        Ordinary initialize=False construction leaves capture to the first
        complete control graph. The offline builder calls this explicitly to
        discover/prebuild reachable modules without unrolling all substeps.
        """
        if self.debug or not self._graph_eligible():
            return
        import warp as wp
        with wp.ScopedCapture(stream=self.bundle.stream):
            self.enqueue_substep(policy_step=True)
            self.enqueue_substep(policy_step=False)

    def enqueue_substep(self, *, policy_step=False, force_refresh=False):
        # There is deliberately no loop over worlds here. All functional plans
        # consume the same W-major arena and one mask entry per world.
        self.bundle.runtime.enqueue_substep(policy_step=policy_step,
                                           force_refresh=force_refresh)

    def enqueue_action(self, action):
        if self.debug or not self._graph_eligible():
            self.discard_control_graph()
        self.stage_action(action)
        if not self.debug:
            self.prepare_control_graph()
            if self._control_graph is not None:
                import warp as wp
                wp.capture_launch(self._control_graph, stream=self.bundle.stream)
                if self._observer is not None:
                    self._observer.notify_control_graph_replay(
                        timestep=None, substeps=self.bundle.runtime.substeps,
                        stream=self.bundle.stream)
                return
        for substep in range(self.bundle.runtime.substeps):
            self.enqueue_substep(policy_step=(substep == 0))

    def download_state(self):
        import warp as wp
        wp.synchronize_stream(self.bundle.stream)
        e = self.bundle.runtime.euler
        return {name: getattr(e, name).numpy() for name in
                ("qpos", "qvel", "time", "ctrl", "qacc", "qacc_warmstart")}

    def close(self):
        self.bind_observer(None)
        self.bundle.close()


from libero.gpu.host.runtime.core.build_lifetime import cold_build_scope


@cold_build_scope
def prepare_native_batch(built, *, states, strict_build_receipts=False, build_root=None, debug=False, **kwargs):
    from libero.gpu.host.build.startup import startup_phase
    from libero.gpu.host.factory import prepare_cuda_physics
    from libero.gpu.host.runtime.step.recovery_final_event_environment_factory import (
        actual_decoder_record, _native_intake_package)
    from libero.gpu.host.runtime.core.product_paths import (
        product_build, mathdx_root, toolchain_scope, cache_root, build_project)
    from libero.gpu.host.build.native import strict_build_receipts_scope
    from libero.gpu.host.build.product_cache import product_cache_scope
    states = tuple(states)
    if type(strict_build_receipts) is not bool:
        raise ValueError('strict_build_receipts must be bool')
    state = stack_reset_states(states)
    W = len(states)
    if "nworld" in kwargs or "state" in kwargs:
        raise ValueError("batch states determine nworld and state")
    if ('model_spec' in kwargs) != ('constraint_package' in kwargs):
        raise ValueError('model_spec and constraint_package must be supplied together')
    # Shared native model parameters are fixed for this bundle's lifetime.
    # Native model replacement constructs fresh device owners and cold tables.
    kwargs.setdefault('cache_collision_parameters', True)
    with build_project(build_root, lock_key=built.metadata['model_sha256']) as project:
        with startup_phase('decoder_intake'):
            built.metadata['decoder_record'] = actual_decoder_record(built)
            if 'model_spec' not in kwargs:
                kwargs['model_spec'], kwargs['constraint_package'] = _native_intake_package(built)
        with product_build():
            sdk = kwargs.pop('mathdx_root', None)
            sdk = mathdx_root() if sdk is None else sdk
            import warp as wp
            architecture = wp.get_device(kwargs.get('device', 'cuda:0')).arch
            kwargs.setdefault('cuda_build_root', project['root'] / 'cuda' / ('sm' + str(architecture)))
            cache = product_cache_scope(cache_root() / 'products') if strict_build_receipts else nullcontext()
            with toolchain_scope(sdk), strict_build_receipts_scope(strict_build_receipts), cache:
                bundle = prepare_cuda_physics(built, state=state, nworld=W, mathdx_root=sdk, debug=debug, **kwargs)
        bundle.build_report = dict(directory=str(project['root']), events=list(project['events']))
        bundle.observation_build_options = dict(build_root=str(project['root']),
            mathdx_root=str(sdk), strict_build_receipts=strict_build_receipts)
    try:
        return NativeBatchedPhysics(bundle, debug=debug)
    except BaseException:
        bundle.close()
        raise
