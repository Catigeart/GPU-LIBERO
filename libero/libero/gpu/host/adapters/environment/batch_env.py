"""Same-task LIBERO environments over one native world-major GPU physics arena.

Fixtures are sampled once at construction and shared by every world. Reset uses
the task's official initialization bank (or a supplied bank); it never constructs
singleton physics environments. Goal logic/render packaging runs on the host at
action boundaries. All worlds' physics advances in one native batch.
"""
import libero.gpu.precision as scalar
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from threading import RLock
from pathlib import Path

import numpy as np
from libero.gpu.host.build.compilation import construction_scope, environment_scope
from libero.gpu.host.build.startup import profile_construction, startup_phase


def _state_bank(values, nq, nv):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] < 1 or result.shape[1] != 1+nq+nv:
        raise ValueError(f'init_states must have shape (M, {1+nq+nv}), M >= 1')
    if not np.isfinite(result).all():
        raise ValueError('initial states must be finite')
    return np.array(result, copy=True, order='C')


def _clone_shell(shell):
    # The model-only sim deliberately raises on every attribute access,
    # including deepcopy probes. Bypass the public __new__ routing as well:
    # reconstructing it with deepcopy would allocate a single-world GPU proxy.
    clone = object.__new__(type(shell))
    memo = {id(shell): clone, id(shell.sim): shell.sim}
    clone.__dict__.update(deepcopy(shell.__dict__, memo))
    return clone


def _prepare_shared_fixtures(shell, model, seed):
    """Original host placement sampling, before static model intake/compilation."""
    import mujoco
    from .task_driver import _PLACEMENT_LOCK
    from robosuite.utils.errors import RandomizationError
    with _PLACEMENT_LOCK:
        previous = np.random.get_state()
        np.random.seed(seed)
        try:
            for attempt in range(100):
                try:
                    placements = shell.placement_initializer.sample()
                    break
                except RandomizationError:
                    if attempt == 99:
                        raise
        finally:
            np.random.set_state(previous)
    for position, quaternion, obj in placements.values():
        if obj.name in shell.fixtures_dict:
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj.root_body)
            if body < 0:
                raise KeyError(obj.root_body)
            model.body_pos[body] = position
            model.body_quat[body] = quaternion


class _WorldBoundary:
    """Read-only task/predicate view into the owner's shared boundary snapshot."""
    physics_backend = 'gpu'

    def __init__(self, owner, world):
        self.owner, self.world = owner, world

    def read_fields(self, names):
        owner = self.owner
        owner._open()
        names = tuple(names)
        snapshot = owner._boundary_snapshot(names)
        return {name: snapshot[name][self.world] for name in names}

    def update_visual_parameters(self, changes):
        owner = self.owner
        owner._open()
        owner.observer.update_visual_parameters(changes, world=self.world, stream=owner.stream)

    def _unsupported(self, *args, **kwargs):
        from .facade import UnsupportedGpuOperation
        raise UnsupportedGpuOperation('mutate/advance the shared arena through GpuBatchEnv')

    write_fields = forward = step = _unsupported

    def close(self):
        self.owner = None


class GpuBatchEnv:
    """Explicit synchronous batch API, with leading environment axis.

    ``step(actions)`` advances all worlds together and returns
    ``(dict[str, ndarray], rewards, successes, infos)``. There is no auto-reset.
    ``reset(id=...)`` and ``set_init_state(..., id=...)`` can select worlds.
    Flat initialization states follow LIBERO's [time, qpos, qvel] convention.
    ``build_root`` selects the persistent compiler cache. ``initialize=False``
    prepares owners without capturing the control graph or resetting; call
    ``reset()`` before stepping. The first step prepares the graph if needed.
    ``ready_session`` explicitly reuses one fully qualified, immutable build in
    this process. Create it with host.build.aot.qualify_ready_session(); the
    default independent construction retains full artifact hash validation.
    Precision defaults to FP32. ``physics_engine='auto'`` selects independent
    CUDA for FP32 and native CUDA for explicit FP64. The explicit ``mjwarp``
    engine remains an optional reference.
    Explicit ``physics_engine='native'`` retains the handwritten FP32 backend.
    """
    physics_backend = 'gpu'
    is_async = False

    def __new__(cls, *args, physics_engine='auto', **kwargs):
        from libero.gpu.host.backends.selection import resolve_engine
        physics_engine = resolve_engine(physics_engine, kwargs.get('precision'))
        if cls is GpuBatchEnv and physics_engine == 'cuda_fp32':
            scalar.select(kwargs.get('precision') or 'fp32')
            from libero.gpu.host.backends.cuda_fp32.environment import CudaFP32BatchEnv
            return object.__new__(CudaFP32BatchEnv)
        if cls is GpuBatchEnv and physics_engine == 'mjwarp':
            scalar.select(kwargs.get('precision') or 'fp32')
            try:
                from libero.gpu.host.backends.mjwarp.environment import MjWarpBatchEnv
            except ModuleNotFoundError as error:
                if error.name not in ('mujoco_warp', 'warp'):
                    raise
                raise ImportError('The explicit MJWarp reference requires its isolated '
                                  'environment and requirements-mjwarp.txt. Default FP32 '
                                  'uses physics_engine="cuda_fp32" without Warp or MJWarp.') from error
            return object.__new__(MjWarpBatchEnv)
        return object.__new__(cls)

    @construction_scope
    @profile_construction
    def __init__(self, bddl_file_name, num_envs=8, *, init_states=None,
                 seed=0, device='cuda:0', memory_budget=None,
                 strict_build_receipts=False, build_root=None, initialize=True,
                 allow_compile=False, debug=False, ready_session=None,
                 profile_startup=False, algorithm_options=None, precision=None,
                 physics_engine='auto', **environment_options):
        self.physics_engine = 'native'
        self.precision = scalar.select(precision)
        if type(num_envs) is not int or not 1 <= num_envs <= 128:
            raise ValueError('num_envs must be an integer in [1, 128]')
        from libero.gpu.host.runtime.core.memory_budget import normalize_memory_budget
        policy = normalize_memory_budget(memory_budget)
        if type(strict_build_receipts) is not bool:
            raise ValueError('strict_build_receipts must be bool')
        if type(initialize) is not bool:
            raise ValueError('initialize must be bool')
        if type(debug) is not bool:
            raise ValueError('debug must be bool')
        if ready_session is not None and allow_compile:
            raise ValueError('ready_session requires allow_compile=False')
        self.debug = debug
        self.allow_compile = allow_compile
        from libero.gpu.host.build.aot import project_root, require_ready, scene_for, configure_warp_cache
        from libero.gpu.host.build.compilation import install_warp_guard
        build_root = project_root(build_root)
        with startup_phase('readiness_scene'):
            if not allow_compile:
                require_ready(build_root, bddl_file_name, num_envs=num_envs, seed=seed,
                              observation_options=environment_options, debug=debug,
                              ready_session=ready_session, return_record=False)
            scene = scene_for(build_root, bddl_file_name, seed)
        configure_warp_cache(build_root)
        install_warp_guard()
        self._lock = RLock()
        self._closed = self._failed = False
        self._initialized = False
        self._snapshot_cache = None
        self._physics = self.observer = self._status = None
        self._facades, self.shells = [], []
        self._goal_plans = []
        self.num_envs = self.env_num = num_envs
        self._rngs = []
        self.seed(seed)
        from libero.gpu.model import build_task_model
        from .batched_physics import prepare_native_batch
        from libero.gpu.host.runtime.step.recovery_final_event_environment_factory import default_integration_state
        options = dict(environment_options)
        for key in ('physics_backend', 'nworld', 'state', 'stream', 'static_model_setup'):
            if key in options:
                raise ValueError(f'{key} is controlled by GpuBatchEnv')
        if options.get('has_renderer', False) or options.get('hard_reset', False):
            raise ValueError('GpuBatchEnv requires offscreen rendering and fixed-model soft reset')
        options.setdefault('use_camera_obs', False)
        options.update(has_renderer=False, has_offscreen_renderer=bool(options['use_camera_obs']), hard_reset=False)
        physical_options = {'robots', 'gripper_types', 'controller_configs', 'initialization_noise',
                            '_gpu_constructor_call', 'xml_string_override', 'compiled_model_file'}
        if scene is not None and not physical_options.intersection(options):
            directory = Path(scene['directory'])
            options.update(xml_string_override=(directory/'scene.xml').read_text(),
                           compiled_model_file=directory/'model.mjb')
            if init_states is None:
                init_states = np.load(directory/'initial_states.npy', allow_pickle=False)
        elif not allow_compile:
            raise ValueError('physical model options require their own offline build profile')
        with startup_phase('model'):
            self.built = build_task_model(bddl_file_name,
                static_model_setup=lambda shell, model: _prepare_shared_fixtures(shell, model, seed),
                **options)
        model = self.built.native_model
        if init_states is None:
            from libero.libero import get_libero_path
            from libero.libero.benchmark.init_state_io import load_init_states
            path = Path(bddl_file_name)
            init_states = load_init_states(Path(get_libero_path('init_states')) /
                                          path.parent.name / (path.stem + '.pruned_init'))
        self.initial_states = _state_bank(init_states, int(model.nq), int(model.nv))
        self._nq, self._nv = int(model.nq), int(model.nv)
        initial = self.initial_states[np.arange(num_envs) % len(self.initial_states)]
        states = []
        for row in initial:
            state = default_integration_state(model)
            state['time'][:] = row[0]
            state['qpos'][0] = row[1:1+self._nq]
            state['qvel'][0] = row[1+self._nq:]
            states.append(state)
        try:
            with startup_phase('physics'):
                self._physics = prepare_native_batch(self.built, states=states,
                    device=device, memory_budget=policy, strict_build_receipts=strict_build_receipts,
                    build_root=build_root, debug=debug,algorithm_options=algorithm_options)
            self.build_report = self._physics.bundle.build_report
            with startup_phase('observations_facades'):
                self._assemble(build_root=build_root, strict_build_receipts=strict_build_receipts)
            if initialize:
                with startup_phase('control_graph'):
                    self._physics.prepare_control_graph()
            if initialize:
                with startup_phase('reset'):
                    self.reset(init_states=initial)
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:
                error.gpu_cleanup_errors = [repr(cleanup)]
            raise

    def _assemble(self, *, build_root=None, strict_build_receipts=False):
        import warp as wp
        from .facade import GpuSimFacade, SharedModelMetadata
        from .resident_runtime import fixed_root_descriptors
        from libero.gpu.host.adapters.observations.adapter import GpuObservationAdapter
        from libero.gpu.host.adapters.observations.runtime_binding import FixedRuntimeObservationBinding
        from libero.gpu.host.bindings.observations import install_batch
        from .goal_read_plan import GoalReadPlan
        from .task_driver import _bind_static_robot_references
        from libero.gpu.host.runtime.step.recovery_status import RecoveryBoundaryStatus
        bundle = self._physics.bundle
        self.stream, self.runtime = bundle.stream, bundle.runtime
        self.action_dim = bundle.controller.action_dim
        self.control_freq = self.built.metadata['control_freq']
        self.substeps = self.runtime.substeps
        self.memory_budget = bundle.memory_budget
        self._fields = fixed_root_descriptors(bundle)
        self._status = RecoveryBoundaryStatus(bundle) if self.debug else None
        self.observer = GpuObservationAdapter(self.built.native_model, self.built.metadata,
                                             self.num_envs, device=self.stream.device)
        self._binding = FixedRuntimeObservationBinding(bundle, self.observer,
                                                       expected_observe=self.runtime.observe).install()
        self.observation_build_report = install_batch(
            self.observer, self._binding, bundle, build_root,
            debug=self.debug, strict_build_receipts=strict_build_receipts)
        self.build_report['observations'] = self.observation_build_report
        self._selection = wp.zeros(self.num_envs, dtype=wp.int32, device=self.stream.device)
        self._all = np.ones(self.num_envs, dtype=np.int32)
        self._bind_shells()
        self._physics.bind_observer(self.observer)
        self._closed = False

    def _bind_shells(self):
        """Share task predicates and facade binding across physical engines."""
        from .facade import GpuSimFacade, SharedModelMetadata
        from .goal_read_plan import GoalReadPlan
        from .task_driver import _bind_static_robot_references
        template = self.built.domain_shell
        # Clone before installing any live CUDA/controller owners on a shell.
        with startup_phase('clone_shells'):
            self.shells = [template] + [_clone_shell(template) for _ in range(self.num_envs-1)]
        with startup_phase('shared_model_metadata'):
            metadata = SharedModelMetadata(self.built.facade_metadata(copy_arrays=False))
        for world, shell in enumerate(self.shells):
            facade = GpuSimFacade(_WorldBoundary(self, world), metadata)
            self._facades.append(facade)
            shell.sim, shell.physics_backend = facade, 'gpu'
            for index, robot in enumerate(shell.robots):
                _bind_static_robot_references(robot, facade, self.built.metadata['robot_metadata'][index])
            self.observer.bind_host_environment(shell, world=world)
            # Default task predicates use their own shell's object-state graph.
            # Keep CPU step/reset shims forbidden; this owner handles lifecycle.
        self._goal_plans = [GoalReadPlan.from_shell(shell) for shell in self.shells]
        self.observer.bind_sample_event_hook(self._sample_event)

    def _open(self):
        if self._closed:
            raise RuntimeError('GPU batch environment is closed')
        if self._failed:
            raise RuntimeError('GPU batch execution failed; close and rebuild required')

    def __len__(self):
        return self.num_envs

    def prepare_substep_fusion(self, build_root, *, allow_compile=False,
                               fast_compile=False, device_lto=False, profile_newton=False,
                               atomic_reductions=None, defer_post=True, single_solve=False, staged=False,
                               motion_passive=False, motion_threads=512,
                               actuation_solve=False, force_threads=256):
        """Opt into native physical-substep CUDA Graph execution.

        This is an experimental native-batch entry. Ordinary execution is
        load-only; an explicit offline caller may permit missing compilation.
        fast_compile selects a separately cached developer build that can
        trade execution speed for shorter assembler optimization.
        profile_newton selects the separately linked instrumentation product;
        ordinary sampling contains no Newton profiling instructions.
        defer_post keeps complete force/sensor output at the last pre-Euler
        forward and both policy forwards, when the built-in sampler permits it.
        Set False to retain complete Post on every physical substep.
        single_solve is an explicit FP32 scheduling candidate. It does not
        change the default native schedule or qualify mathematical equivalence.
        motion_passive selects the optional FP64 motion/passive kernel followed
        by the remaining fused substep; motion_threads configures that producer.
        actuation_solve also replaces entry actuation/smooth preparation and
        reuses healthy unchanged-control force outputs before the controller.
        force_threads configures that producer (64, 128 or 256; 0 disables it).
        staged selects the optional FP64 prepare/Newton/Post-tail decomposition.
        Combined with actuation_solve, it selects the measured NV39 normal
        Init256/Resume64 chain with separate controller and Euler entries inside its admitted arena;
        other shapes and ordered reductions retain the
        force-fused path. This does not change ordinary backend selection.
        It retains three complete solves and the original prefix/recovery paths;
        it is load-only unless offline compilation is explicitly permitted.
        """
        from libero.gpu.host.build.compilation import compilation_scope
        with self._lock:
            self._open()
            with compilation_scope(allow_compile):
                return self._physics.prepare_fused_substep_graph(
                    build_root, allow_compile=allow_compile,
                    fast_compile=fast_compile,
                    device_lto=device_lto, profile_newton=profile_newton,
                    atomic_reductions=atomic_reductions, defer_post=defer_post,
                    single_solve=single_solve, staged=staged,
                    motion_passive=motion_passive, motion_threads=motion_threads,
                    actuation_solve=actuation_solve, force_threads=force_threads)

    def _ids(self, ids):
        if ids is None:
            return tuple(range(self.num_envs))
        if isinstance(ids, (int, np.integer)) and not isinstance(ids, bool):
            ids = [ids]
        if isinstance(ids, bool):
            raise ValueError('invalid world IDs')
        ids = tuple(ids)
        if (len(set(ids)) != len(ids) or any(isinstance(i, bool) or
                not isinstance(i, (int, np.integer)) or not 0 <= i < self.num_envs for i in ids)):
            raise ValueError('invalid or duplicate world IDs')
        return tuple(map(int, ids))

    def seed(self, seed=None):
        with self._lock:
            self._open()
            if isinstance(seed, (int, np.integer)) and not isinstance(seed, bool):
                seeds = [int(seed)+i for i in range(self.num_envs)]
            elif seed is None:
                seeds = [None]*self.num_envs
            else:
                raise ValueError('seed must be an integer or None')
            self._rngs = [np.random.RandomState(s) for s in seeds]
            return seeds

    @contextmanager
    def _mutation(self):
        self._snapshot_cache = None
        try:
            yield
        except BaseException:
            self._failed = True
            raise
        finally:
            self._snapshot_cache = None

    def _sample_event(self, token):
        # Hooks run before host sensors, or once after a real graph replay.
        # Capturing a sampler does not publish a new host-readable event.
        self._snapshot_cache = None

    def _goal_fields(self, ids):
        from libero.gpu.host.adapters.observations.boundary import GOAL_FIELDS
        plans = getattr(self, '_goal_plans', ())
        schemas = tuple(plans[i].fields_for(self.shells[i]) if plans else GOAL_FIELDS for i in ids)
        names = tuple(name for name in GOAL_FIELDS if any(name in schema for schema in schemas))
        return names, schemas

    def _boundary_snapshot(self, names):
        """Complete requested fields within this one synchronized sample event.

        Qualified goal prefetches are private. Explicit/custom consumers still
        receive real canonical fields, including contacts they request later.
        """
        names = tuple(dict.fromkeys(names))
        if self._snapshot_cache is None:
            self._check()
            self._snapshot_cache = {}
        missing = tuple(name for name in names if name not in self._snapshot_cache)
        if missing:
            self._snapshot_cache.update(self._snapshot(missing))
        return self._snapshot_cache

    def _check(self):
        if not self.debug:
            # Host consumers still need completed device writes. This wait
            # performs no packet gathering, source hashing or correctness check.
            import warp as wp
            wp.synchronize_stream(self.stream)
            return None
        record = self._status.synchronize_and_check()
        self._integrations = np.asarray([row[16] for row in record.rows], dtype=np.int64)
        return record

    def _snapshot(self, names=None):
        from libero.gpu.host.adapters.environment.boundary_downloads import BoundaryDownloads
        downloads = getattr(self, '_boundary_downloads', None)
        if downloads is None:
            downloads = self._boundary_downloads = BoundaryDownloads()
        return downloads.download(self, names)

    def _observations(self, ids):
        values = self.observer.observations_many(ids,
            image_convention=self.built.metadata.get('image_convention', 'opengl'))
        if not values:
            return {}
        return {name: np.stack([obs[name] for obs in values]) for name in values[0]}

    def _activate_all(self):
        import warp as wp
        with wp.ScopedStream(self.stream):
            self.runtime.intent.assign(self._all)

    @environment_scope
    def step(self, actions):
        with self._lock:
            self._open()
            if not getattr(self, '_initialized', True):
                raise RuntimeError('reset the GPU batch before stepping')
            actions = np.asarray(actions, dtype=np.float64)
            if self.debug and (actions.shape != (self.num_envs, self.action_dim) or not np.isfinite(actions).all()):
                raise ValueError(f'actions must have shape ({self.num_envs}, {self.action_dim}) and be finite')
            if any(shell.done for shell in self.shells):
                raise ValueError('reset terminated worlds before stepping the batch')
            from libero.gpu.host.adapters.observations.boundary import boundary_predicates
            with self._mutation():
                before = self._integrations.copy() if self.debug else None
                self._activate_all()
                self._physics.enqueue_action(np.ascontiguousarray(actions))
                self._check()
                if self.debug and not np.all(self._integrations-before == self.substeps):
                    raise RuntimeError('GPU batch did not commit every requested physics substep')
                names, schemas = self._goal_fields(tuple(range(self.num_envs)))
                self._snapshot_cache = self._snapshot(names)
                rewards, successes, infos = [], [], []
                for world, (shell, facade) in enumerate(zip(self.shells, self._facades)):
                    shell.timestep += 1
                    shell.cur_time += shell.control_timestep
                    with boundary_predicates(shell, facade, fields=schemas[world]):
                        reward, _, info = shell._post_action(actions[world])
                        success = shell._check_success()
                    rewards.append(reward)
                    successes.append(success)
                    infos.append(dict(info, env_id=world))
                obs = self._observations(tuple(range(self.num_envs)))
                return obs, np.asarray(rewards), np.asarray(successes, dtype=bool), infos

    @environment_scope
    def reset(self, id=None, *, seed=None, init_states=None):
        with self._lock:
            self._open()
            ids = self._ids(id)
            if not getattr(self, '_initialized', True) and ids != tuple(range(self.num_envs)):
                raise RuntimeError('reset all worlds before selecting individual worlds')
            if seed is not None:
                self.seed(seed)
            if init_states is not None:
                rows = self._validate_states(init_states, ids)
            else:
                rows = (np.stack([self.initial_states[self._rngs[i].randint(len(self.initial_states))]
                                  for i in ids]) if ids else np.empty((0, 1+self._nq+self._nv)))
            with self._mutation():
                self._restore(rows, ids, reset=True)
                for i in ids:
                    shell = self.shells[i]
                    shell.timestep, shell.cur_time, shell.done = 0, 0.0, False
                    shell._obs_cache = {}
                self._initialized = True
                return self._observations(ids)

    def _validate_states(self, states, ids):
        rows = np.asarray(states, dtype=np.float64)
        if rows.shape != (len(ids), 1+self._nq+self._nv) or not np.isfinite(rows).all():
            raise ValueError('one finite LIBERO flat state required per selected world')
        return rows.copy()

    @environment_scope
    def set_init_state(self, states, id=None):
        with self._lock:
            self._open()
            ids = self._ids(id)
            rows = self._validate_states(states, ids)
            with self._mutation():
                self._restore(rows, ids, reset=False)
                return self._observations(ids)

    def _restore(self, rows, ids, *, reset):
        import warp as wp
        from libero.gpu.host.runtime.step.shape_environment import _written
        if not ids:
            return
        mask = np.zeros(self.num_envs, dtype=np.int32)
        mask[list(ids)] = 1
        e, bundle = self.runtime.euler, self._physics.bundle
        roots = {'time': e.time, 'qpos': e.qpos, 'qvel': e.qvel}
        payload = {'time': rows[:, 0], 'qpos': rows[:, 1:1+self._nq],
                   'qvel': rows[:, 1+self._nq:]}
        if reset:
            roots.update(ctrl=e.ctrl, qacc_warmstart=e.qacc_warmstart,
                         qfrc_applied=bundle.dynamics.qfrc_applied,
                         xfrc_applied=bundle.dynamics.xfrc_applied)
            for name in set(roots)-set(payload):
                payload[name] = np.zeros((len(ids), roots[name].size//self.num_envs))
        transfers = []
        with wp.ScopedStream(self.stream):
            self._selection.assign(mask)
            for name, array in roots.items():
                source = wp.array(np.ascontiguousarray(payload[name]).reshape(-1),
                                  dtype=scalar.wp_real, device=self.stream.device)
                transfers.append(source)
                width = array.size//self.num_envs
                for row, world in enumerate(ids):
                    wp.copy(array, source, dest_offset=world*width, src_offset=row*width,
                            count=width, stream=self.stream)
            wp.launch(_written, self.num_envs, inputs=[self._selection, self.runtime.state_epoch,
                bundle.expected_state, bundle.foundation.state_valid,
                self.runtime.forward.valid, self.runtime.valid], device=self.stream.device, stream=self.stream)
            if reset:
                bundle.controller.initialize_ctrl_from_reset(self._selection, e.ctrl, stream=self.stream)
                self.observer.reset(self._selection, stream=self.stream)
            self.runtime.forward.enqueue(self._selection, self.stream)
            if reset:
                bundle.controller.bootstrap_from_current_physics(self._selection, stream=self.stream)
            record = self._check()
            if self.debug and any(record.rows[i][18] != 1 for i in ids):
                raise RuntimeError('GPU batch reset did not publish every selected forward')
            from libero.gpu.host.adapters.observations.boundary import boundary_predicates
            names, schemas = self._goal_fields(ids)
            self._snapshot_cache = self._snapshot(names)
            try:
                for offset, world in enumerate(ids):
                    shell = self.shells[world]
                    with boundary_predicates(shell, self._facades[world], fields=schemas[offset]):
                        shell._check_success()
                        shell._post_process()
            finally:
                self._snapshot_cache = None
            self.observer.sample(self._binding.fields(), force=True,
                                 world_mask=self._selection, stream=self.stream)
            self._check()
        # Transfers remain alive until the synchronous restore boundary above.

    @environment_scope
    def get_sim_state(self):
        with self._lock:
            self._open()
            self._check()
            e = self.runtime.euler
            return np.concatenate((e.time.numpy().reshape(-1, 1),
                                   e.qpos.numpy().reshape(self.num_envs, self._nq),
                                   e.qvel.numpy().reshape(self.num_envs, self._nv)), axis=1)

    @environment_scope
    def diagnostics(self):
        with self._lock:
            self._open()
            self._check()
            return dict(physics_backend='gpu', precision=self.precision, scalar_bytes=scalar.itemsize,
                        debug=self.debug, nworld=self.runtime.euler.capacity,
                        model_sha256=self.built.metadata['model_sha256'],
                        substeps=self.substeps, memory_budget=self.memory_budget,
                        actual_integration_count=self.runtime.integration_count.numpy().tolist(),
                        first_error=self.runtime.first_error.numpy().tolist(),
                        observations=getattr(self, 'observation_build_report', None),
                        observation_graph_eligible=self.observer.graph_eligible,
                        goal_fields=[list(plan.fields_for(shell))
                                     for plan, shell in zip(self._goal_plans, self.shells)])

    @environment_scope
    def modify_observable(self, name, attribute, modifier):
        """Modify the shared schema; Python callback objects follow Python ownership.

        Stateful world-specific processors should be supplied by separate custom
        observable owners. A custom sensor/processor switches subsequent actions
        to the same GPU arena's uncaptured substep route.
        """
        with self._lock:
            self._open()
            self._snapshot_cache = None
            try:
                return self.observer.modify_observable(name, attribute, modifier, stream=self.stream)
            finally:
                self._snapshot_cache = None

    @environment_scope
    def add_observable(self, observable):
        """Add independent per-world Observable owners through the shared adapter."""
        with self._lock:
            self._open()
            self._snapshot_cache = None
            try:
                return self.observer.add_observable(observable, stream=self.stream)
            finally:
                self._snapshot_cache = None

    @environment_scope
    def check_success(self):
        with self._lock:
            self._open()
            from libero.gpu.host.adapters.observations.boundary import boundary_predicates
            self._check()
            ids = tuple(range(self.num_envs))
            names, schemas = self._goal_fields(ids)
            self._snapshot_cache = self._snapshot(names)
            try:
                result = []
                for world, (shell, facade) in enumerate(zip(self.shells, self._facades)):
                    with boundary_predicates(shell, facade, fields=schemas[world]):
                        result.append(shell._check_success())
                return np.asarray(result, dtype=bool)
            finally:
                self._snapshot_cache = None

    @environment_scope
    def close(self):
        with self._lock:
            if self._closed:
                return
            stream = getattr(self, 'stream', None)
            scope = nullcontext()
            if stream is not None:
                import warp as wp
                scope = wp.ScopedStream(stream)
            with scope:
                if self._physics is not None:
                    self._physics.discard_control_graph()
                    self._physics.bind_observer(None)
                # Release graphics/facade borrows before the shared physics owner.
                for facade in self._facades:
                    facade.close()
                self._facades.clear()
                if self.observer is not None:
                    self.observer.close()
                if self._physics is not None:
                    self._physics.close()
                if self._status is not None:
                    self._status.release()
                self._snapshot_cache = None
                self._boundary_downloads = None
                self.shells.clear()
                self._goal_plans.clear()
                self._fields = {}
                self._binding = self.observer = self.runtime = self._physics = None
                self._selection = self._status = None
                import gc
                gc.collect()
                if stream is not None:
                    wp.synchronize_stream(stream)
                self._closed = True

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *exc):
        self.close()
