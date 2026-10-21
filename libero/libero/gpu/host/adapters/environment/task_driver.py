"""Real task lifecycle over injected device-resident components.

The original model-only shell supplies task definitions and CPU placement/goal
logic only. Original Robot.reset/_load_controller and env.step/reset are never
invoked. Numerical physics comes exclusively from the runtime bound to sim.
"""
from dataclasses import dataclass
from threading import RLock
from contextlib import contextmanager
import numpy as np
import random
import gc
from .facade import UnsupportedGpuOperation, CompletedColdModelUpdate
from libero.gpu.host.build.compilation import load_only_entry

_PLACEMENT_LOCK = RLock()


def _bind_static_robot_references(robot, sim, meta=None):
    """Bind compiled indices and an existing facade; perform no CPU initialization."""
    robot.sim = sim
    if meta is None:return
    robot.robot_joints = list(meta["joint_names"])
    robot._ref_joint_pos_indexes = list(meta["qpos_indices"])
    robot._ref_joint_vel_indexes = list(meta["qvel_indices"])
    robot._ref_joint_indexes = list(meta["joint_ids"])
    robot._ref_joint_actuator_indexes = list(meta["actuator_ids"])
    gripper = meta.get("gripper", {})
    robot._ref_gripper_joint_pos_indexes = list(gripper.get("qpos_indices", ()))
    robot._ref_gripper_joint_vel_indexes = list(gripper.get("qvel_indices", ()))
    robot._ref_joint_gripper_actuator_indexes = list(gripper.get("actuator_ids", ()))
    if robot.has_gripper: robot.gripper_joints = list(gripper["joint_names"])
    robot.eef_site_id = meta["eef_site_id"]
    robot.eef_cylinder_id = meta["eef_cylinder_id"]


@dataclass
class GpuTaskComponents:
    built: object
    sim: object
    physics: object
    controller: object
    observations: object
    world_mask: object
    world: int = 0
    # One true runtime owner supplies all observed device arrays, not CPU copies.
    observation_fields: object = None
    synchronize_and_check: object = None
    bootstrap_controller: object = None
    stage_action: object = None
    close_runtime: object = None
    observation_hook: object = None
    stream: object = None
    released: bool = False
    parameter_replacement_binder: object = None
    native_model_replacement_binder: object = None
    release_started: bool = False

    def validate(self):
        if self.released: raise RuntimeError('GPU task components were released')
        if getattr(self.sim, "physics_backend", None) != "gpu":
            raise UnsupportedGpuOperation("task requires a real GPU sim facade")
        for name in ("observation_fields", "synchronize_and_check", "bootstrap_controller", "stage_action", "close_runtime", "observation_hook"):
            if not callable(getattr(self, name)):
                raise UnsupportedGpuOperation(f"task runtime assembly lacks {name}")
        if getattr(self.physics, "observe", None) is not self.observation_hook:
            raise UnsupportedGpuOperation("physics substep must bind the actual observation hook")
        if self.stream is None:
            raise UnsupportedGpuOperation("explicit shared GPU stream required")
        for name in ("enqueue_action",):
            if not callable(getattr(self.physics, name, None)):
                raise UnsupportedGpuOperation(f"physics assembly lacks {name}")
        for name in ("reset", "sample", "observations", "close"):
            if not callable(getattr(self.observations, name, None)):
                raise UnsupportedGpuOperation(f"device observation adapter lacks {name}")

    def release(self, *, retire_controller_views=True):
        """Synchronize/close actual owners, then release all borrowed GPU roots."""
        if self.released:return
        self.release_started=True
        if retire_controller_views:
            for robot in self.built.domain_shell.robots:
                invalidate = getattr(getattr(robot, "controller", None), "invalidate", None)
                if callable(invalidate): invalidate()
        self.observations.close()
        self.close_runtime()
        self.released=True
        for name in ('physics','controller','observations','world_mask','observation_fields',
                     'synchronize_and_check','bootstrap_controller','stage_action','observation_hook','parameter_replacement_binder','native_model_replacement_binder'):
            setattr(self,name,None)
        gc.collect()

class GpuTaskEnvironment:
    physics_backend = "gpu"

    @load_only_entry
    def __init__(self, components, *, rebuild=None, initialize=True):
        components.validate()
        self._lock = RLock()
        self._closed = False
        self._unusable = False
        self._replacing = False
        self._needs_reset = False
        self.last_failure = None
        self._rebuild = rebuild
        self._rng = np.random.RandomState()
        self.components = components
        self._install_shell()
        if initialize:
            self._initialize_fresh()

    def __getattr__(self, name):
        # Task metadata is the actual original task shell, including custom
        # objects, parsed BDDL and sampler definitions. Do not accidentally
        # expose inherited CPU execution methods through generic delegation.
        shell = self.__dict__.get("shell")
        if shell is not None and not name.startswith("_"):
            value = getattr(shell, name)
            if not callable(value): return value
        raise AttributeError(name)

    def _install_shell(self, *, reuse_controller_views=False):
        self.built = self.components.built
        self.shell = self.built.domain_shell
        self.sim = self.components.sim
        self.shell.sim = self.sim
        self.shell.physics_backend = "gpu"
        for name in ("step", "reset", "reset_from_xml_string", "_get_observations", "_update_observables"):
            setattr(self.shell, name, getattr(self, name))
        bind_host = getattr(self.components.observations, "bind_host_environment", None)
        if callable(bind_host): bind_host(self.shell, world=self.components.world)
        render = getattr(self.components.observations, "render_gpu", None)
        if callable(render):
            # Capture this generation's owners. Replacement binds a new sim;
            # retained old facades fail their closed-owner checks.
            components = self.components
            self.sim.bind_renderer(load_only_entry(lambda **options: render(
                components.observation_fields(), world=components.world,
                stream=components.stream, **options)),
                context_provider=getattr(components.observations, "public_render_context", None))
        # These are static references only; never call shell._setup_references,
        # because it invokes normal robot/controller initialization.
        static_robots = getattr(self.built, "metadata", {}).get("robot_metadata", ())
        for index, robot in enumerate(self.shell.robots):
            meta = static_robots[index] if index < len(static_robots) else None
            _bind_static_robot_references(robot, self.sim, meta)
            adapter = self.components.controller
            if hasattr(adapter, "public") and callable(getattr(adapter, "validate", None)):
                from libero.gpu.host.runtime.control.public_view import controller_public_view
                previous = getattr(robot, "controller", None)
                component_owner = self.components
                guard = lambda owner=component_owner: not owner.released and not owner.sim._closed
                rebind = getattr(previous, "rebind", None)
                if reuse_controller_views and callable(rebind):
                    robot.controller = rebind(adapter, self.components.world, owner_guard=guard)
                else:
                    invalidate = getattr(previous, "invalidate", None)
                    if callable(invalidate): invalidate()
                    robot.controller = controller_public_view(adapter, self.components.world, owner_guard=guard)
        binder=self.components.parameter_replacement_binder
        if binder is not None:
            if not callable(binder):raise TypeError('parameter replacement binder must be callable')
            binder(self.replace_components_for_model_update)
        native_binder = self.components.native_model_replacement_binder
        if native_binder is not None:
            if not callable(native_binder): raise TypeError("native model replacement binder must be callable")
            native_binder(self.replace_native_components)
        self.shell.obj_body_id = {
            name: self.sim.model.body_name2id(obj.root_body)
            for name, obj in {**self.shell.objects_dict, **self.shell.fixtures_dict}.items()
        }

    @contextmanager
    def _placement_rng(self):
        # Original placement samplers use numpy's global RNG. Give every world
        # its own sequence while serializing only host sampling, not physics.
        with _PLACEMENT_LOCK:
            previous = np.random.get_state()
            np.random.set_state(self._rng.get_state())
            try:
                yield
            finally:
                self._rng.set_state(np.random.get_state())
                np.random.set_state(previous)

    @contextmanager
    def _xml_build_rng(self):
        # XML replacement originally compiles an override without re-running
        # task sampling. The model-only shell rebuild must not consume the
        # next placement's per-world RNG sequence or host global RNG state.
        with _PLACEMENT_LOCK:
            np_previous = np.random.get_state()
            random_previous = random.getstate()
            np.random.set_state(self._rng.get_state())
            try:
                yield
            finally:
                np.random.set_state(np_previous)
                random.setstate(random_previous)
    def _open(self, allow_reset_pending=False):
        if self._replacing: raise RuntimeError('GPU model replacement in progress')
        if self._closed: raise RuntimeError("GPU task environment is closed")
        if self._unusable or getattr(self.sim,"_quarantined",False):
            raise RuntimeError("GPU environment lifecycle failed; close and rebuild required")
        if self._needs_reset and not allow_reset_pending:
            raise RuntimeError("placement reset did not finish; reset retry required")

    @contextmanager
    def _mutation(self, *, placement_retry=False):
        try:
            yield
        except BaseException as exc:
            from robosuite.utils.errors import RandomizationError
            self.last_failure = {"type": type(exc).__name__, "message": str(exc)}
            if placement_retry and isinstance(exc, RandomizationError):
                self._needs_reset = True
            else:
                self._unusable = True
            raise
        else:
            self._needs_reset = False
    @property
    def obj_of_interest(self): return self.shell.obj_of_interest
    @property
    def robots(self): return self.shell.robots
    @property
    def model(self): return self.shell.model
    @property
    def _visualizations(self): return self.shell._visualizations
    @property
    def action_dim(self): return int(getattr(self.components.controller, "action_dim", 7))

    @property
    def language_instruction(self): return self.shell.language_instruction
    @property
    def control_freq(self): return self.shell.control_freq
    @property
    def horizon(self): return self.shell.horizon
    @property
    def timestep(self): return self.shell.timestep
    @property
    def cur_time(self): return self.shell.cur_time
    @property
    def done(self): return self.shell.done

    def observation_spec(self): return self._get_observations()

    def get_robot_state_vector(self, obs):
        return self.shell.get_robot_state_vector(obs)

    def get_object(self, object_name): return self.shell.get_object(object_name)
    def is_fixture(self, object_name): return self.shell.is_fixture(object_name)

    @load_only_entry
    def edit_model_xml(self, xml_str):
        with self._lock:
            self._open()
            return self.shell.edit_model_xml(xml_str)

    @load_only_entry
    def set_xml_processor(self, processor):
        with self._lock:
            self._open()
            # Use the actual cold shell so constructor/replacement code reads
            # the same callback rather than a shadow value on the public proxy.
            self.shell.set_xml_processor(processor)

    @load_only_entry
    def set_camera_pos_quat(self, camera_pos, camera_quat):
        with self._lock:
            self._open()
            return self.shell.set_camera_pos_quat(camera_pos, camera_quat)

    @load_only_entry
    def set_visualization(self):
        with self._lock:
            self._open()
            self.shell.set_visualization()
            self.sim.flush_public_arrays()

    @load_only_entry
    def initialize_renderer(self):
        with self._lock:
            self._open()
            callback=getattr(self.components.observations,'initialize_renderer',None)
            if not callable(callback):raise UnsupportedGpuOperation('GPU renderer cold initialization is not bound')
            return callback(world=self.components.world)

    @property
    def observation_modalities(self):
        return set(self.components.observations.observation_modalities)

    @load_only_entry
    def reward(self, action=None):
        from libero.gpu.host.adapters.observations.boundary import boundary_predicates
        with self._lock:
            self._open()
            self.components.synchronize_and_check()
            with boundary_predicates(self.shell, self.sim): return self.shell.reward(action)

    @load_only_entry
    def visualize(self, vis_settings):
        with self._lock:
            self._open()
            with self._mutation():
                # Original methods modify only site visual bytes; the bound sim
                # exposes public RGBA arrays and flushes them to the visual owner.
                self.shell.visualize(vis_settings)
                self.sim.flush_public_arrays()

    @load_only_entry
    def check_contact(self, geoms_1, geoms_2=None):
        from libero.gpu.host.adapters.observations.boundary import boundary_predicates
        with self._lock:
            self._open()
            self.components.synchronize_and_check()
            with boundary_predicates(self.shell, self.sim):
                return self.shell.check_contact(geoms_1, geoms_2)

    @load_only_entry
    def get_contacts(self, model):
        from libero.gpu.host.adapters.observations.boundary import boundary_predicates
        with self._lock:
            self._open()
            self.components.synchronize_and_check()
            with boundary_predicates(self.shell, self.sim): return self.shell.get_contacts(model)

    @property
    def observation_names(self):
        plan = self.components.observations.plan
        if hasattr(self.components.observations, "observation_names"):
            return self.components.observations.observation_names
        return set(plan.order or tuple(spec.name for spec in plan.specs) +
                   tuple(spec["name"] for spec in plan.image_specs))

    @property
    def enabled_observables(self):
        plan = self.components.observations.plan
        if hasattr(self.components.observations, "enabled_observables"):
            return self.components.observations.enabled_observables
        return {spec.name for spec in plan.specs if spec.enabled} | {
            spec["name"] for spec in plan.image_specs if spec.get("enabled", True)}

    @property
    def active_observables(self):
        plan = self.components.observations.plan
        if hasattr(self.components.observations, "active_observables"):
            return self.components.observations.active_observables
        return {spec.name for spec in plan.specs if spec.active} | {
            spec["name"] for spec in plan.image_specs if spec.get("active", True)}

    @load_only_entry
    def modify_observable(self, observable_name, attribute, modifier):
        with self._lock:
            self._open()
            callback = getattr(self.components.observations, "modify_observable", None)
            if not callable(callback):
                raise UnsupportedGpuOperation("GPU observable schema mutation is not bound")
            return callback(observable_name, attribute, modifier, stream=self.components.stream)

    @load_only_entry
    def add_observable(self, observable):
        with self._lock:
            self._open()
            callback = getattr(self.components.observations, "add_observable", None)
            if not callable(callback): raise UnsupportedGpuOperation("GPU observable addition is not bound")
            return callback(observable, stream=self.components.stream)

    @load_only_entry
    def render(self):
        with self._lock:
            self._open()
            self.sim.flush_public_arrays()
            return self.shell.viewer.render()

    @load_only_entry
    def get_pixel_obs(self):
        with self._lock:
            self._open()
            self.sim.flush_public_arrays()
            return self.shell.viewer.get_pixel_obs()

    @load_only_entry
    def close_renderer(self):
        with self._lock:
            self._open()
            return self.shell.viewer.close()

    @load_only_entry
    def clear_objects(self, object_names):
        names = {object_names} if isinstance(object_names, str) else set(object_names)
        with self._lock:
            self._open()
            with self._mutation():
                for obj in self.model.mujoco_objects:
                    if obj.name in names:
                        self.sim.data.set_joint_qpos(obj.joints[0], np.array((10, 10, 10, 1, 0, 0, 0)))

    @property
    def action_spec(self):
        limits = getattr(self.components.controller, "control_limits", None)
        if limits is not None:
            return tuple(np.array(value, copy=True) for value in limits)
        if self.action_dim == 7:
            return -np.ones(7), np.ones(7)
        raise UnsupportedGpuOperation("GPU controller action limits are not bound")

    def seed(self, seed=None):
        with self._lock:
            self._open()
            self._rng = np.random.RandomState(seed)

    def _reset_robots(self):
        # Reproduce original arm write -> controller constructor forward/cache ->
        # gripper write order. Reset noise is model preparation, not physics.
        for robot in self.robots:
            initial = np.array(robot.init_qpos, dtype=np.float64, copy=True)
            if not self.shell.deterministic_reset:
                noise = robot.initialization_noise
                if noise["type"] == "gaussian":
                    initial += np.random.randn(len(initial)) * noise["magnitude"]
                elif noise["type"] == "uniform":
                    initial += np.random.uniform(-1.0, 1.0, len(initial)) * noise["magnitude"]
                else:
                    raise ValueError("unsupported robot reset noise type")
            for name, value in zip(robot.robot_model.joints, initial):
                self.sim.data.set_joint_qpos(name, value)
            # Callback must reproduce both actual constructor forwards, gather
            # and bootstrap. The original constructor trace has two forwards;
            # it must not call original CPU controller construction.
            self.components.bootstrap_controller(robot, self.components.world_mask)
            self.components.synchronize_and_check()
            if robot.has_gripper and not self.shell.deterministic_reset:
                for name, value in zip(robot.gripper.joints, robot.gripper.init_qpos):
                    self.sim.data.set_joint_qpos(name, value)

    def _reset_internal_gpu(self):
        self.shell.sim_state_initial = self.sim.get_state()
        self.shell.cur_time, self.shell.timestep, self.shell.done = 0.0, 0, False
        self.shell._obs_cache = {}
        self.components.observations.reset(self.components.world_mask, stream=self.components.stream)
        with self._placement_rng():
            self._reset_robots()
            self.shell._reset_task_placements()
        self.components.synchronize_and_check()

    def _initialize_fresh(self):
        # Mirrors _initialize_sim.forward before initial _reset_internal.
        self.sim.forward()
        self.components.synchronize_and_check()
        self._reset_internal_gpu()
        # Original constructor does not add outer reset's final forward/sample.

    def _sample(self, *, force=False):
        c = self.components
        c.observations.sample(c.observation_fields(), force=force,
                              world_mask=c.world_mask, stream=c.stream)

    @load_only_entry
    def _update_observables(self, force=False):
        with self._lock:
            self._open()
            with self._mutation():
                self._sample(force=force)

    @load_only_entry
    def _get_observations(self, force_update=False):
        with self._lock:
            self._open()
            if force_update:
                with self._mutation():
                    self._sample(force=True)
                    self.components.synchronize_and_check()
            else:
                self.components.synchronize_and_check()
            self.sim.synchronize_public_arrays()
            return self.components.observations.observations(
                self.components.world,
                image_convention=getattr(self.built, "metadata", {}).get("image_convention", "opengl"),
            )

    def _replace_components(self, fresh, *, preserve_sim=None):
        fresh.validate()
        old = self.components
        if fresh is old:
            # Explicit in-place reprepare; provider keeps its actual arena owners.
            self._install_shell()
            return
        try:
            old.release()
        except BaseException:
            self._unusable = True
            fresh.release()
            raise
        if preserve_sim is not None:
            released_runtime = preserve_sim._runtime
            preserve_sim.adopt_runtime_from(fresh.sim)
            # Historical release capabilities remain bound to their old arena.
            old.close_runtime = released_runtime.close
            fresh.sim = preserve_sim
            fresh.close_runtime = preserve_sim.close
        self.components = fresh
        self._install_shell()

    def _quarantine_replacement(self,detail):
        self._unusable=True
        self.last_failure={'type':'ModelReplacementFailure','message':str(detail)}
        quarantine=getattr(self.sim,'_quarantine',None)
        if callable(quarantine):quarantine('model_replacement')

    @load_only_entry
    def replace_native_components(self, build_new, source_sim):
        """Restore a freshly sourced model/arena into the stable public facade.

        build_new(old_components) performs actual state/observer snapshot,
        old.release(), fresh model/spec allocation and full restore. It must not
        run task construction/reset/forward or reuse the old physical owner.
        """
        from libero.gpu.host.runtime.core.contracts import _count
        with self._lock:
            self._open(allow_reset_pending=True)
            old = self.components
            if source_sim is not old.sim: raise ValueError("foreign native model update facade")
            if not callable(build_new): raise TypeError("actual native model replacement builder required")
            old_generation = old.sim._model_generation
            if old_generation is None: raise UnsupportedGpuOperation("native update requires actual ModelGeneration")
            next_generation = _count(old_generation.generation + 1, "generation")
            visual_generation = old.sim.visual_generation
            visual_metadata = old.sim.model._metadata
            visual_snapshot = {name: np.array(visual_metadata[name], copy=True)
                               for name in ("site_rgba", "geom_rgba") if name in visual_metadata}
            state = old.sim._world_state
            host_snapshot = None if state is None else {
                "state_epoch": state.state_epoch, "publications": dict(state.publications),
                "sequence": state._sequence,
            }
            self._replacing = True
            fresh = None
            try:
                fresh = build_new(old)
                if not old.released: raise RuntimeError("old arena must release before replacement allocation")
                fresh.validate()
                if fresh.built.domain_shell is not old.built.domain_shell:
                    raise ValueError("native parameter update must preserve original task shell identity")
                fresh_sim = fresh.sim
                generation = fresh_sim._model_generation
                if generation is None or generation is old_generation:
                    raise ValueError("native update requires a freshly built ModelGeneration")
                generation.generation = next_generation
                if generation.parameters_sha256 != generation.spec.static_arrays_sha256:
                    raise ValueError("native replacement must use the actual fresh static readset")
                metadata = fresh_sim.model._metadata
                for name, value in visual_snapshot.items():
                    target = metadata.get(name)
                    if not isinstance(target, np.ndarray) or target.shape != value.shape or target.dtype != value.dtype:
                        raise ValueError("fresh native visual mirror schema differs: " + name)
                    value.flags.writeable = False
                    metadata[name] = value
                fresh_sim.visual_generation = visual_generation
                new_state = fresh_sim._world_state
                if host_snapshot is not None:
                    if new_state is None: raise ValueError("fresh native world state is not bound")
                    new_state.model = generation
                    new_state.state_epoch = host_snapshot["state_epoch"]
                    # The old tags stay stale against the newly sourced model.
                    # Copying host tags never declares fresh device products.
                    new_state.publications = dict(host_snapshot["publications"])
                    new_state._sequence = host_snapshot["sequence"]
                source_sim.adopt_runtime_from(fresh_sim)
                fresh.sim = source_sim
                fresh.close_runtime = source_sim.close
                self.components = fresh
                self._install_shell(reuse_controller_views=True)
                return None
            except BaseException as exc:
                if old.released or old.release_started:
                    self._quarantine_replacement(exc)
                    if fresh is not None:
                        try: fresh.release()
                        except BaseException as cleanup:
                            exc.gpu_cleanup_errors = [repr(cleanup)]
                else:
                    # Preflight/snapshot refusals leave the current owner alive.
                    # A supplier violating release-before-allocation must also
                    # release its uninstalled fresh roots without retiring the
                    # active original shell's controller view.
                    if fresh is not None:
                        try: fresh.release(retire_controller_views=False)
                        except BaseException as cleanup:
                            exc.gpu_cleanup_errors = [repr(cleanup)]
                    self.last_failure = {"type": type(exc).__name__, "message": str(exc)}
                raise
            finally:
                self._replacing = False

    @load_only_entry
    def replace_components_for_model_update(self,build_new,token,model_generation):
        """Install restored actual owners atomically after cold model replacement.

        The source owner prepares a descriptor plan before old release, captures
        and restores actual device data, and returns (fresh_components, plan).
        No constructor, robot reset, implicit forward, or observation sample runs.
        """
        from libero.gpu.host.runtime.core.generation import commit_parameter_descriptor
        with self._lock:
            self._open()
            old=self.components
            if old.sim._model_generation is not model_generation:
                raise ValueError('foreign model generation replacement request')
            if not callable(build_new):raise TypeError('actual cold model builder required')
            if (token.model_identity != model_generation.spec.identity_key or
                    token.base_generation != model_generation.generation or
                    token.base_digest != model_generation.parameters_sha256 or
                    token.next_generation != model_generation.generation+1):
                raise ValueError('stale or foreign cold model update token')
            visual_metadata=getattr(old.sim.model,'_metadata',{})
            visual_snapshot={name:np.array(visual_metadata[name],copy=True)
                             for name in ('site_rgba','geom_rgba') if name in visual_metadata}
            visual_generation=getattr(old.sim,'visual_generation',0)
            state=getattr(old.sim,'_world_state',None)
            host_snapshot=None if state is None else {
                'state_epoch':state.state_epoch,'publications':dict(state.publications),
                'sequence':state._sequence,
            }
            self._replacing=True
            fresh=None
            try:
                fresh,plan=build_new(old)
                if not old.released:raise RuntimeError('old arena must release before new allocation')
                fresh.validate()
                if fresh.built.domain_shell is not old.built.domain_shell:
                    raise ValueError('parameter replacement must preserve the active task shell')
                if plan[0] != token or fresh.sim._model_generation.spec.identity_key != plan[1].identity_key:
                    raise ValueError('fresh native descriptor differs from parameter token/plan')
                # Visual state is independent of the fresh physical MJB/spec.
                # The observer owner has already restored actual visual bytes;
                # preserve this facade mirror without another GPU upload or gen.
                fresh_metadata=getattr(fresh.sim.model,'_metadata',{})
                for name,value in visual_snapshot.items():
                    target=fresh_metadata.get(name)
                    if not isinstance(target,np.ndarray) or target.shape!=value.shape or target.dtype!=value.dtype:
                        raise ValueError('fresh visual mirror schema differs: '+name)
                    value.flags.writeable=False
                    fresh_metadata[name]=value
                fresh.sim.visual_generation=visual_generation
                commit_parameter_descriptor(model_generation,plan)
                fresh.sim._model_generation=model_generation
                newstate=getattr(fresh.sim,'_world_state',None)
                if newstate is not None:
                    newstate.model=model_generation
                    if host_snapshot is not None:
                        newstate.state_epoch=host_snapshot['state_epoch']
                        # Old publication tags remain stale after generation+1;
                        # this host copy never manufactures a new device phase.
                        newstate.publications=dict(host_snapshot['publications'])
                        newstate._sequence=host_snapshot['sequence']
                self._replace_components(fresh, preserve_sim=old.sim)
                return CompletedColdModelUpdate(
                    model_generation,model_generation.spec.identity_key,
                    model_generation.generation,model_generation.parameters_sha256,
                    self._quarantine_replacement,
                    facade=self.sim, runtime=self.sim._runtime, world_state=self.sim._world_state,
                )
            except BaseException as exc:
                self._quarantine_replacement(exc)
                if fresh is not None and fresh is not self.components:
                    quarantine=getattr(fresh.sim,'_quarantine',None)
                    if callable(quarantine):quarantine('model_replacement')
                    try:fresh.release()
                    except BaseException as close_error:
                        exc.gpu_cleanup_errors=[repr(close_error)]
                raise
            finally:
                self._replacing=False
    @load_only_entry
    def reset(self):
        with self._lock:
            self._open(allow_reset_pending=True)
            with self._mutation(placement_retry=True):
                self._needs_reset = False
                if self.shell.hard_reset and not self.shell.deterministic_reset:
                    if self._rebuild is None:
                        raise UnsupportedGpuOperation("hard reset requires model-only rebuild binding")
                    with self._placement_rng(): fresh = self._rebuild(None, self.components)
                    self._replace_components(fresh)
                    self.sim.forward()
                    self.components.synchronize_and_check()
                else:
                    self.sim.reset()
                self._reset_internal_gpu()
                self.sim.forward()
                self._disable_visual_sites()
                return self._get_observations(force_update=True)

    def _disable_visual_sites(self):
        # Equivalent to visualize({vis:False}), without original renderer/robot
        # visualization callbacks or host mutation of device physical state.
        rgba = np.array(self.sim.model._metadata["site_rgba"], copy=True)
        models = list(getattr(self.shell.model, "mujoco_objects", ()))
        for robot in self.robots:
            models.append(robot.robot_model)
            if robot.has_gripper: models.append(robot.gripper)
        for model in models:
            for site in model.sites:
                index = self.sim.model.site_name2id(site)
                if rgba[index,3] > 0.0: rgba[index,3] = -rgba[index,3]
        self.sim.update_visual_parameters({"site_rgba":rgba})

    @load_only_entry
    def reset_from_xml_string(self, xml_string):
        if not isinstance(xml_string, str) or not xml_string.strip():
            raise ValueError('nonempty XML string required')
        with self._lock:
            self._open()
            with self._mutation():
                if self._rebuild is None:
                    raise UnsupportedGpuOperation("XML reset requires a model-only rebuild binding")
                with self._xml_build_rng():
                    fresh = self._rebuild(xml_string, self.components)
                self._replace_components(fresh)
                # Original XML replacement runs _initialize_sim.forward before
                # deterministic reset invokes sim.reset and its final forward.
                self.sim.forward()
                self.components.synchronize_and_check()
                self.shell.deterministic_reset = True
                try:
                    self.sim.reset()
                    self._reset_internal_gpu()
                    self.sim.forward()
                    self._disable_visual_sites()
                    self._get_observations(force_update=True)
                finally:
                    self.shell.deterministic_reset = False

    @load_only_entry
    def get_sim_state(self):
        with self._lock:
            self._open()
            return self.sim.get_state().flatten()

    @load_only_entry
    def set_state(self, value):
        with self._lock:
            self._open()
            value=self.sim.validate_flat_state(value)
            with self._mutation():
                self.sim.set_state_from_flattened(value)

    @load_only_entry
    def regenerate_obs_from_state(self, value):
        from libero.gpu.host.adapters.observations.boundary import boundary_predicates
        with self._lock:
            self._open()
            value = self.sim.validate_flat_state(value)
            with self._mutation():
                self.sim.set_state_from_flattened(value)
                self.sim.forward()
                self.components.synchronize_and_check()
                with boundary_predicates(self.shell, self.sim):
                    self.shell._check_success()
                    self.shell._post_process()
                return self._get_observations(force_update=True)

    @load_only_entry
    def _check_success(self):
        from libero.gpu.host.adapters.observations.boundary import boundary_predicates
        with self._lock:
            self._open()
            self.components.synchronize_and_check()
            with boundary_predicates(self.shell, self.sim): return self.shell._check_success()

    @load_only_entry
    def _post_process(self):
        from libero.gpu.host.adapters.observations.boundary import boundary_predicates
        with self._lock:
            self._open()
            self.components.synchronize_and_check()
            with boundary_predicates(self.shell, self.sim): return self.shell._post_process()
    @load_only_entry
    def step(self, action):
        from libero.gpu.host.adapters.observations.boundary import boundary_predicates
        with self._lock:
            self._open()
            if self.shell.done: raise ValueError("executing action in terminated episode")
            action = np.asarray(action, dtype=np.float64)
            if self.action_dim == 4 and action.ndim == 1 and action.size > 4:
                action = np.concatenate((action[:3], action[-1:]))
            if action.shape != (self.action_dim,) or not np.isfinite(action).all():
                raise ValueError("invalid policy action")
            with self._mutation():
                self.shell.timestep += 1
                self.sim.flush_public_arrays()
                self.components.stage_action(action)
                # Runtime's bound observe hook samples actual device fields after
                # each substep. Do not add one action-end sample here.
                self.components.physics.enqueue_action()
                self.components.synchronize_and_check()
                self.shell.cur_time += self.shell.control_timestep
                with boundary_predicates(self.shell, self.sim):
                    reward, _horizon_done, info = self.shell._post_action(action)
                    done = self.shell._check_success()
                return self._get_observations(), reward, done, info

    @load_only_entry
    def runtime_owner_evidence(self):
        with self._lock:
            self._open()
            return self.sim._call('runtime_owner_evidence')

    @load_only_entry
    def gpu_physics_evidence(self):
        with self._lock:
            self._open()
            return self.sim._call("gpu_physics_evidence")

    @load_only_entry
    def diagnostic_evidence(self):
        with self._lock:
            if self._closed:raise RuntimeError("GPU task environment is closed")
            reader=getattr(self.sim._runtime,"diagnostic_evidence",None)
            if not callable(reader):raise UnsupportedGpuOperation("actual failed-stage diagnostic reader unavailable")
            return {"diagnostic_only":True,"lifecycle_unusable":self._unusable,
                    "reset_pending":self._needs_reset,"last_failure":self.last_failure,
                    "actual":reader()}

    @load_only_entry
    def close(self):
        with self._lock:
            if not self._closed:
                try:
                    self.components.release()
                except BaseException as exc:
                    self._unusable=True
                    self.last_failure={'type':type(exc).__name__,'message':str(exc)}
                    raise
                self._closed = True
                self.components = None
                # Runtime close synchronized its stream. Drop retained controller,
                # observer and callback cycles before the next arena allocation.
                gc.collect()
