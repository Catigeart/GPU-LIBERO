"""LIBERO lifecycle over unmodified MJWarp physics and resident CUDA OSC."""
from pathlib import Path
from threading import RLock
from contextlib import contextmanager

import numpy as np
import warp as wp
import mujoco_warp as mjw

from libero.gpu import precision as scalar
from libero.gpu.host.adapters.environment.batch_env import GpuBatchEnv, _state_bank
from libero.gpu.host.adapters.environment.resident_runtime import ResidentField
from libero.gpu.host.adapters.observations.adapter import GpuObservationAdapter
from libero.gpu.host.build.compilation import construction_scope, install_warp_guard
from libero.gpu.host.build.project_paths import project_paths
from .model import build_scene, capacity_budget
from .controller import controller_product
from .runtime import Runtime


class MjWarpBatchEnv(GpuBatchEnv):
    physics_engine = 'mjwarp'

    @construction_scope
    def __init__(self, bddl_file_name, num_envs=8, *, init_states=None, seed=0,
                 device='cuda:0', build_root=None, initialize=True, allow_compile=False,
                 debug=False, precision='fp32', physics_engine='mjwarp', mjwarp_options=None,
                 memory_budget=None, strict_build_receipts=False, ready_session=None,
                 profile_startup=False, algorithm_options=None,
                 **options):
        self.precision = scalar.select(precision)
        if self.precision != 'fp32' or type(num_envs) is not int or not 1 <= num_envs <= 128:
            raise ValueError('MJWarp requires FP32 and 1..128 worlds')
        if ready_session is not None or algorithm_options is not None or memory_budget is not None:
            raise ValueError('native ready_session/algorithm_options/memory_budget are not MJWarp profiles')
        self.physics_engine = 'mjwarp'
        self.allow_compile, self.debug = allow_compile, debug
        self._lock = RLock()
        self._closed = self._failed = self._initialized = False
        self._snapshot_cache = None
        self._held_contacts = {}
        self._facades, self.shells, self._goal_plans = [], [], []
        self.num_envs = self.env_num = num_envs
        self.seed(seed)
        self.build_report = dict(events=[], physics_engine='mjwarp')
        self.build_root = project_paths(build_root).base
        # Warp 1.18 versions this directory during initialization. Assigning
        # config directly on a later environment would undo that resolution
        # and prepare modules in a different directory from a fresh process.
        from warp._src.build import init_kernel_cache
        init_kernel_cache(self.build_root / 'mjwarp' / 'warp-cache')
        install_warp_guard()
        wp.init()
        self.stream = wp.Stream(device)
        if self.stream.device.arch != 80:
            raise ValueError('the current shared CUDA OSC frontend requires SM80')
        controls = dict(mjwarp_options or {})
        fast_compile = bool(controls.pop('fast_compile', False))
        collision = controls.pop('collision', 'default')
        self.newton_implementation = controls.pop('newton', 'upstream')
        if self.newton_implementation not in ('upstream', 'cuda_hessian', 'cuda_direction', 'cuda_constraint', 'cuda_linesearch', 'cuda_newton'):
            raise ValueError('newton must be upstream, cuda_hessian, cuda_direction, cuda_constraint, cuda_linesearch or cuda_newton')
        self.acceleration_implementation = controls.pop('acceleration', 'upstream')
        if self.acceleration_implementation not in ('upstream', 'cuda'):
            raise ValueError('acceleration must be upstream or cuda')
        self.mass_solve_implementation = controls.pop('mass_solve',
            'cuda' if self.acceleration_implementation=='cuda' else 'upstream')
        if self.mass_solve_implementation not in ('upstream', 'cuda'):
            raise ValueError('mass_solve must be upstream or cuda')
        if self.acceleration_implementation=='cuda' and self.mass_solve_implementation!='cuda':
            raise ValueError('CUDA acceleration requires CUDA mass_solve')
        self.mass_factor_implementation=controls.pop('mass_factor','upstream')
        if self.mass_factor_implementation not in ('upstream','cuda'):
            raise ValueError('mass_factor must be upstream or cuda')
        self.mass_matrix_implementation=controls.pop('mass_matrix','upstream')
        if self.mass_matrix_implementation not in ('upstream','cuda'):
            raise ValueError('mass_matrix must be upstream or cuda')
        self.com_vel_implementation=controls.pop('com_vel','upstream')
        self.bias_implementation=controls.pop('bias','upstream')
        if self.com_vel_implementation not in ('upstream','cuda') or self.bias_implementation not in ('upstream','cuda'):
            raise ValueError('com_vel and bias must be upstream or cuda')
        self.postconstraint_implementation=controls.pop('postconstraint','upstream')
        if self.postconstraint_implementation not in ('upstream','cuda'):
            raise ValueError('postconstraint must be upstream or cuda')
        self.passive_implementation=controls.pop('passive','upstream')
        if self.passive_implementation not in ('upstream','cuda'):
            raise ValueError('passive must be upstream or cuda')
        self.actuation_implementation=controls.pop('actuation','upstream')
        if self.actuation_implementation not in ('upstream','cuda'):
            raise ValueError('actuation must be upstream or cuda')
        self.library_path = controller_product(self.build_root, fast_compile=fast_compile)
        if options.get('has_renderer') or options.get('hard_reset'):
            raise ValueError('MJWarp batch uses a fixed model with soft reset')
        options.setdefault('use_camera_obs', False)
        options.update(has_renderer=False, has_offscreen_renderer=bool(options['use_camera_obs']), hard_reset=False)
        with self._cleanup_on_error(), wp.ScopedStream(self.stream):
            self.built, self.model, bank, self.task, self.identity = build_scene(
                self.build_root, bddl_file_name, seed, options, collision=collision)
            self.initial_states = _state_bank(bank if init_states is None else init_states,
                                              self.model.nq, self.model.nv)
            self._nq, self._nv = self.model.nq, self.model.nv
            self.action_dim = 7
            self.control_freq = self.built.metadata['control_freq']
            self.substeps = int(self.built.metadata['control_timestep'] / self.built.metadata['model_timestep'])
            self.capacities = capacity_budget(self.built.native_model,
                nconmax=controls.pop('nconmax', None), njmax=controls.pop('njmax', None))
            if controls:
                raise ValueError('unknown MJWarp options: ' + ', '.join(controls))
            self.identity.update(collision=collision, worlds=num_envs, capacities=self.capacities,
                                 schedule='upstream-step1-osc-step2', controller_initial_state='reset-state',
                                 warp=wp.__version__, fast_compile=fast_compile, arch=self.stream.device.arch)
            self.data = mjw.make_data(self.built.native_model, nworld=num_envs,
                                      nccdmax=self.capacities['nconmax'], **self.capacities)
            from .newton import Newton, hessian_product, unsupported_reason
            hessian = (hessian_product(self.build_root, fast_compile=fast_compile)
                       if self.newton_implementation in ('cuda_hessian', 'cuda_direction', 'cuda_constraint', 'cuda_linesearch', 'cuda_newton') and
                       unsupported_reason(self.model, self.data) is None else None)
            from libero.gpu.host.runtime.newton.mjwarp_direction import unsupported_reason as direction_reason
            direction = (hessian_product(self.build_root, fast_compile=fast_compile, component='direction')
                         if self.newton_implementation in ('cuda_direction', 'cuda_constraint', 'cuda_linesearch', 'cuda_newton') and hessian is not None and
                         direction_reason(self.model) is None else None)
            constraint = (hessian_product(self.build_root, fast_compile=fast_compile, component='constraint')
                          if self.newton_implementation in ('cuda_constraint', 'cuda_linesearch', 'cuda_newton') and direction is not None else None)
            from .newton import linesearch_reason
            linesearch = (hessian_product(self.build_root, fast_compile=fast_compile, component='linesearch')
                          if self.newton_implementation in ('cuda_linesearch', 'cuda_newton') and constraint is not None and
                          linesearch_reason(self.model) is None else None)
            state = (hessian_product(self.build_root, fast_compile=fast_compile, component='solver_state')
                     if self.newton_implementation == 'cuda_newton' and linesearch is not None else None)
            newton = Newton(self.model, self.data, self.stream,
                            implementation=self.newton_implementation, library=hessian,
                            direction_library=direction, constraint_library=constraint, linesearch_library=linesearch, state_library=state)
            from .acceleration import Acceleration, mass_product, caller_unsupported_reason
            mass_library = (mass_product(self.build_root, fast_compile=fast_compile)
                            if self.mass_solve_implementation == 'cuda' and
                            caller_unsupported_reason(self.model, self.data) is None else None)
            force_library = (mass_product(self.build_root, fast_compile=fast_compile, component='smooth_force')
                if self.acceleration_implementation=='cuda' and mass_library is not None else None)
            acceleration = Acceleration(self.model, self.data, self.stream,
                implementation=self.mass_solve_implementation, library=mass_library,
                force_implementation=self.acceleration_implementation, force_library=force_library)
            self.identity['mass_solve'] = acceleration.identity
            self.identity['newton'] = newton.identity
            from .mass_factor import MassFactors,factor_product,caller_unsupported_reason as factor_reason
            factor_library=(factor_product(self.build_root,fast_compile=fast_compile)
                if self.mass_factor_implementation=='cuda' and factor_reason(self.model,self.data) is None else None)
            factors=MassFactors(self.model,self.data,self.stream,
                implementation=self.mass_factor_implementation,library=factor_library)
            self.identity['mass_factor']=factors.identity
            from .mass_matrix import MassMatrices,matrix_product,caller_unsupported_reason as matrix_reason
            matrix_library=(matrix_product(self.build_root,fast_compile=fast_compile)
                if self.mass_matrix_implementation=='cuda' and matrix_reason(self.model,self.data) is None else None)
            matrices=MassMatrices(self.model,self.data,self.stream,
                implementation=self.mass_matrix_implementation,library=matrix_library)
            self.identity['mass_matrix']=matrices.identity
            from .velocity import Velocity,velocity_product,caller_unsupported_reason as velocity_reason
            motion_qualified=velocity_reason(self.model,self.data) is None
            velocity_library=(velocity_product(self.build_root,fast_compile=fast_compile)
                if self.com_vel_implementation=='cuda' and motion_qualified else None)
            bias_library=(velocity_product(self.build_root,component='bias',fast_compile=fast_compile)
                if self.bias_implementation=='cuda' and motion_qualified else None)
            from .passive import Passive,caller_unsupported_reason as passive_reason
            passive_library=(velocity_product(self.build_root,component='passive',fast_compile=fast_compile)
                if self.passive_implementation=='cuda' and passive_reason(self.model,self.data) is None else None)
            passive_driver=Passive(self.model,self.data,self.stream,
                implementation=self.passive_implementation,library=passive_library)
            self.identity['passive']=passive_driver.identity
            velocity=Velocity(self.model,self.data,self.stream,com_vel=self.com_vel_implementation,
                bias=self.bias_implementation,velocity_library=velocity_library,bias_library=bias_library,passive_driver=passive_driver)
            self.identity['velocity']=velocity.identity
            from .postconstraint import Postconstraint
            post_library=(velocity_product(self.build_root,component='postconstraint',fast_compile=fast_compile)
                if self.postconstraint_implementation=='cuda' and motion_qualified else None)
            postconstraint=Postconstraint(self.model,self.data,self.stream,
                implementation=self.postconstraint_implementation,library=post_library,topology=velocity.topology)
            self.identity['postconstraint']=postconstraint.identity
            from .actuation import Actuation,actuation_product,qualify as qualify_actuation
            actuation_admission=qualify_actuation(self.model,self.data) if self.actuation_implementation=='cuda' else None
            actuation_library=(actuation_product(self.build_root,fast_compile=fast_compile)
                if actuation_admission is not None and actuation_admission['reason'] is None else None)
            actuation=Actuation(self.model,self.data,self.stream,implementation=self.actuation_implementation,
                library=actuation_library,admission=actuation_admission)
            self.identity['actuation']=actuation.identity
            if not allow_compile:
                from .build import require_profile
                require_profile(self)
            self.observer = GpuObservationAdapter(self.built.native_model, self.built.metadata,
                                                  num_envs, device=self.stream.device)
            self._physics = self.runtime = Runtime(self, self.model, self.data, self.library_path,
                                                  newton=newton, acceleration=acceleration,
                                                  mass_factor=factors,mass_matrix=matrices,velocity=velocity,postconstraint=postconstraint,actuation=actuation)
            self._all = np.ones(num_envs, np.int32)
            self._selection = wp.array(self._all, dtype=int, device=self.stream.device)
            self._fields = {name: ResidentField(array, array.shape[1:]) for name, array in self.observation_fields().items()}
            self._bind_shells()
            if initialize:
                self.reset(init_states=self.initial_states[np.arange(num_envs) % len(self.initial_states)])
                self.runtime.prepare_control_graph()

    @contextmanager
    def _cleanup_on_error(self):
        try:
            yield
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:
                error.gpu_cleanup_errors = [repr(cleanup)]
            raise

    def _bind_shells(self):
        super()._bind_shells()
        from .facade import bind_model_compatibility
        for facade in self._facades:
            bind_model_compatibility(facade.model)

    def observation_fields(self):
        if hasattr(self, '_raw_fields'):
            return self._raw_fields
        d, n = self.data, self.num_envs
        def alias(array, width):
            return wp.array(ptr=array.ptr, shape=(n, width), dtype=float, device=self.stream.device)
        m = self.built.native_model
        self._raw_fields = dict(qpos=d.qpos, qvel=d.qvel, body_xpos=alias(d.xpos, m.nbody * 3),
                    body_xquat=alias(d.xquat, m.nbody * 4), body_xmat=alias(d.xmat, m.nbody * 9),
                    geom_xpos=alias(d.geom_xpos, m.ngeom * 3), geom_xmat=alias(d.geom_xmat, m.ngeom * 9),
                    site_xpos=alias(d.site_xpos, m.nsite * 3), site_xmat=alias(d.site_xmat, m.nsite * 9),
                    subtree_com=alias(d.subtree_com, m.nbody * 3))
        return self._raw_fields

    def prepare_substep_fusion(self, build_root, **kwargs):
        if kwargs.get('profile_newton') or kwargs.get('device_lto'):
            raise ValueError('native substep instrumentation is not an MJWarp backend option')
        return self.runtime.prepare_control_graph()

    def _check(self):
        wp.synchronize_stream(self.stream)
        overflow = self.data.overflow.numpy()
        errors = self.runtime.controller.buffers['error'].numpy()
        from mujoco_warp._src.types import OverflowType
        limits = int(OverflowType.ITERATIONS | OverflowType.LS_ITERATIONS)
        fatal = overflow & ~limits
        # Iteration/line-search limits are legal solver exits in CPU 2.3.7
        # and upstream MJWarp. Preserve their warning bits in diagnostics;
        # numerical checks judge their residuals rather than treating a limit
        # as a truncated contact/constraint allocation.
        if np.any(fatal) or np.any(errors):
            self._failed = True
            raise RuntimeError(dict(mjwarp_overflow=overflow.tolist(), controller_error=errors.tolist()))

    def _restore(self, rows, ids, *, reset):
        if not ids:
            return
        with wp.ScopedStream(self.stream):
            from . import kernels
            selected = set(ids)
            mask = np.zeros(self.num_envs, np.int32)
            mask[list(ids)] = 1
            self._selection.assign(mask)
            partial = len(ids) != self.num_envs
            # Upstream forward processes all worlds. In particular it refreshes
            # pre-Euler poses/sensors in worlds whose roots did not change.
            # A selected reset must hold those worlds' public derived state,
            # not merely their roots and Newton warmstart.
            held_names = ('qacc', 'qacc_warmstart', 'qfrc_constraint', 'qfrc_bias',
                'qfrc_smooth', 'cfrc_ext', 'cfrc_int', 'sensordata', 'M', 'xpos',
                'xquat', 'xmat', 'geom_xpos', 'geom_xmat', 'site_xpos', 'site_xmat',
                'subtree_com')
            preserved = {name: wp.clone(getattr(self.data, name)) for name in held_names} if partial else {}
            contacts = self._snapshot(('ncon', 'contact')) if partial else None
            for world in ids:
                self._held_contacts.pop(world, None)
            transfers = []
            fields = dict(time=rows[:, :1], qpos=rows[:, 1:1 + self._nq], qvel=rows[:, 1 + self._nq:])
            if reset:
                fields.update(ctrl=np.zeros((len(ids), self.model.nu)),
                              qacc_warmstart=np.zeros((len(ids), self._nv)),
                              qfrc_applied=np.zeros((len(ids), self._nv)),
                              xfrc_applied=np.zeros((len(ids), self.built.native_model.nbody * 6)))
            for name, values in fields.items():
                target = getattr(self.data, name)
                source = wp.array(np.asarray(values, np.float32).reshape(-1), dtype=float, device=self.stream.device)
                transfers.append(source)
                flat = wp.array(ptr=target.ptr, shape=(target.size * (6 if name == 'xfrc_applied' else 1),),
                                dtype=float, device=self.stream.device)
                width = flat.size // self.num_envs
                for index, world in enumerate(ids):
                    wp.copy(flat, source, dest_offset=world * width, src_offset=index * width,
                            count=width, stream=self.stream)
            wp.launch(kernels.state_written, self.num_envs,
                      [self._selection, self.runtime.state_epoch, self.runtime.reset_events, int(reset)],
                      stream=self.stream)
            self.runtime.forward()
            for name, original in preserved.items():
                target = getattr(self.data, name)
                width = target.size // self.num_envs
                for world in range(self.num_envs):
                    if world not in selected:
                        wp.copy(target.flatten(), original.flatten(), dest_offset=world * width,
                                src_offset=world * width, count=width, stream=self.stream)
            if partial:
                # Contact storage is pooled, not world-major. Hold only its
                # public read view; never splice slots used by upstream physics.
                for world in range(self.num_envs):
                    if world not in selected:
                        self._held_contacts[world] = (int(contacts['ncon'][world]),
                            {k: v.copy() for k, v in contacts['contact'][world].items()})
            if reset:
                self.runtime.controller.bootstrap(self._selection)
                self.observer.reset(self._selection, stream=self.stream)
            self._check()
            names, schemas = self._goal_fields(ids)
            self._snapshot_cache = self._snapshot(names)
            from libero.gpu.host.adapters.observations.boundary import boundary_predicates
            try:
                for offset, world in enumerate(ids):
                    with boundary_predicates(self.shells[world], self._facades[world], fields=schemas[offset]):
                        self.shells[world]._check_success()
                        self.shells[world]._post_process()
            finally:
                self._snapshot_cache = None
            self.observer.sample(self.observation_fields(), force=True, world_mask=self._selection, stream=self.stream)
            self._check()

    def _snapshot(self, names=None):
        from libero.gpu.host.adapters.observations.boundary import GOAL_FIELDS
        names = GOAL_FIELDS if names is None else names
        raw = self.observation_fields()
        shaped = {'body_xpos': (self.built.native_model.nbody, 3),
                  'body_xquat': (self.built.native_model.nbody, 4),
                  'body_xmat': (self.built.native_model.nbody, 9),
                  'geom_xpos': (self.built.native_model.ngeom, 3),
                  'geom_xmat': (self.built.native_model.ngeom, 9),
                  'site_xpos': (self.built.native_model.nsite, 3),
                  'site_xmat': (self.built.native_model.nsite, 9),
                  'subtree_com': (self.built.native_model.nbody, 3)}
        aliases = dict(xpos='body_xpos', xquat='body_xquat', xmat='body_xmat')
        roots = {'time', 'qpos', 'qvel', 'qacc', 'qacc_warmstart', 'ctrl', 'act',
                 'qfrc_applied', 'xfrc_applied', 'qfrc_bias', 'qfrc_smooth',
                 'qfrc_constraint', 'cfrc_ext', 'cfrc_int', 'sensordata'}
        result = {}
        for name in names:
            key = aliases.get(name, name)
            if key in raw:
                result[name] = raw[key].numpy().reshape((self.num_envs,) + shaped[key]) if key in shaped else raw[key].numpy()
            elif name in roots:
                result[name] = getattr(self.data, name).numpy()
        if 'ncon' in names or 'contact' in names:
            size = min(int(self.data.nacon.numpy()[0]), self.data.naconmax)
            worlds = self.data.contact.worldid.numpy()[:size]
            geoms = self.data.contact.geom.numpy()[:size]
            from mujoco_warp._src.types import ContactType
            kinds = self.data.contact.type.numpy()[:size]
            exported = (kinds & int(ContactType.CONSTRAINT | ContactType.PASSIVE)) != 0
            worlds, geoms = worlds[exported], geoms[exported]
            result['ncon'] = np.bincount(worlds, minlength=self.num_envs)
            result['contact'] = [dict(geom1=geoms[worlds == w, 0], geom2=geoms[worlds == w, 1]) for w in range(self.num_envs)]
            for world, (count, contact) in self._held_contacts.items():
                result['ncon'][world] = count
                result['contact'][world] = {k: v.copy() for k, v in contact.items()}
        missing = set(names) - set(result)
        if missing:
            raise KeyError('MJWarp boundary fields unavailable: ' + ', '.join(sorted(missing)))
        return result

    def diagnostics(self):
        self._check()
        return dict(physics_backend='gpu', physics_engine='mjwarp', precision='fp32',
                    actual_integration_count=self.runtime.integration_count.numpy().tolist(),
                    first_error=self.runtime.controller.buffers['error'].numpy().tolist(),
                    substeps=self.substeps, identity=self.identity, capacities=self.capacities,
                    upstream_status=self.data.overflow.numpy().tolist())

    def work_snapshot(self):
        result = {name: getattr(self.runtime, name).numpy().tolist() for name in
                  ('integration_count', 'state_epoch', 'reset_events', 'newton_attempts', 'position_math')}
        result['first_error'] = self.runtime.controller.buffers['error'].numpy().tolist()
        result['upstream_status'] = self.data.overflow.numpy().tolist()
        return result

    def validation_endpoint(self, observations, rewards, successes):
        result = self.runtime.download_state()
        for name in ('qfrc_bias', 'qfrc_smooth', 'qfrc_constraint', 'cfrc_ext', 'cfrc_int'):
            result[name] = getattr(self.data, name).numpy().reshape(self.num_envs, -1)
        fields = self.observation_fields()
        for name in ('body_xpos', 'body_xquat', 'site_xpos', 'site_xmat'):
            result[name] = fields[name].numpy()
        result['sensor'] = self.data.sensordata.numpy()
        result['ncon'] = self._snapshot(('ncon',))['ncon']
        result['nefc'] = self.data.nefc.numpy()
        # Validation uses a basis map exported by actual CPU MuJoCo 2.3.7.
        # Converting an endpoint's stored M requires no CPU physical update.
        layout = np.load(self.build_root / 'mjwarp' / 'mass-layouts' /
                         (Path(self.task['directory']).name + '.npz'))
        old_indices = layout['indices']
        source_indices = np.empty(int(layout['nM']), np.int32)
        for k in range(len(source_indices)):
            i, j = np.argwhere(old_indices == k)[0]
            source_indices[k] = self.runtime.controller.mass_indices_host[i, j]
        if np.any(source_indices < 0):
            raise ValueError('MJWarp mass layout does not cover the CPU packed matrix')
        result['qM'] = self.data.M.numpy()[:, source_indices]
        result.update({'obs.' + key: value for key, value in observations.items()})
        result['rewards_trace'] = np.asarray(rewards).T
        result['successes'] = np.asarray(successes)
        return result

    def close(self):
        if getattr(self, '_closed', True):
            return
        if hasattr(self, 'stream'):
            wp.synchronize_stream(self.stream)
        if getattr(self, 'observer', None) is not None:
            self.observer.close()
        for facade in self._facades:
            facade.close()
        if getattr(self, 'runtime', None) is not None:
            self.runtime._control_graph = self.runtime._substep_fusion = None
            self.runtime._control_graph_owners.clear()
            self.runtime.env = None
        self._facades.clear()
        self.shells.clear()
        self._goal_plans.clear()
        self._held_contacts.clear()
        self._fields = {}
        if hasattr(self, '_raw_fields'):
            self._raw_fields.clear()
        self._selection = self.data = self.model = self.observer = self.runtime = self._physics = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
