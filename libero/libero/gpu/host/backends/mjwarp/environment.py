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
    def __init__(self, bddl_file_name, num_envs=8, *, init_states=None, seed=0, scene_seed=None,
                 device='cuda:0', build_root=None, initialize=True, allow_compile=False,
                 debug=False, precision='fp32', physics_engine='mjwarp', mjwarp_options=None,
                 memory_budget=None, strict_build_receipts=False, ready_session=None,
                 profile_startup=False, algorithm_options=None,
                 **options):
        self.precision = scalar.select(precision)
        if self.precision != 'fp32' or type(num_envs) is not int or num_envs < 1:
            raise ValueError('MJWarp requires FP32 and a positive integer world count')
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
        from .options import expand
        controls,self.implementation_preset=expand(mjwarp_options or {})
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
        self.euler_implementation=controls.pop('euler','upstream')
        if self.euler_implementation not in ('upstream','cuda'):
            raise ValueError('euler must be upstream or cuda')
        self.transmission_implementation=controls.pop('transmission','upstream')
        self.actuator_velocity_implementation=controls.pop('actuator_velocity','upstream')
        if any(v not in ('upstream','cuda') for v in (self.transmission_implementation,self.actuator_velocity_implementation)):
            raise ValueError('transmission and actuator_velocity must be upstream or cuda')
        self.kinematics_implementation=controls.pop('kinematics','upstream')
        if self.kinematics_implementation not in ('upstream','cuda'):
            raise ValueError('kinematics must be upstream or cuda')
        self.com_pos_implementation=controls.pop('com_pos','upstream')
        if self.com_pos_implementation not in ('upstream','cuda'):
            raise ValueError('com_pos must be upstream or cuda')
        self.camlight_implementation=controls.pop('camlight','upstream')
        if self.camlight_implementation not in ('upstream','cuda'):
            raise ValueError('camlight must be upstream or cuda')
        self.fluid_implementation=controls.pop('fluid','upstream')
        if self.fluid_implementation not in ('upstream','cuda'):
            raise ValueError('fluid must be upstream or cuda')
        self.external_wrench_implementation=controls.pop('external_wrench','upstream')
        if self.external_wrench_implementation not in ('upstream','cuda'):
            raise ValueError('external_wrench must be upstream or cuda')
        self.constraint_build_implementation=controls.pop('constraint_build','upstream')
        if self.constraint_build_implementation not in ('upstream','cuda'):
            raise ValueError('constraint_build must be upstream or cuda')
        self.broadphase_implementation=controls.pop('broadphase','upstream')
        if self.broadphase_implementation not in ('upstream','cuda'):
            raise ValueError('broadphase must be upstream or cuda')
        self.plane_collision_implementation=controls.pop('plane_collision','upstream')
        if self.plane_collision_implementation not in ('upstream','cuda'):raise ValueError('plane_collision must be upstream or cuda')
        self.convex_collision_implementation=controls.pop('convex_collision','upstream')
        if self.convex_collision_implementation not in ('upstream','cuda'):raise ValueError('convex_collision must be upstream or cuda')
        self.box_collision_implementation=controls.pop('box_collision','upstream')
        if self.box_collision_implementation not in ('upstream','cuda'):raise ValueError('box_collision must be upstream or cuda')
        self.queries_implementation=controls.pop('queries','upstream')
        if self.queries_implementation not in ('upstream','cuda'):raise ValueError('queries must be upstream or cuda')
        self.publication_implementation=controls.pop('publication','upstream')
        if self.publication_implementation not in ('upstream','cuda'):raise ValueError('publication must be upstream or cuda')
        if self.publication_implementation=='cuda' and self.euler_implementation!='cuda':raise ValueError('CUDA publication requires CUDA Euler')
        self.public_forward_implementation=controls.pop('public_forward','upstream')
        if self.public_forward_implementation not in ('upstream','cuda'):raise ValueError('public_forward must be upstream or cuda')
        self.library_path = controller_product(self.build_root,arch=self.stream.device.arch, fast_compile=fast_compile)
        if options.get('has_renderer') or options.get('hard_reset'):
            raise ValueError('MJWarp batch uses a fixed model with soft reset')
        options.setdefault('use_camera_obs', False)
        options.update(has_renderer=False, has_offscreen_renderer=bool(options['use_camera_obs']), hard_reset=False)
        with self._cleanup_on_error(), wp.ScopedStream(self.stream):
            self.built, self.model, bank, self.task, self.identity = build_scene(
                self.build_root, bddl_file_name, scene_seed, options, collision=collision,prepare=allow_compile)
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
            self.identity.update(collision=collision, preset=self.implementation_preset, worlds=num_envs, capacities=self.capacities,public_forward=self.public_forward_implementation,
                                 schedule='upstream-step1-osc-step2', controller_initial_state='reset-state',
                                 warp=wp.__version__, fast_compile=fast_compile, arch=self.stream.device.arch)
            self.data = mjw.make_data(self.built.native_model, nworld=num_envs,
                                      nccdmax=self.capacities['nconmax'], **self.capacities)
            from .queries import Queries,queries_product
            queries_library=queries_product(self.build_root,arch=self.stream.device.arch,fast_compile=fast_compile) if self.queries_implementation=='cuda' else None
            queries=Queries(self.model,self.data,self.stream,implementation=self.queries_implementation,library=queries_library)
            self.identity['queries']=queries.identity
            from .newton import Newton, hessian_product, unsupported_reason
            hessian = (hessian_product(self.build_root,arch=self.stream.device.arch, fast_compile=fast_compile)
                       if self.newton_implementation in ('cuda_hessian', 'cuda_direction', 'cuda_constraint', 'cuda_linesearch', 'cuda_newton') and
                       unsupported_reason(self.model, self.data) is None else None)
            from libero.gpu.host.runtime.newton.mjwarp_direction import unsupported_reason as direction_reason
            direction = (hessian_product(self.build_root,arch=self.stream.device.arch, fast_compile=fast_compile, component='direction')
                         if self.newton_implementation in ('cuda_direction', 'cuda_constraint', 'cuda_linesearch', 'cuda_newton') and hessian is not None and
                         direction_reason(self.model) is None else None)
            constraint = (hessian_product(self.build_root,arch=self.stream.device.arch, fast_compile=fast_compile, component='constraint')
                          if self.newton_implementation in ('cuda_constraint', 'cuda_linesearch', 'cuda_newton') and direction is not None else None)
            from .newton import linesearch_reason
            linesearch = (hessian_product(self.build_root,arch=self.stream.device.arch, fast_compile=fast_compile, component='linesearch')
                          if self.newton_implementation in ('cuda_linesearch', 'cuda_newton') and constraint is not None and
                          linesearch_reason(self.model) is None else None)
            state = (hessian_product(self.build_root,arch=self.stream.device.arch, fast_compile=fast_compile, component='solver_state')
                     if self.newton_implementation == 'cuda_newton' and linesearch is not None else None)
            newton = Newton(self.model, self.data, self.stream,
                            implementation=self.newton_implementation, library=hessian,
                            direction_library=direction, constraint_library=constraint, linesearch_library=linesearch, state_library=state)
            from .acceleration import Acceleration, mass_product, caller_unsupported_reason
            mass_library = (mass_product(self.build_root,arch=self.stream.device.arch, fast_compile=fast_compile)
                            if self.mass_solve_implementation == 'cuda' and
                            caller_unsupported_reason(self.model, self.data) is None else None)
            force_library = (mass_product(self.build_root,arch=self.stream.device.arch, fast_compile=fast_compile, component='smooth_force')
                if self.acceleration_implementation=='cuda' and mass_library is not None else None)
            acceleration = Acceleration(self.model, self.data, self.stream,
                implementation=self.mass_solve_implementation, library=mass_library,
                force_implementation=self.acceleration_implementation, force_library=force_library)
            self.identity['mass_solve'] = acceleration.identity
            self.identity['newton'] = newton.identity
            from .mass_factor import MassFactors,factor_product,caller_unsupported_reason as factor_reason
            factor_library=(factor_product(self.build_root,arch=self.stream.device.arch,fast_compile=fast_compile)
                if self.mass_factor_implementation=='cuda' and factor_reason(self.model,self.data) is None else None)
            factors=MassFactors(self.model,self.data,self.stream,
                implementation=self.mass_factor_implementation,library=factor_library)
            self.identity['mass_factor']=factors.identity
            from .mass_matrix import MassMatrices,matrix_product,caller_unsupported_reason as matrix_reason
            matrix_library=(matrix_product(self.build_root,arch=self.stream.device.arch,fast_compile=fast_compile)
                if self.mass_matrix_implementation=='cuda' and matrix_reason(self.model,self.data) is None else None)
            matrices=MassMatrices(self.model,self.data,self.stream,
                implementation=self.mass_matrix_implementation,library=matrix_library)
            self.identity['mass_matrix']=matrices.identity
            from .velocity import Velocity,velocity_product,caller_unsupported_reason as velocity_reason
            motion_qualified=velocity_reason(self.model,self.data) is None
            velocity_library=(velocity_product(self.build_root,arch=self.stream.device.arch,fast_compile=fast_compile)
                if self.com_vel_implementation=='cuda' and motion_qualified else None)
            bias_library=(velocity_product(self.build_root,arch=self.stream.device.arch,component='bias',fast_compile=fast_compile)
                if self.bias_implementation=='cuda' and motion_qualified else None)
            from .fluid import Fluid,fluid_product,qualify as qualify_fluid
            fluid_admission=qualify_fluid(self.model,self.data,implementation=self.fluid_implementation)
            fluid_library=fluid_product(self.build_root,arch=self.stream.device.arch,fast_compile=fast_compile) if fluid_admission['selected']=='cuda' else None
            fluid=Fluid(self.model,self.data,self.stream,implementation=self.fluid_implementation,library=fluid_library,admission=fluid_admission)
            self.identity['fluid']=fluid.identity
            from .passive import Passive,caller_unsupported_reason as passive_reason
            passive_library=(velocity_product(self.build_root,arch=self.stream.device.arch,component='passive',fast_compile=fast_compile)
                if self.passive_implementation=='cuda' and passive_reason(self.model,self.data) is None else None)
            passive_driver=Passive(self.model,self.data,self.stream,
                implementation=self.passive_implementation,library=passive_library,fluid=fluid,queries=queries)
            self.identity['passive']=passive_driver.identity
            from .transmission import Transmission,transmission_product,qualify as qualify_transmission
            transmission_admission=qualify_transmission(self.model,self.data,transmission=self.transmission_implementation,
                actuator_velocity=self.actuator_velocity_implementation)
            transmission_library=(transmission_product(self.build_root,arch=self.stream.device.arch,fast_compile=fast_compile)
                if 'cuda' in transmission_admission['selected'].values() else None)
            transmission=Transmission(self.model,self.data,self.stream,transmission=self.transmission_implementation,
                actuator_velocity=self.actuator_velocity_implementation,library=transmission_library,admission=transmission_admission)
            self.identity['transmission']=transmission.identity
            velocity=Velocity(self.model,self.data,self.stream,com_vel=self.com_vel_implementation,
                bias=self.bias_implementation,velocity_library=velocity_library,bias_library=bias_library,passive_driver=passive_driver,transmission=transmission)
            self.identity['velocity']=velocity.identity
            from .external_wrench import ExternalWrench,external_wrench_product,qualify as qualify_external_wrench
            external_admission=qualify_external_wrench(self.model,self.data,implementation=self.external_wrench_implementation)
            external_library=external_wrench_product(self.build_root,arch=self.stream.device.arch,fast_compile=fast_compile) if external_admission['selected']=='cuda' else None
            external_wrench=ExternalWrench(self.model,self.data,self.stream,implementation=self.external_wrench_implementation,library=external_library,admission=external_admission)
            self.identity['external_wrench']=external_wrench.identity
            from .postconstraint import Postconstraint
            post_library=(velocity_product(self.build_root,arch=self.stream.device.arch,component='postconstraint',fast_compile=fast_compile)
                if self.postconstraint_implementation=='cuda' and motion_qualified else None)
            postconstraint=Postconstraint(self.model,self.data,self.stream,
                implementation=self.postconstraint_implementation,library=post_library,topology=velocity.topology,external_wrench=external_wrench)
            self.identity['postconstraint']=postconstraint.identity
            from .actuation import Actuation,actuation_product,qualify as qualify_actuation
            actuation_admission=qualify_actuation(self.model,self.data) if self.actuation_implementation=='cuda' else None
            actuation_library=(actuation_product(self.build_root,arch=self.stream.device.arch,fast_compile=fast_compile)
                if actuation_admission is not None and actuation_admission['reason'] is None else None)
            actuation=Actuation(self.model,self.data,self.stream,implementation=self.actuation_implementation,
                library=actuation_library,admission=actuation_admission)
            self.identity['actuation']=actuation.identity
            from .publication import Publication,publication_product,unsupported_reason as publication_reason
            publication_library=publication_product(self.build_root,arch=self.stream.device.arch,fast_compile=fast_compile) if self.publication_implementation=='cuda' and publication_reason(self.model) is None else None
            publication=Publication(self.model,self.data,self.stream,implementation=self.publication_implementation,library=publication_library)
            self.identity['publication']=publication.identity
            from .euler import Euler,euler_product,caller_unsupported_reason as euler_reason
            euler_library=(euler_product(self.build_root,arch=self.stream.device.arch,fast_compile=fast_compile)
                if self.euler_implementation=='cuda' and euler_reason(self.model,self.data) is None else None)
            euler=Euler(self.model,self.data,self.stream,mass_factor=factors,
                implementation=self.euler_implementation,library=euler_library,publication=publication)
            self.identity['euler']=euler.identity
            from .kinematics import Kinematics,kinematics_product,caller_unsupported_reason as kinematics_reason
            kinematics_library=(kinematics_product(self.build_root,arch=self.stream.device.arch,fast_compile=fast_compile)
                if self.kinematics_implementation=='cuda' and kinematics_reason(self.model,self.data) is None else None)
            kinematics=Kinematics(self.model,self.data,self.stream,implementation=self.kinematics_implementation,
                library=kinematics_library,topology=velocity.topology)
            self.identity['kinematics']=kinematics.identity
            from .com_pos import ComPos,com_pos_product,caller_unsupported_reason as com_pos_reason
            com_pos_library=(com_pos_product(self.build_root,arch=self.stream.device.arch,fast_compile=fast_compile)
                if self.com_pos_implementation=='cuda' and com_pos_reason(self.model,self.data) is None else None)
            com_pos=ComPos(self.model,self.data,self.stream,implementation=self.com_pos_implementation,library=com_pos_library)
            self.identity['com_pos']=com_pos.identity
            from .camlight import CamLight,camlight_product,qualify as qualify_camlight
            camlight_admission=qualify_camlight(self.model,self.data,implementation=self.camlight_implementation)
            camlight_library=(camlight_product(self.build_root,arch=self.stream.device.arch,fast_compile=fast_compile)
                if camlight_admission['selected']=='cuda' else None)
            camlight=CamLight(self.model,self.data,self.stream,implementation=self.camlight_implementation,
                library=camlight_library,admission=camlight_admission)
            self.identity['camlight']=camlight.identity
            from .constraint_build import ConstraintBuild,constraint_build_product,qualify as qualify_constraint_build
            constraint_admission=qualify_constraint_build(self.model,self.data,implementation=self.constraint_build_implementation)
            constraint_library=constraint_build_product(self.build_root,arch=self.stream.device.arch,fast_compile=fast_compile) if constraint_admission['selected']=='cuda' else None
            constraint_build=ConstraintBuild(self.model,self.data,self.stream,implementation=self.constraint_build_implementation,library=constraint_library,admission=constraint_admission)
            self.identity['constraint_build']=constraint_build.identity
            from .collision import Collision,broadphase_product,qualify as qualify_broadphase
            broadphase_admission=qualify_broadphase(self.model,self.data,implementation=self.broadphase_implementation)
            broadphase_library=broadphase_product(self.build_root,arch=self.stream.device.arch,fast_compile=fast_compile) if broadphase_admission['selected']=='cuda' else None
            from .collision import qualify_plane
            plane_admission=qualify_plane(self.model,self.data,implementation=self.plane_collision_implementation)
            plane_library=broadphase_product(self.build_root,arch=self.stream.device.arch,fast_compile=fast_compile,component='plane') if plane_admission['selected']=='cuda' else None
            from .collision import qualify_box
            box_admission=qualify_box(self.model,self.data,implementation=self.box_collision_implementation)
            box_library=broadphase_product(self.build_root,arch=self.stream.device.arch,fast_compile=fast_compile,component='box') if box_admission['selected']=='cuda' else None
            from .collision import qualify_convex
            convex_admission=qualify_convex(self.model,self.data,implementation=self.convex_collision_implementation)
            convex_library=broadphase_product(self.build_root,arch=self.stream.device.arch,fast_compile=fast_compile,component='convex') if convex_admission['selected']=='cuda' else None
            collision_driver=Collision(self.model,self.data,self.stream,broadphase=self.broadphase_implementation,library=broadphase_library,admission=broadphase_admission,planes=self.plane_collision_implementation,plane_library=plane_library,plane_admission=plane_admission,boxes=self.box_collision_implementation,box_library=box_library,box_admission=box_admission,convex=self.convex_collision_implementation,convex_library=convex_library,convex_admission=convex_admission)
            self.identity['broadphase']=collision_driver.identity
            self.identity['plane_collision']=collision_driver.plane_identity
            self.identity['box_collision']=collision_driver.box_identity
            self.identity['convex_collision']=collision_driver.convex_identity
            if not allow_compile:
                from .build import require_profile
                require_profile(self)
            self.observer = GpuObservationAdapter(self.built.native_model, self.built.metadata,
                                                  num_envs, device=self.stream.device)
            self._physics = self.runtime = Runtime(self, self.model, self.data, self.library_path,
                                                  newton=newton, acceleration=acceleration,
                                                  mass_factor=factors,mass_matrix=matrices,velocity=velocity,postconstraint=postconstraint,actuation=actuation,euler=euler,transmission=transmission,kinematics=kinematics,com_pos=com_pos,camlight=camlight,constraint_build=constraint_build,collision=collision_driver,queries=queries)
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
        if kwargs.get('staged'):
            raise ValueError('staged substeps require the native FP64 backend')
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
            self.runtime._forward_graph=None
            self.runtime._forward_graph_owners.clear()
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
