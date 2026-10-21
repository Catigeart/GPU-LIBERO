"""LIBERO lifecycle backed entirely by owned FP32 CUDA resources."""
from pathlib import Path
from threading import RLock
from contextlib import contextmanager
import numpy as np
from libero.gpu import precision as scalar
from libero.gpu.host.adapters.environment.batch_env import GpuBatchEnv,_state_bank
from libero.gpu.host.adapters.environment.resident_runtime import ResidentField
from .observations import ObservationAdapter
from libero.gpu.host.build.compilation import construction_scope
from libero.gpu.host.build.project_paths import project_paths
from . import device as cuda
from .model import prepare,upload
from .data import make_data
from .scene import build_scene,capacity_budget
from .runtime import Runtime


class CudaFP32BatchEnv(GpuBatchEnv):
    physics_engine='cuda_fp32'

    @construction_scope
    def __init__(self,bddl_file_name,num_envs=8,*,init_states=None,seed=0,device='cuda:0',
            build_root=None,initialize=True,allow_compile=False,debug=False,precision='fp32',
            physics_engine='cuda_fp32',cuda_options=None,mjwarp_options=None,memory_budget=None,
            strict_build_receipts=False,ready_session=None,profile_startup=False,algorithm_options=None,**options):
        self.precision=scalar.select(precision)
        if self.precision!='fp32' or type(num_envs) is not int or not 1<=num_envs<=128:
            raise ValueError('Independent FP32 requires 1..128 worlds')
        if ready_session is not None or algorithm_options is not None or memory_budget is not None:
            raise ValueError('Legacy native ready/algorithm/memory-budget profiles are not independent FP32 options')
        controls=dict(cuda_options or {})
        if mjwarp_options:
            old=dict(mjwarp_options);preset=old.pop('preset','cuda_libero')
            if preset!='cuda_libero':raise ValueError('Independent FP32 has no upstream numerical fallback')
            controls.update(old)
        fast=bool(controls.pop('fast_compile',True));collision=controls.pop('collision','default')
        self.physics_engine='cuda_fp32';self.allow_compile=allow_compile;self.debug=debug
        self._lock=RLock();self._closed=self._failed=self._initialized=False
        self._snapshot_cache=None;self._held_contacts={};self._facades=[];self.shells=[];self._goal_plans=[]
        self.num_envs=self.env_num=num_envs;self.seed(seed);self._integrations=np.zeros(num_envs,np.int64)
        self.build_root=project_paths(build_root).base;self.build_report=dict(events=[],physics_engine=self.physics_engine)
        folder=self.build_root/'cmake-control'/('sm80-dev' if fast else 'sm80-release')
        self.cuda_runtime=cuda.Runtime(folder/'cuda_runtime.so')
        if not str(device).startswith('cuda:'):raise ValueError('A CUDA device is required')
        self.stream=cuda.Stream(self.cuda_runtime.device(int(str(device).split(':')[1])))
        options.setdefault('use_camera_obs',False)
        if options.get('has_renderer') or options.get('hard_reset'):raise ValueError('Fixed-model offscreen/soft-reset batch required')
        options.update(has_renderer=False,has_offscreen_renderer=bool(options['use_camera_obs']),hard_reset=False)
        with self._cleanup_on_error(),self.stream:
            self.built,bank,self.task,self.identity=build_scene(self.build_root,bddl_file_name,seed,options,collision=collision,prepare=allow_compile)
            self.initial_states=_state_bank(bank if init_states is None else init_states,self.built.native_model.nq,self.built.native_model.nv)
            self.model=upload(prepare(self.built.native_model));self._nq=self.model.nq;self._nv=self.model.nv
            self.action_dim=7;self.control_freq=self.built.metadata['control_freq']
            self.substeps=int(self.built.metadata['control_timestep']/self.built.metadata['model_timestep'])
            self.capacities=capacity_budget(self.built.native_model,nconmax=controls.pop('nconmax',None),njmax=controls.pop('njmax',None))
            if controls:raise ValueError('Unknown independent FP32 options: '+str(sorted(controls)))
            self.data=make_data(self.built.native_model,self.model,num_envs,self.capacities['nconmax'],self.capacities['njmax'])
            self.identity.update(worlds=num_envs,capacities=self.capacities,physics_engine=self.physics_engine,
                warp_dependency=False,schedule='step1-osc-step2',arch=self.stream.device.arch)
            self.observer=ObservationAdapter(self.built.native_model,self.built.metadata,num_envs,device=self.stream.device)
            self._physics=self.runtime=Runtime(self,fast_compile=fast)
            self._all=np.ones(num_envs,np.int32);self._selection=cuda.array(self._all,dtype=int)
            self._fields={k:ResidentField(v,v.shape[1:]) for k,v in self.observation_fields().items()}
            self._bind_shells()
            if initialize:
                self.reset(init_states=self.initial_states[np.arange(num_envs)%len(self.initial_states)])
                self.runtime.prepare_control_graph()

    def _activate_all(self):
        with self.stream:self.runtime.intent.assign(self._all)


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
        from ..mjwarp.facade import bind_model_compatibility
        for world,facade in enumerate(self._facades):
            bind_model_compatibility(facade.model)
            facade.bind_renderer(lambda world=world,**options:self._render_world(world,**options),
                context_provider=self.observer.public_render_context)

    def _render_world(self,world,**options):
        self._open()
        options={name:value for name,value in options.items() if value is not None}
        with self.stream:
            return self.observer.render_gpu(self.observation_fields(),world=world,stream=self.stream,**options)


    def observation_fields(self):
        if hasattr(self, '_raw_fields'):
            return self._raw_fields
        d, n = self.data, self.num_envs
        def alias(array, width):
            return cuda.array(ptr=array.ptr, shape=(n, width), dtype=float, device=self.stream.device, owner=array)
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
        cuda.synchronize_stream(self.stream)
        overflow = self.data.overflow.numpy()
        errors = self.runtime.controller.buffers['error'].numpy()
        if self.debug:self._integrations=self.runtime.integration_count.numpy()
        from .constants import OverflowType
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
        with cuda.ScopedStream(self.stream):
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
            preserved = {name: cuda.clone(getattr(self.data, name)) for name in held_names} if partial else {}
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
                source = cuda.array(np.asarray(values, np.float32).reshape(-1), dtype=float, device=self.stream.device)
                transfers.append(source)
                flat = cuda.array(ptr=target.ptr, shape=(target.size * (6 if name == 'xfrc_applied' else 1),),
                                dtype=float, device=self.stream.device, owner=target)
                width = flat.size // self.num_envs
                for index, world in enumerate(ids):
                    cuda.copy(flat, source, dest_offset=world * width, src_offset=index * width,
                            count=width, stream=self.stream)
            self.runtime.count(3 if reset else 2,self._selection)
            self.runtime.forward()
            for name, original in preserved.items():
                target = getattr(self.data, name)
                width = target.size // self.num_envs
                for world in range(self.num_envs):
                    if world not in selected:
                        cuda.copy(target.flatten(), original.flatten(), dest_offset=world * width,
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
            from .constants import ContactType
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
        return dict(physics_backend='gpu', physics_engine='cuda_fp32', precision='fp32',
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


    def __enter__(self):
        return self


    def __exit__(self, *args):
        self.close()


    def close(self):
        if getattr(self,'_closed',True):return
        if hasattr(self,'stream'):self.stream.synchronize()
        if getattr(self,'runtime',None) is not None:self.runtime.close()
        if getattr(self,'observer',None) is not None:self.observer.close()
        for facade in self._facades:facade.close()
        self._facades.clear();self.shells.clear();self._goal_plans.clear();self._held_contacts.clear()
        self._fields={};self._raw_fields={}
        if getattr(self,'runtime',None) is not None:self.runtime.env=None
        self._selection=self.data=self.model=self.observer=self.runtime=self._physics=None
        if hasattr(self,'stream'):self.stream.close()
        self._closed=True
