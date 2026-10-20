"""Runtime-facing observation sampling and allowed boundary packaging."""
import libero.gpu.precision as scalar
from contextlib import nullcontext
import numpy as np
import warp as wp
from .schema import plan_from_metadata, pack_observations
from .device import DeviceObservations, DeviceSampledScenes
from .packing import DeviceFieldPacker
from .scene_kinematics import DeviceSceneTransforms
from .boundary import DerivedRenderBridge, RENDER_FIELDS


def plan_from_builder_metadata(metadata):
    robots=metadata["robot_metadata"]
    if len(robots)!=1:raise NotImplementedError("default LIBERO single robot profile only")
    robot=robots[0]
    cameras=[]
    if metadata["use_camera_obs"]:
        for i,name in enumerate(metadata["camera_names"]):
            cameras.append({"name":name,"height":metadata["camera_heights"][i],"width":metadata["camera_widths"][i],
                            "depth":metadata["camera_depths"][i],"segmentation":metadata["camera_segmentations"][i]})
    return plan_from_metadata(**{key:robot[key] for key in ("prefix","joint_qpos_indices","joint_qvel_indices",
                             "gripper_qpos_indices","gripper_qvel_indices","eef_site_id","eef_body_id")},
                             object_body_ids=metadata["ordered_object_body_ids"],control_freq=metadata["control_freq"],
                             cameras=cameras,use_object_obs=metadata["use_object_obs"])


def _layout(lengths):
    result={};at=0
    for name,length in lengths.items():result[name]=(at,length);at+=length
    return result

class GpuObservationAdapter:
    """No host read/physics call in sample(); host packaging only at a boundary.

    The scheduler owns input publications, stream ordering, and per-world validity.
    raw_fields can have intentionally different state epochs. No extra forward is
    requested to align geometry with post-integrator qpos/qvel.
    """
    def __init__(self,native_model,metadata,nworld=1,device="cuda:0",scene_option=None):
        self.model=native_model;self.metadata=metadata;self.nworld=nworld;self.device=device
        self.closed=False;self.valid=True;self.visual_generation=0
        self.host_environments={};self.host_observables={};self._last_sample=None
        self._public_bridge=None;self._pending_renderers=set();self._owned_viewers={}
        self.plan=plan_from_builder_metadata(metadata)
        raw_lengths={"qpos":int(native_model.nq),"qvel":int(native_model.nv),
                     "body_xpos":int(native_model.nbody)*3,"body_xquat":int(native_model.nbody)*4,
                     "site_xpos":int(native_model.nsite)*3}
        raw_layout=_layout(raw_lengths)
        self.raw_packer=DeviceFieldPacker(raw_layout,nworld,device)
        self.lowdim=DeviceObservations(self.plan,raw_layout,nworld,device)
        if scene_option is None and self.plan.image_specs:
            import mujoco
            scene_option=mujoco.MjvOption()
            scene_option.geomgroup[0]=int(metadata.get("render_collision_mesh",False))
            scene_option.geomgroup[1]=int(metadata.get("render_visual_mesh",True))
        self.scene_option=scene_option
        self.scene_transforms=None;self.scene_packer=None;self.scene_cache=None;self.bridges={}
        self.visual_sources={}
        for name in ("site_rgba","geom_rgba"):
            value=np.asarray(getattr(native_model,name),dtype=np.float64).reshape(-1)
            self.visual_sources[name]=wp.array(np.tile(value,nworld),dtype=scalar.wp_real,device=device)
        if self.plan.image_specs:
            self.scene_transforms=DeviceSceneTransforms(native_model,nworld,device)
            shapes={"xpos":(int(native_model.nbody),3),"xmat":(int(native_model.nbody),9),
                    "geom_xpos":(int(native_model.ngeom),3),"geom_xmat":(int(native_model.ngeom),9),
                    "site_xpos":(int(native_model.nsite),3),"site_xmat":(int(native_model.nsite),9),
                    "cam_xpos":(int(native_model.ncam),3),"cam_xmat":(int(native_model.ncam),9),
                    "light_xpos":(int(native_model.nlight),3),"light_xdir":(int(native_model.nlight),3),
                    "subtree_com":(int(native_model.nbody),3),"site_rgba":tuple(native_model.site_rgba.shape),
                    "geom_rgba":tuple(native_model.geom_rgba.shape)}
            lengths={name:int(np.prod(shape)) for name,shape in shapes.items()}
            packed_layout=_layout(lengths)
            self.scene_packer=DeviceFieldPacker(packed_layout,nworld,device)
            scene_layout={name:(packed_layout[name][0],shape) for name,shape in shapes.items()}
            self.scene_cache=DeviceSampledScenes(self.plan.image_specs,scene_layout,nworld,device,metadata.get("segmentation_mappings"))
        self._host_sampling_required=False
        self._host_sampling_reasons=[]
        self._sample_event_token=0
        self._sample_event_hook=None
        self._graph_guard=None
        self._graph_configuration=self._sampling_graph_signature()

    @staticmethod
    def _array_signature(array):
        return (id(array),int(array.ptr),array.dtype,tuple(array.shape),
                tuple(array.strides),str(array.device),bool(array.is_contiguous),
                getattr(array,'capacity',None))

    def _sampling_graph_signature(self):
        """Host-only capture contract; enabled/rate values live behind stable roots."""
        low=self.lowdim
        arrays=list(low.meta)+[getattr(low,name) for name in
            ('cache','observed','sidequat','elapsed','sampled','present','events','all_worlds')]
        arrays.append(self.raw_packer.buffer)
        if self.scene_cache is not None:
            arrays.extend(getattr(self.scene_cache,name) for name in
                ('periods','enabled','elapsed','sampled','generation','parents',
                 'parent_generation','captured','scenes','mask'))
            arrays.append(self.scene_packer.buffer)
        arrays.extend(self.visual_sources[name] for name in sorted(self.visual_sources))
        structure=tuple((s.name,s.modality,s.operation,s.width,tuple(s.indices),
                         tuple(s.dependencies),tuple(s.shape)) for s in self.plan.specs)
        images=tuple(tuple(sorted((key,repr(value)) for key,value in spec.items()
                                  if key not in ('enabled','active','period')))
                     for spec in self.plan.image_specs)
        layout=(id(low),id(self.raw_packer),repr(getattr(self.raw_packer,'layout',None)),
                getattr(low,'source_stride',None),id(self.scene_cache),id(self.scene_packer),
                getattr(self.scene_cache,'ncamera',None),getattr(self.scene_cache,'stride',None),
                repr(getattr(self.scene_packer,'layout',None)))
        return (self.nworld,float(self.metadata['model_timestep']),tuple(self.plan.order),layout,
                structure,images,tuple(self._array_signature(a) for a in arrays))

    @property
    def requires_host_sampling(self):
        return bool(self._host_sampling_required or self.host_observables or self._pending_renderers)

    @property
    def graph_eligible(self):
        """Read-only guard checked before capture and replay; never reads GPU data."""
        if self.closed or not self.valid or self.requires_host_sampling:
            return False
        try:
            if self._sampling_graph_signature()!=self._graph_configuration:
                return False
            guard=self._graph_guard
            return guard is None or bool(guard())
        except (AttributeError,KeyError,TypeError,ValueError,RuntimeError):
            return False

    def bind_graph_guard(self,callback):
        if callback is not None and not callable(callback):
            raise TypeError('observation graph guard must be callable')
        self._graph_guard=callback

    def bind_sample_event_hook(self,callback):
        if callback is not None and not callable(callback):
            raise TypeError('observation sample event hook must be callable')
        self._sample_event_hook=callback

    @property
    def sample_event_token(self):
        """Ordered sample submission id; host readers retain their completion wait."""
        return self._sample_event_token

    def _require_host_sampling(self,reason):
        self._host_sampling_required=True
        if reason not in self._host_sampling_reasons:self._host_sampling_reasons.append(reason)

    def assert_sampling_allowed(self,stream=None):
        if self.closed or not self.valid:
            raise RuntimeError('observation adapter is closed or quarantined')
        if stream is not None and getattr(stream,'is_capturing',False) and not self.graph_eligible:
            raise RuntimeError('observation sampling requires uncaptured host callbacks or current bindings')

    def _record_sample_event(self,timestep,force=False,events=1):
        self._last_sample=(float(timestep),bool(force))
        self._sample_event_token+=events
        if self._sample_event_hook is not None:self._sample_event_hook(self._sample_event_token)

    def notify_control_graph_replay(self,timestep=None,substeps=1,stream=None):
        """Record actual replay submission, never graph construction or a host sample."""
        self.assert_sampling_allowed(stream)
        if stream is not None and getattr(stream,'is_capturing',False):
            raise RuntimeError('cannot notify observation replay during graph capture')
        if type(substeps) is not int or substeps<0:raise ValueError('nonnegative substep count required')
        if substeps==0:return
        if not self.graph_eligible:raise RuntimeError('observation graph is no longer eligible')
        dt=self.metadata['model_timestep'] if timestep is None else timestep
        if not np.isfinite(dt) or dt<=0:raise ValueError('sampling timestep must be positive finite')
        self.lowdim.last_stream=stream or wp.get_stream(self.device)
        if self.scene_cache is not None:self.scene_cache.last_stream=self.lowdim.last_stream
        self._record_sample_event(dt,False,substeps)

    def _synchronize_sampling(self,stream=None):
        streams=(stream,self.lowdim.last_stream,
                 None if self.scene_cache is None else self.scene_cache.last_stream)
        seen=set()
        for value in streams:
            if value is not None and id(value) not in seen:
                wp.synchronize_stream(value);seen.add(id(value))


    @staticmethod
    def _flat(fields):
        aliases={"xpos":"body_xpos","xquat":"body_xquat","xmat":"body_xmat"}
        result={name:value.flatten() if value.ndim!=1 else value for name,value in fields.items()}
        for name,alias in aliases.items():
            if name in result and alias not in result:result[alias]=result[name]
            if alias in result and name not in result:result[name]=result[alias]
        return result

    def sample(self,raw_fields,timestep=None,force=False,world_mask=None,stream=None):
        self.assert_sampling_allowed(stream)
        fields=self._flat(raw_fields)
        timestep=self.metadata["model_timestep"] if timestep is None else timestep
        # Verify all required fields before enqueueing any sampler update.
        for name in self.raw_packer.layout:
            if name not in fields:raise KeyError(f"missing observable publication: {name}")
        if self.scene_cache is not None:
            for name in ("body_xpos","body_xmat","geom_xpos","geom_xmat","site_xpos","site_xmat","subtree_com"):
                if name not in fields:raise KeyError(f"missing camera-scene publication: {name}")
        self.raw_packer.validate(fields)
        if self.scene_cache is not None:
            for name in ("body_xmat","geom_xpos","geom_xmat","site_xmat","subtree_com"):
                expected={"body_xmat":self.model.nbody*9,"geom_xpos":self.model.ngeom*3,"geom_xmat":self.model.ngeom*9,
                          "site_xmat":self.model.nsite*9,"subtree_com":self.model.nbody*3}[name]*self.nworld
                field=fields[name]
                if field.dtype!=scalar.wp_real or field.ndim!=1 or field.size!=expected or field.device!=self.raw_packer.buffer.device:
                    raise ValueError(f"invalid camera scene field: {name}")
        self.valid=False
        try:
            source=self.raw_packer.copy_from(fields,stream=stream)
            self.lowdim.update(source,timestep,force,world_mask,stream)
            if self.scene_cache is not None:
                fields.update(self.scene_transforms.update(fields["body_xpos"],fields["body_xmat"],world_mask,stream))
                fields.update(self.visual_sources)
                scene=self.scene_packer.copy_from(fields,stream=stream)
                self.scene_cache.update(scene,timestep,force,world_mask,stream)
        except Exception:
            raise
        self.valid=True
        self._after_gpu_sample(timestep,force,world_mask,stream)

    def _after_gpu_sample(self,timestep,force=False,world_mask=None,stream=None):
        """Shared completion hook for baseline/direct GPU sampling, once per event.

        Default built-ins only record the completed event. Synchronize and run
        original Python extensions only if an optional renderer/callback needs it.
        """
        self.assert_sampling_allowed(stream)
        # Python executes while recording a graph. No sampling event or Python
        # Observable callback has happened until the recorded graph is launched.
        if stream is not None and getattr(stream,'is_capturing',False):return
        self._record_sample_event(timestep,force)
        if self._pending_renderers:
            if stream is not None:wp.synchronize_stream(stream)
            elif self.raw_packer.buffer.device.is_cuda:wp.synchronize_stream(wp.get_stream(self.device))
            for world in tuple(self._pending_renderers):
                env=self.host_environments[world]
                # The original optional renderer consumes current GPU facade
                # transforms and task XML. Its original dependency errors remain.
                self.initialize_renderer(world)
                self._pending_renderers.discard(world)
        if self.host_observables:
            if stream is not None:wp.synchronize_stream(stream)
            elif self.raw_packer.buffer.device.is_cuda:wp.synchronize_stream(wp.get_stream(self.device))
            selected=np.ones(self.nworld,dtype=np.int32) if world_mask is None else world_mask.numpy()
            for world,compat in self.host_observables.items():
                if selected[world]:compat.update(float(timestep),force=bool(force))

    def observations(self,world=0,image_convention="opengl"):
        if self.closed or not self.valid:raise RuntimeError("observation adapter is closed or quarantined")
        if world in self.host_observables:return self.host_observables[world].pack()
        images={}
        if self.scene_cache is not None:
            # Rendering only now, but using the retained original sample event.
            for spec in self.plan.image_specs:
                h,w,_=spec["shape"]
                key=(h,w)
                if key not in self.bridges:self.bridges[key]=DerivedRenderBridge(self.model,width=w,height=h)
            option=self._public_bridge.vopt if self._public_bridge is not None else self.scene_option
            images=self.scene_cache.render_at_boundary(self.bridges,world,image_convention,option)
        return self.lowdim.pack(world,images)

    def observations_many(self, worlds=None, image_convention="opengl"):
        """Download each shared buffer once, then package selected worlds."""
        if self.closed or not self.valid:
            raise RuntimeError('observation adapter is closed or quarantined')
        worlds = tuple(range(self.nworld)) if worlds is None else tuple(worlds)
        if any(not 0 <= w < self.nworld for w in worlds):
            raise IndexError('observation world outside batch')
        if self.host_observables:
            return [self.observations(w, image_convention) for w in worlds]
        if self.lowdim.last_stream is not None:
            wp.synchronize_stream(self.lowdim.last_stream)
        values = self.lowdim.observed.numpy().reshape(self.nworld, -1)
        scene = self.scene_cache
        snapshot = None
        if scene is not None:
            if scene.last_stream is not None:
                wp.synchronize_stream(scene.last_stream)
            snapshot = (scene.generation.numpy().reshape(self.nworld, scene.ncamera),
                        scene.scenes.numpy().reshape(self.nworld, scene.ncamera, scene.stride),
                        scene.parent_generation.numpy().reshape(self.nworld, scene.ncamera))
            for spec in self.plan.image_specs:
                h, w, _ = spec['shape']
                if (h, w) not in self.bridges:
                    self.bridges[h, w] = DerivedRenderBridge(self.model, width=w, height=h)
        option = self._public_bridge.vopt if self._public_bridge is not None else self.scene_option
        result = []
        for world in worlds:
            images = {} if scene is None else scene.render_at_boundary(
                self.bridges, world, image_convention, option, snapshot=snapshot)
            result.append(self.lowdim.pack(world, images, values=values[world]))
        return result

    def render_gpu(self,raw_fields,width=640,height=480,camera_name=None,depth=False,segmentation=False,world=0,stream=None,mode="offscreen",device_id=-1):
        """Render current device publications at an explicit public boundary.

        raw_fields uses the same device descriptor contract as sample(). Static
        camera/light transforms are evaluated on device, never CPU mj_forward.
        This current scene is separate from retained observation sample events.
        """
        if self.closed or not self.valid:raise RuntimeError("observation adapter is closed or quarantined")
        if not 0<=world<self.nworld:raise IndexError(world)
        assert mode=="offscreen","only offscreen supported for now"
        width,height=int(width),int(height)
        if width<=0 or height<=0:raise ValueError("render dimensions must be positive")
        fields=self._flat(raw_fields)
        shapes={"xpos":(int(self.model.nbody),3),"xmat":(int(self.model.nbody),9),
                "geom_xpos":(int(self.model.ngeom),3),"geom_xmat":(int(self.model.ngeom),9),
                "site_xpos":(int(self.model.nsite),3),"site_xmat":(int(self.model.nsite),9),
                "subtree_com":(int(self.model.nbody),3)}
        for name,shape in shapes.items():
            if name not in fields:raise KeyError("missing published render field: "+name)
            value=fields[name]
            if (value.dtype!=scalar.wp_real or value.ndim!=1 or value.size!=self.nworld*int(np.prod(shape))
                    or value.device!=self.raw_packer.buffer.device):
                raise ValueError("invalid published render device field: "+name)
        if self.scene_transforms is None:self.scene_transforms=DeviceSceneTransforms(self.model,self.nworld,self.device)
        fields.update(self.scene_transforms.update(fields["body_xpos"],fields["body_xmat"],stream=stream))
        shapes.update({"cam_xpos":(int(self.model.ncam),3),"cam_xmat":(int(self.model.ncam),9),
                       "light_xpos":(int(self.model.nlight),3),"light_xdir":(int(self.model.nlight),3)})
        fields.update(self.visual_sources)
        shapes.update({name:tuple(getattr(self.model,name).shape) for name in self.visual_sources})
        if stream is not None:wp.synchronize_stream(stream)
        elif self.raw_packer.buffer.device.is_cuda:wp.synchronize_stream(wp.get_stream(self.device))
        snapshot={name:fields[name].numpy().reshape((self.nworld,)+shape)[world].copy()
                  for name,shape in shapes.items()}
        return self.render(snapshot,width,height,camera_name,depth,segmentation)

    def render(self,fields,width=640,height=480,camera_name=None,depth=False,segmentation=False):
        """Public MjSim.render boundary, using a complete current runtime scene."""
        if self.closed or not self.valid:raise RuntimeError("observation adapter is closed or quarantined")
        width,height=int(width),int(height)
        if width<=0 or height<=0:raise ValueError("render dimensions must be positive")
        key=(height,width)
        if key not in self.bridges:self.bridges[key]=DerivedRenderBridge(self.model,width=width,height=height)
        bridge=self.bridges[key]
        if self._public_bridge is not None and bridge is not self._public_bridge:
            bridge.cam=self._public_bridge.cam
            bridge.vopt=self._public_bridge.vopt
            bridge.pert=self._public_bridge.pert
        if self._public_bridge is None and self.scene_option is not None:bridge.vopt=self.scene_option
        self._public_bridge=bridge
        self.scene_option=bridge.vopt
        bridge.publish(fields)
        return bridge.render(camera_name,depth,segmentation,self.scene_option)

    def reset(self,world_mask=None,stream=None):
        if self.closed or not self.valid:raise RuntimeError("observation adapter is closed or quarantined")
        self.lowdim.reset(world_mask,stream)
        if self.scene_cache is not None:self.scene_cache.reset(world_mask,stream)
        if self.host_observables:
            selected=np.ones(self.nworld,dtype=np.int32) if world_mask is None else world_mask.numpy()
            for world,compat in self.host_observables.items():
                if selected[world]:compat.reset()

    def update_visual_parameters(self,changes,world=0,stream=None):
        if self.closed or not self.valid:raise RuntimeError("observation adapter is closed or quarantined")
        if not 0<=world<self.nworld:raise IndexError(world)
        prepared={}
        for name,value in changes.items():
            if name not in self.visual_sources:raise KeyError(name)
            shape=getattr(self.model,name).shape
            value=np.asarray(value)
            if value.shape!=shape or not np.all(np.isfinite(value)):raise ValueError(name)
            prepared[name]=value.astype(np.float64).reshape(-1).copy()
        self.valid=False
        for name,value in prepared.items():
            with (wp.ScopedStream(stream) if stream is not None else nullcontext()):
                data=wp.array(value,dtype=scalar.wp_real,device=self.device)
            wp.copy(self.visual_sources[name],data,dest_offset=world*data.size,count=data.size,stream=stream)
        self.valid=True
        self.visual_generation+=1
    def snapshot_continuation(self,stream=None):
        from .continuation import snapshot
        packet=snapshot(self,stream)
        packet['requires_host_sampling']=self.requires_host_sampling
        packet['sample_event_token']=self.sample_event_token
        return packet
    def restore_continuation(self,packet,stream=None):
        from .continuation import restore
        token=packet.get('sample_event_token',0)
        if type(token) is not int or token<0:raise ValueError('invalid observation sample event token')
        result=restore(self,packet,stream)
        self._sample_event_token=token
        if self.host_observables or packet.get('requires_host_sampling',False):
            self._require_host_sampling('restored custom observables')
        return result
    def close(self):
        if self.closed:return
        errors=[]
        try:self._synchronize_sampling()
        except Exception as error:errors.append(error)
        for world,viewer in self._owned_viewers.items():
            if viewer is not None:
                try:viewer.close()
                except Exception as error:errors.append(error)
                env=self.host_environments.get(world)
                if env is not None and env.viewer is viewer:env.viewer=None
        self._owned_viewers.clear()
        self.host_environments.clear();self.host_observables.clear()
        for bridge in self.bridges.values():
            try:bridge.close()
            except Exception as error:errors.append(error)
        self.bridges.clear()
        self._pending_renderers.clear()
        self._sample_event_hook=None;self._graph_guard=None
        sampler=getattr(self,'_cuda_direct_sampler',None)
        if sampler is not None:
            baseline=getattr(sampler,'baseline_sample',None)
            if baseline is not None:self.sample=baseline
            sampler.observer=None;sampler.baseline_sample=None;sampler.owner_arrays=()
            sampler.plan=None
            self._cuda_direct_sampler=None
        self._cuda_batch_observations_registration=None
        self.closed=True
        if errors:raise errors[0]
    def public_render_context(self):
        if self.closed or not self.valid:raise RuntimeError("observation adapter is closed or quarantined")
        if self._public_bridge is None:
            key=(int(self.model.vis.global_.offheight),int(self.model.vis.global_.offwidth))
            if key not in self.bridges:self.bridges[key]=DerivedRenderBridge(self.model,width=key[1],height=key[0])
            self._public_bridge=self.bridges[key]
            if self.scene_option is not None:self._public_bridge.vopt=self.scene_option
        return self._public_bridge

    def bind_host_environment(self,env,world=0):
        if not 0<=world<self.nworld:raise IndexError(world)
        self.host_environments[world]=env
        renderer=getattr(env,"renderer","mujoco").lower()
        if renderer not in ("mujoco","default"):
            self._pending_renderers.add(world)
            if self._last_sample is not None:
                self.initialize_renderer(world)
                self._pending_renderers.discard(world)
        elif getattr(env,"has_renderer",False):self.initialize_renderer(world)

    def initialize_renderer(self,world=0):
        if self.closed or not self.valid:raise RuntimeError("observation adapter is closed or quarantined")
        if world not in self.host_environments:raise RuntimeError("renderer requires a live GPU environment binding")
        env=self.host_environments[world]
        renderer=getattr(env,"renderer","mujoco").lower()
        if renderer in ("mujoco","default"):
            if getattr(env,"has_renderer",False) and getattr(env,"viewer",None) is None:
                from robosuite.utils import OpenCVRenderer
                env.viewer=OpenCVRenderer(env.sim)
                if env.render_camera is not None:env.viewer.set_camera(env.sim.model.camera_name2id(env.render_camera))
        else:
            if self._last_sample is None:
                self._pending_renderers.add(world)
                return
            # Model-only construction installed an instance no-op. Resolve the
            # original class implementation explicitly, never the cold shim or
            # the task-driver delegation installed on this same instance.
            callback=getattr(type(env),"initialize_renderer",None)
            if not callable(callback):raise AttributeError("original renderer initialization method unavailable")
            callback(env)
        env.viewer_get_obs=hasattr(getattr(env,"viewer",None),"_get_observations")
        if getattr(env,"viewer",None) is not None:self._owned_viewers[world]=env.viewer
        self._pending_renderers.discard(world)

    @property
    def observation_modalities(self):
        if self.host_observables:
            return {obs.modality for obs in next(iter(self.host_observables.values())).observables.values()}
        return {spec.modality for spec in self.plan.specs}|({"image"} if self.plan.image_specs else set())

    def read_scene_fields(self,raw_fields,names=("cam_xpos","cam_xmat","light_xpos","light_xdir"),world=0,stream=None):
        """Named scene-array boundary with actual GPU body inputs, no GL context.

        Fixed camera/light transforms use the existing device producer. Tracking
        modes use standard scene-only mj_camlight on GPU body/subtree readback.
        Neither path performs CPU physics or substitutes a fabricated pose.
        """
        if self.closed or not self.valid:raise RuntimeError("observation adapter is closed or quarantined")
        if not 0<=world<self.nworld:raise IndexError(world)
        shapes={"cam_xpos":(int(self.model.ncam),3),"cam_xmat":(int(self.model.ncam),9),
                "light_xpos":(int(self.model.nlight),3),"light_xdir":(int(self.model.nlight),3)}
        names=tuple(names)
        if any(name not in shapes for name in names):raise KeyError("unknown camera/light scene field")
        if not names:return {}
        fields=self._flat(raw_fields)
        for name,size in (("body_xpos",self.model.nbody*3),("body_xmat",self.model.nbody*9)):
            if name not in fields:raise KeyError("missing published scene field: "+name)
            value=fields[name]
            if value.dtype!=scalar.wp_real or value.ndim!=1 or value.size!=self.nworld*size or value.device!=self.raw_packer.buffer.device:
                raise ValueError("invalid published scene field: "+name)
        if self.scene_transforms is None:self.scene_transforms=DeviceSceneTransforms(self.model,self.nworld,self.device)
        trans=self.scene_transforms
        subtree=fields.get("subtree_com")
        if trans.host_camlight and (subtree is None or subtree.dtype!=scalar.wp_real or subtree.ndim!=1
                or subtree.size!=self.nworld*self.model.nbody*3 or subtree.device!=self.raw_packer.buffer.device):
            raise ValueError("tracking cameras/lights require published subtree_com")
        outputs=trans.update(fields["body_xpos"],fields["body_xmat"],stream=stream)
        if stream is not None:wp.synchronize_stream(stream)
        elif self.raw_packer.buffer.device.is_cuda:wp.synchronize_stream(wp.get_stream(self.device))
        if not trans.host_camlight:
            return {name:outputs[name].numpy().reshape((self.nworld,)+shapes[name])[world].copy() for name in names}
        import mujoco
        from .render_model import clone_render_model
        container=getattr(self,"_named_scene_container",None)
        if container is None:
            model=clone_render_model(self.model);container=(model,mujoco.MjData(model));self._named_scene_container=container
        model,data=container
        for name,source,shape in (("xpos",fields["body_xpos"],(self.model.nbody,3)),
                                   ("xmat",fields["body_xmat"],(self.model.nbody,9)),
                                   ("subtree_com",subtree,(self.model.nbody,3))):
            np.copyto(getattr(data,name),source.numpy().reshape((self.nworld,)+shape)[world],casting="no")
        mujoco.mj_camlight(model,data)
        return {name:np.array(getattr(data,name),copy=True) for name in names}

    def _enable_host_observables(self):
        from .observable_compatibility import HostObservableCompatibility
        if not self.host_environments:raise RuntimeError("custom observable requires a live GPU environment binding")
        if self._last_sample is None:raise RuntimeError("custom sensor validation requires the completed initial GPU sample")
        for world,env in self.host_environments.items():
            if world not in self.host_observables:
                # Capture existing clocks and held values before constructing the
                # custom extension objects, so a modifier does not reset unrelated
                # original observables or change an in-flight sampling period.
                if self.lowdim.last_stream is not None:wp.synchronize_stream(self.lowdim.last_stream)
                values=self.lowdim.observed.numpy().reshape(self.nworld,-1)[world]
                cached=self.lowdim.cache.numpy().reshape(self.nworld,-1)[world]
                elapsed=self.lowdim.elapsed.numpy().reshape(self.nworld,-1)[world]
                sampled=self.lowdim.sampled.numpy().reshape(self.nworld,-1)[world]
                present=self.lowdim.present.numpy().reshape(self.nworld,-1)[world]
                prior_images={}
                if self.scene_cache is not None:
                    for spec in self.plan.image_specs:
                        h,w,_=spec["shape"]
                        if (h,w) not in self.bridges:self.bridges[(h,w)]=DerivedRenderBridge(self.model,width=w,height=h)
                    from robosuite import macros
                    prior_images=self.scene_cache.render_at_boundary(self.bridges,world,macros.IMAGE_CONVENTION,self.scene_option)
                    scene_elapsed=self.scene_cache.elapsed.numpy().reshape(self.nworld,-1)[world]
                    scene_sampled=self.scene_cache.sampled.numpy().reshape(self.nworld,-1)[world]
                compat=HostObservableCompatibility(env)
                settings={s.name:(s.period,s.enabled,s.active) for s in self.plan.specs}
                settings.update({s["name"]:(s["period"],s["enabled"],s["active"]) for s in self.plan.image_specs})
                for name,(period,enabled,active) in settings.items():
                    if name in compat.observables:
                        observable=compat.observables[name]
                        observable.set_sampling_rate(1.0/period)
                        observable.set_enabled(enabled);observable.set_active(active)
                compat.cache.clear()
                for i,spec in enumerate(self.plan.specs):
                    if spec.name not in compat.observables:continue
                    observable=compat.observables[spec.name]
                    at=self.plan.offsets[spec.name]
                    def shaped(source):
                        value=source[at:at+spec.width].copy().reshape(spec.shape or ())
                        return value[0] if spec.shape==(1,) else value
                    observable._current_observed_value=shaped(values)
                    observable._time_since_last_sample=float(elapsed[i]);observable._sampled=bool(sampled[i])
                    if present[i]:compat.cache[spec.name]=np.array(shaped(cached))
                for i,spec in enumerate(self.plan.image_specs):
                    observable=compat.observables[spec["name"]]
                    observable._current_observed_value=np.array(prior_images[spec["name"]],copy=True)
                    observable._time_since_last_sample=float(scene_elapsed[i]);observable._sampled=bool(scene_sampled[i])
                    compat.cache[spec["name"]]=np.array(prior_images[spec["name"]],copy=True)
                self.host_observables[world]=compat

    @property
    def requires_host_callbacks(self):return self.requires_host_sampling

    @property
    def observation_names(self):
        if self.host_observables:return set(next(iter(self.host_observables.values())).observables)
        return set(self.plan.order)
    @property
    def enabled_observables(self):
        if self.host_observables:return {name for name,obs in next(iter(self.host_observables.values())).observables.items() if obs.is_enabled()}
        return {s.name for s in self.plan.specs if s.enabled}|{s["name"] for s in self.plan.image_specs if s["enabled"]}
    @property
    def active_observables(self):
        if self.host_observables:return {name for name,obs in next(iter(self.host_observables.values())).observables.items() if obs.is_active()}
        return {s.name for s in self.plan.specs if s.active}|{s["name"] for s in self.plan.image_specs if s["active"]}

    def add_observable(self,observable,stream=None):
        if self.closed or not self.valid:raise RuntimeError("observation adapter is closed or quarantined")
        self._require_host_sampling('added observable')
        self._enable_host_observables()
        import copy
        owners=list(self.host_observables.values())
        for compat in owners:
            assert observable.name not in compat.observables,"Observable {} already exists".format(observable.name)
        # Observable clocks and held readings belong to one world. Preserve the
        # supplied object for the first owner and stage independent copies before
        # publishing the rest; Python functions retain their usual closure identity.
        copies=[observable]+[copy.deepcopy(observable) for _ in owners[1:]]
        for compat,value in zip(owners,copies):compat.add(value)

    def modify_observable(self,name,attribute,modifier,stream=None):
        from .observable_compatibility import ATTRIBUTES
        from dataclasses import replace
        if self.closed or not self.valid:raise RuntimeError("observation adapter is closed or quarantined")
        assert name in self.observation_names,"No valid observable with name {} found. Options are: {}".format(name,self.observation_names)
        if attribute not in ATTRIBUTES:raise ValueError("Invalid observable attribute specified: "+str(attribute))
        if attribute not in ("sampling_rate","enabled","active") or self.host_observables:
            self._require_host_sampling('custom observable '+str(attribute))
            self._enable_host_observables()
            for compat in self.host_observables.values():compat.modify(name,attribute,modifier)
            return
        change={attribute:modifier}
        if attribute=="sampling_rate":
            rate=float(modifier)
            if not np.isfinite(rate) or rate<=0:raise ValueError("sampling rate must be positive finite")
            change={"period":1.0/rate}
        else:change={attribute:bool(modifier)}
        specs=tuple(replace(s,**change) if s.name==name else s for s in self.plan.specs)
        images=tuple(dict(s,**change) if s["name"]==name else s for s in self.plan.image_specs)
        new_plan=replace(self.plan,specs=specs,image_specs=images)
        self._synchronize_sampling(stream)
        write_stream=stream or self.lowdim.last_stream or wp.get_stream(self.device)
        self.valid=False
        try:
            with wp.ScopedStream(write_stream):
                self.lowdim.meta[9].assign(np.asarray([int(s.enabled) for s in specs],dtype=np.int32))
                self.lowdim.meta[10].assign(np.asarray([s.period for s in specs],dtype=np.float64))
                if self.scene_cache is not None:
                    self.scene_cache.enabled.assign(np.asarray([int(s["enabled"]) for s in images],dtype=np.int32))
                    self.scene_cache.periods.assign(np.asarray([s["period"] for s in images],dtype=np.float64))
                if attribute=="enabled":
                    for i,s in enumerate(specs):
                        if s.name==name:
                            elapsed=self.lowdim.elapsed.numpy().reshape(self.nworld,-1);elapsed[:,i]=0
                            observed=self.lowdim.observed.numpy().reshape(self.nworld,-1)
                            at=self.plan.offsets[name];observed[:,at:at+s.width]=0
                            self.lowdim.elapsed.assign(elapsed.reshape(-1))
                            self.lowdim.observed.assign(observed.reshape(-1))
                    if self.scene_cache is not None:
                        for i,s in enumerate(images):
                            if s["name"]==name:
                                scene=self.scene_cache;elapsed=scene.elapsed.numpy().reshape(self.nworld,-1);elapsed[:,i]=0
                                scene.elapsed.assign(elapsed.reshape(-1))
                                generations=scene.generation.numpy().reshape(self.nworld,-1)
                                for world in range(self.nworld):
                                    scene.images[world][i]=np.zeros(s["shape"])
                                    scene.rendered_generation[world,i]=generations[world,i]
                self.plan=new_plan;self.lowdim.plan=new_plan;self.lowdim._packing_recipe=None
                if self.scene_cache is not None:self.scene_cache.specs=images
        except Exception:raise
        self.lowdim.last_stream=write_stream
        if self.scene_cache is not None:self.scene_cache.last_stream=write_stream
        self._synchronize_sampling(write_stream)
        self.valid=True
