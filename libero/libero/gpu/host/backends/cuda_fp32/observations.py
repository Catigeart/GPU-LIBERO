"""Independent FP32 observable buffers and CUDA sampling entry points.

The ordered plan, sampled clocks, presence semantics and host packaging are
shared contracts with the original observation adapter. No Warp import/JIT.
"""
import ctypes as ct
import numpy as np
from . import device as cuda
from .bridges import Bridges,Observe,Scene
from libero.gpu.host.adapters.observations.schema import pack_observations
from libero.gpu.host.adapters.observations.packing_recipe import PackingRecipe


def bridge(device):
    stream=cuda.get_stream(device)
    return Bridges(stream.runtime.path.parent/'cuda_bridges_fp32.so',stream)


class DeviceObservations:

    def __init__(self, plan, source_layout, nworld=1, device="cuda:0"):
        self.plan = plan.validate(); self.nworld = nworld; self.device = device
        self.last_stream=None
        self.packing_profile="recipe"
        self._packing_recipe=None
        self.source_stride = max((o+n for o,n in source_layout.values()), default=0)
        offsets = plan.offsets; names = {s.name:i for i,s in enumerate(plan.specs)}
        ops=[]; ix=[]; ixoffs=[]; ds0=[]; ds1=[]; ds2=[]; presence_deps=[]
        for s in plan.specs:
            ixoffs.append(len(ix)); d0=d1=d2=0
            presence_deps.extend([names[d] for d in s.dependencies] + [0]*(3-len(s.dependencies)))
            if s.operation in ("qpos", "qvel", "site_pos", "body_pos", "body_quat"):
                ops.append(0)
                field = {"site_pos":"site_xpos", "body_pos":"body_xpos", "body_quat":"body_xquat"}.get(s.operation,s.operation)
                start, length = source_layout[field]
                if s.operation == "body_quat": local = [s.indices[0]*4+j for j in (1,2,3,0)]
                elif s.operation in ("site_pos","body_pos"): local = [s.indices[0]*3+j for j in range(3)]
                else: local = list(s.indices)
                if len(local) != s.width or any(i<0 or i>=length for i in local):
                    raise ValueError(f"source index out of range: {s.name}")
                ix.extend(start+i for i in local)
            elif s.operation in ("cos","sin"):
                ops.append(1 if s.operation=="cos" else 2)
                d0=offsets[s.dependencies[0]]; d1=names[s.dependencies[0]]
            elif s.operation == "inverse_pose":
                ops.append(3); d0=offsets[s.dependencies[0]]; d1=offsets[s.dependencies[1]]
                d2=names[s.dependencies[0]]
                # The quaternion-presence check is redundant for the supported
                # ordered default plan: pos and quat have identical enabled clocks.
            elif s.operation == "relative_pos":
                ops.append(4); d0=offsets[s.dependencies[0]]; d1=offsets[s.dependencies[1]]; d2=offsets[s.dependencies[2]]
            elif s.operation == "relative_quat":
                ops.append(5); d0=offsets[s.dependencies[0]]; d1=names[s.dependencies[0]]
            else: raise NotImplementedError(s.operation)
            ds0.append(d0); ds1.append(d1); ds2.append(d2)
        def ai(x): return cuda.array(x,dtype=cuda.int32,device=device)
        self.meta = [ai(ops),ai([offsets[s.name] for s in plan.specs]),ai([s.width for s in plan.specs]),
                     ai(ixoffs),ai(ix),ai(ds0),ai(ds1),ai(ds2),ai(presence_deps),ai([int(s.enabled) for s in plan.specs]),
                     cuda.array([s.period for s in plan.specs],dtype=cuda.float32,device=device)]
        total=nworld*plan.width; clocks=nworld*len(plan.specs)
        self.cache=cuda.zeros(total,dtype=cuda.float32,device=device)
        self.observed=cuda.zeros(total,dtype=cuda.float32,device=device)
        self.sidequat=cuda.zeros(total+4,dtype=cuda.float32,device=device)
        self.elapsed=cuda.zeros(clocks,dtype=cuda.float32,device=device)
        self.sampled=cuda.zeros(clocks,dtype=cuda.int32,device=device)
        self.present=cuda.zeros(clocks,dtype=cuda.int32,device=device)
        self.events=cuda.zeros(clocks,dtype=cuda.int32,device=device)
        self.all_worlds=ai([1]*nworld)
        self.bridge=bridge(device)


    def configure_packing(self, profile="recipe"):
        if profile not in ("legacy", "recipe"):
            raise ValueError("unknown observation packing profile")
        self.packing_profile = profile


    def pack(self, world=0, images=None, *, values=None):
        if not 0 <= world < self.nworld: raise IndexError(world)
        if values is None:
            if self.last_stream is not None:cuda.synchronize_stream(self.last_stream)
            values = self.observed.numpy().reshape(self.nworld,-1)[world]
        if self.packing_profile == "recipe":
            if self._packing_recipe is None or not self._packing_recipe.matches(self.plan):
                self._packing_recipe = PackingRecipe.try_create(self.plan)
            if self._packing_recipe is not None:
                return self._packing_recipe.pack(values, images)
        return pack_observations(self.plan, values, images=images)


    def sample_arguments(self,source,timestep,world_mask=None,stream=None):
        if not np.isfinite(timestep) or timestep<=0:raise ValueError('Invalid sampling timestep')
        if source.dtype!=cuda.float32 or source.size!=self.nworld*self.source_stride:raise ValueError('Sampling source layout differs')
        self.last_stream=stream or cuda.get_stream(self.device)
        mask=self.all_worlds if world_mask is None else world_mask
        names=('op','offset','size','index_offset','indices','dep0','dep1','dep2','presence','enabled','periods')
        return Observe(worlds=self.nworld,ns=len(self.plan.specs),width=self.plan.width,stride=self.source_stride,
            **{name:value.ptr for name,value in zip(names,self.meta)},source=source.ptr,
            **{k:getattr(self,k).ptr for k in ('cache','observed','sidequat','elapsed','sampled','present','events')},mask=mask.ptr)
    def update(self,source,timestep,force=False,world_mask=None,stream=None):
        args=self.sample_arguments(source,timestep,world_mask,stream)
        self.bridge.call('observe',ct.byref(args),float(timestep),int(force))

    def reset(self,world_mask=None,stream=None):
        self.last_stream=stream or cuda.get_stream(self.device)
        mask=self.all_worlds if world_mask is None else world_mask
        self.bridge.call('reset_observation',self.nworld,len(self.plan.specs),self.plan.width,mask.ptr,
            self.elapsed.ptr,self.present.ptr,self.observed.ptr,self.cache.ptr,self.sidequat.ptr)


class DeviceSampledScenes:

    def __init__(self,image_specs,scene_layout,nworld=1,device="cuda:0",segmentation_mappings=None):
        from libero.gpu.host.adapters.observations.sampled_images import image_descriptor, segmentation_mapping
        self.descriptors=tuple(image_descriptor(spec) for spec in image_specs)
        self.segmentation_mappings=segmentation_mapping(self.descriptors,segmentation_mappings)
        self.specs=tuple(image_specs);self.layout=scene_layout;self.nworld=nworld;self.device=device
        self.stride=max((offset+int(np.prod(shape)) for offset,shape in scene_layout.values()),default=0)
        self.ncamera=len(image_specs)
        self.last_stream=None
        n=nworld*self.ncamera
        self.periods=cuda.array([s["period"] for s in image_specs],dtype=cuda.float32,device=device)
        self.enabled=cuda.array([int(s["enabled"]) for s in image_specs],dtype=cuda.int32,device=device)
        self.elapsed=cuda.zeros(n,dtype=cuda.float32,device=device)
        self.sampled=cuda.zeros(n,dtype=cuda.int32,device=device)
        self.generation=cuda.zeros(n,dtype=cuda.int32,device=device)
        indexes={spec["name"]:i for i,spec in enumerate(image_specs)}
        self.parents=cuda.array([indexes[camera+"_image"] if kind=="depth" else -1 for camera,kind in self.descriptors],dtype=cuda.int32,device=device)
        self.parent_generation=cuda.zeros(n,dtype=cuda.int32,device=device)
        self.captured=cuda.zeros(n,dtype=cuda.int32,device=device)
        self.scenes=cuda.zeros(n*self.stride,dtype=cuda.float32,device=device)
        self.mask=cuda.array([1]*nworld,dtype=cuda.int32,device=device)
        self.rendered_generation=np.zeros((nworld,self.ncamera),dtype=np.int32)
        self.images=[[np.zeros(tuple(s["shape"])) for s in self.specs] for _ in range(nworld)]
        self.bridge=bridge(device)


    def render_at_boundary(self,bridge,world=0,image_convention="opengl",scene_option=None, *, snapshot=None):
        from libero.gpu.host.adapters.observations.sampled_images import render_sampled_images
        if not 0<=world<self.nworld:raise IndexError(world)
        if snapshot is None:
            if self.last_stream is not None:cuda.synchronize_stream(self.last_stream)
            snapshot=(self.generation.numpy().reshape(self.nworld,self.ncamera),
                      self.scenes.numpy().reshape(self.nworld,self.ncamera,self.stride),
                      self.parent_generation.numpy().reshape(self.nworld,self.ncamera))
        generations,scenes,parents=(value[world] for value in snapshot)
        # MuJoCo's scene-only MjData container stores doubles. Widen the
        # published FP32 sample exactly at this CPU/GL boundary, preserving
        # its pose epoch and the strict render bridge's dtype contract.
        scenes=np.asarray(scenes,dtype=np.float64)
        render_sampled_images(self.specs,self.descriptors,self.segmentation_mappings,self.layout,
                             scenes,generations,self.rendered_generation[world],self.images[world],
                             bridge,image_convention,scene_option,
                             parents)
        return {s["name"]:self.images[world][i] for i,s in enumerate(self.specs)}


    def update(self,source,timestep,force=False,world_mask=None,stream=None):
        if not np.isfinite(timestep) or timestep<=0:raise ValueError('Invalid scene sampling timestep')
        if source.dtype!=cuda.float32 or source.size!=self.nworld*self.stride:raise ValueError('Scene sampling source differs')
        self.last_stream=stream or cuda.get_stream(self.device)
        mask=self.mask if world_mask is None else world_mask
        args=Scene(worlds=self.nworld,cameras=self.ncamera,stride=self.stride,source=source.ptr,scene=self.scenes.ptr,
            **{k:getattr(self,k).ptr for k in ('periods','enabled','elapsed','sampled','generation','parents','parent_generation','captured')},mask=mask.ptr)
        self.bridge.call('capture_scene',ct.byref(args),float(timestep),int(force))

    def reset(self,world_mask=None,stream=None):
        mask=self.mask if world_mask is None else world_mask
        self.last_stream=stream or cuda.get_stream(self.device)
        self.bridge.call('reset_observation',self.nworld,self.ncamera,0,mask.ptr,self.elapsed.ptr,
            self.captured.ptr,self.scenes.ptr,self.scenes.ptr,self.scenes.ptr)
        selected=mask.numpy();generation=self.generation.numpy().reshape(self.nworld,self.ncamera)
        for world in range(self.nworld):
            if selected[world]:
                self.images[world]=[np.zeros(tuple(s['shape'])) for s in self.specs]
                self.rendered_generation[world]=generation[world]


class DeviceFieldPacker:

    def __init__(self,layout,nworld=1,device="cuda:0"):
        self.layout=dict(layout);self.nworld=nworld;self.device=device
        self.stride=max((o+n for o,n in layout.values()),default=0)
        occupied=set()
        for name,(offset,length) in layout.items():
            if offset<0 or length<0:raise ValueError(name)
            region=set(range(offset,offset+length))
            if occupied.intersection(region):raise ValueError(f"overlapping source layout: {name}")
            occupied.update(region)
        self.buffer=cuda.zeros(nworld*self.stride,dtype=cuda.float32,device=device)
        self.bridge=bridge(device)
        from .fused_observations import FusedRawPack,FIELDS
        self.fused_raw=(FusedRawPack(self,self.buffer.stream.runtime.path.parent/'fp32_fused_observations.so')
                        if self.stride>0 and set(self.layout)==set(FIELDS) else None)


    def validate(self,fields):
        for name,(offset,length) in self.layout.items():
            source=fields[name]
            if source.dtype!=cuda.float32 or source.size!=self.nworld*length:
                raise ValueError(f"field shape/type mismatch: {name}")
            if source.device!=self.buffer.device:raise ValueError(f"field device mismatch: {name}")
            if source.ndim!=1:raise ValueError(f"flatten device field before packing: {name}")
        return fields


    def copy_from(self,fields,stream=None):
        self.validate(fields)
        if self.fused_raw is not None:
            self.fused_raw.enqueue(fields)
            return self.buffer
        for name,(offset,width) in self.layout.items():
            if width:self.bridge.call('pack',self.nworld,width,width,self.stride,offset,fields[name].ptr,self.buffer.ptr)
        return self.buffer


class DeviceSceneTransforms:

    def __init__(self,model,nworld=1,device="cuda:0"):
        import mujoco
        self.host_camlight=bool(np.any(np.asarray(model.cam_mode)!=int(mujoco.mjtCamLight.mjCAMLIGHT_FIXED))
                               or np.any(np.asarray(model.light_mode)!=int(mujoco.mjtCamLight.mjCAMLIGHT_FIXED)))
        # Fixed modes retain the original GPU producer. Non-fixed modes receive
        # scene-only mj_camlight at the explicit GL boundary from GPU body fields;
        # zero device scratch is never treated as the final camera publication.
        self.nbody=int(model.nbody);self.ncam=int(model.ncam);self.nlight=int(model.nlight)
        self.nworld=nworld;self.device=device
        self.all_worlds=cuda.array([1]*nworld,dtype=cuda.int32,device=device)
        def a(value,dtype):return cuda.array(np.asarray(value).reshape(-1),dtype=dtype,device=device)
        mats=np.zeros((self.ncam,9))
        for i in range(self.ncam):mujoco.mju_quat2Mat(mats[i],np.asarray(model.cam_quat[i]))
        self.camera_static=[a(model.cam_bodyid,cuda.int32),a(model.cam_pos,cuda.float32),a(mats,cuda.float32)]
        self.light_static=[a(model.light_bodyid,cuda.int32),a(model.light_pos,cuda.float32),a(model.light_dir,cuda.float32)]
        self.cam_xpos=cuda.zeros(nworld*self.ncam*3,dtype=cuda.float32,device=device)
        self.cam_xmat=cuda.zeros(nworld*self.ncam*9,dtype=cuda.float32,device=device)
        self.light_xpos=cuda.zeros(nworld*self.nlight*3,dtype=cuda.float32,device=device)
        self.light_xdir=cuda.zeros(nworld*self.nlight*3,dtype=cuda.float32,device=device)
        self.bridge=bridge(device)


    def update(self,body_xpos,body_xmat,world_mask=None,stream=None):
        if self.host_camlight:raise ValueError('Only FIXED cameras/lights are admitted')
        mask=self.all_worlds if world_mask is None else world_mask
        for count,rotation,metadata,position,axes in ((self.ncam,1,self.camera_static,self.cam_xpos,self.cam_xmat),
                (self.nlight,0,self.light_static,self.light_xpos,self.light_xdir)):
            if count:self.bridge.call('scene_transform',self.nworld,self.nbody,count,rotation,mask.ptr,
                *[a.ptr for a in metadata],body_xpos.ptr,body_xmat.ptr,position.ptr,axes.ptr)
        return dict(cam_xpos=self.cam_xpos,cam_xmat=self.cam_xmat,light_xpos=self.light_xpos,light_xdir=self.light_xdir)


from libero.gpu.host.adapters.observations.adapter import GpuObservationAdapter


class ObservationAdapter(GpuObservationAdapter):
    """FP32 device fields published into the CPU renderer's FP64 container."""

    def _sample_lowdim(self,fields,timestep,force,world_mask,stream):
        if self.raw_packer.fused_raw is None:
            return super()._sample_lowdim(fields,timestep,force,world_mask,stream)
        self.raw_packer.fused_raw.sample(self.lowdim,fields,timestep,force,world_mask,stream)

    def render(self,fields,width=640,height=480,camera_name=None,depth=False,segmentation=False):
        from libero.gpu.host.adapters.observations.boundary import RENDER_FIELDS
        published={name:np.asarray(value,dtype=np.float64) if name in RENDER_FIELDS else value
                   for name,value in fields.items()}
        return super().render(published,width,height,camera_name,depth,segmentation)
