"""Explicit cold-reprepare boundary byte transport; no sensor recomputation."""
import libero.gpu.precision as scalar
import copy
from contextlib import nullcontext
import dataclasses
import json
import numpy as np
import warp as wp

LOWDIM_FIELDS = ("cache","observed","sidequat","elapsed","sampled","present","events")
SCENE_FIELDS = ("elapsed","sampled","generation","scenes","parent_generation","captured")
TRANSFORM_FIELDS = ("cam_xpos","cam_xmat","light_xpos","light_xdir")

def arrays(adapter):
    result = {"lowdim."+name:getattr(adapter.lowdim,name) for name in LOWDIM_FIELDS}
    result["raw_packer.buffer"] = adapter.raw_packer.buffer
    for name,value in adapter.visual_sources.items():result["visual."+name]=value
    if adapter.scene_cache is not None:
        for name in SCENE_FIELDS:result["scene."+name]=getattr(adapter.scene_cache,name)
        for name in TRANSFORM_FIELDS:result["transform."+name]=getattr(adapter.scene_transforms,name)
        result["scene_packer.buffer"] = adapter.scene_packer.buffer
    return result

def schema(adapter):
    return json.dumps({"version":1,"nworld":adapter.nworld,
                       "specs":[dataclasses.asdict(s) for s in adapter.plan.specs],
                       "images":adapter.plan.image_specs,
                       "segmentation_mappings":None if adapter.scene_cache is None else adapter.scene_cache.segmentation_mappings,
                       "scene_layout":None if adapter.scene_cache is None else adapter.scene_cache.layout},
                      sort_keys=True)

def _snapshot(adapter, stream=None):
    if adapter.closed or not adapter.valid:raise RuntimeError("invalid observer cannot snapshot")
    streams = [stream,adapter.lowdim.last_stream]
    if adapter.scene_cache is not None:streams.append(adapter.scene_cache.last_stream)
    for value in streams:
        if value is not None:wp.synchronize_stream(value)
    packet={"schema":schema(adapter),"arrays":{name:value.numpy().copy() for name,value in arrays(adapter).items()},
            "visual_generation":adapter.visual_generation,"host_scene":None,
            "host_observables":{world:compat.snapshot_owner() for world,compat in adapter.host_observables.items()},
            "last_sample":adapter._last_sample,"render_state":None,
            "viewer_owner":adapter,"viewers":dict(adapter._owned_viewers)}
    if adapter._public_bridge is not None:
        bridge=adapter._public_bridge
        def properties(value,names):return {name:copy.deepcopy(getattr(value,name)) for name in names if hasattr(value,name)}
        packet["render_state"]={"camera":properties(bridge.cam,("type","fixedcamid","trackbodyid","lookat","distance","azimuth","elevation")),
                                "option":properties(bridge.vopt,("label","frame","geomgroup","sitegroup","jointgroup","tendongroup","actuatorgroup","flags","bvh_depth")),
                                "perturb":properties(bridge.pert,("select","skinselect","active","refpos","refquat","localpos","scale"))}
    if adapter.scene_cache is not None:
        packet["host_scene"]={"rendered_generation":adapter.scene_cache.rendered_generation.copy(),
                              "images":[[image.copy() for image in row] for row in adapter.scene_cache.images]}
    return packet

def snapshot(adapter, stream=None):
    try:
        return _snapshot(adapter,stream)
    except Exception:
        adapter.valid=False
        raise

def restore(adapter, packet, stream=None):
    if adapter.closed or not adapter.valid:raise RuntimeError("invalid observer cannot restore")
    # A fresh owner starts with construction defaults; restore user descriptor
    # settings before transporting clocks. Never accept changed IDs or shapes.
    old_schema=json.loads(packet["schema"])
    new_schema=json.loads(schema(adapter))
    def structural(value):
        value=copy.deepcopy(value)
        for spec in value["specs"]+value["images"]:
            for key in ("period","enabled","active"):spec.pop(key,None)
        return value
    if structural(old_schema)!=structural(new_schema):raise ValueError("continuation observation schema differs")
    for old,new in zip(old_schema["specs"]+old_schema["images"],new_schema["specs"]+new_schema["images"]):
        for key in ("period","enabled","active"):
            if old[key]!=new[key]:
                adapter.modify_observable(old["name"],"sampling_rate" if key=="period" else key,
                                          1.0/old[key] if key=="period" else old[key],stream=stream)
    adapter.valid=False
    try:
        if packet["schema"]!=schema(adapter):raise ValueError("continuation observation schema differs")
        targets=arrays(adapter)
        if set(packet["arrays"])!=set(targets):raise ValueError("continuation buffer readset differs")
        prepared={}
        for name,target in targets.items():
            value=packet["arrays"][name]
            dtype=np.dtype(np.float64 if target.dtype==scalar.wp_real else np.int32)
            if not isinstance(value,np.ndarray) or value.shape!=target.shape or value.dtype!=dtype:
                raise ValueError("continuation dtype/shape mismatch: "+name)
            # Transport bytes, including untouched scratch, without numerical
            # recasting or finite filtering that would silently alter payloads.
            prepared[name]=value.copy()
        host=copy.deepcopy(packet["host_scene"])
        if adapter.scene_cache is None:
            if host is not None:raise ValueError("unexpected continuation images")
        else:
            scene=adapter.scene_cache
            if (not isinstance(host,dict) or host["rendered_generation"].shape!=scene.rendered_generation.shape
                    or host["rendered_generation"].dtype!=np.int32 or len(host["images"])!=scene.nworld):
                raise ValueError("continuation image metadata differs")
            for row in host["images"]:
                if len(row)!=scene.ncamera:raise ValueError("continuation camera count differs")
                for image,spec in zip(row,scene.specs):
                    if (not isinstance(image,np.ndarray) or image.shape!=tuple(spec["shape"])
                            or image.dtype not in (np.dtype(np.uint8),np.dtype(np.float64),np.dtype(np.float32),np.dtype(np.int32))):
                        raise ValueError("continuation image shape differs")
        from .observable_compatibility import HostObservableCompatibility
        host_observables={}
        for world,payload in packet.get("host_observables",{}).items():
            if world not in adapter.host_environments:raise ValueError("custom observable environment owner not bound")
            host_observables[world]=HostObservableCompatibility.restore_owner(payload,adapter.host_environments[world])
        generation=packet["visual_generation"]
        if type(generation) is not int or generation<0:raise ValueError("invalid visual generation")
        uploads=[]
        for name,value in prepared.items():
            target=targets[name]
            with (wp.ScopedStream(stream) if stream is not None else nullcontext()):
                upload=wp.array(value,dtype=target.dtype,device=target.device)
            uploads.append(upload)
            wp.copy(target,upload,stream=stream)
        if stream is not None:wp.synchronize_stream(stream)
        elif targets:wp.synchronize_device(next(iter(targets.values())).device)
        if host is not None:
            adapter.scene_cache.rendered_generation=host["rendered_generation"]
            adapter.scene_cache.images=host["images"]
            adapter.scene_cache.last_stream=stream
        adapter.lowdim.last_stream=stream
        adapter.visual_generation=generation
        adapter.host_observables=host_observables
        adapter._last_sample=packet.get("last_sample")
        render_state=packet.get("render_state")
        if render_state is not None:
            adapter.valid=True
            try:bridge=adapter.public_render_context()
            finally:adapter.valid=False
            for key,target in (("camera",bridge.cam),("option",bridge.vopt),("perturb",bridge.pert)):
                for name,value in render_state[key].items():
                    current=getattr(target,name)
                    if isinstance(current,np.ndarray):np.copyto(current,value,casting="no")
                    else:setattr(target,name,value)
        previous=packet.get("viewer_owner")
        for world,viewer in packet.get("viewers",{}).items():
            environment=adapter.host_environments.get(world)
            if environment is None:raise ValueError("viewer task shell owner not bound")
            fresh=adapter._owned_viewers.get(world)
            if fresh is not None and fresh is not viewer:fresh.close()
            # The original OpenCV renderer stores sim; NVISII stores env. Move
            # their actual callback/configuration/history objects to the new owner.
            if hasattr(viewer,"sim"):viewer.sim=environment.sim
            if hasattr(viewer,"env"):viewer.env=environment
            environment.viewer=viewer;adapter._owned_viewers[world]=viewer
            if previous is not None and previous is not adapter:previous._owned_viewers.pop(world,None)
    except Exception:
        raise
    adapter.valid=True