"""Ordered device observable sampling. One thread owns one world and its cache.

No sensor is freshly evaluated during host packaging. CUDA execution/validation is
scheduled externally. CPU Warp execution is useful only for independent testing.
"""
import libero.gpu.precision as scalar
import numpy as np
import warp as wp
from .schema import pack_observations
from .packing_recipe import PackingRecipe

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": True})

@wp.func
def rotation32(x: scalar.wp_real, y: scalar.wp_real, z: scalar.wp_real, w: scalar.wp_real):
    q = wp.vec4(wp.float32(w), wp.float32(x), wp.float32(y), wp.float32(z))
    n = wp.dot(q, q)
    result = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    if n >= wp.float32(4.0) * wp.float32(2.220446049250313e-16):
        scale = wp.sqrt(scalar.wp_real(2.0) / scalar.wp_real(n))
        q = wp.vec4(wp.float32(scalar.wp_real(q[0])*scale), wp.float32(scalar.wp_real(q[1])*scale),
                    wp.float32(scalar.wp_real(q[2])*scale), wp.float32(scalar.wp_real(q[3])*scale))
        result = wp.mat33(wp.float32(scalar.wp_real(1.0)-scalar.wp_real(q[2]*q[2])-scalar.wp_real(q[3]*q[3])), q[1]*q[2]-q[3]*q[0], q[1]*q[3]+q[2]*q[0],
                          q[1]*q[2]+q[3]*q[0], wp.float32(scalar.wp_real(1.0)-scalar.wp_real(q[1]*q[1])-scalar.wp_real(q[3]*q[3])), q[2]*q[3]-q[1]*q[0],
                          q[1]*q[3]-q[2]*q[0], q[2]*q[3]+q[1]*q[0], wp.float32(scalar.wp_real(1.0)-scalar.wp_real(q[1]*q[1])-scalar.wp_real(q[2]*q[2])))
    return result

@wp.func
def matrix_quaternion32(m: wp.mat33):
    # Same float32 symmetric K and positive-w convention as robosuite mat2quat.
    # Dominant eigenvector computed by power iteration, independently validated
    # against numpy.linalg.eigh. Largest-diagonal seed avoids 180-degree w=0 trap.
    k = wp.mat44(m[0,0]-m[1,1]-m[2,2], m[0,1]+m[1,0], m[0,2]+m[2,0], m[2,1]-m[1,2],
                 m[0,1]+m[1,0], m[1,1]-m[0,0]-m[2,2], m[1,2]+m[2,1], m[0,2]-m[2,0],
                 m[0,2]+m[2,0], m[1,2]+m[2,1], m[2,2]-m[0,0]-m[1,1], m[1,0]-m[0,1],
                 m[2,1]-m[1,2], m[0,2]-m[2,0], m[1,0]-m[0,1], m[0,0]+m[1,1]+m[2,2]) / 3.0
    largest = int(0)
    for i in range(1, 4):
        if k[i,i] > k[largest,largest]: largest = i
    q = wp.vec4(0.0, 0.0, 0.0, 0.0)
    q[largest] = 1.0
    for iteration in range(24):
        q = k * q
        q = q / wp.sqrt(wp.dot(q, q))
    if q[3] < 0.0: q = -q
    return q

@wp.kernel
def update_kernel(
    nspec: int, width: int, source_stride: int,
    op: wp.array(dtype=wp.int32), offsets: wp.array(dtype=wp.int32), sizes: wp.array(dtype=wp.int32),
    index_offsets: wp.array(dtype=wp.int32), indices: wp.array(dtype=wp.int32),
    dep0: wp.array(dtype=wp.int32), dep1: wp.array(dtype=wp.int32), dep2: wp.array(dtype=wp.int32),
    presence_deps: wp.array(dtype=wp.int32), enabled: wp.array(dtype=wp.int32), periods: wp.array(dtype=scalar.wp_real),
    source: wp.array(dtype=scalar.wp_real),
    cache: wp.array(dtype=scalar.wp_real), observed: wp.array(dtype=scalar.wp_real),
    sidequat: wp.array(dtype=scalar.wp_real),
    elapsed: wp.array(dtype=scalar.wp_real), sampled: wp.array(dtype=wp.int32),
    present: wp.array(dtype=wp.int32), world_mask: wp.array(dtype=wp.int32),
    timestep: scalar.wp_real, force: int, sampled_event: wp.array(dtype=wp.int32)):
    world = wp.tid()
    if world_mask[world] == 0: return
    b = world * width
    sb = world * nspec
    for s in range(nspec):
        sampled_event[sb+s] = 0
        if enabled[s] != 0:
            e = elapsed[sb+s] + timestep
            elapsed[sb+s] = e
            take = (sampled[sb+s] == 0 and periods[s] >= e) or force != 0
            fallback = e >= periods[s] and sampled[sb+s] == 0
            if take or fallback:
                at = b+offsets[s]
                d0 = b+dep0[s]
                d1 = b+dep1[s]
                d2 = b+dep2[s]
                for j in range(sizes[s]):
                    value = scalar.wp_real(0.0)
                    if op[s] == 0:
                        value = source[world*source_stride+indices[index_offsets[s]+j]]
                    elif op[s] == 1:
                        if present[sb+presence_deps[s*3]] != 0: value = wp.cos(cache[d0+j])
                    elif op[s] == 2:
                        if present[sb+presence_deps[s*3]] != 0: value = wp.sin(cache[d0+j])
                    cache[at+j] = value
                if op[s] == 3:
                    # world_pose_in_gripper from already-sampled eef cache.
                    if present[sb+presence_deps[s*3+1]] == 0 or present[sb+presence_deps[s*3]] == 0:
                        for row in range(4):
                            cache[at+row*4+row] = scalar.wp_real(1.0)
                    else:
                        r = rotation32(cache[d1], cache[d1+1], cache[d1+2], cache[d1+3])
                        for row in range(3):
                            t = scalar.wp_real(0.0)
                            for col in range(3):
                                cache[at+row*4+col] = scalar.wp_real(r[col,row])
                                t -= scalar.wp_real(r[col,row]) * scalar.wp_real(wp.float32(cache[d0+col]))
                            cache[at+row*4+3] = t
                        cache[at+15] = scalar.wp_real(1.0)
                elif op[s] == 4:
                    if present[sb+presence_deps[s*3]] != 0 and present[sb+presence_deps[s*3+1]] != 0 and present[sb+presence_deps[s*3+2]] != 0:
                        # Object pose and inverse gripper pose are already cached.
                        ro = rotation32(cache[d1], cache[d1+1], cache[d1+2], cache[d1+3])
                        rr = wp.mat33(0.0)
                        for row in range(3):
                            t = cache[d2+row*4+3]
                            for col in range(3):
                                t += cache[d2+row*4+col]*scalar.wp_real(wp.float32(cache[d0+col]))
                                v = scalar.wp_real(0.0)
                                for k in range(3): v += cache[d2+row*4+k] * scalar.wp_real(ro[k,col])
                                rr[row,col] = wp.float32(v)
                            cache[at+row] = t
                        q = matrix_quaternion32(rr)
                        for j in range(4): sidequat[b+offsets[s]+j] = scalar.wp_real(q[j])
                elif op[s] == 5:
                    if present[sb+presence_deps[s*3]] != 0:
                        for j in range(4): cache[at+j] = sidequat[d0+j]
                for j in range(sizes[s]): observed[at+j] = cache[at+j]
                present[sb+s] = 1
                sampled[sb+s] = 1
                sampled_event[sb+s] = 1
            if e >= periods[s]:
                elapsed[sb+s] = e % periods[s]
                sampled[sb+s] = 0

@wp.kernel
def reset_kernel(nspec: int, width: int, mask: wp.array(dtype=wp.int32),
                 elapsed: wp.array(dtype=scalar.wp_real), present: wp.array(dtype=wp.int32),
                 observed: wp.array(dtype=scalar.wp_real), cache: wp.array(dtype=scalar.wp_real), sidequat: wp.array(dtype=scalar.wp_real)):
    world = wp.tid()
    if mask[world] == 0: return
    # Original Observable.reset does not clear _sampled; preserve this quirk.
    for s in range(nspec):
        elapsed[world*nspec+s] = scalar.wp_real(0.0)
        present[world*nspec+s] = 0
    for j in range(width):
        observed[world*width+j] = scalar.wp_real(0.0)
        cache[world*width+j] = scalar.wp_real(0.0)
        sidequat[world*width+j] = scalar.wp_real(0.0)

class DeviceObservations:
    """Static plan plus persistent per-world buffers. No transfers in update().

    source is a device-resident flat FP64 buffer assembled from raw fields in
    source_layout order. Copies between device fields are allowed; host downloads
    are forbidden until an observation boundary. source_layout maps field names to
    (offset, number_of_elements_per_world).
    """
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
        def ai(x): return wp.array(x,dtype=wp.int32,device=device)
        self.meta = [ai(ops),ai([offsets[s.name] for s in plan.specs]),ai([s.width for s in plan.specs]),
                     ai(ixoffs),ai(ix),ai(ds0),ai(ds1),ai(ds2),ai(presence_deps),ai([int(s.enabled) for s in plan.specs]),
                     wp.array([s.period for s in plan.specs],dtype=scalar.wp_real,device=device)]
        total=nworld*plan.width; clocks=nworld*len(plan.specs)
        self.cache=wp.zeros(total,dtype=scalar.wp_real,device=device)
        self.observed=wp.zeros(total,dtype=scalar.wp_real,device=device)
        self.sidequat=wp.zeros(total+4,dtype=scalar.wp_real,device=device)
        self.elapsed=wp.zeros(clocks,dtype=scalar.wp_real,device=device)
        self.sampled=wp.zeros(clocks,dtype=wp.int32,device=device)
        self.present=wp.zeros(clocks,dtype=wp.int32,device=device)
        self.events=wp.zeros(clocks,dtype=wp.int32,device=device)
        self.all_worlds=ai([1]*nworld)

    def update(self, source, timestep, force=False, world_mask=None, stream=None):
        if not np.isfinite(timestep) or timestep<=0:raise ValueError("sampling timestep must be positive finite")
        if world_mask is not None and (world_mask.dtype!=wp.int32 or world_mask.ndim!=1 or world_mask.size!=self.nworld or world_mask.device!=self.observed.device):
            raise ValueError("world mask must be a same-device int32 vector")
        if source.dtype != scalar.wp_real or source.ndim!=1 or source.device!=self.observed.device or source.size != self.nworld*self.source_stride:
            raise ValueError("source must be flat FP64 with exact world stride")
        self.last_stream=(stream or wp.get_stream(self.device)) if self.observed.device.is_cuda else None
        wp.launch(update_kernel,dim=self.nworld,inputs=[len(self.plan.specs),self.plan.width,self.source_stride,
                  *self.meta,source,self.cache,self.observed,self.sidequat,self.elapsed,self.sampled,self.present,
                  self.all_worlds if world_mask is None else world_mask,scalar.wp_real(timestep),int(force),self.events],device=self.device,stream=stream)

    def reset(self, world_mask=None, stream=None):
        self.last_stream=(stream or wp.get_stream(self.device)) if self.observed.device.is_cuda else None
        wp.launch(reset_kernel,dim=self.nworld,inputs=[len(self.plan.specs),self.plan.width,
                  self.all_worlds if world_mask is None else world_mask,self.elapsed,self.present,self.observed,self.cache,self.sidequat],device=self.device,stream=stream)

    def configure_packing(self, profile="recipe"):
        if profile not in ("legacy", "recipe"):
            raise ValueError("unknown observation packing profile")
        self.packing_profile = profile

    def pack(self, world=0, images=None, *, values=None):
        if not 0 <= world < self.nworld: raise IndexError(world)
        if values is None:
            if self.last_stream is not None:wp.synchronize_stream(self.last_stream)
            values = self.observed.numpy().reshape(self.nworld,-1)[world]
        if self.packing_profile == "recipe":
            if self._packing_recipe is None or not self._packing_recipe.matches(self.plan):
                self._packing_recipe = PackingRecipe.try_create(self.plan)
            if self._packing_recipe is not None:
                return self._packing_recipe.pack(values, images)
        return pack_observations(self.plan, values, images=images)
@wp.kernel
def capture_scene_kernel(ncamera: int, scene_stride: int,
                         source: wp.array(dtype=scalar.wp_real), scene: wp.array(dtype=scalar.wp_real),
                         periods: wp.array(dtype=scalar.wp_real), enabled: wp.array(dtype=wp.int32),
                         elapsed: wp.array(dtype=scalar.wp_real), sampled: wp.array(dtype=wp.int32),
                         generation: wp.array(dtype=wp.int32), parents: wp.array(dtype=wp.int32),
                         parent_generation: wp.array(dtype=wp.int32), captured: wp.array(dtype=wp.int32),
                         mask: wp.array(dtype=wp.int32),
                         timestep: scalar.wp_real, force: int):
    world=wp.tid()
    if mask[world]==0: return
    for camera in range(ncamera):
        s=world*ncamera+camera
        if enabled[camera]!=0:
            e=elapsed[s]+timestep
            take=(sampled[s]==0 and periods[camera]>=e) or force!=0
            fallback=e>=periods[camera] and sampled[s]==0
            if take or fallback:
                parent=parents[camera]
                if parent>=0:
                    ps=world*ncamera+parent
                    for j in range(scene_stride):scene[s*scene_stride+j]=scene[ps*scene_stride+j]
                    parent_generation[s]=generation[ps] if captured[ps]!=0 else 0
                else:
                    for j in range(scene_stride):scene[s*scene_stride+j]=source[world*scene_stride+j]
                    parent_generation[s]=0
                generation[s]+=1
                captured[s]=1
                sampled[s]=1
            if e>=periods[camera]:
                e=e%periods[camera]
                sampled[s]=0
            elapsed[s]=e

class DeviceSampledScenes:
    """Capture derived GPU scenes at original camera sample times.

    Download/render occurs later using the retained sample event. RGB and raw
    depth share the original RGB render event; segmentation retains its own event.
    """
    def __init__(self,image_specs,scene_layout,nworld=1,device="cuda:0",segmentation_mappings=None):
        from .sampled_images import image_descriptor, segmentation_mapping
        self.descriptors=tuple(image_descriptor(spec) for spec in image_specs)
        self.segmentation_mappings=segmentation_mapping(self.descriptors,segmentation_mappings)
        self.specs=tuple(image_specs);self.layout=scene_layout;self.nworld=nworld;self.device=device
        self.stride=max((offset+int(np.prod(shape)) for offset,shape in scene_layout.values()),default=0)
        self.ncamera=len(image_specs)
        self.last_stream=None
        n=nworld*self.ncamera
        self.periods=wp.array([s["period"] for s in image_specs],dtype=scalar.wp_real,device=device)
        self.enabled=wp.array([int(s["enabled"]) for s in image_specs],dtype=wp.int32,device=device)
        self.elapsed=wp.zeros(n,dtype=scalar.wp_real,device=device)
        self.sampled=wp.zeros(n,dtype=wp.int32,device=device)
        self.generation=wp.zeros(n,dtype=wp.int32,device=device)
        indexes={spec["name"]:i for i,spec in enumerate(image_specs)}
        self.parents=wp.array([indexes[camera+"_image"] if kind=="depth" else -1 for camera,kind in self.descriptors],dtype=wp.int32,device=device)
        self.parent_generation=wp.zeros(n,dtype=wp.int32,device=device)
        self.captured=wp.zeros(n,dtype=wp.int32,device=device)
        self.scenes=wp.zeros(n*self.stride,dtype=scalar.wp_real,device=device)
        self.mask=wp.array([1]*nworld,dtype=wp.int32,device=device)
        self.rendered_generation=np.zeros((nworld,self.ncamera),dtype=np.int32)
        self.images=[[np.zeros(tuple(s["shape"])) for s in self.specs] for _ in range(nworld)]

    def update(self,source,timestep,force=False,world_mask=None,stream=None):
        if not np.isfinite(timestep) or timestep<=0:raise ValueError("sampling timestep must be positive finite")
        if world_mask is not None and (world_mask.dtype!=wp.int32 or world_mask.ndim!=1 or world_mask.size!=self.nworld or world_mask.device!=self.scenes.device):
            raise ValueError("world mask must be a same-device int32 vector")
        if source.dtype!=scalar.wp_real or source.ndim!=1 or source.device!=self.scenes.device or source.size!=self.nworld*self.stride:
            raise ValueError("scene source must be flat FP64 with exact world stride")
        self.last_stream=(stream or wp.get_stream(self.device)) if self.scenes.device.is_cuda else None
        wp.launch(capture_scene_kernel,dim=self.nworld,inputs=[self.ncamera,self.stride,source,self.scenes,
                  self.periods,self.enabled,self.elapsed,self.sampled,self.generation,self.parents,self.parent_generation,self.captured,
                  self.mask if world_mask is None else world_mask,scalar.wp_real(timestep),int(force)],device=self.device,stream=stream)

    def render_at_boundary(self,bridge,world=0,image_convention="opengl",scene_option=None, *, snapshot=None):
        from .sampled_images import render_sampled_images
        if not 0<=world<self.nworld:raise IndexError(world)
        if snapshot is None:
            if self.last_stream is not None:wp.synchronize_stream(self.last_stream)
            snapshot=(self.generation.numpy().reshape(self.nworld,self.ncamera),
                      self.scenes.numpy().reshape(self.nworld,self.ncamera,self.stride),
                      self.parent_generation.numpy().reshape(self.nworld,self.ncamera))
        generations,scenes,parents=(value[world] for value in snapshot)
        render_sampled_images(self.specs,self.descriptors,self.segmentation_mappings,self.layout,
                             scenes,generations,self.rendered_generation[world],self.images[world],
                             bridge,image_convention,scene_option,
                             parents)
        return {s["name"]:self.images[world][i] for i,s in enumerate(self.specs)}

    def reset(self,world_mask=None,stream=None):
        if self.last_stream is not None:wp.synchronize_stream(self.last_stream)
        selected=np.ones(self.nworld,dtype=np.int32) if world_mask is None else world_mask.numpy()
        # reset_kernel width zero resets clocks/presence without touching sampled.
        dummy=self.captured
        wp.launch(reset_kernel,dim=self.nworld,inputs=[self.ncamera,0,self.mask if world_mask is None else world_mask,
                  self.elapsed,dummy,self.scenes,self.scenes,self.scenes],device=self.device,stream=stream)
        for world in range(self.nworld):
            if selected[world]:
                for i,spec in enumerate(self.specs):self.images[world][i]=np.zeros(tuple(spec["shape"]))
                self.rendered_generation[world]=self.generation.numpy().reshape(self.nworld,self.ncamera)[world]
