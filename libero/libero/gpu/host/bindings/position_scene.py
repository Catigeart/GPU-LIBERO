"""Optional handwritten fusion of the environment observer's fixed-mode scene route."""
import libero.gpu.precision as scalar
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
from pathlib import Path
from types import MethodType
import warp as wp
from libero.gpu.host.build.native import NativePlan


def install_scene(owner,body_xpos,body_xmat,build_root):
    """Cold bind DeviceSceneTransforms; hot update supports caller mask and input pointers."""
    if hasattr(owner,'_cuda_scene_plan') and hasattr(owner,'_scene_cache'):return dict(owner._cuda_scene_plan.manifest,cache='actual input/output bits')
    baseline_update=owner.update
    stages=[]
    # Per-item bit snapshots include actual geometry, static local inputs and
    # retained output. No host version guessing and no CPU hot math.
    owner._scene_cache={}
    for label,count,stride in (('camera',owner.ncam,36),('light',owner.nlight,24)):
        owner._scene_cache[label]=[
            wp.zeros(owner.nworld*count*stride,dtype=scalar.wp_real,device=body_xpos.device),
            *[wp.zeros(owner.nworld*count,dtype=wp.int32,device=body_xpos.device) for _ in range(4)]]

    if owner.ncam:
        stages.append(('libero_scene::camera_cached',[owner.nbody,owner.ncam,body_xpos,body_xmat,
            owner.all_worlds,*owner.camera_static,owner.cam_xpos,owner.cam_xmat,*owner._scene_cache['camera']],'block'))
    if owner.nlight:
        stages.append(('libero_scene::light_cached',[owner.nbody,owner.nlight,body_xpos,body_xmat,
            owner.all_worlds,*owner.light_static,owner.light_xpos,owner.light_xdir,*owner._scene_cache['light']],'block'))
    if not stages:return dict(name='empty_scene',kernels=0)
    plan=NativePlan('handwritten_scene',[(GPU_ROOT / 'cuda/observations/position_scene.cuh')],stages,
        owner.nworld,build_root,device=body_xpos.device)
    def update(self,bp,bm,world_mask=None,stream=None):
        for field,size in ((bp,self.nworld*self.nbody*3),(bm,self.nworld*self.nbody*9)):
            if field.dtype!=scalar.wp_real or field.ndim!=1 or field.size!=size or not field.is_contiguous:
                raise ValueError('scene input must be flat contiguous FP64 body arrays')
        mask=self.all_worlds if world_mask is None else world_mask
        if mask.dtype!=wp.int32 or mask.shape!=(self.nworld,) or mask.device!=bp.device:
            raise ValueError('scene mask must be resident world int32 array')
        def overlaps(a,b):
            return a.size and b.size and a.ptr < b.ptr+b.size*wp.types.type_size_in_bytes(b.dtype) and b.ptr < a.ptr+a.size*wp.types.type_size_in_bytes(a.dtype)
        outputs=[self.cam_xpos,self.cam_xmat,self.light_xpos,self.light_xdir]
        reads=[bp,bm,*self.camera_static,*self.light_static]
        if any(overlaps(a,b) for a in reads for b in outputs) or any(
                overlaps(outputs[i],outputs[j]) for i in range(4) for j in range(i)):
            return baseline_update(bp,bm,world_mask,stream)
        descriptors=[]
        if self.ncam:descriptors.append((*self.camera_static,self.cam_xpos,self.cam_xmat))
        if self.nlight:descriptors.append((*self.light_static,self.light_xpos,self.light_xdir))
        for stage,descriptor in enumerate(descriptors):
            for j,value in enumerate(descriptor):plan.set_pointer(stage,5+j,value,stream)

            plan.set_pointer(stage,2,bp,stream)
            plan.set_pointer(stage,3,bm,stream)
            plan.set_pointer(stage,4,mask,stream)
        plan.launch(stream=stream)
        return {k:getattr(self,k) for k in ('cam_xpos','cam_xmat','light_xpos','light_xdir')}
    owner.update=MethodType(update,owner);owner._cuda_scene_plan=plan
    return dict(plan.manifest,stages=len(stages),scope='fixed-mode observer scene transforms')


def install_scene_from_environment(environment,build_root,*,recapture=True,dataflow_profile='compatible'):
    """Install after GpuTaskComponents assembly, preserving both registered hooks.

    FixedRuntimeObservationBinding owns forward.scene_transform and observe;
    both call this observer's update dynamically, so their identity is retained.
    The environment factory creates its observer after preparing physics.
    """
    components=getattr(environment,'components',environment)
    observer=components.observations
    from .observations import install_environment as install_observations
    observations_manifest=install_observations(environment,Path(build_root)/'observations',
                                               dataflow_profile=dataflow_profile)
    existing=getattr(components,'cuda_scene_manifest',None)
    if existing is not None and getattr(observer.scene_transforms,'_cuda_scene_plan',None) is not None:
        # Initial environment creation and the recursive model assembly hook
        # can both reach this installer. One actual observer owns one plan.
        return dict(existing,installed=True)
    if observer.scene_transforms is None:
        return dict(name='observer_scene',installed=False,reason='camera observations disabled',observations=observations_manifest)
    fields=components.observation_fields()
    manifest=install_scene(observer.scene_transforms,fields['body_xpos'],fields['body_xmat'],build_root)
    runtime=components.physics
    if recapture and hasattr(runtime,'_reset_forward_graph'):
        # The earlier physics preparation graph predates environment observer
        # construction. Capture the actual observer callback in recovery too.
        with wp.ScopedCapture(stream=runtime.stream,force_module_load=True) as capture:
            wp.capture_if(getattr(runtime,'reset_predicate',runtime.recovery.reset_mask),on_true=runtime._reset_forward,stream=runtime.stream)
        runtime._reset_forward_graph=capture.graph
    manifest=dict(manifest,observations=observations_manifest,cache='actual input/output bits')
    components.cuda_scene_manifest=manifest
    # The assembled components retain their real prepared bundle in the bound
    # observation callback. Extend its trace rather than creating another owner.
    binding=getattr(components.observation_fields,'__self__',None)
    bundle=getattr(binding,'prepared',None)
    plan=getattr(observer.scene_transforms,'_cuda_scene_plan',None)
    if bundle is not None and plan is not None:
        if hasattr(bundle,'cuda_native_plans') and plan not in bundle.cuda_native_plans:
            bundle.cuda_native_plans=(*bundle.cuda_native_plans,plan)
        if hasattr(bundle,'cuda_manifests'):
            bundle.cuda_manifests.append(plan.manifest)
        if hasattr(bundle,'cuda_selection_manifest'):
            bundle.cuda_selection_manifest['observer_scene']=dict(manifest,installed=True)
    return dict(manifest,installed=True,callback='FixedRuntimeObservationBinding._scene_forward and GpuObservationAdapter.sample')


def install_scene_from_bundle(bundle,build_root):
    """Bind an already installed observation hook; bare physics has none yet."""
    callback=bundle.runtime.forward.scene_transform
    binding=getattr(callback,'__self__',None)
    observer=getattr(binding,'observer',None)
    if observer is None or observer.scene_transforms is None:
        return dict(name='observer_scene',installed=False,reason='observer constructed after physics preparation')
    fields=binding.fields()
    return install_scene(observer.scene_transforms,fields['body_xpos'],fields['body_xmat'],build_root)
