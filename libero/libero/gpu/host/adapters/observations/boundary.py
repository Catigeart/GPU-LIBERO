"""Action/observation-boundary predicates and direct-derived-data rendering.

These bridges do not call mj_forward, mj_step, or sim.forward. A read must be a
single runtime snapshot, even when its fields intentionally have different epochs.
"""
from contextlib import contextmanager
from types import SimpleNamespace
from collections import namedtuple
from collections.abc import Mapping
ContactPair = namedtuple("ContactPair", "geom1 geom2")
import numpy as np

GOAL_FIELDS = ("qpos", "body_xpos", "body_xquat", "site_xpos", "site_xmat", "contact", "ncon")
RENDER_FIELDS = ("xpos", "xmat", "geom_xpos", "geom_xmat", "site_xpos", "site_xmat",
                 "cam_xpos", "cam_xmat", "light_xpos", "light_xdir", "subtree_com")

class _SnapshotData:
    def __init__(self, snapshot, model, load_fields=None):
        self._snapshot = {}; self._model = model
        self._load_fields = load_fields
        self._include(snapshot)

    def _include(self, snapshot):
        snapshot = dict(snapshot)
        if "contact" in snapshot:
            contacts=snapshot["contact"];ncon=int(np.asarray(snapshot["ncon"]).reshape(-1)[0])
            if isinstance(contacts,Mapping):
                if len(contacts["geom1"])!=len(contacts["geom2"]):raise ValueError("contact ID arrays have different capacities")
                contacts=tuple(ContactPair(int(a),int(b)) for a,b in zip(contacts["geom1"],contacts["geom2"]))
            elif isinstance(contacts,np.ndarray) and contacts.dtype.names:
                contacts=tuple(ContactPair(int(c["geom1"]),int(c["geom2"])) for c in contacts)
            else:
                contacts=tuple(ContactPair(int(c.geom1),int(c.geom2)) for c in contacts)
            if ncon<0 or ncon>len(contacts):raise ValueError("contact count exceeds published contact capacity")
            snapshot["contact"]=contacts
            snapshot["ncon"]=ncon
        self._snapshot.update(snapshot)

    def __getattr__(self, name):
        if name not in self._snapshot and name in GOAL_FIELDS and self._load_fields is not None:
            names = ("contact", "ncon") if name in ("contact", "ncon") else (name,)
            self._include(self._load_fields(names))
        if name not in self._snapshot:
            raise AttributeError(f"field {name} was not included in the boundary snapshot")
        return self._snapshot[name]
    def get_site_xpos(self, name):
        return self.site_xpos[self._model.site_name2id(name)]
    def get_site_xmat(self, name):
        return self.site_xmat[self._model.site_name2id(name)].reshape(3, 3)
    def get_body_xpos(self, name):
        return self.body_xpos[self._model.body_name2id(name)]
    def get_body_xquat(self, name):
        return self.body_xquat[self._model.body_name2id(name)]

class _VisualModel:
    def __init__(self, model):
        self._model = model
        metadata=getattr(model,"_metadata",None)
        source=metadata["site_rgba"] if metadata is not None and "site_rgba" in metadata else model.site_rgba
        self.site_rgba = np.array(source, copy=True)
    def __getattr__(self, name):
        return getattr(self._model, name)

@contextmanager
def boundary_predicates(env, facade, *, fields=GOAL_FIELDS):
    """Pin original ObjectState/predicate reads to one coherent boundary snapshot.

    Execute the original check_success/post_process methods inside this context in
    their original order. No robot/controller calculations may run inside it. The
    caller owns environment serialization; one env cannot be stepped concurrently.
    Visual site-alpha modifications are captured and published on exit, including
    when a predicate itself mutates object visual-affordance state.
    """
    # The batch owns one synchronized sample event and caches incremental reads.
    # Prefetch hints may underestimate a custom callback's read set. Complete
    # those reads here while serialization is held, without rerunning callbacks.
    fields = tuple(fields)
    if "contact" in fields and "ncon" not in fields:
        fields += ("ncon",)
    snapshot = facade.snapshot_fields(fields)
    model = _VisualModel(facade.model)
    old_sim = env.sim
    view = SimpleNamespace(model=model, data=_SnapshotData(snapshot, model, facade.snapshot_fields))
    before_rgba = model.site_rgba.copy()
    env.sim = view
    try:
        yield view
    finally:
        # A retained view must never lazily fetch from a later simulation event.
        view.data._load_fields = None
        env.sim = old_sim
        if not np.array_equal(before_rgba, model.site_rgba):
            facade.update_visual_parameters({"site_rgba": model.site_rgba})

def close_native_renderer(renderer):
    """Release the local MuJoCo 2.3 renderer as well as newer close() APIs."""
    close=getattr(renderer,"close",None)
    if close is not None:
        close()
        return
    # Local 2.3.7 Renderer has no close/context-manager methods. Its source owns
    # these two resources, each with an explicit free() method.
    gl_context=renderer._gl_context
    try:
        gl_context.make_current()
        renderer._mjr_context.free()
    finally:
        gl_context.free()

class DerivedRenderBridge:
    """Render an already-published scene; never recompute geometry from qpos.

    A dedicated native MjData is a scene-input container only. Input must include
    every required derived field, captured at the camera observable's sample event.
    It must not be an action-end state substituted for an earlier sampled state.
    """
    def __init__(self, native_model, width=128, height=128, max_geom=10000):
        import mujoco
        if int(native_model.ntendon) or int(getattr(native_model, "nflex", 0)):
            raise NotImplementedError("rendering tendon/flex paths requires a derived-data schema extension")
        from .render_model import clone_render_model
        # Independent graphics owner: rgba and framebuffer edits must not mutate
        # the physical intake MJB or its immutable parameter-array readset.
        native_model=clone_render_model(native_model)
        width,height=int(width),int(height)
        if width<=0 or height<=0:raise ValueError("render dimensions must be positive")
        # Match MjRenderContext's growable offscreen framebuffer for user sizes.
        # These visualization-only model options do not alter physical parameters.
        native_model.vis.global_.offwidth=max(int(native_model.vis.global_.offwidth),width)
        native_model.vis.global_.offheight=max(int(native_model.vis.global_.offheight),height)
        self.model = native_model
        self.data = mujoco.MjData(native_model)
        self.renderer = mujoco.Renderer(native_model, height=height, width=width, max_geom=max_geom)
        self.cam=mujoco.MjvCamera()
        self.cam.fixedcamid=0 if native_model.ncam else -1
        self.cam.type=mujoco.mjtCamera.mjCAMERA_FIXED if native_model.ncam else mujoco.mjtCamera.mjCAMERA_FREE
        if not native_model.ncam:mujoco.mjv_defaultFreeCamera(native_model,self.cam)
        self.vopt=mujoco.MjvOption()
        self.pert=mujoco.MjvPerturb()
        self.scn=self.renderer._scene
        self.con=self.renderer._mjr_context
        self._host_camlight=bool(np.any(np.asarray(native_model.cam_mode)!=int(mujoco.mjtCamLight.mjCAMLIGHT_FIXED))
                                  or np.any(np.asarray(native_model.light_mode)!=int(mujoco.mjtCamLight.mjCAMLIGHT_FIXED)))
        self.closed = False
        self.scene_valid = False

    def publish(self, fields):
        if self.closed:
            raise RuntimeError("renderer is closed")
        prepared = {}
        targets = {}
        try:
            for name in RENDER_FIELDS:
                value = fields.get(name)
                if value is None and name == "xpos": value = fields.get("body_xpos")
                if value is None and name == "xmat": value = fields.get("body_xmat")
                target = getattr(self.data, name)
                if value is None:
                    if target.size: raise KeyError(f"missing published render field: {name}")
                    continue
                value = np.asarray(value)
                if value.size != target.size or value.dtype != target.dtype:
                    raise ValueError(f"render shape/dtype mismatch: {name}")
                if not np.all(np.isfinite(value)):
                    raise ValueError(f"nonfinite render field: {name}")
                prepared[name] = value.reshape(target.shape).copy()
                targets[name] = target
            # Retain model visual parameters at the same camera sample event as
            # geometry. A later post_process alpha toggle must not change old RGB.
            for name in ("site_rgba", "geom_rgba"):
                if name not in fields: continue
                target = getattr(self.model, name)
                value = np.asarray(fields[name])
                if value.size != target.size or not np.all(np.isfinite(value)):
                    raise ValueError(f"invalid sampled visual field: {name}")
                converted = value.astype(target.dtype).reshape(target.shape)
                if not np.array_equal(converted.astype(value.dtype).reshape(value.shape),value):
                    raise ValueError(f"sampled visual field is not losslessly representable: {name}")
                prepared[name] = converted.copy()
                targets[name] = target
        except Exception:
            self.scene_valid = False
            raise
        self.scene_valid = False
        try:
            for name,value in prepared.items():
                np.copyto(targets[name],value,casting="no")
        except Exception:
            # Native scene storage may now be partially changed. Block use until
            # a complete validated publication succeeds.
            raise
        if self._host_camlight:
            import mujoco
            # Scene-only ABI preparation for tracking/target cameras and lights.
            # body/subtree fields above are actual GPU publications. This helper
            # writes camera/light transforms only; no physics is recomputed.
            mujoco.mj_camlight(self.model,self.data)
        self.scene_valid = True
    def render(self, camera_name=None, depth=False, segmentation=False, scene_option=None):
        """Return the original MjSim bottom-up RGB/raw-depth or (type, id) IDs.

        mjr_readPixels supplies raw normalized FP32 depth. Renderer.render() is
        intentionally not used for depth because it converts values to meters.
        """
        import mujoco
        if self.closed:raise RuntimeError("renderer is closed")
        if not self.scene_valid:raise RuntimeError("no complete valid scene publication")
        renderer=self.renderer
        renderer._gl_context.make_current()
        camera=self.cam
        if camera_name is not None:
            if isinstance(camera_name,str):
                camera_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_CAMERA,camera_name)
                if camera_id<0:raise ValueError("No camera with name "+camera_name)
            else:camera_id=int(camera_name)
            if camera_id<-1 or camera_id>=self.model.ncam:raise ValueError("camera ID outside model")
            camera.fixedcamid=camera_id
            camera.type=mujoco.mjtCamera.mjCAMERA_FREE if camera_id==-1 else mujoco.mjtCamera.mjCAMERA_FIXED
        # None retains the context camera, matching original binding_utils.
        mujoco.mjv_updateScene(self.model,self.data,scene_option or self.vopt,self.pert,camera,
                              mujoco.mjtCatBit.mjCAT_ALL,renderer._scene)
        scene=renderer._scene
        flags=scene.flags.copy()
        try:
            if segmentation:
                scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT]=True
                scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR]=True
            mujoco.mjr_render(renderer._rect,scene,renderer._mjr_context)
            rgb=np.empty((renderer.height,renderer.width,3),dtype=np.uint8)
            raw_depth=np.empty((renderer.height,renderer.width),dtype=np.float32) if depth else None
            mujoco.mjr_readPixels(rgb,raw_depth,renderer._rect,renderer._mjr_context)
            if segmentation:
                encoded=rgb.astype(np.uint32)
                encoded=encoded[:,:,0]+encoded[:,:,1]*256+encoded[:,:,2]*65536
                encoded[encoded>=scene.ngeom+1]=0
                ids=np.full((scene.ngeom+1,2),-1,dtype=np.int32)
                for geom in scene.geoms[:scene.ngeom]:
                    if geom.segid!=-1:ids[geom.segid+1]=(geom.objtype,geom.objid)
                rgb=ids[encoded]
            return (rgb,raw_depth) if depth else rgb
        finally:
            np.copyto(scene.flags,flags)

    def render_rgb(self,camera_name,image_convention="opengl",scene_option=None,depth=False):
        if image_convention not in ("opengl","opencv"):raise ValueError("unknown image convention")
        result=self.render(camera_name,depth=depth,scene_option=scene_option)
        def orient(value):return value.copy() if image_convention=="opengl" else value[::-1].copy()
        if depth:return tuple(orient(value) for value in result)
        return orient(result)

    def render_segmentation(self,camera_name,image_convention="opengl",scene_option=None):
        if image_convention not in ("opengl","opencv"):raise ValueError("unknown image convention")
        result=self.render(camera_name,segmentation=True,scene_option=scene_option)
        return result.copy() if image_convention=="opengl" else result[::-1].copy()

    def upload_texture(self,tex_id):
        import mujoco
        if self.closed:raise RuntimeError("renderer is closed")
        self.renderer._gl_context.make_current()
        mujoco.mjr_uploadTexture(self.model,self.con,int(tex_id))

    def close(self):
        if not self.closed:
            try:
                close_native_renderer(self.renderer)
            finally:
                self.closed = True

    def __enter__(self): return self
    def __exit__(self, *args): self.close()
