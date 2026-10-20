"""New case-specific D pose adapter for genuinely captured C roots.

Original two-model FrozenModel/pins are untouched. This adapter has no inherited
acceptance; exact captured static read-set and model identity are mandatory.
"""
from __future__ import annotations
import libero.gpu.precision as scalar
import hashlib,json
from types import SimpleNamespace, MappingProxyType
import numpy as np
import warp as wp
wp.set_module_options({"enable_backward": False, "fuse_fp": True, "fast_math": False})
from libero.gpu.reference.position.body_kinematics.workspace import STATIC, _read_set_hash
from libero.gpu.reference.position.body_kinematics.kernel import Model, Data, Diagnostic, _compute, _commit
from libero.gpu.reference.position.kinematics_frames.frames import MODEL_FIELDS, FLOAT_FIELDS, _tail_static_sha256, _status, _body, _objects, _zero_frame, _zero_tmp, _copy_position, _copy_matrix, _position, _matrix

class CapturedStatic:
    __slots__=("dims","arrays","readset_sha256","model_source_sha256","native_frame_static","actual_model_binary_sha256","sealed_frame_sha256","_sealed")
    def __delattr__(self,name):
        raise AttributeError("captured model attributes cannot be deleted")
    def __setattr__(self,name,value):
        if getattr(self,"_sealed",False):raise AttributeError("captured model identity is immutable")
        object.__setattr__(self,name,value)
    def __init__(self, arrays,dims,*,expected_readset_sha256,model_source_sha256,model_binary_bytes):
        # Actual saved original MJB bytes, not a caller-declared 64-character name.
        if not isinstance(model_binary_bytes,bytes) or not model_binary_bytes or hashlib.sha256(model_binary_bytes).hexdigest()!=model_source_sha256:
            raise ValueError("actual captured model bytes/SHA required")
        import tempfile,os,mujoco
        fd,path=tempfile.mkstemp(suffix=".mjb")
        try:
            with os.fdopen(fd,"wb") as f:f.write(model_binary_bytes)
            native=mujoco.MjModel.from_binary_path(path)
        finally:
            os.unlink(path)
        native_dims={k:int(getattr(native,k)) for k in ("nq","nbody","njnt","nmocap")}
        if native_dims!=dict(dims):raise ValueError("actual compiled MJB dimensions differ")
        native_static={k:np.ascontiguousarray(getattr(native,k),dtype=np.dtype(dt)) for k,(dt,_) in STATIC.items()}
        if _read_set_hash(native_static,native_dims)!=expected_readset_sha256:
            raise ValueError("actual compiled MJB static source mismatch")
        self.native_frame_static=MappingProxyType({k:np.ascontiguousarray(getattr(native,k),dtype=np.float64 if k in FLOAT_FIELDS else np.int32) for k in MODEL_FIELDS})
        for v in self.native_frame_static.values():v.flags.writeable=False
        self.actual_model_binary_sha256=hashlib.sha256(model_binary_bytes).hexdigest()
        self.sealed_frame_sha256=_tail_static_sha256(self.native_frame_static)
        self.dims=MappingProxyType(dict(dims));self.arrays=MappingProxyType({k:np.array(v,copy=True,order="C") for k,v in arrays.items()});n=self.dims
        if set(n)!={"nq","nbody","njnt","nmocap"} or any(type(v)!=int or v<0 for v in n.values()) or n["nbody"]<1:
            raise ValueError("captured model dimensions")
        if set(self.arrays)!=set(STATIC):raise ValueError("complete captured kinematics readset")
        for key,(dt,shape) in STATIC.items():
            a=self.arrays[key]
            if not isinstance(a,np.ndarray) or a.dtype!=np.dtype(dt) or a.shape!=shape(n) or not a.flags.c_contiguous or (dt=="float64" and not np.isfinite(a).all()):raise ValueError("static schema "+key)
        if _read_set_hash(self.arrays,n)!=expected_readset_sha256:raise ValueError("captured static readset SHA")
        for a in self.arrays.values():a.flags.writeable=False
        self.readset_sha256=expected_readset_sha256;self.model_source_sha256=model_source_sha256
        a=self.arrays
        if int(a["body_parentid"][0])!=0 or np.any(a["body_jntnum"]<0) or np.any(a["body_mocapid"]<-1) or np.any(a["body_mocapid"]>=n["nmocap"]):raise ValueError("captured body topology")
        for b in range(1,n["nbody"]):
            if not 0<=int(a["body_parentid"][b])<b:raise ValueError("captured parent order")
            count=int(a["body_jntnum"][b]);start=int(a["body_jntadr"][b])
            if count and not 0<=start<start+count<=n["njnt"]:raise ValueError("captured joint range")
        for j,t in enumerate(a["jnt_type"]):
            if int(t) not in (0,1,2,3):raise ValueError("captured source joint kind")
            q=int(a["jnt_qposadr"][j]);size={0:7,1:4,2:1,3:1}[int(t)]
            if not 0<=q<q+size<=n["nq"]:raise ValueError("captured qpos span")

        object.__setattr__(self,"_sealed",True)

class CapturedBodyProducer:
    def __init__(self, model:CapturedStatic, capacity:int, device='cuda:0'):
        if not isinstance(model,CapturedStatic):raise TypeError('requires exact captured static')
        if not isinstance(capacity,int) or capacity<=0:raise ValueError('capacity')
        self.model=model;self.capacity=capacity;self.device=device
        if _read_set_hash(model.arrays,model.dims)!=model.readset_sha256:raise ValueError("static upload SHA drift")
        upload_dims=dict(model.dims);upload_arrays={k:np.array(v,copy=True,order="C") for k,v in model.arrays.items()}
        self.actual_upload_readset_sha256=_read_set_hash(upload_arrays,upload_dims)
        if self.actual_upload_readset_sha256!=model.readset_sha256:raise ValueError("actual static upload snapshot SHA")
        n=upload_dims;nb=n['nbody'];nj=n['njnt'];nm=n['nmocap'];nq=n['nq']
        m=Model()
        for target,source,dtype in (('qpos0','qpos0',scalar.wp_real),('parent','body_parentid',wp.int32),('jnum','body_jntnum',wp.int32),('jadr','body_jntadr',wp.int32),('mocapid','body_mocapid',wp.int32),('bpos','body_pos',scalar.wp_real),('bquat','body_quat',scalar.wp_real),('jtype','jnt_type',wp.int32),('qadr','jnt_qposadr',wp.int32),('jaxis','jnt_axis',scalar.wp_real),('jpos','jnt_pos',scalar.wp_real)):
            setattr(m,target,wp.array(np.ascontiguousarray(upload_arrays[source].reshape(-1)),dtype=dtype,device=device))
        m.nq=nq;m.nb=nb;m.nj=nj;m.nm=nm;self.static=m
        d=Data();g=Diagnostic()
        def f(shape):return wp.zeros(shape,scalar.wp_real,device=device)
        def i(shape):return wp.zeros(shape,wp.int32,device=device)
        d.active=i(capacity);d.qpos=f((capacity,max(1,nq)));d.mocap_pos=f((capacity,max(1,nm*3)));d.mocap_quat=f((capacity,max(1,nm*4)))
        d.qtmp=f((capacity,max(1,nq)));d.mqtmp=f((capacity,max(1,nm*4)))
        d.xpos=f((capacity,nb*3));d.xquat=f((capacity,nb*4));d.xmat=f((capacity,nb*9));d.xanchor=f((capacity,max(1,nj*3)));d.xaxis=f((capacity,max(1,nj*3)))
        d.world_xipos=f((capacity,3));d.world_ximat=f((capacity,9));d.status=i(capacity)
        for name,size in (('norm_before',nj),('mocap_before',nm),('pre_pos',nb*3),('pre_quat',nb*4),('joint_pos',nj*3),('joint_quat',nj*4),('body_norm_before',nb),('trig_arg',nj),('trig_sin',nj),('trig_cos',nj)):
            setattr(g,name,f((capacity,max(1,size))))
        for name,size in (('body_route',nb),('joint_route',nj),('body_mat_identity',nb),('joint_rot_identity',nj),('hinge_zero',nj),('offcenter_rot_identity',nj),('norm_branch',nj),('mocap_branch',nm),('body_norm_branch',nb)):
            setattr(g,name,i((capacity,max(1,size))))
        self.data=d;self.diag=g
        for name in ('active','qpos','mocap_pos','mocap_quat','xpos','xquat','xmat','xanchor','xaxis','world_xipos','world_ximat','status'):
            setattr(self,name,getattr(d,name))
    def launch(self,stream=None):
        wp.launch(_compute,dim=self.capacity,inputs=[self.static,self.data,self.diag],device=self.device,stream=stream)
        wp.launch(_commit,dim=self.capacity,inputs=[self.static,self.data,self.diag],device=self.device,stream=stream)

@wp.kernel(enable_backward=False)
def _held_body(active: wp.array(dtype=wp.int32), body_xpos: wp.array(dtype=scalar.wp_real),
          body_xquat: wp.array(dtype=scalar.wp_real), body_xmat: wp.array(dtype=scalar.wp_real),
          body_ipos: wp.array(dtype=scalar.wp_real), body_iquat: wp.array(dtype=scalar.wp_real),
          body_sameframe: wp.array(dtype=wp.int32), xipos: wp.array(dtype=scalar.wp_real),
          ximat: wp.array(dtype=scalar.wp_real), body_tmpquat: wp.array(dtype=scalar.wp_real), nbody: int):
    tid = wp.tid()
    w = tid // nbody
    i = tid - w * nbody
    if active[w] != 1:
        return
    if i == 0:
        _zero_frame(xipos, ximat, tid)
        _zero_tmp(body_tmpquat, tid)
        ximat[9 * tid] = scalar.wp_real(1.0)
        ximat[9 * tid + 4] = scalar.wp_real(1.0)
        ximat[9 * tid + 8] = scalar.wp_real(1.0)
    elif body_sameframe[i] == 1:
        _copy_position(body_xpos, xipos, tid, tid)
        _copy_matrix(body_xmat, ximat, tid, tid)
        _zero_tmp(body_tmpquat, tid)
    else:
        _position(body_xpos, body_xmat, body_ipos, xipos, tid, i, tid)
        _matrix(body_xquat, body_iquat, ximat, body_tmpquat, tid, i, tid)

@wp.kernel(enable_backward=False)
def _held_objects(active: wp.array(dtype=wp.int32), body_xpos: wp.array(dtype=scalar.wp_real),
             body_xquat: wp.array(dtype=scalar.wp_real), body_xmat: wp.array(dtype=scalar.wp_real),
             xipos: wp.array(dtype=scalar.wp_real), ximat: wp.array(dtype=scalar.wp_real),
             bodyid: wp.array(dtype=wp.int32), local_pos: wp.array(dtype=scalar.wp_real),
             local_quat: wp.array(dtype=scalar.wp_real), sameframe: wp.array(dtype=wp.int32),
             out_pos: wp.array(dtype=scalar.wp_real), out_mat: wp.array(dtype=scalar.wp_real),
             tmpquat: wp.array(dtype=scalar.wp_real),
             nbody: int, nobject: int):
    tid = wp.tid()
    w = tid // nobject
    i = tid - w * nobject
    if active[w] != 1:
        return
    b = w * nbody + bodyid[i]
    mode = sameframe[i]
    if mode == 0:
        _position(body_xpos, body_xmat, local_pos, out_pos, b, i, tid)
        _matrix(body_xquat, local_quat, out_mat, tmpquat, b, i, tid)
    elif mode == 1:
        _copy_position(body_xpos, out_pos, b, tid)
        _copy_matrix(body_xmat, out_mat, b, tid)
        _zero_tmp(tmpquat, tid)
    else:
        _copy_position(xipos, out_pos, b, tid)
        _copy_matrix(ximat, out_mat, b, tid)
        _zero_tmp(tmpquat, tid)


class CapturedFrameProducer:
    def __init__(self, model: dict, nworld: int, *, expected_static_sha256, device="cuda:0"):
        if nworld < 1:
            raise ValueError("nworld must be positive")
        model={k:np.array(v,copy=True,order="C") for k,v in model.items()}
        missing = set(MODEL_FIELDS) - set(model)
        extra = set(model) - set(MODEL_FIELDS)
        if missing or extra:
            raise ValueError(f"static field set differs: missing={sorted(missing)}, extra={sorted(extra)}")
        prepared = {}
        count = None
        for name in MODEL_FIELDS:
            a = np.asarray(model[name])
            dtype = np.dtype("float64") if name in FLOAT_FIELDS else np.dtype("int32")
            if a.dtype != dtype or not a.flags.c_contiguous:
                raise TypeError(f"{name}: require contiguous {dtype}")
            if name == "body_ipos":
                count = a.shape[0]
            prepared[name] = a
        b = count
        g = prepared["geom_bodyid"].shape[0]
        s = prepared["site_bodyid"].shape[0]
        shapes = {"body_ipos": (b, 3), "body_iquat": (b, 4), "body_sameframe": (b,),
                  "geom_bodyid": (g,), "geom_pos": (g, 3), "geom_quat": (g, 4), "geom_sameframe": (g,),
                  "site_bodyid": (s,), "site_pos": (s, 3), "site_quat": (s, 4), "site_sameframe": (s,)}
        for name, shape in shapes.items():
            if prepared[name].shape != shape:
                raise ValueError(f"{name}: expected {shape}, got {prepared[name].shape}")
        if b < 1 or g < 1:
            raise ValueError("nonempty body, geom, site arrays required")
        if not np.all(np.isin(prepared["body_sameframe"], (0, 1))):
            raise ValueError("body_sameframe must be 0/1")
        for kind in ("geom", "site"):
            if not np.all(np.isin(prepared[f"{kind}_sameframe"], (0, 1, 2))):
                raise ValueError(f"{kind}_sameframe must be 0/1/2")
            ids = prepared[f"{kind}_bodyid"]
            if np.any(ids < 0) or np.any(ids >= b):
                raise ValueError(f"{kind}_bodyid out of bounds")
        for name in FLOAT_FIELDS:
            if not np.all(np.isfinite(prepared[name])):
                raise ValueError(f"{name} nonfinite")
        self.tail_static_sha256 = _tail_static_sha256(prepared)
        if self.tail_static_sha256 != expected_static_sha256:
            raise ValueError("captured frame static readset mismatch")
        self.device = wp.get_device(device)
        self.nworld, self.nbody, self.ngeom, self.nsite = nworld, b, g, s
        self.static = {name: wp.array(a.reshape(-1), dtype=(scalar.wp_real if name in FLOAT_FIELDS else wp.int32), device=self.device)
                       for name, a in prepared.items()}
        self.body_xpos = wp.empty(nworld * b * 3, dtype=scalar.wp_real, device=self.device)
        self.body_xquat = wp.empty(nworld * b * 4, dtype=scalar.wp_real, device=self.device)
        self.body_xmat = wp.empty(nworld * b * 9, dtype=scalar.wp_real, device=self.device)
        self.active_world = wp.empty(nworld, dtype=wp.int32, device=self.device)
        self.xipos = wp.empty(nworld * b * 3, dtype=scalar.wp_real, device=self.device)
        self.ximat = wp.empty(nworld * b * 9, dtype=scalar.wp_real, device=self.device)
        self.geom_xpos = wp.empty(nworld * g * 3, dtype=scalar.wp_real, device=self.device)
        self.geom_xmat = wp.empty(nworld * g * 9, dtype=scalar.wp_real, device=self.device)
        self.site_xpos = wp.empty(nworld * s * 3, dtype=scalar.wp_real, device=self.device)
        self.site_xmat = wp.empty(nworld * s * 9, dtype=scalar.wp_real, device=self.device)
        self.body_tmpquat = wp.empty(nworld * b * 4, dtype=scalar.wp_real, device=self.device)
        self.geom_tmpquat = wp.empty(nworld * g * 4, dtype=scalar.wp_real, device=self.device)
        self.site_tmpquat = wp.empty(nworld * s * 4, dtype=scalar.wp_real, device=self.device)
        self.status = wp.empty(nworld, dtype=wp.int32, device=self.device)

    def launch(self, *, stream=None):
        wp.launch(_status, dim=self.nworld, inputs=[self.active_world, self.status], device=self.device, stream=stream)
        wp.launch(_held_body, dim=self.nworld * self.nbody,
                  inputs=[self.active_world, self.body_xpos, self.body_xquat, self.body_xmat,
                          self.static["body_ipos"], self.static["body_iquat"], self.static["body_sameframe"],
                          self.xipos, self.ximat, self.body_tmpquat, self.nbody], device=self.device, stream=stream)
        for kind, count in (("geom", self.ngeom), ("site", self.nsite)):
            if count == 0:continue
            wp.launch(_held_objects, dim=self.nworld * count,
                      inputs=[self.active_world, self.body_xpos, self.body_xquat, self.body_xmat,
                              self.xipos, self.ximat, self.static[f"{kind}_bodyid"],
                              self.static[f"{kind}_pos"], self.static[f"{kind}_quat"],
                              self.static[f"{kind}_sameframe"], getattr(self, f"{kind}_xpos"),
                              getattr(self, f"{kind}_xmat"), getattr(self, f"{kind}_tmpquat"),
                              self.nbody, count], device=self.device, stream=stream)


@wp.kernel(enable_backward=False)
def _frame_gate(active:wp.array(dtype=wp.int32), body_status:wp.array(dtype=wp.int32), source_epoch:wp.array(dtype=wp.int64), expected_epoch:wp.array(dtype=wp.int64), first_error:wp.array(dtype=wp.int32), gate:wp.array(dtype=wp.int32)):
    w=wp.tid()
    gate[w]=0
    if active[w]==0:return
    if active[w]!=1:
        if first_error[w]==0:first_error[w]=108
        return
    if first_error[w]!=0:return
    if source_epoch[w]<wp.int64(0) or expected_epoch[w]<wp.int64(0) or source_epoch[w]!=expected_epoch[w]:
        first_error[w]=109
        return
    if body_status[w]!=0:
        first_error[w]=body_status[w]
        return
    gate[w]=1

@wp.kernel(enable_backward=False)
def _pose_publish(gate:wp.array(dtype=wp.int32), frame_status:wp.array(dtype=wp.int32), source_epoch:wp.array(dtype=wp.int64), written_epoch:wp.array(dtype=wp.int64), valid:wp.array(dtype=wp.int32),first_error:wp.array(dtype=wp.int32)):
    w=wp.tid()
    valid[w]=0
    if gate[w]!=1:return
    if frame_status[w]!=0:
        if first_error[w]==0:first_error[w]=100+frame_status[w]
        return
    if first_error[w]!=0:return
    written_epoch[w]=source_epoch[w]
    valid[w]=1

@wp.kernel
def _source_static_word(actual:wp.array(dtype=wp.uint32),word:int,expected:wp.uint32,source_active:wp.array(dtype=wp.int32),first_error:wp.array(dtype=wp.int32)):
    w=wp.tid()
    if source_active[w]==1 and actual[word]!=expected:
        wp.atomic_cas(first_error,w,0,111)

@wp.kernel
def _source_active_gate(source:wp.array(dtype=wp.int32),first_error:wp.array(dtype=wp.int32),body_active:wp.array(dtype=wp.int32)):
    w=wp.tid()
    body_active[w]=source[w]
    if first_error[w]!=0:
        body_active[w]=0

class PairedPoseProducer:
    """Actual same-device body→frame chain; host snapshots only at boundaries."""
    __slots__=("stream","stream_handle","stream_context","body","frames","gate","expected_epoch","source_epoch","written_epoch","valid","first_error","capacity","device","_root_upload_buffers","actual_root_upload_history","_structural_bindings","_bindings_sealed","_static_witnesses","input_active","__weakref__")
    def __init__(self,model:CapturedStatic,frame_static,capacity,*,expected_frame_sha256,stream,device="cuda:0"):
        actual_device=wp.get_device(device)
        if not isinstance(stream,wp.Stream) or stream.device!=actual_device:raise ValueError("same actual device construction stream required")
        self.stream=stream
        self.stream_handle=int(stream.cuda_stream)
        self.stream_context=int(actual_device.context)
        with wp.ScopedStream(stream):
            actual_frames={k:np.array(v,copy=True,order="C") for k,v in frame_static.items()}
            if model.sealed_frame_sha256!=expected_frame_sha256 or _tail_static_sha256(model.native_frame_static)!=model.sealed_frame_sha256 or _tail_static_sha256(actual_frames)!=model.sealed_frame_sha256:
                raise ValueError("actual compiled MJB frame static mismatch")
            self.body=CapturedBodyProducer(model,capacity,device)
            self.frames=CapturedFrameProducer(actual_frames,capacity,expected_static_sha256=expected_frame_sha256,device=device)
            if self.frames.nbody!=model.dims["nbody"]:raise ValueError("cross captured model body count")
            self.frames.body_xpos=self.body.xpos.flatten()
            self.frames.body_xquat=self.body.xquat.flatten()
            self.frames.body_xmat=self.body.xmat.flatten()
            self.gate=wp.zeros(capacity,dtype=wp.int32,device=device)
            self.frames.active_world=self.gate
            self.expected_epoch=wp.full(capacity,-1,dtype=wp.int64,device=device)
            self.source_epoch=wp.zeros(capacity,dtype=wp.int64,device=device)
            self.written_epoch=wp.full(capacity,-1,dtype=wp.int64,device=device)
            self.valid=wp.zeros(capacity,dtype=wp.int32,device=device)
            self.first_error=wp.zeros(capacity,dtype=wp.int32,device=device)
            self.capacity=capacity;self.device=device
            self._root_upload_buffers=[]
            self.actual_root_upload_history=[]
            self.input_active=wp.zeros(capacity,dtype=wp.int32,device=device)
            witnesses=[]
            # Independent immutable-upload mirrors are device read-only witnesses.
            # Compare every original static word before any mathematical writer.
            body_sources={'qpos0':'qpos0','parent':'body_parentid','jnum':'body_jntnum','jadr':'body_jntadr','mocapid':'body_mocapid','bpos':'body_pos','bquat':'body_quat','jtype':'jnt_type','qadr':'jnt_qposadr','jaxis':'jnt_axis','jpos':'jnt_pos'}
            source_arrays=[(getattr(self.body.static,target),model.arrays[source]) for target,source in body_sources.items()]
            source_arrays+=[(array,actual_frames[key]) for key,array in self.frames.static.items()]
            for array,original in source_arrays:
                if not isinstance(array,wp.array) or not array.size:continue
                original=np.ascontiguousarray(original,dtype=np.float64 if array.dtype==scalar.wp_real else np.int32)
                logical_bytes=array.size*wp.types.type_size_in_bytes(array.dtype)
                if not array.is_contiguous or logical_bytes%4 or original.nbytes!=logical_bytes:raise ValueError("static word source schema")
                view=wp.array(ptr=array.ptr,dtype=wp.uint32,shape=(logical_bytes//4,),capacity=array.capacity,device=actual_device)
                view._ref=array
                # Frozen host scalar literals from the independently MJB-derived
                # input, not a mutable GPU expected buffer or observed output.
                expected=tuple(int(v) for v in original.reshape(-1).view(np.uint32))
                witnesses.append((view,expected,array))
            self._static_witnesses=tuple(witnesses)
        # Retain the actual upload owners and every structural binding. Dynamic
        # device contents may change; replacing their source owners may not.
        self._structural_bindings=[]
        def keep(path):
            value=self._binding_value(path)
            self._structural_bindings.append((path,value,self._binding_token(value)))
        for name in ("body","frames","capacity","device","stream","stream_handle","stream_context","gate","expected_epoch","source_epoch","written_epoch","valid","first_error","input_active","_static_witnesses"):
            keep((name,))
        for name in ("model","static","data","diag","capacity","device","actual_upload_readset_sha256","active","qpos","mocap_pos","mocap_quat","xpos","xquat","xmat","xanchor","xaxis","world_xipos","world_ximat","status"):
            keep(("body",name))
        for parent in ("static","data","diag"):
            obj=getattr(self.body,parent)
            for name,_constructor in type(obj)._constructors:
                keep(("body",parent,name))
        for name,value in vars(self.frames).items():
            keep(("frames",name))
            if name=="static":
                for key in value:keep(("frames","static",key))
        for i,witness in enumerate(self._static_witnesses):
            for j in range(3):keep(("_static_witnesses",i,j))
        self._structural_bindings=tuple(self._structural_bindings)
        self._bindings_sealed=True
    def __setattr__(self,name,value):
        if getattr(self,"_bindings_sealed",False) and name in ("_structural_bindings","_bindings_sealed","_static_witnesses","__class__","validate_source_bindings","_binding_value","_binding_token","upload_roots","launch","four_leaves"):
            raise AttributeError("producer source bindings are immutable")
        object.__setattr__(self,name,value)
    def __delattr__(self,name):
        if getattr(self,"_bindings_sealed",False) and name in ("_structural_bindings","_bindings_sealed","_static_witnesses","__class__","validate_source_bindings","_binding_value","_binding_token","upload_roots","launch","four_leaves"):
            raise AttributeError("producer source bindings cannot be deleted")
        object.__delattr__(self,name)
    def _binding_value(self,path):
        value=self
        for part in path:
            value=value[part] if isinstance(value,(dict,tuple,list)) else getattr(value,part)
        return value
    @staticmethod
    def _binding_token(value):
        if isinstance(value,wp.array):
            return (int(value.ptr or 0),str(value.dtype),tuple(value.shape),tuple(value.strides),int(value.capacity or 0),str(value.device),int(value.device.context),id(value._ref))
        if isinstance(value,dict):return tuple(sorted(value))
        if isinstance(value,(str,int,float,bool,type(None))):return value
        if hasattr(value,"__dict__"):return (id(value),tuple(sorted(vars(value))))
        return id(value)
    def validate_source_bindings(self):
        for path,owner,token in self._structural_bindings:
            actual=self._binding_value(path)
            if (not isinstance(owner,(str,int,float,bool,type(None))) and actual is not owner) or self._binding_token(actual)!=token:
                raise ValueError("actual producer source binding replacement: "+".".join(map(str,path)))
        if self.body.actual_upload_readset_sha256!=self.body.model.readset_sha256 or self.frames.tail_static_sha256!=self.body.model.sealed_frame_sha256:
            raise ValueError("actual upload provenance mismatch")
    def upload_roots(self,*,active,qpos,mocap_pos,mocap_quat,source_epoch,expected_epoch,stream):
        self.validate_source_bindings()
        if not isinstance(stream,wp.Stream) or int(stream.cuda_stream)!=self.stream_handle or int(stream.device.context)!=self.stream_context:raise ValueError("root upload stream/context mismatch")
        n=self.body.model.dims;N=self.capacity
        values={"active":(active,(N,),np.int32),"qpos":(qpos,(N,n["nq"]),np.float64),"mocap_pos":(mocap_pos,(N,n["nmocap"],3),np.float64),"mocap_quat":(mocap_quat,(N,n["nmocap"],4),np.float64),"source_epoch":(source_epoch,(N,),np.int64),"expected_epoch":(expected_epoch,(N,),np.int64)}
        # Snapshot and bind exactly the bytes used for upload, retaining device
        # staging owners through all subsequent same-stream consumption.
        prepared={}
        for name,(a,shape,dt) in values.items():
            if not isinstance(a,np.ndarray) or a.dtype!=dt or a.shape!=shape or not a.flags.c_contiguous:raise ValueError("actual roots schema "+name)
            prepared[name]=np.array(a,copy=True,order="C")
        binding={name:{"dtype":a.dtype.str,"shape":list(a.shape),"sha256":hashlib.sha256(a.tobytes()).hexdigest(),"bytes":a.nbytes} for name,a in prepared.items()}
        with wp.ScopedStream(stream):
            for name,a in prepared.items():
                if a.size==0:continue
                dst=self.input_active if name=="active" else getattr(self,name) if name in ("source_epoch","expected_epoch") else getattr(self.body,name)
                source=wp.array(a.reshape(dst.shape),dtype=dst.dtype,device=self.device)
                self._root_upload_buffers.append(source)
                wp.copy(dst,source,stream=stream)
        self.actual_root_upload_history.append(binding)
    def launch(self,stream):
        self.validate_source_bindings()
        if not isinstance(stream,wp.Stream) or int(stream.cuda_stream)!=self.stream_handle or int(stream.device.context)!=self.stream_context:raise ValueError("producer stream/context mismatch")
        for actual,expected,_owner in self._static_witnesses:
            for word,value in enumerate(expected):
                wp.launch(_source_static_word,dim=self.capacity,inputs=[actual,word,wp.uint32(value),self.input_active,self.first_error],device=self.device,stream=stream)
        wp.launch(_source_active_gate,dim=self.capacity,inputs=[self.input_active,self.first_error,self.body.active],device=self.device,stream=stream)
        self.body.launch(stream=stream)
        wp.launch(_frame_gate,dim=self.capacity,inputs=[self.body.active,self.body.status,self.source_epoch,self.expected_epoch,self.first_error,self.gate],device=self.device,stream=stream)
        self.frames.launch(stream=stream)
        wp.launch(_pose_publish,dim=self.capacity,inputs=[self.gate,self.frames.status,self.source_epoch,self.written_epoch,self.valid,self.first_error],device=self.device,stream=stream)
    def four_leaves(self):
        self.validate_source_bindings()
        # Same resident arrays; consumer must read valid/written_epoch/first_error.
        return {"geom_xpos":self.frames.geom_xpos,"geom_xmat":self.frames.geom_xmat,"body_xipos":self.frames.xipos,"body_ximat":self.frames.ximat}
