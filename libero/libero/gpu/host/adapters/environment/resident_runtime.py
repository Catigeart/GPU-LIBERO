"""Actual device-array boundary adapter for a prepared resident runtime.

All numerical stages are explicit source-qualified enqueue callbacks supplied by
assembly. Missing reset/step1/parameter support is rejected before mutation.
Only boundary transfers occur here; no CPU forward, solve or controller math.
"""
import libero.gpu.precision as scalar
from dataclasses import dataclass
from threading import RLock
import numpy as np
from uuid import uuid4
from .facade import UnsupportedGpuOperation


@dataclass(frozen=True)
class ResidentField:
    array: object
    shape: tuple
    world_axis: bool = True

    def __post_init__(self):
        shape=tuple(self.shape)
        if any(isinstance(d,bool) or not isinstance(d,(int,np.integer)) or d<0 for d in shape):
            raise ValueError("nonnegative integer resident field dimensions required")
        object.__setattr__(self,"shape",tuple(int(d) for d in shape))

    @property
    def size(self): return int(np.prod(self.shape,dtype=np.int64)) if self.shape else 1


def _token(array):
    return (id(array),int(array.ptr),str(array.dtype),tuple(array.shape),tuple(array.strides),str(array.device))


class ResidentPhysicsFacadeRuntime:
    physics_backend = "gpu"

    def __init__(self, *, runtime, fields, stream, world=0, nworld=1,
                 synchronize_and_check, close_runtime, state_written=None,
                 reset=None, step1=None, update_parameters=None,
                 update_visual=None, special_readers=None, evidence=None, update_parameters_transaction=None,
                 spatial_queries=None):
        import warp as wp
        self.wp=wp;self.runtime=runtime;self.stream=stream
        self.world,self.nworld=world,nworld
        self._lock=RLock();self._closed=False;self._failed=False;self._pending_uploads=[]
        if isinstance(world,bool) or not isinstance(world,int) or not 0<=world<nworld:
            raise ValueError('invalid resident world index')
        if not stream.device.is_cuda:raise UnsupportedGpuOperation('CUDA resident runtime required')
        if not callable(synchronize_and_check) or not callable(close_runtime):
            raise ValueError('actual status check and release callbacks required')
        self._check,self._close=synchronize_and_check,close_runtime
        self._state_written,self._reset,self._step1=state_written,reset,step1
        self._parameters,self._visual,self._evidence=update_parameters,update_visual,evidence
        self._parameter_transaction=update_parameters_transaction
        self._spatial_queries=spatial_queries
        self.allocation_id=uuid4().hex
        self.fields=dict(fields);self.special_readers=dict(special_readers or {})
        self._tokens={}
        for name,field in self.fields.items():
            if not isinstance(field,ResidentField):raise TypeError(f'field descriptor required: {name}')
            a=field.array
            if a.device!=stream.device:raise ValueError(f'field GPU mismatch: {name}')
            if a.size!=field.size*(nworld if field.world_axis else 1):
                raise ValueError(f'field extent mismatch: {name}')
            # Flat-offset copies are legal only for contiguous actual owners.
            if not a.is_contiguous:raise ValueError(f'field must be contiguous: {name}')
            self._tokens[name]=_token(a)
        for name in ('qpos','qvel','time'):
            if name not in self.fields:raise ValueError(f'state root required: {name}')

    def _open(self):
        if self._closed:raise RuntimeError('resident facade runtime is closed')
        if self._failed:raise RuntimeError('resident boundary mutation failed; rebuild required')
        for name,field in self.fields.items():
            if _token(field.array)!=self._tokens[name]:raise RuntimeError(f'resident root changed: {name}')

    def _required(self, callback, name):
        if not callable(callback):raise UnsupportedGpuOperation(f'actual resident callback missing: {name}')
        return callback

    def runtime_owner_evidence(self):
        """Actual allocation lifetime/schema only; no physics or acceptance claim."""
        with self._lock:
            self._open()
            return {'allocation_id':self.allocation_id,'world':self.world,
                    'nworld':self.nworld,'device':str(self.stream.device),
                    'roots':{name:{'ptr':int(field.array.ptr),
                              'dtype':str(field.array.dtype),
                              'shape':list(field.array.shape),
                              'strides':list(field.array.strides),
                              'world_shape':list(field.shape),
                              'world_axis':field.world_axis}
                             for name,field in self.fields.items()}}

    def read_fields(self,names):
        with self._lock:
            self._open()
            missing=set(names)-set(self.fields)-set(self.special_readers)
            if missing:raise UnsupportedGpuOperation(f'unmapped resident fields: {sorted(missing)}')
            self._check()
            self._pending_uploads.clear()
            result={}
            with self.wp.ScopedStream(self.stream):
                for name in names:
                    if name in self.special_readers:
                        result[name]=self.special_readers[name](self.world)
                        continue
                    field=self.fields[name]
                    flat=field.array.numpy().reshape(-1)
                    offset=self.world*field.size if field.world_axis else 0
                    value=flat[offset:offset+field.size].copy().reshape(field.shape)
                    result[name]=value.item() if not field.shape else value
            return result

    def _bind_spatial_queries(self, queries):
        from .public_spatial import PublicSpatialQueries
        with self._lock:
            self._open()
            if self._spatial_queries is not None:
                raise RuntimeError('public spatial owner already bound')
            if not isinstance(queries, PublicSpatialQueries):
                raise TypeError('actual public spatial query owner required')
            if (queries.bundle.runtime is not self.runtime or queries.stream is not self.stream
                    or queries.nworld != self.nworld
                    or queries.bundle.model_pack.model is not queries.built.native_model):
                raise ValueError('foreign public spatial query owner')
            for name, field in (('qvel', 'qvel'), ('body_pos', 'body_xpos'),
                                ('geom_pos', 'geom_xpos'), ('site_pos', 'site_xpos'),
                                ('com', 'subtree_com')):
                if queries.roots[name] is not self.fields[field].array:
                    raise ValueError('foreign public spatial input root: ' + name)
            self._spatial_queries=queries

    def read_jacobian(self,kind,index,component):
        with self._lock:
            self._open();self._check()
            queries=self._spatial_queries
            if queries is None:raise UnsupportedGpuOperation('actual GPU spatial query owner missing')
            return self._required(getattr(queries,'read_jacobian',None),'GPU jacobian query')(
                self.world,kind,index,component,self.stream)

    def read_velocity(self,kind,index,component):
        with self._lock:
            self._open();self._check()
            queries=self._spatial_queries
            if queries is None:raise UnsupportedGpuOperation('actual GPU spatial query owner missing')
            return self._required(getattr(queries,'read_velocity',None),'GPU velocity query')(
                self.world,kind,index,component,self.stream)

    def _prepare_write(self,values):
        payload={}
        for name,value in values.items():
            if name not in ('time','qpos','qvel','ctrl','qacc_warmstart','qfrc_applied','xfrc_applied') or name not in self.fields:
                raise UnsupportedGpuOperation(f'flat-state write not declared: {name}')
            field=self.fields[name]
            if not field.world_axis:raise ValueError('state field must be per-world')
            if field.array.dtype!=scalar.wp_real:raise ValueError('FP64 state owner required')
            value=np.asarray(value,dtype=np.float64)
            if value.shape!=field.shape or not np.isfinite(value).all():
                raise ValueError(f'invalid state write: {name}')
            payload[name]=np.ascontiguousarray(value.reshape(-1))
        return payload

    def write_fields(self,values):
        with self._lock:
            self._open();self._check();callback=self._required(self._state_written,'state_written')
            payload=self._prepare_write(values)
            # Allocate all host-source transfers before any actual root copy.
            with self.wp.ScopedStream(self.stream):
                sources={name:self.wp.array(value,dtype=scalar.wp_real,device=self.stream.device)
                         for name,value in payload.items()}
            self._pending_uploads.extend(sources.values())
            try:
                for name,source in sources.items():
                    field=self.fields[name]
                    self.wp.copy(field.array,source,dest_offset=self.world*field.size,
                                 count=field.size,stream=self.stream)
                callback(tuple(payload),self.world,self.stream)
            except BaseException:
                self._failed=True;raise

    def write_joint_qpos(self,start,stop,value):
        with self._lock:
            self._open();self._check();callback=self._required(self._state_written,'state_written')
            field=self.fields['qpos']
            value=np.asarray(value,dtype=np.float64)
            if not 0<=start<stop<=field.size or value.shape!=(stop-start,) or not np.isfinite(value).all():
                raise ValueError('invalid joint qpos slice')
            with self.wp.ScopedStream(self.stream):
                source=self.wp.array(value,dtype=scalar.wp_real,device=self.stream.device)
            self._pending_uploads.append(source)
            try:
                self.wp.copy(field.array,source,dest_offset=self.world*field.size+start,count=stop-start,stream=self.stream)
                callback(('qpos',),self.world,self.stream)
            except BaseException:self._failed=True;raise

    def forward(self):
        with self._lock:
            self._open()
            self.runtime.forward.enqueue(self.runtime.intent,self.stream)

    def step(self):
        with self._lock:self._open();self.runtime.enqueue_physics_step()

    def step1(self):
        with self._lock:self._open();self._required(self._step1,'step1')(self.world,self.stream)

    def reset(self):
        with self._lock:self._open();self._required(self._reset,'reset')(self.world,self.stream)

    def update_model_parameters(self,changes):
        with self._lock:
            self._open();self._check()
            callback=self._required(self._parameters,'model parameter upload')
            try:return callback(changes,self.world,self.stream)
            except BaseException:self._failed=True;raise

    def update_model_parameters_transaction(self,changes,token,model_generation):
        with self._lock:
            self._open();self._check()
            if not callable(self._parameter_transaction):return self.update_model_parameters(changes)
            try:return self._parameter_transaction(changes,self.world,self.stream,token,model_generation)
            except BaseException:self._failed=True;raise
    def update_visual_parameters(self,changes):
        with self._lock:
            self._open();self._check()
            callback=self._required(self._visual,'visual parameter upload')
            try:return callback(changes,self.world,self.stream)
            except BaseException:self._failed=True;raise

    def gpu_physics_evidence(self):
        with self._lock:
            self._open();self._check()
            self._pending_uploads.clear()
            return self._required(self._evidence,'actual GPU evidence')(self.world)

    def diagnostic_evidence(self):
        """Read actual failed-stage facts for forensics, never a valid output claim."""
        with self._lock:
            if self._closed:raise RuntimeError('resident runtime is closed')
            self.wp.synchronize_stream(self.stream)
            payload=self._required(self._evidence,'actual GPU diagnostic evidence')(self.world)
            return {'diagnostic_only':True,'boundary_quarantined':self._failed,'actual':payload}

    def close(self):
        with self._lock:
            if not self._closed:
                try:
                    # Query plans borrow the physics roots; release them first.
                    if self._spatial_queries is not None:self._spatial_queries.close()
                    self._spatial_queries=None
                    self._close()
                except BaseException:
                    self._failed=True
                    raise
                self._pending_uploads.clear()
                self.fields.clear();self.special_readers.clear();self._tokens.clear()
                self.runtime=None
                self._check=self._close=self._state_written=self._reset=self._step1=None
                self._parameters=self._visual=self._evidence=self._parameter_transaction=None
                self._closed=True


def fixed_root_descriptors(bundle):
    """Describe actual existing fixed pipeline owners, without allocating physics.

    Geometry fields are the last forward's outputs, while qpos/qvel/time are live
    Euler roots. Their mixed ages are deliberately retained after integration.
    """
    model=bundle.model_pack.model
    e=bundle.runtime.euler
    pos=bundle.dynamics.stages.position
    roots={
        'time':ResidentField(e.time,()),
        'qpos':ResidentField(e.qpos,(int(model.nq),)),
        'qvel':ResidentField(e.qvel,(int(model.nv),)),
        'ctrl':ResidentField(e.ctrl,(int(model.nu),)),
        'qfrc_applied':ResidentField(bundle.dynamics.qfrc_applied,(int(model.nv),)),
        'xfrc_applied':ResidentField(bundle.dynamics.xfrc_applied,(int(model.nbody),6)),
        'qacc':ResidentField(e.qacc,(int(model.nv),)),
        'qacc_warmstart':ResidentField(e.qacc_warmstart,(int(model.nv),)),
        'qfrc_bias':ResidentField(bundle.dynamics.public['qfrc_bias'],(int(model.nv),)),
        'qfrc_smooth':ResidentField(e.qfrc_smooth,(int(model.nv),)),
        'qfrc_constraint':ResidentField(e.qfrc_constraint,(int(model.nv),)),
        'qM':ResidentField(e.qM,(int(model.nM),)),
        'body_xpos':ResidentField(pos.body.xpos,(int(model.nbody),3)),
        'body_xquat':ResidentField(pos.body.xquat,(int(model.nbody),4)),
        'body_xmat':ResidentField(pos.body.xmat,(int(model.nbody),9)),
        'site_xpos':ResidentField(pos.frames.site_xpos,(int(model.nsite),3)),
        'site_xmat':ResidentField(pos.frames.site_xmat,(int(model.nsite),9)),
        'geom_xpos':ResidentField(pos.frames.geom_xpos,(int(model.ngeom),3)),
        'geom_xmat':ResidentField(pos.frames.geom_xmat,(int(model.ngeom),9)),
        'subtree_com':ResidentField(pos.mass.subtree_com,(int(model.nbody),3)),
        'sensordata':ResidentField(e.sensordata,(int(model.nsensordata),)),
        'ncon':ResidentField(bundle.runtime.forward.transport.ncon,()),
    }
    return roots


def fixed_contact_reader(bundle):
    """Read actual transported contact pairs at a synchronized action boundary."""
    transport=bundle.runtime.forward.transport
    def read(world):
        import warp as wp
        with wp.ScopedStream(bundle.stream):
            count=int(transport.ncon.numpy().reshape(-1)[world])
            arrays={name:transport.inputs[name].numpy().reshape(bundle.runtime.euler.capacity,-1)[world]
                    for name in ('geom1','geom2')}
            if count<0 or any(count>value.size for value in arrays.values()):
                raise RuntimeError('actual contact count exceeds resident capacity')
            return {name:value[:count].copy() for name,value in arrays.items()}
    return read
