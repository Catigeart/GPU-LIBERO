"""Lazy public Jacobian/velocity queries over actual resident GPU roots.

No forward is implicit. Query outputs are private and never publish or alter
physics state. Python performs cold integer topology assembly and host packaging.
"""
import libero.gpu.precision as scalar
from pathlib import Path
from threading import RLock
import numpy as np


def _integer_vector(model, name, count):
    raw = np.asarray(getattr(model, name))
    if raw.shape != (count,) or raw.dtype.kind not in 'iu':
        raise ValueError('native integer topology schema: ' + name)
    if raw.size and (raw.min() < -(2**31) or raw.max() >= 2**31):
        raise ValueError('native topology integer range: ' + name)
    return np.array(raw, dtype=np.int32, copy=True)


def native_ancestor_topology(model):
    """Reproduce mj_jac's fixed-body skip and exact dof-parent walk."""
    nb,nv,ng,ns = (int(getattr(model,n)) for n in ('nbody','nv','ngeom','nsite'))
    if nb < 1 or nv < 1 or ng < 0 or ns < 0:
        raise ValueError('positive native body/velocity dimensions required')
    parent=_integer_vector(model,'body_parentid',nb)
    root=_integer_vector(model,'body_rootid',nb)
    num=_integer_vector(model,'body_dofnum',nb)
    adr=_integer_vector(model,'body_dofadr',nb)
    dp=_integer_vector(model,'dof_parentid',nv)
    geom=_integer_vector(model,'geom_bodyid',ng)
    site=_integer_vector(model,'site_bodyid',ns)
    if parent[0]!=0 or root[0]!=0 or num[0]!=0:
        raise ValueError('native world body topology differs')
    if np.any((dp < -1)|(dp >= nv)) or np.any((geom<0)|(geom>=nb)) or np.any((site<0)|(site>=nb)):
        raise ValueError('native topology ID out of range')
    for dof in range(nv):
        seen=set();cursor=dof
        while cursor>=0:
            if cursor in seen:raise ValueError('cyclic native dof parent chain')
            seen.add(cursor);cursor=int(dp[cursor])
    mask=np.zeros((nb,nv),dtype=np.int32)
    for body in range(nb):
        seen=set();cursor=body;last=body
        while cursor:
            if cursor<0 or cursor>=nb or cursor in seen:
                raise ValueError('invalid native body parent chain')
            seen.add(cursor);last=cursor;cursor=int(parent[cursor])
        if root[body]!=last or num[body]<0 or (num[body]>0 and (adr[body]<0 or adr[body]+num[body]>nv)):
            raise ValueError('invalid native body root/dof block')
        cursor=body
        while cursor and num[cursor]==0:cursor=int(parent[cursor])
        if cursor:
            dof=int(adr[cursor]+num[cursor]-1)
            while dof>=0:
                mask[body,dof]=1;dof=int(dp[dof])
    return {'body_rootid':root,'ancestor':mask,'geom_bodyid':geom,'site_bodyid':site}


def _token(array):
    return (id(array),int(array.ptr),str(array.dtype),tuple(array.shape),tuple(array.strides),str(array.device))


class PublicSpatialQueries:
    """One cold plan per actual owner; build only on the first leaf request."""
    def __init__(self, built, bundle, *, validate_owner, build_root=None):
        import warp as wp
        if not callable(validate_owner):raise TypeError('actual owner/source validator required')
        self.wp,self.built,self.bundle=wp,built,bundle
        self._validate_owner=validate_owner
        self._lock=RLock();self._closed=False;self._plans=None
        from libero.gpu.host.runtime.core.product_paths import build_directory
        self.build_root=Path(build_root) if build_root is not None else build_directory('public-spatial')
        model=built.native_model
        self.nb,self.nv,self.ng,self.ns=(int(getattr(model,n)) for n in ('nbody','nv','ngeom','nsite'))
        self.nworld=int(bundle.runtime.euler.capacity)
        self.stream=bundle.stream;self.device=self.stream.device
        if not self.device.is_cuda:raise ValueError('public spatial queries require actual CUDA')
        pos=bundle.dynamics.stages.position
        self.roots={'cdof':pos.mass.cdof,'com':pos.mass.subtree_com,'body_pos':pos.body.xpos,
            'geom_pos':pos.frames.geom_xpos,'site_pos':pos.frames.site_xpos,'qvel':bundle.runtime.euler.qvel}
        widths={'cdof':6*self.nv,'com':3*self.nb,'body_pos':3*self.nb,
            'geom_pos':3*self.ng,'site_pos':3*self.ns,'qvel':self.nv}
        for name,array in self.roots.items():
            if array.device!=self.device or array.dtype!=scalar.wp_real or not array.is_contiguous or array.size!=self.nworld*widths[name]:
                raise ValueError('actual public spatial root differs: '+name)
        self._tokens={name:_token(array) for name,array in self.roots.items()}
        self.topology=native_ancestor_topology(model)

    def _open(self):
        if self._closed:raise RuntimeError('public spatial owner closed')
        self._validate_owner()
        for name,array in self.roots.items():
            if _token(array)!=self._tokens[name]:raise RuntimeError('public spatial root changed: '+name)

    def _prepare(self):
        if self._plans is not None:return
        from libero.gpu.host.runtime.core.product_paths import mathdx_root, toolchain_scope, cache_root
        from libero.gpu.host.build.product_cache import product_cache_scope
        # Public leaf calls run outside factory's original construction scope.
        # Resolve and bind the actual bundled SDK/cache authority here once.
        sdk=mathdx_root()
        with toolchain_scope(sdk), product_cache_scope(Path(getattr(self.bundle,'build_project_root',self.build_root))/'products'):
            self._prepare_native()

    def _prepare_native(self):
        from libero.gpu.host.paths import GPU_ROOT
        from libero.gpu.host.build.native import NativePlan
        wp=self.wp
        with wp.ScopedStream(self.stream):
            self.static={name:wp.array(value,dtype=wp.int32,device=self.device)
                         for name,value in self.topology.items()}
            self.active=wp.zeros(self.nworld,dtype=wp.int32,device=self.device)
            self.kind=wp.zeros(self.nworld,dtype=wp.int32,device=self.device)
            self.index=wp.zeros(self.nworld,dtype=wp.int32,device=self.device)
            self.status=wp.zeros(self.nworld,dtype=wp.int32,device=self.device)
            self.jp=wp.zeros((self.nworld,3,self.nv),dtype=scalar.wp_real,device=self.device)
            self.jr=wp.zeros((self.nworld,3,self.nv),dtype=scalar.wp_real,device=self.device)
            self.vp=wp.zeros((self.nworld,3),dtype=scalar.wp_real,device=self.device)
            self.vr=wp.zeros((self.nworld,3),dtype=scalar.wp_real,device=self.device)
        args=[self.active,self.kind,self.index,self.status,self.static['body_rootid'],self.static['ancestor'],
            self.static['geom_bodyid'],self.static['site_bodyid'],self.roots['cdof'],self.roots['com'],
            self.roots['body_pos'],self.roots['geom_pos'],self.roots['site_pos'],self.jp,self.jr,self.nb,self.ng,self.ns]
        jac=('public_spatial_cuda::jacobian',args,'lane0')
        here=GPU_ROOT/'cuda/runtime'
        self._jacobian_plan=NativePlan('public_spatial_jacobian',[here/'public_spatial.cuh'],[jac],
            self.nworld,self.build_root/'jacobian',device=self.device,defines={'SPATIAL_NV':self.nv})
        velocity=('public_spatial_cuda::velocity',[self.active,self.status,self.jp,self.jr,self.roots['qvel'],self.vp,self.vr],'block')
        self._velocity_plan=NativePlan('public_spatial_velocity',[here/'public_spatial.cuh'],[jac,velocity],
            self.nworld,self.build_root/'velocity',device=self.device,mathdx=True,
            bridges=[here/'public_spatial_bridge.cu'],defines={'SPATIAL_NV':self.nv,'DX_SM':self.device.arch*10})
        self._plans=(self._jacobian_plan,self._velocity_plan)

    def read(self,world,kind,index,component,*,velocity=False,stream=None):
        with self._lock:
            self._open()
            if stream is not None and stream is not self.stream:raise ValueError('public spatial shared stream required')
            if isinstance(world,bool) or not isinstance(world,int) or not 0<=world<self.nworld:raise ValueError('invalid public spatial world')
            counts={'body':self.nb,'geom':self.ng,'site':self.ns}
            if kind not in counts or isinstance(index,bool) or not isinstance(index,(int,np.integer)) or not 0<=index<counts[kind]:
                raise ValueError('invalid native public spatial query')
            if component not in ('translation','rotation'):raise ValueError('invalid public spatial component')
            self._prepare()
            active=np.zeros(self.nworld,np.int32);active[world]=1
            kinds=np.zeros(self.nworld,np.int32);kinds[world]={'body':0,'geom':1,'site':2}[kind]
            ids=np.zeros(self.nworld,np.int32);ids[world]=index
            with self.wp.ScopedStream(self.stream):
                # Queries own these buffers; the actual physics activity and
                # status roots are neither read as admission nor overwritten.
                self.active.assign(active);self.kind.assign(kinds);self.index.assign(ids)
                (self._velocity_plan if velocity else self._jacobian_plan).launch(self.stream)
            self.wp.synchronize_stream(self.stream)
            self._open()
            code=int(self.status.numpy().reshape(-1)[world])
            if code:raise RuntimeError('GPU public spatial query status: '+str(code))
            output=(self.vp if component=='translation' else self.vr) if velocity else (self.jp if component=='translation' else self.jr)
            value=output.numpy()[world].copy()
            if not np.isfinite(value).all():raise RuntimeError('nonfinite GPU public spatial output')
            return value

    def read_jacobian(self,world,kind,index,component,stream=None):
        return self.read(world,kind,index,component,stream=stream)

    def read_velocity(self,world,kind,index,component,stream=None):
        return self.read(world,kind,index,component,velocity=True,stream=stream)

    def close(self):
        with self._lock:
            if self._closed:return
            self.wp.synchronize_stream(self.stream)
            self._closed=True
            self._plans=None;self._jacobian_plan=self._velocity_plan=None
            self.roots.clear();self._tokens.clear()
            for name in ('static','active','kind','index','status','jp','jr','vp','vr'):
                self.__dict__.pop(name,None)
            self._validate_owner=None;self.built=self.bundle=None
