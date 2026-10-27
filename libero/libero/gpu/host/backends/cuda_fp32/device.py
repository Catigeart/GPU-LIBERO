"""Small device-buffer and CUDA resource owners; no numerical compiler/JIT.

Logical shapes omit fixed vector/matrix components, matching the maintained
CUDA ABIs. Host storage shapes include them. Views retain the allocation owner.
All allocations and typed bindings happen before capture.
"""
from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import ctypes as ct
import math
import numpy as np


@dataclass(frozen=True)
class DType:
    name: str
    scalar: str
    components: tuple = ()

    @property
    def numpy_dtype(self): return np.dtype(self.scalar)
    @property
    def itemsize(self): return self.numpy_dtype.itemsize * math.prod(self.components)
    @property
    def _shape_(self): return self.components


float32=DType('float32','float32');float64=DType('float64','float64')
int32=DType('int32','int32');int64=DType('int64','int64')
uint8=DType('uint8','uint8');boolean=DType('bool','bool')
vec2=DType('vec2','float32',(2,));vec3=DType('vec3','float32',(3,))
vec2i=DType('vec2i','int32',(2,));vec3i=DType('vec3i','int32',(3,))
quat=DType('quat','float32',(4,));mat33=DType('mat33','float32',(3,3))
spatial_vector=DType('spatial_vector','float32',(6,))
_active_stream=ContextVar('flash_cuda_stream',default=None)
_stream_scopes=ContextVar('flash_cuda_stream_scopes',default=())


def dtype(value):
    if isinstance(value,DType):return value
    if value is float:return float32
    if value is int:return int32
    if value is bool:return boolean
    lookup={'float32':float32,'float64':float64,'int32':int32,'int64':int64,'uint8':uint8,'bool':boolean}
    return lookup[np.dtype(value).name]


class Runtime:
    """Load an independently built C ABI, with typed entry points."""
    def __init__(self,library):
        self.path=Path(library).resolve();self.lib=ct.CDLL(str(self.path))
        ptr=ct.c_void_p;size=ct.c_size_t;integer=ct.c_int
        signatures=dict(device=[integer,ct.POINTER(integer),ct.POINTER(size),ct.POINTER(size)],
            get_device=[ct.POINTER(integer)],set_device=[integer],
            attribute=[integer,integer,ct.POINTER(integer)],
            allocate=[ct.POINTER(ptr),size],free=[ptr],stream_create=[ct.POINTER(ptr)],
            stream_destroy=[ptr],synchronize=[ptr],copy=[ptr,ptr,size,integer,ptr],
            zero=[ptr,size,ptr],capture_begin=[ptr],capture_end=[ptr,ct.POINTER(ptr)],
            graph_instantiate=[ptr,ct.POINTER(ptr)],graph_launch=[ptr,ptr],
            graph_destroy=[ptr,ptr],event_create=[ct.POINTER(ptr)],event_record=[ptr,ptr],
            event_elapsed=[ptr,ptr,ct.POINTER(ct.c_float)],event_destroy=[ptr],
            store_integer=[ptr,integer,ptr],decrement_integer=[ptr,ptr],
            while_begin=[ptr,ptr,ct.POINTER(ptr)],while_end=[ptr,ptr])
        for name,args in signatures.items():
            try:function=getattr(self.lib,'flash_runtime_'+name)
            except AttributeError as error:
                from libero.gpu.host.build.compilation import MissingCompiledProduct
                raise MissingCompiledProduct('CUDA resource ABI is missing '+name+'; rebuild flash_cuda_runtime offline in '+str(self.path.parent)) from error
            function.argtypes=args;function.restype=integer
        self.lib.flash_runtime_error.argtypes=[integer];self.lib.flash_runtime_error.restype=ct.c_char_p

    def call(self,name,*args):
        error=getattr(self.lib,'flash_runtime_'+name)(*args)
        if error:raise RuntimeError('CUDA '+name+': '+self.lib.flash_runtime_error(error).decode())

    def device(self,ordinal=0):return Device(self,ordinal)


class Device:
    _cuda_native=True
    def __init__(self,runtime,ordinal):
        self.runtime=runtime;self.ordinal=ordinal;arch=ct.c_int();free=ct.c_size_t();total=ct.c_size_t()
        runtime.call('device',ordinal,ct.byref(arch),ct.byref(free),ct.byref(total))
        self.arch=arch.value;self.free_bytes=free.value;self.total_bytes=total.value;self.is_cuda=True
        count=ct.c_int();runtime.call('attribute',ordinal,16,ct.byref(count));self.sm_count=count.value
    def __str__(self):return 'cuda:'+str(self.ordinal)

    @contextmanager
    def scope(self):
        previous=ct.c_int()
        self.runtime.call('get_device',ct.byref(previous))
        if previous.value!=self.ordinal:self.runtime.call('set_device',self.ordinal)
        try:yield
        finally:
            if previous.value!=self.ordinal:self.runtime.call('set_device',previous.value)

    def call(self,name,*args):
        with self.scope():self.runtime.call(name,*args)


class Stream:
    _cuda_native=True
    def __init__(self,device):
        self.device=device;self.runtime=device.runtime;self.handle=ct.c_void_p();self.capturing=False
        self.device.call('stream_create',ct.byref(self.handle));self._closed=False;self._pending_host=[]
        self.device.stream=self
    @property
    def cuda_stream(self):return self.handle.value
    @property
    def is_capturing(self):return self.capturing
    def synchronize(self):
        self.device.call('synchronize',self.handle);self._pending_host.clear()
    def close(self):
        if not self._closed:
            self.synchronize();self.device.call('stream_destroy',self.handle);self._closed=True
    def __enter__(self):
        scope=self.device.scope();scope.__enter__()
        token=_active_stream.set(self)
        _stream_scopes.set(_stream_scopes.get()+((scope,token),));return self
    def __exit__(self,*args):
        scopes=_stream_scopes.get();scope,token=scopes[-1]
        _stream_scopes.set(scopes[:-1]);_active_stream.reset(token);scope.__exit__(*args)


def current_stream(device=None):
    stream=_active_stream.get()
    # A lazy adapter may allocate outside a lexical scope, but it must still
    # supply the actual owned Device, never a process-wide implicit CUDA stream.
    if stream is None and isinstance(device,Device):stream=getattr(device,'stream',None)
    if stream is None:raise RuntimeError('Explicit CUDA stream scope required')
    if stream._closed:raise RuntimeError('CUDA stream owner is closed')
    if device is not None and device is not stream.device:raise ValueError('CUDA device owner differs')
    return stream


class Allocation:
    def __init__(self,stream,nbytes):
        if stream.capturing:raise RuntimeError('Allocate workspace before Graph capture')
        self.runtime=stream.runtime;self.device=stream.device;self.pointer=ct.c_void_p();self.nbytes=nbytes;self.closed=False
        self.device.call('allocate',ct.byref(self.pointer),nbytes)
    def close(self):
        if not self.closed:self.device.call('free',self.pointer);self.closed=True
    def __del__(self):
        if not getattr(self,'closed',True):
            try:self.close()
            except Exception:pass


class Array:
    """Contiguous device field or alias; only explicit transfer operations."""
    _cuda_native=True
    def __init__(self,data=None,*,shape=None,dtype=float32,device=None,ptr=None,owner=None):
        if ptr is not None and owner is None:raise ValueError('A borrowed device view must retain its allocation owner')
        self.dtype=globals()['dtype'](dtype)
        stream=owner.stream if ptr is not None else current_stream(device)
        self.device=stream.device;self.stream=stream
        host=None if data is None else np.ascontiguousarray(data,dtype=self.dtype.numpy_dtype)
        components=math.prod(self.dtype.components)
        if shape is None:
            if host is None:raise ValueError('Array needs shape or host data')
            shape=host.shape[:-len(self.dtype.components)] if self.dtype.components else host.shape
            if self.dtype.components and not shape:shape=(host.size//components,)
        self.shape=(shape,) if isinstance(shape,int) else tuple(int(x) for x in shape)
        if any(x<0 for x in self.shape):raise ValueError('Negative device shape')
        self.ndim=len(self.shape);self.size=math.prod(self.shape);self.capacity=self.size*self.dtype.itemsize
        if self.size*math.prod(self.dtype.components)>2**31-1:
            raise ValueError('Device array scalar span exceeds CUDA int32 indexing')
        self.strides=tuple(math.prod(self.shape[i+1:])*self.dtype.itemsize for i in range(self.ndim))
        self.is_contiguous=True
        if ptr is not None:
            if owner is None:raise ValueError('A borrowed device view must retain its allocation owner')
            if getattr(owner,'device',self.device) is not self.device:
                raise ValueError('Borrowed CUDA view device differs from its allocation owner')
            self.owner=owner;self.ptr=int(ptr)
        else:
            self.owner=Allocation(stream,self.capacity);self.ptr=self.owner.pointer.value or 0
        if host is not None:self.assign(host)

    @property
    def storage_shape(self):return self.shape+self.dtype.components
    def numpy(self):
        if self.stream.capturing:raise RuntimeError('Host download inside Graph capture')
        result=np.empty(self.storage_shape,dtype=self.dtype.numpy_dtype)
        self.stream.device.call('copy',result.ctypes.data,self.ptr,self.capacity,2,self.stream.cuda_stream)
        self.stream.synchronize();return result
    def assign(self,source):
        if self.stream.capturing:raise RuntimeError('Host upload inside Graph capture')
        host=np.array(source,dtype=self.dtype.numpy_dtype,order='C',copy=True)
        if host.nbytes!=self.capacity:raise ValueError('Device assignment byte span differs')
        self.stream.device.call('copy',self.ptr,host.ctypes.data,self.capacity,1,self.stream.cuda_stream)
        # Retain the private upload until a real boundary synchronizes. This
        # keeps action submission asynchronous without borrowing caller memory.
        self.stream._pending_host.append(host)
    def zero_(self):self.stream.device.call('zero',self.ptr,self.capacity,self.stream.cuda_stream)
    def fill_(self,value):
        if value==0:self.zero_()
        else:self.assign(np.full(self.storage_shape,value,dtype=self.dtype.numpy_dtype))
    def flatten(self):return self.reshape((self.size,))
    def reshape(self,shape):
        if math.prod(shape)!=self.size:raise ValueError('Device reshape span differs')
        return Array(shape=shape,dtype=self.dtype,device=self.device,ptr=self.ptr,owner=self)


def empty(shape,dtype=float32,device=None):return Array(shape=shape,dtype=dtype,device=device)
def zeros(shape,dtype=float32,device=None):
    result=empty(shape,dtype,device);result.zero_();return result
def zeros_like(source):return zeros(source.shape,source.dtype,source.device)
def empty_like(source):return empty(source.shape,source.dtype,source.device)
def full(shape,value,dtype=int32,device=None):
    result=empty(shape,dtype,device);result.fill_(value);return result
def clone(source):
    result=empty_like(source);copy(result,source);return result
def copy(target,source,*,stream=None,count=None,dest_offset=0,src_offset=0):
    stream=stream or target.stream
    if target.dtype!=source.dtype or target.device is not source.device:raise ValueError('Device copy type/owner differs')
    if stream.device is not target.device:raise ValueError('Device copy stream owner differs')
    n=source.size if count is None else int(count);item=source.dtype.itemsize
    if min(n,dest_offset,src_offset)<0 or dest_offset+n>target.size or src_offset+n>source.size:raise ValueError('Device copy exceeds field span')
    stream.device.call('copy',target.ptr+dest_offset*item,source.ptr+src_offset*item,n*item,3,stream.cuda_stream)


def store_integer(field,value):
    if field.dtype!=int32 or field.size!=1:raise ValueError('Expected one int32 conditional counter')
    field.stream.device.call('store_integer',field.ptr,int(value),field.stream.cuda_stream)


class While:
    def __init__(self,counter):self.counter=counter;self.scope=ct.c_void_p()
    def __enter__(self):
        s=self.counter.stream
        if not s.capturing:raise RuntimeError('Conditional while requires a captured Graph')
        s.device.call('while_begin',s.cuda_stream,self.counter.ptr,ct.byref(self.scope));return self
    def __exit__(self,*args):
        s=self.counter.stream;s.device.call('while_end',s.cuda_stream,self.scope)


class Graph:
    def __init__(self,stream,owners=()):
        self.stream=stream;self.owners=list(owners);self.handle=ct.c_void_p();self.executable=ct.c_void_p();self.closed=False
    def __enter__(self):
        self.stream.__enter__()
        try:
            self.stream.device.call('capture_begin',self.stream.cuda_stream)
            self.stream.capturing=True
            return self
        except BaseException:
            self.stream.__exit__(None,None,None)
            raise
    def __exit__(self,kind,value,traceback):
        try:
            self.stream.capturing=False
            self.stream.device.call('capture_end',self.stream.cuda_stream,ct.byref(self.handle))
            if kind is None:self.stream.device.call('graph_instantiate',self.handle,ct.byref(self.executable))
            else:self.close()
        finally:self.stream.__exit__(kind,value,traceback)
    def launch(self):
        if self.closed or not self.executable.value:raise RuntimeError('Graph is not executable')
        self.stream.device.call('graph_launch',self.executable,self.stream.cuda_stream)
    def close(self):
        if not self.closed:
            self.stream.synchronize();self.stream.device.call('graph_destroy',self.handle,self.executable);self.closed=True;self.owners.clear()


# Buffer vocabulary used by existing pointer-only numerical bindings.
array=Array
ScopedStream=lambda stream:stream
synchronize_stream=lambda stream:stream.synchronize()
get_stream=lambda device:device.stream


def __getattr__(name):
    if name=='bool':return boolean
    raise AttributeError(name)
