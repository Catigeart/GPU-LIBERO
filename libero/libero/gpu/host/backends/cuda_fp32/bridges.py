"""Typed load-only controller/observable transport entry points."""
import ctypes as ct
from pathlib import Path


class Gather(ct.Structure):
    _fields_=[(k,ct.c_int) for k in ('worlds','site','nv','nq','nmass','nsites')]+[(k,ct.c_void_p) for k in (
        'qi','vi','mi','qpos','qvel','bias','mass','jp','jr','position','rotation',
        'out_mass','out_jac','out_q','out_v','out_bias','out_pos','out_rot','out_vel')]


class Observe(ct.Structure):
    _fields_=[(k,ct.c_int) for k in ('worlds','ns','width','stride')]+[(k,ct.c_void_p) for k in (
        'op','offset','size','index_offset','indices','dep0','dep1','dep2','presence','enabled','periods','source',
        'cache','observed','sidequat','elapsed','sampled','present','mask','events')]


class Scene(ct.Structure):
    _fields_=[(k,ct.c_int) for k in ('worlds','cameras','stride')]+[(k,ct.c_void_p) for k in (
        'source','scene','periods','enabled','elapsed','sampled','generation','parents','parent_generation','captured','mask')]


def _load_library(path):
    """Bind and validate once, when an owned stream first needs this library."""
    library=ct.CDLL(str(path))
    p=ct.c_void_p;i=ct.c_int
    signatures=dict(points=[i,i,i,p,p,p],gather=[ct.POINTER(Gather),p],scatter=[i,i,p,p,p,i,p],
        masked=[i,i,p,p,p,p],bootstrap=[i]+[p]*10,count=[i,i]+[p]*7,
        pack=[i,i,i,i,i,p,p,p],observe=[ct.POINTER(Observe),ct.c_float,i,p],
        reset_observation=[i,i,i]+[p]*7,capture_scene=[ct.POINTER(Scene),ct.c_float,i,p],
        scene_transform=[i,i,i,i]+[p]*9,collision_reset=[p,p,p])
    for name,args in signatures.items():
        f=getattr(library,'flash_bridge_'+name);f.argtypes=args;f.restype=i
    for name,structure in (('gather',Gather),('observe',Observe),('scene',Scene)):
        f=getattr(library,'flash_bridge_'+name+'_bytes');f.restype=ct.c_size_t
        if f()!=ct.sizeof(structure):raise ValueError('Bridge ABI mismatch: '+name)
    return library


class Bridges:
    """Stream-local calls into a shared, stateless C ABI.

    Components using the same stream reuse its validated library bindings.
    Products stay fixed for the lifetime of the process: ctypes/dlopen does not
    provide same-path hot reload after an offline rebuild. There is no runtime
    file polling or process-wide cache keeping closed environments alive.
    """
    def __init__(self,path,stream):
        self.path=Path(path).resolve();self.stream=stream
        libraries=getattr(stream,'_bridge_libraries',None)
        if libraries is None:
            libraries={};stream._bridge_libraries=libraries
        if self.path not in libraries:
            libraries[self.path]=_load_library(self.path)
        self.library=libraries[self.path]

    def call(self,name,*args):
        with self.stream.device.scope():
            code=getattr(self.library,'flash_bridge_'+name)(*args,self.stream.cuda_stream)
        if code:raise RuntimeError('CUDA bridge '+name+': error '+str(code))
