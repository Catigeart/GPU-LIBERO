"""Typed load-only controller/observable transport entry points."""
import ctypes as ct
import hashlib
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


class Bridges:
    def __init__(self,path,stream):
        from libero.gpu.host.paths import GPU_ROOT
        self.path=Path(path);self.stream=stream;self.library=ct.CDLL(str(self.path))
        self.library.flash_bridge_source_id.restype=ct.c_char_p
        expected=hashlib.sha256(':'.join(hashlib.sha256((GPU_ROOT/k).read_bytes()).hexdigest() for k in
            ('cuda/runtime/bridges.cu','cuda/runtime/bridge_types.cuh','cmake/CudaBridges.cmake')).encode()).hexdigest()
        if self.library.flash_bridge_source_id().decode()!=expected:raise RuntimeError('Stale CUDA bridge product; rebuild offline')
        p=ct.c_void_p;i=ct.c_int
        signatures=dict(points=[i,i,i,p,p,p],gather=[ct.POINTER(Gather),p],scatter=[i,i,p,p,p,i,p],
            masked=[i,i,p,p,p,p],bootstrap=[i]+[p]*10,count=[i,i]+[p]*7,
            pack=[i,i,i,i,i,p,p,p],observe=[ct.POINTER(Observe),ct.c_float,i,p],
            reset_observation=[i,i,i]+[p]*7,capture_scene=[ct.POINTER(Scene),ct.c_float,i,p],
            scene_transform=[i,i,i,i]+[p]*9)
        for name,args in signatures.items():
            f=getattr(self.library,'flash_bridge_'+name);f.argtypes=args;f.restype=i
        for name,structure in (('gather',Gather),('observe',Observe),('scene',Scene)):
            f=getattr(self.library,'flash_bridge_'+name+'_bytes');f.restype=ct.c_size_t
            if f()!=ct.sizeof(structure):raise ValueError('Bridge ABI mismatch: '+name)
    def call(self,name,*args):
        code=getattr(self.library,'flash_bridge_'+name)(*args,self.stream.cuda_stream)
        if code:raise RuntimeError('CUDA bridge '+name+': error '+str(code))
