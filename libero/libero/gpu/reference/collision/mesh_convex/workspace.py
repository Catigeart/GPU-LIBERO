"""Preallocated C07 CUDA boundary; upload/download are outside production."""
from __future__ import annotations
import libero.gpu.precision as scalar
import ctypes as ct
from pathlib import Path
import numpy as np

H2D,D2H=1,2
STATIC={'vert':np.float32,'graph':np.int32,'vertadr':np.int32,
        'vertnum':np.int32,'graphadr':np.int32,'dataid':np.int32,
        'geomtype':np.int32}
DYNAMIC={'pose':(np.float64,(25,)),'pair':(np.int32,(2,)),
         'active':(np.int32,()),'entered':(np.int32,())}
OUTPUT={'raw':(np.float64,(10,)),'count':(np.int32,()),
        'status':(np.int32,()),'diag':(np.int32,(12,)),
        'supports':(np.int32,(256,3)),'portal':(np.float64,(4,9))}


def validate_mpr_parameters(model):
    for name,dtype in (('maxiter',np.int32),('tolerance',np.float64)):
        value=model[name]
        if not isinstance(value,np.ndarray) or value.shape!=() or value.dtype!=np.dtype(dtype):
            raise TypeError(f'{name}: expected original scalar {np.dtype(dtype)}')
    maxiter=model['maxiter'].item();tolerance=model['tolerance'].item()
    if maxiter<0 or not np.isfinite(tolerance) or tolerance<0:
        raise ValueError('MPR parameters')
    return maxiter,tolerance


class MeshMPRWorkspace:
    def __init__(self,model:dict,n:int,library:str|Path):
        if n<1:raise ValueError('N must be positive')
        # Keep the original scalar ABI: .item()/int() would silently truncate
        # negative fractional iterations and accept float32 tolerances.
        maxiter,tolerance=validate_mpr_parameters(model)
        self.N=n;self.stream=0;self.graph=None;self.graph_exec=None;self.owned_stream=None
        self.cuda=ct.CDLL('libcudart.so');self.lib=ct.CDLL(str(Path(library).resolve()))
        self.cuda.cudaMalloc.argtypes=[ct.POINTER(ct.c_void_p),ct.c_size_t]
        self.cuda.cudaMalloc.restype=ct.c_int
        self.cuda.cudaFree.argtypes=[ct.c_void_p];self.cuda.cudaFree.restype=ct.c_int
        self.cuda.cudaMemcpy.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_size_t,ct.c_int]
        self.cuda.cudaMemcpy.restype=ct.c_int
        self.cuda.cudaDeviceSynchronize.argtypes=[]
        self.cuda.cudaDeviceSynchronize.restype=ct.c_int
        self.lib.launch_mesh_mpr.argtypes=[ct.c_void_p]*7+[ct.c_int]*5+[scalar.ct_real]+[ct.c_void_p]*4+[ct.c_int]+[ct.c_void_p]*7
        self.lib.launch_mesh_mpr.restype=ct.c_int
        self.model={}
        for name,dtype in STATIC.items():
            a=model[name]
            if not isinstance(a,np.ndarray) or a.dtype!=dtype:
                raise TypeError(f'{name}: expected original {np.dtype(dtype)}')
            self.model[name]=np.ascontiguousarray(a)
        self.maxiter=maxiter
        self.tolerance=tolerance
        if self.model['vert'].ndim!=2 or self.model['vert'].shape[1]!=3:
            raise ValueError('mesh_vert must be float32 [nvert,3]')
        if self.model['graph'].ndim!=1 or any(self.model[x].ndim!=1 for x in
                ('vertadr','vertnum','graphadr','dataid','geomtype')):
            raise ValueError('static layout')
        if any(len(self.model[x])!=len(self.model['vertadr']) for x in ('vertnum','graphadr')):
            raise ValueError('mesh metadata count')
        if len(self.model['dataid'])!=len(self.model['geomtype']):raise ValueError('geom metadata count')
        vadr=self.model['vertadr'].astype(np.int64)
        vnum=self.model['vertnum'].astype(np.int64)
        gadr=self.model['graphadr'].astype(np.int64)
        if np.any(vadr<0) or np.any(vnum<=0) or np.any(vadr+vnum>len(self.model['vert'])):
            raise ValueError('mesh vertex address/length')
        if np.any((gadr<-1)|(gadr>=len(self.model['graph']))):
            raise ValueError('mesh graph address')
        mesh=self.model['geomtype']==7
        if np.any((self.model['dataid'][mesh]<0)|
                  (self.model['dataid'][mesh]>=len(vadr))):
            raise ValueError('Mesh geom dataid')
        graph=self.model['graph']
        for adr in gadr[gadr>=0]:
            nv=int(graph[adr]);end=int(adr+2+2*nv)
            if nv<=0 or end>=len(graph):raise ValueError('graph hull header')
            edgeadr=graph[adr+2:adr+2+nv]
            global_id=graph[adr+2+nv:adr+2+2*nv]
            if np.any((global_id<0)|(global_id>=vnum[np.flatnonzero(gadr==adr)[0]])):
                raise ValueError('graph global vertex IDs')
            base=end
            for local in edgeadr:
                cursor=base+int(local)
                if cursor<base or cursor>=len(graph):raise ValueError('graph edge address')
                while cursor<len(graph) and graph[cursor]>=0:
                    if graph[cursor]>=nv:raise ValueError('graph neighbor local ID')
                    cursor+=1
                if cursor>=len(graph):raise ValueError('graph neighbor terminator')
        self.ptr={};self.spec={}
        try:
            for name,a in self.model.items():
                self._alloc(name,a.shape,a.dtype);self._copy_in(name,a)
            self.synchronize()
            for name,(dtype,tail) in {**DYNAMIC,**OUTPUT}.items():
                self._alloc(name,(n,)+tail,np.dtype(dtype))
        except Exception:
            self.close();raise

    @staticmethod
    def _check(code,label):
        if code:raise RuntimeError(f'{label}: CUDA error {code}')

    def _alloc(self,name,shape,dtype):
        count=int(np.prod(shape));ptr=ct.c_void_p()
        self._check(self.cuda.cudaMalloc(ct.byref(ptr),count*np.dtype(dtype).itemsize),'cudaMalloc '+name)
        self.ptr[name]=ptr;self.spec[name]=(shape,np.dtype(dtype))

    def _copy_in(self,name,a):
        self._check(self.cuda.cudaMemcpy(self.ptr[name],ct.c_void_p(a.ctypes.data),a.nbytes,H2D),
                    'cudaMemcpy H2D '+name)

    def _input(self,name,a):
        shape,dtype=self.spec[name]
        if not isinstance(a,np.ndarray) or a.shape!=shape or a.dtype!=dtype:
            raise TypeError(f'{name}: expected original {shape} {dtype}')
        return np.ascontiguousarray(a)

    def upload(self,pose,pair,active,entered):
        arrays=locals()
        for name in DYNAMIC:
            self._copy_in(name,self._input(name,arrays[name]))
        # Graph launches use a nonblocking stream; complete these default-
        # stream copies before the caller can launch on that stream.
        self.synchronize()

    def prefill(self,**outputs):
        if set(outputs)!=set(OUTPUT):raise ValueError('all output buffers required')
        for name in OUTPUT:self._copy_in(name,self._input(name,outputs[name]))
        self.synchronize()

    def launch(self):
        p=self.ptr
        args=[p[x] for x in STATIC]
        args += [len(self.model['vert']),len(self.model['graph']),
                 len(self.model['vertadr']),len(self.model['dataid']),self.maxiter,
                 self.tolerance]
        args += [p[x] for x in DYNAMIC]+[self.N]
        args += [p[x] for x in OUTPUT]+[ct.c_void_p(self.stream)]
        self._check(self.lib.launch_mesh_mpr(*args),'launch_mesh_mpr')

    def synchronize(self):self._check(self.cuda.cudaDeviceSynchronize(),'cudaDeviceSynchronize')

    def download(self):
        self.synchronize();out={}
        for name in OUTPUT:
            shape,dtype=self.spec[name];a=np.empty(shape,dtype)
            self._check(self.cuda.cudaMemcpy(ct.c_void_p(a.ctypes.data),self.ptr[name],a.nbytes,D2H),
                        'cudaMemcpy D2H '+name)
            out[name]=a
        return out

    def capture(self):
        if self.graph_exec is not None:raise RuntimeError('graph already captured')
        self.synchronize()
        c=self.cuda
        c.cudaStreamCreateWithFlags.argtypes=[ct.POINTER(ct.c_void_p),ct.c_uint]
        c.cudaStreamBeginCapture.argtypes=[ct.c_void_p,ct.c_int]
        c.cudaStreamEndCapture.argtypes=[ct.c_void_p,ct.POINTER(ct.c_void_p)]
        # CUDA 13 runtime ABI: graph exec*, graph, unsigned long long flags.
        c.cudaGraphInstantiate.argtypes=[ct.POINTER(ct.c_void_p),ct.c_void_p,ct.c_ulonglong]
        c.cudaGraphLaunch.argtypes=[ct.c_void_p,ct.c_void_p]
        c.cudaGraphExecDestroy.argtypes=[ct.c_void_p];c.cudaGraphDestroy.argtypes=[ct.c_void_p]
        c.cudaStreamDestroy.argtypes=[ct.c_void_p]
        stream=ct.c_void_p();self._check(c.cudaStreamCreateWithFlags(ct.byref(stream),1),'stream create')
        self.owned_stream=stream;self.stream=stream.value
        self._check(c.cudaStreamBeginCapture(stream,0),'graph begin');self.launch()
        graph=ct.c_void_p();self._check(c.cudaStreamEndCapture(stream,ct.byref(graph)),'graph end')
        self.graph=graph;exec_=ct.c_void_p()
        self._check(c.cudaGraphInstantiate(ct.byref(exec_),graph,0),'graph instantiate')
        self.graph_exec=exec_

    def graph_launch(self):
        if self.graph_exec is None:raise RuntimeError('graph not captured')
        self._check(self.cuda.cudaGraphLaunch(self.graph_exec,ct.c_void_p(self.stream)),'graph launch')

    def close(self):
        if self.graph_exec:self.cuda.cudaGraphExecDestroy(self.graph_exec);self.graph_exec=None
        if self.graph:self.cuda.cudaGraphDestroy(self.graph);self.graph=None
        if self.owned_stream:self.cuda.cudaStreamDestroy(self.owned_stream);self.owned_stream=None
        for ptr in self.ptr.values():self.cuda.cudaFree(ptr)
        self.ptr.clear()

    def __del__(self):
        if hasattr(self,'ptr'):self.close()
