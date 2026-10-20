"""Diagnostic capacity-specialized CUDA core, lowered from retained stage bodies.

One block owns a world. Flat transpose/assembly grids become block loops;
world stages and S07 retain their arithmetic and run on lane zero. Diagnostic
tail slots remain source-equivalent after stop. Original entry is retained.
"""
import libero.gpu.precision as scalar
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
from libero.gpu.host.runtime.core.product_paths import nvcc_path
import ctypes as ct
import hashlib
import inspect
import json
import re
import subprocess
from pathlib import Path
from types import MethodType

import numpy as np
import warp as wp
from warp._src.context import ModuleBuilder


class Slot(int):
    pass


def _balanced(source, start):
    depth = 1
    pos = start+1
    while depth:
        if source[pos] == '{': depth += 1
        if source[pos] == '}': depth -= 1
        pos += 1
    return pos


def _lower(source, required):
    """Keep device helpers; replace global forward wrappers by explicit-index calls."""
    source = source[source.index('// CUDA Thread Block Cluster'):]
    source = re.sub(r'^#.*$', '', source, flags=re.M)
    functions = {}
    pieces = []
    previous = 0
    pattern = re.compile(r'extern "C" __global__ void (\w+)\((.*?)\)\s*\{', re.S)
    for match in pattern.finditer(source):
        end = _balanced(source, match.end()-1)
        pieces.append(source[previous:match.start()]); previous=end
        name, signature = match.group(1,2)
        if not name.endswith('_cuda_kernel_forward'): continue
        key = next((key for key in required if name.startswith(key+'_')),None)
        if key is None: continue
        body = source[match.end():end-1]
        loop = re.search(r'for \(size_t _idx = .*?\)\s*\{',body,re.S)
        if loop is None:
            inner,count=re.subn(r'const size_t _idx = .*?;','',body,count=1,flags=re.S)
            if count!=1:raise ValueError(('unexpected Warp wrapper',name))
            inner=inner.replace('wp::tile_shared_storage_t tile_mem;','')
        else:
            inner_end=_balanced(body,loop.end()-1)
            inner=body[loop.end():inner_end-1]
        inner=inner.replace('wp::tile_shared_storage_t::init();','')
        declarations=[x.strip() for x in signature.split(',')]
        arg_names=[x.rsplit(' ',1)[1] for x in declarations]
        functions[key]=(name,declarations,arg_names)
        pieces.append('__device__ __noinline__ void '+name+'(size_t _idx, '+signature+') {\n'+inner+'\n}\n')
    pieces.append(source[previous:])
    if set(functions)!=set(required): raise ValueError(('missing CUDA stages',set(required)-set(functions)))
    lowered=''.join(pieces)
    lowered=re.sub(r'wp::int\b','wp::int32',lowered)
    lowered=re.sub(r'wp::float\b','wp::float32',lowered)
    return lowered,functions


def _record(solver):
    records=[]; mode='prefix'
    original_launch=wp.launch
    original_s07=solver.ws.launch_s07
    original_loop=solver._loop_body
    source_solver=getattr(type(solver),'fusion_iteration_original',None)
    recorded_loop=original_loop if source_solver is None else MethodType(source_solver._loop_body,solver)
    had_loop='_loop_body' in vars(solver)
    had_s07='launch_s07' in vars(solver.ws)
    def launch(kernel,dim=None,inputs=None,outputs=None,**kwargs):
        values=list(inputs or [])+list(outputs or [])
        if kernel.key=='_tape_ls_buffers':
            from .parallel_tape import _parallel_tape_ls_buffers
            nv=solver.ws.qacc.shape[1]
            kernel=_parallel_tape_ls_buffers
            dim=solver.ws.nworld*(nv+4*solver.ws.rows+3)
            values.append(nv)
        records.append((mode,kernel,dim,values))
    def s07(*args,**kwargs): records.append((mode,'S07',None,[]))
    def loop(k):
        nonlocal mode
        if k==0:
            mode='loop';recorded_loop(Slot(0));mode='tail'
    wp.launch=launch;solver.ws.launch_s07=s07;solver._loop_body=loop
    try: solver.launch_device()
    finally:
        wp.launch=original_launch
        if had_s07:solver.ws.launch_s07=original_s07
        else:del solver.ws.launch_s07
        if had_loop:solver._loop_body=original_loop
        else:del solver._loop_body
    if not any(r[0]=='loop' for r in records):raise ValueError('positive maxiter required')
    return records


class PersistentCudaCore:
    def __init__(self,solver,build_root,*,block_size=64):
        ws=solver.ws
        device=wp.get_device(ws.device)
        if block_size not in (32,64,128):raise ValueError('block size')
        if ws.maxiter<1:raise ValueError('persistent candidate requires maxiter >= 1')
        self.solver=solver;self.ws=ws;self.block_size=block_size
        self.original=solver.launch_device
        self.event_owner=inspect.getclosurevars(solver.launch_device.__func__).nonlocals.get('owner')
        records=_record(solver)
        self.root=Path(build_root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        groups={}
        for _,kernel,_,_ in records:
            if kernel!='S07':groups.setdefault(kernel.module,set()).add(kernel.key)
        snippets=[];functions={};sources={}
        for index,(module,keys) in enumerate(groups.items()):
            options=module.options|dict(block_dim=block_size,output_arch=device.arch,enable_backward=False)
            raw=ModuleBuilder(module,options).codegen('cuda')
            (self.root/('stage_module_%d.cu'%index)).write_text(raw)
            lowered,funcs=_lower(raw,keys)
            namespace='stage_module_%d'%index
            snippets.append('namespace '+namespace+' {\n'+lowered+'\n}\n')
            for key,info in funcs.items():functions[(module,key)]=(namespace,info)
            sources[module.name]=hashlib.sha256(raw.encode()).hexdigest()
        ls=(GPU_ROOT / 'reference/newton/newton_linesearch/kernel.cu')
        native=ls.read_text().split('extern "C" int s07_launch')[0]
        native=native.replace('__global__ void search_kernel(','__device__ __noinline__ void search_kernel(int w, ')
        native=native.replace('  int w = blockIdx.x*blockDim.x+threadIdx.x;','')
        fields=[];field_values=[];field_index={};statements={k:[] for k in ('prefix','loop','tail')}
        owners=[]
        def bind(value,cuda_type):
            if isinstance(value,Slot):return 'iteration_slot'
            token=(id(value),cuda_type) if isinstance(value,wp.array) else (value,cuda_type)
            if token not in field_index:
                name='a%d'%len(fields);field_index[token]=name
                box=value.__ctype__() if isinstance(value,wp.array) else (
                    scalar.ct_real(value) if 'float64' in cuda_type else ct.c_int32(value))
                fields.append((name,cuda_type,type(box)));field_values.append(box)
                if isinstance(value,wp.array):owners.append(value)
            return 'p->'+field_index[token]
        for mode,kernel,dim,args in records:
            if kernel=='S07':
                statements[mode].append('if(threadIdx.x==0) search_kernel(w,p->ls,%d,%d,%d,%d,%d);\n__syncthreads();'%(ws.nworld,ws.qacc.shape[1],ws.qM.shape[1],ws.rows,ws.eval_trace.shape[1]))
                continue
            namespace,(name,decls,arg_names)=functions[(kernel.module,kernel.key)]
            if len(args)!=len(decls)-1:raise ValueError('kernel argument layout')
            bound=[bind(value,decl.rsplit(' ',1)[0]) for value,decl in zip(args,decls[1:])]
            shape=(dim,) if isinstance(dim,int) else tuple(dim)
            total=int(np.prod(shape));perworld=total//ws.nworld
            if total%ws.nworld:raise ValueError(('grid cannot partition by world',kernel.key,dim))
            bounds='bounds_%d'%len(statements[mode])
            code='wp::launch_bounds_t<%d> %s = {}; %s.size=%d; '%(len(shape),bounds,bounds,total)
            code+=' '.join('%s.shape[%d]=%d;'%(bounds,i,n) for i,n in enumerate(shape))
            code+='\nfor(int local=threadIdx.x;local<%d;local+=blockDim.x) %s::%s(w*%d+local,%s,%s);\n__syncthreads();'%(perworld,namespace,name,perworld,bounds,','.join(bound))
            statements[mode].append('{'+code+'}')
        Param=type('Parameters',(ct.Structure,),{'_fields_':[(n,t) for n,_,t in fields]+[('ls',type(ws._s07_args))]})
        self.params=Param(*field_values,ws._s07_args)
        declarations='\n'.join(t+' '+n+';' for n,t,_ in fields)+'\nS07Args ls;'
        native=re.sub(r'^#include.*$','',native,flags=re.M)
        headers='''#define NDEBUG
#define WP_NO_BFLOAT16
#define WP_ENABLE_CUDA 1
#define WP_TILE_BLOCK_DIM 64
#include "builtin.h"
#include "deterministic.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#define builtin_tid1d() wp::tid(_idx, dim)
#define builtin_tid2d(x,y) wp::tid(x,y,_idx,dim)
#define builtin_tid3d(x,y,z) wp::tid(x,y,z,_idx,dim)
#define builtin_tid4d(x,y,z,w) wp::tid(x,y,z,w,_idx,dim)
#define builtin_block_dim() wp::block_dim()
'''
        headers=headers.replace('WP_TILE_BLOCK_DIM 64','WP_TILE_BLOCK_DIM %d'%block_size)
        body='\n'.join(statements['prefix'])+'\nfor(int iteration_slot=0;iteration_slot<%d;++iteration_slot){\n'%ws.maxiter+'\n'.join(statements['loop'])+'\n}\n'+'\n'.join(statements['tail'])
        source=headers+native+'\n'+''.join(snippets)+'\nstruct Parameters {\n'+declarations+'\n};\nstatic_assert(sizeof(Parameters)==%d,"host parameter layout");\n'%ct.sizeof(Param)
        source+='__global__ void persistent_newton(const Parameters* p){int w=blockIdx.x;\n'+body+'\n}\n'
        source+='extern "C" int launch_persistent(const void* p, int worlds, void* stream){persistent_newton<<<worlds,%d,0,(cudaStream_t)stream>>>((const Parameters*)p);return (int)cudaGetLastError();}\n'%block_size
        digest=hashlib.sha256((source+'|fma=true|prec-div=true|prec-sqrt=true').encode()).hexdigest()
        source_path=self.root/(digest+'.cu');target=self.root/(digest+'.so')
        source_path.write_text(source)
        include=Path(wp.__file__).parent/'native'
        command=[nvcc_path(),'-shared','-Xcompiler','-fPIC','-O3','--fmad=true','--prec-div=true','--prec-sqrt=true','-arch=sm_%d'%device.arch,'-I'+str(include),str(source_path),'-o',str(target)]
        if not target.exists():
            result=subprocess.run(command,capture_output=True,text=True)
            (self.root/'compile.log').write_text(result.stdout+result.stderr)
            if result.returncode:raise RuntimeError('persistent CUDA compilation failed: '+str(self.root/'compile.log'))
        self.library=ct.CDLL(str(target));self.library.launch_persistent.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p];self.library.launch_persistent.restype=ct.c_int
        raw=np.frombuffer(bytes(self.params),dtype=np.uint8).copy()
        with wp.ScopedStream(ws.stream):
            self.device_params=wp.array(raw,dtype=wp.uint8,device=ws.device)
        self.owners=tuple(owners);self.tokens=tuple((a.ptr,a.shape,a.dtype) for a in self.owners)
        self.bindings=tuple((owner,key,value) for owner in (solver,ws)
                            for key,value in vars(owner).items() if isinstance(value,wp.array))
        self.capacity=(ws.nworld,ws.rows,ws.contacts,ws.maxiter,ws.warmstart_enabled)
        self.manifest=dict(floating_policy={'fma':True,'fast_math':False,'prec_div':True,'prec_sqrt':True},source=str(source_path),sha256=digest,command=command,stage_sources=sources,
            nv=ws.qacc.shape[1],nM=ws.qM.shape[1],rows=ws.rows,contacts=ws.contacts,nworld=ws.nworld,
            maxiter=ws.maxiter,block_size=block_size,stage_counts={k:len(v) for k,v in statements.items()},
            diagnostic_tail_preserved=True,scope='CUDA solver core; adapter admission/public commit remain separate')
        (self.root/'MANIFEST.json').write_text(json.dumps(self.manifest,indent=2)+'\n')
    def _validate_launch(self):
        ws=self.ws
        if self.event_owner is not None:
            from libero.gpu.host.runtime.step.newton_final_event import _token
            owner=self.event_owner
            roots=(self.solver.candidate,self.solver.candidate_attempt,self.solver.candidate_completed,
                   self.solver.candidate_origin,self.solver.source_stage,self.solver.rejected_final_entry,owner.n_attempt)
            if tuple(_token(a) for a in roots)!=self.solver._event_tokens:
                raise ValueError('final event owner changed')
        if (ws.nworld,ws.rows,ws.contacts,ws.maxiter,ws.warmstart_enabled)!=self.capacity:
            raise ValueError('persistent capacity/configuration changed; cold rebuild required')
        if any(getattr(owner,key) is not value for owner,key,value in self.bindings):
            raise ValueError('persistent array binding changed')
        if tuple((a.ptr,a.shape,a.dtype) for a in self.owners)!=self.tokens:raise ValueError('persistent roots changed')
    def launch(self):
        self._validate_launch()
        result=self.library.launch_persistent(self.device_params.ptr,self.ws.nworld,self.ws.stream.cuda_stream)
        if result:raise RuntimeError(('persistent CUDA launch',result))
    def install(self):
        core=self
        self.solver.launch_device=MethodType(lambda owner:core.launch(),self.solver)
        self.solver.persistent_cuda_core=self
        return self
    def restore(self):
        self.solver.launch_device=self.original
