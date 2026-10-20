"""Diagnostic device capture of genuine post-assemble/pre-factor S05 inputs.

No solver math changes. Cold derived class inserts private copies around the
original factor_update launch. Overflow invalidates packet completeness.
"""
import libero.gpu.precision as scalar
import ast
import hashlib
import inspect
import textwrap
import numpy as np
import warp as wp

wp.set_module_options({'enable_backward':False,'fuse_fp':True,'fast_math':False})

@wp.kernel(enable_backward=False)
def _select(active:wp.array(dtype=wp.int32),status:wp.array(dtype=wp.int32),
            attempts:wp.array(dtype=wp.int64),accepted:wp.array(dtype=wp.int32),
            slot:wp.array(dtype=wp.int32),overflow:wp.array(dtype=wp.int32),
            ordinal:wp.array(dtype=wp.int64),capacity:int,nworld:int):
    w=wp.tid()
    attempts[w]+=wp.int64(1)
    slot[w]=-1
    if active[w]!=0 and status[w]==0:
        k=accepted[w]
        if k>=capacity:
            overflow[w]=1
            return
        slot[w]=k
        accepted[w]=k+1
        ordinal[k*nworld+w]=attempts[w]

@wp.kernel(enable_backward=False)
def _copy_f64(src:wp.array(dtype=scalar.wp_real),dst:wp.array(dtype=scalar.wp_real),
              slot:wp.array(dtype=wp.int32),stride:int,nworld:int):
    tid=wp.tid();w=tid//stride;k=slot[w]
    if k>=0:dst[(k*nworld+w)*stride+tid%stride]=src[tid]

@wp.kernel(enable_backward=False)
def _copy_i32(src:wp.array(dtype=wp.int32),dst:wp.array(dtype=wp.int32),
              slot:wp.array(dtype=wp.int32),stride:int,nworld:int):
    tid=wp.tid();w=tid//stride;k=slot[w]
    if k>=0:dst[(k*nworld+w)*stride+tid%stride]=src[tid]

FIELDS=('count','ncon','active','kind','J','D','state','oldstate','qM','madr',
        'parentid','simplenum','efc_id','contact_dim','contact_H','ncone','H',
        'Hcone','nnz','nupdate','local','ltj','vec','status','route')
STATIC=('madr','parentid','simplenum')
OUTPUTS=('H','Hcone','nnz','nupdate','local','ltj','vec','status','route')

def instrument_s05_method(original):
    """Reject source-layout changes; retain every original launch AST."""
    source=textwrap.dedent(inspect.getsource(original));tree=ast.parse(source)
    fn=tree.body[0]
    launches=[i for i,n in enumerate(fn.body) if isinstance(n,ast.Expr)
              and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='wp.launch']
    if len(launches)!=3:raise ValueError('unexpected S05 launch layout')
    kernels=[ast.unparse(fn.body[i].value.args[0]) for i in launches]
    if kernels!=['s05.validate','s05.assemble','s05.factor_update']:raise ValueError('S05 source order drift')
    before=ast.parse('self.factor_capture.before(kind_arr)').body[0]
    after=ast.parse('self.factor_capture.after()').body[0]
    i=launches[2];fn.body[i:i]=[before];fn.body.insert(i+2,after)
    product=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    recovered=ast.parse(product);recovered.body[0].body.pop(i+2);recovered.body[0].body.pop(i)
    if ast.dump(recovered,include_attributes=False)!=ast.dump(ast.parse(source),include_attributes=False):raise ValueError('capture modified retained math')
    namespace=original.__globals__.copy()
    exec(compile(ast.fix_missing_locations(tree),'<genuine-s05-pre-factor-capture>','exec'),namespace)
    return namespace[original.__name__],dict(original_method_sha256=hashlib.sha256(source.encode()).hexdigest(),generated_method_sha256=hashlib.sha256(product.encode()).hexdigest(),retained_launch_ast_exact=True,source=product)

class FactorCapture:
    def __init__(self,solver,capacity=32,max_bytes=512*1024*1024):
        if not 1<=capacity<=101:raise ValueError('capacity outside diagnostic schedule')
        self.solver=solver;self.ws=ws=solver.ws;self.capacity=capacity;self.nworld=ws.nworld
        n=ws.nworld
        self.device=wp.get_device(ws.device)
        if n<1 or capacity*n>=2**31:raise ValueError('capture offset overflow')
        self.sources=dict(count=ws.count,ncon=ws.ncon,active=ws.active,kind=solver._kind_direct,
            J=solver._J_flat,D=solver._D_flat,state=solver._state_flat,oldstate=solver._oldstate_flat,
            qM=solver._qM_flat,madr=ws.madr,parentid=ws.parentid,simplenum=ws.simplenum,
            efc_id=solver._id_flat,contact_dim=solver._dim_flat,contact_H=solver._H_contact_flat,
            ncone=ws.ncone,H=solver._H_flat,Hcone=solver._Hcone_flat,nnz=ws.nnz,nupdate=ws.nupdate,
            local=solver._H_local_flat,ltj=solver._H_ltj_flat,vec=solver._H_vec_flat,status=ws.status,route=ws.route)
        total=0
        for name,src in self.sources.items():
            if src.dtype not in (wp.int32,scalar.wp_real) or src.device!=self.device:raise ValueError('typed capture source mismatch')
            if not src.is_contiguous:raise ValueError('noncontiguous source')
            if name not in STATIC and (src.size%n or capacity*src.size>=2**31):raise ValueError('perworld source layout overflow')
            total+=(src.size if name in STATIC else capacity*src.size)*(4 if src.dtype==wp.int32 else 8)
            if name in OUTPUTS:total+=capacity*src.size*(4 if src.dtype==wp.int32 else 8)
        if total>max_bytes:raise ValueError('cold capture byte budget exceeded')
        self.byte_budget=total;self.pre={};self.post={};self.static={}
        with wp.ScopedStream(ws.stream):
            self.attempts=wp.zeros(n,dtype=wp.int64,device=ws.device)
            self.accepted=wp.zeros(n,dtype=wp.int32,device=ws.device)
            self.slot=wp.full(n,-1,dtype=wp.int32,device=ws.device)
            self.overflow=wp.zeros(n,dtype=wp.int32,device=ws.device)
            self.ordinal=wp.zeros(capacity*n,dtype=wp.int64,device=ws.device)
            for name,src in self.sources.items():
                if name in STATIC:
                    dst=wp.empty(src.size,dtype=src.dtype,device=ws.device);wp.copy(dst,src,stream=ws.stream);self.static[name]=dst
                else:self.pre[name]=wp.zeros(capacity*src.size,dtype=src.dtype,device=ws.device)
                if name in OUTPUTS:self.post[name]=wp.zeros(capacity*src.size,dtype=src.dtype,device=ws.device)
    def _copy(self,destination):
        ws=self.ws
        for name,dst in destination.items():
            src=self.sources[name];stride=src.size//ws.nworld
            kernel=_copy_i32 if src.dtype==wp.int32 else _copy_f64
            wp.launch(kernel,src.size,inputs=[src,dst,self.slot,stride,ws.nworld],device=ws.device,stream=ws.stream)
    def before(self,kind_arr):
        # kind may be the solver's full/incremental/cone-specific owner.
        if kind_arr.dtype!=wp.int32 or kind_arr.size!=self.nworld or kind_arr.device!=self.device:raise ValueError('kind descriptor mismatch')
        self.sources['kind']=kind_arr
        ws=self.ws
        wp.launch(_select,ws.nworld,inputs=[ws.active,ws.status,self.attempts,self.accepted,self.slot,self.overflow,self.ordinal,self.capacity,ws.nworld],device=ws.device,stream=ws.stream)
        self._copy(self.pre)
    def after(self):self._copy(self.post)
    def download(self):
        """Caller must be outside capture/hot loop; explicit stream fence."""
        wp.synchronize_stream(self.ws.stream)
        result={name:array.numpy() for name,array in dict(attempts=self.attempts,accepted=self.accepted,overflow=self.overflow,ordinal=self.ordinal).items()}
        for prefix,group in (('pre',self.pre),('post',self.post),('static',self.static)):
            for name,array in group.items():result[prefix+'.'+name]=array.numpy()
        if np.any(result['overflow']):raise ValueError('capture overflow: incomplete pre-factor witness')
        return result

def factor_capture_solver_type(original_type,capacity=32,max_bytes=512*1024*1024):
    owner=next(cls for cls in original_type.__mro__ if '_s05_launch' in cls.__dict__)
    method,proof=instrument_s05_method(owner.__dict__['_s05_launch'])
    class CapturedSolver(original_type):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.factor_capture=FactorCapture(self,capacity,max_bytes)
    CapturedSolver._s05_launch=method
    CapturedSolver.capture_source_proof=proof
    return CapturedSolver
