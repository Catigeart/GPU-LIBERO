"""Diagnostic S04 candidate: parallel finite/envelope validation only.

Structural and cone safety guards and all update arithmetic retain source
AST/order. A row/column rank preserves the original finite error priority;
any finite error precedes envelope errors. No production qualification.
"""
import libero.gpu.precision as scalar
from pathlib import Path
import ast
import copy
import hashlib
import importlib.util
import json
import sys
import warp as wp


@wp.kernel(enable_backward=False)
def _finite_cap(active: wp.array(dtype=wp.int32), structural: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32), jar: wp.array(dtype=scalar.wp_real),
    D: wp.array(dtype=scalar.wp_real), R: wp.array(dtype=scalar.wp_real), loss: wp.array(dtype=scalar.wp_real),
    J: wp.array(dtype=scalar.wp_real), first: wp.array(dtype=wp.int32),
    envelope: wp.array(dtype=wp.int32), rows: int, nv: int):
    tid=wp.tid()
    w=tid//rows
    r=tid-w*rows
    if active[w] != 1 or structural[w] != 0 or r >= count[w]:
        return
    row=w*rows+r
    base=r*(nv+1)
    if (not wp.isfinite(jar[row]) or not wp.isfinite(D[row]) or
            not wp.isfinite(R[row]) or not wp.isfinite(loss[row])):
        wp.atomic_min(first,w,base)
    for j in range(nv):
        if not wp.isfinite(J[row*nv+j]):
            wp.atomic_min(first,w,base+j+1)
    cap=scalar.wp_real(10000.0)
    lower=scalar.wp_real(-10000.0)
    d_cap=scalar.wp_real(2.0e16)
    if (jar[row]>cap or jar[row]<lower or D[row]>d_cap or D[row]<lower or
            R[row]>cap or R[row]<lower or loss[row]>cap or loss[row]<lower):
        wp.atomic_or(envelope,w,1)
    for j in range(nv):
        value=J[row*nv+j]
        if value>cap or value<lower:
            wp.atomic_or(envelope,w,1)


@wp.kernel(enable_backward=False)
def _resolve_validation(structural: wp.array(dtype=wp.int32), first: wp.array(dtype=wp.int32),
                        envelope: wp.array(dtype=wp.int32), result: wp.array(dtype=wp.int32), nv:int):
    w=wp.tid()
    result[w]=structural[w]
    if structural[w] == 0:
        if first[w] != 1073741823:
            if first[w] % (nv+1) == 0:
                result[w]=8
            else:
                result[w]=9
        elif envelope[w] != 0:
            result[w]=10


def validation_source():
    from libero.gpu.reference.constraints.constraint_update import kernel as original
    path=Path(original.__file__)
    raw=path.read_bytes()
    tree=ast.parse(raw.decode('utf-8-sig'))
    fn=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='_update')
    finite=[]
    envelope=[]
    for i,node in enumerate(fn.body):
        if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='r':
            text=ast.unparse(node)
            if 'wp.isfinite(jar[' in text:finite.append(i)
            if 'jar[rb + r] > cap' in text:envelope.append(i)
    if len(finite)!=1 or len(envelope)!=1:
        raise ValueError('S04 retained validation AST no longer matches source proof')
    prefix=copy.deepcopy(fn)
    prefix.name='_structural_prefix'
    prefix.body=prefix.body[:finite[0]]
    for name in ('first','envelope'):
        prefix.args.args.append(ast.arg(arg=name,annotation=ast.parse('wp.array(dtype=wp.int32)',mode='eval').body))
    prefix.body[2:2]=ast.parse('first[w]=1073741823\nenvelope[w]=0').body
    body=copy.deepcopy(fn)
    body.name='_update_prevalidated'
    body.args.args.append(ast.arg(arg='prevalidated',annotation=ast.parse('wp.array(dtype=wp.int32)',mode='eval').body))
    body.body=[node for i,node in enumerate(body.body) if i not in (finite[0],envelope[0])]
    body.body[2:2]=ast.parse('if prevalidated[w]!=0:\n    status[w]=prevalidated[w]\n    return').body
    product="import warp as wp\nwp.set_module_options({'enable_backward':False,'fuse_fp':True,'fast_math':False})\nfrom libero.gpu.reference.constraints.constraint_update.kernel import _flip_sign, _cone_t, _u\n\n"
    product+=ast.unparse(ast.fix_missing_locations(prefix))+'\n\n'
    product+=ast.unparse(ast.fix_missing_locations(body))+'\n'
    manifest=dict(source_path=str(path),source_sha256=hashlib.sha256(raw).hexdigest(),
                  generated_sha256=hashlib.sha256(product.encode()).hexdigest(),
                  removed_only_validation_loops=finite+envelope,
                  retained_math_ast=True,acceptance='diagnostic-pending')
    return product,manifest


def load_validation(build_root):
    product,manifest=validation_source()
    identity=hashlib.sha256((product+Path(__file__).read_text(encoding='utf-8-sig')).encode()).hexdigest()
    name='_libero_s04_parallel_validation_'+identity
    root=Path(build_root).resolve()/name
    root.mkdir(parents=True,exist_ok=True)
    path=root/'kernel.py'
    encoded=product.encode()
    if path.exists() and path.read_bytes()!=encoded:
        raise ValueError('parallel validation source cache drift')
    if not path.exists():path.write_bytes(encoded)
    manifest=dict(manifest,candidate_sha256=identity,
                  generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    metadata=root/'manifest.json'
    packed=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode()
    if metadata.exists() and metadata.read_bytes()!=packed:
        raise ValueError('parallel validation manifest drift')
    if not metadata.exists():metadata.write_bytes(packed)
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module
        spec.loader.exec_module(module)
    return sys.modules[name],manifest


def parallel_validation_solver_type(source_solver_type, *, build_root):
    module,manifest=load_validation(build_root)
    math_module=next(sys.modules[cls.__module__] for cls in source_solver_type.__mro__ if hasattr(sys.modules.get(cls.__module__), 'NV') and hasattr(sys.modules.get(cls.__module__), 'K'))
    nv=int(math_module.NV)
    class ParallelValidationSolver(source_solver_type):
        def __init__(self,workspace):
            super().__init__(workspace)
            if workspace.rows*(nv+1)>=1073741823:
                raise ValueError('finite-priority rank exceeds certified int32 encoding')
            self.validation_manifest=manifest
            self._prefix_status=wp.zeros(workspace.nworld,dtype=wp.int32,device=workspace.device)
            self._finite_first=wp.zeros(workspace.nworld,dtype=wp.int32,device=workspace.device)
            self._envelope=wp.zeros(workspace.nworld,dtype=wp.int32,device=workspace.device)
            self._prevalidated=wp.zeros(workspace.nworld,dtype=wp.int32,device=workspace.device)

        def _s04_update(self,jar,flag):
            ws=self.ws
            args=[ws.active,ws.count,ws.ne,ws.nf,ws.ncon,flag,jar,self._D_flat,
                  self._R_flat,self._loss_flat,self._type_flat,self._id_flat,self._J_flat,
                  self._mu_flat,self._fr_flat,self._dim_flat,self._force_flat,self._state_flat,
                  self._H_contact_flat,ws.cost,ws.status,ws.rows,ws.contacts,nv]
            prefix=list(args)
            prefix[20]=self._prefix_status
            wp.launch(module._structural_prefix,ws.nworld,
                      inputs=[*prefix,self._finite_first,self._envelope],device=ws.device,stream=ws.stream)
            wp.launch(_finite_cap,ws.nworld*ws.rows,
                      inputs=[ws.active,self._prefix_status,ws.count,jar,self._D_flat,self._R_flat,
                              self._loss_flat,self._J_flat,self._finite_first,self._envelope,ws.rows,nv],
                      device=ws.device,stream=ws.stream)
            wp.launch(_resolve_validation,ws.nworld,
                      inputs=[self._prefix_status,self._finite_first,self._envelope,self._prevalidated,nv],
                      device=ws.device,stream=ws.stream)
            wp.launch(module._update_prevalidated,ws.nworld,inputs=[*args,self._prevalidated],
                      device=ws.device,stream=ws.stream)
    return ParallelValidationSolver

