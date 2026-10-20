"""Cold opt-in collision fusion. Native and parallel-row boundaries are retained."""
import libero.gpu.precision as scalar
import ast
import copy
import hashlib
import inspect
from pathlib import Path
import textwrap
import warp as wp
from libero.gpu.host.runtime.collision import chain
from libero.gpu.host.runtime.collision import parallel_finalize_copy as pc
from libero.gpu.host.runtime.collision import detectors
from libero.gpu.reference.collision.collision_tree import kernel as tree_module
from libero.gpu.reference.collision.collision_dispatch import kernel as dispatch_module
from libero.gpu.reference.collision.contact_finalize import kernel as finalize_module
from libero.gpu.host.assembly.compose.compose import compose, _load
from libero.gpu.host.assembly.compose.backup import _overlap


class _OutputCopies:
    """Copy-only grid; preserves ordered fallback for overlapping fields."""
    def __init__(self, ws, restore=False):
        self.ws, self.restore = ws, restore
        self.fields = tuple((name, ws.outputs[name], ws.before_outputs[name]) for name in ws.outputs)
        self.pairs = tuple((before, arr) if restore else (arr, before) for _, arr, before in self.fields)
        destinations = [target for source, target in self.pairs]
        self.fallback = any(_overlap(dst, src) for dst in destinations for src, _ in self.pairs)
        self.fallback |= any(_overlap(a, b) for i, a in enumerate(destinations) for b in destinations[i+1:])
        self.fallback |= restore and any(_overlap(dst, ws.active) for dst in destinations)
        args = ['active: wp.array(dtype=wp.int32)'] if restore else []
        body = ['    cell = wp.tid()']
        types = {scalar.wp_real:scalar.name, wp.float32:'float32', wp.int32:'int32'}
        self.inputs = [ws.active] if restore else []
        for i, (source, target) in enumerate(self.pairs):
            if not source.is_contiguous or not target.is_contiguous or source.size != target.size:
                raise ValueError('contiguous matching C output buffers required')
            if not source.size: continue
            dtype=types[source.dtype]
            args += [f's{i}: wp.array(dtype=wp.{dtype})', f't{i}: wp.array(dtype=wp.{dtype})']
            condition=f'cell < {source.size}'
            if restore:
                condition += f' and active[cell // {source.size // ws.nworld}] == 0'
            body.append(f'    if {condition}: t{i}[cell] = s{i}[cell]')
            self.inputs += [source, target]
        code='@wp.kernel(enable_backward=False)\ndef copy_fields('+', '.join(args)+'):\n'+'\n'.join(body)+'\n'
        self.kernel=_load('collision_output_restore' if restore else 'collision_output_backup',code,{'wp':wp}).copy_fields
        self.dim=max(a.size for a,b in self.pairs)

    def launch(self, stream):
        if tuple(self.ws.outputs) != tuple(name for name, _, _ in self.fields):
            raise ValueError('C output inventory changed')
        if any(self.ws.outputs[name] is not arr or self.ws.before_outputs[name] is not before for name,arr,before in self.fields):
            raise ValueError('C output ownership changed')
        fallback = self.fallback or (self.restore and any(_overlap(target, self.ws.active) for source, target in self.pairs))
        if not fallback:
            # Input binding replaces active during cold preparation; use its actual retained owner.
            inputs = [self.ws.active] + self.inputs[1:] if self.restore else self.inputs
            wp.launch(self.kernel, self.dim, inputs=inputs,device=self.ws.device,stream=stream)
        else:
            for source,target in self.pairs:
                if not self.restore: wp.copy(target,source,stream=stream)
                else:
                    kernel={wp.int32:chain._restore_inactive_i32,scalar.wp_real:chain._restore_inactive_f64,wp.float32:chain._restore_inactive_f32}[target.dtype]
                    wp.launch(kernel,target.size,inputs=[self.ws.active,source,target,target.size//self.ws.nworld],device=self.ws.device,stream=stream)


def _generated_launch():
    namespace=dict(vars(chain))
    namespace['__name__']=__name__
    namespace['__file__']=__file__
    fn=ast.parse(textwrap.dedent(inspect.getsource(chain.CollisionChainWorkspace.launch))).body[0]

    def inline(method, owner, skip=0):
        body=ast.parse(textwrap.dedent(inspect.getsource(method))).body[0].body[skip:]
        prefix='_inlined_'+owner.replace('.','_')+'_'
        globals_=method.__globals__
        class Rewrite(ast.NodeTransformer):
            def visit_Name(self,node):
                if node.id=='self': return ast.parse(owner,mode='eval').body
                if method is pc.parallel_copy_launch and node.id in ('n','pairs','max_raw','raw_f'):
                    return ast.parse({'n':'N','pairs':'self.call_capacity','max_raw':'50','raw_f':'10'}[node.id],mode='eval').body
                if node.id=='s' and owner=='self.dispatch': return ast.parse('self.dispatch.static',mode='eval').body
                if node.id=='w' and owner=='self': return ast.Name(id='self',ctx=node.ctx)
                if node.id in globals_ and node.id not in ('wp',):
                    name=prefix+node.id; namespace[name]=globals_[node.id]
                    return ast.Name(id=name,ctx=node.ctx)
                return node
        return [Rewrite().visit(node) for node in body]

    expanded=[]
    for node in fn.body:
        call=node.value if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) else None
        target=ast.unparse(call.func) if call else ''
        if target=='self._stage_guard':
            status,stage,codes,stream=call.args
            codes=list(codes.elts)+[ast.Constant(-999)]*3
            text='wp.launch(_stage_error_kernel, dim=N, inputs=[STATUS, self.active, STAGE, C0, C1, C2, self.fe_stage,self.fe_code,self.fe_item,self.effective_active],device=self.device,stream=stream)'
            item=ast.parse(text).body[0]
            replacements={'STATUS':status,'STAGE':stage,**{f'C{i}':codes[i] for i in range(3)}}
            class Replace(ast.NodeTransformer):
                def visit_Name(self,n): return copy.deepcopy(replacements.get(n.id,n))
            expanded.append(Replace().visit(item))
        elif target=='self.tree.launch': expanded+=inline(tree_module.CollisionTreeWorkspace.launch,'self.tree')
        elif target=='self.dispatch.launch': expanded+=inline(dispatch_module.CollisionDispatchWorkspace.launch,'self.dispatch',skip=1)
        elif target=='self.detect.launch': expanded+=inline(detectors.DetectorWorkspace.launch,'self.detect',skip=1)
        elif target=='self.finalize.launch': expanded+=inline(finalize_module.ContactFinalizeWorkspace.launch,'self.finalize')
        elif target=='wp.launch' and ast.unparse(call.args[0])=='_calls_to_finalize_kernel':
            expanded+=inline(pc.parallel_copy_launch,'self',skip=2)
        elif isinstance(node,ast.For) and ast.unparse(node.iter)=='self.outputs.items()':
            is_restore=any(isinstance(n,ast.Assign) for n in node.body)
            expanded.append(ast.parse('self._fusion_restore.launch(stream)' if is_restore else 'self._fusion_backup.launch(stream)').body[0])
        else: expanded.append(node)

    def launch_info(node):
        if not isinstance(node,ast.Expr) or not isinstance(node.value,ast.Call):return None
        call=node.value
        if ast.unparse(call.func)!='wp.launch':return None
        kws={k.arg:k.value for k in call.keywords}
        if ast.unparse(kws.get('dim',ast.Constant(None))) not in ('N','self.tree.nworld','self.dispatch.nworld','self.detect.nworld','self.finalize.nworld','n'):
            return None
        kernel=eval(compile(ast.fix_missing_locations(ast.Expression(call.args[0])),__file__,'eval'),namespace)
        return kernel,kws['inputs'].elts

    groups=[]; result=[]; pending=[]
    def flush():
        if len(pending)<2:result.extend(pending);pending.clear();return
        stages=[]; inputs=[]
        for i,node in enumerate(pending):
            kernel,args=launch_info(node)
            names=tuple(inspect.signature(kernel.func).parameters)
            if len(names)!=len(args): raise ValueError('C source parameter count changed')
            mapping={name:f'g{i}_{name}' for name in names}
            stages.append((kernel,mapping));inputs.extend(args)
        name=f'_collision_group_{len(groups)+1}'
        kernel=compose(name,stages);namespace[name]=kernel
        result.append(ast.Expr(ast.Call(ast.Attribute(ast.Name('wp',ast.Load()),'launch',ast.Load()),[ast.Name(name,ast.Load())],
            [ast.keyword('dim',ast.Name('N',ast.Load())),ast.keyword('inputs',ast.List(inputs,ast.Load())),ast.keyword('device',ast.Attribute(ast.Name('self',ast.Load()),'device',ast.Load())),ast.keyword('stream',ast.Name('stream',ast.Load()))])))
        groups.append({'name':name,'stages':[k.key for k,m in stages],'original_launches':len(stages)})
        pending.clear()
    for node in expanded:
        if launch_info(node):pending.append(node)
        else:flush();result.append(node)
    flush()
    expected = [
        ['_input_error_kernel','_effective_active_kernel','_bounds_kernel','_stage_error_kernel','_broadphase_kernel','_stage_error_kernel','_stagebuild_kernel','_stage_error_kernel'],
        ['_tree_kernel','_stage_error_kernel','_req_to_dispatch_kernel','_dispatch','_stage_error_kernel','_dispatch_to_calls_kernel','_reset_error_kernel'],
        ['_detect_kernel','_detect_error_kernel','_begin'],
        ['_assemble','_chain_error_kernel','_effective_active_kernel','_restore_failed_contacts_kernel','_body_counters_kernel'],
    ]
    if [group['stages'] for group in groups] != expected:
        raise ValueError('C fusion stage inventory changed; cold source review required')
    fn.body=result
    module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    generated=ast.unparse(module)+'\n'
    exec(compile(module,__file__+':cold-launch','exec'),namespace)
    return namespace['launch'],groups,generated


# Construction source-derived, no mutation of any original module/class.
_LAUNCH,_GROUPS,_SOURCE=_generated_launch()
class FusedCollisionChainWorkspace(chain.CollisionChainWorkspace):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._fusion_backup=_OutputCopies(self)
        self._fusion_restore=_OutputCopies(self,restore=True)
    launch=_LAUNCH
    candidate_source_manifest={'schema':'collision-compose-r03','groups':_GROUPS,'parallel_rows_retained':True,'native_launch_retained':True,'generated_source':_SOURCE, 'source_files': {str(Path(m.__file__).resolve()): hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (chain, pc, detectors, tree_module, dispatch_module, finalize_module)}, 'candidate_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def source_derived_workspace_class():
    return FusedCollisionChainWorkspace





# Preserve the existing capacity planner, native extension and retained binding
# constructor, replacing only its cold workspace-class resolver.
from libero.gpu.host.runtime.collision.parallel_copy_runtime_factory import make_parallel_copy_adapter as _original_adapter
from libero.gpu.host.assembly.compose.factory import _derive
make_fused_adapter = _derive(_original_adapter, {'source_derived_workspace_class': source_derived_workspace_class})

