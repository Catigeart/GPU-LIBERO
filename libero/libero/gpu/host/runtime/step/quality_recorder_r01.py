"""Cold passive copies at actual OSC and Newton consumption boundaries."""
import ast
import inspect
import textwrap
import warp as wp
from libero.gpu.host.runtime.dynamics import runtime_adapter as d_original
from . import shape_physics as f_original


def _token(a):
    return (id(a),int(a.ptr),a.shape,a.strides,a.dtype,str(a.device))

class DevicePacket:
    def __init__(self,sources,stream):
        self.sources=dict(sources)
        self.stream=stream
        self.tokens={k:_token(a) for k,a in self.sources.items()}
        with wp.ScopedStream(stream):
            self.copies={k:wp.zeros_like(a) for k,a in self.sources.items()}
        self.copy_tokens={k:_token(a) for k,a in self.copies.items()}
    def capture(self):
        if {k:_token(a) for k,a in self.sources.items()}!=self.tokens:
            raise ValueError("quality source owner changed")
        if {k:_token(a) for k,a in self.copies.items()}!=self.copy_tokens:
            raise ValueError("quality witness owner changed")
        for k,a in self.sources.items():wp.copy(self.copies[k],a,stream=self.stream)
    def download(self):
        wp.synchronize_stream(self.stream)
        return {k:a.numpy() for k,a in self.copies.items()}


def _derived_method(method,namespace,*,before_call,before_hook,after_hook=None):
    tree=ast.parse(textwrap.dedent(inspect.getsource(method)))
    count=0
    def hook(name):return ast.Expr(ast.Call(ast.Attribute(ast.Name("self",ast.Load()),name,ast.Load()),[],[]))
    class Insert(ast.NodeTransformer):
        def visit_Expr(self,node):
            nonlocal count
            if isinstance(node.value,ast.Call) and ast.unparse(node.value.func)==before_call:
                count+=1
                return [hook(before_hook),node]+([hook(after_hook)] if after_hook else [])
            return node
    tree=Insert().visit(tree)
    if count!=1:raise ValueError("one original producer boundary required")
    # Erasing inserted hooks must reproduce the complete original AST.
    baseline=ast.parse(textwrap.dedent(inspect.getsource(method)))
    restored=ast.parse(ast.unparse(tree))
    class Erase(ast.NodeTransformer):
        def visit_Expr(self,node):
            if isinstance(node.value,ast.Call) and ast.unparse(node.value.func) in ("self."+before_hook,"self."+str(after_hook)):
                return None
            return node
    restored=Erase().visit(restored)
    if ast.dump(restored,include_attributes=False)!=ast.dump(baseline,include_attributes=False):
        raise ValueError("passive insertion changed original method AST")
    ast.fix_missing_locations(tree)
    local=dict(namespace)
    exec(compile(tree,method.__code__.co_filename+":quality-cold","exec"),local)
    return local[method.__name__]

class RuntimeControllerAdapter(d_original.RuntimeControllerAdapter):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        c=self.controller
        fields=tuple(dict.fromkeys((*c.WIDTHS,*self._physical_names,"active","policy_step","error","cache_valid","needs_update")))
        sources={k:getattr(c,k) for k in fields}
        sources.update(parent_valid=self.parents.valid,parent_written=self.parents.written,
            parent_expected=self.parents.expected,first_error=self.first_error,event_before=self.event,
            written_before=self.written,consumed_parent_before=self.consumed_parent)
        self.quality_control=DevicePacket(sources,self.stream)
    def _quality_control_capture(self):self.quality_control.capture()

RuntimeControllerAdapter.enqueue=_derived_method(d_original.RuntimeControllerAdapter.enqueue,
    vars(d_original),before_call="c.launch",before_hook="_quality_control_capture")

class ShapePhysicsForward(f_original.ShapePhysicsForward):
    def __init__(self,*args,**kwargs):
        kwargs['capture_entry_witness'] = True
        super().__init__(*args,**kwargs)
        n=self.n
        extra={k:getattr(n.ws,k) for k in ("cone_dim","cone_mu","cone_friction")}
        extra.update(expected_newton=self.expected_newton,expected_forward=self.expected_forward,
            state_epoch=self.d.state_epoch,first_error=self.d.first_error,upstream=self.n_upstream,
            mask=self.mask,projection_status=self.projection_status)
        extra.update({"canonical_before__"+k:a for k,a in n.public_arrays.items() if isinstance(a,wp.array)})
        self.quality_preN=DevicePacket(extra,self.stream)
        after={"canonical_after__"+k:a for k,a in n.public_arrays.items() if isinstance(a,wp.array)}
        after.update({"work__"+k:getattr(n.ws,k) for k in ("err_code","solver_exit","s08d_status","qacc","force","state")})
        event=getattr(n,"final_event",None)
        if event is not None:
            after.update({"final_event__"+k:getattr(event,k) for k in ("candidate","candidate_attempt","candidate_completed","candidate_origin","source_stage","rejected_final_entry")})
        self.quality_postN=DevicePacket(after,self.stream)
    def _quality_preN_capture(self):self.quality_preN.capture()
    def _quality_postN_capture(self):self.quality_postN.capture()
    def download_quality(self):
        self.validate()
        wp.synchronize_stream(self.stream)
        packet={"preN__"+k:a.numpy() for k,a in self.entry_witness.items()}
        packet.update(self.quality_preN.download());packet.update(self.quality_postN.download())
        return packet

ShapePhysicsForward.enqueue=_derived_method(f_original.ShapePhysicsForward.enqueue,
    vars(f_original),before_call="self.n.enqueue",before_hook="_quality_preN_capture",after_hook="_quality_postN_capture")
