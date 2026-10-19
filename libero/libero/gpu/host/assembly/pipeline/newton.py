"""Cold compose expansion of the native Newton schedule.

Only adjacent one-thread-per-world launches compose. Transpose, Hessian
assembly and S07 keep their original stream synchronization boundaries.
The original stage bodies, masked loop and diagnostic tape are retained.
"""
import ast
import copy
import hashlib
import inspect
import textwrap

import warp as wp
from libero.gpu.host.assembly.compose.compose import compose, _load
from libero.gpu.host.assembly.compose.newton import FusedNewtonMixin, publish_newton
from libero.gpu.host.runtime.step.newton_final_event import FinalEventAdapter, _final_event


_INLINE = {"_warmstart_select", "_jacvec_sub", "_mulM", "_gauss", "_count",
           "_s04_update", "_s05_launch", "_s06_launch", "_s08a", "_s08b",
           "_s08c", "_s08d"}


def _compile_schedule(original, method_name, terminal=(), extra_namespace=None, identity_suffix="",
                      kernel_identity_suffix=None):
    method = getattr(original, method_name)
    namespace = dict(method.__globals__)
    namespace.update(extra_namespace or {})
    fn = ast.parse(textwrap.dedent(inspect.getsource(method))).body[0]
    identity = hashlib.sha256((original.__module__ + identity_suffix).encode()).hexdigest()[:16]
    # Compiled kernels take owner arrays as arguments. Only the Python launcher
    # captures a process-local owner and therefore needs a unique module name.
    kernel_identity = (identity if kernel_identity_suffix is None else
        hashlib.sha256((original.__module__ + kernel_identity_suffix).encode()).hexdigest()[:16])
    kernels, groups = [], []

    def expand(statements):
        result = []
        for statement in statements:
            call = statement.value if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call) else None
            if (call is not None and isinstance(call.func, ast.Attribute)
                    and ast.unparse(call.func.value) == "self" and call.func.attr in _INLINE):
                helper = getattr(original, call.func.attr)
                namespace.update(helper.__globals__)
                tree = ast.parse(textwrap.dedent(inspect.getsource(helper))).body[0]
                parameters = [a.arg for a in tree.args.args[1:]]
                if len(parameters) != len(call.args) or call.keywords:
                    raise ValueError((call.func.attr, "helper call shape changed"))
                replacements = dict(zip(parameters, call.args))

                class Substitute(ast.NodeTransformer):
                    def visit_Name(self, node):
                        return copy.deepcopy(replacements[node.id]) if node.id in replacements else node

                body = [Substitute().visit(copy.deepcopy(s)) for s in tree.body
                        if not (isinstance(s, ast.Assign) and ast.unparse(s) == "ws = self.ws")
                        and not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                                 and isinstance(s.value.value, str))]
                if call.func.attr == "_jacvec_sub":
                    qacc_name = ast.unparse(replacements["qacc"])
                    aliases = {"ws.qacc": "self._qacc_flat", "ws.qacc_smooth": "self._qacc_smooth_flat",
                               "ws.qacc_warmstart": "self._qacc_warmstart_flat"}
                    if qacc_name not in aliases:
                        raise ValueError(("unsupported jacvec alias", qacc_name))
                    class FlatAlias(ast.NodeTransformer):
                        def visit_Subscript(self, node):
                            if isinstance(node.value, ast.Name) and node.value.id == "flat":
                                return ast.parse(aliases[qacc_name], mode="eval").body
                            return self.generic_visit(node)
                    body = [FlatAlias().visit(s) for s in body
                            if not (isinstance(s, ast.Assign) and any(isinstance(t, ast.Name)
                                    and t.id == "flat" for t in s.targets))]
                result.extend(expand(body))
            else:
                for attribute in ("body", "orelse"):
                    if hasattr(statement, attribute):
                        setattr(statement, attribute, expand(getattr(statement, attribute)))
                result.append(statement)
        return result

    def descriptor(statement):
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            return None
        call = statement.value
        if ast.unparse(call.func) == "self._glue":
            if len(call.args) != 2 or call.keywords:
                return None
            kernel = eval(ast.unparse(call.args[0]), namespace)
            return kernel, copy.deepcopy(call.args[1].elts)
        if ast.unparse(call.func) != "wp.launch":
            return None
        if len(call.args) < 2 or ast.unparse(call.args[1]) != "ws.nworld":
            return None
        kernel = eval(ast.unparse(call.args[0]), namespace)
        args = []
        for keyword in call.keywords:
            if keyword.arg in ("inputs", "outputs"):
                if not isinstance(keyword.value, ast.List):
                    raise ValueError("explicit original launch arguments required")
                args.extend(copy.deepcopy(keyword.value.elts))
        return kernel, args

    def fuse(statements):
        output, pending = [], []

        def flush():
            if not pending:
                return
            stages, bindings = [], {}
            for stage_index, (kernel, expressions) in enumerate(pending):
                params = list(inspect.signature(kernel.func).parameters)
                if len(params) != len(expressions):
                    raise ValueError((kernel.key, len(params), len(expressions)))
                mapping = {p: "s%d_%s" % (stage_index, p) for p in params}
                stages.append((kernel, mapping))
                bindings.update({mapping[p]: expression for p, expression in zip(params, expressions)})
            index = len(kernels)
            kernel = compose("newton_r03_%s_%s_%d" % (kernel_identity, method_name, index), stages)
            kernels.append(kernel)
            groups.append(tuple(k.key for k, _ in pending))
            statement = ast.parse("wp.launch(_schedule_kernels[0], ws.nworld, inputs=[], device=ws.device, stream=ws.stream)").body[0]
            statement.value.args[0].slice = ast.Constant(index)
            statement.value.keywords[0].value = ast.List(
                elts=[bindings[p] for p in kernel.fusion_argument_names], ctx=ast.Load())
            output.append(statement)
            pending.clear()

        for statement in statements:
            desc = descriptor(statement)
            if desc is not None:
                pending.append(desc)
                continue
            flush()
            for attribute in ("body", "orelse"):
                if hasattr(statement, attribute):
                    setattr(statement, attribute, fuse(getattr(statement, attribute)))
            output.append(statement)
        flush()
        return output

    expanded = expand(fn.body)
    # Optional final-event/publication stages append to the final world group;
    # no event state is read by any earlier solver stage.
    for kernel, expressions in terminal:
        key = "_terminal_%d" % len(namespace)
        namespace[key] = kernel
        statement = ast.parse("wp.launch(" + key + ", ws.nworld, inputs=[], device=ws.device, stream=ws.stream)").body[0]
        statement.value.keywords[0].value = ast.List(
            elts=[ast.parse(expression, mode="eval").body for expression in expressions], ctx=ast.Load())
        expanded.append(statement)
    fn.body = fuse(expanded)
    fn.decorator_list = []
    ast.fix_missing_locations(fn)
    namespace["_schedule_kernels"] = tuple(kernels)
    source = ast.unparse(fn) + "\n"
    # This module contains a Python launcher only. Registering an empty Warp
    # module would trigger a dummy compilation for every unique owner name.
    function = getattr(_load("newton_r03_launcher_" + identity + "_" + method_name,
                             source, namespace, warp_module=False), method_name)
    return function, tuple(groups), tuple(kernels), source


def fused_solver_class(package, base_class=None):
    """Cold native-dimension solver class; no mutation of original modules."""
    original = package.solver.FwdConstraintSolver
    base = original if base_class is None else base_class
    schedule = _compile_schedule(original, "launch_device")
    iteration = _compile_schedule(original, "_loop_body")
    return type("FusedNewtonR03Solver", (base,), {
        "__module__": original.__module__,
        "launch_device": schedule[0], "_loop_body": iteration[0],
        "fusion_schedule_groups": schedule[1], "fusion_schedule_kernels": schedule[2],
        "fusion_schedule_source": schedule[3],
        "fusion_iteration_groups": iteration[1], "fusion_iteration_kernels": iteration[2],
        "fusion_iteration_source": iteration[3], "fusion_iteration_original": original})


class FusedFinalEventAdapter(FusedNewtonMixin, FinalEventAdapter):
    """Final event and publication join the last solver world group.

    The public commit is still a separate parallel grid. The original event
    root tokens are checked before the solver, as in FinalEventAdapter.
    """
    def __init__(self, workspace, *, rules, solver_factory):
        super().__init__(workspace, rules=rules, solver_factory=solver_factory)
        original = solver_factory.fusion_iteration_original
        event_factory = self._solver_factory
        owner = self
        event_args = ["ws.qacc", "ws.empty", "ws.newton_world", "ws.solver_exit",
                      "ws.err_code", "ws.s08d_status", "owner.n_attempt", "self.candidate",
                      "self.candidate_attempt", "self.candidate_completed", "self.candidate_origin",
                      "self.source_stage", "self.rejected_final_entry", "ws.qacc.shape[1]"]
        publication_values = dict(upstream="owner._r03_upstream", enabled="owner._r03_enabled",
            error="ws.err_code", valid="owner.valid", written="owner.written",
            publication_error="owner.publication_error", expected="owner.n_expected",
            n_written="owner.n_written")
        terminal = ((_final_event, event_args), (publish_newton,
                    [publication_values[p] for p in publish_newton.fusion_argument_names]))
        compiled = _compile_schedule(original, "launch_device", terminal,
                                     {"owner": owner}, ":event-owner:" + str(id(owner)),
                                     kernel_identity_suffix=":event-publication")
        launch = compiled[0]

        def launch_device(self):
            from libero.gpu.host.runtime.step.newton_final_event import _token
            roots = (self.candidate, self.candidate_attempt, self.candidate_completed,
                     self.candidate_origin, self.source_stage, self.rejected_final_entry, owner.n_attempt)
            if tuple(_token(a) for a in roots) != self._event_tokens:
                raise ValueError("final event owner changed")
            launch(self)

        self._solver_factory = type("FusedEventPublicationSolver", (event_factory,), {
            "__module__": original.__module__, "launch_device": launch_device,
            "fusion_schedule_groups": compiled[1], "fusion_schedule_kernels": compiled[2],
            "fusion_schedule_source": compiled[3]})


def _enqueue_method():
    method = FusedNewtonMixin.enqueue
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    count = 0

    class RemovePublish(ast.NodeTransformer):
        def visit_Expr(self, node):
            nonlocal count
            call = node.value
            if isinstance(call, ast.Call) and ast.unparse(call.func) == "wp.launch" and call.args and ast.unparse(call.args[0]) == "publish_newton":
                count += 1
                return None
            return self.generic_visit(node)

        def visit_With(self, node):
            node = self.generic_visit(node)
            node.body = ast.parse("self._r03_upstream = upstream_status\nself._r03_enabled = enabled").body + node.body
            return node

    tree = RemovePublish().visit(tree)
    if count != 1:
        raise ValueError("one original publication launch required")
    ast.fix_missing_locations(tree)
    return _load("newton_r03_event_enqueue", ast.unparse(tree) + "\n",
                 dict(method.__globals__)).enqueue


FusedFinalEventAdapter.enqueue = _enqueue_method()


def install_cuda_entry(adapter,core):
    """Keep live entry ownership checks while fusing device admission and commit."""
    from types import MethodType
    from libero.gpu.host.assembly.compose import newton as entry
    def enqueue(self,upstream_status,active=None,expected_epoch=None,publication_status=None):
        if not self._sealed:raise RuntimeError("seal bindings before enqueue")
        w=self.ws
        enabled=self._all_enabled if active is None else active
        publication=self._no_publication_error if publication_status is None else publication_status
        epoch=self.n_attempt if expected_epoch is None else expected_epoch
        for name,array,template in (("upstream_status",upstream_status,w.err_code),
            ("active",enabled,w.active),("publication_status",publication,self.publication_error),
            ("expected_epoch",epoch,self.n_expected)):
            self._check_array(name,array,template)
        protected={getattr(w,name).ptr for name in entry.original.RESET_INT}
        if upstream_status.ptr in protected:raise ValueError("upstream status must not alias reset-owned scratch")
        if enabled.ptr in protected|{w.active.ptr,w.empty.ptr,w.newton_world.ptr}:
            raise ValueError("active mask must not alias N-owned scratch")
        self._r03_upstream=upstream_status;self._r03_enabled=enabled
        if core.manifest.get('external_tail_stages',0):
            prevalidate=getattr(self,'cuda_external_tail_prevalidate',None)
            sealed=getattr(self,'cuda_external_tail_prevalidate_ref',None)
            if prevalidate is None or prevalidate is not sealed:
                raise ValueError("external post validation hook changed")
            # Validate post owners before any joint-kernel write, including
            # blocked invocations. Late sensor callbacks cannot protect writes.
            prevalidate(self.stream)
        core.launch_entry(upstream_status,enabled,publication,epoch,expected_epoch is not None)
        return self.public_arrays
    adapter.enqueue=MethodType(enqueue,adapter)
    adapter.cuda_entry_core=core
    return adapter
