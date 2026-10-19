"""Cold composition of independent-world kernels, with unchanged stage bodies.

Each original kernel becomes a device function. A stage-local early return
returns from that function, so the following stage still runs as before.
Only the one-dimensional tid is supplied by the fused caller. No arithmetic,
checks, array writes, or floating-point options are rewritten.
"""
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
import ast
import copy
import hashlib
import inspect
import linecache
from pathlib import Path
import sys
import textwrap
import types

import warp as wp

_CACHE = {}


def _load(name, source, namespace, *, warp_module=True):
    digest = hashlib.sha256(source.encode()).hexdigest()
    module_name = __name__ + "._" + name + "_" + digest[:16]
    if module_name in sys.modules:
        return sys.modules[module_name]
    filename = str((GPU_ROOT / 'host/assembly/compose/_generated') / (name + "_" + digest + ".py"))
    # Warp inspects Python functions during compilation. Retain exact source,
    # including for cold-generated functions that do not need a disk cache.
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    module = types.ModuleType(module_name)
    module.__dict__.update(namespace)
    module.__name__, module.__file__ = module_name, filename
    sys.modules[module_name] = module
    exec(compile(source, filename, "exec"), module.__dict__)
    if warp_module:
        wp.set_module_options({"enable_backward": False, "fuse_fp": True,
                               "fast_math": False}, module=module_name)
    module.generated_source = source
    return module


def _function(kernel):
    raw = textwrap.dedent(inspect.getsource(kernel.func))
    tree = ast.parse(raw)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    fn.name = "_body"
    fn.decorator_list = [ast.parse("wp.func", mode="eval").body]
    fn.args.args.insert(0, ast.arg(arg="world_index", annotation=ast.Name(id="int", ctx=ast.Load())))

    class Tid(ast.NodeTransformer):
        def visit_Call(self, node):
            if (isinstance(node.func, ast.Attribute) and
                    isinstance(node.func.value, ast.Name) and
                    node.func.value.id == "wp" and node.func.attr == "tid"):
                if node.args or node.keywords:
                    raise ValueError("only independent one-dimensional world kernels can compose")
                return ast.Name(id="world_index", ctx=ast.Load())
            return self.generic_visit(node)

    fn = Tid().visit(fn)
    ast.fix_missing_locations(fn)
    # Identical function text may belong to different native model packages,
    # whose NV constants and helper functions differ. Keep their cold bindings
    # separate rather than reusing a module based on function text alone.
    owner = hashlib.sha256(kernel.func.__module__.encode()).hexdigest()[:16]
    module = _load(kernel.key + "_body_" + owner, ast.unparse(fn) + "\n", dict(kernel.func.__globals__))
    return module._body, fn.args.args[1:], raw


def compose(name, stages):
    """stages: (original_kernel, original-argument -> fused-argument mapping).

    Only use when one world owns every mutated cell and there is no grid-wide
    producer/consumer dependency between stages. Scalar expressions belong in
    the launcher; every mapping value is a simple argument name.
    """
    key = (name, tuple((k, tuple(mapping.items())) for k, mapping in stages))
    if key in _CACHE:
        return _CACHE[key]
    args, calls, namespace, originals, argument_types = {}, [], {"wp": wp}, [], {}
    for index, (kernel, mapping) in enumerate(stages):
        function, original_args, raw = _function(kernel)
        if set(mapping) != {a.arg for a in original_args}:
            raise ValueError((name, kernel.key, "stage arguments differ"))
        function_name = "_stage_" + str(index)
        namespace[function_name] = function
        annotations = inspect.get_annotations(kernel.func, eval_str=True)
        actuals = [ast.Name(id="world_index", ctx=ast.Load())]
        for arg in original_args:
            target = mapping[arg.arg]
            if not target.isidentifier():
                raise ValueError("simple fused argument required")
            actual_type = annotations[arg.arg]
            if target in args and not wp.types.types_equal(argument_types[target], actual_type):
                raise TypeError((target, "stage argument types differ"))
            if target not in args:
                # Struct names such as Model/Data are local to their source
                # modules. Bind their actual Warp types explicitly rather than
                # assuming the spelling is available in the fused namespace.
                type_name = "_type_%d_%s" % (index, arg.arg)
                namespace[type_name] = actual_type
                argument_types[target] = actual_type
                args[target] = ast.arg(arg=target, annotation=ast.Name(id=type_name, ctx=ast.Load()))
            actuals.append(ast.Name(id=target, ctx=ast.Load()))
        calls.append(ast.Expr(value=ast.Call(func=ast.Name(id=function_name, ctx=ast.Load()),
                                           args=actuals, keywords=[])))
        originals.append(hashlib.sha256(raw.encode()).hexdigest())
    fn = ast.FunctionDef(name=name, args=ast.arguments(posonlyargs=[], args=list(args.values()),
        vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
        body=[ast.parse("world_index = wp.tid()").body[0], *calls],
        decorator_list=[ast.parse("wp.kernel(enable_backward=False)", mode="eval").body])
    ast.fix_missing_locations(fn)
    owners = "|".join(k.func.__module__ + ":" + k.key for k, _ in stages)
    binding = hashlib.sha256(owners.encode()).hexdigest()[:16]
    module = _load(name + "_" + binding, ast.unparse(fn) + "\n", namespace)
    kernel = getattr(module, name)
    kernel.fusion_argument_names = tuple(args)
    kernel.fusion_stage_sha256 = tuple(originals)
    kernel.fusion_source = module.generated_source
    _CACHE[key] = kernel
    return kernel


def identity_mapping(kernel, **renames):
    return {name: renames.get(name, name) for name in inspect.signature(kernel.func).parameters}
