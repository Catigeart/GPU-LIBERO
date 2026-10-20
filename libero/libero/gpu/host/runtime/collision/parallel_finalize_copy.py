"""Opt-in candidate: original C call-to-finalize copies in disjoint GPU rows.

The original chain module is not mutated. A source-derived subclass substitutes
one launch expression with begin/copy/commit launches on the original stream.
This changes scheduling only; the original full-capacity write set is retained.
"""
import libero.gpu.precision as scalar
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
import ast
import hashlib
import inspect
from pathlib import Path
import textwrap
import warp as wp

from . import chain as original

EXPECTED_CHAIN_SHA256 = '935072553c89d29e83e593e57b5e59a58f92d828b120730b5c45ada93a276f0e'


@wp.kernel(enable_backward=False)
def _begin(c_count: wp.array(dtype=wp.int32),
           det_status: wp.array(dtype=wp.int32),
           f_active: wp.array(dtype=wp.int32),
           f_call_count: wp.array(dtype=wp.int32),
           f_status: wp.array(dtype=wp.int32),
           p_counts: wp.array(dtype=wp.int32),
           active: wp.array(dtype=wp.int32), pairs: int, max_raw: int):
    w = wp.tid()
    f_active[w] = active[w]
    f_call_count[w] = 0
    f_status[w] = 0
    p_counts[w] = 0
    if active[w] == 0:
        return
    nc = c_count[w]
    if nc < 0:
        f_status[w] = 1
        return
    # Preserve original check order and first failing detector exactly.
    for j in range(nc):
        if det_status[w * pairs + j] != 0:
            f_active[w] = 0
            f_status[w] = 1
            return
    p_counts[w] = pairs * max_raw


@wp.kernel(enable_backward=False)
def _rows(c_g1: wp.array(dtype=wp.int32), c_g2: wp.array(dtype=wp.int32),
          c_t1: wp.array(dtype=wp.int32), c_t2: wp.array(dtype=wp.int32),
          c_ipair: wp.array(dtype=wp.int32), c_user: wp.array(dtype=wp.int32),
          c_segment: wp.array(dtype=wp.int32), c_sort: wp.array(dtype=wp.int32),
          c_margin: wp.array(dtype=scalar.wp_real),
          det_raw: wp.array(dtype=scalar.wp_real), det_raw_count: wp.array(dtype=wp.int32),
          f_raw: wp.array(dtype=scalar.wp_real), f_raw_count: wp.array(dtype=wp.int32),
          f_types: wp.array(dtype=wp.int32), f_segment: wp.array(dtype=wp.int32),
          f_sort: wp.array(dtype=wp.int32),
          p_route: wp.array(dtype=wp.int32), p_usermargin: wp.array(dtype=scalar.wp_real),
          p_counts: wp.array(dtype=wp.int32), f_status: wp.array(dtype=wp.int32),
          f_active: wp.array(dtype=wp.int32), pairs: int, max_raw: int, raw_f: int):
    row = wp.tid()
    rows_per_world = pairs * max_raw
    w = row // rows_per_world
    if f_active[w] == 0 or f_status[w] != 0 or p_counts[w] == 0:
        return
    slot = row // max_raw
    r = row % max_raw
    # src and dst are identical indices, including the accepted borrowed alias.
    for k in range(raw_f):
        f_raw[row * raw_f + k] = det_raw[row * raw_f + k]
    p_route[row * 4] = c_g1[slot]
    p_route[row * 4 + 1] = c_g2[slot]
    p_route[row * 4 + 2] = c_ipair[slot]
    p_route[row * 4 + 3] = c_user[slot]
    p_usermargin[row] = c_margin[slot]
    if r == 0:
        f_raw_count[slot] = det_raw_count[slot]
        f_types[slot * 2] = c_t1[slot]
        f_types[slot * 2 + 1] = c_t2[slot]
        f_segment[slot] = c_segment[slot]
        f_sort[slot] = c_sort[slot]


@wp.kernel(enable_backward=False)
def _commit(c_count: wp.array(dtype=wp.int32), f_call_count: wp.array(dtype=wp.int32),
            f_active: wp.array(dtype=wp.int32), f_status: wp.array(dtype=wp.int32),
            p_counts: wp.array(dtype=wp.int32)):
    w = wp.tid()
    if f_active[w] != 0 and f_status[w] == 0 and p_counts[w] != 0:
        f_call_count[w] = c_count[w]


def parallel_copy_launch(workspace, stream):
    w = workspace
    n, pairs, max_raw, raw_f = w.nworld, w.call_capacity, 50, 10
    wp.launch(_begin, dim=n, inputs=[w.detect.call_count, w.detect.det_status,
        w.finalize.active, w.finalize.call_count, w.finalize.status,
        w.finalize.params.counts, w.effective_active, pairs, max_raw],
        device=w.device, stream=stream)
    wp.launch(_rows, dim=n * pairs * max_raw, inputs=[
        w.detect.call_g1, w.detect.call_g2, w.detect.call_type1, w.detect.call_type2,
        w.call_ipair, w.call_user, w.call_segment, w.call_sort, w.detect.call_margin,
        w.detect.raw, w.detect.raw_count, w.finalize.raw, w.finalize.raw_count,
        w.finalize.types, w.finalize.segment, w.finalize.sort_enabled,
        w.finalize.params.route, w.finalize.params.usermargin, w.finalize.params.counts,
        w.finalize.status, w.finalize.active, pairs, max_raw, raw_f],
        device=w.device, stream=stream)
    wp.launch(_commit, dim=n, inputs=[w.detect.call_count, w.finalize.call_count,
        w.finalize.active, w.finalize.status, w.finalize.params.counts],
        device=w.device, stream=stream)


def source_derived_workspace_class():
    """Build an isolated class; require the exact original chain source epoch."""
    source_path = Path(original.__file__).resolve()
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if source_sha != EXPECTED_CHAIN_SHA256:
        raise RuntimeError('parallel-copy original chain source epoch changed')
    original_method = textwrap.dedent(inspect.getsource(original.CollisionChainWorkspace.launch))
    tree = ast.parse(original_method)
    class Substitute(ast.NodeTransformer):
        replacements = 0
        def visit_Expr(self, node):
            value = node.value
            if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                    and isinstance(value.func.value, ast.Name) and value.func.value.id == 'wp'
                    and value.func.attr == 'launch' and value.args
                    and isinstance(value.args[0], ast.Name)
                    and value.args[0].id == '_calls_to_finalize_kernel'):
                self.replacements += 1
                return ast.copy_location(ast.Expr(value=ast.Call(
                    func=ast.Name(id='_parallel_copy_launch', ctx=ast.Load()),
                    args=[ast.Name(id='self', ctx=ast.Load()), ast.Name(id='stream', ctx=ast.Load())],
                    keywords=[])), node)
            return self.generic_visit(node)
    substitute = Substitute()
    tree = ast.fix_missing_locations(substitute.visit(tree))
    if substitute.replacements != 1:
        raise RuntimeError('parallel-copy substitution must match exactly one original launch')
    generated = ast.unparse(tree) + '\n'
    namespace = dict(vars(original))
    namespace['__name__'] = __name__
    namespace['__file__'] = str((GPU_ROOT / 'host/runtime/collision/parallel_finalize_copy.py'))
    namespace['_parallel_copy_launch'] = parallel_copy_launch
    exec(compile(tree, str((GPU_ROOT / 'host/runtime/collision/parallel_finalize_copy.py')) + ':source-derived-launch', 'exec'), namespace)
    candidate_class = type('ParallelCopyCollisionChainWorkspace', (original.CollisionChainWorkspace,),
                           {'launch': namespace['launch'], '__module__': __name__})
    candidate_class.candidate_source_manifest = {
        'schema': 'collision-parallel-finalize-copy-source-candidate-v1',
        'original_chain_path': str(source_path), 'original_chain_sha256': source_sha,
        'original_method_sha256': hashlib.sha256(original_method.encode()).hexdigest(),
        'candidate_module_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'generated_method_sha256': hashlib.sha256(generated.encode()).hexdigest(),
        'generated_method_source': generated,
        'substitutions': 1, 'raw_f': 10, 'max_raw': 50,
        'scope': 'source candidate only; endpoints/writes/performance await actual independent qualification'}
    return candidate_class