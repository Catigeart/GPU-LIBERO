"""Cold S position fusion; contact and parallel reference barriers remain.

Each combined stage owns an entire world's rows. The original arithmetic,
validation, prefix writes and stage-local returns are retained by compose.
Works with both the original workspace and native-dimension generated types.
"""
import libero.gpu.precision as scalar
import ast
import copy
import hashlib
import inspect
import textwrap

from .compose import compose, _load

_TYPES = {}


def _tree(function):
    return ast.parse(textwrap.dedent(inspect.getsource(function)))


def _launches(function):
    return [n.value for n in _tree(function).body[0].body
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
            and ast.unparse(n.value.func) == 'wp.launch']


class _SegmentOperands(ast.NodeTransformer):
    def __init__(self, segment):
        self.segment = segment

    def visit_Name(self, node):
        replacements = {'self': 'self.' + self.segment,
                        's': 'self.' + self.segment + '.static'}
        if self.segment == 's03':
            replacements.update(timestep='scalar.wp_real(self.s03.timestep)',
                                impratio='scalar.wp_real(self.s03.impratio)',
                                refsafe='int(self.s03.refsafe)')
        if node.id in replacements:
            return ast.parse(replacements[node.id], mode='eval').body
        return node


def _group(name, entries):
    stages, operands = [], []
    for index, (call, globals_, segment) in enumerate(entries):
        kernel = eval(compile(ast.Expression(call.args[0]), '<S kernel>', 'eval'), globals_)
        parameters = tuple(inspect.signature(kernel.func).parameters)
        inputs = next(k.value for k in call.keywords if k.arg == 'inputs')
        if len(parameters) != len(inputs.elts):
            raise ValueError('S original positional argument count changed')
        mapping = {arg: 'stage_%d_%s' % (index, arg) for arg in parameters}
        stages.append((kernel, mapping))
        for expression in inputs.elts:
            expression = copy.deepcopy(expression)
            if segment:
                expression = _SegmentOperands(segment).visit(expression)
            operands.append(ast.unparse(ast.fix_missing_locations(expression)))
    return compose(name, stages), operands


def fused_workspace_type(original_type):
    """Return a cold subclass; instantiate this before sealing S ownership."""
    if original_type in _TYPES:
        return _TYPES[original_type]
    position = original_type.launch_position
    launches = _launches(position)
    expected = ('_position_gate', '_reset_stage_status', '_contact_incoming',
                '_gate_active_status', '_assemble', '_prediag',
                '_merge_stage_status', '_commit_position')
    if tuple(ast.unparse(c.args[0]).split('.')[-1] for c in launches) != expected:
        raise ValueError('S source launch schedule changed')

    def cold_initialize(self, *args, **kwargs):
        original_type.__init__(self, *args, **kwargs)
        scalar = self.s02.launch.__func__
        impedance = self.s03.launch.__func__
        if len(_launches(scalar)) != 1 or len(_launches(impedance)) != 1:
            raise ValueError('S segment launch schedule changed')
        entries = [(c, position.__globals__, None) for c in launches]
        before = [entries[0], entries[1],
                  (_launches(scalar)[0], scalar.__globals__, 's02'),
                  entries[2], entries[3]]
        after = [entries[4], entries[5],
                 (_launches(impedance)[0], impedance.__globals__, 's03'),
                 entries[6], entries[7]]
        self._fusion_s_before, before = _group('s_rows_before', before)
        self._fusion_s_after, after = _group('s_rows_after', after)
        metadata = _tree(impedance).body[0].body
        metadata = [copy.deepcopy(n) for n in metadata
                    if isinstance(n, ast.Assign) and any(
                        ast.unparse(t) == 'self.last_launch_scalars' for t in n.targets)]
        if len(metadata) != 1:
            raise ValueError('S exact by-value metadata snapshot changed')
        metadata = _SegmentOperands('s03').visit(metadata[0])
        source = 'def run(self, stream=None):\n    N, R, C = self.nworld, self.rows, self.contacts\n'
        source += '    wp.launch(self._fusion_s_before, dim=N, inputs=[' + ', '.join(before) + '], device=self.device, stream=stream)\n'
        source += '    self.s01.launch(stream=stream)\n'
        source += textwrap.indent(ast.unparse(ast.fix_missing_locations(metadata)), '    ') + '\n'
        source += '    wp.launch(self._fusion_s_after, dim=N, inputs=[' + ', '.join(after) + '], device=self.device, stream=stream)\n'
        namespace_id = hashlib.sha256(original_type.__module__.encode()).hexdigest()[:16]
        module = _load('s_rows_launcher_' + namespace_id, source, dict(position.__globals__))
        self._fusion_s_run = module.run

    def launch_position(self, stream=None):
        self._fusion_s_run(self, stream=stream)

    result = type('Fused' + original_type.__name__, (original_type,),
                  {'__init__': cold_initialize, 'launch_position': launch_position,
                   '__module__': __name__})
    _TYPES[original_type] = result
    return result
