"""Forward admission/projection fusion, preserving singleton protocol views.

No cross-world gate is flattened: each SharedProtocol(1) still runs at tid=0.
Backend begin remains its qualified method, with its host validations intact.
"""
import ast
import inspect
import textwrap
import warp as wp
from libero.gpu.host.runtime.foundation import device as foundation_device
from libero.gpu.host.runtime.foundation.runtime_adapter import FoundationForwardAdapter, _copy_admission, _copy_event
from libero.gpu.host.runtime.step import physics, shape_projection
from libero.gpu.host.runtime.step.shape_physics import ShapePhysicsForward
from libero.gpu.host.runtime.dynamics import runtime_adapter as dynamics_runtime
from libero.gpu.host.assembly.compose.backup import FusedShapePhysicsForward as BackupForward
from libero.gpu.host.assembly.compose.compose import compose, identity_mapping

@wp.kernel(enable_backward=False)
def _batch_admission(source: wp.array(dtype=wp.int32), target: wp.array(dtype=wp.int32), world: int):
    lane = wp.tid()
    target[world + lane] = source[lane]

@wp.kernel(enable_backward=False)
def _batch_event(source: wp.array(dtype=wp.int64), target: wp.array(dtype=wp.int64), world: int):
    lane = wp.tid()
    target[world + lane] = source[lane]

_fields = ('errors', 'first_stage', 'first_local', 'first_item', 'first_event')
_stages = [(foundation_device._gate, identity_mapping(foundation_device._gate))]
_stages.append((_batch_admission, dict(source='active', target='out_admitted', world='out_world')))
for _name in _fields:
    _kernel = _batch_event if _name == 'first_event' else _batch_admission
    _stages.append((_kernel, dict(source=_name, target='out_' + _name, world='out_world')))
gate_and_copy = compose('forward_gate_and_copy', _stages)
begin_and_plan = compose('forward_begin_and_plan', [
    (dynamics_runtime._begin, identity_mapping(dynamics_runtime._begin, error='first_error')),
    (physics._increment_plan, identity_mapping(physics._increment_plan, first='first_error', expected='expected_plan')),
])


def _backend_begin_with_plan(self, mask, expected_plan, stream):
    # This is the unchanged current RuntimeDynamicsAdapter.begin_forward prefix.
    if self.begin_forward.__func__ is not dynamics_runtime.LiveDynamicsAdapter.begin_forward:
        raise ValueError('fused begin requires unchanged qualified dynamics begin method')
    self.validate(stream)
    dynamics_runtime._array(mask, (self.capacity,), wp.int32, self.device)
    dynamics_runtime._array(expected_plan, (self.capacity,), wp.int64, self.device)
    wp.copy(self.requested_mask, mask, stream=self.stream)
    inputs = [mask, self.first_error, self.state_epoch, self.source_state_epoch,
              self.forward_epoch, self.active,
              *[self.valid[k] for k in ('position', 'velocity', 'bias', 'smooth')], expected_plan]
    self._launch(begin_and_plan, inputs)


def _gate_copy(self, w, gate, count, actual, expected, valid):
    # Same validations as SharedProtocol.gate, before the original launch point.
    gate._stage(101)
    gate._state()
    gate._require([('admission', self._mask_views[w], 'int32'), ('count', count, 'int32'),
                   ('producer_epoch', actual, 'int64'), ('expected_epoch', expected, 'int64'),
                   ('producer_valid', valid, 'int32'), ('event', actual, 'int64')])
    args = dict(admission=self._mask_views[w], count=count, producer_epoch=actual,
                expected_epoch=expected, producer_valid=valid, event=actual,
                active=gate.active, valid=gate.valid, capacity=gate.capacity, stage=101,
                out_admitted=self.admitted, out_world=w)
    args.update({name: getattr(gate, name) for name in _fields})
    args.update({'out_' + name: getattr(self, name) for name in _fields})
    wp.launch(gate_and_copy, dim=gate.active.size, inputs=[args[name] for name in gate_and_copy.fusion_argument_names],
              device=self.stream.device, stream=self.stream)


def _foundation_begin(with_plan=False):
    tree = ast.parse(textwrap.dedent(inspect.getsource(FoundationForwardAdapter.begin_forward)))
    class Replace(ast.NodeTransformer):
        count = 0
        def visit_For(self, node):
            if ast.unparse(node.iter) == 'enumerate(zip(self.gates, self._views))':
                self.count += 1
                node.body = ast.parse('count, actual, expected, valid = row\nself._fusion_gate_copy(w, gate, count, actual, expected, valid)').body
                return node
            return self.generic_visit(node)
    replace = Replace()
    tree = replace.visit(tree)
    if replace.count == 0:
        # New schema: a single W-wide SharedProtocol with batch copies.
        # Match the exact original gate/copy block before replacing only its
        # dispatch. All retained owner/source/epoch validations remain intact.
        fn=tree.body[0]
        candidates=[i for i,node in enumerate(fn.body)
            if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call)
            and ast.unparse(node.value.func)=='gate.gate']
        if len(candidates)!=1:
            raise ValueError('one original singleton loop or batch gate required')
        i=candidates[0]
        expected_gate='gate.gate(mask, count, actual, expected, valid, actual, stage=101)'
        expected_copy='wp.copy(self.admitted, gate.active, stream=self.stream)'
        expected_errors='for name, root, _ in self._error_roots:\n    wp.copy(root, getattr(gate, name), stream=self.stream)'
        actual=[ast.dump(n,include_attributes=False) for n in fn.body[i:i+3]]
        expected=[ast.dump(ast.parse(text).body[0],include_attributes=False)
                  for text in (expected_gate,expected_copy,expected_errors)]
        if actual != expected:
            raise ValueError('original batch gate and error-copy sequence drifted')
        fn.body[i:i+3]=ast.parse('self._fusion_gate_copy(0, gate, count, actual, expected, valid)').body
    elif replace.count != 1:
        raise ValueError('exactly one singleton foundation gate loop required')
    if with_plan:
        fn = tree.body[0]
        fn.name = 'begin_forward_with_plan'
        fn.args.kwonlyargs.append(ast.arg(arg='plan'))
        fn.args.kw_defaults.append(None)
        candidates = [i for i, node in enumerate(fn.body) if isinstance(node, ast.Expr)
                      and isinstance(node.value, ast.Call)
                      and ast.unparse(node.value.func) == 'self.backend.begin_forward']
        if len(candidates) != 1:
            raise ValueError('one original backend begin required')
        fn.body[candidates[0]] = ast.parse('_backend_begin_with_plan(self.backend, self.admitted, plan, self.stream)').body[0]
    ast.fix_missing_locations(tree)
    namespace = dict(FoundationForwardAdapter.begin_forward.__globals__)
    namespace['_backend_begin_with_plan'] = _backend_begin_with_plan
    exec(compile(tree, __file__ + ':foundation', 'exec'), namespace)
    return namespace['begin_forward_with_plan' if with_plan else 'begin_forward']


class FusedFoundationForwardAdapter(FoundationForwardAdapter):
    _fusion_gate_copy = _gate_copy
    begin_forward = _foundation_begin()
    begin_forward_with_plan = _foundation_begin(True)


project_newton = compose('forward_project_newton', [
    (physics._n_mask, identity_mapping(physics._n_mask, active='d_active', error='n_upstream')),
    (shape_projection.project_counts, identity_mapping(shape_projection.project_counts)),
    (shape_projection.project_cone_heads, identity_mapping(shape_projection.project_cone_heads)),
    (physics._increment_plan, identity_mapping(physics._increment_plan, mask='request_mask', first='first', expected='expected_newton')),
])


def _project(self):
    n, x, sw = self.n.ws, self._s_views, self.s.workspace
    args = dict(d_active=self.d.active, first=self.d.first_error, smooth=self.d.valid['smooth'],
                reference=sw.reference_ready, sstatus=sw.status, admitted=self.n_admitted,
                n_upstream=self.n_upstream, source_counts=self._counts, source_ncon=self.transport.ncon,
                newton=n.newton_world, projection_status=self.projection_status, error_item=self.projection_item,
                rows=n.rows, contacts=n.contacts, nv=self.runtime_shape.nv,
                row_type=x['type'], row_id=x['id'], contact_dim=x['dim'], contact_mu=x['mu'],
                contact_friction=x['friction'], request_mask=self.mask, expected_newton=self.expected_newton)
    args.update({name: getattr(n, name) for name in ('nefc', 'ne', 'nf', 'ncon', 'count', 'empty', 'active',
                                                  'cone_dim', 'cone_mu', 'cone_friction')})
    self._launch(project_newton, [args[name] for name in project_newton.fusion_argument_names])


def _shape_enqueue(step1=False):
    original_method = ShapePhysicsForward.enqueue_step1 if step1 else ShapePhysicsForward.enqueue
    tree = ast.parse(textwrap.dedent(inspect.getsource(original_method)))
    function = tree.body[0]
    begin_calls = [i for i, node in enumerate(function.body) if isinstance(node, ast.Assign)
                   and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'self.f.begin_forward']
    if len(begin_calls) != 1:
        raise ValueError('one original F begin required')
    i = begin_calls[0]
    if ast.unparse(function.body[i + 1].value.args[0]) != '_increment_plan':
        raise ValueError('forward plan increment must immediately follow begin')
    function.body[i:i + 2] = ast.parse('admitted = self.f.begin_forward_with_plan(self.mask, stream=self.stream, plan=self.expected_forward)').body
    if step1:
        ast.fix_missing_locations(tree)
        namespace = dict(original_method.__globals__)
        exec(compile(tree, __file__ + ':step1', 'exec'), namespace)
        return namespace['enqueue_step1']
    begin = next(i for i, node in enumerate(function.body) if isinstance(node, ast.Expr)
                 and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'self._launch'
                 and ast.unparse(node.value.args[0]) == '_n_mask')
    end = next(i for i in range(begin + 1, len(function.body)) if isinstance(function.body[i], ast.Expr)
               and isinstance(function.body[i].value, ast.Call)
               and ast.unparse(function.body[i].value.func) == 'self._launch'
               and ast.unparse(function.body[i].value.args[0]) == '_increment_plan')
    original = function.body[begin:end + 1]
    kernels = [ast.unparse(node.value.args[0]) for node in original if isinstance(node, ast.Expr)]
    if kernels != ['_n_mask', 'project_counts', 'project_cone_heads', '_increment_plan']:
        raise ValueError('original four-stage N projection sequence required')
    # Keep n/x local aliases because the final forward publication consumes n.
    function.body[begin:end + 1] = ast.parse('n, x = self.n.ws, self._s_views\nself._fusion_project_newton()').body
    copies = [i for i, node in enumerate(function.body) if isinstance(node, ast.For)
              and ast.unparse(node.iter) == 'self.entry_witness.items()']
    if len(copies) != 1:
        raise ValueError('one retained r01 backup copy loop required')
    function.body[copies[0]] = ast.parse('if self._fusion_entry_copy is not None:\n    self._fusion_entry_copy.launch(self.stream)').body[0]
    ast.fix_missing_locations(tree)
    namespace = dict(ShapePhysicsForward.enqueue.__globals__)
    exec(compile(tree, __file__ + ':shape', 'exec'), namespace)
    return namespace['enqueue']


class FusedShapePhysicsForward(BackupForward):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The native schedule recorder must see the actual entry owners before
        # any forward has run. Previously eager graph capture supplied these
        # bindings as a side effect of calling enqueue during construction.
        self.n._r03_upstream = self.n_upstream
        self.n._r03_enabled = self.mask

    _fusion_project_newton = _project
    enqueue = _shape_enqueue()
    enqueue_step1 = _shape_enqueue(True)
