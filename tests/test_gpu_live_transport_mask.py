"""Exercise cold C/S stage selection and the existing admission reference on CPU.

Only selected function ASTs are loaded: importing Warp, allocating a GPU, and
compiling generated CUDA are deliberately unnecessary for this regression.
"""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch


GPU = Path(__file__).resolve().parents[1] / 'libero/libero/gpu'
COLLISION = GPU / 'host/bindings/collision.py'
WIDTHS = {'dist': 1, 'pos': 3, 'frame': 9, 'margin': 1, 'friction': 5,
          'solref': 2, 'solreffriction': 2, 'solimp': 5,
          'dim': 1, 'geom1': 1, 'geom2': 1, 'exclude': 1}


class Array:
    def __init__(self, values, dtype='i32'):
        self.values = list(values)
        self.dtype = dtype
        self.ptr = id(self)


def load_functions(path, names, namespace, *, kernels=False):
    nodes = [copy.deepcopy(n) for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise AssertionError('missing source functions: ' + str(names))
    if kernels:
        for node in nodes:
            node.decorator_list = []
            node.returns = None
            for arg in node.args.args:
                arg.annotation = None
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


class LiveTransportMaskTests(unittest.TestCase):
    def setUp(self):
        self.world = 0
        self.wp = SimpleNamespace(array=Array, int32='i32', uint32='u32', int64='i64',
                                  float32='f32', float64='f64', tid=lambda: self.world)
        self.namespace = {'wp': self.wp, 'np': SimpleNamespace(floating=float)}
        load_functions(COLLISION, {'_dataflow_stages', '_guard_inactive'}, self.namespace)
        load_functions(GPU / 'host/runtime/collision/chain.py',
                       {'_effective_active_kernel'}, self.namespace, kernels=True)
        load_functions(GPU / 'host/runtime/step/contact_transport.py',
                       {'_extract'}, self.namespace, kernels=True)

    def plan(self, worlds, *, profile='fast', diagnostics=False):
        self.nworld = worlds
        source_capacity, target_capacity = 4, 3
        a = lambda values, dtype='i32': Array(values, dtype)
        self.ws = SimpleNamespace(_cuda_collision_dataflow_profile=profile,
            _cuda_collision_diagnostics=diagnostics, before_outputs={},
            active=a([1] * worlds), fe_code=a([0] * worlds),
            effective_active=a([1] * worlds), contact_capacity=source_capacity,
            finalize=SimpleNamespace(ncon=a([2] * worlds),
                contact_f=a(range(worlds * source_capacity * 65), 'f64'),
                contact_i=a(range(worlds * source_capacity * 5))))
        self.d = SimpleNamespace(active=a([1] * worlds),
            valid={'position': a([1] * worlds)}, first_error=a([0] * worlds))
        self.transport = SimpleNamespace(collision=SimpleNamespace(workspace=self.ws),
            dynamics=self.d, inputs={name: a([-100 - i] * (worlds * target_capacity * width),
                'i32' if name in ('dim', 'geom1', 'geom2', 'exclude') else 'f64')
                for i, (name, width) in enumerate(WIDTHS.items())},
            active=a([0] * worlds), ncon=a([-7] * worlds),
            source_capacity=source_capacity, capacity=target_capacity)
        self.storage = {'contact_f': a([999] * (worlds * source_capacity * 28), 'f64'),
                        'contact_i': a([999] * (worlds * source_capacity * 5))}
        original = [
            ('collision_cuda::cf_effective',
             [self.ws.active, self.ws.fe_code, self.ws.effective_active], 'lane0'),
            ('collision_cuda::cf_calls_finalize', [], 'block'),
            None,
        ]
        self.namespace['_workspace_stages'] = lambda ws: original
        self.namespace['_dataflow_storage'] = lambda ws: self.storage
        metadata = SimpleNamespace(prepare_collision_metadata=lambda ws: {})
        with patch.dict(sys.modules, {'libero.gpu.host.build.collision_metadata': metadata}):
            self.stages = self.namespace['_dataflow_stages'](self.ws, self.transport)
        self.mapped, self.wrapper_source = self.namespace['_guard_inactive'](self.ws, self.stages)

    def invoke_admission(self, requested):
        """Mock plan dispatch; use real Python kernels for effective/S masks.

        Contact math/commit is not emulated. The source public C slabs already
        contain a successful prefix, as before a masked refresh or reset.
        """
        self.ws.active.values[:] = requested
        self.d.active.values[:] = requested
        tail = self.mapped[-1]
        for w, mask in enumerate(requested):
            self.world = w
            self.namespace['_effective_active_kernel'](self.ws.active.values,
                self.ws.fe_code.values, self.ws.effective_active.values)
            # _guard_inactive wrappers return on requested0. A direct tail
            # must still execute the established reference's admission clear.
            if tail is not self.stages[-1] and mask == 0:
                continue
            self.namespace['_extract'](self.d.active.values,
                self.d.valid['position'].values, self.d.first_error.values,
                self.ws.fe_code.values, self.ws.effective_active.values,
                self.ws.finalize.ncon.values, self.ws.finalize.contact_f.values,
                self.ws.finalize.contact_i.values, self.transport.active.values,
                self.transport.ncon.values,
                *[self.transport.inputs[name].values for name in WIDTHS],
                self.transport.source_capacity, self.transport.capacity)

    def held_state(self):
        return copy.deepcopy((self.transport.ncon.values,
            {name: array.values for name, array in self.transport.inputs.items()},
            self.ws.finalize.contact_f.values, self.ws.finalize.contact_i.values,
            self.ws.fe_code.values, self.d.first_error.values))

    def test_selected_commit_tail_keeps_original_owners_and_unconditional_dispatch(self):
        for profile, diagnostics, name in (
                ('fast', False, 'cf_commit_contacts_transport_live'),
                ('fast', True, 'cf_commit_contacts_transport'),
                ('compatible', False, 'cf_commit_contacts_transport')):
            with self.subTest(profile=profile, diagnostics=diagnostics):
                self.plan(2, profile=profile, diagnostics=diagnostics)
                tail = self.stages[-1]
                self.assertEqual(tail[0], 'collision_cuda::' + name)
                self.assertIs(self.mapped[-1], tail)
                self.assertIs(self.mapped[0], self.stages[0])
                self.assertEqual(tail[2], 'block')
                expected = [self.d.active, self.d.valid['position'], self.d.first_error,
                    self.ws.fe_code, self.ws.effective_active, self.ws.finalize.ncon,
                    self.storage['contact_f'], self.storage['contact_i'],
                    self.ws.finalize.contact_f, self.ws.finalize.contact_i,
                    self.transport.active, self.transport.ncon,
                    *self.transport.inputs.values(), 4, 3]
                self.assertEqual(tail[1], expected)
                if profile == 'fast':
                    self.assertIsNot(self.mapped[1], self.stages[1])
                    self.assertIn('if(requested[w]==0)return;', self.wrapper_source)
                    self.assertNotIn(name + '(w', self.wrapper_source)
                else:
                    self.assertIs(self.mapped, self.stages)
                    self.assertEqual(self.wrapper_source, '')

    def test_controller_refresh_skip_clears_previous_success_and_holds_leaves(self):
        for worlds in (1, 2, 8, 64):
            with self.subTest(worlds=worlds):
                self.plan(worlds)
                self.invoke_admission([1] * worlds)
                self.assertEqual(self.transport.active.values, [1] * worlds)
                held = self.held_state()
                self.invoke_admission([0] * worlds)
                self.assertEqual(self.transport.active.values, [0] * worlds)
                self.assertEqual(self.held_state(), held)

    def test_selected_reset_clears_only_unselected_admission_with_held_contact_bytes(self):
        for worlds in (2, 8, 64):
            with self.subTest(worlds=worlds):
                self.plan(worlds)
                self.invoke_admission([1] * worlds)
                held = self.held_state()
                selected = [int(w % 2 == 0) for w in range(worlds)]
                self.invoke_admission(selected)
                self.assertEqual(self.transport.active.values, selected)
                self.assertEqual(self.held_state(), held)

    def test_requested_world_source_refusal_clears_admission_without_replacing_held_fields(self):
        for reason in ('c_error', 'd_error', 'position_invalid', 'negative_ncon',
                       'source_capacity', 'target_capacity'):
            with self.subTest(reason=reason):
                self.plan(1)
                self.invoke_admission([1])
                if reason == 'c_error':
                    self.ws.fe_code.values[0] = 3
                elif reason == 'd_error':
                    self.d.first_error.values[0] = 19
                elif reason == 'position_invalid':
                    self.d.valid['position'].values[0] = 0
                else:
                    self.ws.finalize.ncon.values[0] = {
                        'negative_ncon': -1, 'source_capacity': 5, 'target_capacity': 4}[reason]
                held = self.held_state()
                self.invoke_admission([1])
                self.assertEqual(self.transport.active.values, [0])
                self.assertEqual(self.held_state(), held)

    def test_successful_empty_prefix_remains_admitted(self):
        self.plan(2)
        self.ws.finalize.ncon.values[:] = [0, 0]
        self.invoke_admission([1, 1])
        self.assertEqual(self.transport.active.values, [1, 1])
        self.assertEqual(self.transport.ncon.values, [0, 0])
        self.invoke_admission([0, 1])
        self.assertEqual(self.transport.active.values, [0, 1])
        self.assertEqual(self.transport.ncon.values, [0, 0])


if __name__ == '__main__':
    unittest.main()
