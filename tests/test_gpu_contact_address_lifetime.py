"""Contact-generation bridge regressions with no Warp/GPU runtime import.

The CPU oracle follows MuJoCo 2.3.7 engine_collision_driver.c:mj_collideGeoms
(new efc_address=-1), engine_core_constraint.c:mj_instantiateContact, and
engine_support.c:mj_contactForce (negative address returns zero force).
These tests execute the actual bridge kernel body as Python; they do not run
MuJoCo contactForce, a physics step, GPU code or a compiler.
"""
import ast
import itertools
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
STEP = ROOT / 'libero/libero/gpu/host/runtime/step'
SOURCE = STEP / 'post_contact_address.py'


class Array:
    _pointers = itertools.count(100)

    def __init__(self, shape, value, *, dtype='int32', device='cuda:0'):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self.ptr = next(self._pointers)
        self.is_contiguous = True
        self.strides = tuple(4 * self._product(self.shape[i + 1:])
                             for i in range(len(self.shape)))
        self.data = [value] * self._product(self.shape)

    @staticmethod
    def _product(values):
        result = 1
        for value in values:
            result *= value
        return result

    def _index(self, index):
        if isinstance(index, tuple):
            return index[0] * self.shape[1] + index[1]
        return index

    def __getitem__(self, index):
        return self.data[self._index(index)]

    def __setitem__(self, index, value):
        self.data[self._index(index)] = value


class PythonKernelRuntime:
    int32 = 'int32'

    def full(self, shape, value, **options):
        return Array(shape, value, **options)

    def tid(self):
        return self.index

    def launch(self, kernel, *, dim, inputs, device, stream):
        for self.index in itertools.product(*(range(n) for n in dim)):
            kernel(*inputs)


def load_bridge():
    nodes = []
    for node in ast.parse(SOURCE.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if isinstance(node, ast.FunctionDef):
                node.decorator_list = []
                node.returns = None
                for arg in node.args.args:
                    arg.annotation = None
            nodes.append(node)
    runtime = PythonKernelRuntime()
    namespace = {'wp': runtime}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace['PostContactAddressBridge']


def cpu_current_address(exclude, assigned_address):
    # CPU collision initializes every new contact; only included constraints
    # replace this sentinel. No-DOF contacts also leave/reset the sentinel.
    return assigned_address if exclude == 0 else -1


def cpu_elliptic_force(address, force, dim=3):
    if address < 0:
        return [0.0] * 6
    return list(force[address:address + dim]) + [0.0] * (6 - dim)


class ContactAddressLifetimeTests(unittest.TestCase):
    def prepared(self, worlds=2, contacts=4):
        source = SimpleNamespace(nworld=worlds, contacts=contacts,
            status=Array((worlds,), 0), committed=Array((worlds,), 1),
            ncon=Array((worlds,), contacts),
            contact_exclude=Array((worlds, contacts), 0),
            contact_address=Array((worlds, contacts), -1))
        active = Array((worlds,), 1)
        stream = SimpleNamespace(device=active.device)
        bridge = load_bridge()(source, active=active, stream=stream)
        return source, active, bridge

    def test_included_to_excluded_drops_stale_force_for_each_exclusion(self):
        force = [float(i + 1) for i in range(18)]
        for exclusion in (1, 2, 3):
            with self.subTest(exclusion=exclusion):
                source, _, bridge = self.prepared(contacts=1)
                source.contact_address[0, 0] = 9
                bridge.enqueue()
                self.assertEqual(bridge.current[0, 0], 9)
                # A new excluded contact reuses the slot. Standalone S holds9.
                source.contact_exclude[0, 0] = exclusion
                bridge.enqueue()
                expected = cpu_current_address(exclusion, assigned_address=9)
                self.assertEqual(bridge.current[0, 0], expected)
                self.assertEqual(source.contact_address[0, 0], 9)
                self.assertEqual(cpu_elliptic_force(bridge.current[0, 0], force), [0.0] * 6)
                self.assertNotEqual(cpu_elliptic_force(source.contact_address[0, 0], force), [0.0] * 6)

    def test_excluded_contact_cannot_raise_stale_address_range_error(self):
        source, _, bridge = self.prepared(contacts=1)
        source.contact_address[0, 0] = 9
        source.contact_exclude[0, 0] = 1
        bridge.enqueue()
        current_nefc, dim = 9, 3
        self.assertGreater(source.contact_address[0, 0], current_nefc - dim)
        # Post skips negative addresses before applying this range predicate.
        self.assertLess(bridge.current[0, 0], 0)

    def test_final_s_exclusion_and_reordered_contact_use_current_generation(self):
        source, _, bridge = self.prepared(contacts=1)
        source.contact_address[0, 0] = 3
        bridge.enqueue()
        # C may have included this contact; S's final no-DOF exclusion wins.
        source.contact_exclude[0, 0] = 3
        bridge.enqueue()
        self.assertEqual(bridge.current[0, 0], -1)
        # A later contact at the same slot/body identity gets its new address.
        source.contact_exclude[0, 0] = 0
        source.contact_address[0, 0] = 12
        bridge.enqueue()
        self.assertEqual(bridge.current[0, 0], 12)

    def test_masks_failed_commits_and_invalid_counts_hold_private_output(self):
        cases = [('active', 0), ('active', 2), ('status', 4),
                 ('committed', 0), ('ncon', -1), ('ncon', 5)]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                source, active, bridge = self.prepared()
                source.contact_address.data[:] = list(range(8))
                bridge.enqueue()
                before = list(bridge.current.data)
                source.contact_exclude.data[:] = [3] * 8
                (active if field == 'active' else getattr(source, field))[0] = value
                bridge.enqueue()
                self.assertEqual(bridge.current.data[:4], before[:4])
                self.assertEqual(bridge.current.data[4:], [-1] * 4)
                self.assertEqual(source.contact_address.data, list(range(8)))

    def test_current_prefix_growth_shrink_and_world_independence(self):
        for worlds in (1, 2, 8, 64):
            with self.subTest(worlds=worlds):
                source, _, bridge = self.prepared(worlds)
                self.assertEqual(bridge.current.data, [-1] * (worlds * 4))
                for w in range(worlds):
                    for c in range(4):
                        source.contact_address[w, c] = 4 * w + c
                    source.ncon[w] = 2
                    source.contact_exclude[w, 1] = 1
                bridge.enqueue()
                for w in range(worlds):
                    self.assertEqual([bridge.current[w, c] for c in range(4)], [4 * w, -1, -1, -1])
                    source.ncon[w] = 4
                bridge.enqueue()
                before = list(bridge.current.data)
                for w in range(worlds):
                    source.ncon[w] = 0
                    source.contact_exclude[w, 3] = 3
                bridge.enqueue()
                self.assertEqual(bridge.current.data, before)  # Post reads only ncon.

    def test_retained_source_and_output_owner_guards(self):
        source, _, bridge = self.prepared()
        source.contact_exclude = Array((2, 4), 0)
        with self.assertRaisesRegex(ValueError, 'source owner changed'):
            bridge.enqueue()
        source, _, bridge = self.prepared()
        bridge.current = Array((2, 4), -1)
        with self.assertRaisesRegex(ValueError, 'retained owner changed'):
            bridge.enqueue()
        with self.assertRaisesRegex(ValueError, 'foreign.*stream'):
            bridge.validate(SimpleNamespace(device='cuda:0'))
        source, _, bridge = self.prepared()
        bridge.stream = SimpleNamespace(device='cuda:0')
        with self.assertRaisesRegex(ValueError, 'retained owner changed'):
            bridge.enqueue()

    def test_normalize_precedes_dynamic_post_snapshot(self):
        path = SOURCE
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'capture_post_with_current_address')
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        events = []
        bridge = SimpleNamespace(enqueue=lambda **kw: events.append(('normalize', kw['stream'])))
        post = SimpleNamespace(capture_before_constraint=lambda **kw: events.append(('old', kw['stream'])))
        post.capture_before_constraint = lambda **kw: events.append(('new snapshot', kw['stream']))
        namespace['capture_post_with_current_address'](bridge, post, 'stream')
        self.assertEqual(events, [('normalize', 'stream'), ('new snapshot', 'stream')])

    def test_shape_and_generic_preparation_bind_the_same_current_owner_and_hook(self):
        hook = next(n for n in ast.parse(SOURCE.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'capture_post_with_current_address')
        for filename in ('shape_prepare.py', 'generic_prepare.py'):
            with self.subTest(route=filename):
                tree = ast.parse((STEP / filename).read_text())
                assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)]
                sources = next(n.value for n in assignments
                               if any(isinstance(t, ast.Name) and t.id == 'sources' for t in n.targets))
                address = next(k.value for k in sources.keywords if k.arg == 'contact_efc_address')
                callback = next(n for n in assignments
                                if any(isinstance(t, ast.Attribute) and t.attr == 'before_constraint'
                                       for t in n.targets))
                events = []
                current = object()
                bridge = SimpleNamespace(current=current,
                    enqueue=lambda **kw: events.append(('normalize', kw['stream'])))
                post = SimpleNamespace(capture_before_constraint=lambda **kw:
                    events.append(('snapshot', kw['stream'])))
                physics = SimpleNamespace()
                namespace = dict(address_bridge=bridge, post=post, physics=physics)
                exec(compile(ast.Module(body=[hook, callback], type_ignores=[]), filename, 'exec'), namespace)
                self.assertIs(eval(compile(ast.Expression(address), filename, 'eval'), namespace), current)
                physics.before_constraint('stream')
                self.assertEqual(events, [('normalize', 'stream'), ('snapshot', 'stream')])


if __name__ == '__main__':
    unittest.main()
