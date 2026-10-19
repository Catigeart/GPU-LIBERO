"""Position history selection and byte transactions without CUDA or compilation."""
import ast
from copy import deepcopy
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch


GPU = Path(__file__).resolve().parents[1] / 'libero/libero/gpu'
SOURCE = GPU / 'host/bindings/position.py'
DTYPE_BYTES = {'f64': 8, 'i32': 4, 'i64': 8}
# The held public fields deliberately exclude body/frame temporary diagnostics.
HOLD_FIELDS = (
    ('subtree_com', 'f64'), ('cinert', 'f64'), ('cdof', 'f64'), ('crb', 'f64'),
    ('result_kind', 'i32'), ('first_code', 'i32'), ('first_stage', 'i32'),
    ('body.qpos', 'f64'), ('body.xpos', 'f64'), ('body.xquat', 'f64'),
    ('body.xmat', 'f64'), ('body.xanchor', 'f64'), ('body.xaxis', 'f64'),
    ('body.status', 'i32'), ('frame.xipos', 'f64'), ('frame.ximat', 'f64'),
    ('frame.geom_xpos', 'f64'), ('frame.geom_xmat', 'f64'),
    ('frame.site_xpos', 'f64'), ('frame.site_xmat', 'f64'), ('frame.status', 'i32'),
    ('factor.qm', 'f64'), ('factor.ld', 'f64'), ('factor.inv', 'f64'),
    ('factor.sqrt_inv', 'f64'), ('factor.status', 'i32'), ('tr.length', 'f64'),
    ('tr.moment', 'f64'), ('tr.position_status', 'i32'), ('valid', 'i32'),
    ('written', 'i64'),
)


class Array:
    def __init__(self, shape, dtype, cells=None):
        self.shape, self.dtype = shape, dtype
        self.ndim = len(shape)
        self.device, self.is_contiguous = 'cuda:0', True
        self.ptr = id(self)
        self.cells = list(cells) if cells is not None else []


def token(a):
    return id(a), a.ptr, a.shape, a.dtype, a.device, a.is_contiguous


class Hold:
    def __init__(self, worlds):
        self.fields, self.views, self.before = {}, {}, {}
        # Byte patterns include signed zero and distinct NaN payloads. No float
        # conversion may normalize those bytes while the history is copied.
        patterns = (0, 1 << 63, 0x7ff8000000000001, 0x7ff8000000000002,
                    0xffffffffffffffff, 0x123456789abcdef0)
        for index, (name, dtype) in enumerate(HOLD_FIELDS):
            width = index % 5 + 1
            size = DTYPE_BYTES[dtype]
            cells = [patterns[(index + i) % len(patterns)].to_bytes(8, 'little')[:size]
                     for i in range(worlds * width)]
            self.fields[name] = self.views[name] = Array((worlds, width), dtype, cells)
            self.before[name] = Array((worlds, width), dtype,
                                      [b'\xa5' * size] * (worlds * width))
        self.tokens = {name: (token(a), token(self.before[name]))
                       for name, a in self.views.items()}

    def validate(self, fields):
        if fields.keys() != self.fields.keys():
            raise ValueError('hold field names changed')
        for name, a in self.views.items():
            if self.tokens[name] != (token(a), token(self.before[name])):
                raise ValueError('hold retained buffer changed')

    def snapshot(self, stream):
        raise AssertionError('the unmasked snapshot must not be dispatched')

    def restore_unrequested(self, mask, stream):
        self.validate(self.fields)
        for name, a in self.views.items():
            width = a.shape[1]
            for world, selected in enumerate(mask.cells):
                if selected == 0:
                    start = world * width
                    a.cells[start:start + width] = self.before[name].cells[start:start + width]


class NativePlan:
    """Interpret only the existing cf_copy::hold_* ABI selected by the bridge."""
    def __init__(self, name, sources, stages, worlds, root, **options):
        self.name, self.sources, self.stages = name, sources, stages
        self.worlds, self.root, self.options = worlds, root, options
        self.manifest = {'name': name}
        self.payload_cells, self.calls = 0, 0

    def launch(self, stream=None):
        self.calls += 1
        for function, (mask, source, before, width), mode in self.stages:
            assert function == 'cf_copy::hold_' + source.dtype
            assert mode == 'block'
            for world in range(self.worlds):
                if mask.cells[world] != 0:
                    continue
                start = world * width
                before.cells[start:start + width] = source.cells[start:start + width]
                self.payload_cells += width


class PhysicsPlan:
    """A stateful F stand-in: inactive clears and selected failures still run."""
    def __init__(self, d):
        self.d, self.calls = d, 0
        self.manifest = {'name': 'position'}

    def launch(self, stream=None):
        self.calls += 1
        for world, selected in enumerate(self.d.requested_mask.cells):
            # These diagnostics are outside the held inventory. Masking the
            # whole F invocation would incorrectly retain their old values.
            self.d.diagnostics[world] = 0
            for index, a in enumerate(self.d._holds['position'].views.values()):
                width, size = a.shape[1], DTYPE_BYTES[a.dtype]
                start = world * width
                value = b'\0' * size
                if selected == 1:
                    value = bytes([17 + index]) * size
                a.cells[start:start + width] = [value] * width
            if selected == 1:
                self.d.diagnostics[world] = -7 if world == self.d.failed_world else 3


def load_functions():
    names = {'_create_position_snapshot_plan', 'install', 'install_dataflow'}
    nodes = [n for n in ast.parse(SOURCE.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {'GPU_ROOT': GPU, 'Path': Path, 'MethodType': MethodType,
                 'NativePlan': NativePlan, '__package__': 'libero.gpu.host.bindings'}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace


class PositionSnapshotMaskTests(unittest.TestCase):
    def setUp(self):
        self.namespace = load_functions()
        self.warp = SimpleNamespace(float64='f64', int32='i32', int64='i64')
        self.scene = SimpleNamespace(install_scene_from_bundle=lambda *a: {})
        self.modules = patch.dict(sys.modules, {
            'warp': self.warp,
            'libero.gpu.host.bindings.position_scene': self.scene,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def adapter(self, mask, failed_world=-1):
        worlds = len(mask)
        d = SimpleNamespace(capacity=worlds, device='cuda:0', stream=object(),
            requested_mask=Array((worlds,), 'i32', mask),
            _holds={'position': Hold(worlds)}, diagnostics=[91] * worlds,
            failed_world=failed_world, validate=lambda stream: None,
            position=SimpleNamespace(validate_bindings=lambda: None, factor=object()),
            transmission=SimpleNamespace(validate_bindings=lambda: None))
        return d

    def install(self, d, route):
        physics = PhysicsPlan(d)
        self.namespace.update(create_plan=lambda *a: physics,
            create_dataflow_plan=lambda *a, **k: physics, position_stages=lambda p: [])
        bundle = SimpleNamespace(dynamics=d, runtime=SimpleNamespace(forward=SimpleNamespace(c=None)))
        self.namespace[route](bundle, Path('/unused-build-root'))
        return physics, d._cuda_position_snapshot_plan

    def test_all_fields_bind_original_buffers_in_the_snapshot_direction(self):
        d = self.adapter([0, 1])
        hold, plan = self.namespace['_create_position_snapshot_plan'](d, Path('/build'))
        self.assertIs(hold, d._holds['position'])
        self.assertEqual(len(plan.stages), 31)
        self.assertEqual(plan.sources, [GPU / 'cuda/runtime/copies.cuh'])
        self.assertEqual(plan.worlds, 2)
        self.assertFalse(plan.options['stage_barriers'])
        self.assertNotIn('world_guard', plan.options)
        for (name, source), (function, args, mode) in zip(hold.views.items(), plan.stages):
            self.assertEqual(function, 'cf_copy::hold_' + source.dtype)
            self.assertEqual(args, [d.requested_mask, source, hold.before[name], source.shape[1]])
            self.assertEqual(mode, 'block')

    def test_masked_snapshot_matches_full_snapshot_then_original_restore(self):
        for worlds in (1, 2, 8, 64):
            masks = [[1] * worlds, [0] * worlds,
                     [(0, 1, -1, 2)[w % 4] for w in range(worlds)]]
            for route in ('install', 'install_dataflow'):
                for mask in masks:
                    with self.subTest(worlds=worlds, route=route, mask=mask[:8]):
                        d = self.adapter(mask, failed_world=1 if worlds > 1 else 0)
                        reference = self.adapter(mask, failed_world=d.failed_world)
                        ref_hold = reference._holds['position']
                        for name, a in ref_hold.views.items():
                            ref_hold.before[name].cells[:] = a.cells
                        PhysicsPlan(reference).launch()
                        ref_hold.restore_unrequested(reference.requested_mask, reference.stream)
                        physics, snapshot = self.install(d, route)
                        history = {name: a.cells[:] for name, a in d._holds['position'].before.items()}
                        d.launch_position()
                        self.assertEqual(physics.calls, 1)
                        self.assertEqual(d.diagnostics, reference.diagnostics)
                        for name, a in d._holds['position'].views.items():
                            self.assertEqual(a.cells, ref_hold.views[name].cells, name)
                            width = a.shape[1]
                            for world, selected in enumerate(mask):
                                if selected != 0:
                                    start = world * width
                                    self.assertEqual(d._holds['position'].before[name].cells[start:start + width],
                                                     history[name][start:start + width])
                        expected = mask.count(0) * sum(a.shape[1] for a in ref_hold.views.values())
                        self.assertEqual(snapshot.payload_cells, expected)

    def test_same_mask_pointer_reads_new_device_values_on_each_replay(self):
        d = self.adapter([1, 0])
        physics, snapshot = self.install(d, 'install_dataflow')
        mask = d.requested_mask
        d.launch_position()
        current = {name: a.cells[:] for name, a in d._holds['position'].views.items()}
        mask.cells[:] = [0, 1]
        d.launch_position()
        self.assertIs(d.requested_mask, mask)
        self.assertEqual(physics.calls, 2)
        self.assertEqual(snapshot.calls, 2)
        for name, a in d._holds['position'].views.items():
            width = a.shape[1]
            self.assertEqual(a.cells[:width], current[name][:width])
            self.assertIs(snapshot.stages[list(d._holds['position'].views).index(name)][1][0], mask)

    def test_retained_history_or_hold_replacement_is_rejected_before_f(self):
        for replacement in ('hold', 'history'):
            with self.subTest(replacement=replacement):
                d = self.adapter([0, 1])
                physics, snapshot = self.install(d, 'install_dataflow')
                if replacement == 'hold':
                    d._holds['position'] = Hold(2)
                else:
                    hold = d._holds['position']
                    hold.before['body.xpos'] = deepcopy(hold.before['body.xpos'])
                with self.assertRaisesRegex(ValueError, 'hold owner|retained buffer'):
                    d.launch_position()
                self.assertEqual(physics.calls, 0)
                self.assertEqual(snapshot.calls, 0)

    def test_world_extent_or_mask_type_drift_is_rejected_before_plan_construction(self):
        for attr, value in (('shape', (1,)), ('dtype', 'f64'), ('is_contiguous', False)):
            with self.subTest(attr=attr):
                d = self.adapter([0, 1])
                setattr(d.requested_mask, attr, value)
                with self.assertRaisesRegex(ValueError, 'retained world mask'):
                    self.namespace['_create_position_snapshot_plan'](d, Path('/build'))


if __name__ == '__main__':
    unittest.main()
