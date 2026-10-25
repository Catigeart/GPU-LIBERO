"""Unused reference compiles are deferred; actual legacy gates remain enforced."""
import ast
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

GPU = Path(__file__).resolve().parents[1] / 'libero/libero/gpu'
with patch.dict(sys.modules, {'libero.gpu.host.paths': SimpleNamespace(
        GPU_ROOT=GPU, REPO_ROOT=GPU.parents[2])}):
    bridge_module = runpy.run_path(str(GPU / 'host/runtime/newton/s07_bridge.py'))
Bridge = bridge_module['S07DeviceBridge']


class DeferredNativeBuildTests(unittest.TestCase):
    def test_complete_native_capture_begins_forward_before_prefix_capture(self):
        tree = ast.parse((GPU/'host/assembly/components.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_capture_reset_forward')
        state = {'capturing': False, 'begun': False, 'calls': 0}
        class Capture:
            graph = object()
            def __init__(self, **kwargs):
                pass
            def __enter__(self):
                state['capturing'] = True
                return self
            def __exit__(self, *args):
                state['capturing'] = False
        def complete_forward():
            self.assertTrue(state['capturing'])
            state['begun'] = True
            state['calls'] += 1
        wp = SimpleNamespace(ScopedCapture=Capture, capture_if=lambda *a, on_true, **k: on_true())
        namespace = {'wp': wp}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'native_capture', 'exec'), namespace)
        loop = SimpleNamespace(stream=object(), recovery=SimpleNamespace(reset_mask=object()),
                               _reset_forward=complete_forward)
        namespace['_capture_reset_forward'](loop)
        self.assertTrue(state['begun'])
        self.assertEqual(state['calls'], 1)
        self.assertIs(loop._reset_forward_graph, Capture.graph)

    def test_newton_recording_binds_actual_entry_owners_without_a_forward(self):
        tree = ast.parse((GPU/'host/assembly/pipeline/forward.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FusedShapePhysicsForward')
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        class Base:
            def __init__(self):
                self.n = SimpleNamespace()
                self.n_upstream, self.mask = object(), object()
            def enqueue(self, *args):
                raise AssertionError('construction must not execute physics')
        node = ast.ClassDef(name='Forward', bases=[ast.Name(id='Base', ctx=ast.Load())],
                            keywords=[], body=[init], decorator_list=[])
        namespace = {'Base': Base}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                     'cold_forward', 'exec'), namespace)
        owner = namespace['Forward']()
        self.assertIs(owner.n._r03_upstream, owner.n_upstream)
        self.assertIs(owner.n._r03_enabled, owner.mask)

    def test_deferred_bridge_still_rejects_actual_unqualified_product(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Bridge, '_compile', side_effect=AssertionError('unexpected PTX product')) as compile_, \
                    patch.object(Bridge, '_driver') as driver:
                bridge = Bridge(build_dir=Path(directory), nvrtc_path='unused', defer_compile=True)
                compile_.assert_not_called()
                with self.assertRaisesRegex(AssertionError, 'unexpected PTX'):
                    bridge.load()
                compile_.assert_called_once()
                driver.assert_not_called()

    def test_standalone_bridge_keeps_eager_default(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Bridge, '_compile', return_value=b'qualified-test-product') as compile_:
                bridge = Bridge(build_dir=Path(directory), nvrtc_path='unused')
                compile_.assert_called_once()
                self.assertEqual(bridge._ptx, b'qualified-test-product')

    def test_deferred_bridge_does_not_read_source_or_discover_nvrtc(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Path, 'read_bytes', side_effect=AssertionError('source read')), \
                    patch.dict(Bridge.__init__.__globals__, {
                        'default_nvrtc': lambda: (_ for _ in ()).throw(AssertionError('NVRTC lookup'))}):
                bridge = Bridge(kernel_source=Path(directory)/'absent.cu',
                                build_dir=Path(directory), defer_compile=True)
                self.assertIsNone(bridge.nvrtc_path)
                self.assertIsNone(bridge._ptx)

    def test_source_drift_is_rejected_on_actual_standalone_use(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'drift.cu'
            source.write_text('invalid frozen source')
            bridge = Bridge(kernel_source=source, build_dir=Path(directory),
                            nvrtc_path='unused', defer_compile=True)
            with self.assertRaisesRegex(AssertionError, 'source drift'), \
                    patch.object(Bridge, '_driver', side_effect=AssertionError('unexpected driver')):
                bridge.load()

    def test_reset_graph_captures_once_and_never_launches_a_child_graph_in_capture(self):
        tree = ast.parse((GPU/'host/runtime/step/recovery_runtime_r02.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CurrentWrittenRecoveringRuntime')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_conditional_reset_forward')
        events = []
        graph = object()
        class Capture:
            def __init__(self, **kwargs):
                self.graph = graph
            def __enter__(self):
                events.append('begin')
                return self
            def __exit__(self, *args):
                events.append('end')
        wp = SimpleNamespace(ScopedCapture=Capture,
            capture_if=lambda *a, **k: events.append('if'),
            capture_launch=lambda g, **k: events.append(('launch', g)))
        namespace = {'wp': wp}
        exec(compile(ast.Module(body=[method], type_ignores=[]), 'conditional_reset', 'exec'), namespace)
        owner = SimpleNamespace(stream=SimpleNamespace(is_capturing=False), _reset_forward_graph=None,
            reset_predicate=object(), _reset_forward=lambda: None,
            _update_reset_forward_condition=lambda: events.append('predicate'))
        fn = namespace['_conditional_reset_forward']
        fn(owner)
        fn(owner)
        self.assertEqual(events.count('begin'), 1)
        self.assertEqual(events.count(('launch', graph)), 2)
        owner.stream.is_capturing = True
        fn(owner)
        self.assertEqual(events[-1], 'if')
        self.assertEqual(events.count('begin'), 1)


if __name__ == '__main__':
    unittest.main()
