"""Fixed scene reads and environment cleanup, without optional GPU packages."""
import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from threading import RLock
from types import SimpleNamespace
from unittest.mock import patch

from libero.gpu.host.backends.mjwarp import scene_cache
from libero.gpu.host.build.compilation import MissingCompiledProduct
from libero.gpu.host.runtime.core.model_bytes import model_bytes, load_native_model

GPU = Path(__file__).resolve().parents[1] / 'libero/libero/gpu'


def scene_files():
    path = GPU / 'host/backends/cuda_fp32/scene.py'
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)
                and n.name == '_scene_files')
    scope = dict(contextmanager=contextmanager, __package__='libero.gpu.host.backends.cuda_fp32')
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope['_scene_files']


class SceneLifetimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        binding = patch.object(scene_cache, 'binding', return_value={'engine': {'python': 'fixed'}})
        binding.start()
        self.addCleanup(binding.stop)
        inputs = patch.object(scene_cache, 'offline_inputs', return_value={'assets': {}, 'library_sha256': 'lib'})
        inputs.start()
        self.addCleanup(inputs.stop)
        self.files = scene_files()

    def publish(self):
        folder, _, _, record = scene_cache.locate(self.root, 'xml', {}, {}, prepare=True)
        folder.mkdir(parents=True)
        raw = b'offline model'
        (folder / 'model.mjb').write_bytes(raw)
        (folder / 'receipt.json').write_text(json.dumps(dict(schema='modern-scene-v2', binding=record,
            offline_inputs={'assets': {}, 'library_sha256': 'lib'},
            model_sha256=hashlib.sha256(raw).hexdigest(), processed_xml_sha256='processed')))
        return folder

    def test_repeated_runtime_loads_never_open_a_writer_or_modify_fixed_inputs(self):
        folder = self.publish()
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob('*') if p.is_file()}
        with patch('libero.gpu.host.runtime.core.product_paths.build_unit_lock', side_effect=AssertionError('runtime writer')):
            for _ in range(3):
                with self.files(self.root, 'xml', {}, {}, None, prepare=False) as row:
                    self.assertEqual(row[0], folder)
                    self.assertIsNotNone(row[2])
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob('*') if p.is_file()})

    def test_missing_input_fails_without_creating_runtime_directories(self):
        with self.assertRaises(MissingCompiledProduct):
            with self.files(self.root, 'xml', {}, {}, None, prepare=False):
                pass
        self.assertEqual(list(self.root.iterdir()), [])
        folder = self.publish()
        (folder / 'model.mjb').unlink()
        before = set(self.root.rglob('*'))
        with self.assertRaises(MissingCompiledProduct):
            with self.files(self.root, 'xml', {}, {}, None, prepare=False):
                pass
        self.assertEqual(set(self.root.rglob('*')), before)

    def test_offline_preparation_rechecks_the_scene_after_waiting_for_writer(self):
        @contextmanager
        def writer(folder):
            self.assertFalse(folder.exists())
            self.publish()  # A preceding writer completed while we waited.
            yield
        with patch('libero.gpu.host.runtime.core.product_paths.build_unit_lock', writer):
            with self.files(self.root, 'xml', {}, {}, None, prepare=True) as row:
                self.assertIsNotNone(row[2])

    def test_runtime_project_reads_do_not_open_the_offline_writer_lock(self):
        from libero.gpu.host.build.compilation import compilation_scope
        from libero.gpu.host.runtime.core.product_paths import build_project
        with compilation_scope(False), build_project(self.root) as project:
            self.assertEqual(project['root'], self.root)
        self.assertEqual(list(self.root.iterdir()), [])
        with compilation_scope(False), self.assertRaises(MissingCompiledProduct):
            with build_project(self.root/'missing'):
                pass
        self.assertFalse((self.root/'missing').exists())


class EnvironmentCloseTests(unittest.TestCase):
    def test_fp32_close_continues_after_observer_failure_without_reclosing_runtime(self):
        path = GPU / 'host/backends/cuda_fp32/environment.py'
        cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'close')
        scope = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), scope)
        calls = []
        def observer_close():
            calls.append('observer')
            if calls.count('observer') == 1:
                raise RuntimeError('observer failure')
        runtime = SimpleNamespace(close=lambda: calls.append('runtime'), env=object())
        env = SimpleNamespace(_lock=RLock(), _closed=False, runtime=runtime, _physics=runtime,
            observer=SimpleNamespace(close=observer_close),
            stream=SimpleNamespace(synchronize=lambda: None, close=lambda: calls.append('stream')),
            _facades=[], shells=[], _goal_plans=[], _held_contacts={}, _fields={}, _raw_fields={},
            built=object(), initial_states=object(), data=object(), model=object(), _selection=object())
        with self.assertRaisesRegex(RuntimeError, 'observer failure'):
            scope['close'](env)
        self.assertIsNone(env.runtime)
        self.assertTrue(env._failed)
        self.assertIsNone(runtime.env)
        scope['close'](env)
        scope['close'](env)
        self.assertEqual(calls, ['runtime', 'observer', 'observer', 'stream'])
        self.assertTrue(env._closed)
        self.assertIsNone(env.built)
        self.assertIsNone(env.data)


class SharedCodeTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('cc') and importlib.util.find_spec('warp'), 'requires cc and Warp host bindings')
    def test_stateless_code_loads_from_one_fixed_file_without_a_compiler_or_key(self):
        from libero.gpu.host.build import native
        from libero.gpu.host.build.compilation import compilation_scope
        from libero.gpu.host.build.product_cache import product_cache_scope
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'probe.c'
            source.write_text('int answer(void) { return 42; }\n')
            binary = root / 'probe.so'
            subprocess.run(['cc', '-shared', '-fPIC', str(source), '-o', str(binary)], check=True, capture_output=True)
            raw = binary.read_bytes()
            def build(command, **kwargs):
                Path(command[command.index('-o') + 1]).write_bytes(raw)
                return SimpleNamespace(returncode=0, stdout='', stderr='')
            with patch.object(native.wp, 'get_device', return_value=SimpleNamespace(is_cuda=True, arch=80)), \
                 patch.object(native, 'nvcc_path', return_value='/explicit/compiler'), \
                 patch.object(native, 'compiler_dependencies', return_value=[]), \
                 patch.object(native, 'toolchain_record', return_value={'components': [native.file_record(source)]}), \
                 patch.object(native.subprocess, 'run', side_effect=build), \
                 product_cache_scope(root/'products'):
                kwargs = dict(required_symbols=('answer',), shared=True)
                _, prepared = native._compile_library_receipted('probe', 'actual admitted source', root/'offline', **kwargs)
                key = prepared['product_cache']['key']
                inputs = Path(prepared['product_cache']['directory'])/'inputs'
                inputs.mkdir(exist_ok=True)
                (inputs/'unit0.cu').write_text('actual admitted source')
                with compilation_scope(False), \
                     patch.object(native, 'nvcc_path', side_effect=AssertionError('runtime compiler discovery')), \
                     patch.object(native, 'compiler_dependencies', side_effect=AssertionError('runtime compiler query')), \
                     patch.object(native, 'toolchain_record', side_effect=AssertionError('runtime toolchain query')):
                    first, a = native._compile_library_receipted('probe', 'actual admitted source', root/'first', **kwargs)
                    second, b = native._compile_library_receipted('probe', 'actual admitted source', root/'second', **kwargs)
                self.assertEqual(a['library'], b['library'])
                self.assertTrue(a['shared_product'])
                self.assertIsNot(first, second)
                self.assertEqual((first.answer(), second.answer()), (42, 42))
                self.assertFalse(list((root/'first').glob('*.so')))
                self.assertFalse(list((root/'second').glob('*.so')))


class ModelTransportTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('mujoco'), 'requires MuJoCo model binding')
    def test_native_buffer_serialization_matches_file_and_clone_identity(self):
        import copy
        import mujoco
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint/><geom size=".1"/></body></worldbody></mujoco>')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reference.mjb'
            mujoco.mj_saveModel(model, str(path), None)
            expected = path.read_bytes()
        self.assertEqual(model_bytes(model), expected)
        if hasattr(os, 'memfd_create'):
            with patch('os.memfd_create', side_effect=OSError('unavailable')):
                self.assertEqual(model_bytes(model), expected)
        clone = copy.copy(model)
        self.assertIsNot(clone, model)
        self.assertEqual(model_bytes(clone), expected)
        self.assertIs(load_native_model(model=model, expected_sha256=hashlib.sha256(expected).hexdigest())[0], model)
        clone.body_pos[1, 0] += .1
        with self.assertRaises(ValueError):
            model_bytes(clone, hashlib.sha256(expected).hexdigest())

    @unittest.skipUnless(hasattr(os, 'memfd_create') and importlib.util.find_spec('mujoco'),
                         'requires anonymous memory files and MuJoCo')
    def test_model_serialization_releases_anonymous_file_on_error(self):
        import mujoco
        descriptors = []
        def fail(model, filename, buffer):
            self.assertIsNone(buffer)
            fd = int(filename.rsplit('/', 1)[-1])
            os.fstat(fd)
            descriptors.append(fd)
            raise RuntimeError('serialization failed')
        with patch.object(mujoco, 'mj_saveModel', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'serialization failed'):
                model_bytes(object())
        self.assertEqual(len(descriptors), 1)
        with self.assertRaises(OSError):
            os.fstat(descriptors[0])

    @unittest.skipUnless(importlib.util.find_spec('mujoco'), 'requires MuJoCo model binding')
    def test_newton_consumers_accept_memory_owners_before_gpu_preparation(self):
        import mujoco
        model = mujoco.MjModel.from_xml_string('<mujoco><option cone="elliptic"/>'
            '<worldbody><body><joint/><geom size=".1"/></body></worldbody></mujoco>')
        spec = SimpleNamespace(model_sha256=hashlib.sha256(model_bytes(model)).hexdigest(),
            dimensions=SimpleNamespace(nv=model.nv, nM=model.nM))
        class ReachedGpuPreparation(Exception): pass
        def candidate(nv, nm, root):
            self.assertEqual((nv, nm), (model.nv, model.nM))
            raise ReachedGpuPreparation()
        for name in ('live_api.py', 'live_copy_api.py', 'live_guard_copy_api.py'):
            with self.subTest(consumer=name):
                path = GPU / 'host/runtime/newton' / name
                functions = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)
                    and (n.name == '_validate_native' or n.name.startswith('prepare_live'))]
                scope = dict(load_native_model=load_native_model, mujoco=mujoco, SimpleNamespace=SimpleNamespace,
                    Path=Path, load_candidate=candidate, ColdNewtonRuntimeAdapter=object,
                    _AdmittedModel=lambda admission: SimpleNamespace(_validate=lambda: None))
                exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), scope)
                function = scope[next(n.name for n in functions if n.name.startswith('prepare_live'))]
                with self.assertRaises(ReachedGpuPreparation):
                    function(model=model, model_spec=spec, rows=8, contacts=4, nworld=2,
                             build_root='unused', live_inputs={})

    @unittest.skipUnless(importlib.util.find_spec('mujoco'), 'requires MuJoCo model binding')
    def test_native_replacement_has_no_model_files_and_rejects_owner_drift(self):
        import mujoco
        if mujoco.__version__ != '2.3.7':
            self.skipTest('native compatibility intake uses MuJoCo 2.3.7')
        from libero.gpu.model.factory import BuiltTaskModel, native_model_metadata
        from libero.gpu.host.runtime.core.contracts import ContractError
        from libero.gpu.host.runtime.core.native_model_update import prepare_native_model_update
        from libero.gpu.host.runtime.core.product_paths import product_build
        model = mujoco.MjModel.from_xml_string('<mujoco><option cone="elliptic"/>'
            '<worldbody><body><joint/><geom size=".1"/></body></worldbody></mujoco>')
        built = BuiltTaskModel(model, '<mujoco/>', None, native_model_metadata(model))
        before = model_bytes(model)
        with product_build() as root:
            plan = prepare_native_model_update(built, {}, {'timestep': float(model.opt.timestep)*2})
            self.assertIsNot(plan.built.native_model, model)
            self.assertEqual(model_bytes(model), before)
            self.assertEqual(plan.built.native_model.opt.timestep, model.opt.timestep*2)
            self.assertFalse(list(root.rglob('*.mjb')))
            self.assertFalse(list(root.rglob('*PREFLIGHT.json')))
            self.assertIsNone(plan.mjb_path)
            plan.validate_base()
            plan.built.native_model.body_pos[1, 0] += .1
            with self.assertRaises(ContractError):
                plan.validate_base()
        self.assertFalse(root.exists())


if __name__ == '__main__':
    unittest.main()
