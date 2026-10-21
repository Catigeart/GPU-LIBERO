"""Compatibility construction authority and storage routing; no CUDA work."""
from contextlib import contextmanager
import ast
import gc
import hashlib
import inspect
import linecache
import os
from pathlib import Path
import sys
import runpy
import shutil
import subprocess
import threading
import types
import weakref
import time
import tempfile
from types import SimpleNamespace
import unittest
from threading import RLock
from unittest.mock import patch

from libero.gpu.host.build.compilation import compilation_scope, compilation_allowed
from libero.gpu.host.build import build_artifacts
from libero.gpu.host.build.library_lifetime import load_cdll
from libero.gpu.host.build.build_receipts import file_record
from libero.gpu.host.build.product_cache import product_cache_scope
from libero.gpu.host.runtime.core.product_paths import (
    product_build, build_project, current_build_project, private_product_root,
    current_build_lifetime, record_build)
from libero.gpu.host.runtime.core.build_lifetime import cold_build_scope
GPU = Path(__file__).resolve().parents[1] / 'libero/libero/gpu'
# Load only the stdlib construction boundaries; the optional robosuite facade
# is not needed to test compiler authority in the minimal CI environment.
product_factory = SimpleNamespace(**runpy.run_path(str(GPU / 'host/adapters/environment/product_factory.py')))
tree = ast.parse((GPU / 'host/adapters/environment/factory.py').read_text())
namespace = dict(__name__=__name__, __package__='libero.gpu.host.adapters.environment',
                 _LOCK=RLock(), _FACTORIES={})
nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
         node.name in ('register_gpu_task_factory', 'install_model_only_gpu_factory')]
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(GPU / 'host/adapters/environment/factory.py'), 'exec'), namespace)
entry = SimpleNamespace(**namespace)


class CompatibilityBuildTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        environment = patch.dict(os.environ, {'LIBERO_GPU_CACHE_DIR': str(self.root),
                                              'LIBERO_GPU_DISK_MIN_FREE_BYTES': '0'})
        environment.start()
        self.addCleanup(environment.stop)
        guard = patch('libero.gpu.host.build.compilation.install_warp_guard')
        self.warp_guard = guard.start()
        self.addCleanup(guard.stop)
        modules = patch.dict(sys.modules, {'libero.gpu.host.build.aot': SimpleNamespace(
            configure_warp_cache=lambda root: None)})
        modules.start()
        self.addCleanup(modules.stop)

    def test_normal_construction_is_load_only_and_uses_selected_project(self):
        selected = self.root / 'selected'
        with build_project(selected), product_factory.compatibility_scope():
            self.assertFalse(compilation_allowed())
            self.assertEqual(current_build_project()['root'], selected)
            self.warp_guard.assert_called_once_with()

    def test_offline_authority_is_explicit_and_outer_freeze_wins(self):
        with compilation_scope(True), product_factory.compatibility_scope():
            self.assertTrue(compilation_allowed())
        with compilation_scope(False), compilation_scope(True), product_factory.compatibility_scope():
            self.assertFalse(compilation_allowed())

    def test_nested_provider_preserves_outer_event_log(self):
        (self.root / 'project').mkdir()  # Offline preparation precedes the frozen provider.
        with compilation_scope(False) as outer:
            with product_factory.compatibility_scope():
                record_build('cache-check', 'prepared product')
            self.assertEqual(len(outer['events']), 1)
            self.assertEqual(outer['events'][0]['commands_executed'], [])

    def test_private_signed_products_are_reclaimed_without_deleting_project(self):
        selected = self.root / 'project'
        selected.mkdir()
        with product_cache_scope(self.root / 'products', authority_root=self.root / 'authority'):
            with product_build() as run:
                target = private_product_root(selected / 'collision')
                target.mkdir(parents=True)
                (target / 'private.so').write_bytes(b'private image')
                self.assertTrue(target.is_relative_to(run))
                self.assertEqual(private_product_root(run / 'cuda'), run / 'cuda')
            self.assertFalse(run.exists())
        self.assertTrue(selected.exists())
        self.assertFalse((selected / 'collision').exists())
        self.assertEqual(private_product_root(selected), selected)

    def test_replacement_restores_project_scope_before_preparation(self):
        selected = self.root / 'project'
        @cold_build_scope(project_root=lambda: selected)
        def replace():
            self.assertEqual(current_build_project()['root'], selected)
        replace()
        self.assertIsNone(current_build_project())

    def test_random_private_paths_preserve_build_identity_but_inputs_invalidate(self):
        captured = []
        class Authority:
            def obtain_artifacts(self, identity, symbols, *, names, build, validate, exports):
                validate()
                captured.append(identity)
                exports({name: b'cached image' for name in names})
                return None, {}, {'hit': True}
        def request(label, source='device source', flag='-O2', architecture=80):
            folder = self.root / label
            folder.mkdir()
            unit, header = folder / 'unit.cu', folder / 'header.cuh'
            unit.write_text(source); header.write_text('header')
            target = folder / ('random-' + label + '.so')
            with patch.object(build_artifacts, 'current_product_authority', return_value=Authority()), \
                 patch.object(build_artifacts, 'compiler_dependencies', return_value=[file_record(header)]), \
                 patch.object(build_artifacts, 'toolchain_record', return_value={'compiler': 'pinned compiler'}):
                build_artifacts.receipted_artifact_build(
                    commands=[['nvcc', flag, '-shared', str(unit), '-o', str(target)]],
                    units=[unit], outputs={'product.so': target}, architecture=architecture,
                    source_closure={str(header): file_record(header)['sha256']},
                    source_aliases={str(header): 'generated/header.cuh'},
                    verify=lambda: self.assertEqual(target.read_bytes(), b'cached image'))
            return target
        first = request('first')
        second = request('second')
        self.assertEqual(captured[0], captured[1])
        self.assertNotEqual(first.stat().st_ino, second.stat().st_ino)
        request('source-change', source='changed source')
        request('flag-change', flag='-O3')
        request('architecture-change', architecture=90)
        self.assertTrue(all(identity != captured[0] for identity in captured[2:]))

    def test_provider_scope_covers_complete_environment_initialization(self):
        events = []
        @contextmanager
        def scope():
            events.append('enter')
            try:
                yield
            finally:
                events.append('exit')
        def provider(built):
            events.append('components')
            return SimpleNamespace(validate=lambda: None)
        provider.construction_scope = scope
        def environment(*args, **kwargs):
            events.append('initialize')
            return object()
        with patch.dict(entry._FACTORIES, {}, clear=True), patch.dict(sys.modules, {
            'libero.gpu.model': SimpleNamespace(build_task_model=lambda *a, **k: object()),
            'libero.gpu.host.adapters.environment.task_driver': SimpleNamespace(GpuTaskEnvironment=environment)}):
            constructor = entry.install_model_only_gpu_factory('scope-test', provider)
            constructor(bddl_file_name='fixture.bddl')
        self.assertEqual(events, ['enter', 'components', 'initialize', 'exit'])

    def test_python_launcher_releases_owner_but_keeps_live_source(self):
        tree = ast.parse((GPU / 'host/assembly/compose/compose.py').read_text())
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
                 node.name in ('_load', '_release_python_source')]
        namespace = dict(__name__='compatibility_launcher_test', GPU_ROOT=GPU,
            scalar=SimpleNamespace(bits=64), hashlib=hashlib, linecache=linecache,
            types=types, sys=sys, weakref=weakref, _PYTHON_SOURCES={},
            _PYTHON_SOURCE_LOCK=threading.RLock())
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(GPU / 'host/assembly/compose/compose.py'), 'exec'), namespace)
        class Owner:
            value = 42
        owner = Owner(); held = weakref.ref(owner)
        source = 'def launch():\n    return owner.value\n'
        first = namespace['_load']('owner', source, {'owner': owner}, warp_module=False)
        second = namespace['_load']('owner', source, {'owner': owner}, warp_module=False)
        filename, name = first.__file__, first.__name__
        function = second.launch
        del owner, first, second
        gc.collect()
        self.assertNotIn(name, sys.modules)
        self.assertIsNotNone(held())
        self.assertEqual(function(), 42)
        self.assertEqual(inspect.getsource(function), source)
        del function
        gc.collect()
        self.assertIsNone(held())
        self.assertNotIn(filename, linecache.cache)

    def test_capsule_instances_are_private_and_retire_with_their_owner(self):
        import numpy as np
        source = GPU / 'host/runtime/collision/capsule_extension.py'
        node = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'prepare_workspace_extension')
        prepared = []
        def prepare(device):
            value = dict(execution_binding={}, build_path='private-' + str(len(prepared)),
                         build_sha256='checked')
            prepared.append(value)
            return value
        cached = {}
        namespace = dict(Path=Path, GPU_ROOT=GPU, ROUTES=((3, 5), (3, 6), (3, 7)),
            _PREPARED_NATIVE=cached, sha=lambda path: 'checked',
            generate_sources=lambda path: (self.root / 'shared-source', {}),
            runpy=SimpleNamespace(run_path=lambda path: dict(prepare_native=prepare)))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        def workspace():
            return SimpleNamespace(_device_input_binding=None, _launch_started=False, device='cuda:0',
                detect=SimpleNamespace(route_id=SimpleNamespace(assign=lambda value: None)))
        detector = SimpleNamespace(DET_CONVEX_MPR=1, DET_CAPSULE_BOX_NATIVE=2,
            build_route_ids=lambda: np.zeros((8, 8), dtype=np.int32))
        owners = []
        try:
            with patch.dict(sys.modules, {'libero.gpu.host.runtime.collision.detectors': detector}):
                first, second = workspace(), workspace()
                for ws in (first, second):
                    with product_build():
                        owner = current_build_lifetime().retain()
                        owners.append(owner)
                        namespace['prepare_workspace_extension'](ws, self.root)
                self.assertIsNot(first.detect.native, second.detect.native)
                self.assertEqual(len(cached), 2)
                owners.pop(0).close()
                self.assertEqual(list(cached.values()), [second.detect.native])
                self.assertEqual(first.detect.native['build_path'], 'private-0')
                owners.pop().close()
                self.assertFalse(cached)
        finally:
            for owner in owners:
                owner.close()

    def test_digest_cache_invalidates_same_mtime_edits_and_replacement(self):
        path = self.root / 'source.cuh'
        path.write_bytes(b'first')
        first = file_record(path)
        stamp = path.stat().st_mtime_ns
        path.write_bytes(b'other')
        os.utime(path, ns=(stamp, stamp))
        changed = file_record(path)
        self.assertNotEqual(first['sha256'], changed['sha256'])
        fresh = self.root / 'replacement'
        fresh.write_bytes(b'third')
        os.utime(fresh, ns=(stamp, stamp))
        os.replace(fresh, path)
        self.assertNotEqual(changed['sha256'], file_record(path)['sha256'])
        self.assertEqual(file_record(path)['sha256'], hashlib.sha256(b'third').hexdigest())

    def test_unchanged_old_input_hash_is_reused_and_new_edit_is_rehashed(self):
        path = self.root / 'toolchain-input'
        path.write_bytes(b'first')
        time.sleep(2.05)  # Older than the deliberate coarse-timestamp window.
        reads = []
        original = Path.open
        def opened(target, *args, **kwargs):
            if (args[0] if args else kwargs.get('mode')) == 'rb':
                reads.append(target)
            return original(target, *args, **kwargs)
        with patch.object(Path, 'open', autospec=True, side_effect=opened):
            first = file_record(path)
            self.assertEqual(file_record(path), first)
            self.assertEqual(len(reads), 1)
            stamp = path.stat().st_mtime_ns
            path.write_bytes(b'other')
            os.utime(path, ns=(stamp, stamp))
            self.assertNotEqual(file_record(path)['sha256'], first['sha256'])
            self.assertEqual(len(reads), 2)

    @unittest.skipUnless(shutil.which('cc') and Path('/proc/self/maps').is_file(), 'ELF mapping test requires cc and procfs')
    def test_deleted_private_library_unmaps_after_last_function_owner(self):
        source = self.root / 'probe.c'
        source.write_text('int flash_answer(void) { return 42; }\n')
        path = self.root / 'probe.so'
        subprocess.run(['cc', '-shared', '-fPIC', str(source), '-o', str(path)], check=True, capture_output=True)
        library = load_cdll(path)
        function = library.flash_answer
        self.assertEqual(function(), 42)
        path.unlink()
        del library
        gc.collect()
        self.assertIn(str(path), Path('/proc/self/maps').read_text())
        self.assertEqual(function(), 42)  # A borrowed function keeps its owner.
        del function
        gc.collect()
        self.assertNotIn(str(path), Path('/proc/self/maps').read_text())

if __name__ == '__main__':
    unittest.main()
