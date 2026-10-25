"""Load-only API boundaries and offline persistence; no CUDA/physics execution."""
import ast
from contextlib import contextmanager
import errno
import hashlib
from functools import wraps
import json
import os
from pathlib import Path
import sys
import tempfile
from threading import RLock
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from libero.gpu.host.build import compatibility
from libero.gpu.host.build.compilation import (
    compilation_allowed, compilation_scope, current_scope, load_only_entry,
    MissingCompiledProduct, require_compilation)
from libero.gpu.host.build.product_cache import (
    ProductCache, product_cache_scope, current_product_cache,
    require_product_persistence)

GPU = Path(__file__).resolve().parents[1] / 'libero/libero/gpu'


def task_driver():
    # Exercise actual entry bodies without importing optional robosuite/CUDA.
    path = GPU / 'host/adapters/environment/task_driver.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GpuTaskEnvironment')
    scope = dict(load_only_entry=load_only_entry, RLock=RLock, contextmanager=contextmanager,
                 np=SimpleNamespace(random=SimpleNamespace(RandomState=lambda: None)))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope['GpuTaskEnvironment']


class CacheHardeningTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def authority(self, **kwargs):
        return ProductCache(self.root / 'products', **kwargs)

    def request(self, authority, **kwargs):
        options = dict(build=lambda: b'completed binary', validate=lambda: None, exports=lambda raw: None)
        options.update(kwargs)
        return authority.obtain({'source': 'current'}, ['entry'], **options)

    def test_runtime_entry_restores_scope_and_keeps_explicit_offline_authority(self):
        @load_only_entry
        def lazy():
            require_compilation('native or Warp', 'missing module')
        self.assertIsNone(current_scope())
        with self.assertRaises(MissingCompiledProduct):
            lazy()
        self.assertIsNone(current_scope())
        self.assertTrue(compilation_allowed())
        with compilation_scope(True) as outer:
            lazy()
            @load_only_entry
            def nested():
                self.assertIs(current_scope(), outer)
                lazy()
            nested()
            with compilation_scope(False):
                with self.assertRaises(MissingCompiledProduct):
                    lazy()

    def test_direct_sim_facade_calls_are_also_load_only(self):
        path = GPU / 'host/adapters/environment/facade.py'
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_serialized')
        scope = dict(wraps=wraps, load_only_entry=load_only_entry)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
        @scope['_serialized']
        def forward(self):
            require_compilation('native', 'lazy forward')
        sim = SimpleNamespace(_lock=RLock())
        with self.assertRaises(MissingCompiledProduct):
            forward(sim)
        with compilation_scope(True):
            forward(sim)

    def test_warp_codegen_guard_blocks_a_late_runtime_cache_miss(self):
        from libero.gpu.host.build.compilation import install_warp_guard
        calls = []
        class Module:
            name = 'late-warp'
            def _compile(self):
                return False
            def _run_codegen(self):
                calls.append('compiled')
        context = SimpleNamespace(Module=Module)
        source = SimpleNamespace(context=context)
        warp = SimpleNamespace(_src=source)
        with patch.dict(sys.modules, {'warp': warp, 'warp._src': source, 'warp._src.context': context}):
            install_warp_guard()
            @load_only_entry
            def render():
                Module()._run_codegen()
            with self.assertRaises(MissingCompiledProduct):
                render()
            self.assertEqual(calls, [])
            with compilation_scope(True):
                render()
            self.assertEqual(calls, ['compiled'])

    def test_real_driver_constructor_step_reset_observation_and_render_boundaries(self):
        driver = task_driver()
        def miss(*args, **kwargs):
            require_compilation('native or Warp', 'late module')
        with patch.object(driver, '_install_shell', lambda self: None), \
             patch.object(driver, '_initialize_fresh', lambda self: miss()):
            with self.assertRaises(MissingCompiledProduct):
                driver(SimpleNamespace(validate=lambda: None))
            with compilation_scope(True):
                driver(SimpleNamespace(validate=lambda: None))
        # A missing product at the actual call sites must be blocked without
        # the external frozen scope used by the earlier lifecycle verifier.
        imports = {
            'robosuite.utils.errors': SimpleNamespace(RandomizationError=type('RandomizationError', (Exception,), {})),
            'libero.gpu.host.adapters.observations.boundary': SimpleNamespace(boundary_predicates=None)}
        with patch.dict(sys.modules, imports):
            for method, args in (('step', [object()]), ('reset', []), ('_get_observations', []),
                                 ('initialize_renderer', []), ('render', []),
                                 ('replace_native_components', [None, None]),
                                 ('replace_components_for_model_update', [None, None, None])):
                with self.subTest(method=method):
                    env = driver.__new__(driver)
                    env._lock = RLock()
                    env._open = miss  # First guarded operation of the real body.
                    with self.assertRaises(MissingCompiledProduct):
                        getattr(env, method)(*args)
                    self.assertIsNone(current_scope())

    def test_optional_private_build_and_required_persistence_have_distinct_contracts(self):
        authority = self.authority(max_bytes=1)
        _, _, status = self.request(authority)
        self.assertFalse(status['persisted'])
        with require_product_persistence():
            with self.assertRaises(OSError) as error:
                self.request(authority)
        self.assertEqual(error.exception.errno, errno.ENOSPC)
        self.assertIn('cannot be persisted', str(error.exception))
        self.assertFalse(list(authority.root.glob('*/product.so')))
        self.assertFalse(self.request(authority)[2]['persisted'])  # Policy restored on failure.

    def test_public_product_cache_bounds_are_configurable_without_overriding_explicit_bounds(self):
        with patch.dict(os.environ, {'LIBERO_GPU_PRODUCT_CACHE_MAX_BYTES': '1',
                                    'LIBERO_GPU_PRODUCT_CACHE_MAX_ENTRIES': '2'}):
            with product_cache_scope(self.root / 'products') as authority:
                self.assertEqual((authority.max_bytes, authority.max_entries), (1, 2))
                with require_product_persistence(), self.assertRaises(OSError):
                    self.request(authority)
            with product_cache_scope(self.root / 'products',
                                     max_bytes=1024**2, max_entries=3) as authority:
                self.assertEqual((authority.max_bytes, authority.max_entries), (1024**2, 3))
                self.assertTrue(self.request(authority)[2]['persisted'])

    def test_nested_cache_persists_and_frozen_reuse_never_rebuilds(self):
        with require_product_persistence(), product_cache_scope(
                self.root / 'products'):
            authority = current_product_cache()
            self.assertTrue(self.request(authority)[2]['persisted'])
        def compile_():
            require_compilation('native', 'unit')
            self.fail('cache hit must not rebuild')
        with compilation_scope(False):
            self.assertTrue(self.request(authority, build=compile_)[2]['hit'])
            next(authority.root.glob('*/product.so')).write_bytes(b'corrupt')
            def load(raw):
                if raw == b'corrupt':
                    raise OSError('invalid shared library')
            with self.assertRaisesRegex(OSError, 'invalid shared library'):
                self.request(authority, build=compile_, exports=load)

    def test_loading_ignores_build_toolchain_but_checks_code_and_architecture(self):
        authority = self.authority()
        identity = dict(source='actual source', architecture=80,
                        toolchain=dict(environment_sha256='offline shell', components=['actual compiler bytes']))
        authority.obtain(identity, ['entry'], build=lambda: b'completed', validate=lambda: None, exports=lambda raw: None)
        receipts = {p: p.read_bytes() for p in authority.root.glob('*/receipt.json')}
        current = dict(identity, toolchain=dict(identity['toolchain'], environment_sha256='training shell'))
        with compilation_scope(False):
            _, receipt, status = authority.obtain(current, ['entry'],
                build=lambda: self.fail('runtime build'), validate=lambda: None, exports=lambda raw: None)
            self.assertTrue(status['hit'])
            self.assertEqual(receipt['identity'], identity)
            authority.obtain(dict(current, toolchain={'components':['compiler removed']}), ['entry'],
                build=lambda: self.fail('runtime build'), validate=lambda: self.fail('runtime toolchain validation'),
                exports=lambda raw: None)
            for changed in (dict(current, source='edited source'), dict(current, architecture=90)):
                with self.assertRaises(MissingCompiledProduct):
                    authority.obtain(changed, ['entry'], build=lambda: self.fail('runtime build'),
                                     validate=lambda: None, exports=lambda raw: None)
        self.assertTrue(all(p.read_bytes() == raw for p, raw in receipts.items()))

    def test_ordinary_artifacts_and_legacy_cache_need_no_signing_key(self):
        cache = self.authority()
        identity = {'source': 'two artifacts', 'architecture': 80}
        values = {'module.so': b'library', 'module.ptx': b'ptx'}
        _, receipt, status = cache.obtain_artifacts(identity, ['entry'], names=values,
            build=lambda: values, validate=lambda: None, exports=lambda files: None)
        folder = Path(status['directory'])
        self.assertEqual((folder/'module.so').read_bytes(), b'library')
        self.assertEqual((folder/'module.ptx').read_bytes(), b'ptx')
        self.assertFalse((folder/'product.so').exists())  # No base64 bundle.
        self.assertNotIn('signature', json.loads((folder/'receipt.json').read_text()))
        self.assertFalse((self.root/'authority').exists())
        with compilation_scope(False):
            cache.obtain_artifacts(identity, ['entry'], names=values,
                build=lambda: self.fail('runtime compile'), validate=lambda: self.fail('runtime validation'),
                exports=lambda files: self.assertEqual(files, values))
            (folder/'module.ptx').unlink()
            with self.assertRaises(MissingCompiledProduct):
                cache.obtain_artifacts(identity, ['entry'], names=values,
                    build=lambda: self.fail('runtime compile'), validate=lambda: None, exports=lambda files: None)

        legacy = cache.legacy_root/'old-build'
        legacy.mkdir(parents=True)
        raw = b'legacy library'
        old_identity = {'source': 'legacy code', 'architecture': 80}
        old_receipt = dict(schema='authorized-native-product-v1', completed=True,
            build_key=legacy.name, identity=old_identity, required_symbols=['entry'],
            binary_sha256=hashlib.sha256(raw).hexdigest(), binary_bytes=len(raw))
        (legacy/'product.so').write_bytes(raw)
        (legacy/'receipt.json').write_text(json.dumps({'receipt':old_receipt, 'signature':'retired, unused'}))
        with compilation_scope(False), patch('libero.gpu.host.build.product_cache._lock',
                side_effect=AssertionError('runtime cache write')):
            loaded = ProductCache(self.root/'products')
            _, _, result = loaded.obtain(old_identity, ['entry'], build=lambda: self.fail('runtime compile'),
                validate=lambda: self.fail('runtime toolchain scan'), exports=lambda content: self.assertEqual(content, raw))
        self.assertEqual(Path(result['directory']), legacy)

    def test_offline_dependency_change_rebuilds_same_product_selection(self):
        cache = self.authority()
        first = dict(source='same CUDA', dependencies=[{'sha256':'first header'}])
        second = dict(source='same CUDA', dependencies=[{'sha256':'changed header'}])
        _, _, a = cache.obtain(first, ['entry'], build=lambda: b'first binary',
            validate=lambda: None, exports=lambda raw: None)
        _, _, b = cache.obtain(second, ['entry'], build=lambda: b'new binary',
            validate=lambda: None, exports=lambda raw: None)
        self.assertEqual(a['key'], b['key'])
        self.assertFalse(b['hit'])
        self.assertEqual((Path(b['directory'])/'product.so').read_bytes(), b'new binary')

    def test_load_only_authority_opens_no_writer_lock_or_missing_cache_directory(self):
        authority = self.authority()
        self.request(authority)
        before = {p: p.stat().st_mtime_ns for p in self.root.rglob('*')}
        with compilation_scope(False), \
             patch('libero.gpu.host.build.product_cache._lock', side_effect=AssertionError('runtime writer lock')):
            loaded = self.authority()
            self.assertTrue(self.request(loaded)[2]['hit'])
            with self.assertRaises(MissingCompiledProduct):
                ProductCache(self.root/'absent')
        self.assertEqual(before, {p: p.stat().st_mtime_ns for p in self.root.rglob('*')})

    def test_multi_artifact_budget_cannot_publish_partial_ready_bundle(self):
        authority = self.authority(max_bytes=1)
        with require_product_persistence(), self.assertRaises(OSError):
            authority.obtain_artifacts({'source': 'both'}, ['entry'], names=['module.so', 'module.ptx'],
                build=lambda: {'module.so': b'binary', 'module.ptx': b'PTX'},
                validate=lambda: None, exports=lambda raw: None)
        self.assertFalse(list(authority.root.glob('*/receipt.json')))

    def test_capacity_failure_keeps_prior_products_and_counts_unpaired_binary(self):
        authority = self.authority()
        self.request(authority)
        prior = {p: p.read_bytes() for p in authority.root.glob('*/*') if p.name != 'build.lock'}
        orphan = authority.root / 'interrupted'
        orphan.mkdir()
        authority.max_bytes = 4096
        for name in ('product.so', '0' * 32 + '.so'):
            with self.subTest(artifact=name):
                path = orphan / name
                path.write_bytes(b'x' * 4096)  # No usable receipt.
                with require_product_persistence(), self.assertRaises(OSError):
                    authority.obtain({'source': 'another'}, ['entry'], build=lambda: b'new',
                                     validate=lambda: None, exports=lambda raw: None)
                path.unlink()
        self.assertTrue(all(path.read_bytes() == raw for path, raw in prior.items()))

    def test_failed_product_publication_cleans_its_temporary_files(self):
        authority = self.authority()
        with patch('libero.gpu.host.build.product_cache.os.replace', side_effect=OSError(errno.ENOSPC, 'full')):
            with self.assertRaises(OSError):
                self.request(authority)
        files = [p.name for p in authority.root.glob('*/*')]
        self.assertEqual(files, ['build.lock'])

    def test_offline_prepare_replaces_stale_complete_with_failed_on_full_cache(self):
        # Real prepare() + real product cache, with only the environment and
        # source index substituted. No dependency on optional simulation packages.
        project = self.root / 'project'
        compatibility._write_status(project / 'compatibility', {'status': 'complete', 'old': True})
        def construct(**kwargs):
            self.assertTrue(compilation_allowed())
            with product_cache_scope(self.root / 'products', max_bytes=1):
                self.request(current_product_cache())
            self.fail('unpersisted constructor must fail')
        modules = {
            'numpy': SimpleNamespace(),
            'libero.libero.envs': SimpleNamespace(OffScreenRenderEnv=construct),
            'libero.gpu.precision': SimpleNamespace(bits=64),
            'libero.gpu.host.build.aot': SimpleNamespace(configure_warp_cache=lambda *a, **kw: None)}
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {'LIBERO_GPU_CACHE_DIR': str(self.root)}), \
             patch('libero.gpu.host.build.compilation.install_warp_guard'), \
             patch('libero.gpu.host.runtime.core.specialization_cache.index_packages') as index:
            with self.assertRaises(OSError):
                compatibility.prepare(project, [dict(key='task', bddl='task.bddl')])
        state = json.loads((project / 'compatibility/BUILD.json').read_text())
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(state['error']['type'], 'OSError')
        self.assertNotIn('old', state)
        index.assert_not_called()
        self.assertFalse(list((project / 'compatibility').glob('*.tmp')))


if __name__ == '__main__':
    unittest.main()
