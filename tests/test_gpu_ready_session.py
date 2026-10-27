"""CPU-only compatibility handle and selected-task runtime checks."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from libero.gpu.host.build import aot
from libero.gpu.host.build.compilation import (MissingCompiledProduct,
    compilation_allowed, compilation_scope, require_compilation)


_spec = importlib.util.spec_from_file_location('_ready_session_fixture',
    Path(__file__).with_name('test_gpu_aot_build.py'))
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)


class ReadySessionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        contract, products, native, warp = _fixture.observation_fixture(aot, self.root)
        row = dict(key='suite/first.bddl', source_identity='old-source', real_bits=aot.scalar.bits,
            num_envs=8, seed=0, model_sha256='model', native_artifacts=native,
            warp_required=sorted(warp), warp_artifacts=warp, observation_products=products)
        self.record = dict(schema='libero-unified-aot-build-v1', real_bits=aot.scalar.bits,
            status='complete', num_envs=8, seed=0, source_identity='old-source',
            observation_contract=contract, tasks={row['key']: row}, artifacts={**native, **warp})
        self.manifest = self.root/'AOT_BUILD.json'
        self.save()
        self.addCleanup(patch.stopall)
        patch.object(aot, 'project_root', side_effect=lambda root: Path(root).resolve()).start()

    def save(self):
        self.manifest.write_text(json.dumps(self.record))

    def qualify(self):
        return aot.qualify_ready_session(self.root, num_envs=8, seed=0)

    def ready(self, session=None, **kwargs):
        return aot.require_ready(self.root, '/bddl/suite/first.bddl',
            num_envs=8, seed=0, ready_session=session, **kwargs)

    def test_handle_and_direct_calls_do_not_hash_or_scan_sources(self):
        with (patch.object(aot, '_sha', side_effect=AssertionError('runtime hash')),
              patch.object(aot, 'source_identity', side_effect=AssertionError('runtime source identity')),
              patch.object(aot, '_source_identity_files', side_effect=AssertionError('source scan'))):
            session = self.qualify()
            self.assertIsNone(self.ready(session, return_record=False))
            self.ready()
            (self.root/'unrelated.py').write_text('changed host implementation')
            self.ready(session)
        self.assertFalse(session.info['immutable_tree_required'])
        self.assertEqual(session.info['validation'], 'manifest-format-and-selected-profile-products')
        self.assertEqual(session.info['tasks'], 1)

    def test_unrelated_incomplete_tasks_do_not_invalidate_the_selected_task(self):
        self.record['tasks']['suite/other.bddl'] = {'incomplete': True}
        self.record['artifacts']['unrelated/missing.so'] = 'historical'
        self.save()
        self.ready(self.qualify())

    def test_manifest_is_reread_without_a_permanent_invalidated_state(self):
        session = self.qualify()
        self.record['status'] = 'building'
        self.save()
        with self.assertRaises(MissingCompiledProduct):
            self.ready(session)
        self.record['status'] = 'complete'
        self.record['source_identity'] = 'another-host-revision'
        self.save()
        self.assertEqual(self.ready(session)['source_identity'], 'another-host-revision')

    def test_missing_current_product_fails_then_can_be_repaired(self):
        row = self.record['tasks']['suite/first.bddl']
        target = self.root/next(iter(row['observation_products']['profiles']['no-camera']['native_artifacts']))
        raw = target.read_bytes()
        session = self.qualify()
        target.unlink()
        with self.assertRaisesRegex(MissingCompiledProduct, 'Required task product is missing'):
            self.ready(session)
        target.write_bytes(raw)
        self.ready(session)

    def test_explicit_build_selection_and_close_are_preserved(self):
        session = self.qualify()
        for arguments in (dict(num_envs=2, seed=0), dict(num_envs=8, seed=1)):
            with self.subTest(arguments=arguments):
                aot.require_ready(self.root, '/bddl/suite/first.bddl', ready_session=session, **arguments)
        with self.assertRaises(MissingCompiledProduct):
            aot.require_ready(self.root/'other', '/bddl/suite/first.bddl', num_envs=8,
                              seed=0, ready_session=session)
        self.ready(session)['tasks'].clear()
        session.record['tasks'].clear()
        session.info['root'] = 'other'
        self.ready(session)
        with session:
            self.ready(session)
        session.close()
        with self.assertRaisesRegex(MissingCompiledProduct, 'closed'):
            self.ready(session)

    def test_no_compilation_or_cache_writes(self):
        before = {p.relative_to(self.root): (p.read_bytes(), p.stat().st_mtime_ns)
                  for p in self.root.rglob('*') if p.is_file()}
        with compilation_scope(False):
            self.ready(self.qualify(), return_record=False)
            self.assertFalse(compilation_allowed())
            with self.assertRaises(MissingCompiledProduct):
                require_compilation('any GPU product', 'absent')
        after = {p.relative_to(self.root): (p.read_bytes(), p.stat().st_mtime_ns)
                 for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_batch_passes_the_compatibility_handle_before_gpu_construction(self):
        spec = importlib.util.spec_from_file_location('_ready_session_batch',
            Path(__file__).resolve().parents[1]/
            'libero/libero/gpu/host/adapters/environment/batch_env.py')
        batch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(batch)
        session = self.qualify()
        class StopBeforeGpu(RuntimeError):
            pass
        def ready(*args, **kwargs):
            self.assertFalse(compilation_allowed())
            self.assertIs(kwargs['ready_session'], session)
            self.assertFalse(kwargs['return_record'])
        with (patch.object(aot, 'require_ready', side_effect=ready) as check,
              patch.object(aot, 'scene_for', side_effect=StopBeforeGpu)):
            with self.assertRaises(StopBeforeGpu):
                batch.GpuBatchEnv('/bddl/suite/first.bddl', num_envs=8,
                    build_root=self.root, ready_session=session, physics_engine='native')
        check.assert_called_once()
        with patch.object(aot, 'require_ready') as check:
            with self.assertRaisesRegex(ValueError, 'requires allow_compile=False'):
                batch.GpuBatchEnv('/bddl/suite/first.bddl', num_envs=8,
                    build_root=self.root, ready_session=session, allow_compile=True, physics_engine='native')
        check.assert_not_called()


if __name__ == '__main__':
    unittest.main()
