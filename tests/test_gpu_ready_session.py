"""CPU-only readiness reuse and its explicit immutable-tree boundary."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

from libero.gpu.host.build import aot
from libero.gpu.host.build.compilation import (MissingCompiledProduct,
    compilation_allowed, compilation_scope, require_compilation)
from libero.gpu.host.build.ready_session import ReadySession


# Reuse the existing independent file-backed bank fixture, without loading GPU
# modules, constructing an environment or invoking unittest's CLI entry point.
_spec = importlib.util.spec_from_file_location('_ready_session_fixture',
    Path(__file__).with_name('test_gpu_aot_build.py'))
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)


class ReadySessionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base/'bank'
        self.root.mkdir()
        self.source = self.base/'runtime.py'
        self.source.write_bytes(b'initial runtime')
        contract, products, native, warp = _fixture.observation_fixture(aot, self.root)
        self.contract = contract
        row = dict(key='suite/first.bddl', source_identity='source', num_envs=8, seed=0,
            model_sha256='model', native_artifacts=native, warp_required=sorted(warp),
            warp_artifacts=warp, observation_products=products)
        second = deepcopy(row)
        second['key'] = 'suite/second.bddl'
        self.record = dict(status='complete', num_envs=8, seed=0, source_identity='source',
            observation_contract=contract, tasks={row['key']: row, second['key']: second},
            artifacts={**native, **warp})
        self.manifest = self.root/'AOT_BUILD.json'
        self.save()
        self.addCleanup(patch.stopall)
        patch.object(aot, 'source_identity', return_value='source').start()
        patch.object(aot, 'observation_product_contract', return_value=contract).start()
        patch.object(aot, '_source_identity_files',
            side_effect=lambda **kwargs: {p.name: p for p in sorted(self.base.glob('*.py'))}).start()

    def save(self):
        self.manifest.write_text(json.dumps(self.record))

    def qualify(self):
        return aot.qualify_ready_session(self.root, num_envs=8, seed=0)

    def ready(self, session=None, task='first', **kwargs):
        return aot.require_ready(self.root, '/bddl/suite/'+task+'.bddl',
            num_envs=8, seed=0, ready_session=session, **kwargs)

    def test_session_hashes_every_artifact_once_then_reuses_without_hashing(self):
        with patch.object(aot, '_sha', wraps=aot._sha) as sha:
            session = self.qualify()
            self.assertEqual(sha.call_count, len(self.record['artifacts']))
        with (patch.object(aot, '_sha', side_effect=AssertionError('repeated content hash')),
                patch.object(aot, 'source_identity', side_effect=AssertionError('repeated source hash')),
                patch.object(aot, 'observation_product_contract', side_effect=AssertionError('repeated contract hash'))):
            self.assertIsNone(self.ready(session, return_record=False))
            self.assertEqual(self.ready(session, task='second')['tasks'], self.record['tasks'])
        self.assertEqual(session.info['artifacts'], len(self.record['artifacts']))
        self.assertEqual(session.info['tasks'], 2)

    def test_default_independent_readiness_still_rehashes_the_whole_bank(self):
        with patch.object(aot, '_sha', wraps=aot._sha) as sha:
            self.ready()
            self.ready(task='second')
        self.assertEqual(sha.call_count, 2*len(self.record['artifacts']))

    def test_qualification_checks_all_tasks_and_the_full_artifact_table(self):
        for name in ('key', 'source_identity', 'num_envs', 'seed'):
            with self.subTest(stale=name):
                original = self.record['tasks']['suite/second.bddl'][name]
                self.record['tasks']['suite/second.bddl'][name] = 'stale'
                self.save()
                with self.assertRaisesRegex(MissingCompiledProduct, 'Task qualification differs'):
                    self.qualify()
                self.record['tasks']['suite/second.bddl'][name] = original
        self.record['tasks']['suite/second.bddl']['observation_products'] = {}
        self.save()
        with self.assertRaisesRegex(MissingCompiledProduct, 'Observation products'):
            self.qualify()

    def test_each_ordinary_artifact_change_permanently_invalidates_the_session(self):
        target = self.root/next(iter(self.record['artifacts']))
        original = target.read_bytes()
        for change in ('same-size-restored-mtime', 'replace', 'delete'):
            with self.subTest(change=change):
                target.write_bytes(original)
                session = self.qualify()
                prior = target.stat()
                if change == 'same-size-restored-mtime':
                    target.write_bytes(b'x'*len(original))
                    os.utime(target, ns=(prior.st_atime_ns, prior.st_mtime_ns))
                elif change == 'replace':
                    replacement = target.with_suffix('.replacement')
                    replacement.write_bytes(original)
                    replacement.replace(target)
                else:
                    target.unlink()
                with self.assertRaisesRegex(MissingCompiledProduct, 'Ready session is invalid'):
                    self.ready(session)
                target.write_bytes(original)
                with self.assertRaisesRegex(MissingCompiledProduct, 'Ready session is invalid'):
                    self.ready(session)

    def test_source_set_rewrite_and_manifest_change_are_not_cached_as_ready(self):
        for kind in ('source', 'new-source', 'manifest'):
            with self.subTest(kind=kind):
                session = self.qualify()
                if kind == 'source':
                    self.source.write_bytes(b'updated runtime')
                elif kind == 'new-source':
                    (self.base/'added.py').write_bytes(b'new runtime')
                else:
                    self.manifest.write_text(self.manifest.read_text()+'\n')
                with self.assertRaises(MissingCompiledProduct):
                    self.ready(session)

    def test_foreign_request_cannot_broaden_the_qualified_scope(self):
        session = self.qualify()
        for arguments in (dict(num_envs=2, seed=0), dict(num_envs=8, seed=1)):
            with self.subTest(arguments=arguments), self.assertRaises(MissingCompiledProduct):
                aot.require_ready(self.root, '/bddl/suite/first.bddl', ready_session=session, **arguments)
        with self.assertRaises(MissingCompiledProduct):
            aot.require_ready(self.base, '/bddl/suite/first.bddl', num_envs=8, seed=0,
                ready_session=session)
        for kwargs in (dict(task='absent'), dict(debug=True),
                       dict(observation_options=dict(use_camera_obs=True, camera_heights=64))):
            with self.subTest(kwargs=kwargs), self.assertRaises(MissingCompiledProduct):
                self.ready(session, **kwargs)
        self.ready(session)

    def test_records_are_detached_and_close_process_and_serialization_boundaries_hold(self):
        session = self.qualify()
        self.ready(session)['tasks'].clear()
        session.record['artifacts'].clear()
        session.info['root'] = 'other'
        self.assertEqual(len(self.ready(session)['tasks']), 2)
        with self.assertRaises(TypeError):
            pickle.dumps(session)
        with patch('libero.gpu.host.build.ready_session.os.getpid', return_value=os.getpid()+1):
            with self.assertRaisesRegex(MissingCompiledProduct, 'process boundary'):
                self.ready(session)
        with self.assertRaises(MissingCompiledProduct):
            self.ready(session)
        with self.qualify() as owned:
            self.ready(owned)
        with self.assertRaisesRegex(MissingCompiledProduct, 'closed'):
            self.ready(owned)

    def test_a_write_after_hashing_is_not_accepted_by_the_final_snapshot(self):
        target = self.root/next(iter(self.record['artifacts']))
        original_sha = aot._sha
        def race(path):
            digest = original_sha(path)
            if Path(path) == target:
                target.write_bytes(b'changed after successful hash')
            return digest
        with patch.object(aot, '_sha', side_effect=race), self.assertRaisesRegex(
                MissingCompiledProduct, 'changed during ready session qualification'):
            self.qualify()

    def test_a_source_write_during_identity_hashing_is_rejected(self):
        def race():
            self.source.write_bytes(b'updated during source hash')
            return 'source'
        with patch.object(aot, 'source_identity', side_effect=race), self.assertRaisesRegex(
                MissingCompiledProduct, 'changed during ready session qualification'):
            self.qualify()

    def test_artifact_paths_and_symlinks_are_rejected(self):
        baseline = deepcopy(self.record)
        for spelling in ('../outside.so', str(self.base/'outside.so')):
            self.record['artifacts'][spelling] = 'a'*64
            self.save()
            with self.subTest(path=spelling), self.assertRaisesRegex(ValueError, 'invalid unified artifact'):
                self.qualify()
            self.record = deepcopy(baseline)
        self.save()
        target = self.root/next(iter(self.record['artifacts']))
        moved = self.base/'outside.so'
        target.rename(moved)
        target.symlink_to(moved)
        with self.assertRaisesRegex(ValueError, 'non-symlink file'):
            self.qualify()
        target.unlink()
        moved.rename(target)
        session = self.qualify()
        directory = target.parent
        outside = self.base/'outside-directory'
        directory.rename(outside)
        directory.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(MissingCompiledProduct, 'non-symlink artifact directories'):
            self.ready(session)

    def test_qualification_has_no_compiler_authority_or_cache_writes(self):
        before = {p.relative_to(self.root): (p.read_bytes(), p.stat().st_mtime_ns)
                  for p in self.root.rglob('*') if p.is_file()}
        with compilation_scope(False):
            session = self.qualify()
            self.ready(session, return_record=False)
            self.assertFalse(compilation_allowed())
            with self.assertRaises(MissingCompiledProduct):
                require_compilation('any GPU product', 'absent')
        after = {p.relative_to(self.root): (p.read_bytes(), p.stat().st_mtime_ns)
                 for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        with self.assertRaises(TypeError):
            ReadySession(None)

    def test_batch_explicitly_passes_the_session_before_any_gpu_construction(self):
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
                    build_root=self.root, ready_session=session)
        check.assert_called_once()
        with patch.object(aot, 'require_ready') as check:
            with self.assertRaisesRegex(ValueError, 'requires allow_compile=False'):
                batch.GpuBatchEnv('/bddl/suite/first.bddl', num_envs=8,
                    build_root=self.root, ready_session=session, allow_compile=True)
        check.assert_not_called()


if __name__ == '__main__':
    unittest.main()
