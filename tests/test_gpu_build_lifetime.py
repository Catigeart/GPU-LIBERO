"""Temporary runtime owners and retryable cleanup; no legacy cache or CUDA required."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from libero.gpu.host.runtime.core.build_lifetime import managed_build, cold_build_scope
from libero.gpu.host.runtime.core.product_paths import product_build, current_build_lifetime


def bundle():
    return SimpleNamespace(close=lambda: None)


class BuildLifetimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = patch.dict(os.environ, {'LIBERO_GPU_CACHE_DIR': str(self.root)})
        environment.start()
        self.addCleanup(environment.stop)

    def test_scope_exit_keeps_live_runtime_and_close_reclaims_it(self):
        @managed_build
        def prepare():
            (current_build_lifetime().path / 'evidence.json').write_text('{}')
            return bundle()
        with product_build() as path:
            runtime = prepare()
        self.assertTrue((path / 'evidence.json').is_file())
        runtime.close()
        runtime.close()
        self.assertFalse(path.exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_failed_constructor_and_empty_scope_reclaim_private_files(self):
        @managed_build
        def fail():
            (current_build_lifetime().path / 'partial.mjb').write_bytes(b'x' * 65536)
            raise ValueError('constructor failure')
        with self.assertRaisesRegex(ValueError, 'constructor failure'):
            fail()
        with product_build() as path:
            pass
        self.assertFalse(path.exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_failed_gpu_close_retains_files_for_retry(self):
        calls = []
        @managed_build
        def prepare():
            def close():
                calls.append(True)
                if len(calls) == 1:
                    raise RuntimeError('still live')
            return SimpleNamespace(close=close)
        runtime = prepare()
        with self.assertRaisesRegex(RuntimeError, 'still live'):
            runtime.close()
        self.assertTrue(runtime.disk_build_root.exists())
        runtime.close()
        self.assertFalse(runtime.disk_build_root.exists())

    def test_temporary_cleanup_retry_does_not_repeat_successful_gpu_close(self):
        calls = []
        @managed_build
        def prepare():
            return SimpleNamespace(close=lambda: calls.append('gpu'))
        runtime = prepare()
        from libero.gpu.host.runtime.core import build_lifetime
        original = build_lifetime.TemporaryDirectory.cleanup
        fail = [True]
        def cleanup(temporary):
            if fail[0]:
                fail[0] = False
                raise OSError('injected temporary cleanup failure')
            return original(temporary)
        with patch.object(build_lifetime.TemporaryDirectory, 'cleanup', cleanup):
            with self.assertRaisesRegex(OSError, 'injected'):
                runtime.close()
            self.assertTrue(runtime.disk_build_root.exists())
            runtime.close()
        self.assertEqual(calls, ['gpu'])
        self.assertFalse(runtime.disk_build_root.exists())

    def test_retirement_failure_keeps_only_failed_callbacks_and_allows_retry(self):
        calls, held = [], []
        @managed_build
        def prepare():
            owner = current_build_lifetime()
            held.append(owner)
            def fail_once():
                calls.append('retry')
                if calls.count('retry') == 1:
                    raise OSError('retirement failure')
            owner.retire_callbacks.extend((fail_once, lambda: calls.append('other')))
            return SimpleNamespace(close=lambda: calls.append('gpu'))
        runtime = prepare()
        with self.assertRaisesRegex(OSError, 'retirement failure'):
            runtime.close()
        self.assertEqual(calls, ['gpu', 'retry', 'other'])
        self.assertTrue(runtime.disk_build_root.exists())
        runtime.close()
        self.assertEqual(calls, ['gpu', 'retry', 'other', 'retry'])
        self.assertFalse(runtime.disk_build_root.exists())

    def test_replacement_does_not_delete_fresh_generation(self):
        @managed_build
        def prepare():
            return bundle()
        old = prepare()
        @cold_build_scope
        def replace():
            path = current_build_lifetime().path
            (path / 'parameter.json').write_text('{}')
            old.close()
            fresh = prepare()
            self.assertEqual(fresh.disk_build_root, path)
            return fresh
        fresh = replace()
        self.assertFalse(old.disk_build_root.exists())
        self.assertTrue((fresh.disk_build_root / 'parameter.json').is_file())
        fresh.close()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_runtime_scopes_do_not_register_storage_or_create_model_copies(self):
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        @managed_build
        def prepare():
            return bundle()
        held = prepare()
        for _ in range(12):
            other = prepare()
            self.assertNotEqual(held.disk_build_root, other.disk_build_root)
            other.close()
        held.close()
        after = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

if __name__ == '__main__':
    unittest.main()
