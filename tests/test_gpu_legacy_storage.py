"""Offline maintenance of retired storage: leases, ownership and crash recovery."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from libero.gpu.host.build.legacy_storage import DiskCache, directory_bytes


class LegacyStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.environment = patch.dict(os.environ, {
            'LIBERO_GPU_CACHE_DIR': str(self.root), 'LIBERO_GPU_DISK_MIN_FREE_BYTES': '0',
            'LIBERO_GPU_DISK_MAX_BYTES': str(8 * 1024 ** 2),
            'LIBERO_GPU_RUNS_MAX_BYTES': str(8 * 1024 ** 2),
            'LIBERO_GPU_MODELS_MAX_BYTES': str(8 * 1024 ** 2),
            'LIBERO_GPU_SCENES_MAX_BYTES': str(8 * 1024 ** 2)})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.cache = DiskCache()

    def test_process_crash_releases_lease_and_reaps_only_owned_files(self):
        legacy = self.root / 'builds/environment-historical'
        legacy.mkdir(parents=True)
        (legacy / 'model.mjb').write_bytes(b'keep')
        code = ('from libero.gpu.host.build.legacy_storage import DiskCache; '
                'import os; c=DiskCache(); lease=c.new_run(); '
                '(lease.path / "partial").write_bytes(b"x" * 65536); '
                'print(lease.path, flush=True); os._exit(0)')
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
        path = Path(result.stdout.strip().splitlines()[-1])
        self.assertTrue(path.exists())
        self.cache.collect()
        self.assertFalse(path.exists())
        self.assertEqual((legacy / 'model.mjb').read_bytes(), b'keep')

    def test_other_process_cannot_reap_live_run(self):
        lease = self.cache.new_run()
        code = 'from libero.gpu.host.build.legacy_storage import DiskCache; DiskCache().collect()'
        subprocess.run([sys.executable, '-c', code], check=True, capture_output=True)
        self.assertTrue(lease.path.exists())
        lease.close()
        self.assertFalse(lease.path.exists())

    def test_registration_crash_cutpoints_recover_without_unregistered_runs(self):
        code = '''
import json, os, sys
from pathlib import Path
from libero.gpu.host.build.legacy_storage import DiskCache
c = DiskCache()
cut = sys.argv[1]
save = c._save
count = 0
def interrupted_save(catalog):
    global count
    count += 1
    if count == 1 and cut == 'before-intent':
        os._exit(0)
    save(catalog)
    if (count == 1 and cut == 'intent') or (count == 2 and cut == 'ready'):
        os._exit(0)
c._save = interrupted_save
sync = c._sync_directory
def interrupted_sync(path):
    sync(path)
    if path == c.root / 'runs' and cut == 'mkdir':
        os._exit(0)
c._sync_directory = interrupted_sync
replace = os.replace
def interrupted_replace(source, target):
    if Path(source).name.startswith('.libero-storage-'):
        if cut == 'partial-marker':
            Path(source).write_text('{')
            os._exit(0)
        replace(source, target)
        if cut == 'marker':
            os._exit(0)
    else:
        replace(source, target)
os.replace = interrupted_replace
c.new_run()
raise AssertionError('crash cutpoint not reached')
'''
        legacy = self.cache.root / 'runs/environment-unregistered-legacy'
        legacy.mkdir(parents=True)
        (legacy / 'keep').write_bytes(b'unregistered')
        for cut in ('before-intent', 'intent', 'mkdir', 'partial-marker', 'marker', 'ready'):
            with self.subTest(cut=cut):
                subprocess.run([sys.executable, '-c', code, cut], check=True, capture_output=True, text=True)
                self.assertEqual(self.cache.collect(), {})
                self.assertEqual(list(legacy.parent.iterdir()), [legacy])
                self.assertEqual((legacy / 'keep').read_bytes(), b'unregistered')

    def test_damaged_legacy_marker_is_retained_without_poisoning_other_collection(self):
        path = self.cache.root / 'runs/environment-damaged-legacy'
        path.mkdir(parents=True)
        (path / '.libero-storage.json').write_text('{')
        (path / 'keep').write_bytes(b'unknown ownership')
        with self.cache._catalog() as catalog:
            catalog[str(path)] = dict(kind='runs', touched=0)
        ordinary = self.cache.new_run()
        ordinary.close()
        rows = self.cache.collect()
        self.assertFalse(ordinary.path.exists())
        self.assertEqual((path / 'keep').read_bytes(), b'unknown ownership')
        self.assertIn('ownership', rows[str(path)]['collection_error'])

    def test_interrupted_removal_resumes_after_marker_was_deleted(self):
        code = '''
import os, shutil
from libero.gpu.host.build.legacy_storage import DiskCache
c = DiskCache()
lease = c.new_run()
(lease.path / 'large-partial').write_bytes(b'x' * 65536)
def interrupted_remove(path):
    (path / '.libero-storage.json').unlink()
    os._exit(0)
shutil.rmtree = interrupted_remove
print(lease.path, flush=True)
lease.close()
'''
        result = subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)
        path = Path(result.stdout.strip().splitlines()[-1])
        self.assertTrue((path / 'large-partial').exists())
        self.assertFalse((path / '.libero-storage.json').exists())
        self.assertEqual(self.cache.collect(), {})
        self.assertFalse(path.exists())

    def test_removal_journal_does_not_authorize_a_replacement_directory(self):
        lease = self.cache.new_run()
        path = lease.path
        with self.cache._catalog() as catalog:
            row = catalog[str(path)]
            info = path.stat()
            row.update(state='removing', removal_identity=[info.st_dev, info.st_ino + 1])
        (path / '.libero-storage.json').unlink()
        (path / 'keep').write_bytes(b'replacement')
        lease.close()
        self.assertTrue((path / 'keep').exists())
        self.assertIn('ownership', self.cache.collect()[str(path)]['collection_error'])

    def test_foreign_marker_or_symlink_never_grants_collection_rights(self):
        lease = self.cache.new_run()
        marker = lease.path / '.libero-storage.json'
        marker.write_text(json.dumps(dict(schema='foreign-cache')))
        lease.close()
        self.assertTrue(lease.path.exists())
        target = self.root / 'foreign'
        target.mkdir()
        (target / 'keep').write_bytes(b'foreign data')
        link = self.cache.root / 'runs/environment-symlink'
        link.symlink_to(target, target_is_directory=True)
        with self.cache._catalog() as catalog:
            catalog[str(link)] = dict(kind='runs', touched=0)
        rows = self.cache.collect()
        self.assertIn('symlink', rows[str(link)]['collection_error'])
        self.assertEqual((target / 'keep').read_bytes(), b'foreign data')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            with self.cache._catalog() as catalog:
                self.cache._register(catalog, link, 'models')
        self.assertFalse((target / '.libero-storage.json').exists())

    def test_preflight_hardlinks_do_not_double_charge_shared_bytes(self):
        lease = self.cache.new_run()
        source = lease.path / 'a'
        source.write_bytes(b'x' * 65536)
        before = directory_bytes(lease.path)
        os.link(source, lease.path / 'b')
        self.assertEqual(directory_bytes(lease.path), before)
        lease.close()

    def test_prepared_scenes_leave_legacy_collection_without_file_changes(self):
        folder = self.root / 'project/mjwarp/models/offline-input'
        folder.mkdir(parents=True)
        (folder / 'model.mjb').write_bytes(b'fixed offline model')
        (folder / '.libero-storage.json').write_text('legacy marker')
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in folder.iterdir()}
        with self.cache._catalog() as catalog:
            catalog[str(folder)] = dict(kind='scenes', touched=0)
        self.cache.max_bytes = 0
        self.assertEqual(self.cache.collect(), {})
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in folder.iterdir()})


    @unittest.skipUnless(importlib.util.find_spec('mujoco'), 'native model test requires MuJoCo')
    def test_shared_model_survives_other_process_close(self):
        import mujoco
        import numpy as np
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=".1"/></worldbody></mujoco>')
        raw = np.empty(mujoco.mj_sizeModel(model), np.uint8)
        mujoco.mj_saveModel(model, None, raw)
        record, lease = self.cache.model(model, hashlib.sha256(raw).hexdigest())
        code = ('import sys,mujoco; from libero.gpu.host.build.legacy_storage import DiskCache; '
                'c=DiskCache(); r,l=c.model(mujoco.MjModel.from_binary_path(sys.argv[1]),sys.argv[2]); '
                'print("ready",flush=True); sys.stdin.readline(); l.close()')
        child = subprocess.Popen([sys.executable, '-c', code, record['path'], record['sha256']],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(), 'ready')
            lease.close()
            self.cache.max_bytes = 100
            self.cache.collect()
            self.assertTrue(Path(record['path']).is_file())
            child.communicate('\n', timeout=15)
            self.assertEqual(child.returncode, 0)
            self.cache.collect()
            self.assertFalse(Path(record['path']).exists())
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate()
            lease.close()

if __name__ == '__main__':
    unittest.main()
