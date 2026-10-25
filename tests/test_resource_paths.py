"""CPU-only resource compatibility tests; never import simulation or compile CUDA."""
import builtins
import contextlib
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(os.environ.get('LIBERO_TEST_SOURCE', str(Path(__file__).resolve().parents[1])))
WORK = Path(os.environ.get('LIBERO_TEST_TMP', str(Path(tempfile.gettempdir()) / 'libero-resource-path-tests')))
WORK.mkdir(parents=True, exist_ok=True)
os.environ['LIBERO_CONFIG_PATH'] = str(WORK / 'import-config')
sys.path.insert(0, str(SOURCE))
from libero import libero as api
from libero.libero import resources as r


def entry(name, payload):
    return {'path': name, 'size': len(payload), 'git_blob_sha1':
            hashlib.sha1(b'blob ' + str(len(payload)).encode() + b'\0' + payload).hexdigest()}


class ResourcesTest(unittest.TestCase):
    def setUp(self):
        self.case = tempfile.TemporaryDirectory(dir=WORK)
        self.root = Path(self.case.name)
        self.env = patch.dict(os.environ, {'LIBERO_RESOURCE_PATH': str(self.root / 'cache')})
        self.env.start()
    def tearDown(self):
        self.env.stop()
        self.case.cleanup()
    def manifest(self):
        data = {'folder/a.txt': b'original a', 'b.txt': b'original b'}
        manifest = {'format': 1, 'revision': 'fixture-revision', 'repository': 'fixture/repo',
                    'groups': {'bddl_files': [entry(k, v) for k, v in data.items()]}}
        path = self.root / 'manifest.json'
        path.write_text(json.dumps(manifest))
        return data, path
    def test_import_noninteractive_and_no_download(self):
        code = ('import builtins,urllib.request; '\
                'builtins.input=lambda *a: (_ for _ in ()).throw(AssertionError("input")); '\
                'urllib.request.urlopen=lambda *a,**k: (_ for _ in ()).throw(AssertionError("network")); '\
                'from libero.libero import get_libero_path; '\
                'print(get_libero_path("benchmark_root"))')
        env = dict(os.environ, LIBERO_CONFIG_PATH=str(self.root/'fresh-config'))
        completed = subprocess.run([sys.executable, '-c', code], env=env, cwd=SOURCE,
                                   input='', text=True, capture_output=True, timeout=10)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue((self.root/'fresh-config/config.yaml').is_file())
    def test_existing_config_preserved_and_external_assets_reused(self):
        custom = self.root/'external-assets'; custom.mkdir()
        config = self.root/'config.yaml'
        content = 'assets: '+str(custom)+'\ndatasets: '+str(self.root/'dataset')+'\ncustom_key: preserve\n'
        config.write_text(content)
        with patch.object(api, 'config_file', str(config)):
            self.assertEqual(api.get_libero_path('assets'), str(custom))
            self.assertEqual(api.get_libero_path('datasets'), str(self.root/'dataset'))
            self.assertEqual(config.read_text(), content)
            with self.assertRaises(AssertionError): api.get_libero_path('unknown')
    def test_missing_explicit_user_resource_is_actionable(self):
        config = self.root/'config.yaml'; config.write_text('assets: '+str(self.root/'absent')+'\n')
        with patch.object(api, 'config_file', str(config)):
            with self.assertRaisesRegex(FileNotFoundError, 'has not been changed'):
                api.get_libero_path('assets')
        self.assertEqual(config.read_text(), 'assets: '+str(self.root/'absent')+'\n')
    def test_complete_group_checksums_resumption_and_warm_no_network(self):
        data, path = self.manifest(); requests = []
        def urlopen(request, timeout):
            requests.append(request.full_url)
            name = 'folder/a.txt' if request.full_url.endswith('folder/a.txt') else 'b.txt'
            return contextlib.closing(io.BytesIO(data[name]))
        with patch.object(r, '_MANIFEST_PATH', path), patch.object(r.urllib.request, 'urlopen', urlopen):
            destination = Path(r.ensure_resource_group('bddl_files'))
            self.assertEqual(len(requests), 2)
            self.assertTrue((destination/'.libero-resources.json').is_file())
            self.assertEqual((destination/'folder/a.txt').read_bytes(), data['folder/a.txt'])
            self.assertEqual(r.ensure_resource_group('bddl_files'), str(destination))
            self.assertEqual(len(requests), 2)
            (destination/'b.txt').unlink()
            r.ensure_resource_group('bddl_files')
            self.assertEqual(len(requests), 3)
    def test_corrupt_download_fails_without_complete_marker(self):
        data, path = self.manifest()
        with patch.object(r, '_MANIFEST_PATH', path), patch.object(r.urllib.request, 'urlopen',
                lambda *a, **k: contextlib.closing(io.BytesIO(b'bad'))):
            with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
                r.ensure_resource_group('bddl_files')
            self.assertFalse((r.managed_resource_path('bddl_files')/'.libero-resources.json').exists())
            self.assertFalse(list(r.resource_root().rglob('.libero-download-*')))
    def test_offline_failure(self):
        _, path = self.manifest()
        with patch.object(r, '_MANIFEST_PATH', path), patch.object(r.urllib.request, 'urlopen',
                side_effect=AssertionError('should not download')):
            with self.assertRaisesRegex(FileNotFoundError, 'LIBERO_AUTO_DOWNLOAD'):
                r.ensure_resource_group('bddl_files', download=False)
    def test_path_traversal_refused(self):
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            r._download_file(self.root, entry('../outside', b'bad'),
                             {'repository':'fixture', 'revision':'fixture'}, 'bddl_files')
    def test_source_group_uses_existing_original_resources(self):
        package = self.root/'bundled/libero/libero'
        manifest = {'format': 1, 'revision': 'fixture', 'repository': 'fixture/repo',
                    'groups': {}}
        files = {}
        for key, folder in r._GROUP_FOLDERS.items():
            name = 'sample.bddl' if key == 'bddl_files' else 'sample.bin'
            payload = b'fixture resource'
            files[key] = package/folder/name
            files[key].parent.mkdir(parents=True, exist_ok=True)
            files[key].write_bytes(payload)
            manifest['groups'][key] = [entry(name, payload)]
        path = self.root/'bundled-manifest.json'
        path.write_text(json.dumps(manifest))
        with patch.object(api, '__file__', str(package/'__init__.py')), \
                patch.object(r, '_MANIFEST_PATH', path), \
                patch.object(r.urllib.request, 'urlopen', side_effect=AssertionError('network')):
            defaults = api.get_default_path_dict()
            for key, folder in r._GROUP_FOLDERS.items():
                self.assertEqual(defaults[key], str(package/folder))
            # Missing or truncated bundled resources must fall back per group.
            files['assets'].unlink()
            defaults = api.get_default_path_dict()
            self.assertEqual(defaults['assets'], str(self.root/'cache/assets'))
            self.assertEqual(defaults['init_states'], str(package/'init_files'))
            files['init_states'].write_bytes(b'partial')
            self.assertEqual(api.get_default_path_dict()['init_states'],
                             str(self.root/'cache/init_files'))
        self.assertFalse((self.root/'cache').exists())
    def test_source_only_install_chooses_cache_without_import_download(self):
        package = self.root/'source-only/libero/libero'; package.mkdir(parents=True)
        # A regular outer package prevents an editable install from overriding this fixture.
        shutil.copy2(SOURCE/'libero/__init__.py', package.parent/'__init__.py')
        for name in ['__init__.py','resources.py','upstream_resources.json']:
            shutil.copy2(SOURCE/'libero/libero'/name, package/name)
        # Partial assets/XML must not trick completeness into accepting the wheel.
        (package/'assets').mkdir(); (package/'assets/stub.xml').write_text('<mujoco/>')
        code = 'from libero.libero import get_default_path_dict; import json; print(json.dumps(get_default_path_dict()))'
        env = dict(os.environ, LIBERO_CONFIG_PATH=str(self.root/'source-only-config'),
                   PYTHONPATH=str(self.root/'source-only'))
        completed = subprocess.run([sys.executable,'-c',code],env=env,cwd=self.root,
                                   capture_output=True,text=True,timeout=10)
        self.assertEqual(completed.returncode,0,completed.stderr)
        defaults = json.loads(completed.stdout)
        self.assertEqual(defaults['assets'], str(self.root/'cache/assets'))
        self.assertEqual(defaults['benchmark_root'], str(package))
    def test_cli_all_groups_check_without_network(self):
        import yaml
        config_dir = self.root/"cli-config"
        config_dir.mkdir()
        mapping = {}
        for key in r._GROUP_FOLDERS:
            directory = self.root/("custom-"+key)
            directory.mkdir()
            mapping[key] = str(directory)
        (config_dir/"config.yaml").write_text(yaml.safe_dump(mapping))
        env = dict(os.environ, LIBERO_CONFIG_PATH=str(config_dir),
                   PYTHONPATH=str(SOURCE))
        completed = subprocess.run([sys.executable,"-m","libero.libero.resource_cli","--check"],
                                   env=env,cwd=SOURCE,capture_output=True,text=True,timeout=10)
        self.assertEqual(completed.returncode,0,completed.stderr)
        self.assertIn("bddl_files:",completed.stdout)

    def test_parallel_new_config_atomic_and_no_overwrite(self):
        config = self.root/'parallel-config'
        env = dict(os.environ, LIBERO_CONFIG_PATH=str(config))
        command = [sys.executable,'-c','from libero.libero import get_libero_path; print(get_libero_path("benchmark_root"))']
        procs = [subprocess.Popen(command,env=env,cwd=SOURCE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                 for _ in range(6)]
        for proc in procs:
            out, err = proc.communicate(timeout=10)
            self.assertEqual(proc.returncode,0,err)
        import yaml
        self.assertEqual(yaml.safe_load((config/'config.yaml').read_text())['benchmark_root'],
                         str(SOURCE/'libero/libero'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
