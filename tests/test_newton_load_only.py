"""Newton offline/load-only boundary and actual-library resource contracts."""
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from libero.gpu.host.build import newton as build
from libero.gpu.host.build.compilation import compilation_scope, MissingCompiledProduct
from libero.gpu.host.runtime.newton import native_product as native


class NativeProductTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'kernel.so'
        self.path.write_bytes(b'fake CUDA image')
        self.values = [1, 24, 128, 4096, 80, 60, 0, 1024, 256, 49152, 1024]
        def query(out, count):
            self.assertEqual(count, 11)
            for i, value in enumerate(self.values): out[i] = value
            return 0
        self.library = NS(_name=str(self.path), newton_kernel_info=Mock(side_effect=query),
                          launch_persistent=Mock(), launch_entry=Mock())
        self.options = dict(parameter_bytes=24, block_threads=128, shared_bytes=4096,
                            architecture=80, combined_entry=True)

    def load(self):
        with patch.object(native, 'load_cdll', return_value=self.library), \
                patch.object(Path, 'read_bytes', side_effect=AssertionError('binary hashing')):
            return native.load_product(self.path, **self.options)

    def test_query_once_and_reuse_without_files_or_tools(self):
        library, resources = self.load()
        core = NS(library=library, _resource_library=library, _resource_info=resources,
                  manifest=dict(library_file=str(self.path), shared_bytes=4096))
        self.path.unlink()  # The retained loaded image owns its kernel.
        self.assertEqual(native.retained_resources(core, shared_limit_bytes=49152)['total_shared_bytes'], 5120)
        self.assertEqual(native.retained_resources(core, shared_limit_bytes=8192)['shared_limit_bytes'], 8192)
        self.library.newton_kernel_info.assert_called_once()
        core.library = NS(_name=str(self.path))
        with self.assertRaisesRegex(ValueError, 'owner replaced'):
            native.retained_resources(core, shared_limit_bytes=49152)

    def test_parameter_precision_arch_and_launch_mismatch(self):
        for index in range(5):
            with self.subTest(field=index):
                self.values[index] += 1
                with self.assertRaisesRegex(ValueError, 'ABI, launch configuration or architecture'):
                    self.load()
                self.values[index] -= 1

    def test_query_failure_and_missing_export(self):
        self.library.newton_kernel_info.side_effect = lambda *args: 999
        with self.assertRaisesRegex(RuntimeError, 'query failed: 999'): self.load()
        del self.library.newton_kernel_info
        with self.assertRaisesRegex(MissingCompiledProduct, 'query ABI missing'): self.load()

    def test_capacity_threads_and_stricter_policy(self):
        self.values[7] = 49152
        with self.assertRaisesRegex(ValueError, 'shared memory exceeds'): self.load()
        self.values[7], self.values[8] = 1024, 64
        with self.assertRaisesRegex(ValueError, 'block size exceeds'): self.load()
        self.values[8], self.values[9] = 256, 1024
        with self.assertRaisesRegex(ValueError, 'shared memory exceeds'): self.load()


class CatalogueProductTests(unittest.TestCase):
    def test_declared_instance_loads_without_source_sdk_or_compiler_access(self):
        from libero.gpu.host.runtime.newton import catalogue
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'cmake-control/sm80-dev/native_newton_nv21.so'
            path.parent.mkdir(parents=True)
            path.write_bytes(b'prebuilt CUDA')
            config = dict(nv=21,nM=86,rows=30032,contacts=5000,maxiter=100,
                block_threads=128,precision_bits=64,architecture=80,warmstart=True,
                use_dx=True,use_blas=True,phase_profiling=False,
                parameter_bytes=2040,shared_bytes=19232)
            values = (21,86,30032,5000,100,128,64,80,1,1,1,0)
            lib = NS(native_newton_configuration=Mock(side_effect=lambda index:values[index]))
            with patch.dict(os.environ, {'LIBERO_GPU_BUILD_MODE':'dev'}), \
                    patch.object(catalogue,'load_product',return_value=(lib,{})) as load, \
                    patch.object(Path,'read_text',side_effect=AssertionError('source/SDK read')):
                result = catalogue.load(config,build_root=root,mathdx_root='/absent/sdk')
                self.assertIs(result['library'],lib)
                self.assertFalse(result['metadata']['source_generation'])
                self.assertEqual(load.call_args.args[0],path)
                with self.assertRaisesRegex(ValueError,'configuration differs'):
                    catalogue.load(dict(config,maxiter=50),build_root=root,mathdx_root='/absent/sdk')

    def test_missing_instance_requires_offline_build(self):
        from libero.gpu.host.runtime.newton import catalogue
        with TemporaryDirectory() as directory, \
                patch.dict(os.environ,{'LIBERO_GPU_BUILD_MODE':'dev'}), \
                patch.object(catalogue,'load_product') as load:
            with self.assertRaisesRegex(MissingCompiledProduct,'libero.gpu.build'):
                catalogue.load(dict(nv=21,architecture=80),build_root=directory,mathdx_root=None)
            load.assert_not_called()


class ProductSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = dict(precision_bits=64, architecture=80, nv=21, nM=99, rows=128,
            contacts=20, nworld=2, maxiter=100, warmstart=True, block_threads=128,
            shared_bytes=4096, parameter_bytes=24, combined_entry=True, use_dx=True,
            use_blas=True, dx_bytes=1024, hessian_k=16, newton_islands='off', phase_profiling=False)
        self.args = dict(template='bound-layout-and-schedule', headers=[build.GPU_ROOT/'cuda/newton/math.cuh'],
            config=self.config, build_root=self.root/'bank', runtime_root=self.root/'owner')
        self.inputs = dict(units=[], dependencies=[], generator=[], project=[], toolchain=dict(components=[]))
        self.compile_calls = []
        self.prepare_calls = 0
        self.prepared_input_version = 0
        def prepare(template, headers, config, root, sdk):
            self.prepare_calls += 1
            target = self.root/'compile'/'newton.so'
            target.parent.mkdir(exist_ok=True)
            inputs = dict(self.inputs, version=self.prepared_input_version)
            return target, [['fake-nvcc', '-o', str(target)]], inputs, dict(source='offline-only.cu')
        def compile(command, **kwargs):
            self.compile_calls.append(command)
            Path(command[-1]).write_bytes(b'compiled library')
            return NS(returncode=0, stdout='', stderr='')
        self.prepare_patch = patch.object(build, '_prepare_build', side_effect=prepare)
        self.compiler_patch = patch.object(build.subprocess, 'run', side_effect=compile)
        self.validate_patch = patch.object(build, '_validate_inputs')
        self.load_patch = patch.object(native, 'load_product', side_effect=lambda path, **kw:
            (NS(_name=str(path)), dict(library_file=str(path))))
        for context in (self.prepare_patch, self.compiler_patch, self.validate_patch, self.load_patch):
            context.start(); self.addCleanup(context.stop)

    def prepare(self):
        with compilation_scope(True): return build.obtain_product(**self.args)

    def test_reuse_without_sdk_source_compiler_or_runtime_locks(self):
        self.prepare()
        self.assertEqual(len(self.compile_calls), 1)
        from libero.gpu.host.runtime.core import product_paths
        with compilation_scope(False), \
                patch.object(build, '_prepare_build', side_effect=AssertionError('offline preparation')), \
                patch.object(build, 'nvcc_path', side_effect=AssertionError('compiler lookup')), \
                patch.object(product_paths, 'build_unit_lock', side_effect=AssertionError('runtime build lock')):
            result = build.obtain_product(**dict(self.args, mathdx_root='/missing/SDK'))
        self.assertEqual(result['metadata']['compile_commands_executed'], [])
        self.assertNotIn('MANIFEST.json', [p.name for p in (self.root/'owner').iterdir()])
        self.assertEqual(self.prepare_calls, 1)
        self.assertEqual(len(self.compile_calls), 1)

    def test_dimensions_layout_precision_and_algorithm_choose_distinct_products(self):
        self.prepare()
        variants = [dict(self.args, template='different layout')]
        for key, value in dict(precision_bits=32, architecture=90, rows=256,
                               maxiter=50, warmstart=False, combined_entry=False, block_threads=64,
                               newton_islands='exact', phase_profiling=True).items():
            variants.append(dict(self.args, config=dict(self.config, **{key: value})))
        with compilation_scope(False):
            for args in variants:
                with self.subTest(args=args['config']), self.assertRaises(MissingCompiledProduct):
                    build.obtain_product(**args)
        self.assertEqual(len(self.compile_calls), 1)

    def test_world_count_reuses_the_same_code_but_source_schedule_still_matters(self):
        self.prepare()
        with compilation_scope(False):
            for worlds in (1, 8, 4096):
                result = build.obtain_product(**dict(self.args,
                    config=dict(self.config, nworld=worlds)))
                self.assertEqual(result['metadata']['compile_commands_executed'], [])
            with self.assertRaises(MissingCompiledProduct):
                build.obtain_product(**dict(self.args, template='different world-dependent code'))
        self.assertEqual(len(self.compile_calls), 1)

    def test_previous_world_count_product_remains_load_only_compatible(self):
        from libero.gpu.host.build.product_cache import product_cache_scope
        product = self.prepare()
        folder = Path(product['metadata']['product_build']['cache']['directory'])
        receipt = json.loads((folder/'receipt.json').read_text())
        identity = receipt['identity']['build']
        identity['config']['nworld'] = 2
        artifacts = {name:(folder/name).read_bytes() for name in ('newton.so','product.json')}
        description = json.loads(artifacts['product.json'])
        description['selection']['config']['nworld'] = 2
        artifacts['product.json'] = json.dumps(description).encode()
        with compilation_scope(True), product_cache_scope(self.args['build_root']/'products') as cache:
            cache.obtain_artifacts(identity, ('launch_persistent','newton_kernel_info','launch_entry'),
                names=('newton.so','product.json'), build=lambda:artifacts,
                validate=lambda:None, exports=lambda artifacts:None)
        (folder/'receipt.json').unlink()
        with compilation_scope(False), patch.object(build, '_prepare_build',
                side_effect=AssertionError('runtime compilation preparation')):
            result = build.obtain_product(**self.args)
        self.assertEqual(result['metadata']['compile_commands_executed'], [])
        self.assertEqual(len(self.compile_calls), 1)

    def test_offline_input_change_rebuilds_same_runtime_selection(self):
        self.prepare(); self.prepare()
        self.assertEqual(len(self.compile_calls), 1)
        self.prepared_input_version = 1
        self.prepare()
        self.assertEqual(len(self.compile_calls), 2)
        with compilation_scope(False): build.obtain_product(**self.args)

    def test_failed_query_never_publishes_a_completed_product(self):
        with patch.object(native, 'load_product', side_effect=ValueError('bad ABI')):
            with self.assertRaisesRegex(ValueError, 'bad ABI'): self.prepare()
        with compilation_scope(False), self.assertRaises(MissingCompiledProduct):
            build.obtain_product(**self.args)

    def test_failed_compile_never_publishes_a_completed_product(self):
        with patch.object(build.subprocess, 'run', return_value=NS(returncode=1, stdout='', stderr='compile error')):
            with self.assertRaisesRegex(RuntimeError, 'compilation failed'): self.prepare()
        with compilation_scope(False), self.assertRaises(MissingCompiledProduct):
            build.obtain_product(**self.args)

    def test_publication_input_failure_does_not_replace_completed_product(self):
        self.prepare()
        self.prepared_input_version = 1
        with patch.object(build, '_validate_inputs', side_effect=[None, ValueError('changed input')]):
            with self.assertRaisesRegex(ValueError, 'changed input'): self.prepare()
        with compilation_scope(False):
            result = build.obtain_product(**self.args)
        self.assertTrue(result['metadata']['product_build']['cache']['hit'])

    def test_existing_cache_budget_is_honored(self):
        from libero.gpu.host.build.product_cache import require_product_persistence
        with patch.dict(os.environ, {'LIBERO_GPU_PRODUCT_CACHE_MAX_BYTES': '1'}), \
                require_product_persistence(), self.assertRaises(OSError):
            self.prepare()
        with compilation_scope(False), self.assertRaises(MissingCompiledProduct):
            build.obtain_product(**self.args)


class PreparedInstanceTests(unittest.TestCase):
    def test_runtime_checks_dimensions_presence_and_offline_checks_freshness(self):
        from libero.gpu.host.build import aot
        with TemporaryDirectory() as directory:
            root = Path(directory)
            package = root/'instance'; package.mkdir()
            (package/'candidate.json').write_text(json.dumps(dict(nv=21, nM=99)))
            source = package/'workspace.py'; source.write_text('# original\n')
            files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in package.iterdir()}
            (root/'NEWTON_INSTANCES.json').write_text(json.dumps({'21:99': dict(directory=str(package), files=files)}))
            source.write_text('# comment only\n')
            with compilation_scope(False), patch.object(aot, '_sha', side_effect=AssertionError('runtime hashing')):
                self.assertEqual(aot.offline_newton_source(root, 21, 99)[1]['nv'], 21)
                (package/'candidate.json').write_text(json.dumps(dict(nv=43, nM=99)))
                with self.assertRaisesRegex(ValueError, 'dimensions differ'):
                    aot.offline_newton_source(root, 21, 99)
                (package/'candidate.json').write_text(json.dumps(dict(nv=21, nM=99)))
                source.unlink()
                with self.assertRaises(MissingCompiledProduct): aot.offline_newton_source(root, 21, 99)
            source.write_text('# changed\n')
            with compilation_scope(True), self.assertRaisesRegex(ValueError, 'instance changed'):
                aot.offline_newton_source(root, 21, 99)


if __name__ == '__main__':
    unittest.main()
