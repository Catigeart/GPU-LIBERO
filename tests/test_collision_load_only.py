"""Collision loader contracts without CUDA compilation or a GPU context."""
from contextlib import ExitStack, nullcontext
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from libero.gpu.host.build.compilation import compilation_scope, MissingCompiledProduct
from libero.gpu.host.runtime.collision import native
from libero.gpu.host.runtime.core import product_paths


class CollisionLoadOnlyTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.dev = SimpleNamespace(is_cuda=True, arch=80)
        size = native.scalar.itemsize

        def abi(out):
            out[:] = [size, 4, 4, 13 * size, 0, size, 4 * size]
            return 7

        def resources(values, attrs):
            values[:] = [101, 202, 16, 128, 32]
            attrs[:] = [80, 80, 40, 256, 0, 12000, 12000, 64]
            return 0

        self.library = SimpleNamespace(_handle=123, native_private_abi=Mock(side_effect=abi),
            native_kernel_identity=Mock(side_effect=resources), launch_native=Mock(return_value=0))
        for owner, name, value in (
                (native, 'load_cdll', Mock(return_value=self.library)),
                (native.wp, 'ScopedDevice', lambda device: nullcontext()),
                (native.wp, 'get_device', lambda device: self.dev),
                (native, 'nvcc_path', lambda: str(self.root / 'absent-nvcc')),
                (native, '_native_source_bindings', lambda header: {}),
                (native, '_CACHE', {})):
            self.stack.enter_context(patch.object(owner, name, value))
        self.stack.enter_context(patch.dict(os.environ, {
            'P06C_NATIVE_BUILD_ROOT': str(self.root),
            'P06C_NVCC': str(self.root / 'absent-nvcc'),
            'P06C_CCBIN': str(self.root / 'absent-g++')}))
        self.stack.enter_context(compilation_scope(False))
        self.stack.enter_context(patch.object(native.subprocess, 'run',
            side_effect=AssertionError('runtime invoked an external tool')))
        self.stack.enter_context(patch.object(native, '_sha',
            side_effect=AssertionError('runtime read a compiler/artifact fingerprint')))

    def test_load_checks_abi_and_resources_without_provenance_io(self):
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('image/SDK read')), \
                patch.object(Path, 'read_text', side_effect=AssertionError('process map read')):
            loaded = native._bind_native_product(self.root, self.dev)
        self.assertIs(loaded['library'], self.library)
        self.assertEqual(loaded['execution_binding']['shared_bytes'], 128)
        self.assertEqual(loaded['execution_binding']['kernel_attributes']['max_threads_per_block'], 256)
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertEqual(len(self.library.launch_native.argtypes), 33)

    def test_abi_resource_and_export_errors_still_propagate(self):
        self.library.native_private_abi.side_effect = lambda out: 0
        with self.assertRaisesRegex(ValueError, 'ABI mismatch'):
            native._bind_native_product(self.root, self.dev)
        self.library.native_kernel_identity.assert_not_called()
        del self.library.native_private_abi
        with self.assertRaises(AttributeError):
            native._bind_native_product(self.root, self.dev)

    def test_cuda_resource_query_failure_is_not_accepted(self):
        self.library.native_kernel_identity.side_effect = lambda values, attrs: 98
        with self.assertRaisesRegex(RuntimeError, '98'):
            native._bind_native_product(self.root, self.dev)

    def test_product_cache_hit_creates_no_identity_or_build_report(self):
        def materialize(**kwargs):
            for path in kwargs['outputs'].values():
                path.write_bytes(b'cached product')
            kwargs['verify']()
            return {'compile_commands_executed': [], 'cache': {'hit': True}}

        with patch.object(native, 'current_product_cache', return_value=object()), \
                patch.object(native, 'receipted_artifact_build', side_effect=materialize):
            loaded = native._prepare_native(self.dev)
        self.assertEqual({p.name for p in Path(loaded['build_path']).iterdir()},
                         {'libcollision.so', 'collision.ptx'})
        self.assertEqual(loaded['product_build']['compile_commands_executed'], [])

    def legacy_product(self):
        header = (native.GPU_ROOT / 'cuda/common/vector.cuh').resolve()
        key = hashlib.sha256(json.dumps({'sources': {}, 'fma': True,
            'real_bits': native.scalar.bits, 'forced_include': str(header)}, sort_keys=True).encode()).hexdigest()
        target = self.root / ('native-' + key[:24] + '-sm80')
        target.mkdir()
        compiler = str((self.root / 'absent-nvcc').resolve())
        flags = [*native.scalar.native_flags(), '-I' + str(native.GPU_ROOT / 'cuda/common'),
            '-std=c++17', '-O2', '-arch=sm_80', '--fmad=true', '--ftz=false',
            '--prec-div=true', '--prec-sqrt=true', '-rdc=true', '-include', str(header),
            '-ccbin', str((self.root / 'absent-g++').resolve())]
        unit = str(native.MODULE / 'native_kernel.cu')
        commands = [[compiler, *flags, '-shared', '-Xcompiler', '-fPIC',
            '-Xlinker', '-rpath', '-Xlinker', str(Path(compiler).parent.parent / 'lib64'),
            unit, '-o', str(target / 'libcollision.so')],
            [compiler, *flags, '--ptx', unit, '-o', str(target / 'collision.ptx')]]
        # Old compiler/artifact proof fields need not be present or readable.
        record = dict(sources={}, exit_code=0, argv=commands, cwd=str(target))
        (target / 'BUILD.json').write_text(json.dumps(record))
        for name in ('libcollision.so', 'collision.ptx'):
            (target / name).write_bytes(b'cached product')
        return target, record

    def test_legacy_cache_reuses_recipe_without_compiler_or_log_checks(self):
        target, record = self.legacy_product()
        with patch.object(native, 'current_product_cache', return_value=None):
            loaded = native._prepare_native(self.dev)
            self.assertEqual(loaded['build_path'], str(target))
            self.assertIs(native._prepare_native(self.dev), loaded)
            native._CACHE.clear()
            record['exit_code'] = 1
            (target / 'BUILD.json').write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'source/status mismatch'):
                native._prepare_native(self.dev)

    def test_missing_product_cannot_start_runtime_compilation(self):
        with patch.object(native, 'current_product_cache', return_value=None):
            with self.assertRaises(MissingCompiledProduct):
                native._prepare_native(self.dev)

    def test_cache_retirement_preserves_new_owner(self):
        first = SimpleNamespace(path=self.root, retire_callbacks=[])
        second = SimpleNamespace(path=self.root, retire_callbacks=[])
        older, newer = object(), object()
        with patch.object(product_paths, 'current_build_lifetime', return_value=first):
            native._store_native_cache('key', older, self.root / 'one')
        with patch.object(product_paths, 'current_build_lifetime', return_value=second):
            native._store_native_cache('key', newer, self.root / 'two')
        first.retire_callbacks[0]()
        self.assertIs(native._CACHE['key'], newer)
        second.retire_callbacks[0]()
        self.assertNotIn('key', native._CACHE)

    def test_launch_propagates_error_and_rejects_other_device(self):
        names = ('call_g1 call_g2 call_type1 call_type2 call_margin call_count active xpos xmat '
                 'geom_size geom_rbound geom_dataid mesh_vert mesh_vertadr mesh_vertnum '
                 'mesh_graphadr mesh_graph native_raw native_count native_status native_diag '
                 'native_frame0 native_framewritten').split()
        ws = SimpleNamespace(**{name: SimpleNamespace(ptr=1234) for name in names},
            device=self.dev, native={'library': self.library}, nworld=2, pairs=32,
            ngeom=10, nmesh=1, nvert=8, ngraph=16, mpr_iterations=50,
            mpr_tolerance=1e-6, multiccd=1)
        stream = SimpleNamespace(device=self.dev, cuda_stream=100)
        self.library.launch_native.return_value = 9
        with self.assertRaisesRegex(RuntimeError, 'launch error 9'):
            native.launch_native(ws, stream)
        with patch.object(native.wp, 'get_device', side_effect=lambda device: device):
            stream.device = 'another-device'
            with self.assertRaisesRegex(ValueError, 'device mismatch'):
                native.launch_native(ws, stream)


if __name__ == '__main__':
    unittest.main()
