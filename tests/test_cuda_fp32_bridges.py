"""Stream-owned CUDA bridge bindings, without loading CUDA."""
import ctypes as ct
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from libero.gpu.host.backends.cuda_fp32 import bridges


class BridgeLifetimeTests(unittest.TestCase):
    def library(self):
        library = MagicMock()
        for name, structure in (('gather', bridges.Gather), ('observe', bridges.Observe),
                                ('scene', bridges.Scene)):
            getattr(library, 'flash_bridge_' + name + '_bytes').return_value = ct.sizeof(structure)
        library.flash_bridge_collision_reset.return_value = 0
        return library

    def test_same_stream_reuses_validated_bindings_and_resolved_path(self):
        stream = SimpleNamespace(device=SimpleNamespace(scope=nullcontext),cuda_stream=41)
        library = self.library()
        with patch.object(bridges.ct, 'CDLL', return_value=library) as load:
            first = bridges.Bridges('/products/cuda_bridges_fp32.so', stream)
            second = bridges.Bridges('/products/../products/cuda_bridges_fp32.so', stream)
        self.assertIs(first.library, second.library)
        load.assert_called_once_with('/products/cuda_bridges_fp32.so')
        library.flash_bridge_source_id.assert_not_called()
        for name in ('gather', 'observe', 'scene'):
            getattr(library, 'flash_bridge_' + name + '_bytes').assert_called_once_with()
        second.call('collision_reset', 11, 12)
        library.flash_bridge_collision_reset.assert_called_once_with(11, 12, 41)
        self.assertEqual(library.flash_bridge_collision_reset.argtypes, [ct.c_void_p] * 3)

    def test_streams_keep_independent_call_contexts_and_library_owners(self):
        first_stream, second_stream = SimpleNamespace(device=SimpleNamespace(scope=nullcontext),cuda_stream=41), SimpleNamespace(device=SimpleNamespace(scope=nullcontext),cuda_stream=42)
        first_library, second_library = self.library(), self.library()
        with patch.object(bridges.ct, 'CDLL', side_effect=[first_library, second_library]):
            first = bridges.Bridges('/products/cuda_bridges_fp32.so', first_stream)
            second = bridges.Bridges('/products/cuda_bridges_fp32.so', second_stream)
        first.call('collision_reset', 11, 12)
        second.call('collision_reset', 21, 22)
        first_library.flash_bridge_collision_reset.assert_called_once_with(11, 12, 41)
        second_library.flash_bridge_collision_reset.assert_called_once_with(21, 22, 42)
        self.assertIsNot(first_stream._bridge_libraries, second_stream._bridge_libraries)

    def test_different_products_on_one_stream_do_not_alias(self):
        stream = SimpleNamespace(device=SimpleNamespace(scope=nullcontext),cuda_stream=41)
        with patch.object(bridges.ct, 'CDLL', side_effect=[self.library(), self.library()]):
            first = bridges.Bridges('/one/cuda_bridges_fp32.so', stream)
            second = bridges.Bridges('/two/cuda_bridges_fp32.so', stream)
        self.assertIsNot(first.library, second.library)

    def test_failed_abi_admission_is_not_reused(self):
        stream = SimpleNamespace(device=SimpleNamespace(scope=nullcontext),cuda_stream=41)
        wrong, correct = self.library(), self.library()
        wrong.flash_bridge_gather_bytes.return_value = 0
        with patch.object(bridges.ct, 'CDLL', side_effect=[wrong, correct]) as load:
            with self.assertRaisesRegex(ValueError, 'Bridge ABI mismatch: gather'):
                bridges.Bridges('/products/cuda_bridges_fp32.so', stream)
            self.assertEqual(stream._bridge_libraries, {})
            bridge = bridges.Bridges('/products/cuda_bridges_fp32.so', stream)
        self.assertIs(bridge.library, correct)
        self.assertEqual(load.call_count, 2)

    def test_source_files_are_not_read_and_cuda_errors_remain_active(self):
        stream = SimpleNamespace(device=SimpleNamespace(scope=nullcontext),cuda_stream=41)
        library = self.library()
        with patch('pathlib.Path.read_bytes', side_effect=AssertionError('source read')), \
             patch.object(bridges.ct, 'CDLL', return_value=library):
            bridges.Bridges('/products/cuda_bridges_fp32.so', stream)
        library.flash_bridge_source_id.assert_not_called()
        library.flash_bridge_collision_reset.return_value = 17
        with patch.object(bridges.ct, 'CDLL', return_value=library):
            bridge = bridges.Bridges('/products/cuda_bridges_fp32.so', stream)
        with self.assertRaisesRegex(RuntimeError, 'CUDA bridge collision_reset: error 17'):
            bridge.call('collision_reset', 11, 12)


if __name__ == '__main__':
    unittest.main()
