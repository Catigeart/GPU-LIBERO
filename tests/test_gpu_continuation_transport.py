"""Restore state without a second download; full verification is opt-in."""
from contextlib import nullcontext
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
import warp as wp
from libero.gpu.host.runtime.step import continuation as c


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        wp.init()
        self.bundle = NS(stream=object(), controller=None, model_spec=NS(model_sha256='model'))
        self.fields = {
            'state': wp.array(np.array([[1., np.nan], [-0., 4.]], np.float64), dtype=wp.float64, device='cpu'),
            'count': wp.array(np.array([2, 5], np.int64), dtype=wp.int64, device='cpu'),
            'vector': wp.array(np.ones((2, 3), np.float64), dtype=wp.vec3d, device='cpu'),
        }
        for target, name, value in (
            (c.wp, 'ScopedStream', lambda stream: nullcontext()),
            (c, 'continuation_arrays', lambda b: self.fields),
        ):
            patcher = patch.object(target, name, value)
            patcher.start(); self.addCleanup(patcher.stop)
        patcher = patch.object(c.wp, 'synchronize_stream')
        self.sync = patcher.start(); self.addCleanup(patcher.stop)

    def replace_targets(self):
        self.fields = {k: wp.zeros(a.shape, dtype=a.dtype, device='cpu') for k, a in self.fields.items()}

    def test_default_transport_has_no_payload_hash_or_readback(self):
        with patch.object(c.hashlib, 'sha256', side_effect=AssertionError('payload hashing')):
            packet = c.continuation_snapshot(self.bundle)
            self.replace_targets()
            with patch.object(wp.array, 'numpy', side_effect=AssertionError('restored-state readback')):
                result = c.restore_continuation(self.bundle, packet)
        self.assertTrue(result['transport_completed'])
        self.assertIsNone(result['data_bitwise_equal'])
        self.assertEqual(self.sync.call_count, 2)
        for name, array in self.fields.items():
            self.assertEqual(array.numpy().tobytes(), packet['data'][name].tobytes())
        self.assertTrue(all('sha256' not in h for h in packet['headers'].values()))

    def test_all_layouts_validated_before_any_write(self):
        packet = c.continuation_snapshot(self.bundle)
        self.replace_targets()
        packet['data']['count'] = np.zeros(3, dtype=np.int64)
        with patch.object(wp.array, 'assign', side_effect=AssertionError('premature upload')):
            with self.assertRaisesRegex(ValueError, 'shape/dtype'):
                c.restore_continuation(self.bundle, packet)
        packet['data'].pop('count')
        with self.assertRaisesRegex(ValueError, 'field set'):
            c.restore_continuation(self.bundle, packet)

    def test_explicit_verification_detects_corruption_and_upload_errors(self):
        packet = c.continuation_snapshot(self.bundle, verify=True)
        self.replace_targets()
        result = c.restore_continuation(self.bundle, packet, verify=True)
        self.assertTrue(result['data_bitwise_equal'])
        packet['data']['count'][0] += 1
        with self.assertRaisesRegex(ValueError, 'snapshot drift'):
            c.restore_continuation(self.bundle, packet, verify=True)
        packet['data']['count'][0] -= 1
        self.replace_targets()
        with patch.object(wp.array, 'assign', return_value=None):
            with self.assertRaisesRegex(ValueError, 'byte transport'):
                c.restore_continuation(self.bundle, packet, verify=True)

    def test_stream_failure_is_propagated(self):
        packet = c.continuation_snapshot(self.bundle)
        self.replace_targets()
        self.sync.side_effect = RuntimeError('transfer failure')
        with self.assertRaisesRegex(RuntimeError, 'transfer failure'):
            c.restore_continuation(self.bundle, packet)

    def test_native_replacement_uses_same_transport_after_model_validation(self):
        packet = c.continuation_snapshot(self.bundle)
        self.replace_targets()
        plan = NS(validate_base=lambda: None)
        with patch.object(c, 'validate_native_controller_transition', return_value='checked') as transition, \
             patch.object(wp.array, 'numpy', side_effect=AssertionError('restored-state readback')):
            result = c.restore_native_model_continuation(self.bundle, packet, plan)
        transition.assert_called_once_with(self.bundle, packet, plan)
        self.assertEqual(result['controller_static_transition'], 'checked')
        self.assertTrue(result['transport_completed'])
        self.assertIsNone(result['data_bitwise_equal'])


if __name__ == '__main__':
    unittest.main()
