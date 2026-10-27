"""Offline-built OSC replay versus independent SVD; never compile at runtime.

Set LIBERO_OSC_REPLAY_LIBRARY to libflash_osc_mixed_replay_probe_fp32.so.
The sibling cuda_runtime.so supplies resource ownership only.
"""
import ctypes as ct
import os
from pathlib import Path
import unittest

import numpy as np


@unittest.skipUnless(os.environ.get('LIBERO_OSC_REPLAY_LIBRARY'), 'offline OSC replay product required')
class OscMixedPrecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from libero.gpu.host.backends.cuda_fp32 import device as cuda
        cls.cuda = cuda
        path = Path(os.environ['LIBERO_OSC_REPLAY_LIBRARY'])
        cls.library = ct.CDLL(str(path))
        cls.library.flash_osc_replay_trace_bytes.restype = ct.c_size_t
        cls.trace_dtype = np.dtype([('scale', 'f8'), ('maxcorr', 'f8'),
            ('singular', 'f8', (6,)), ('sweeps', 'i4'), ('rotations', 'i4')])
        assert cls.trace_dtype.itemsize == cls.library.flash_osc_replay_trace_bytes()
        cls.library.flash_osc_replay.argtypes = [ct.c_void_p]*5 + [ct.c_int, ct.c_int, ct.c_void_p]
        cls.library.flash_osc_replay.restype = ct.c_int
        cls.runtime = cuda.Runtime(path.parent / 'cuda_runtime.so')
        cls.stream = cuda.Stream(cls.runtime.device(0))

    @classmethod
    def tearDownClass(cls):
        cls.stream.close()

    def replay(self, matrices):
        n, count = matrices.shape[-1], len(matrices)
        padded = np.zeros((count, 6, 6), np.float64)
        padded[:, :n, :n] = matrices
        cuda, stream = self.cuda, self.stream
        with stream:
            inputs = cuda.array(padded, dtype=cuda.float64, device=stream.device)
            output = cuda.zeros_like(inputs)
            rank = cuda.zeros(count, dtype=cuda.int32, device=stream.device)
            ok = cuda.zeros_like(rank)
            trace = cuda.zeros((count, self.trace_dtype.itemsize), dtype=cuda.uint8, device=stream.device)
            error = self.library.flash_osc_replay(inputs.ptr, output.ptr, rank.ptr, ok.ptr,
                trace.ptr, n, count, stream.cuda_stream)
            self.assertEqual(error, 0)
            result = output.numpy()[:, :n, :n]
            return result, rank.numpy(), ok.numpy(), trace.numpy().reshape(-1).view(self.trace_dtype)

    def test_zero_and_rank_deficient_two_worlds(self):
        for n in (3, 6):
            matrix = np.zeros((2, n, n), np.float32)
            matrix[1, :2, :2] = [[1, 2], [2, 4]]
            inverse, rank, ok, trace = self.replay(matrix)
            np.testing.assert_array_equal(rank, [0, 1])
            np.testing.assert_array_equal(ok, [1, 1])
            np.testing.assert_allclose(inverse, np.linalg.pinv(matrix.astype(np.float64)), atol=1e-14)
            self.assertEqual(trace['sweeps'][0], 0)

    def test_scaled_ill_conditioned_fp32_inputs(self):
        rng = np.random.default_rng(1701)
        for n in (3, 6):
            matrices = []
            for condition in (1., 1e3, 1e6):
                for scale in (1e-20, 1., 1e20):
                    for _ in range(4):
                        q, _ = np.linalg.qr(rng.normal(size=(n, n)))
                        matrices.append((scale * (q * np.geomspace(1., 1./condition, n)) @ q.T).astype(np.float32))
            matrices = np.asarray(matrices).astype(np.float64)
            inverse, rank, ok, trace = self.replay(matrices)
            np.testing.assert_array_equal(ok, np.ones(len(matrices), np.int32))
            np.testing.assert_array_equal(rank, np.full(len(matrices), n, np.int32))
            reference = np.linalg.pinv(matrices, rcond=1e-15)
            for a, p, expected in zip(matrices, inverse, reference):
                relative = np.linalg.norm(p-expected) / np.linalg.norm(expected)
                budget = 64*n*np.finfo(np.float64).eps*np.linalg.cond(a)
                self.assertLess(relative, budget)
                residual = np.linalg.norm(a@p@a-a) / np.linalg.norm(a)
                self.assertLess(residual, budget)
            self.assertTrue(np.all(trace['sweeps'] <= 64))
            self.assertTrue(np.isfinite(trace['maxcorr']).all())

    def test_fp32_roundoff_direction_does_not_become_retained_rank(self):
        # Exact outer product [5,-5,-4]^T [5,-5,-4]. The former FP32 Jacobi
        # kept a spurious 1.85e-6 singular value, exhausted 64 sweeps and failed.
        a = np.array([[25, -25, -20], [-25, 25, 20], [-20, 20, 16]], np.float32)
        matrix = np.stack((a, 16*a))
        inverse, rank, ok, trace = self.replay(matrix)
        np.testing.assert_array_equal(rank, [1, 1])
        np.testing.assert_array_equal(ok, [1, 1])
        np.testing.assert_allclose(inverse, np.linalg.pinv(matrix.astype(float)), rtol=2e-14, atol=1e-16)
        self.assertTrue(np.all(trace['maxcorr'] < 2e-15))

    def test_controller_torque_and_fp64_cache(self):
        from libero.gpu.host.backends.cuda_fp32.controller import Controller
        from libero.gpu.host.runtime.control import ControllerSolveData
        cuda, stream = self.cuda, self.stream
        library = ct.CDLL(str(Path(os.environ['LIBERO_OSC_REPLAY_LIBRARY']).parent / 'mjwarp_controller_fp32.so'))
        launch = library.flash_mjwarp_controller_launch
        launch.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p]
        launch.restype = ct.c_int
        rng = np.random.default_rng(91)
        with stream:
            buffers = {name: cuda.zeros((2, width), dtype=cuda.float32, device=stream.device)
                       for name, width in Controller.WIDTHS.items()}
            for name in ('active', 'error', 'policy', 'needs', 'quat_valid', 'cache_valid',
                         'derived_valid', 'cache_hits'):
                buffers[name] = cuda.zeros(2, dtype=cuda.int32, device=stream.device)
            buffers['ranks'] = cuda.zeros((2, 3), dtype=cuda.int32, device=stream.device)
            buffers['derived_ranks'] = cuda.zeros_like(buffers['ranks'])
            buffers['derived_key'] = cuda.zeros((2, 91), dtype=cuda.float32, device=stream.device)
            buffers['derived_data'] = cuda.zeros((2, 206), dtype=cuda.float64, device=stream.device)
            buffers['quat_cast'] = cuda.zeros((2, 4), dtype=cuda.float32, device=stream.device)
            host = {}
            def assign(name, value):
                host[name] = np.broadcast_to(np.asarray(value, np.float32), buffers[name].shape).copy()
                buffers[name].assign(host[name])
            raw = rng.normal(size=(2, 7, 7))
            mass = raw @ raw.transpose(0, 2, 1) + 2*np.eye(7)
            assign('mass', mass.reshape(2, 49))
            assign('jacobian', rng.normal(size=(2, 42)))
            assign('q', rng.normal(size=(2, 7)))
            assign('q0', rng.normal(size=(2, 7)))
            assign('qvel', rng.normal(size=(2, 7)))
            assign('bias', rng.normal(size=(2, 7)))
            assign('goal_pos', rng.normal(size=(2, 3)))
            assign('ee_pos', rng.normal(size=(2, 3)))
            assign('goal_rot', np.eye(3).reshape(9))
            assign('ee_rot', np.eye(3).reshape(9))
            assign('ee_velocity', rng.normal(size=(2, 6)))
            assign('gains', np.r_[np.full(6, 150.), np.full(6, 2*np.sqrt(150.))])
            assign('range', np.tile([-100., 100.], 9))
            buffers['active'].fill_(1)
            buffers['cache_valid'].fill_(1)
            args = ControllerSolveData()
            for name, array in buffers.items():
                setattr(args, name, array.ptr)
            args.diagnostics = 1
            for uncouple in (0, 1):
                args.uncouple = uncouple
                packed = cuda.array(np.frombuffer(bytes(args), np.uint8).copy(), dtype=cuda.uint8, device=stream.device)
                buffers['derived_valid'].zero_()
                buffers['needs'].zero_()
                self.assertEqual(launch(packed.ptr, 2, stream.cuda_stream), 0)
                np.testing.assert_array_equal(buffers['error'].numpy(), [0, 0])
                first = buffers['raw_tau'].numpy()
                for w in range(2):
                    m = host['mass'][w].reshape(7, 7).astype(np.float64)
                    j = host['jacobian'][w].reshape(6, 7).astype(np.float64)
                    x = np.linalg.solve(m, j.T)
                    a = j @ x
                    full = np.linalg.pinv(a, rcond=1e-15)
                    project = np.eye(7) - x @ full @ j
                    error = np.r_[host['goal_pos'][w].astype(float)-host['ee_pos'][w], np.zeros(3)]
                    desired = host['gains'][w, :6].astype(float)*error - host['gains'][w, 6:].astype(float)*host['ee_velocity'][w]
                    wrench = (np.r_[np.linalg.pinv(a[:3, :3]) @ desired[:3],
                                    np.linalg.pinv(a[3:, 3:]) @ desired[3:]] if uncouple else full @ desired)
                    pd = 10*(host['q0'][w].astype(float)-host['q'][w]) - 6.324555320336759*host['qvel'][w]
                    expected = j.T @ wrench + host['bias'][w] + project.T @ m @ pd
                    np.testing.assert_allclose(first[w], expected, rtol=2e-6, atol=2e-5)
                    np.testing.assert_allclose(buffers['ctrl'].numpy()[w, :7],
                                               np.clip(expected, -100, 100), rtol=2e-6, atol=2e-5)
                    cached = buffers['derived_data'].numpy()[w]
                    np.testing.assert_allclose(cached[49:85].reshape(6, 6), a, rtol=2e-14, atol=2e-14)
                buffers['needs'].zero_()
                self.assertEqual(launch(packed.ptr, 2, stream.cuda_stream), 0)
                np.testing.assert_array_equal(buffers['raw_tau'].numpy(), first)
                self.assertTrue(np.all(buffers['cache_hits'].numpy() > 0))
            # Exercise rank-deficient full/position/orientation groups in the
            # actual controller's warp and both half-warps, with valid SPD M.
            assign('mass', np.tile(np.eye(7).reshape(1, 49), (2, 1)))
            jacobian = np.zeros((2, 6, 7), np.float32)
            jacobian[:, :3, 0] = [5, -5, -4]
            jacobian[:, 3:, 1] = [1, 2, 3]
            assign('jacobian', jacobian.reshape(2, 42))
            buffers['derived_valid'].zero_()
            buffers['needs'].zero_()
            self.assertEqual(launch(packed.ptr, 2, stream.cuda_stream), 0)
            np.testing.assert_array_equal(buffers['error'].numpy(), [0, 0])
            np.testing.assert_array_equal(buffers['ranks'].numpy(), [[2, 1, 1], [2, 1, 1]])
            self.assertTrue(np.isfinite(buffers['ctrl'].numpy()).all())

    def test_offline_fused_source_ids_include_compute_precision(self):
        from libero.gpu.host.backends.cuda_fp32 import fused_controller, fused_control_drive
        directory = Path(os.environ['LIBERO_OSC_REPLAY_LIBRARY']).parent
        for filename, symbol, module in (
                ('fp32_fused_controller.so', 'flash_fp32_controller_source_id', fused_controller),
                ('fp32_fused_control_drive.so', 'flash_fp32_control_drive_source_id', fused_control_drive)):
            library = ct.CDLL(str(directory / filename))
            query = getattr(library, symbol)
            query.restype = ct.c_char_p
            self.assertEqual(query().decode(), module.source_id())


if __name__ == '__main__':
    unittest.main()
