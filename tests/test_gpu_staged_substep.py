"""Offline selection of the coarse FP64 product; no CUDA initialization."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from libero.gpu.host.build import substep_native as build
from libero.gpu.host.runtime.step.staged_motion_plan import build_motion_plan, motion_threads
from libero.gpu.host.runtime.step.staged_motion_plan import passive_motion_plan as motion_plan
import numpy as np


class StagedProductTests(unittest.TestCase):
    def test_force_selection_retains_the_native_substep_graph_owners(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        import warp as wp
        from libero.gpu.host.adapters.environment.batched_physics import NativeBatchedPhysics
        physics = NativeBatchedPhysics.__new__(NativeBatchedPhysics)
        physics.debug = False
        physics._observer = None
        runtime = SimpleNamespace(substeps=3)
        physics.bundle = SimpleNamespace(runtime=runtime, stream=object())
        capture = SimpleNamespace(graph=object())
        fusion = MagicMock()
        with patch('libero.gpu.host.runtime.step.peeled_force.PeeledForceSubstep',
                   return_value=fusion) as factory, patch.object(wp, 'ScopedCapture') as scope:
            scope.return_value.__enter__.return_value = capture
            result = physics.prepare_fused_substep_graph('offline',actuation_solve=True)
        self.assertIs(result,fusion)
        self.assertIs(physics._substep_fusion,fusion)
        self.assertIs(physics._control_graph,capture.graph)
        self.assertEqual(factory.call_args.kwargs['force_threads'],256)
        self.assertEqual(factory.call_args.kwargs['motion_threads'],512)
        self.assertFalse(factory.call_args.kwargs['allow_build'])
        self.assertEqual(fusion.enqueue_substep.call_count,3)
        with self.assertRaisesRegex(ValueError,'select staged'):
            physics.prepare_fused_substep_graph('offline',staged=True,actuation_solve=True)

    def test_motion_product_keeps_the_native_schedule_and_independent_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'project/fused_substep.cu'
            source.parent.mkdir()
            source.write_text('// source fixture\n')
            (source.parent/'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.24)\n')
            with patch.object(build.scalar, 'bits', 64), patch.object(build, 'nvcc_path', return_value=str(root/'nvcc')):
                fused = build.prepare(source, root/'cache', root/'sdk', 80, nv=57)
                motion = build.prepare_motion(source, root/'cache', root/'sdk', 80, nv=57)
                force = build.prepare_force(source, root/'cache', root/'sdk', 80, nv=57)
            self.assertEqual(motion['folder'], fused['folder'])
            self.assertNotEqual(motion['identity'], fused['identity'])
            self.assertEqual(motion['target'], 'flash_substep_motion_nv57')
            self.assertEqual(Path(motion['library']).name, 'substep_motion_nv57.so')
            self.assertEqual(motion['config']['solves_per_substep'], 3)
            self.assertEqual(force['folder'], fused['folder'])
            self.assertEqual(force['target'], 'flash_substep_force_nv57')
            self.assertEqual(force['config']['solves_per_substep'], 3)
            self.assertEqual(force['config']['force_schedule'], 'entry-reuse-controller')
            self.assertFalse(force['source_generation'])

    def test_staged_product_has_its_own_identity_and_reuses_build_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'project/fused_substep.cu'
            source.parent.mkdir()
            source.write_text('// source fixture\n')
            (source.parent / 'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.24)\n')
            with patch.object(build.scalar, 'bits', 64), patch.object(build, 'nvcc_path', return_value=str(root / 'nvcc')):
                fused = build.prepare(source, root / 'cache', root / 'sdk', 80, nv=21)
                staged = build.prepare_staged(source, root / 'cache', root / 'sdk', 80, nv=21)
                ordered = build.prepare_staged(source, root / 'cache', root / 'sdk', 80,
                                               nv=21, atomic_reductions=False)
            self.assertEqual(fused['folder'], staged['folder'])
            self.assertNotEqual(fused['identity'], staged['identity'])
            self.assertEqual(staged['target'], 'flash_substep_staged_nv21')
            self.assertEqual(Path(staged['library']).name, 'substep_staged_nv21.so')
            self.assertEqual(ordered['target'], 'flash_substep_staged_nv21_ordered')
            self.assertEqual(staged['config']['solves_per_substep'], 3)
            self.assertEqual(staged['commands'][1][staged['commands'][1].index('--target') + 1], staged['target'])
            with patch.object(build.scalar, 'bits', 64), patch.object(build, 'nvcc_path', return_value=str(root / 'nvcc')):
                all_staged = build.prepare_all(source, root / 'cache', root / 'sdk', 80, staged=True)
            self.assertEqual(all_staged['target'], 'flash_staged_substeps_all')
            self.assertEqual(len(all_staged['libraries']), 13)
            self.assertTrue(all('substep_staged' in Path(p).name for p in all_staged['libraries']))

    def test_scalar_and_schedule_domain_is_explicit(self):
        for bits, options in ((32, {}), (64, {'single_solve': True}), (64, {'profile_newton': True})):
            with self.subTest(bits=bits, options=options), patch.object(build.scalar, 'bits', bits):
                with self.assertRaisesRegex(ValueError, 'FP64, three solves'):
                    build.prepare_staged('unused', 'unused', 'unused', 80, **options)

    def test_load_only_missing_product_never_executes_a_build(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(build.subprocess, 'run') as execute:
            manifest = dict(folder=directory, library=str(Path(directory) / 'staged.so'))
            with self.assertRaises(build.MissingCompiledProduct):
                build.finish(manifest, allow_build=False)
            execute.assert_not_called()


class MotionPlanTests(unittest.TestCase):
    def test_peeled_bias_requires_contiguous_subtrees(self):
        self.assertIsNone(motion_plan([0, 0, 1, 0, 1]))
        plan = motion_plan([0, 0, 1, 1, 0])
        np.testing.assert_array_equal(plan[-5:], [5, 3, 1, 1, 1])

    def test_peeled_topology_uses_component_fallback_outside_fp32_domain(self):
        self.assertIsNone(motion_plan([0] * 34))
        self.assertIsNone(motion_plan([0] + list(range(64))))

    def test_shared_prefix_has_one_public_owner(self):
        # Paths [1,2,4], [1,3], [5]; body1 is redundantly computed but only
        # the first path publishes it. Parent IDs need not form DFS intervals.
        plan = build_motion_plan([0, 0, 1, 1, 2, 0])
        np.testing.assert_array_equal(plan, [3, 0, 2, 5, 6, 1, 3, 1, 2, 4, 5,
                                            -1, 0, 1, 0, 1, 2])
        self.assertEqual(motion_threads(plan, 6), 128)

    def test_large_bodies_have_no_fixed_shared_array_limit(self):
        parent = [0] + list(range(100))
        plan = build_motion_plan(parent)
        self.assertEqual(plan[0], 1)
        self.assertEqual(motion_threads(plan, len(parent)), 128)
        plan = build_motion_plan([0] * 40)
        self.assertEqual(motion_threads(plan, 40), 256)

    def test_expensive_duplicated_prefix_keeps_component_fallback(self):
        # A deep trunk with many leaves would duplicate substantial work.
        parent = [0] + list(range(20)) + [20] * 20
        self.assertEqual(motion_threads(build_motion_plan(parent), len(parent)), 0)

    def test_invalid_dependency_rejected(self):
        for parent in ([], [1], [0, 1], [0, 2, 0], [0, -1]):
            with self.subTest(parent=parent), self.assertRaises(ValueError):
                build_motion_plan(parent)
