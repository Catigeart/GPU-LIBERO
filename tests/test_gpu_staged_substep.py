"""Offline selection of the coarse FP64 product; no CUDA initialization."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from libero.gpu.host.build import substep_native as build


class StagedProductTests(unittest.TestCase):
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
