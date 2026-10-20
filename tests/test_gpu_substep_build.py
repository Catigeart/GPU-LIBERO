"""Offline CMake cache ownership and runtime load-only boundary."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class SubstepBuildTests(unittest.TestCase):
    def test_single_solve_candidate_has_separate_admission_but_reuses_products(self):
        from libero.gpu.host.build import substep_native
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'project/kernel.cu'
            source.parent.mkdir()
            source.write_text('// fixture\n')
            (source.parent / 'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.24)\n')
            with patch.object(substep_native, 'nvcc_path', return_value=str(root / 'nvcc')):
                with patch.object(substep_native.scalar, 'bits', 32):
                    normal = substep_native.prepare(source, root / 'cache', root / 'sdk', 80)
                    single = substep_native.prepare(source, root / 'cache', root / 'sdk', 80,
                                                   single_solve=True)
                    self.assertEqual(normal['config']['solves_per_substep'], 3)
                    self.assertEqual(single['config']['solves_per_substep'], 1)
                    self.assertNotEqual(normal['identity'], single['identity'])
                    self.assertEqual(normal['library'], single['library'])
                    self.assertEqual(normal['commands'], single['commands'])
                with patch.object(substep_native.scalar, 'bits', 64):
                    with self.assertRaisesRegex(ValueError, 'FP32 candidate'):
                        substep_native.prepare(source, root / 'cache', root / 'sdk', 80,
                                               single_solve=True)

    def test_wheel_sources_find_the_packaged_cmake_entry_without_compiling(self):
        from libero.gpu.host.build import substep_native
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gpu = root / 'site-packages/libero/gpu'
            source = gpu / 'cuda/step/fused_substep.cu'
            source.parent.mkdir(parents=True)
            source.write_text('// installed CUDA source\n')
            project = gpu
            (project / 'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.24)\n')
            with patch.object(substep_native, 'nvcc_path', return_value=str(root / 'nvcc')):
                manifest = substep_native.prepare(source, root / 'cache', root / 'sdk', 80)
            self.assertEqual(manifest['config']['project'], str(project))
            self.assertIn(str(project), manifest['commands'][0])
            with patch.object(substep_native.subprocess, 'run') as execute:
                source = root / 'no-project/kernel.cu'
                with self.assertRaisesRegex(FileNotFoundError, 'No maintained CMake project'):
                    substep_native.prepare(source, root / 'cache', root / 'sdk', 80)
                execute.assert_not_called()

    def test_thread_configuration_changes_identity_and_rejects_old_manifest(self):
        from libero.gpu.host.build import substep_native
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'project/kernel.cu'
            source.parent.mkdir()
            source.write_text('// fixture\n')
            (source.parent / 'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.24)\n')
            with patch.object(substep_native, 'nvcc_path', return_value=str(root / 'nvcc')):
                first = substep_native.prepare(source, root / 'cache', root / 'sdk', 80)
                cache = Path(first['folder']) / 'CMakeCache.txt'
                entries = [f'CMAKE_HOME_DIRECTORY:STRING={source.parent}']
                entries += [arg[2:].split('=', 1)[0] + ':STRING=' + arg.split('=', 1)[1]
                            for arg in first['commands'][0] if arg.startswith('-D')]
                cache.write_text('\n'.join(entries) + '\n')
                self.assertTrue(substep_native._configured(first))
                cache.write_text(cache.read_text().replace('FLASH_SUBSTEP_THREADS:STRING=64',
                                                          'FLASH_SUBSTEP_THREADS:STRING=128'))
                self.assertFalse(substep_native._configured(first))
                second = substep_native.prepare(source, root / 'cache', root / 'sdk', 80)
                self.assertEqual(second['config']['threads_per_world'], 128)
                self.assertIn('-DFLASH_SUBSTEP_THREADS=128', second['commands'][0])
                self.assertNotEqual(first['identity'], second['identity'])
                self.assertEqual(first['folder'], second['folder'])
                self.assertTrue(substep_native._configured(second))

    def test_source_edits_keep_the_cmake_build_tree_and_runtime_is_load_only(self):
        from libero.gpu.host.build import substep_native
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);source=folder/'project/kernel.cu';source.parent.mkdir()
            (source.parent/'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.24)\n')
            header=source.with_suffix('.cuh');header.write_text('int value=1;\n')
            source.write_text('#include "kernel.cuh"\n')
            compiler=folder/'nvcc';compiler.write_text('fixture compiler\n')
            archive=folder/'sdk/lib/libcusolverdx.a';archive.parent.mkdir(parents=True);archive.write_bytes(b'fixture archive')
            with patch.object(substep_native,'nvcc_path',return_value=str(compiler)):
                first=substep_native.prepare(source,folder/'cache',folder/'sdk',80)
                self.assertEqual(first['commands'][0][0],'cmake')
                self.assertIn(str(source.parent),first['commands'][0])
                self.assertFalse(list((folder/'cache').rglob('*.cu')))
                self.assertTrue(first['config']['atomic_reductions'])
                ordered=substep_native.prepare(source,folder/'cache',folder/'sdk',80,atomic_reductions=False)
                self.assertEqual(first['folder'],ordered['folder'])
                self.assertNotEqual(first['identity'],ordered['identity'])
                self.assertEqual(ordered['target'],'flash_substep_ordered')
                self.assertFalse(ordered['config']['atomic_reductions'])
                self.assertTrue(ordered['config']['default_atomic_reductions'])
                fast=substep_native.prepare(source,folder/'cache',folder/'sdk',80,fast_compile=True)
                self.assertNotEqual(first['library'],fast['library'])
                profile=substep_native.prepare(source,folder/'cache',folder/'sdk',80,profile_newton=True)
                self.assertEqual(first['folder'],profile['folder'])
                self.assertNotEqual(first['library'],profile['library'])
                self.assertEqual(profile['target'],'flash_substep_newton_profile')
                self.assertEqual(Path(profile['library']).name,'substep_newton_profile.so')
                small=substep_native.prepare(source,folder/'cache',folder/'sdk',80,nv=21)
                self.assertEqual(first['folder'],small['folder'])
                self.assertEqual(small['target'],'flash_substep_nv21')
                self.assertEqual(Path(small['library']).name,'substep_nv21.so')
                all_shapes=substep_native.prepare_all(source,folder/'cache',folder/'sdk',80)
                self.assertEqual(first['folder'],all_shapes['folder'])
                self.assertEqual(all_shapes['target'],'flash_substeps_all')
                self.assertEqual(len(all_shapes['libraries']),13)
                self.assertIn('flash_substeps_all',all_shapes['commands'][1])
                ordered_all=substep_native.prepare_all(source,folder/'cache',folder/'sdk',80,atomic_reductions=False)
                self.assertEqual(ordered_all['target'],'flash_substeps_ordered_all')
                self.assertTrue(all(p.endswith('_ordered.so') for p in ordered_all['libraries']))
                header.write_text('int value=2;\n')
                second=substep_native.prepare(source,folder/'cache',folder/'sdk',80)
            self.assertEqual(first['identity'],second['identity'])
            self.assertEqual(first['folder'],second['folder'])
            from libero.gpu.host.build.compilation import compilation_scope, MissingCompiledProduct
            with compilation_scope(False),patch.object(substep_native.subprocess,'run') as execute:
                with self.assertRaises(MissingCompiledProduct):substep_native.finish(first)
                execute.assert_not_called()
            from types import SimpleNamespace
            Path(profile['library']).write_bytes(b'profile fixture')
            with compilation_scope(False),patch.object(substep_native,'_configured',return_value=True), \
                    patch.object(substep_native.subprocess,'run',return_value=SimpleNamespace(
                        returncode=0,stdout='ninja: no work to do.\n')) as execute:
                substep_native.finish(profile)
                self.assertEqual(execute.call_args.args[0][-1],'flash_substep_newton_profile')


if __name__ == "__main__":
    unittest.main()
