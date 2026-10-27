"""Prepared FP32 products work without source certificates or a GPU."""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
from libero.gpu.host.backends.cuda_fp32 import device, fused_tail, scene_inputs
from libero.gpu.host.backends.mjwarp import scene_cache, scene_inputs as reference_inputs
from libero.gpu.host.build.compilation import MissingCompiledProduct


class LoadOnlyTests(unittest.TestCase):
    def test_previous_exact_scene_key_loads_without_migration_or_scanning(self):
        task = dict(key='suite/task', files={'scene.xml':'scene'})
        options = dict(use_camera_obs=False, ignore_done=True)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(scene_cache, 'engine_identity', return_value={'python':'3.16.0'}):
            root = Path(directory)
            legacy = dict(scene_cache.binding('<mujoco/>', options, task),
                          options=options, scene_parameters=None)
            key = hashlib.sha256(json.dumps(legacy, sort_keys=True).encode()).hexdigest()
            old = root/'mjwarp/models'/key
            old.mkdir(parents=True)
            (old/'model.mjb').write_bytes(b'prepared model')
            (old/'receipt.json').write_text(json.dumps(dict(schema='modern-scene-v2', binding=legacy)))
            with patch.object(Path, 'glob', side_effect=AssertionError('runtime directory scan')), \
                    patch.object(scene_cache, 'offline_inputs', side_effect=AssertionError('offline validation')):
                folder, model, _, _ = scene_cache.locate(root, '<mujoco/>', options, task, prepare=False)
            self.assertEqual(model, old/'model.mjb')
            self.assertFalse(folder.exists())

    def test_verified_scene_reindex_shares_existing_model_storage(self):
        import sys
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'old.mjb'
            source.write_bytes(b'complete immutable CPU model')
            value = hashlib.sha256(source.read_bytes()).hexdigest()
            built = NS(xml='<mujoco/>', native_model=object(), metadata=dict(model_sha256=value))
            row = dict(model_sha256=value,
                       processed_xml_sha256=hashlib.sha256(built.xml.encode()).hexdigest())
            save = Mock(side_effect=AssertionError('must reuse the verified model'))
            with patch.dict(sys.modules, {'mujoco': NS(mj_saveModel=save)}), \
                    patch.object(scene_cache, 'offline_inputs', return_value={}):
                scene_cache.finish(root/'new', built, row, {}, prepare=True,
                                   xml=built.xml, compiled_model=source)
            self.assertEqual(source.stat().st_ino, (root/'new/model.mjb').stat().st_ino)
            save.assert_not_called()

    def test_host_camera_and_episode_options_do_not_require_another_cpu_model(self):
        task = dict(key='suite/task', files={'scene.xml': 'fixed-scene'})
        with patch.object(scene_cache, 'engine_identity', return_value={'python': '3.16.0'}):
            plain = scene_cache.binding('<mujoco/>', {}, task)
            cameras = scene_cache.binding('<mujoco/>', dict(use_camera_obs=True,
                has_offscreen_renderer=True, camera_heights=256,
                camera_names=['agentview'], ignore_done=True, horizon=1000,
                controller_configs={'type':'OSC_POSE','input_max':2}), task)
            self.assertEqual(plain, cameras)
            legacy = dict(plain, options=dict(use_camera_obs=False, ignore_done=True))
            self.assertEqual(scene_cache._data_binding(legacy), plain)
            self.assertNotEqual(plain, scene_cache.binding('<mujoco/>', dict(robots=['Other']), task))
            self.assertNotEqual(plain, scene_cache.binding('<mujoco model="changed"/>', {}, task))

    def test_runtime_binds_exports_without_source_files(self):
        library = Mock()
        with patch.object(device.ct, 'CDLL', return_value=library), \
             patch.object(Path, 'read_bytes', side_effect=AssertionError('source read')):
            runtime = device.Runtime('/offline/runtime.so')
        library.flash_runtime_source_id.assert_not_called()
        self.assertEqual(runtime.lib.flash_runtime_allocate.argtypes,
                         [ct.POINTER(ct.c_void_p), ct.c_size_t])

    def test_fused_tail_retains_abi_architecture_and_launch_errors(self):
        euler = NS(stream=NS(device=NS(arch=80), cuda_stream=17), arguments=fused_tail.EulerInputs())
        library = NS(flash_fp32_tail_arguments_bytes=Mock(return_value=ct.sizeof(fused_tail.EulerInputs)),
                     flash_fp32_tail_arch=Mock(return_value=80), flash_fp32_tail_advance=Mock(return_value=0))
        with patch.object(fused_tail.ct, 'CDLL', return_value=library), \
             patch.object(Path, 'is_file', return_value=True), \
             patch.object(Path, 'read_bytes', side_effect=AssertionError('source read')):
            tail = fused_tail.FusedAdvance(euler, '/offline/tail.so')
            tail.enqueue()
            library.flash_fp32_tail_advance.return_value = 7
            with self.assertRaisesRegex(RuntimeError, 'CUDA error 7'):
                tail.enqueue()
            library.flash_fp32_tail_arguments_bytes.return_value = 1
            with self.assertRaisesRegex(ValueError, 'ABI'):
                fused_tail.FusedAdvance(euler, '/offline/tail.so')
            library.flash_fp32_tail_arguments_bytes.return_value = ct.sizeof(fused_tail.EulerInputs)
            library.flash_fp32_tail_arch.return_value = 90
            with self.assertRaisesRegex(ValueError, 'architecture'):
                fused_tail.FusedAdvance(euler, '/offline/tail.so')


class SceneInputsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root/'mjwarp/scene-inputs/task.npz'
        self.path.parent.mkdir(parents=True)
        np.savez(self.path, body_names=np.array(['world', 'body']),
                 body_pos=np.zeros((2, 3)), body_quat=np.tile([1., 0., 0., 0.], (2, 1)))
        self.task = dict(key='suite/task', seed=1, directory='/catalog/task', dimensions={'nbody': 2},
                         files={'model.mjb': 'model', 'scene.xml': 'scene', 'initial_states.npy': 'initial'})
        self.row = dict(task='suite/task', seed=1, model_sha256='model', scene_sha256='scene',
                        init_sha256='initial', mujoco='2.3.7', exporter_sha256='obsolete-source',
                        data_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.receipt = self.path.with_suffix('.json')
        self.receipt.write_text(json.dumps(self.row))

    def test_current_and_legacy_inputs_ignore_exporter_source_but_keep_data_binding(self):
        for schema in (None, 'cuda-fp32-catalog-v1'):
            if schema is not None:
                self.row['exporter_schema'] = schema
                self.receipt.write_text(json.dumps(self.row))
            for reader in (scene_inputs.read, reference_inputs.read):
                with self.subTest(schema=schema, reader=reader.__module__):
                    with patch.object(Path, 'read_bytes', side_effect=AssertionError('runtime file hashing')):
                        data, _ = reader(self.root, self.task)
                    self.assertEqual(data['body_pos'].shape, (2, 3))
        self.row['seed'] = 3
        self.receipt.write_text(json.dumps(self.row))
        with self.assertRaises(MissingCompiledProduct):
            scene_inputs.read(self.root, self.task)

    def test_format_and_layout_fail_at_runtime_content_changes_fail_offline(self):
        self.row['exporter_schema'] = 'unknown'
        self.receipt.write_text(json.dumps(self.row))
        with self.assertRaisesRegex(MissingCompiledProduct, 'format'):
            scene_inputs.read(self.root, self.task)
        self.row.pop('exporter_schema')
        self.receipt.write_text(json.dumps(self.row))
        wrong = dict(self.task, dimensions={'nbody': 3})
        with self.assertRaisesRegex(ValueError, 'layout'):
            scene_inputs.read(self.root, wrong)
        self.path.write_bytes(b'changed data')
        with self.assertRaises(MissingCompiledProduct):
            scene_inputs.read(self.root, self.task, verify=True)

    def test_old_scene_discovery_is_offline_only_despite_source_changes(self):
        binding = dict(engine={'python': '3.16.0', 'library_sha256': 'lib'}, task='suite/task', assets={'mesh': 'bytes'})
        old = self.root/'mjwarp/models/old-source-dependent-key'
        old.mkdir(parents=True)
        model = old/'model.mjb'
        model.write_bytes(b'prepared model')
        (old/'receipt.json').write_text(json.dumps(dict(
            binding=dict(binding, sources={'former.py': 'retired'}, scene_parameters=self.row),
            model_sha256=hashlib.sha256(model.read_bytes()).hexdigest())))
        before = {p: p.read_bytes() for p in old.iterdir()}
        parameters = dict(self.row, exporter_sha256='new-exporter')
        with patch.object(scene_cache, 'binding', return_value=binding), \
             patch.object(scene_cache, 'offline_inputs', return_value={'assets': {'mesh': 'bytes'}, 'library_sha256': 'lib'}):
            with self.assertRaises(MissingCompiledProduct):
                scene_cache.locate(self.root, 'xml', {}, self.task, prepare=False, scene_parameters=parameters)
            folder, selected, _, record = scene_cache.locate(self.root, 'xml', {}, self.task,
                prepare=True, scene_parameters=parameters)
            self.assertNotEqual(folder, old)
            self.assertEqual(selected, model)
            self.assertNotIn('sources', record)
            self.assertNotIn('exporter_sha256', record['scene_parameters'])
            with self.assertRaises(MissingCompiledProduct):
                scene_cache.locate(self.root, 'xml', {}, self.task, prepare=False,
                    scene_parameters=dict(parameters, data_sha256='different-payload'))
        self.assertEqual({p: p.read_bytes() for p in old.iterdir()}, before)


if __name__ == '__main__':
    unittest.main()
