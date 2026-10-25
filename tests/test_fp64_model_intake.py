"""Direct model intake, with host-only domain and ownership regressions."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('mujoco') and importlib.util.find_spec('warp'),
                     'requires MuJoCo and Warp host types')
class ModelIntakeTests(unittest.TestCase):
    def setUp(self):
        import mujoco
        from libero.gpu.host.runtime.core.contracts import ModelDimensions, ModelSpec
        from libero.gpu.host.runtime.core.registry import CapabilityRegistry, KernelCandidate
        from libero.gpu.host.runtime.foundation.runtime_adapter import STATIC, _native_field_digest
        self.model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint/>'
            '<geom size=".1"/><site/></body></worldbody></mujoco>')
        dims = ModelDimensions(**{name: int(getattr(self.model, name))
                                 for name in ModelDimensions.__dataclass_fields__})
        self.spec = ModelSpec(dims, '1'*64, '2'*64, '3'*64, {})
        self.arrays = {name: np.array(getattr(self.model, name), copy=True) for name in STATIC}
        self.metadata = dict(dimensions=dims.as_dict(), model_sha256=self.spec.model_sha256,
            topology_sha256=self.spec.topology_sha256, static_arrays_sha256=self.spec.static_arrays_sha256,
            options={}, array_sha256={name: _native_field_digest(name, a) for name, a in self.arrays.items()},
            robot_metadata=[{'joint_ids': [0]}])
        self.registry = CapabilityRegistry()
        self.registry.register_candidate(KernelCandidate('test', 'D', '4'*64, 'test',
            self.spec.layout_version, 'CUDA', 'fp64', 'test',
            model_sha256s=(self.spec.model_sha256,)))

    def test_kinematic_intake_needs_no_source_or_metadata_files(self):
        from libero.gpu.host.runtime.foundation.runtime_adapter import PreparedKinematicModel
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('source read')):
            prepared = PreparedKinematicModel(self.arrays, self.metadata, self.spec,
                registry=self.registry, candidate_id='test', mode='diagnostic')
            prepared.validate()
        np.testing.assert_array_equal(prepared.arrays['body_pos'], self.arrays['body_pos'])
        self.arrays['body_pos'][1, 0] = 10
        self.assertNotEqual(prepared.arrays['body_pos'][1, 0], 10)
        self.assertFalse(prepared.arrays['body_pos'].flags.writeable)
        with self.assertRaisesRegex(ValueError, 'field digest differs'):
            PreparedKinematicModel(self.arrays, self.metadata, self.spec,
                registry=self.registry, candidate_id='test', mode='diagnostic')

    def test_native_fields_keep_dimensions_array_values_and_domain_checks(self):
        from libero.gpu.host.runtime.dynamics.specialization import native_fields
        options = dict(registry=self.registry, candidate_id='test', mode='diagnostic')
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('source read')):
            fields, _ = native_fields(self.arrays, self.metadata, self.spec, ('body_pos',), **options)
        np.testing.assert_array_equal(fields['body_pos'], self.arrays['body_pos'])
        self.assertFalse(fields['body_pos'].flags.writeable)
        wrong = deepcopy(self.metadata)
        wrong['dimensions']['nv'] += 1
        with self.assertRaisesRegex(ValueError, 'dimensions differ'):
            native_fields(self.arrays, wrong, self.spec, ('body_pos',), **options)
        with self.assertRaisesRegex(ValueError, 'admission refused'):
            native_fields(self.arrays, self.metadata, self.spec, ('body_pos',),
                          **dict(options, candidate_id='missing'))

    def test_factories_pass_one_metadata_snapshot_without_serializing_model(self):
        import mujoco
        from libero.gpu.host.runtime.dynamics import runtime_factory, wide_factory
        built = NS(native_model=self.model, metadata=self.metadata)
        for factory in (runtime_factory, wide_factory):
            with self.subTest(factory=factory.__name__):
                captured = {}
                def adapter(**kwargs):
                    captured.update(kwargs)
                    return NS()
                with patch.object(Path, 'read_bytes', side_effect=AssertionError('source read')), \
                     patch.object(factory, 'GenericRuntimeDynamicsAdapter', side_effect=adapter), \
                     patch.object(mujoco, 'mj_saveModel', side_effect=AssertionError('model serialization')):
                    _, intake = factory.prepare_from_built(built, 2, directory='unused',
                        decoder_record={'path': '/retired/decoder.py', 'sha256': '5'*64},
                        state_epoch=None, first_error=None, stream=None)
                self.assertIsNot(captured['metadata'], self.metadata)
                self.assertEqual(captured['metadata'], self.metadata)
                self.assertIs(intake['model_record']['model'], self.model)
                self.assertIsNone(intake['metadata_record'])
                self.assertEqual(captured['source_records'], ())
                for cid in intake['candidate_ids'].values():
                    candidate = captured['registry'].candidates[cid]
                    self.assertIsNone(candidate.source_sha256)
                    self.assertEqual(dict(candidate.source_files), {})
                    self.assertTrue(captured['registry'].admit(cid, captured['spec'], mode='diagnostic').execution_allowed)
                    self.assertTrue(captured['registry'].admit(cid, captured['spec']).execution_allowed)

    def test_components_share_one_immutable_snapshot_without_rehashing(self):
        from libero.gpu.host.runtime.core import static_readset as store
        from libero.gpu.host.runtime.dynamics.specialization import native_fields
        from libero.gpu.host.runtime.foundation.runtime_adapter import PreparedKinematicModel
        with patch.object(store, 'native_field_digest', wraps=store.native_field_digest) as digest:
            arrays = store.PreparedModelArrays(self.arrays, self.metadata, self.spec)
            self.assertEqual(digest.call_count, len(self.arrays))
        options = dict(registry=self.registry, candidate_id='test', mode='production')
        with patch.object(store, 'native_field_digest', side_effect=AssertionError('component rehash')):
            first, _ = native_fields(arrays, self.metadata, self.spec, ('body_pos',), **options)
            second, _ = native_fields(arrays, self.metadata, self.spec, ('body_pos',), **options)
            body = PreparedKinematicModel(arrays, self.metadata, self.spec, **options)
            body.validate()
        self.assertIs(first['body_pos'], second['body_pos'])
        self.assertIs(first['body_pos'], body.arrays['body_pos'])
        with self.assertRaises(ValueError):
            first['body_pos'].setflags(write=True)
        self.arrays['body_pos'][:] = 12
        self.assertFalse(np.any(first['body_pos'] == 12))
        first['body_pos'].shape = (first['body_pos'].size,)
        with self.assertRaisesRegex(ValueError, 'layout'):
            native_fields(arrays, self.metadata, self.spec, ('body_pos',), **options)
        with self.assertRaisesRegex(ValueError, 'owner/layout'):
            body.validate()

    def test_live_newton_model_is_retained_and_external_file_identity_is_checked(self):
        import mujoco
        from libero.gpu.host.runtime.core.model_bytes import load_native_model
        with patch.object(mujoco, 'mj_saveModel', side_effect=AssertionError('model serialization')):
            actual, path = load_native_model(model=self.model, expected_sha256='1'*64)
        self.assertIs(actual, self.model)
        self.assertIsNone(path)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.mjb'
            mujoco.mj_saveModel(self.model, str(path), None)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(mujoco, 'mj_saveModel', side_effect=AssertionError('reserialization')):
                actual, _ = load_native_model(model_path=path, expected_sha256=digest)
            self.assertEqual(actual.nv, self.model.nv)
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                load_native_model(model_path=path, expected_sha256='0'*64)
        with self.assertRaisesRegex(TypeError, 'MjModel'):
            load_native_model(model=object(), expected_sha256='1'*64)


if __name__ == '__main__':
    unittest.main()
