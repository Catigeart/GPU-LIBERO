"""Native model and generated package ownership without source certificates."""
from pathlib import Path
import unittest
from unittest.mock import patch

import mujoco

from libero.gpu.model.factory import BuiltTaskModel, native_model_metadata
from libero.gpu.host.runtime.core.contracts import ContractError
from libero.gpu.host.runtime.core.portable_intake import describe_native_model, prepare_native_intake
from libero.gpu.host.runtime.core.product_paths import product_build


@unittest.skipUnless(mujoco.__version__ == '2.3.7', 'native intake requires CPU reference MuJoCo 2.3.7')
class PortableIntakeManagementTests(unittest.TestCase):
    def setUp(self):
        self.xml = '<mujoco><option cone="elliptic"/><worldbody><body name="body">' \
                   '<joint/><geom size=".1"/></body></worldbody></mujoco>'
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.built = BuiltTaskModel(self.model, self.xml, None, native_model_metadata(self.model))

    def test_physical_metadata_matches_without_source_reads(self):
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('source read')):
            fresh = native_model_metadata(self.model, include_source_provenance=False)
            first = describe_native_model(self.built)
            self.built.metadata['decoder_pin'] = {'source_sha256': 'retired diagnostic metadata'}
            self.assertEqual(first, describe_native_model(self.built))
        self.assertNotIn('source_sha256', fresh['decoder_pin'])
        for key in fresh:
            if key != 'decoder_pin':
                self.assertEqual(fresh[key], self.built.metadata[key], key)

    def test_actual_parameter_and_policy_changes_are_still_rejected(self):
        self.model.body_pos[1, 0] += .1
        with self.assertRaisesRegex(ContractError, 'native model changed'):
            describe_native_model(self.built)
        self.model.body_pos[1, 0] -= .1
        self.built.metadata['mutable_parameters'] = ('body_mass',)
        with self.assertRaisesRegex(ContractError, 'permissions require'):
            describe_native_model(self.built)

    def test_native_callback_still_requires_gpu_implementation(self):
        previous = mujoco.get_mjcb_control()
        try:
            mujoco.set_mjcb_control(lambda model, data: None)
            with self.assertRaisesRegex(ContractError, 'callback requires'):
                describe_native_model(self.built)
        finally:
            mujoco.set_mjcb_control(previous)

    def test_generated_package_validation_uses_model_not_source_proof(self):
        with product_build():
            binding = prepare_native_intake(self.built)
            self.assertTrue((binding.package / 'identity.py').is_file())
            with patch.object(Path, 'read_bytes', side_effect=AssertionError('source proof read')):
                self.assertIs(binding.validate(), binding.spec)
                self.model.opt.timestep *= 2
                with self.assertRaisesRegex(ContractError, 'native model changed'):
                    binding.validate()

    def test_owner_replacement_and_explicit_invalidation(self):
        with product_build():
            binding = prepare_native_intake(self.built)
            self.built.native_model = mujoco.MjModel.from_xml_string(self.xml)
            with self.assertRaisesRegex(ContractError, 'owner changed'):
                binding.validate()
            self.built.native_model = self.model
            binding.invalidate()
            with self.assertRaisesRegex(ContractError, 'invalidated'):
                binding.validate()

    def producer_reads_forbidden(self):
        original = Path.read_bytes
        def read(path):
            if path.name == 'native_model_update.py':
                raise AssertionError('runtime producer authentication')
            return original(path)
        return patch.object(Path, 'read_bytes', read)

    def test_legacy_schema_producer_is_ignored_but_permissions_are_checked(self):
        from libero.gpu.host.runtime.core.native_model_update import publish_native_model_write_schema
        with self.producer_reads_forbidden():
            schema = publish_native_model_write_schema(self.built)
            self.assertNotIn('producer', schema)
            self.built.metadata['native_write_schema']['producer'] = {
                'path': '/retired/native_model_update.py', 'sha256': 'old'}
            describe_native_model(self.built)
            self.built.metadata['native_write_schema']['topology_writable'] = True
            with self.assertRaisesRegex(ContractError, 'policy differs'):
                describe_native_model(self.built)

    def test_model_update_transaction_does_not_authenticate_producer(self):
        from libero.gpu.host.runtime.core.native_model_update import prepare_native_model_update
        with product_build(), self.producer_reads_forbidden():
            old = prepare_native_intake(self.built)
            self.built._gpu_native_intake_binding = old
            timestep = self.model.opt.timestep
            plan = prepare_native_model_update(self.built, {}, {'timestep': timestep * 2})
            self.assertNotIn('producer_sha256', plan.preflight)
            self.assertEqual(self.model.opt.timestep, timestep)
            self.assertEqual(plan.built.native_model.opt.timestep, timestep * 2)
            self.assertIsNot(plan.built.native_model, self.model)
            plan.validate_base()
            self.assertIs(plan.mark_committed(), plan.built)
            self.assertFalse(old.active)
            with self.assertRaisesRegex(ContractError, 'no longer prepared'):
                plan.validate_base()
            with self.assertRaisesRegex(ContractError, 'cannot be discarded'):
                plan.discard()

    def test_model_update_rejects_readonly_and_invalid_parameter_values(self):
        import numpy as np
        from libero.gpu.host.runtime.core.native_model_update import prepare_native_model_update
        with self.producer_reads_forbidden():
            for field, payload, message in (
                    ('jnt_qposadr', self.model.jnt_qposadr.copy(), 'readonly/structural'),
                    ('body_mass', -np.ones_like(self.model.body_mass), 'nonnegative'),
                    ('body_mass', self.model.body_mass.astype(np.float32), 'shape/dtype')):
                with self.subTest(field=field, message=message):
                    with self.assertRaisesRegex(ContractError, message):
                        prepare_native_model_update(self.built, {field: payload})


if __name__ == '__main__':
    unittest.main()
