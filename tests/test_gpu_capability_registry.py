"""Runtime capabilities survive removal of development source/report files."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from libero.gpu.host.runtime.core.contracts import ContractError, ModelDimensions, ModelSpec, _hash
from libero.gpu.host.runtime.core.registry import CapabilityRegistry, KernelCandidate
from libero.gpu.host.runtime.core.qualification import AcceptanceRecord, QualifiedRegistry
from libero.gpu.host.runtime.core.qualification import load_qualified_registry


class CapabilityRegistryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = ModelSpec(ModelDimensions(2, 2, 1, 0, 3, 2, 1, 0, 2),
                               '1'*64, '2'*64, '3'*64, {})
        self.source = self.root/'kernel.cu'
        self.source.write_text('offline source')
        files = {str(self.source): hashlib.sha256(self.source.read_bytes()).hexdigest()}
        self.candidate = KernelCandidate('fixture', 'D', _hash(files), 'fixture',
            self.model.layout_version, 'sm80', 'fp64', 'author',
            specialization_keys=(self.model.specialization_key,), source_files=files,
            maximum_counts={'nv': 2}, mutable_parameters=('body_pos',),
            generation_rebind_fields=('body_pos',))
        self.report = self.root/'report.json'
        self.report.write_text('offline numerical result')
        self.record = AcceptanceRecord('fixture', self.candidate.source_sha256,
            self.model.identity_key, 'accepted', hashlib.sha256(self.report.read_bytes()).hexdigest(),
            str(self.report), 'reviewer', True, True, True, 'fixture',
            permitted_parameter_fields=('body_pos',))

    def registry(self):
        registry = CapabilityRegistry()
        registry.register_candidate(self.candidate)
        return registry

    def test_registration_and_execution_do_not_open_development_files(self):
        self.source.unlink()
        self.report.unlink()
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('runtime filesystem read')):
            registry = self.registry()
            self.assertTrue(registry.admit('fixture', self.model).execution_allowed)
            self.assertTrue(registry.admit('fixture', self.model, parameter_fields=('body_pos',)).execution_allowed)
            self.assertFalse(registry.admit('fixture', self.model, parameter_fields=('body_mass',)).execution_allowed)
            self.assertFalse(registry.admit('fixture', self.model, required_scope='other').execution_allowed)

    def test_numerical_verdicts_do_not_change_structural_execution(self):
        registry = QualifiedRegistry()
        registry.register_candidate(self.candidate)
        for mode in ('production', 'diagnostic'):
            self.assertTrue(registry.admit('fixture', self.model, mode=mode).execution_allowed)
        registry.record_acceptance(replace(self.record, status='rejected'))
        self.assertTrue(registry.admit('fixture', self.model).execution_allowed)
        with self.assertRaisesRegex(ContractError, 'qualification absent'):
            registry.qualified_record('fixture', self.model)
        changed = replace(self.model, dimensions=replace(self.model.dimensions, nv=3))
        admission = registry.admit('fixture', changed)
        self.assertFalse(admission.execution_allowed)
        self.assertIn('nv: structural capacity insufficient', admission.reasons)
        self.assertFalse(hasattr(CapabilityRegistry(), 'acceptances'))

    def test_generation_requires_explicit_supported_fields_and_both_model_domains(self):
        registry = self.registry()
        derived = replace(self.model, model_sha256='4'*64)
        self.assertFalse(registry.admit_generation('fixture', self.model, derived).execution_allowed)
        self.assertTrue(registry.admit_generation('fixture', self.model, derived,
                        parameter_fields=('body_pos',)).execution_allowed)
        self.assertFalse(registry.admit_generation('fixture', self.model, derived,
                         parameter_fields=('body_mass',)).execution_allowed)
        fixed = replace(self.candidate, candidate_id='fixed', specialization_keys=(),
                        model_sha256s=(self.model.model_sha256,))
        registry.register_candidate(fixed)
        self.assertFalse(registry.admit_generation('fixed', self.model, derived,
                         parameter_fields=('body_pos',)).execution_allowed)
        changed = replace(derived, topology_sha256='5'*64)
        self.assertFalse(registry.admit_generation('fixture', self.model, changed,
                         parameter_fields=('body_pos',)).execution_allowed)

    def test_source_metadata_is_not_a_runtime_execution_requirement(self):
        candidate = replace(self.candidate, source_sha256=None, source_files={}, author='')
        registry = CapabilityRegistry.from_document({'candidate': candidate.to_dict()})
        self.assertTrue(registry.admit('fixture', self.model).execution_allowed)
        self.assertFalse(registry.admit('fixture', self.model,
                                       parameter_fields=('body_mass',)).execution_allowed)
        qualified = QualifiedRegistry()
        qualified.register_candidate(candidate)
        with self.assertRaisesRegex(ContractError, 'acceptance source'):
            qualified.record_acceptance(self.record)
        malformed = replace(candidate, source_files=self.candidate.source_files)
        registry = CapabilityRegistry.from_document({'candidate': malformed.to_dict()})
        self.assertTrue(registry.admit('fixture', self.model).execution_allowed)
        with self.assertRaisesRegex(ContractError, 'bundle digest'):
            qualified.register_candidate(malformed)

    def test_explicit_offline_qualification_still_verifies_sources_and_reports(self):
        path = self.root/'qualification.json'
        raw = json.dumps({'candidate': self.candidate.to_dict(), 'acceptance': asdict(self.record)}).encode()
        path.write_bytes(raw)
        pin = hashlib.sha256(raw).hexdigest()
        registry = load_qualified_registry(path, pin)
        self.assertTrue(registry.admit('fixture', self.model).execution_allowed)
        self.report.write_text('changed report')
        with self.assertRaisesRegex(ContractError, 'independent report content differs'):
            load_qualified_registry(path, pin)
        self.source.unlink()
        with self.assertRaisesRegex(ContractError, 'candidate source unavailable'):
            load_qualified_registry(path, pin)


if __name__ == '__main__':
    unittest.main()
