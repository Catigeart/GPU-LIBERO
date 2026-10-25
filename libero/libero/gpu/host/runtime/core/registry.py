"""Runtime implementation domains; numerical review is an offline concern."""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Mapping
from types import MappingProxyType
from .contracts import ContractError, ModelSpec, _count, _digest


@dataclass(frozen=True)
class KernelCandidate:
    candidate_id: str
    component: str
    source_sha256: str | None
    algorithm: str
    layout_version: str
    architecture: str
    math_profile: str
    author: str
    specialization_keys: tuple[str, ...] = ()
    model_sha256s: tuple[str, ...] = ()
    required_features: tuple[str, ...] = ()
    forbidden_features: tuple[str, ...] = ()
    dimension_equalities: Mapping[str, int] = field(default_factory=dict)
    maximum_counts: Mapping[str, int] = field(default_factory=dict)
    mutable_parameters: tuple[str, ...] = ()
    source_files: Mapping[str, str] = field(default_factory=dict)
    execution_scope: str = 'component-runtime'
    generation_rebind_fields: tuple[str, ...] = ()

    def __post_init__(self):
        if not all((self.candidate_id, self.component, self.layout_version, self.architecture)):
            raise ContractError('candidate identity, component, layout and architecture required')
        for name in ('specialization_keys', 'model_sha256s', 'required_features', 'forbidden_features', 'mutable_parameters', 'generation_rebind_fields'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        source_files = dict(self.source_files)
        # Historical provenance fields are inert at runtime; offline tools verify them.
        object.__setattr__(self, 'source_files', MappingProxyType(source_files))
        if not self.execution_scope:
            raise ContractError('candidate execution scope required')
        if not (self.specialization_keys or self.model_sha256s):
            raise ContractError('candidate needs explicit model/specialization scope')
        for key in self.specialization_keys + self.model_sha256s:
            _digest(key, 'candidate scope')
        for name, count in dict(self.dimension_equalities, **MappingProxyType(dict(self.maximum_counts))).items():
            _count(count, name)
        for mapping in (self.dimension_equalities, self.maximum_counts):
            if any(name in {'qpos', 'qvel', 'qacc', 'time', 'ctrl', 'warmstart', 'nefc', 'ncon'} for name in mapping):
                raise ContractError('live values/counts are not structural admission predicates')
        object.__setattr__(self, 'dimension_equalities', MappingProxyType(dict(self.dimension_equalities)))
        object.__setattr__(self, 'maximum_counts', MappingProxyType(dict(self.maximum_counts)))

    def to_dict(self):
        result = {}
        for name, value in vars(self).items():
            if isinstance(value, Mapping):
                result[name] = dict(value)
            elif isinstance(value, tuple):
                result[name] = list(value)
            else:
                result[name] = value
        return result

    def structural_reasons(self, model: ModelSpec):
        reasons = []
        if model.layout_version != self.layout_version:
            reasons.append('layout version differs')
        if self.specialization_keys and model.specialization_key not in self.specialization_keys:
            reasons.append('specialization outside candidate scope')
        if self.model_sha256s and model.model_sha256 not in self.model_sha256s:
            reasons.append('model outside candidate scope')
        if set(self.required_features) - set(model.features):
            reasons.append('required structural features absent')
        if set(self.forbidden_features) & set(model.features):
            reasons.append('unsupported structural feature present')
        dimensions = dict(model.dimensions.as_dict(), **dict(model.extra_dimensions))
        for name, value in self.dimension_equalities.items():
            if dimensions.get(name) != value:
                reasons.append(f'{name}: dimension differs')
        for name, maximum in self.maximum_counts.items():
            if name not in dimensions or dimensions[name] > maximum:
                reasons.append(f'{name}: structural capacity insufficient')
        return tuple(reasons)


@dataclass(frozen=True)
class Admission:
    candidate_id: str
    model_identity_key: str
    structural_ready: bool
    execution_allowed: bool
    mode: str
    reasons: tuple[str, ...]


class CapabilityRegistry:
    def __init__(self):
        self.candidates = {}

    @classmethod
    def from_document(cls, document):
        """Load candidates, never trust cached execution/accepted flags in a receipt."""
        registry = cls()
        entries = document.get('entries', [document])
        for row in entries:
            payload = row.get('candidate')
            if payload is None:
                raise ContractError('full candidate descriptor required, receipt alone cannot license execution')
            try:
                candidate = KernelCandidate(**payload)
            except TypeError as exc:
                raise ContractError('candidate descriptor fields differ from schema') from exc
            if row.get('candidate_id', candidate.candidate_id) != candidate.candidate_id:
                raise ContractError('candidate receipt id differs from descriptor')
            registry.register_candidate(candidate)
        # Numerical records are registered separately from capabilities.
        return registry

    @classmethod
    def from_file(cls, path, expected_sha256=None):
        raw = Path(path).read_bytes()
        if expected_sha256 is not None:
            _digest(expected_sha256, 'registry manifest')
            if hashlib.sha256(raw).hexdigest() != expected_sha256:
                raise ContractError('registry manifest content differs from sealed SHA256')
        return cls.from_document(json.loads(raw))

    def register_candidate(self, candidate: KernelCandidate):
        prior = self.candidates.get(candidate.candidate_id)
        if prior is not None and prior != candidate:
            raise ContractError('candidate id cannot change source/capability; use new id')
        self.candidates[candidate.candidate_id] = candidate

    def admit(self, candidate_id, model: ModelSpec, *, mode='production', parameter_fields=(), required_scope=None):
        """Check executable structure; development verdicts never gate runtime."""
        if mode not in ('production', 'diagnostic'):
            raise ContractError('admission mode must be production/diagnostic')
        candidate = self.candidates.get(candidate_id)
        if candidate is None:
            return Admission(candidate_id, model.identity_key, False, False, mode,
                             ('candidate not registered',))
        reasons = list(candidate.structural_reasons(model))
        if required_scope is not None and candidate.execution_scope != required_scope:
            reasons.append('candidate execution scope is insufficient')
        if set(parameter_fields) - set(candidate.mutable_parameters):
            reasons.append('parameter change outside candidate capability')
        return Admission(candidate_id, model.identity_key, not reasons, not reasons, mode, tuple(reasons))

    def admit_generation(self, candidate_id, base: ModelSpec, derived: ModelSpec, *,
                         parameter_fields=(), required_scope='component-runtime'):
        """Check both model domains and an explicitly declared parameter update.

        Actual field/topology validation belongs to the model-update boundary.
        A development report cannot extend the implementation's model domain.
        """
        before = self.admit(candidate_id, base, parameter_fields=parameter_fields, required_scope=required_scope)
        after = self.admit(candidate_id, derived, parameter_fields=parameter_fields, required_scope=required_scope)
        reasons = list(dict.fromkeys(before.reasons + after.reasons))
        candidate = self.candidates.get(candidate_id)
        if not base.shares_specialization(derived):
            reasons.append('generation changed structural specialization')
        if base.identity_key != derived.identity_key and not parameter_fields:
            reasons.append('generation requires explicit parameter fields')
        allowed_fields = set(base.mutable_parameters) & set(derived.mutable_parameters)
        if candidate is not None:
            allowed_fields &= set(candidate.generation_rebind_fields)
        if set(parameter_fields) - allowed_fields:
            reasons.append('generation parameter fields outside supported scope')
        return Admission(candidate_id, derived.identity_key, not reasons, not reasons,
                         'production-generation', tuple(reasons))

    def support_row(self, task_id, model, components: Mapping[str, str],
                    required_components=('F', 'D', 'C', 'S', 'N', 'Euler', 'observables', 'environment', 'renderer')):
        entries = {component: self.admit(candidate_id, model, required_scope='component-runtime')
                   for component, candidate_id in components.items()}
        missing = tuple(sorted(set(required_components) - set(components)))
        return {'task_id': task_id, 'model_identity_key': model.identity_key,
                'specialization_key': model.specialization_key,
                'status': 'supported' if entries and not missing and all(a.execution_allowed for a in entries.values()) else 'unsupported',
                'missing_components': list(missing),
                'components': {name: {'candidate_id': a.candidate_id, 'structural_ready': a.structural_ready,
                                     'execution_allowed': a.execution_allowed, 'reasons': list(a.reasons)}
                               for name, a in entries.items()}}
