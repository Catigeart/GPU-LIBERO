"""Cold loading of an explicitly pinned independent acceptance receipt."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass
from .contracts import ContractError, _digest, _hash
from .registry import CapabilityRegistry


@dataclass(frozen=True)
class AcceptanceRecord:
    candidate_id: str
    source_sha256: str
    model_identity_key: str
    status: str
    report_sha256: str
    report_path: str
    reviewer: str
    numerical_pass: bool
    actual_writes_pass: bool
    provenance_pass: bool
    scope: str
    permitted_parameter_fields: tuple[str, ...] = ()
    execution_scope: str = 'component-runtime'
    generation_rebind_fields: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ('source_sha256', 'model_identity_key', 'report_sha256'):
            _digest(getattr(self, name), name)
        if not all(isinstance(value, bool) for value in (self.numerical_pass, self.actual_writes_pass, self.provenance_pass)):
            raise ContractError('acceptance verdicts must be booleans')
        object.__setattr__(self, 'permitted_parameter_fields', tuple(self.permitted_parameter_fields))
        if self.status not in ('pending', 'accepted', 'rejected'):
            raise ContractError('unknown acceptance status')
        if not all((self.report_path, self.reviewer, self.scope)):
            raise ContractError('independent report path/reviewer/scope required')
        if self.status == 'accepted' and not all((self.numerical_pass, self.actual_writes_pass, self.provenance_pass)):
            raise ContractError('acceptance requires numerical, writes and provenance qualification')


@dataclass(frozen=True)
class QualifiedGenerationBridge:
    """Independent native immutable/readset equivalence and legal mutation witness."""
    candidate_id: str
    source_sha256: str
    base_model_identity: str
    derived_model_identity: str
    changed_fields: tuple[str, ...]
    report_path: str
    report_sha256: str
    reviewer: str
    immutable_arrays_equal: bool
    topology_equal: bool
    options_equal: bool
    parameter_rules_pass: bool

    def __post_init__(self):
        for name in ('source_sha256', 'base_model_identity', 'derived_model_identity', 'report_sha256'):
            _digest(getattr(self, name), name)
        if not self.report_path or not self.reviewer:
            raise ContractError('generation bridge requires independent report')
        object.__setattr__(self, 'changed_fields', tuple(sorted(set(self.changed_fields))))
        verdicts = (self.immutable_arrays_equal, self.topology_equal, self.options_equal, self.parameter_rules_pass)
        if not all(isinstance(value, bool) and value for value in verdicts):
            raise ContractError('generation bridge requires exact immutable/topology/options equality and legal source rules')


class QualifiedRegistry(CapabilityRegistry):
    """Offline review records, separate from inherited structural execution checks."""
    def __init__(self):
        super().__init__()
        self.acceptances = {}
        self.generation_bridges = {}

    def register_candidate(self, candidate):
        if candidate.source_sha256 is not None:
            _digest(candidate.source_sha256, 'candidate source')
        if candidate.source_files and _hash(dict(candidate.source_files)) != candidate.source_sha256:
            raise ContractError('candidate source bundle digest differs from file manifest')
        super().register_candidate(candidate)

    def record_acceptance(self, record: AcceptanceRecord):
        """Attach a verdict; the offline qualification loader verifies its files."""
        candidate = self.candidates.get(record.candidate_id)
        if candidate is None or candidate.source_sha256 != record.source_sha256:
            raise ContractError('acceptance source does not match registered candidate')
        if record.execution_scope != candidate.execution_scope:
            raise ContractError('acceptance execution scope differs from candidate')
        if record.status == 'accepted' and record.reviewer == candidate.author:
            raise ContractError('candidate author cannot independently accept own candidate')
        if set(record.permitted_parameter_fields) - set(candidate.mutable_parameters):
            raise ContractError('acceptance claims parameter changes outside candidate capability')
        self.acceptances[(record.candidate_id, record.model_identity_key)] = record

    def record_generation_bridge(self, bridge: QualifiedGenerationBridge):
        candidate = self.candidates.get(bridge.candidate_id)
        if candidate is None or candidate.source_sha256 != bridge.source_sha256:
            raise ContractError('generation bridge candidate/source differs')
        if bridge.reviewer == candidate.author:
            raise ContractError('candidate author cannot independently qualify generation bridge')
        if set(bridge.changed_fields) - set(candidate.generation_rebind_fields):
            raise ContractError('candidate does not support these generation source rebindings')
        self.generation_bridges[(bridge.candidate_id, bridge.base_model_identity,
                                 bridge.derived_model_identity)] = bridge

    def qualified_record(self, candidate_id, model, *, parameter_fields=()):
        record = self.acceptances.get((candidate_id, model.identity_key))
        if (record is None or record.status != 'accepted' or
                set(parameter_fields) - set(record.permitted_parameter_fields)):
            raise ContractError('independent numerical qualification absent')
        return record


def _verify_file(path, expected, label):
    try:
        actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as exc:
        raise ContractError(f'{label} unavailable: {path}') from exc
    if actual != expected:
        raise ContractError(f'{label} content differs: {path}')


def load_qualified_registry(path, expected_sha256, *, required_scope=None):
    """Load authored independent records, rechecking reports and actual sources.

    Unlike ordinary from_file, this is an explicit qualification operation: the
    caller must pin the reviewed receipt SHA. Cached admitted flags are ignored.
    It does not authorize fields, models or scopes outside each exact record.
    """
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ContractError('qualified receipt content differs from trusted pin')
    document = json.loads(raw)
    registry = QualifiedRegistry.from_document(document)
    for row in document.get('entries', [document]):
        candidate = registry.candidates[row['candidate']['candidate_id']]
        for source, expected in candidate.source_files.items():
            _verify_file(source, expected, 'candidate source')
        payloads = row.get('acceptance_records')
        if payloads is None:
            payload = row.get('acceptance')
            payloads = [] if payload is None else [payload]
        if not payloads:
            raise ContractError('explicit independent acceptance records required')
        for payload in payloads:
            try:
                record = AcceptanceRecord(**payload)
            except TypeError as exc:
                raise ContractError('acceptance record schema mismatch') from exc
            if record.candidate_id != candidate.candidate_id:
                raise ContractError('acceptance candidate identity differs from receipt')
            if required_scope is not None and record.execution_scope != required_scope:
                raise ContractError('qualification scope differs from requested scope')
            _verify_file(record.report_path, record.report_sha256, 'independent report')
            registry.record_acceptance(record)
    return registry
