"""Explicit registry gate for independently qualified Euler component dispatch.

A component qualification does not license a complete LIBERO episode. The
caller must also satisfy full-runtime admission for all other required stages.
"""
import hashlib
import json
from pathlib import Path
from .runtime_adapter import ResidentEulerAdapter
from .specialization import validate_specialization


def _qualified_fixture_digest(record, model_sha256, fixture_sha256):
    """Follow only SHA-bound JSON evidence links from independent acceptance."""
    seen=set()
    def visit(path,digest,depth):
        if depth>6 or (path,digest) in seen:return False
        seen.add((path,digest));raw=Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=digest:
            raise ValueError('Euler acceptance evidence link changed')
        data=json.loads(raw)
        if data.get('model_sha256')==model_sha256 and data.get('fixture_files',{}).get('FIXTURE.json')==fixture_sha256:
            return True
        def links(value):
            if isinstance(value,dict):
                if isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str) and value['path'].endswith('.json'):
                    yield value['path'],value['sha256']
                for item in value.values():yield from links(item)
            elif isinstance(value,list):
                for item in value:yield from links(item)
        return any(visit(next_path,next_digest,depth+1) for next_path,next_digest in links(data))
    return visit(record.report_path,record.report_sha256,0)


def workspace_array_bytes(specialization, capacity):
    """Exact owned-array storage; excludes context/cache and caller buffers."""
    d=specialization['dimensions']
    nq,nv,nm,nj,nu,ns=(d[n] for n in ('nq','nv','nM','njnt','nu','nsensordata'))
    nquat=nq-nv  # valid after unique joint-span validation
    per_world=8*(2*nq+19*nv+4*nm+2+12*nquat+nu+ns)+4*(8+3*nquat+2*nv)
    static=8*nv+4*(3*nj+3*nv)
    return static+int(capacity)*per_world


def qualified_euler_adapter(model, model_spec, registry, candidate_id, *,
                            fixture_manifest, fixture_sha256, capacity=1,
                            device='cuda:0', parameter_fields=(), max_array_bytes=None):
    """Construct only after exact source/model and Euler-static admission.

    fixture_sha256 must come from the independent acceptance evidence, not
    be computed from an untrusted current fixture by the caller. Actual native
    topology/damping/timestep are validated again before CUDA allocation.
    body_pos/body_quat reset changes are handled by the outer runtime's model
    generation contract and do not alter Euler's own static arrays.
    """
    if isinstance(capacity,bool) or int(capacity)!=capacity or not 0<capacity<=2147483647:
        raise ValueError('Euler world capacity must fit positive int32')
    admission=registry.admit(candidate_id, model_spec, mode='production',
                             parameter_fields=parameter_fields,
                             required_scope='component-runtime')
    if not admission.execution_allowed:
        raise ValueError('Euler component admission denied: '+ '; '.join(admission.reasons))
    candidate=registry.candidates[candidate_id]
    if candidate.component!='Euler':raise ValueError('wrong component candidate')
    record = registry.qualified_record(candidate_id, model_spec, parameter_fields=parameter_fields)
    if not _qualified_fixture_digest(record,model_spec.model_sha256,fixture_sha256):
        raise ValueError('Euler fixture is not bound by independent acceptance evidence')
    manifest=Path(fixture_manifest);raw=manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=fixture_sha256:
        raise ValueError('Euler independently sealed fixture changed')
    sealed=json.loads(raw)
    if sealed['model_sha256']!=model_spec.model_sha256:
        raise ValueError('Euler fixture model does not match admitted model')
    actual,_=validate_specialization(model)
    if actual['specialization_key']!=sealed['specialization']['specialization_key']:
        raise ValueError('actual native Euler static parameters differ from qualification')
    expected_dims=dict(model_spec.dimensions.as_dict(),**dict(model_spec.extra_dimensions))
    if any(expected_dims.get(n)!=v for n,v in actual['dimensions'].items()):
        raise ValueError('actual native Euler dimensions differ from admitted descriptor')
    array_bytes=workspace_array_bytes(actual,capacity)
    if max_array_bytes is not None and array_bytes>max_array_bytes:
        raise MemoryError('Euler owned-array reservation exceeds supplied budget')
    from .generic_workspace import GenericEulerWorkspace
    workspace=GenericEulerWorkspace(model,capacity,device,mode='diagnostic')
    # Internal class retains its candidate identity; this adapter records the
    # successful external admission instead of changing old kernel licences.
    adapter=ResidentEulerAdapter(workspace=workspace)
    adapter.admission=admission
    adapter.qualification_scope='component-runtime'
    adapter.euler_static_specialization_key=actual['specialization_key']
    adapter.fixture_manifest_sha256=fixture_sha256
    adapter.owned_array_bytes=array_bytes
    return adapter




