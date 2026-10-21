"""Static-only model preparation for a private cold arena transaction."""
from types import SimpleNamespace
import hashlib
import copy
import uuid
import numpy as np


def prepare_parameter_model(bundle, built, changes, token, model_generation):
    """Validate actual payload and prepare a fresh native/static identity.

    No MjData, CPU forward, solver, state integration or dynamic truth is used.
    The returned native model remains private until the arena transaction ends.
    """
    import mujoco
    from libero.gpu.model.factory import native_model_metadata, BuiltTaskModel
    from libero.gpu.host.runtime.core.generation import derive_native_parameter_spec, prepare_parameter_descriptor
    from libero.gpu.host.runtime.core.contracts import ModelSpec
    from libero.gpu.host.runtime.constraints.identity import STATIC_SCHEMA
    from libero.gpu.host.runtime.constraints.specialization import generate_specialization
    if set(changes) != set(token.fields) or not changes or set(changes)-{'body_pos','body_quat'}:
        raise ValueError('cold physics transaction must match declared body parameter token')
    model=built.native_model
    allowed=set(int(i) for i in built.domain_shell.obj_body_id.values())
    payload={}
    for name,value in changes.items():
        value=np.asarray(value)
        source=np.asarray(getattr(model,name))
        if value.dtype != np.float64 or value.shape != source.shape or not np.isfinite(value).all():
            raise ValueError('body parameter payload dtype/shape/finite mismatch: '+name)
        altered=np.flatnonzero(np.any(np.ascontiguousarray(value).view(np.uint64) !=
                                    np.ascontiguousarray(source).view(np.uint64),axis=1))
        if set(map(int,altered))-allowed:
            raise ValueError('body parameter update includes undeclared fixture/object IDs')
        if name=='body_quat' and not np.allclose(np.sum(value*value,axis=1),1.0,rtol=0,atol=1e-12):
            raise ValueError('body quaternion source must be unit normalized')
        payload[name]=np.array(value,copy=True,order='C')
    digests={name:hashlib.sha256(str(a.dtype).encode()+repr(a.shape).encode()+a.tobytes()).hexdigest()
             for name,a in payload.items()}
    if model_generation.prepare_parameter_update(digests) != token:
        raise ValueError('actual cold payload differs from prepared parameter digest')
    # The native decoder computes fresh all-array/header digests. Existing task
    # naming/robot selector metadata remains explicit source metadata.
    native=copy.copy(bundle.model_pack.model)
    actual_base=native_model_metadata(native)
    actual_spec=ModelSpec.from_factory_metadata(actual_base,features=bundle.model_spec.features)
    if actual_spec.identity_key != bundle.model_spec.identity_key:
        raise ValueError("loaded continuation base MJB differs from current native identity")
    for name,value in payload.items():getattr(native,name)[:]=value
    metadata=native_model_metadata(native)
    for name,value in built.metadata.items():
        if name not in metadata:metadata[name]=value
    derived,preflight=derive_native_parameter_spec(bundle.model_spec,built.metadata,metadata,
                                                  declared_fields=tuple(changes))
    plan=prepare_parameter_descriptor(model_generation,token,derived,metadata_preparation=preflight)
    from libero.gpu.host.runtime.core.product_paths import build_directory, current_build_lifetime
    directory=None
    from ..core.model_bytes import model_bytes
    digest=hashlib.sha256(model_bytes(native)).hexdigest()
    if digest != derived.model_sha256:raise ValueError('actual parameter model digest differs')
    static={name:np.ascontiguousarray(getattr(native,name)).copy() for name in STATIC_SCHEMA}
    facts={name:int(getattr(native,name)) for name in ('neq','ntendon','nplugin','nmocap','na','nq','nv','nu','nM','nbody','ngeom','nsite','njnt')}
    facts.update({name:int(getattr(native.opt,name)) for name in ('solver','cone','jacobian','disableflags','enableflags','noslip_iterations')})
    facts.update({name:float(getattr(native.opt,name)) for name in ('timestep','impratio')})
    manifest=dict(model_sha256=digest,model_facts=facts,
        dimensions=dict(metadata['dimensions'],nconmax=int(native.nconmax)),mujoco_version=mujoco.__version__,
        static={name:dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest()) for name,a in static.items()})
    from libero.gpu.host.runtime.core.product_paths import current_build_project, record_build
    project = current_build_project()
    if project is None:
        directory=build_directory('parameter')
        package=directory/('parameter_s_'+uuid.uuid4().hex)
        generate_specialization(manifest,static,package)
    else:
        from libero.gpu.host.runtime.core.specialization_cache import generated_package
        package, _, hit = generated_package(manifest, static, generate_specialization,
                                            project['root'] / 'specializations')
        record_build('constraint-source', package.name,
                     () if hit else (('generate-specialization', str(package)),))
    fresh=BuiltTaskModel(native,built.xml,built.domain_shell,metadata)
    return SimpleNamespace(built=fresh,spec=derived,constraint_package=package,
        descriptor_plan=plan,preflight=preflight,directory=directory)
