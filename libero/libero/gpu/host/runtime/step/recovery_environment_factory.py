"""Real raw-native candidate factory for model-only task environments.

Only diagnostic execution is requested here. Source admission and numerical
qualification belong to the runtime owners; this factory grants neither.
"""
import numpy as np
from libero.gpu.host.runtime.core.model_bytes import actual_decoder_record
import hashlib
from pathlib import Path
from collections.abc import Mapping
from libero.gpu.host.adapters.environment.assembly import assemble_single_world_task
from libero.gpu.host.adapters.environment.resident_runtime import ResidentPhysicsFacadeRuntime, fixed_root_descriptors, fixed_contact_reader
from libero.gpu.host.adapters.environment.facade import UnsupportedGpuOperation


def default_integration_state(model):
    """Original reset defaults, derived from compiled model rather than forward."""
    if int(model.na) or int(model.nmocap):
        raise UnsupportedGpuOperation('activation/mocap reset family requires explicit runtime support')
    return {
        'qpos':np.array(model.qpos0,dtype=np.float64,copy=True,order='C').reshape(1,int(model.nq)),
        'qvel':np.zeros((1,int(model.nv)),np.float64),
        'time':np.zeros(1,np.float64),
        'ctrl':np.zeros((1,int(model.nu)),np.float64),
        'qacc_warmstart':np.zeros((1,int(model.nv)),np.float64),
        'qfrc_applied':np.zeros((1,int(model.nv)),np.float64),
        'xfrc_applied':np.zeros((1,int(model.nbody),6),np.float64),
    }




def _native_intake_package(built):
    """Pure exact native static intake; no MjData or CPU physics."""
    import json,tempfile,uuid
    from libero.gpu.host.runtime.core.generation import model_spec_for_native_metadata
    from libero.gpu.host.runtime.constraints.identity import STATIC_SCHEMA
    from libero.gpu.host.runtime.constraints.specialization import generate_specialization
    from .shape_schema import RuntimeShape
    RuntimeShape.from_native_metadata(built.metadata)
    model=built.native_model
    bddl=Path(built.metadata["bddl_file_name"])
    task=bddl.parent.name+"/"+bddl.stem
    review=Path("/srv/libero-split-receive-20261004-r01/alltasks-evidence/20261009-r01/review-env/digest-scheme-r01/RESULT.json")
    spec=model_spec_for_native_metadata(built.metadata,review,
        "87f40f6f3e9c9ff4e6acd72c24362e9e3de7a5aedfccfd1178f5c143dc9c658f",task)
    static={name:np.ascontiguousarray(getattr(model,name)).copy() for name in STATIC_SCHEMA}
    facts={name:int(getattr(model,name)) for name in ("neq","ntendon","nplugin","nmocap","na","nq","nv","nu","nM","nbody","ngeom","nsite","njnt")}
    facts.update({name:int(getattr(model.opt,name)) for name in ("solver","cone","jacobian","disableflags","enableflags","noslip_iterations")})
    facts.update({name:float(getattr(model.opt,name)) for name in ("timestep","impratio")})
    manifest=dict(model_sha256=spec.model_sha256,model_facts=facts,
        dimensions=dict(built.metadata["dimensions"],nconmax=int(model.nconmax)),
        static={name:dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest()) for name,a in static.items()})
    directory=Path(tempfile.mkdtemp(prefix="shape-intake-",dir=Path("/proc/self/cwd/tmp").resolve()))
    package=directory/("shape_s_"+uuid.uuid4().hex)
    generate_specialization(manifest,static,package)
    (directory/"INTAKE_SOURCE.json").write_text(json.dumps(dict(task=task,model_sha256=spec.model_sha256,model_identity=spec.identity_key,cpu_physics=False,production=False),indent=2))
    return spec,package

def create(built):
    """Callable runtime-factory entry, requiring actual owner modules/callbacks."""
    from libero.gpu.host.runtime.core.namespace import require_canonical_gpu_namespace
    require_canonical_gpu_namespace()
    try:
        from libero.gpu.host.runtime.step.recovery_wide_factory import prepare_wide_recovery_physics as prepare_shape_physics
        from libero.gpu.host.runtime.step.recovery_environment import make_boundary_callbacks
    except ModuleNotFoundError as exc:
        if exc.name in ('libero.gpu.host.runtime.step.shape_prepare',
                        'libero.gpu.host.runtime.step.shape_environment'):
            raise UnsupportedGpuOperation('actual raw-native runtime factory is not yet installed') from exc
        raise
    decoder_record=actual_decoder_record(built)
    built.metadata['decoder_record']=decoder_record
    state=default_integration_state(built.native_model)
    spec,package=_native_intake_package(built)
    prepared=prepare_shape_physics(built,state=state,mode='diagnostic',with_controller=True,
        model_spec=spec,constraint_package=package)
    return assemble_prepared(built,prepared)


def assemble_prepared(built,prepared,*,observation_continuation=None):
    """Bind an already-prepared actual arena; optionally restore old observation bytes.

    The caller owns physical continuation restore and must do it before calling.
    This function never resets, advances, samples, or allocates a physics arena.
    """
    from libero.gpu.host.runtime.step.recovery_environment import make_boundary_callbacks
    from libero.gpu.host.adapters.observations.adapter import GpuObservationAdapter
    from libero.gpu.host.adapters.observations.runtime_binding import FixedRuntimeObservationBinding
    callbacks=None;observer=None
    try:
        callbacks=make_boundary_callbacks(prepared,built)
        names=('reset','step1','state_written','synchronize_and_check','close_runtime',
               'evidence','update_parameters','update_visual')
        if not isinstance(callbacks,Mapping) or any(not callable(callbacks.get(name)) for name in names):
            raise UnsupportedGpuOperation('actual boundary callbacks must provide all declared functions')
        if prepared.model_pack.model is not built.native_model:
            raise UnsupportedGpuOperation('raw prepared model owner differs from compiled task')
        observer=GpuObservationAdapter(built.native_model,built.metadata,1,
                                       device=prepared.stream.device)
        if observation_continuation is not None:
            observer.restore_continuation(observation_continuation,stream=prepared.stream)
        binding=FixedRuntimeObservationBinding(prepared,observer,expected_observe=prepared.runtime.observe)
        def visual(changes,world,stream):
            callbacks['update_visual'](changes,world,stream)
            observer.update_visual_parameters(changes,world=world,stream=stream)
        boundary=ResidentPhysicsFacadeRuntime(
            runtime=prepared.runtime,fields=fixed_root_descriptors(prepared),stream=prepared.stream,
            synchronize_and_check=callbacks['synchronize_and_check'],close_runtime=callbacks['close_runtime'],
            state_written=callbacks['state_written'],reset=callbacks['reset'],step1=callbacks['step1'],
            update_parameters=callbacks['update_parameters'],update_visual=visual,
            update_parameters_transaction=callbacks.get('update_parameters_transaction'),
            special_readers={'contact':fixed_contact_reader(prepared)},evidence=callbacks['evidence'],
        )
        components=assemble_single_world_task(built,prepared,boundary,binding,prepared.model_spec)
        components.parameter_replacement_binder=callbacks.get('bind_parameter_replacement')
        return components
    except BaseException as exc:
        cleanup=[]
        if observer is not None:cleanup.append(observer.close)
        release=callbacks.get('close_runtime') if isinstance(callbacks,Mapping) else None
        if not callable(release):release=getattr(prepared,'close',None)
        if callable(release):cleanup.append(release)
        cleanup_errors=[]
        for close in cleanup:
            try:close()
            except BaseException as close_error:cleanup_errors.append(repr(close_error))
        if cleanup_errors:exc.gpu_cleanup_errors=cleanup_errors
        raise

def rebuild_components(built,previous):
    """Preflight host layout, release old arena, then allocate the new one."""
    model=built.native_model
    from .shape_schema import RuntimeShape
    RuntimeShape.from_native_metadata(built.metadata)
    default_integration_state(model)
    actual_decoder_record(built)
    previous.release()
    return create(built)


create.rebuild_components=rebuild_components