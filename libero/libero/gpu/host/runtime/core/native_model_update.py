"""Exact cold native-model replacement from public parameter writes.

Only static CPU model bytes are prepared here. GPU owners must snapshot their
actual continuation, release the old arena before allocation and restore without
reset/forward. A new owner/source identity never inherits an acceptance bridge.
"""

from dataclasses import dataclass
import copy
import hashlib

import mujoco
import numpy as np

from .contracts import ContractError, _hash
from .portable_intake import describe_native_model, prepare_native_intake
from .model_bytes import model_bytes

# Public physical inputs, not compiled indices/addresses, name tables, resource
# bytes or derived compiler caches such as body_subtreemass/dof_invweight0.
_PHYSICAL_ARRAYS = frozenset('''
qpos0 qpos_spring
body_pos body_quat body_ipos body_iquat body_mass body_inertia body_gravcomp body_user
jnt_pos jnt_axis jnt_stiffness jnt_range jnt_margin jnt_solref jnt_solimp jnt_limited jnt_user
 dof_armature dof_damping dof_frictionloss dof_solref dof_solimp
geom_pos geom_quat geom_size geom_friction geom_margin geom_gap geom_solmix geom_solref geom_solimp geom_rgba geom_fluid geom_user geom_contype geom_conaffinity geom_condim geom_priority
site_pos site_quat site_size site_rgba site_user
cam_pos cam_quat cam_fovy cam_ipd cam_user
light_pos light_dir light_diffuse light_specular light_ambient light_attenuation light_cutoff light_exponent light_active light_castshadow
actuator_ctrlrange actuator_forcerange actuator_actrange actuator_dynprm actuator_gainprm actuator_biasprm actuator_gear actuator_lengthrange actuator_cranklength actuator_ctrllimited actuator_forcelimited actuator_actlimited actuator_user
tendon_range tendon_margin tendon_stiffness tendon_damping tendon_frictionloss tendon_solref_lim tendon_solimp_lim tendon_solref_fri tendon_solimp_fri tendon_limited tendon_rgba tendon_width tendon_user
eq_data eq_solref eq_solimp eq_active
sensor_cutoff sensor_noise sensor_user
'''.split())
_NONNEGATIVE = frozenset('''body_mass body_inertia dof_armature dof_damping dof_frictionloss jnt_stiffness geom_size site_size geom_friction tendon_stiffness tendon_damping tendon_frictionloss tendon_width sensor_cutoff sensor_noise'''.split())
_QUATERNIONS = frozenset(('body_quat', 'body_iquat', 'geom_quat', 'site_quat', 'cam_quat'))


def native_model_write_schema(built):
    """Declare actual parameter arrays/options; unknown/read-only fields stay so."""
    from libero.gpu.model.factory import BuiltTaskModel
    if type(built) is not BuiltTaskModel or not isinstance(built.native_model, mujoco.MjModel):
        raise ContractError('actual installed native model owner required')
    model = built.native_model
    arrays = {}
    readonly = []
    for name, layout in built.metadata['static_array_inventory'].items():
        value = np.asarray(getattr(model, name))
        if list(value.shape) != layout['shape'] or value.dtype.str != layout['dtype']:
            raise ContractError('native public model layout changed: ' + name)
        if name in _PHYSICAL_ARRAYS:
            arrays[name] = {'shape': list(value.shape), 'dtype': value.dtype.str}
        else:
            readonly.append(name)
    options = {}
    for name in built.metadata['options']:
        value = getattr(model.opt, name)
        a = np.asarray(value)
        if a.dtype.kind not in 'iufb':
            continue
        options[name] = {'shape': list(a.shape), 'dtype': a.dtype.str,
                         'scalar': not isinstance(value, np.ndarray)}
    return {'schema': 'native-model-public-write-v1', 'arrays': arrays,
            'options': options, 'readonly_arrays': sorted(readonly),
            'rebuild_required': True, 'topology_writable': False}


def publish_native_model_write_schema(built):
    """Publish explicit current parameter permissions before initial ModelSpec creation."""
    schema = native_model_write_schema(built)
    built.metadata['native_write_schema'] = copy.deepcopy(schema)
    built.metadata['mutable_parameters'] = tuple(sorted(schema['arrays']))
    return schema


def _inherit_controller_timing(built, robot_metadata):
    """Direct native writes retain the original controller constructor timing."""
    previous = built.metadata.get('robot_metadata', ())
    if len(previous) != len(robot_metadata):
        raise ContractError('native update changed robot metadata count')
    for old, fresh in zip(previous, robot_metadata):
        if any(old.get(key) != fresh.get(key) for key in ('name', 'naming_prefix')):
            raise ContractError('native update changed robot selector/order')
        value = old.get('controller_model_timestep', built.metadata['options']['timestep'])
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ContractError('original controller construction timestep required')
        fresh['controller_model_timestep'] = value
    return robot_metadata


def _inherit_scheduler_timing(built, metadata):
    """Keep original robosuite cached substep timing, separate from native dt."""
    source = built.metadata.get('scheduler_model_timestep', built.metadata.get('model_timestep'))
    if source is None and built.domain_shell is not None:
        source = getattr(built.domain_shell, 'model_timestep', None)
    if source is not None:
        value = float(source)
        if not np.isfinite(value) or value <= 0:
            raise ContractError('original cached scheduler timestep required')
        metadata['scheduler_model_timestep'] = value
        metadata['model_timestep'] = value
    # control_timestep/control_freq, including source precision, were copied
    # from the source metadata. Native option writes do not rebuild those caches.
    return metadata

def _checked_array(name, value, original):
    value = np.asarray(value)
    if value.shape != original.shape or value.dtype != original.dtype:
        raise ContractError('native model parameter shape/dtype differs: ' + name)
    if value.dtype.kind not in 'iufb' or not np.isfinite(value).all():
        raise ContractError('native model parameter must be finite numeric: ' + name)
    if name in _NONNEGATIVE and np.any(value < 0):
        raise ContractError('native model parameter must be nonnegative: ' + name)
    if name in _QUATERNIONS:
        if not np.allclose(np.sum(value * value, axis=-1), 1., rtol=0., atol=1.e-12):
            raise ContractError('native model quaternion must be unit length: ' + name)
    return np.array(value, copy=True, order='C')


def _checked_option(name, value, original):
    original_array = np.asarray(original)
    value_array = np.asarray(value)
    if (value_array.shape != original_array.shape or value_array.dtype.kind not in 'iufb' or
            not np.isfinite(value_array).all()):
        raise ContractError('native option shape/finite differs: ' + name)
    if original_array.dtype.kind in 'iu':
        if value_array.dtype.kind not in 'iu' or np.any(value_array < np.iinfo(original_array.dtype).min) or np.any(value_array > np.iinfo(original_array.dtype).max):
            raise ContractError('native integer option required: ' + name)
    elif value_array.dtype.kind not in 'iufb':
        raise ContractError('native numeric option required: ' + name)
    if name == 'timestep' and not float(value_array) > 0:
        raise ContractError('native timestep must be positive')
    with np.errstate(over='ignore', invalid='ignore'):
        cast = np.array(value_array, dtype=original_array.dtype, copy=True)
    if not np.isfinite(cast).all():
        raise ContractError('native option overflows its actual dtype: ' + name)
    return cast.item() if cast.ndim == 0 else cast


@dataclass
class NativeModelUpdatePlan:
    source_built: object
    source_native_model: object
    base_spec: object
    base_descriptor: dict
    built: object
    intake_binding: object
    array_fields: tuple
    option_fields: tuple
    source_generation: int
    directory: object
    state: str = 'prepared'

    @property
    def spec(self): return self.intake_binding.spec
    @property
    def constraint_package(self): return self.intake_binding.package
    @property
    def mjb_path(self): return None

    def validate_base(self):
        if self.state != 'prepared':
            raise ContractError('native update plan is no longer prepared')
        if self.source_built.native_model is not self.source_native_model:
            raise ContractError('native update source owner changed')
        spec, descriptor = describe_native_model(self.source_built)
        if spec.identity_key != self.base_spec.identity_key or descriptor != self.base_descriptor:
            raise ContractError('native update source identity changed')
        self.intake_binding.validate()
        if (hashlib.sha256(model_bytes(self.source_native_model)).hexdigest() != self.base_spec.model_sha256 or
                hashlib.sha256(model_bytes(self.built.native_model)).hexdigest() != self.spec.model_sha256):
            raise ContractError('native model update bytes changed')
        return self

    def discard(self, reason='cold model update rejected before publication'):
        if self.state == 'committed':
            raise ContractError('committed native update cannot be discarded')
        self.intake_binding.invalidate(reason)
        self.state = 'discarded'

    def mark_committed(self):
        """Call after actual GPU assembly/continuation and external owner swap."""
        self.validate_base()
        self.state = 'committed'
        old = getattr(self.source_built, '_gpu_native_intake_binding', None)
        if old is not None:
            old.invalidate('actual native model owner replaced')
        return self.built


def prepare_native_model_update(built, array_changes, option_changes=None, *, cache_dir=None,
                                source_generation=0):
    """Preflight full actual native replacement, with old owner left untouched.

    The caller holds its task/model lock. Payloads are copied before cloning.
    On any rejection no public owner is replaced or model bytes mutated. The
    plan gives the GPU transaction its fresh built/spec/constraint_package;
    callers preserve dynamic state via their actual continuation APIs.
    """
    from libero.gpu.model.factory import BuiltTaskModel, native_model_metadata, robot_static_metadata
    if type(source_generation) is not int or source_generation < 0:
        raise ContractError('nonnegative source model generation required')
    option_changes = {} if option_changes is None else dict(option_changes)
    array_changes = dict(array_changes)
    if any(type(name) is not str or not name for name in tuple(array_changes) + tuple(option_changes)):
        raise ContractError('native model update field names must be nonempty strings')
    if not array_changes and not option_changes:
        raise ContractError('native replacement requires a nonempty update')
    schema = native_model_write_schema(built)
    if set(array_changes) - set(schema['arrays']):
        raise ContractError('native model field is readonly/structural: ' + repr(sorted(set(array_changes) - set(schema['arrays']))))
    if set(option_changes) - set(schema['options']):
        raise ContractError('unknown/readonly native model option: ' + repr(sorted(set(option_changes) - set(schema['options']))))
    model = built.native_model
    payload = {name: _checked_array(name, value, np.asarray(getattr(model, name)))
               for name, value in array_changes.items()}
    option_payload = {name: _checked_option(name, value, getattr(model.opt, name))
                      for name, value in option_changes.items()}
    base_spec, base_descriptor = describe_native_model(built)
    if hashlib.sha256(model_bytes(model)).hexdigest() != base_spec.model_sha256:
        raise ContractError('native update source model changed')
    native = copy.copy(model)
    original = native_model_metadata(native)
    for field in ('dimensions', 'options', 'model_sha256', 'static_arrays_sha256', 'topology_sha256', 'array_sha256'):
        if original[field] != built.metadata[field]:
            raise ContractError('exact MJB clone differs from source: ' + field)
    for name, value in payload.items():
        getattr(native, name)[...] = value
    for name, value in option_payload.items():
        original_value = getattr(native.opt, name)
        if isinstance(original_value, np.ndarray): original_value[...] = value
        else: setattr(native.opt, name, value)
    metadata = native_model_metadata(native)
    if metadata['dimensions'] != original['dimensions'] or metadata['static_array_inventory'] != original['static_array_inventory']:
        raise ContractError('native parameter update changed structural dimensions/layout')
    changed_arrays = {name for name in metadata['array_sha256']
                      if metadata['array_sha256'][name] != original['array_sha256'][name]}
    if changed_arrays - set(payload):
        raise ContractError('native parameter update changed undeclared arrays')
    changed_options = {name for name in metadata['options'] if metadata['options'][name] != original['options'][name]}
    if changed_options - set(option_payload):
        raise ContractError('native parameter update changed undeclared options')
    for name, value in built.metadata.items():
        if name not in metadata:
            metadata[name] = copy.deepcopy(value)
    metadata['mutable_parameters'] = tuple(sorted(schema['arrays']))
    metadata['native_model_timestep'] = float(native.opt.timestep)
    if built.domain_shell is not None and getattr(built.domain_shell, 'robots', None) is not None:
        metadata['robot_metadata'] = _inherit_controller_timing(built,
            robot_static_metadata(built.domain_shell, native, metadata['name_maps']))
    _inherit_scheduler_timing(built, metadata)
    fresh = BuiltTaskModel(native, built.xml, built.domain_shell, metadata)
    publish_native_model_write_schema(fresh)
    # No identical-family bridge: actual new readsets/options select fresh
    # specialization and each component checks its own source/owner/capacity.
    intake = prepare_native_intake(fresh, cache_dir=cache_dir)
    fresh._gpu_native_intake_binding = intake
    if hashlib.sha256(model_bytes(native)).hexdigest() != intake.spec.model_sha256:
        intake.invalidate('actual update model differs')
        raise ContractError('actual update model differs from fresh intake')
    preflight = {
        'schema': 'native-model-full-replacement-preflight-v1',
        'base_model_identity': base_spec.identity_key, 'new_model_identity': intake.spec.identity_key,
        'source_owner_version': base_descriptor['owner_version_sha256'],
        'new_owner_version': intake.descriptor['owner_version_sha256'],
        'array_fields': sorted(payload), 'option_fields': sorted(option_payload),
        'source_generation': source_generation,
        'source_array_payload_sha256': {name: hashlib.sha256(a.tobytes()).hexdigest()
                                        for name, a in payload.items()},
        'option_payload_sha256': _hash({name: value.tolist() if isinstance(value, np.ndarray) else value
                                       for name, value in option_payload.items()}),
        'actual_model_storage': 'native-model', 'actual_model_sha256': intake.spec.model_sha256,
        'old_owner_unchanged': True, 'CPU_physics': False,
        'generation_bridge_granted': False, 'GPU_continuation_restored': False,
        'GPU_owner_published': False, 'numerical_acceptance_published': False,
    }
    plan = NativeModelUpdatePlan(built, model, base_spec, base_descriptor, fresh, intake,
                                tuple(sorted(payload)), tuple(sorted(option_payload)),
                                source_generation, None)
    plan.preflight = preflight
    plan.validate_base()
    return plan
