"""Cold intake of the actual installed model; no captured-task whitelist.

This binds descriptors and generated S sources to their real owner and bytes.
It never publishes numerical acceptance, a ready device, or a kernel license.
"""

from dataclasses import dataclass
import hashlib
import inspect
import os
from pathlib import Path
import uuid
from types import MappingProxyType

import mujoco
import numpy as np

from .contracts import ContractError, ModelSpec, _hash

_METADATA_FIELDS = (
    'dimensions', 'options', 'stat', 'stat_descriptors', 'name_maps',
    'joint_qpos_slices', 'model_sha256', 'static_arrays_sha256',
    'topology_sha256', 'array_sha256', 'decoder_pin', 'static_array_inventory',
)
_POLICY = {'schema': 'native-intake-policy-v1', 'layout_version': 'gpu-runtime-v1',
           'fuse_fp': True, 'fast_math': False, 'precise_div_sqrt': True,
           'flush_to_zero': False, 'source_admission_required': True}
_FEATURE_FIELDS = (
    ('joint_types', 'jnt_type', 'mjtJoint'),
    ('geom_types', 'geom_type', 'mjtGeom'),
    ('contact_dimensions', 'geom_condim', None),
    ('pair_dimensions', 'pair_dim', None),
    ('transmission_types', 'actuator_trntype', 'mjtTrn'),
    ('actuator_dynamics', 'actuator_dyntype', 'mjtDyn'),
    ('actuator_gain', 'actuator_gaintype', 'mjtGain'),
    ('actuator_bias', 'actuator_biastype', 'mjtBias'),
    ('sensor_types', 'sensor_type', 'mjtSensor'),
    ('sensor_stages', 'sensor_needstage', 'mjtStage'),
    ('equality_types', 'eq_type', 'mjtEq'),
    ('tendon_wrap_types', 'wrap_type', 'mjtWrap'),
    ('joint_limits', 'jnt_limited', None),
    ('actuator_control_limits', 'actuator_ctrllimited', None),
    ('actuator_force_limits', 'actuator_forcelimited', None),
    ('equality_active', 'eq_active', None),
)
_NUMERIC_FIELDS = (
    'dof_damping', 'dof_armature', 'dof_frictionloss', 'jnt_stiffness',
    'jnt_margin', 'tendon_stiffness', 'tendon_damping', 'tendon_frictionloss',
    'body_gravcomp', 'geom_margin', 'geom_gap', 'sensor_cutoff', 'sensor_noise',
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_feature_tokens(model):
    """Use actual typed fields with the census feature vocabulary, without files."""
    if not isinstance(model, mujoco.MjModel):
        raise ContractError('actual native MjModel required')
    tokens = []
    for category, field, enum_name in _FEATURE_FIELDS:
        if not hasattr(model, field):
            tokens.append(category + ':unavailable')
            continue
        enum = getattr(mujoco, enum_name) if enum_name else None
        for value in np.unique(np.asarray(getattr(model, field))):
            label = None
            if enum is not None:
                try:
                    label = enum(int(value)).name
                except (ValueError, TypeError):
                    pass
            tokens.append(f'{category}:{label or int(value)}')
    for category, field in (('plugin_count', 'nplugin'), ('mocap_count', 'nmocap'),
                            ('explicit_pairs', 'npair'), ('excluded_pairs', 'nexclude')):
        if int(getattr(model, field)):
            tokens.append(category + ':nonzero')
    if np.any(model.geom_fluid):
        tokens.append('fluid_nonzero')
    for field in _NUMERIC_FIELDS:
        if hasattr(model, field) and np.count_nonzero(getattr(model, field)):
            tokens.append(field + ':nonzero')
    jacobian = int(model.opt.jacobian)
    resolved = 'dense' if jacobian == 0 or jacobian == 2 and model.nv < 60 else 'sparse'
    tokens.extend((f'jacobian:{resolved}', f'cone:{int(model.opt.cone)}',
                   f'solver:{int(model.opt.solver)}', f'integrator:{int(model.opt.integrator)}'))
    return tuple(sorted(set(tokens)))


def _checked_metadata(built):
    from libero.gpu.model.factory import BuiltTaskModel, native_model_metadata
    if type(built) is not BuiltTaskModel or not isinstance(built.native_model, mujoco.MjModel):
        raise ContractError('actual installed BuiltTaskModel owner required')
    source = Path(inspect.getfile(type(built))).resolve()
    pin = built.metadata.get('decoder_pin', {})
    if (pin.get('module') != type(built).__module__ or
            pin.get('source_sha256') != _sha(source)):
        raise ContractError('actual decoder source identity differs')
    fresh = native_model_metadata(built.native_model)
    for field in _METADATA_FIELDS:
        if built.metadata.get(field) != fresh[field]:
            raise ContractError('native model changed since construction: ' + field)
    # Python callbacks are outside the native array readset. They cannot silently
    # replace native math behind the GPU source contracts.
    for name in sorted(n for n in dir(mujoco) if n.startswith('get_mjcb_')):
        if getattr(mujoco, name)() is not None:
            raise ContractError('native callback requires a GPU implementation: ' + name[4:])
    return fresh, {'path': str(source), 'sha256': _sha(source)}


def _checked_model_policy(built, metadata):
    """Consume current publisher permissions only after the actual native check."""
    layout = built.metadata.get('layout_version', _POLICY['layout_version'])
    if layout != _POLICY['layout_version']:
        raise ContractError('native intake layout policy requires a separate implementation')
    declared = built.metadata.get('mutable_parameters', ('body_pos', 'body_quat'))
    if not isinstance(declared, (tuple, list)) or any(type(name) is not str for name in declared):
        raise ContractError('native parameter policy must be explicit field names')
    if len(declared) != len(set(declared)):
        raise ContractError('native parameter policy has duplicate fields')
    if 'native_write_schema' in built.metadata:
        from .native_model_update import native_model_write_schema
        actual_schema = native_model_write_schema(built)
        if built.metadata['native_write_schema'] != actual_schema:
            raise ContractError('native parameter policy differs from actual current publisher schema')
        if set(declared) != set(actual_schema['arrays']):
            raise ContractError('native parameter policy differs from published physical fields')
    elif set(declared) != {'body_pos', 'body_quat'}:
        raise ContractError('extended native parameter permissions require the actual publisher')
    metadata = dict(metadata)
    metadata['layout_version'] = layout
    metadata['mutable_parameters'] = tuple(declared)
    return metadata

def describe_native_model(built):
    """Return an exact fresh ModelSpec and provenance, without granting execution."""
    metadata, decoder = _checked_metadata(built)
    metadata = _checked_model_policy(built, metadata)
    features = native_feature_tokens(built.native_model)
    spec = ModelSpec.from_factory_metadata(metadata, features=features)
    descriptor = {
        'schema': 'portable-native-model-intake-v1',
        'model_metadata': spec.to_factory_metadata(), 'features': list(features),
        'model_identity': spec.identity_key, 'specialization_key': spec.specialization_key,
        'decoder_record': decoder,
        'owner': {'process_id': os.getpid(), 'built_token': id(built),
                  'native_model_token': id(built.native_model),
                  'producer_type': type(built).__module__ + '.' + type(built).__qualname__},
        'policy': dict(_POLICY),
        'policy_sha256': _hash(_POLICY),
        'native_readset_sha256': _hash({k: metadata[k] for k in _METADATA_FIELDS}),
        'cpu_physics': False, 'kernel_license_granted': False,
        'numerical_acceptance_published': False, 'device_ready_published': False,
    }
    descriptor['owner_version_sha256'] = _hash({
        'owner': descriptor['owner'], 'native_readset': descriptor['native_readset_sha256'],
        'policy': descriptor['policy_sha256']})
    return spec, descriptor


@dataclass
class NativeIntakeBinding:
    """Process-local cold owner/source binding; invalidate before mutation/release."""
    built: object
    native_model: object
    spec: ModelSpec
    package: Path
    descriptor: dict
    pins: dict
    active: bool = True

    def invalidate(self, reason='explicit model/source/policy update'):
        self.active = False
        self.invalidated_reason = str(reason)

    def validate(self):
        """Cold validation only; callers keep normal hot admission/owner guards."""
        if not self.active:
            raise ContractError('native intake binding was invalidated')
        if self.built.native_model is not self.native_model:
            raise ContractError('native intake owner changed')
        for path, digest in self.pins.items():
            if _sha(path) != digest:
                raise ContractError('native intake source/build artifact changed: ' + path)
        spec, descriptor = describe_native_model(self.built)
        if (spec.identity_key != self.spec.identity_key or
                descriptor != self.descriptor or _hash(_POLICY) != self.descriptor['policy_sha256']):
            raise ContractError('native intake model/policy binding changed')
        return self.spec


def prepare_native_intake(built, *, cache_dir=None):
    """Generate current S specialization from native bytes, retaining exact owner.

    Owner bindings are always fresh; identical generated sources may use the
    persistent project cache. Structural/static/capacity and source checks remain.
    The returned binding validates cold changes and supports explicit revocation.
    """
    from libero.gpu.host.runtime.constraints.identity import STATIC_SCHEMA
    from libero.gpu.host.runtime.constraints.specialization import generate_specialization
    spec, descriptor = describe_native_model(built)
    model = built.native_model
    static = {name: np.array(getattr(model, name), copy=True, order='C')
              for name in STATIC_SCHEMA}
    facts = {name: int(getattr(model, name)) for name in (
        'neq', 'ntendon', 'nplugin', 'nmocap', 'na', 'nq', 'nv', 'nu', 'nM',
        'nbody', 'ngeom', 'nsite', 'njnt')}
    facts.update({name: int(getattr(model.opt, name)) for name in (
        'solver', 'cone', 'jacobian', 'disableflags', 'enableflags', 'noslip_iterations')})
    facts.update({name: float(getattr(model.opt, name)) for name in ('timestep', 'impratio')})
    manifest = {
        'model_sha256': spec.model_sha256, 'model_facts': facts,
        'dimensions': dict(built.metadata['dimensions'], nconmax=int(model.nconmax)),
        'static': {name: {'shape': list(a.shape), 'dtype': str(a.dtype),
                          'sha256': hashlib.sha256(a.tobytes()).hexdigest()}
                   for name, a in static.items()},
    }
    # Validate before allocating a directory. Unsupported state/solver families
    # keep the original S rejection rather than being silently omitted.
    from libero.gpu.host.runtime.constraints.specialization import validate_manifest
    validate_manifest(manifest, static)
    from .product_paths import current_build_project, record_build, build_directory
    project = current_build_project()
    if project is None:
        # Explicit standalone source preparation owns its output directory.
        directory = build_directory('intake')
        package = directory / ('shape_s_' + uuid.uuid4().hex)
        generated = generate_specialization(manifest, static, package)
    else:
        from .specialization_cache import generated_package
        package, generated, hit = generated_package(manifest, static, generate_specialization,
                                                    project['root'] / 'specializations')
        record_build('constraint-source', package.name,
                     () if hit else (('generate-specialization', str(package)),))
    pins = {entry['path']: entry['sha256'] for group in ('template_pins', 'generated_pins')
            for entry in generated[group].values()}
    for path in (Path(__file__), Path(__file__).with_name('contracts.py'),
                 Path(inspect.getfile(generate_specialization)), package / '__init__.py',
                 Path(descriptor['decoder_record']['path']),
                 Path(descriptor['decoder_record']['path']).with_name('cold_assets.py'),
                 package / 'MODEL_MANIFEST.json', package / 'SOURCE_GENERATION.json'):
        pins[str(path.resolve())] = _sha(path)
    binding = NativeIntakeBinding(built, model, spec, package, descriptor, MappingProxyType(pins))
    binding.validate()
    return binding
