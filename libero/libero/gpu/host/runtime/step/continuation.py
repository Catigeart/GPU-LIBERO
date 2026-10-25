"""Synchronized boundary transport for cold arena replacement, never CPU physics."""
import libero.gpu.precision as scalar
import hashlib
import numpy as np
import warp as wp


def _scalar_layout(array):
    """Match Warp's NumPy layout without downloading an array to discover it."""
    from warp._src.types import dtype_to_numpy
    dtype = array.dtype
    if hasattr(dtype, 'numpy_dtype'):
        return tuple(array.shape), np.dtype(dtype.numpy_dtype())
    if hasattr(dtype, '_wp_scalar_type_'):
        return (*array.shape, *dtype._shape_), np.dtype(dtype_to_numpy(dtype._wp_scalar_type_))
    return tuple(array.shape), np.dtype(dtype_to_numpy(dtype))


def continuation_arrays(bundle):
    """Explicit dynamic/public read set. Model static dictionaries are excluded."""
    result = {}
    def add(prefix, mapping):
        for key,array in mapping.items():
            if not isinstance(array,wp.array):
                raise TypeError('continuation field is not a device array: '+prefix+'.'+key)
            result[prefix+'.'+key] = array
    def attrs(prefix, owner, names, required=()):
        for key in names:
            array=getattr(owner,key,None)
            if isinstance(array,wp.array):result[prefix+'.'+key]=array
            elif key in required:raise ValueError('required continuation field missing: '+prefix+'.'+key)
    loop,d = bundle.runtime,bundle.dynamics
    attrs('Euler',loop.euler,('qpos','qvel','time','ctrl','qacc','qacc_warmstart','qM','qH','qHDiagInv',
                             'qfrc_smooth','qfrc_constraint','sensordata'),
          required=('qpos','qvel','time','ctrl','qacc','qacc_warmstart','sensordata'))
    attrs('loop',loop,('state_epoch','valid','written','first_error','intent','integration_count'))
    add('D.public',d.public)
    add('D.valid',d.valid);add('D.written',d.written)
    attrs('D',d,('qfrc_applied','xfrc_applied','cvel','cdof_dot','forward_epoch','source_state_epoch',
                 'controller_parent_valid','controller_parent_written'),required=('qfrc_applied','xfrc_applied'))
    attrs('body',d.position.body,('qpos','xpos','xquat','xmat','xanchor','xaxis'))
    attrs('frames',d.position.frames,('geom_xpos','geom_xmat','site_xpos','site_xmat','xipos','ximat'))
    attrs('mass',d.position,('subtree_com','cinert','cdof'))
    attrs('factor',d.position.factor,('qm','ld','inv','sqrt_inv'))
    attrs('forward',loop.forward,('expected_forward','expected_newton','valid','written','error'))
    attrs('foundation',bundle.foundation,('expected_state_epoch','state_valid'))
    add('transport',bundle.runtime.forward.transport.inputs)
    attrs('transport',bundle.runtime.forward.transport,('ncon','active'))
    final=bundle.collision.workspace.finalize
    attrs('C.public',final,('contact_f','contact_i','ncon'),required=('contact_f','contact_i','ncon'))
    workspace=bundle.constraints.workspace
    attrs('S',workspace,('counts','status','committed','reference_ready','ref_epoch',
        'efc_type','efc_id','efc_pos','efc_margin','efc_frictionloss','efc_J','efc_R','efc_D','efc_KBIP',
        'efc_vel','efc_aref','efc_diagApprox','efc_diagA','contact_mu','contact_address','contact_exclude'))
    add('N.public',bundle.newton.public_arrays)
    attrs('N',bundle.newton,('n_attempt','n_expected','n_written','valid'))
    add('post.public',bundle.post.public)
    attrs('post',bundle.post,('valid','written'))
    if bundle.controller is not None:
        add('controller',bundle.controller.public_arrays())
        attrs('controller.input',bundle.controller.controller,('q0','action'),
              required=('q0','action'))
    return result


def controller_static_headers(bundle):
    """Compare real immutable OSC configuration; never restore it as state."""
    if bundle.controller is None:return {}
    result={}
    for name in ('gains','ctrl_range'):
        array=getattr(bundle.controller.controller,name,None)
        if not isinstance(array,wp.array):
            raise ValueError('required controller static field missing: '+name)
        value=array.numpy()
        result[name]=dict(shape=list(value.shape),dtype=value.dtype.str,
            sha256=hashlib.sha256(value.tobytes()).hexdigest())
    return result


def continuation_snapshot(bundle, *, verify=False):
    wp.synchronize_stream(bundle.stream)
    arrays=continuation_arrays(bundle)
    data={key:array.numpy().copy() for key,array in arrays.items()}
    headers={key:dict(shape=list(value.shape),dtype=value.dtype.str,
        bytes=value.nbytes) for key,value in data.items()}
    if verify:
        for key,value in data.items():
            headers[key]['sha256']=hashlib.sha256(value.tobytes()).hexdigest()
    return dict(schema='gpu-boundary-continuation-v1',data=data,headers=headers,
        controller_static_headers=controller_static_headers(bundle),
        source_model_sha256=bundle.model_spec.model_sha256,
        source_scope='actual resident pre-replacement bytes; mixed-age derived/cache retained',
        CPU_physics=False,missing_scope=['observer history supplied by observer owner',
                                      'private scratch not visible through canonical/public read set'])


def _restore_arrays(bundle, snapshot, *, verify=False):
    arrays = continuation_arrays(bundle)
    if set(arrays) != set(snapshot['data']) or set(arrays) != set(snapshot['headers']):
        raise ValueError('new arena continuation field set differs')
    # Validate the complete transfer before enqueueing its first write.
    for key, array in arrays.items():
        value, header = snapshot['data'][key], snapshot['headers'][key]
        if not isinstance(value, np.ndarray):
            raise TypeError('continuation field must be a NumPy array: ' + key)
        layout = _scalar_layout(array)
        if (layout != (value.shape, value.dtype) or list(value.shape) != header['shape'] or
                value.dtype.str != header['dtype'] or value.nbytes != header['bytes']):
            raise ValueError('new arena continuation shape/dtype differs: ' + key)
        if verify and hashlib.sha256(value.tobytes()).hexdigest() != header.get('sha256'):
            raise ValueError('diagnostic boundary snapshot drift: ' + key)
    with wp.ScopedStream(bundle.stream):
        for key, array in arrays.items():
            array.assign(snapshot['data'][key])
    wp.synchronize_stream(bundle.stream)
    if verify:
        for key, array in arrays.items():
            if array.numpy().tobytes() != snapshot['data'][key].tobytes():
                raise ValueError('actual continuation byte transport differs: ' + key)
    return len(arrays)


def restore_continuation(bundle, snapshot, *, verify=False):
    """Restore retained state; full readback verification is explicitly opt-in."""
    if controller_static_headers(bundle) != snapshot['controller_static_headers']:
        raise ValueError('new arena controller static configuration differs')
    count = _restore_arrays(bundle, snapshot, verify=verify)
    return dict(schema='gpu-boundary-continuation-restore-v1', fields=count,
        transport_completed=True, data_bitwise_equal=True if verify else None,
        new_model_sha256=bundle.model_spec.model_sha256,
        derived_age='retained old generation; new forward must republish current generation',
        CPU_physics=False)


def validate_native_controller_transition(bundle, snapshot, plan):
    """Check actual new static owners independently from retained dynamic bytes.

    Old snapshot headers remain unchanged. A declared actuator_ctrlrange update
    can select new exact controller ranges; no other static difference is aliased.
    """
    if snapshot['source_model_sha256'] != plan.base_spec.model_sha256:
        raise ValueError('native continuation belongs to another source model')
    if (bundle.model_pack.model is not plan.built.native_model or
            bundle.model_spec.identity_key != plan.spec.identity_key):
        raise ValueError('native continuation new model owner/spec differs')
    before = snapshot['controller_static_headers']
    after = controller_static_headers(bundle)
    if set(before) != set(after):
        raise ValueError('native continuation controller static field set differs')
    changed = {name for name in before if before[name] != after[name]}
    if not changed:
        return {'schema': 'native-controller-static-transition-v1',
                'changed_fields': [], 'old_headers': before, 'new_headers': after,
                'actual_new_model_ranges_exact': True, 'headers_aliased': False}
    if changed != {'ctrl_range'} or 'actuator_ctrlrange' not in plan.array_fields:
        raise ValueError('undeclared native controller static change: ' + repr(sorted(changed)))
    owner = bundle.controller.controller
    actual = owner.ctrl_range.numpy()
    model = plan.built.native_model
    robot_metadata = plan.built.metadata.get('robot_metadata', ())
    if len(robot_metadata) != 1:
        raise ValueError('native controller range transition requires one explicit robot')
    robot = robot_metadata[0]
    ids = list(robot['actuator_ids']) + list(robot.get('gripper', {}).get('actuator_ids', ()))
    if ids != list(range(int(model.nu))):
        raise ValueError('native controller range ordering requires explicit implementation')
    expected = np.tile(np.asarray(model.actuator_ctrlrange, np.float64).reshape(1, -1),
                       (actual.shape[0], 1))
    if (actual.shape != expected.shape or actual.dtype != expected.dtype or
            actual.tobytes() != expected.tobytes()):
        raise ValueError('new controller ranges differ from actual new native parameters')
    return {'schema': 'native-controller-static-transition-v1',
            'changed_fields': sorted(changed), 'old_headers': before, 'new_headers': after,
            'actual_new_model_ranges_exact': True, 'headers_aliased': False,
            'declared_native_fields': list(plan.array_fields)}


def restore_native_model_continuation(bundle, snapshot, plan, *, verify=False):
    """Restore exact old dynamic bytes under an explicit new-static transition.

    A new physical identity does not make old derived values current: model
    publication invalidates forward/step freshness after this byte transport.
    Dimension/capacity drift remains rejected; nothing is truncated or padded.
    """
    plan.validate_base()
    transition = validate_native_controller_transition(bundle, snapshot, plan)
    count = _restore_arrays(bundle, snapshot, verify=verify)
    return {'schema': 'gpu-native-model-continuation-restore-v1', 'fields': count,
            'transport_completed': True, 'data_bitwise_equal': True if verify else None, 'new_model_sha256': bundle.model_spec.model_sha256,
            'controller_static_transition': transition,
            'derived_age': 'retained prior generation; explicit forward must republish',
            'CPU_physics': False, 'generation_bridge_granted': False}