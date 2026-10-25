"""Independent cold N1 recovery assembly, with unchanged shape physics order."""
from dataclasses import replace
import hashlib
import inspect
import numpy as np
import warp as wp
from . import shape_prepare
from .shape_assembly import ShapeAssembly
from .recovery_event_runtime import EventRecoveringRuntime as RecoveringShapeRuntime
from libero.gpu.host.runtime.dynamics.recovery import ROOT_WIDTHS, ResetBinding, model_root_defaults, DataRecoveryWorkspace


def _recovery_loop(*, model, runtime_type=None, recovery_workspace_type=None, **kwargs):
    runtime_type = RecoveringShapeRuntime if runtime_type is None else runtime_type
    recovery_workspace_type = DataRecoveryWorkspace if recovery_workspace_type is None else recovery_workspace_type
    e, forward = kwargs['euler'], kwargs['forward']
    dimensions = {key: int(getattr(model, key)) for key in
                  ('nq', 'nv', 'nM', 'njnt', 'nu', 'na', 'nbody', 'nmocap',
                   'nuserdata', 'nsensordata', 'nplugin', 'npluginstate')}
    roots = dict(qpos=e.qpos, qvel=e.qvel, qacc=e.qacc,
                 qacc_warmstart=e.qacc_warmstart, ctrl=e.ctrl, time=e.time,
                 sensordata=e.sensordata, qfrc_applied=forward.d.qfrc_applied,
                 xfrc_applied=forward.d.xfrc_applied)
    defaults = model_root_defaults(dimensions, qpos0=np.asarray(model.qpos0),
        body_mocapid=np.asarray(model.body_mocapid), body_pos=np.asarray(model.body_pos),
        body_quat=np.asarray(model.body_quat))
    # Current shape intake rejects activation/mocap/plugin state. Nonempty
    # userdata without a real resident owner cannot be guessed or recreated.
    required = {name for name, width in ROOT_WIDTHS.items() if width(dimensions)}
    if not required <= roots.keys():
        raise NotImplementedError('missing current native root owner: ' + str(required - roots.keys()))
    bindings = {name: ResetBinding(roots[name], defaults[name], 'root') for name in required}
    W = e.capacity
    stream = kwargs['stream']
    with wp.ScopedStream(stream):
        active = wp.zeros(W, dtype=wp.int32, device=stream.device)
        warnings = wp.zeros((W, 8), dtype=wp.int32, device=stream.device)
        lastinfo = wp.zeros((W, 8), dtype=wp.int32, device=stream.device)
    recovery = recovery_workspace_type(dimensions, defaults, bindings, nworld=W,
        active=active, warning_number=warnings, warning_lastinfo=lastinfo,
        stream=stream, current_bindings=lambda: bindings, mode='diagnostic')
    loop = runtime_type(recovery=recovery, **kwargs)
    loop.recovery_inventory_scope = 'native required roots only; derived arena rebuilt by physical forward'
    loop.recovery_scope_gaps = ('full native header/derived inventory qualification',
                               'numeric-origin failed N work vs canonical-held classification',
                               'actual task115/116 spontaneous recovery')
    return loop



def _event_cold_factories():
    from .generic_prepare import _cold_factories
    from .newton_final_event import prepare_final_event_runtime
    collision, unused_newton, choice=_cold_factories()
    if choice!="cn":raise ValueError("Final event candidate requires explicit CN cold scope")
    return collision,prepare_final_event_runtime,choice


def prepare_recovery_event_physics(*args, _assembly=None, **kwargs):
    """Construct the recovery route with explicit, private constructor choices."""
    from libero.gpu.host.runtime.dynamics.wide_factory import prepare_from_built
    from libero.gpu.reference.integration.euler_integrator.wide_angle import WideAngleEulerWorkspace

    assembly = ShapeAssembly(dynamics_factory=prepare_from_built,
        euler_type=WideAngleEulerWorkspace, runtime_factory=_recovery_loop,
        cold_factories=_event_cold_factories)
    if _assembly is not None:
        assembly = replace(assembly, **{name: value for name, value in vars(_assembly).items()
                                       if value is not None})
    bundle = shape_prepare.prepare_shape_physics(*args, _assembly=assembly, **kwargs)
    bundle.recovery_source_manifest = dict(original_shape_prepare_sha256=
        hashlib.sha256(inspect.getsource(shape_prepare).encode()).hexdigest(),
        construction='explicit-constructor-arguments', physics_math_changed=False,
        controller_reset_invoked=False,
        qualification='diagnostic roots/conditional-graph forward/Euler only')
    return bundle
