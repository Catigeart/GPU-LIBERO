"""P06-D staged dynamics: position/velocity/bias/control/actuation/acceleration.

This namespace owns the staged entry points that compose the accepted Phase05
device chains into the MuJoCo REAL_C(2.3)REAL_C(.7) forward-stage boundaries.  Production
kernels keep the accepted source order (engine_forward.c SHA
d8957dc8c2b2aa7ac690ab56e7425deb7632e5df40c609cf9914ab44553c51d5).  The
accepted D09 smooth kernel is retained unchanged; the nonzero-xfrc acceleration
path is the new three-RN -> seeded D02 -> N06 factor/solve combinator.
"""

from importlib import import_module

_EXPORTS = {
    **{name:'rules' for name in ('FIXED_DIMS','STAGE_MAP','FixedModelRules','load_rules')},
    **{name:'witness' for name in ('StageWitness','STAGE_ORDER','FIELD_GROUPS')},
    'ControlClock':'control_clock','SmoothXfrcSolveWorkspace':'acceleration',
    'RnNodes':'acceleration','TransmissionWorkspace':'transmission',
    'ActuationForceWorkspace':'actuation_split','DynamicsStages':'stages',
}


def __getattr__(name):
    if name not in _EXPORTS:raise AttributeError(name)
    value=getattr(import_module('.'+_EXPORTS[name],__name__),name)
    globals()[name]=value
    return value

__all__ = [
    'FIXED_DIMS', 'STAGE_MAP', 'FixedModelRules', 'load_rules',
    'StageWitness', 'STAGE_ORDER', 'FIELD_GROUPS',
    'ControlClock',
    'SmoothXfrcSolveWorkspace', 'RnNodes',
    'TransmissionWorkspace', 'ActuationForceWorkspace',
    'DynamicsStages',
]
