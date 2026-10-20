"""P06-D staged dynamics: position/velocity/bias/control/actuation/acceleration.

This namespace owns the staged entry points that compose the accepted Phase05
device chains into the MuJoCo REAL_C(2.3)REAL_C(.7) forward-stage boundaries.  Production
kernels keep the accepted source order (engine_forward.c SHA
d8957dc8c2b2aa7ac690ab56e7425deb7632e5df40c609cf9914ab44553c51d5).  The
accepted D09 smooth kernel is retained unchanged; the nonzero-xfrc acceleration
path is the new three-RN -> seeded D02 -> N06 factor/solve combinator.
"""

from .rules import FIXED_DIMS, STAGE_MAP, FixedModelRules, load_rules
from .witness import StageWitness, STAGE_ORDER, FIELD_GROUPS
from .control_clock import ControlClock
from .acceleration import SmoothXfrcSolveWorkspace, RnNodes
from .transmission import TransmissionWorkspace
from .actuation_split import ActuationForceWorkspace
from .stages import DynamicsStages

__all__ = [
    'FIXED_DIMS', 'STAGE_MAP', 'FixedModelRules', 'load_rules',
    'StageWitness', 'STAGE_ORDER', 'FIELD_GROUPS',
    'ControlClock',
    'SmoothXfrcSolveWorkspace', 'RnNodes',
    'TransmissionWorkspace', 'ActuationForceWorkspace',
    'DynamicsStages',
]
