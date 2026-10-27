"""Cold geometry admission for explicit independent C construction."""


def prepare_collision_admission(static,*,context=None,options=None):
    from .cold_geometry import CollisionColdAdmission
    return CollisionColdAdmission(static, context=context, options=options)
