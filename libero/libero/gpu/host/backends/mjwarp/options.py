"""Explicit complete shipped-domain preset; individual selectors stay available."""
CUDA_LIBERO={name:'cuda' for name in (
    'acceleration','mass_solve','mass_factor','mass_matrix','com_vel','bias',
    'postconstraint','passive','actuation','euler','transmission','actuator_velocity',
    'kinematics','com_pos','camlight','fluid','external_wrench','constraint_build',
    'broadphase','plane_collision','box_collision','convex_collision','queries',
    'publication','public_forward')}
CUDA_LIBERO['newton']='cuda_newton'


def expand(controls):
    controls=dict(controls);preset=controls.pop('preset','upstream')
    if preset not in ('upstream','cuda_libero'):raise ValueError('MJWarp preset must be upstream or cuda_libero')
    if preset=='cuda_libero':
        for key,value in CUDA_LIBERO.items():controls.setdefault(key,value)
    return controls,preset
