"""Cold configuration and resident native control implementations."""

import ctypes as ct


class ControllerSolveData(ct.Structure):
    _fields_ = [
        ("mass", ct.c_void_p),
        ("jacobian", ct.c_void_p),
        ("q", ct.c_void_p),
        ("qvel", ct.c_void_p),
        ("bias", ct.c_void_p),
        ("q0", ct.c_void_p),
        ("ee_pos", ct.c_void_p),
        ("ee_rot", ct.c_void_p),
        ("ee_velocity", ct.c_void_p),
        ("action", ct.c_void_p),
        ("goal_pos", ct.c_void_p),
        ("goal_rot", ct.c_void_p),
        ("gripper", ct.c_void_p),
        ("active", ct.c_void_p),
        ("gains", ct.c_void_p),
        ("range", ct.c_void_p),
        ("ctrl", ct.c_void_p),
        ("raw_tau", ct.c_void_p),
        ("error", ct.c_void_p),
        ("ranks", ct.c_void_p),
        ("policy", ct.c_void_p),
        ("needs", ct.c_void_p),
        ("mass_inv", ct.c_void_p),
        ("A_full", ct.c_void_p),
        ("A_pos", ct.c_void_p),
        ("A_ori", ct.c_void_p),
        ("lambda_full", ct.c_void_p),
        ("lambda_pos", ct.c_void_p),
        ("lambda_ori", ct.c_void_p),
        ("nullspace_matrix", ct.c_void_p),
        ("position_error", ct.c_void_p),
        ("orientation_error", ct.c_void_p),
        ("quat_input", ct.c_void_p),
        ("quat_cast", ct.c_void_p),
        ("quat_result", ct.c_void_p),
        ("quat_valid", ct.c_void_p),
        ("nullspace_torque", ct.c_void_p),
        ("clipped", ct.c_void_p),
        ("diagnostics", ct.c_int32),
        ("uncouple", ct.c_int32),
        ("derived_valid", ct.c_void_p),
        ("derived_key", ct.c_void_p),
        ("derived_data", ct.c_void_p),
        ("derived_ranks", ct.c_void_p),
        ("cache_hits", ct.c_void_p),
        ("pm", ct.c_void_p),
        ("pj", ct.c_void_p),
        ("pq", ct.c_void_p),
        ("pv", ct.c_void_p),
        ("pp", ct.c_void_p),
        ("pr", ct.c_void_p),
        ("pe", ct.c_void_p),
        ("cache_valid", ct.c_void_p),
        ("action_config", ct.c_void_p),
    ]


def osc_action_config(config, gripper):
    """Robosuite Controller.scale_action parameters and Panda formatter speed."""
    import numpy as np
    values = [np.broadcast_to(np.asarray(config.get(name, default), dtype=float), (6,))
              for name, default in (('input_min', -1), ('input_max', 1),
                ('output_min', [-.05]*3+[-.5]*3), ('output_max', [.05]*3+[.5]*3))]
    result = np.concatenate((*values, np.array([float(gripper['speed'])])))
    if not np.isfinite(result).all() or np.any(values[0] >= values[1]):
        raise ValueError('OSC action ranges must be finite with input_min < input_max')
    return result
