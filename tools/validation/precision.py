"""Frozen-input sensitivity and backward error for lower-precision acceleration.

Mass expansion is MuJoCo's algebraic packed-to-dense operation, not a CPU
physics step. Endpoint state envelopes remain independent of this screen.
"""
from pathlib import Path
import numpy as np


def mass_model(task, *, layout_root=None):
    if layout_root is not None:
        import hashlib
        path = Path(task['directory']) / 'model.mjb'
        layout = np.load(Path(layout_root) / (path.parent.name + '.npz'))
        if (str(layout['mujoco']) != '2.3.7' or str(layout['model_sha256']) !=
                hashlib.sha256(path.read_bytes()).hexdigest()):
            raise ValueError('portable mass layout differs from the CPU 2.3.7 reference')
        return dict(indices=layout['indices'], nv=int(layout['nv']), nM=int(layout['nM']))
    import mujoco
    if mujoco.__version__ != '2.3.7':
        raise ValueError('mass layout reference requires MuJoCo 2.3.7')
    return mujoco.MjModel.from_binary_path(str(Path(task['directory']) / 'model.mjb'))


def dense_mass(model, packed):
    if isinstance(model, dict):
        indices = model['indices']
        packed = np.asarray(packed, np.float64).reshape(-1)
        if packed.size != model['nM']:
            raise ValueError('portable packed mass width differs')
        result = np.zeros(indices.shape, np.float64)
        result[indices >= 0] = packed[indices[indices >= 0]]
        return result
    import mujoco
    result = np.empty((model.nv, model.nv), np.float64)
    mujoco.mj_fullM(model, result, np.ascontiguousarray(packed, dtype=np.float64))
    return result


def sensitivity_floor(matrix, acceleration, metric):
    """Normwise first-order input rounding bound; only frozen inputs are used."""
    unit = np.finfo(np.float32).eps / 2
    order = np.inf if metric == 'max' else 2
    condition = float(np.linalg.cond(matrix, order))
    product = condition * unit
    if not np.isfinite(product) or product >= 1:
        raise ValueError('frozen mass conditioning cannot bound FP32 acceleration')
    # Perturbations in A and b each have a unit-roundoff allowance.
    ratio = 2 * product / (1 - product)
    scale = float(np.max(np.abs(acceleration))) if metric == 'max' else float(np.sqrt(np.mean(acceleration**2)))
    return ratio * scale, condition


def frozen_acceleration_floors(previous, model, worlds):
    result, report = {}, []
    for world in range(worlds):
        matrix = dense_mass(model, previous['qM'][world])
        for field in ('qacc', 'qacc_warmstart'):
            values = np.asarray(previous[field][world], np.float64).reshape(-1)
            for metric in ('max', 'rms'):
                floor, condition = sensitivity_floor(matrix, values, metric)
                result[field, world, metric] = floor
                report.append(dict(field=field, world=world, metric=metric,
                                   frozen_mass_condition=condition, sensitivity_floor=floor))
    return result, report


def balance_residual(matrix, acceleration, smooth_force, constraint_force):
    """Independent FP64 evaluation of the candidate's FP32 force balance."""
    a, smooth, constraint = (np.asarray(v, np.float64).reshape(-1)
                             for v in (acceleration, smooth_force, constraint_force))
    matrix = np.asarray(matrix, np.float64)
    residual = float(np.linalg.norm(matrix @ a - smooth - constraint, np.inf))
    scale = float(np.linalg.norm(matrix, np.inf) * np.linalg.norm(a, np.inf) +
                  np.linalg.norm(smooth, np.inf) + np.linalg.norm(constraint, np.inf))
    error = residual / max(scale, np.finfo(np.float64).tiny)
    unit = np.finfo(np.float32).eps / 2
    gamma = len(a) * unit / (1 - len(a) * unit)
    limit = 8 * gamma
    return dict(absolute_residual=residual, backward_error=error, limit=limit,
                passed=bool(np.isfinite(error) and error <= limit))


def acceleration_residuals(actual, model, worlds):
    result = []
    for world in range(worlds):
        matrix = dense_mass(model, actual['qM'][world])
        for field in ('qacc', 'qacc_warmstart'):
            result.append(dict(field=field, world=world,
                **balance_residual(matrix, actual[field][world], actual['qfrc_smooth'][world],
                                   actual['qfrc_constraint'][world])))
    return result
