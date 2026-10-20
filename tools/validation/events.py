"""Integer trajectory work and floating simulation time are separate checks."""
import numpy as np


def work_events(receipt, stage, *, substeps=None):
    worlds = receipt['worlds']
    count = receipt.get('substeps', substeps)
    if count is not None and (type(count) is not int or count < 1):
        raise ValueError('substeps must be a positive integer')
    counters, status = stage.get('counters', {}), stage.get('status', {})
    integration = counters.get('integration_count')
    resets, errors = counters.get('reset_events'), status.get('first_error')
    expected = receipt['input']['steps'] * count if count is not None else None
    complete = expected is not None and integration is not None and resets is not None and errors is not None
    passed = (integration == [expected] * worlds and resets == [0] * worlds and
              errors == [0] * worlds) if complete else None
    solves = receipt.get('build_config', {}).get('solves_per_substep')
    newton = counters.get('newton_attempts')
    expected_solves = expected * solves if expected is not None and solves is not None else None
    if expected_solves is not None:
        passed = bool(passed and newton == [expected_solves] * worlds)
    return dict(passed=passed, expected_integrations_per_world=expected,
                expected_solves_per_world=expected_solves, newton_attempts=newton,
                integration_count=integration, reset_events=resets, first_error=errors,
                scope='measured interval; unknown when required integer metadata is absent')


def simulation_time(actual, reference, precision):
    if precision not in ('fp64', 'fp32'):
        raise ValueError('unknown simulation time precision')
    a, b = np.asarray(actual, np.float64), np.asarray(reference, np.float64)
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        return dict(passed=False, max_difference=None, limit=None)
    limit = max(1e-10, 1024 * np.finfo(np.float32).eps * max(1., float(np.max(np.abs(b))))) if precision == 'fp32' else 1e-10
    difference = float(np.max(np.abs(a - b)))
    return dict(passed=difference <= limit, max_difference=difference, limit=limit)
