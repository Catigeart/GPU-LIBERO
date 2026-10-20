"""Reference-only trajectory envelopes for independent upstream FP32 replays.

This is a separate engine-equivalence policy, not a change to the native/CPU
regression budgets. The candidate never sets its own tolerance. One-substep
checks still use the original FP32 roundoff screen.
"""
import numpy as np


def compare_replays(actual, first, second):
    checks = []
    for name in first:
        a, x, y = (np.asarray(values[name], np.float64) for values in (actual, first, second))
        if a.shape != x.shape or x.shape != y.shape or a.ndim < 1:
            raise ValueError('replay shapes differ: ' + name)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('upstream reference is not finite: ' + name)
        for world in range(len(x)):
            av, xv, yv = (v[world].reshape(-1) for v in (a, x, y))
            for metric in ('max', 'rms'):
                def measure(value):
                    return float(np.max(abs(value))) if metric == 'max' else float(np.sqrt(np.mean(value * value)))
                spread = measure(xv - yv)
                scale = max(1., measure(xv), measure(yv))
                floor = 1024 * np.finfo(np.float32).eps * scale
                # One decimal order, as required for the upstream FP32 path.
                # Native/CPU regression policies keep their existing budgets.
                limit = max(floor, 10 * spread)
                error = max(measure(av - xv), measure(av - yv))
                if name in ('ctrl', 'time'):
                    limit = 0.  # Same recorded commands and number/order of time advances.
                checks.append(dict(field=name, world=world, metric=metric,
                    error=error, raw_self_spread=spread, roundoff_floor=floor, limit=limit,
                    passed=bool(np.isfinite(error) and error <= limit)))
    return checks
