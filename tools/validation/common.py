"""Shared inputs, frozen-reference comparisons and control-step statistics."""
import hashlib
import json
from pathlib import Path

import numpy as np

FIELDS = ('qpos', 'qvel', 'qacc', 'qacc_warmstart', 'ctrl', 'time')
SUITES = ('libero_10', 'libero_90', 'libero_goal', 'libero_object', 'libero_spatial')
CASES = ('varying', 'varying_cache_miss', 'zero')


def actions(step, worlds, case='varying'):
    """The established per-world sinusoidal OSC sequence, unchanged."""
    if case not in CASES or worlds < 1:
        raise ValueError('unknown action case or invalid world count')
    result = np.zeros((worlds, 7), np.float64)
    if case != 'zero':
        phase = .19 * np.arange(worlds)[:, None]
        result[:, :3] = .12 * np.sin(.31 * step + np.arange(3) * .7 + phase)
        result[:, 3:6] = .05 * np.sin(.23 * step + np.arange(3) * .5 + phase)
        result[:, 6] = .2 * np.sin(.2 * step + phase[:, 0])
    return result


def select_tasks(rows, selectors):
    """Resolve suite names or zero-based suite:task IDs without duplicates."""
    selected = []
    for selector in selectors:
        matches = [row for row in rows if selector == row['suite'] or
                   selector == f"{row['suite']}:{row['task_id']}" or
                   selector == 'all' and row['suite'] in SUITES]
        if not matches:
            raise ValueError('unknown task selection: ' + selector)
        for row in matches:
            if row not in selected:
                selected.append(row)
    return selected


def label(task, worlds, case):
    return f"{task['suite']}__{task['task_id']:03d}_n{worlds}_{case}"


def input_record(task, bank, *, worlds, warmup, steps, seed, case):
    sequence = np.stack([actions(k, worlds, case) for k in range(warmup + steps)])
    bank = np.ascontiguousarray(bank[:worlds], dtype=np.float64)
    record = dict(task=task['key'], model_sha256=task['model_sha256'],
                  worlds=worlds, warmup=warmup, steps=steps, seed=seed,
                  state_shape=list(bank.shape), state_sha256=hashlib.sha256(bank.tobytes()).hexdigest(),
                  actions_shape=list(sequence.shape), actions_sha256=hashlib.sha256(sequence.tobytes()).hexdigest(),
                  cameras=False, control_freq=20)
    record['signature'] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    # A forced position-cache miss changes the GPU implementation path, not
    # physical inputs. Its CPU endpoint can reuse the ordinary varying case.
    record['case'] = case
    return record


def load_endpoint(path, prefix=''):
    with np.load(path, allow_pickle=False) as values:
        result = {key[len(prefix):]: values[key].copy() for key in values.files if key.startswith(prefix)}
    if not result:
        raise ValueError('no endpoint fields for prefix ' + repr(prefix) + ': ' + str(path))
    return result


def compare_endpoints(actual, previous, cpu, worlds, *, baseline_samples=(), multiplier=3., floor=1e-10):
    """Always include the original old endpoint; never derive budgets from new.

    Additional samples must be independently obtained frozen OLD endpoints.
    This function never collects samples or expands a budget automatically.
    Missing fields, mismatched shapes and nonfinite references fail explicitly.
    """
    if worlds < 1 or not np.isfinite(multiplier) or multiplier <= 0 or not np.isfinite(floor) or floor < 0:
        raise ValueError('invalid comparison policy')
    for name, values in [('candidate', actual), ('baseline', previous), ('cpu', cpu)]:
        missing = set(FIELDS) - values.keys()
        if missing:
            raise ValueError(name + ' lacks required fields: ' + repr(sorted(missing)))
    checks = []
    for field in sorted(previous.keys() & cpu.keys()):
        if field not in actual:
            raise ValueError('candidate lacks reference field: ' + field)
        arrays = [np.asarray(values[field], dtype=np.float64)
                  for values in (actual, previous, cpu, *(sample for sample in baseline_samples if field in sample))]
        shape = arrays[0].shape
        if not shape or any(not a.shape or a.shape[0] != worlds or a.size != arrays[0].size for a in arrays):
            raise ValueError('endpoint shape mismatch: ' + field)
        if not all(np.isfinite(a).all() for a in arrays[1:]):
            raise ValueError('nonfinite frozen reference: ' + field)
        if not arrays[0].size:
            continue
        value, old, ref, *samples = [a.reshape(worlds, -1) for a in arrays]
        for world in range(worlds):
            for metric, fn in [('max', lambda x: float(np.max(np.abs(x)))),
                               ('rms', lambda x: float(np.sqrt(np.mean(x * x))))]:
                error = fn(value[world] - ref[world])
                previous_error = max(fn(sample[world] - ref[world]) for sample in (old, *samples))
                limit = max(multiplier * previous_error, floor)
                change = fn(value[world] - old[world])
                checks.append(dict(field=field, world=world, metric=metric,
                                   error=error if np.isfinite(error) else None,
                                   previous_error=previous_error,
                                   endpoint_change=change if np.isfinite(change) else None,
                                   limit=limit, passed=bool(np.isfinite(error) and error <= limit)))
    return checks


def timing_summary(milliseconds, elapsed_s, worlds):
    """Throughput counts world CONTROL steps, never physical substeps."""
    values = np.asarray(milliseconds, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('nonempty finite nonnegative step timings required')
    if not np.isfinite(elapsed_s) or elapsed_s <= 0 or worlds < 1:
        raise ValueError('positive elapsed time and world count required')
    return dict(control_ms=values.tolist(), elapsed_s=float(elapsed_s),
                throughput=float(worlds * values.size / elapsed_s),
                mean_ms=float(values.mean()), std_ms=float(values.std()),
                p95_ms=float(np.percentile(values, 95)),
                cv=float(values.std() / values.mean()) if values.mean() else 0.)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
