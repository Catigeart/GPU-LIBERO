"""Compare matching CPU/GPU receipts; retain event mismatches separately."""
import csv
import json

import numpy as np

from .common import load_endpoint, save_json


def summarize(cpu_root, gpu_root, output):
    rows = []
    for path in sorted(gpu_root.glob('*.json')):
        gpu = json.loads(path.read_text())
        if gpu.get('status') != 'qualified' or not gpu.get('passed'):
            continue
        cpu_path = cpu_root / path.name
        if not cpu_path.exists() and gpu.get('case') == 'varying_cache_miss':
            cpu_path = cpu_root / path.name.replace('_varying_cache_miss.json', '_varying.json')
        if not cpu_path.exists():
            raise ValueError('CPU receipt missing: ' + str(cpu_path))
        cpu = json.loads(cpu_path.read_text())
        if cpu.get('status') != 'recorded_cpu' or not cpu.get('passed'):
            raise ValueError('CPU receipt is not a valid reference: ' + str(cpu_path))
        if cpu['input']['signature'] != gpu['input']['signature']:
            raise ValueError('CPU/GPU input signatures differ: ' + path.name)
        if not cpu['performance_measured'] or not gpu['performance_measured']:
            continue
        c = next(value for value in cpu['runs'].values() if 'elapsed_s' in value)
        g = gpu['runs']['performance']
        a = load_endpoint(gpu_root / (path.stem + '-performance.npz'))
        b = load_endpoint(cpu_path.with_suffix('.npz'))
        consistent = bool(np.allclose(a['time'], b['time'], rtol=0, atol=1e-10))
        rows.append(dict(task=gpu['task'], suite=gpu['suite'], task_id=gpu['task_id'],
            worlds=gpu['worlds'], steps=gpu['input']['steps'], time_event_consistent=consistent,
            cpu_elapsed_s=c['elapsed_s'], gpu_elapsed_s=g['elapsed_s'],
            cpu_throughput=c['throughput'], gpu_throughput=g['throughput'],
            gpu_speedup=c['elapsed_s'] / g['elapsed_s'], cpu_mean_ms=c['mean_ms'],
            gpu_mean_ms=g['mean_ms'], cpu_std_ms=c['std_ms'], gpu_std_ms=g['std_ms']))
    if not rows:
        raise ValueError('no matching qualified CPU/GPU single-measurement receipts')

    def aggregate(items):
        if not items:
            return dict(tasks=0)
        work = sum(row['worlds'] * row['steps'] for row in items)
        cpu = sum(row['cpu_elapsed_s'] for row in items)
        gpu = sum(row['gpu_elapsed_s'] for row in items)
        return dict(tasks=len(items), cpu_throughput=work / cpu, gpu_throughput=work / gpu,
                    gpu_speedup=cpu / gpu, faster_than_cpu=sum(row['gpu_speedup'] > 1 for row in items))
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'task-summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(all_tasks=aggregate(rows),
                  time_event_consistent=aggregate([row for row in rows if row['time_event_consistent']]),
                  timing='world control steps; one measurement; cold/reset/graph preparation excluded', rows=rows)
    save_json(output / 'comparison.json', result)
    return result
