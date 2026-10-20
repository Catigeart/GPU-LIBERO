"""Compare matching CPU/GPU receipts; retain event mismatches separately."""
import csv
import json

from .common import load_endpoint, save_json
from .events import simulation_time, work_events


def summarize(cpu_root, gpu_root, output, *, substeps=None):
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
        time_check = simulation_time(a['time'], b['time'], gpu.get('precision', 'fp64'))
        events = work_events(gpu, g, substeps=substeps)
        rows.append(dict(task=gpu['task'], suite=gpu['suite'], task_id=gpu['task_id'],
            precision=gpu.get('precision', 'fp64'), worlds=gpu['worlds'], steps=gpu['input']['steps'],
            work_event_consistent=events['passed'], simulation_time_consistent=time_check['passed'],
            simulation_time_difference=time_check['max_difference'], simulation_time_limit=time_check['limit'],
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
                  work_event_consistent=aggregate([row for row in rows if row['work_event_consistent'] is True]),
                  simulation_time_consistent=aggregate([row for row in rows if row['simulation_time_consistent']]),
                  timing='world control steps; one measurement; cold/reset/graph preparation excluded', rows=rows)
    save_json(output / 'comparison.json', result)
    return result


def summarize_precision(fp64_root, fp32_root, output, *, substeps=None):
    """Paired saved receipts; failed/missing cells remain visible and untimed.

    This does not requalify endpoints. It reports the numerical qualification
    and load observations recorded by each worker, plus integer work checks.
    """
    roots = {'fp64': fp64_root, 'fp32': fp32_root}
    groups = {}
    for precision, root in roots.items():
        for path in sorted(root.glob('*.json')):
            receipt = json.loads(path.read_text())
            if receipt.get('backend') != 'gpu' or 'task_id' not in receipt:
                continue
            if receipt.get('precision', 'fp64') != precision:
                raise ValueError('receipt precision differs from selected folder: ' + str(path))
            key = (receipt['suite'], receipt['task_id'], receipt['worlds'], receipt['case'])
            groups.setdefault(key, {})[precision] = (path, receipt)
    if not groups:
        raise ValueError('no GPU precision receipts')
    rows, evidence = [], []
    for key, cells in sorted(groups.items()):
        row = dict(suite=key[0], task_id=key[1], worlds=key[2], case=key[3])
        if len(cells) == 2 and all(r.get('passed') and r.get('status') == 'qualified' for _, r in cells.values()):
            a, b = (cells[p][1] for p in ('fp64', 'fp32'))
            if not a.get('input', {}).get('signature') or a['input']['signature'] != b.get('input', {}).get('signature'):
                raise ValueError('precision input signatures differ: ' + str(key))
            # Configurations may differ in scalar precision only.
            configs = [{k: v for k, v in r.get('build_config', {}).items() if k != 'real_bits'} for r in (a, b)]
            if configs[0] != configs[1]:
                raise ValueError('precision build policies differ: ' + str(key))
        qualified = True
        for precision in roots:
            if precision not in cells:
                row[precision + '_status'] = 'missing'
                qualified = False
                continue
            path, receipt = cells[precision]
            if receipt.get('input', {}).get('steps') is not None:
                row['steps'] = receipt['input']['steps']
            prefix = precision + '_'
            row[prefix + 'status'] = receipt['status']
            row[prefix + 'passed'] = receipt.get('passed', False)
            qualified &= receipt.get('passed', False) and receipt['status'] == 'qualified'
            stages = receipt.get('runs', {})
            checks = [c for stage in stages.values() for c in stage.get('checks', [])]
            residuals = [c for stage in stages.values() for c in stage.get('acceleration_residuals', [])]
            row[prefix + 'unconditioned_acceleration_failures'] = sum(
                not c.get('unconditioned_passed', True) for c in checks if c['field'] in ('qacc', 'qacc_warmstart'))
            row[prefix + 'max_backward_error'] = max((c['backward_error'] for c in residuals), default=None)
            for field in ('qpos', 'qvel', 'qacc', 'ctrl', 'time'):
                row[prefix + field + '_change_from_frozen_fp64'] = max(
                    (c['endpoint_change'] for c in checks if c['field'] == field and c['metric'] == 'max'
                     and c['endpoint_change'] is not None), default=None)
            for name in ('creation_s', 'initial_reset_s', 'graph_prepare_s'):
                row[prefix + name] = receipt.get(name)
            perf = stages.get('performance', {})
            for name in ('throughput', 'elapsed_s', 'mean_ms', 'std_ms', 'cv', 'p95_ms'):
                row[prefix + name] = perf.get(name)
            events = [work_events(receipt, stage, substeps=substeps) for stage in stages.values()]
            row[prefix + 'work_event_consistent'] = (
                all(e['passed'] is True for e in events) if events and all(e['passed'] is not None for e in events) else None)
            row[prefix + 'runtime_compilations'] = len(receipt.get('runtime_compilations', []))
            evidence.append(dict(precision=precision, receipt=str(path), work_events=events,
                error=receipt.get('error'),
                shared_device_policy=receipt.get('shared_device_policy'),
                gpu_load_before=receipt.get('gpu_load_before'), gpu_load_after=receipt.get('gpu_load_after'),
                native_resources=receipt.get('native_resources'), comparison_policy=receipt.get('comparison_policy')))
            qualified &= bool(receipt.get('performance_measured') and perf.get('elapsed_s'))
        row['paired_qualified'] = bool(qualified)
        if qualified:
            row['fp32_over_current_fp64'] = row['fp64_elapsed_s'] / row['fp32_elapsed_s']
        rows.append(row)
    valid = [row for row in rows if row['paired_qualified']]
    aggregate = dict(matched_tasks=len(valid), requested_tasks=len(rows), repeats=1, physics_executed=False,
        runtime_compilation_events=sum(row.get(p + '_runtime_compilations', 0) for row in rows for p in roots),
        all_work_events_verified=bool(valid) and len(valid) == len(rows) and all(
            row.get(p + '_work_event_consistent') is True for row in valid for p in roots),
        timing='world control steps; current paired precisions; load observations do not prove an idle device')
    if valid:
        work = sum(row['worlds'] * row['steps'] for row in valid)
        for precision in roots:
            elapsed = sum(row[precision + '_elapsed_s'] for row in valid)
            aggregate[precision + '_total_s'] = elapsed
            aggregate[precision + '_throughput'] = work / elapsed
        aggregate['fp32_over_current_fp64'] = aggregate['fp64_total_s'] / aggregate['fp32_total_s']
    output.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(name for row in rows for name in row))
    with (output / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    result = dict(aggregate=aggregate, rows=rows, evidence=evidence)
    save_json(output / 'comparison.json', result)
    return result
