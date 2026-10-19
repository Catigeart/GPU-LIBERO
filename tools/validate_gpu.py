"""Stable CPU/GPU validation entry. No compilation or execution on import."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run', help='record or check trajectories; GPU execution is load-only')
    run.add_argument('--backend', choices=('cpu', 'gpu'), required=True)
    run.add_argument('--tasks', nargs='+', default=['libero_10'], help='suite names, suite:ID or all')
    run.add_argument('--build-root', type=Path, required=True, help='existing project containing TASK_INSTANCES.json')
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--worlds', type=int, default=2)
    run.add_argument('--warmup', type=int, default=5)
    run.add_argument('--steps', type=int, default=20)
    run.add_argument('--seed', type=int, default=0)
    run.add_argument('--case', choices=('varying', 'varying_cache_miss', 'zero'), default='varying')
    run.add_argument('--device', default='cuda:0')
    run.add_argument('--fast-compile', action='store_true', help='select existing dev products; does not compile')
    run.add_argument('--device-lto', action='store_true', help='select existing LTO products; does not link')
    run.add_argument('--performance', action='store_true', help='time once, only after candidate correctness passes')
    run.add_argument('--record-reference', action='store_true', help='record a frozen GPU baseline, without claiming qualification')
    run.add_argument('--cpu-reference', type=Path)
    run.add_argument('--baseline-reference', type=Path)
    run.add_argument('--baseline-sample-reference', type=Path, action='append', default=[],
                     help='additional explicitly recorded OLD GPU references, never auto-collected')
    run.add_argument('--diagnose-n1', action='store_true', help='after N2 numerical failure only, use existing matching N1 references')
    run.add_argument('--worker-task', help=argparse.SUPPRESS)

    compare = commands.add_parser('compare', help='compare saved endpoints without executing physics')
    for name in ('candidate', 'baseline', 'cpu'):
        compare.add_argument('--' + name, type=Path, required=True)
        compare.add_argument('--' + name + '-prefix', default='')
    compare.add_argument('--old-sample', type=Path, action='append', default=[],
                         help='additional verified frozen OLD endpoint; original baseline is always included')
    compare.add_argument('--old-sample-prefix', default='')
    compare.add_argument('--worlds', type=int, default=2)
    compare.add_argument('--output', type=Path, required=True)

    summary = commands.add_parser('summarize', help='summarize new matching CPU/GPU receipts')
    summary.add_argument('--cpu', type=Path, required=True)
    summary.add_argument('--gpu', type=Path, required=True)
    summary.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'compare':
        from validation.common import compare_endpoints, load_endpoint, save_json
        checks = compare_endpoints(load_endpoint(args.candidate, args.candidate_prefix),
            load_endpoint(args.baseline, args.baseline_prefix), load_endpoint(args.cpu, args.cpu_prefix),
            args.worlds, baseline_samples=[load_endpoint(path, args.old_sample_prefix) for path in args.old_sample])
        result = dict(passed=all(c['passed'] for c in checks), checks=checks,
                      candidate=str(args.candidate), baseline=str(args.baseline), cpu=str(args.cpu),
                      extra_old_samples=[str(path) for path in args.old_sample],
                      multiplier=3, absolute_floor=1e-10, physics_executed=False,
                      scope='endpoint error scale only; caller must establish same inputs and frozen sample provenance')
        save_json(args.output, result)
        print(json.dumps(dict(passed=result['passed'], checks=len(checks))))
        return 0 if result['passed'] else 2
    if args.command == 'summarize':
        from validation.summary import summarize
        result = summarize(args.cpu, args.gpu, args.output)
        print(json.dumps({key: value for key, value in result.items() if key != 'rows'}))
        return 0
    if not 1 <= args.worlds <= 128 or args.warmup < 0 or args.steps < 1:
        parser.error('worlds must be 1..128, warmup >= 0 and steps > 0')
    if args.diagnose_n1 and args.worlds != 2:
        parser.error('--diagnose-n1 requires the primary run to use N=2')
    if args.backend == 'gpu' and not args.record_reference and not (args.cpu_reference and args.baseline_reference):
        parser.error('GPU validation requires --cpu-reference and --baseline-reference, or explicit --record-reference')
    if args.backend == 'cpu':
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'LP_NUM_THREADS'):
            os.environ[name] = '1'
    from validation.common import label, save_json, select_tasks
    from validation.runner import run_task
    args.build_root = args.build_root.resolve()
    args.output = args.output.resolve()
    tasks = select_tasks(json.loads((args.build_root / 'TASK_INSTANCES.json').read_text())['tasks'],
                         [args.worker_task] if args.worker_task else args.tasks)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker_task:
        return run_task(args, tasks[0])
    attempts = []
    # Bound allocator/context lifetime. Each task gets one worker, while its
    # correctness and timing runs share the same environment and captured graph.
    original = sys.argv[1:]
    for task in tasks:
        selector = f"{task['suite']}:{task['task_id']}"
        stem = label(task, args.worlds, args.case)
        log = args.output / (stem + '.log')
        with log.open('x') as stream:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), *original,
                                      '--worker-task', selector], stdout=stream, stderr=subprocess.STDOUT)
        row = dict(task=selector, worlds=args.worlds, returncode=process.returncode, log=str(log))
        if process.returncode == 2 and args.diagnose_n1 and args.backend == 'gpu':
            diagnostic = label(task, 1, args.case)
            available = all((root / (diagnostic + '.json')).is_file()
                            for root in (args.cpu_reference, args.baseline_reference) if root is not None)
            if args.cpu_reference and args.baseline_reference and available:
                command = [sys.executable, str(Path(__file__).resolve()), *original,
                           '--worker-task', selector, '--worlds', '1']
                # The primary failure remains failed even when N1 passes.
                command = [value for value in command if value not in ('--diagnose-n1', '--performance')]
                diagnostic_log = args.output / (diagnostic + '.log')
                with diagnostic_log.open('x') as stream:
                    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
                row['n1_diagnostic'] = dict(returncode=result.returncode, log=str(diagnostic_log))
            else:
                row['n1_diagnostic'] = dict(skipped='matching N1 CPU and frozen GPU references are absent')
        attempts.append(row)
        print(json.dumps(row), flush=True)
        save_json(args.output / 'summary.json', dict(passed=all(r['returncode'] == 0 for r in attempts),
            requested=len(tasks), completed=len(attempts), n1_policy='only after N2 numerical/work failure', attempts=attempts))
        if process.returncode:
            return process.returncode if process.returncode > 0 else 3
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
