"""Measure complete same-task GPU batch steps, excluding setup/reset/policy."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', default='libero_10')
    parser.add_argument('--task-ids', type=int, nargs='+')
    parser.add_argument('--num-envs', type=int, default=8)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--precision', choices=('fp64', 'fp32'), default=None)
    parser.add_argument('--physics-engine', choices=('auto', 'native', 'mjwarp'), default='auto')
    parser.add_argument('--fast-compile', action='store_true', help='select an existing MJWarp dev-controller profile')
    parser.add_argument('--camera', action='store_true', help='two RGB cameras at 128x128')
    parser.add_argument('--memory-limit-gib', type=float)
    parser.add_argument('--build-root', type=Path,
                        help='persistent compiler cache (default: LIBERO_GPU_CACHE_DIR/project)')
    parser.add_argument('--build-only', action='store_true',
                        help='prepare compiled products and owners; do not reset, warm up, or step')
    parser.add_argument('--output', type=Path, default=Path('results/gpu_batch.json'))
    parser.add_argument('--progress-output', type=Path,
                        help='live progress JSON; its IO is outside measured step blocks')
    parser.add_argument('--progress-every', type=int, default=100)
    args = parser.parse_args()
    if min(args.steps, args.repeats) < 1 or args.warmup < 0:
        parser.error('steps/repeats must be positive; warmup must be nonnegative')
    if args.progress_every < 1:
        parser.error('progress-every must be positive')
    def progress(stage, task_id, **fields):
        if args.progress_output is None:
            return
        value = dict(stage=stage, task_id=task_id, updated_at=time.time(), **fields)
        args.progress_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.progress_output.with_suffix('.'+str(os.getpid())+'.tmp')
        temporary.write_text(json.dumps(value, indent=2)+'\n')
        temporary.replace(args.progress_output)
    from libero.gpu import precision as scalar
    from libero.gpu.host.backends.selection import resolve_engine
    engine = resolve_engine(args.physics_engine, args.precision)
    selected_precision = scalar.select(args.precision or ('fp32' if engine == 'mjwarp' else None))
    if engine == 'mjwarp' and args.memory_limit_gib is not None:
        parser.error('MJWarp profiles use nconmax/njmax capacity budgets; --memory-limit-gib is native-only')
    from libero.libero import benchmark
    from libero.libero.envs import GpuBatchEnv
    suite = benchmark.get_benchmark(args.suite)()
    task_ids = list(range(suite.n_tasks)) if args.task_ids is None else args.task_ids
    if any(not 0 <= i < suite.n_tasks for i in task_ids):
        parser.error('task ID outside the selected suite')
    policy = {} if args.memory_limit_gib is None else {
        'limit_bytes': int(args.memory_limit_gib * 1024**3)}
    report = dict(suite=args.suite, num_envs=args.num_envs, steps=args.steps,
                  warmup=args.warmup, repeats=args.repeats, seed=args.seed,
                  use_camera_obs=args.camera, device=args.device, build_only=args.build_only,
                  allow_compile=False, debug=False,
                  precision=selected_precision, physics_engine=engine,
                  timing='Continuous step blocks; setup, reset, warmup and progress IO excluded',
                  tasks=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for task_id in task_ids:
        task = suite.get_task(task_id)
        print(f'Preparing task {task_id}: {task.name}', flush=True)
        progress('preparing', task_id)
        start = time.perf_counter()
        try:
            env = GpuBatchEnv(suite.get_task_bddl_file_path(task_id), args.num_envs,
                              init_states=suite.get_task_init_states(task_id), seed=args.seed,
                              device=args.device, memory_budget=policy or None,
                              precision=selected_precision, physics_engine=engine,
                              **({'mjwarp_options': {'fast_compile': args.fast_compile}} if engine == 'mjwarp' else {}),
                              build_root=args.build_root, initialize=False,
                              allow_compile=False, debug=False,
                              use_camera_obs=args.camera, ignore_done=True)
        except MemoryError as error:
            row = dict(task_id=task_id, task=task.name, status='insufficient_memory',
                       setup_seconds=time.perf_counter()-start, error=str(error))
            report['tasks'].append(row)
            args.output.write_text(json.dumps(report, indent=2))
            print(json.dumps(row), flush=True)
            continue
        with env:
            constructor_seconds = time.perf_counter() - start
            graph_capture_seconds = 0.0
            if not args.build_only:
                from libero.gpu.host.build.compilation import compilation_scope
                graph_start = time.perf_counter()
                with compilation_scope(False):
                    env._physics.prepare_control_graph()
                graph_capture_seconds = time.perf_counter() - graph_start
            setup = time.perf_counter() - start
            build = dict(env.build_report)
            if engine == 'mjwarp':
                build['directory'] = str(env.build_root / 'mjwarp')
            compiler_invocations = sum(len(e['commands_executed']) for e in build['events']
                                       if e['kind'] != 'constraint-source')
            if compiler_invocations:
                raise RuntimeError('benchmark preparation invoked a compiler')
            print(f'Build cache: {build["directory"]}; '
                  f'compiler invocations: {compiler_invocations}', flush=True)
            if args.build_only:
                diagnostics = env.diagnostics()
                if any(diagnostics['actual_integration_count']):
                    raise RuntimeError('build-only preparation advanced physics')
                row = dict(task_id=task_id, task=task.name, status='built',
                           setup_seconds=setup, build=build,
                           compiler_invocations=compiler_invocations, diagnostics=diagnostics)
                report['tasks'].append(row)
                args.output.write_text(json.dumps(report, indent=2))
                print(json.dumps(row), flush=True)
                continue
            # Pre-generated bounded actions; inference/file IO is outside timing.
            actions = np.stack([
                np.random.RandomState(args.seed+i).uniform(
                    -.05, .05, (args.warmup+args.steps, env.action_dim))
                for i in range(args.num_envs)], axis=1)
            samples, failures = [], []
            for repeat in range(args.repeats):
                progress('resetting', task_id, repeat=repeat)
                reset_start = time.perf_counter()
                env.reset(seed=args.seed)
                reset_seconds = time.perf_counter()-reset_start
                progress('warmup', task_id, repeat=repeat, steps=args.warmup)
                warmup_start = time.perf_counter()
                for completed_warmup, action in enumerate(actions[:args.warmup], 1):
                    env.step(action)
                    progress('warmup', task_id, repeat=repeat,
                             completed_steps=completed_warmup, total_steps=args.warmup)
                warmup_seconds = time.perf_counter()-warmup_start
                before = env.diagnostics()
                elapsed = 0.0
                step_seconds = []
                measured = actions[args.warmup:]
                progress('measuring', task_id, repeat=repeat, completed_steps=0,
                         total_steps=args.steps, measured_seconds=elapsed)
                # One continuous rollout; only progress IO is between timers.
                for offset in range(0, args.steps, args.progress_every):
                    block = measured[offset:offset+args.progress_every]
                    start = time.perf_counter()
                    for action in block:
                        step_start = time.perf_counter()
                        env.step(action)
                        step_seconds.append(time.perf_counter()-step_start)
                    elapsed += time.perf_counter()-start  # step checks/waits.
                    completed = offset+len(block)
                    progress('measuring', task_id, repeat=repeat, completed_steps=completed,
                             total_steps=args.steps, measured_seconds=elapsed)
                    if args.progress_output is not None:
                        print(f'Measured {completed}/{args.steps} steps; '
                              f'{elapsed:.3f} seconds', flush=True)
                after = env.diagnostics()
                delta = np.array(after['actual_integration_count'])-before['actual_integration_count']
                valid = bool(np.all(delta == args.steps*env.substeps) and not any(after['first_error']))
                if not valid:
                    from libero.gpu.host.runtime.step.recovery_status import _sources
                    names = ('runtime_error','foundation_error','newton_error','forward_error',
                        'model_epoch','expected_model_epoch','model_ready','state_epoch','expected_state',
                        'forward_epoch','expected_forward','newton_attempt','newton_written',
                        'expected_newton','integration_count','step_valid','forward_valid','newton_valid')
                    failures.append(dict(expected_integrations=args.steps*env.substeps,
                        actual_integrations=delta.tolist(), before=before, after=after,
                        boundary_sources={name:array.numpy().tolist()
                            for name,array in zip(names,_sources(env._physics.bundle))}))
                compiler_invocations = sum(len(e['commands_executed']) for e in build['events']
                                           if e['kind'] != 'constraint-source')
                if compiler_invocations:
                    raise RuntimeError('benchmark execution invoked a compiler')
                milliseconds = np.asarray(step_seconds)*1000
                samples.append(dict(valid=valid, seconds=elapsed,
                    ms_per_batch=elapsed*1000/args.steps if valid else None,
                    env_steps_per_second=args.num_envs*args.steps/elapsed if valid else None,
                    per_env_steps_per_second=args.steps/elapsed if valid else None,
                    reset_seconds=reset_seconds, warmup_seconds=warmup_seconds,
                    integration_delta=delta.tolist(), step_seconds=step_seconds,
                    step_ms_min=float(np.min(milliseconds)),
                    step_ms_median=float(np.median(milliseconds)),
                    step_ms_p95=float(np.percentile(milliseconds, 95)),
                    step_ms_max=float(np.max(milliseconds)),
                    step_ms_std=float(np.std(milliseconds)),
                    step_time_cv=float(np.std(milliseconds)/np.mean(milliseconds))))
                if not valid:break
            row = dict(task_id=task_id, task=task.name,
                       status='physics_error' if failures else 'ok', failures=failures, setup_seconds=setup,
                       constructor_seconds=constructor_seconds,
                       graph_capture_seconds=graph_capture_seconds,
                       build=build, compiler_invocations=compiler_invocations,
                       samples=samples, diagnostics=after,
                       median_ms_per_batch=None if failures else float(np.median([s['ms_per_batch'] for s in samples])),
                       median_env_steps_per_second=None if failures else float(np.median([s['env_steps_per_second'] for s in samples])))
            report['tasks'].append(row)
            args.output.write_text(json.dumps(report, indent=2))
            progress('failed' if failures else 'complete', task_id, completed_steps=args.steps,
                     total_steps=args.steps, measured_seconds=samples[-1]['seconds'])
            print(json.dumps({k:v for k,v in row.items() if k not in ('build','diagnostics')}), flush=True)


if __name__ == '__main__':
    main()
