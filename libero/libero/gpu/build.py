"""Instantiate all shipped LIBERO tasks and compile one reusable GPU project."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precision', choices=('fp64', 'fp32'), default=None)
    parser.add_argument('--build-root', type=Path)
    parser.add_argument('--num-envs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--catalog-jobs', type=int, default=4)
    parser.add_argument('--compatibility', action='store_true',
                        help='prepare signed public OffScreenRenderEnv FP64 products instead of the batch catalogue')
    parser.add_argument('--suite', default='libero_10',
                        choices=('libero_spatial', 'libero_object', 'libero_goal', 'libero_90', 'libero_10'))
    parser.add_argument('--task-ids', type=int, nargs='+', help='compatibility profile task IDs (default: selected suite)')
    args = parser.parse_args()
    if min(args.num_envs, args.jobs, args.catalog_jobs) < 1:
        parser.error('environment and worker counts must be positive')
    import libero.gpu.precision as scalar
    scalar.select(args.precision or ('fp64' if args.compatibility else None))
    if args.compatibility:
        if scalar.bits != 64:
            parser.error('--compatibility requires --precision fp64')
        if args.device != 'cuda:0':
            parser.error('the compatibility entry uses cuda:0; select the visible device before launching')
        from libero.gpu.host.build.compatibility import prepare
        from libero.gpu.host.build.project_paths import project_paths
        from libero.libero import benchmark
        suite = benchmark.get_benchmark(args.suite)()
        ids = list(range(suite.n_tasks)) if args.task_ids is None else args.task_ids
        if any(not 0 <= i < suite.n_tasks for i in ids):
            parser.error('task ID outside the selected suite')
        tasks = [dict(key=args.suite + '/' + suite.get_task(i).name,
                      bddl=suite.get_task_bddl_file_path(i)) for i in ids]
        result = prepare(project_paths(args.build_root).base, tasks,
                         progress=lambda event: print(json.dumps(event), flush=True))
        print(json.dumps(dict(status=result['status'], tasks=len(result['tasks']),
                              directory=result['project'], profile='compatibility')), flush=True)
        return
    if args.task_ids is not None:
        parser.error('--task-ids is a compatibility profile option')
    from libero.gpu.host.build.aot import compile_catalog
    from libero.gpu.host.build.project_paths import prepare_precision_project
    from libero.gpu.host.build.task_catalog import prepare_catalog, write_parameter_catalog
    from libero.gpu.host.paths import GPU_ROOT
    root = prepare_precision_project(args.build_root).precision
    def census(row, done, total):
        print(json.dumps(dict(phase='instantiate', done=done, total=total, task=row['key'])), flush=True)
    tasks = prepare_catalog(root, jobs=args.catalog_jobs, seed=args.seed, progress=census)
    parameters = GPU_ROOT / 'data/libero_task_parameters.json'
    write_parameter_catalog(tasks, parameters)
    def built(row, done, total):
        print(json.dumps(dict(phase='compile', done=done, total=total, task=row['key'],
            native_compiler_invocations=row['native_compiler_invocations'],
            warp_compilations=row['warp_compilations'], seconds=row['seconds'])), flush=True)
    result = compile_catalog(root, tasks, num_envs=args.num_envs, seed=args.seed,
                             device=args.device, jobs=args.jobs, progress=built)
    print(json.dumps(dict(status=result['status'], tasks=len(result['tasks']),
                          artifacts=len(result['artifacts']), directory=str(root))), flush=True)


if __name__ == '__main__':
    main()
