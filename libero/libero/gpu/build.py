"""Instantiate all shipped LIBERO tasks and compile one reusable GPU project."""
import argparse
import json
from pathlib import Path


def main():
    from libero.gpu.host.build.aot import project_root, compile_catalog
    from libero.gpu.host.build.task_catalog import prepare_catalog, write_parameter_catalog
    from libero.gpu.host.paths import GPU_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-root', type=Path)
    parser.add_argument('--num-envs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--catalog-jobs', type=int, default=4)
    args = parser.parse_args()
    if min(args.num_envs, args.jobs, args.catalog_jobs) < 1:
        parser.error('environment and worker counts must be positive')
    root = project_root(args.build_root)
    root.mkdir(parents=True, exist_ok=True)
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
