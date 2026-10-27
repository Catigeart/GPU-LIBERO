"""Compile every declared CUDA model instance with CMake; no environment construction."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precision', choices=('fp64', 'fp32'), default=None)
    parser.add_argument('--physics-engine', choices=('native', 'cuda_fp32'), default=None)
    parser.add_argument('--build-root', type=Path)
    parser.add_argument('--num-envs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--build-mode', choices=('dev', 'release', 'all'), default='dev',
                        help='compiler profile (default: current fast-compile dev profile)')
    parser.add_argument('--ordered-reductions', action='store_true',
                        help='also prepare the native ordered-reduction comparison products')
    parser.add_argument('--arch', type=int, default=80)
    parser.add_argument('--mathdx-root', type=Path)
    parser.add_argument('--compatibility', action='store_true',
                        help='prepare public OffScreenRenderEnv FP64 products instead of the batch catalogue')
    parser.add_argument('--suite', default='libero_10',
                        choices=('libero_spatial', 'libero_object', 'libero_goal', 'libero_90', 'libero_10'))
    parser.add_argument('--task-ids', type=int, nargs='+', help='compatibility profile task IDs (default: selected suite)')
    args = parser.parse_args()
    if min(args.num_envs, args.jobs) < 1:
        parser.error('environment and worker counts must be positive')
    import libero.gpu.precision as scalar
    scalar.select(args.precision or ('fp64' if args.compatibility else None))
    if args.mathdx_root is not None:
        os.environ['LIBERO_MATHDX_ROOT'] = str(args.mathdx_root.expanduser().resolve())
    from libero.gpu.host.build.batch import build_modes
    modes = build_modes(args.build_mode)
    if args.compatibility:
        if scalar.bits != 64:
            parser.error('--compatibility requires --precision fp64')
        if args.physics_engine == 'cuda_fp32':
            parser.error('--compatibility requires the native physics engine')
        from libero.gpu.host.build.compatibility import prepare
        from libero.gpu.host.build.project_paths import project_paths
        from libero.libero import benchmark
        suite = benchmark.get_benchmark(args.suite)()
        ids = list(range(suite.n_tasks)) if args.task_ids is None else args.task_ids
        if any(not 0 <= i < suite.n_tasks for i in ids):
            parser.error('task ID outside the selected suite')
        tasks = [dict(key=args.suite + '/' + suite.get_task(i).name,
                      bddl=suite.get_task_bddl_file_path(i)) for i in ids]
        result = prepare(project_paths(args.build_root).base, tasks, device=args.device,
                         progress=lambda event: print(json.dumps(event), flush=True))
        print(json.dumps(dict(status=result['status'], tasks=len(result['tasks']),
                              directory=result['project'], profile='compatibility')), flush=True)
        return
    if args.task_ids is not None:
        parser.error('--task-ids is a compatibility profile option')
    engine = args.physics_engine or ('cuda_fp32' if scalar.bits == 32 else 'native')
    if engine == 'cuda_fp32' and scalar.bits != 32:
        parser.error('cuda_fp32 requires --precision fp32')
    if engine == 'native' and scalar.bits != 64:
        parser.error('The native catalogue build is FP64; use the explicit developer targets for legacy FP32')
    if args.ordered_reductions and scalar.bits != 64:
        parser.error('--ordered-reductions requires FP64')
    from libero.gpu.host.build.batch import compile_project
    result = compile_project(args.build_root, precision='fp'+str(scalar.bits), modes=modes,
        arch=args.arch, mathdx_root=args.mathdx_root, jobs=args.jobs,
        ordered_reductions=args.ordered_reductions)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
