"""Explicit offline preparation; no Warp installation or physics execution.

Consumes the common CPU scene catalogue, then compiles only the independent
runtime, transport and shared handwritten CUDA targets. Scene preparation uses
CPU MuJoCo; it neither uploads a world nor advances simulation.
"""
import argparse
import gc
import json
from pathlib import Path
import subprocess

COMPONENTS = ('controller', 'hessian', 'direction', 'constraint', 'linesearch',
    'solver_state', 'mass_solve', 'smooth_force', 'mass_factor', 'mass_matrix',
    'velocity', 'bias', 'postconstraint', 'passive', 'actuation', 'euler',
    'transmission', 'kinematics', 'com_pos', 'camlight', 'fluid', 'external_wrench',
    'constraint_build', 'broadphase', 'plane', 'box', 'convex', 'queries', 'publication')


def prepare_project(build_root, *, tasks=('all',), modes=('dev',), skip_build=False,
                    jobs=2, arch=80, mathdx_root=None):
    """Shared implementation for the unified entry and the legacy CLI."""
    if jobs < 1:
        raise ValueError('--jobs must be positive')
    from libero.gpu.host.paths import REPO_ROOT, GPU_ROOT
    from libero.gpu.host.build.project_paths import project_paths
    from libero.libero import get_libero_path
    from .scene import build_scene
    from .model import prepare
    from .layout_specs import MODEL_ARRAYS
    root = project_paths(build_root,precision='fp32').base
    catalogue = json.loads((root / 'TASK_INSTANCES.json').read_text())['tasks']
    selected = [t for t in catalogue if any(s in ('all', t['suite'], t['suite']+':'+str(t['task_id'])) for s in tasks)]
    if not selected:
        raise ValueError('No matching catalogue tasks')
    rows = []
    for task in selected:
        built, bank, _, identity = build_scene(root,
            Path(get_libero_path('bddl_files')) / task['key'], task['seed'],
            dict(use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False,
                 hard_reset=False, ignore_done=True), prepare=True)
        prepared = prepare(built.native_model)
        row = dict(task=task['key'], nv=built.native_model.nv, nq=built.native_model.nq,
                   arrays=len(MODEL_ARRAYS), physics_integrations=0, identity=identity)
        rows.append(row)
        print(json.dumps(dict(phase='cpu-scene', task=task['key'], done=len(rows), total=len(selected))), flush=True)
        del built, bank, prepared
        gc.collect()
    folders = []
    for mode in modes:
        folder = root / 'cmake-control' / f'sm{arch}-{mode}'
        folders.append(str(folder))
        if skip_build:
            continue
        source = REPO_ROOT if (REPO_ROOT/'CMakeLists.txt').is_file() else GPU_ROOT
        configure = ['cmake', '-S', str(source), '-B', str(folder), '-G', 'Ninja',
            '-DCMAKE_BUILD_TYPE=Release', '-DFLASH_CUDA_ARCH='+str(arch),
            '-DFLASH_BUILD_NATIVE=OFF',
            '-DFLASH_FAST_COMPILE='+('OFF' if mode == 'release' else 'ON')]
        if mathdx_root:
            configure.append('-DFLASH_MATHDX_ROOT='+str(mathdx_root))
        subprocess.run(configure, check=True)
        targets = ['flash_cuda_runtime', 'flash_cuda_bridges_fp32', 'flash_fp32_fused_tail',
                   'flash_fp32_fused_observations', 'flash_fp32_fused_position', 'flash_fp32_fused_kinematic_mass',
                   'flash_fp32_fused_motion', 'flash_fp32_fused_motion_passive', 'flash_fp32_fused_control_prepare',
                   'flash_fp32_fused_controller', 'flash_fp32_fused_actuation_solve', 'flash_fp32_fused_euler_step', 'flash_fp32_fused_control_drive'] + ['flash_mjwarp_'+c+'_fp32' for c in COMPONENTS]
        subprocess.run(['cmake', '--build', str(folder), '--parallel', str(jobs), '--target', *targets], check=True)
    return dict(tasks=rows, cuda_directory=folders[0], cuda_directories=folders,
                warp_dependency=False, physics_integrations=0, scene_backend='CPU MuJoCo '+rows[0]['identity']['mujoco'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-root', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', default=['all'])
    parser.add_argument('--skip-build', action='store_true')
    parser.add_argument('--release', action='store_true')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--arch', type=int, default=80)
    parser.add_argument('--mathdx-root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    report = prepare_project(args.build_root, tasks=args.tasks,
        modes=('release',) if args.release else ('dev',), skip_build=args.skip_build,
        jobs=args.jobs, arch=args.arch, mathdx_root=args.mathdx_root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(prepared=len(report['tasks']), physics_integrations=0, warp_dependency=False)), flush=True)


if __name__ == '__main__':
    main()
