"""Explicit offline MJWarp preparation and immutable load-only receipts."""
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_files(*, include_hessian=False, include_direction=False, include_constraint=False, include_linesearch=False, include_state=False, include_mass=False, include_force=False, include_factor=False, include_matrix=False, include_velocity=False, include_bias=False, include_postconstraint=False, include_passive=False, include_actuation=False):
    import mujoco_warp
    from libero.gpu.host.paths import GPU_ROOT
    from libero.gpu.host.build.aot import source_identity
    files = {str(p.relative_to(GPU_ROOT)): digest(p) for p in Path(__file__).parent.glob('*.py')}
    files['project_source_identity'] = source_identity()
    for name in ('data/mjwarp_source_pin.json', 'cuda/control/mjwarp_controller.cu',
                 'host/backends/selection.py', 'host/adapters/environment/batch_env.py'):
        files[name] = digest(GPU_ROOT / name)
    files.update({'upstream/' + str(p.relative_to(Path(mujoco_warp.__file__).parent)): digest(p)
                  for p in Path(mujoco_warp.__file__).parent.rglob('*.py')})
    if include_hessian or include_direction or include_constraint or include_linesearch or include_state:
        for name in ('cuda/newton/mjwarp_hessian.cu', 'cuda/newton/mjwarp_hessian.cuh',
                     'cmake/MjwarpHessian.cmake', 'host/runtime/newton/mjwarp_hessian.py'):
            files[name] = digest(GPU_ROOT / name)
    if include_direction or include_constraint or include_linesearch or include_state:
        for name in ('cuda/newton/mjwarp_direction.cu', 'cuda/newton/mjwarp_direction.cuh',
                     'cmake/MjwarpDirection.cmake', 'host/runtime/newton/mjwarp_direction.py'):
            files[name] = digest(GPU_ROOT / name)
    if include_constraint or include_linesearch or include_state:
        for name in ('cuda/newton/mjwarp_constraint.cu', 'cuda/newton/mjwarp_constraint.cuh',
                     'cmake/MjwarpConstraint.cmake', 'host/runtime/newton/mjwarp_constraint.py'):
            files[name] = digest(GPU_ROOT / name)
    if include_linesearch or include_state:
        for name in ('cuda/newton/mjwarp_linesearch.cu', 'cuda/newton/mjwarp_linesearch.cuh',
                     'cuda/newton/mjwarp_products.cuh', 'cmake/MjwarpLinesearch.cmake', 'host/runtime/newton/mjwarp_linesearch.py'):
            files[name] = digest(GPU_ROOT / name)
    if include_state:
        for name in ('cuda/newton/mjwarp_solver_state.cu', 'cuda/newton/mjwarp_solver_state.cuh',
                     'cmake/MjwarpSolverState.cmake', 'host/runtime/newton/mjwarp_solver_state.py'):
            files[name] = digest(GPU_ROOT / name)
    if include_mass or include_force:
        for name in ('cuda/dynamics/mjwarp_mass_solve.cu','cuda/dynamics/mjwarp_mass_solve.cuh',
                     'cmake/MjwarpMassSolve.cmake','host/runtime/dynamics/mjwarp_mass_solve.py'):
            files[name] = digest(GPU_ROOT / name)
    if include_force:
        for name in ('cuda/dynamics/mjwarp_smooth_force.cu','cuda/dynamics/mjwarp_smooth_force.cuh',
                     'cmake/MjwarpSmoothForce.cmake','host/runtime/dynamics/mjwarp_smooth_force.py'):
            files[name] = digest(GPU_ROOT / name)
    if include_factor:
        for name in ('cuda/dynamics/mjwarp_mass_factor.cu','cuda/dynamics/mjwarp_mass_factor.cuh',
                     'cmake/MjwarpMassFactor.cmake','host/runtime/dynamics/mjwarp_mass_factor.py'):
            files[name] = digest(GPU_ROOT / name)
    if include_matrix:
        for name in ('cuda/dynamics/mjwarp_mass_matrix.cu','cuda/dynamics/mjwarp_mass_matrix.cuh',
                     'cmake/MjwarpMassMatrix.cmake','host/runtime/dynamics/mjwarp_mass_matrix.py'):
            files[name] = digest(GPU_ROOT / name)
    if include_velocity or include_bias or include_postconstraint:
        for name in ('cuda/dynamics/mjwarp_spatial.cuh','host/runtime/dynamics/mjwarp_motion.py'):
            files[name]=digest(GPU_ROOT/name)
    if include_passive or include_actuation:files['host/runtime/dynamics/mjwarp_motion.py']=digest(GPU_ROOT/'host/runtime/dynamics/mjwarp_motion.py')
    for component,enabled in (('velocity',include_velocity),('bias',include_bias),('postconstraint',include_postconstraint),('passive',include_passive),('actuation',include_actuation)):
        if enabled:
            for name in (f'cuda/dynamics/mjwarp_{component}.cu',f'cuda/dynamics/mjwarp_{component}.cuh',
                         f'cmake/Mjwarp{component.title()}.cmake',f'host/runtime/dynamics/mjwarp_{component}.py'):
                files[name]=digest(GPU_ROOT/name)
    return files


def profile_sources(env):
    mass = getattr(env, 'mass_solve_implementation', 'upstream') == 'cuda'
    mode = getattr(env, 'newton_implementation', 'upstream')
    flags = {key: True for key, value in (
        ('include_hessian', mode=='cuda_hessian'), ('include_direction', mode=='cuda_direction'),
        ('include_constraint', mode=='cuda_constraint'), ('include_linesearch', mode=='cuda_linesearch'),
        ('include_state', mode=='cuda_newton'), ('include_mass', mass),
        ('include_force', getattr(env,'acceleration_implementation','upstream')=='cuda'),
        ('include_factor', getattr(env,'mass_factor_implementation','upstream')=='cuda'),
        ('include_matrix', getattr(env,'mass_matrix_implementation','upstream')=='cuda'),
        ('include_velocity', getattr(env,'com_vel_implementation','upstream')=='cuda'),
        ('include_bias', getattr(env,'bias_implementation','upstream')=='cuda'),
        ('include_postconstraint', getattr(env,'postconstraint_implementation','upstream')=='cuda'),
        ('include_passive', getattr(env,'passive_implementation','upstream')=='cuda'),
        ('include_actuation', getattr(env,'actuation_implementation','upstream')=='cuda')) if value}
    return source_files(**flags)


def profile_path(env):
    binding = dict(identity=env.identity, sources=profile_sources(env),
                   controller_sha256=digest(env.library_path))
    key = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    return env.build_root / 'mjwarp' / 'profiles' / (key + '.json')


def require_profile(env):
    from libero.gpu.host.build.compilation import MissingCompiledProduct
    path = profile_path(env)
    if not path.is_file():
        raise MissingCompiledProduct('Prepare this MJWarp profile offline: ' + str(path))
    row = json.loads(path.read_text())
    if (row['identity'] != env.identity or row['sources'] != profile_sources(env) or
            row['controller_sha256'] != digest(env.library_path)):
        raise MissingCompiledProduct('MJWarp profile is stale; prepare it offline: ' + str(path))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-root', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', default=['libero_10:4'])
    parser.add_argument('--worlds', type=int, default=2)
    parser.add_argument('--fast-compile', action='store_true')
    parser.add_argument('--collision', choices=('default', 'primitive'), default='default')
    parser.add_argument('--nconmax', type=int)
    parser.add_argument('--njmax', type=int)
    parser.add_argument('--newton', choices=('upstream', 'cuda_hessian', 'cuda_direction', 'cuda_constraint', 'cuda_linesearch', 'cuda_newton'), default='upstream')
    parser.add_argument('--mass-solve', choices=('upstream', 'cuda'))
    parser.add_argument('--acceleration', choices=('upstream','cuda'), default='upstream')
    parser.add_argument('--mass-factor', choices=('upstream','cuda'), default='upstream')
    parser.add_argument('--mass-matrix', choices=('upstream','cuda'), default='upstream')
    parser.add_argument('--com-vel',choices=('upstream','cuda'),default='upstream')
    parser.add_argument('--bias',choices=('upstream','cuda'),default='upstream')
    parser.add_argument('--postconstraint',choices=('upstream','cuda'),default='upstream')
    parser.add_argument('--passive',choices=('upstream','cuda'),default='upstream')
    parser.add_argument('--actuation',choices=('upstream','cuda'),default='upstream')
    parser.add_argument('--audit-models', action='store_true', help='import all selected models without physics')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    import libero.gpu.precision as scalar
    scalar.select('fp32')
    import warp as wp
    from libero.libero import get_libero_path
    from libero.libero.envs import GpuBatchEnv
    from libero.gpu.host.build.compilation import compilation_scope
    from libero.gpu.host.build.project_paths import project_paths
    from .model import build_scene, capacity_budget
    root = project_paths(args.build_root).base
    tasks = json.loads((root / 'TASK_INSTANCES.json').read_text())['tasks']
    selected = []
    for task in tasks:
        if any(selector in ('all', task['suite'], task['suite'] + ':' + str(task['task_id'])) for selector in args.tasks):
            selected.append(task)
    if not selected:
        raise ValueError('no matching offline tasks')
    rows = []
    for task in selected:
        bddl = str(Path(get_libero_path('bddl_files')) / task['key'])
        if args.audit_models:
            import gc
            try:
                built, model, bank, _, identity = build_scene(root, bddl, 0, dict(
                    use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False,
                    hard_reset=False, ignore_done=True))
                row = dict(task=task['key'], passed=True, dimensions=built.metadata['dimensions'],
                           identity=identity, capacities=capacity_budget(built.native_model))
                del built, model, bank
            except Exception as error:
                row = dict(task=task['key'], passed=False, error=repr(error))
            gc.collect()
        else:
            sources = source_files(include_hessian=args.newton=='cuda_hessian',
                include_direction=args.newton=='cuda_direction', include_constraint=args.newton=='cuda_constraint',
                include_linesearch=args.newton=='cuda_linesearch', include_state=args.newton=='cuda_newton',
                include_mass=args.mass_solve=='cuda', include_force=args.acceleration=='cuda',
                include_factor=args.mass_factor=='cuda',include_matrix=args.mass_matrix=='cuda',
                include_velocity=args.com_vel=='cuda',include_bias=args.bias=='cuda',include_postconstraint=args.postconstraint=='cuda',include_passive=args.passive=='cuda',include_actuation=args.actuation=='cuda')
            with compilation_scope(True) as preparation:
                options = dict(fast_compile=args.fast_compile, collision=args.collision, newton=args.newton,
                               acceleration=args.acceleration,mass_factor=args.mass_factor,mass_matrix=args.mass_matrix,
                               com_vel=args.com_vel,bias=args.bias,postconstraint=args.postconstraint,passive=args.passive,actuation=args.actuation)
                if args.mass_solve is not None:options['mass_solve']=args.mass_solve
                for name in ('nconmax', 'njmax'):
                    if getattr(args, name) is not None:
                        options[name] = getattr(args, name)
                env = GpuBatchEnv(bddl, args.worlds, physics_engine='mjwarp', precision='fp32',
                    build_root=root, allow_compile=True, initialize=False, use_camera_obs=False,
                    ignore_done=True, mjwarp_options=options)
                try:
                    env.reset(seed=0, init_states=env.initial_states[:args.worlds])
                    # Compile every unique operator, including both policy
                    # flags, without capturing 25 duplicate substeps offline.
                    # Runtime captures its full graph using these exact ops.
                    with env.runtime.newton.capture_owners(), wp.ScopedStream(env.stream), \
                            wp.ScopedCapture(stream=env.stream):
                        env.runtime.enqueue_substep(True)
                        env.runtime.enqueue_substep(False)
                    wp.synchronize_stream(env.stream)
                    if any(env.runtime.integration_count.numpy()):
                        raise RuntimeError('offline MJWarp preparation advanced physics')
                    if profile_sources(env) != sources:
                        raise RuntimeError('source changed during offline MJWarp preparation')
                    row = dict(identity=env.identity, sources=sources,
                               controller_sha256=digest(env.library_path),
                               compilation_events=env.build_report['events'] + preparation['events'],
                               actual_integration_count=env.runtime.integration_count.numpy().tolist(), passed=True)
                    path = profile_path(env)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.exists():
                        existing = json.loads(path.read_text())
                        for key in ('identity', 'sources', 'controller_sha256'):
                            if existing[key] != row[key]:
                                raise RuntimeError('MJWarp profile binding collision: ' + str(path))
                    else:
                        path.write_text(json.dumps(row, indent=2) + '\n')
                    row = dict(task=task['key'], profile=str(path), passed=True,
                               compilation_events=len(row['compilation_events']))
                finally:
                    env.close()
        rows.append(row)
        print(json.dumps(row), flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(dict(passed=all(r['passed'] for r in rows), rows=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
