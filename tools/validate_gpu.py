"""Stable CPU/GPU validation entry. No compilation or execution on import."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    # Load standalone commands without importing unrelated reference qualifiers.
    standalone={'fp64-motion':'fp64_motion','cuda-runtime':'cuda_runtime','cuda-model':'cuda_model',
                'cuda-complete':'cuda_complete','cuda-functions':'cuda_functions',
                'cuda-lifecycle':'cuda_lifecycle',
                'mjwarp-convex-regression':'mjwarp_convex_regression'}
    if len(sys.argv)>1 and sys.argv[1] in standalone:
        from importlib import import_module
        module=import_module('validation.'+standalone[sys.argv[1]])
        parser=argparse.ArgumentParser(description=module.__doc__)
        module.add_arguments(parser)
        return module.main(parser.parse_args(sys.argv[2:]))
    parser = argparse.ArgumentParser(description=__doc__, epilog=
        'Standalone commands: fp64-motion, cuda-runtime, cuda-model, cuda-functions, '
        'cuda-lifecycle, cuda-complete, mjwarp-convex-regression. '
        'Use COMMAND --help for their arguments.')
    commands = parser.add_subparsers(dest='command', required=True)
    from validation.mjwarp_reference import add_arguments
    add_arguments(commands.add_parser('mjwarp-reference', help='check FP32 adapter against raw upstream MJWarp'))
    from validation.mjwarp_stages import add_arguments as add_stage_arguments
    add_stage_arguments(commands.add_parser('mjwarp-stages', help='profile resident FP32 stages and freeze Newton inputs'))
    from validation.mjwarp_hessian import add_arguments as add_hessian_arguments
    add_hessian_arguments(commands.add_parser('mjwarp-hessian', help='replay the local handwritten FP32 Hessian leaf'))
    from validation.mjwarp_newton import add_arguments as add_newton_arguments
    add_newton_arguments(commands.add_parser('mjwarp-newton', help='replay Newton with an explicit CUDA Hessian'))
    from validation.mjwarp_direction import add_arguments as add_direction_arguments
    add_direction_arguments(commands.add_parser('mjwarp-direction', help='replay handwritten gradient and direction'))
    from validation.mjwarp_constraint import add_arguments as add_constraint_arguments
    add_constraint_arguments(commands.add_parser('mjwarp-constraint', help='replay handwritten constraint update'))
    from validation.mjwarp_linesearch import add_arguments as add_linesearch_arguments
    add_linesearch_arguments(commands.add_parser('mjwarp-linesearch', help='replay handwritten Newton line search'))
    from validation.mjwarp_solver_state import add_arguments as add_solver_state_arguments
    add_solver_state_arguments(commands.add_parser('mjwarp-solver-state', help='replay Newton initialization and stopping'))
    from validation.mjwarp_mass_solve import add_arguments as add_mass_arguments
    add_mass_arguments(commands.add_parser('mjwarp-mass-solve', help='replay handwritten mass backsubstitution'))
    from validation.mjwarp_acceleration import add_arguments as add_acceleration_arguments
    add_acceleration_arguments(commands.add_parser('mjwarp-acceleration', help='replay complete handwritten acceleration'))
    from validation.mjwarp_mass_factor import add_arguments as add_factor_arguments
    add_factor_arguments(commands.add_parser('mjwarp-mass-factor', help='qualify mass factorization then fused factor/solve'))
    from validation.mjwarp_mass_matrix import add_arguments as add_matrix_arguments
    add_matrix_arguments(commands.add_parser('mjwarp-mass-matrix', help='qualify CRB and CSR mass assembly on frozen inputs'))
    from validation.mjwarp_motion import add_arguments as add_motion_arguments
    add_motion_arguments(commands.add_parser('mjwarp-motion', help='qualify com_vel, bias RNE, then their composition'))
    from validation.mjwarp_postconstraint import add_arguments as add_post_arguments
    add_post_arguments(commands.add_parser('mjwarp-postconstraint', help='qualify postconstraint RNE on frozen acceleration and external wrenches'))
    from validation.mjwarp_passive import add_arguments as add_passive_arguments
    add_passive_arguments(commands.add_parser('mjwarp-passive', help='qualify joint spring/damping, passive-force sum, and composition'))
    from validation.mjwarp_actuation import add_arguments as add_actuation_arguments
    add_actuation_arguments(commands.add_parser('mjwarp-actuation', help='qualify stateless actuator force, projection, joint limits, and composition'))
    from validation.mjwarp_euler import add_arguments as add_euler_arguments
    add_euler_arguments(commands.add_parser('mjwarp-euler', help='qualify Euler damping preparation and state advancement on frozen N=2 inputs'))
    from validation.mjwarp_transmission import add_arguments as add_transmission_arguments
    add_transmission_arguments(commands.add_parser('mjwarp-transmission', help='qualify scalar transmission, generic CSR velocity and the actuation chain'))
    from validation.mjwarp_convex_collision import add_arguments as add_convex_collision_arguments
    add_convex_collision_arguments(commands.add_parser('mjwarp-convex-collision', help='qualify pooled production convex CCD on frozen N=2 contacts'))
    from validation.mjwarp_box_manifold import add_arguments as add_manifold_arguments
    add_manifold_arguments(commands.add_parser('mjwarp-box-manifold', help='qualify box-only multicontact on frozen original EPA faces'))
    from validation.mjwarp_convex_ccd import add_arguments as add_convex_ccd_arguments
    add_convex_ccd_arguments(commands.add_parser('mjwarp-convex-ccd', help='qualify rigid shape handling and EPA on frozen N=2 requests'))
    from validation.mjwarp_convex_gjk import add_arguments as add_convex_gjk_arguments
    add_convex_gjk_arguments(commands.add_parser('mjwarp-convex-gjk', help='qualify frozen GJK loop, support cache and cutoff contracts'))
    from validation.mjwarp_convex_simplex import add_arguments as add_convex_simplex_arguments
    add_convex_simplex_arguments(commands.add_parser('mjwarp-convex-simplex', help='qualify reachable ordered GJK simplex projection against independent convex QP'))
    from validation.mjwarp_convex_support import add_arguments as add_convex_support_arguments
    add_convex_support_arguments(commands.add_parser('mjwarp-convex-support', help='qualify rigid support building blocks without claiming production convex replacement'))
    from validation.mjwarp_box import add_arguments as add_box_arguments
    add_box_arguments(commands.add_parser('mjwarp-box', help='qualify capsule/box manifolds and independent geometric residuals on N=2 inputs'))
    from validation.mjwarp_plane import add_arguments as add_plane_arguments
    add_plane_arguments(commands.add_parser('mjwarp-plane', help='qualify plane manifolds and contact material/publication on frozen N=2 inputs'))
    from validation.mjwarp_broadphase import add_arguments as add_broadphase_arguments
    add_broadphase_arguments(commands.add_parser('mjwarp-broadphase', help='qualify NXN candidates and bounding predicates on frozen N=2 inputs'))
    from validation.mjwarp_constraint_build import add_arguments as add_constraint_build_arguments
    add_constraint_build_arguments(commands.add_parser('mjwarp-constraint-build', help='qualify dense rigid constraint row construction on frozen N=2 inputs'))
    from validation.mjwarp_external_wrench import add_arguments as add_external_wrench_arguments
    add_external_wrench_arguments(commands.add_parser('mjwarp-external-wrench', help='qualify applied/contact postconstraint wrenches on frozen N=2 inputs'))
    from validation.mjwarp_fluid import add_arguments as add_fluid_arguments
    add_fluid_arguments(commands.add_parser('mjwarp-fluid', help='qualify inertia-box wrench, projection and passive composition on frozen N=2 inputs'))
    from validation.mjwarp_scene_inputs import add_arguments as add_scene_input_arguments
    add_scene_input_arguments(commands.add_parser('mjwarp-scene-inputs',help='export original CPU2.3.7 fixed fixture parameters and pose oracles offline'))
    from validation.mjwarp_complete import add_arguments as add_complete_arguments
    add_complete_arguments(commands.add_parser('mjwarp-complete',help='prepare/audit and compare complete CUDA with original MJWarp across shipped tasks'))
    from validation.mjwarp_endpoint_report import add_arguments as add_endpoint_report_arguments
    add_endpoint_report_arguments(commands.add_parser('mjwarp-endpoint-report', help='analyze retained endpoints on CPU, keeping strict and scoped conclusions separate'))
    from validation.mjwarp_public_forward import add_arguments as add_public_forward_arguments
    add_public_forward_arguments(commands.add_parser('mjwarp-public-forward',help='qualify complete public refresh and no-upstream numerical Graph ownership'))
    from validation.mjwarp_publication import add_arguments as add_publication_arguments
    add_publication_arguments(commands.add_parser('mjwarp-publication',help='qualify time, overflow and public warmstart publication on N=2 inputs'))
    from validation.mjwarp_queries import add_arguments as add_queries_arguments
    add_queries_arguments(commands.add_parser('mjwarp-queries', help='qualify rigid Jacobians, force/torque sensors and gravity on frozen N=2 inputs'))
    from validation.mjwarp_camlight import add_arguments as add_camlight_arguments
    add_camlight_arguments(commands.add_parser('mjwarp-camlight', help='qualify FIXED camera/light world frames on frozen N=2 inputs'))
    from validation.mjwarp_com_pos import add_arguments as add_com_pos_arguments
    add_com_pos_arguments(commands.add_parser('mjwarp-com-pos', help='qualify subtree CoM, spatial inertia and motion axes on frozen N=2 inputs'))
    from validation.mjwarp_kinematics import add_arguments as add_kinematics_arguments
    add_kinematics_arguments(commands.add_parser('mjwarp-kinematics', help='qualify rigid body, inertial, geom and site kinematics on frozen N=2 inputs'))
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
    run.add_argument('--precision', choices=('fp64', 'fp32'), default='fp64',
                     help='process scalar policy; GPU products must already exist')
    run.add_argument('--physics-engine', choices=('auto', 'native', 'mjwarp'), default='auto',
                     help='GPU physical engine; MJWarp requires its isolated environment and FP32')
    run.add_argument('--mjwarp-collision', choices=('default', 'primitive'), default='default')
    run.add_argument('--mjwarp-nconmax', type=int)
    run.add_argument('--mjwarp-njmax', type=int)
    run.add_argument('--execution', choices=('fused', 'components'), default='fused',
                     help='single-substep fusion or the public component graph')
    run.add_argument('--fast-compile', action='store_true', help='select existing dev products; does not compile')
    run.add_argument('--device-lto', action='store_true', help='select existing LTO products; does not link')
    run.add_argument('--single-solve', action='store_true',
                     help='explicit native FP32 fused scheduling candidate; not the default')
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
    compare.add_argument('--precision', choices=('fp64', 'fp32'), default='fp64')
    compare.add_argument('--model', type=Path, help='matching model.mjb; required for FP32 sensitivity')
    compare.add_argument('--output', type=Path, required=True)

    summary = commands.add_parser('summarize', help='summarize new matching CPU/GPU receipts')
    summary.add_argument('--cpu', type=Path, required=True)
    summary.add_argument('--gpu', type=Path, required=True)
    summary.add_argument('--output', type=Path, required=True)
    summary.add_argument('--substeps', type=int, help='explicit substeps for legacy receipts without this metadata')

    paired = commands.add_parser('summarize-precision', help='compare saved FP64/FP32 timings without physics')
    paired.add_argument('--fp64', type=Path, required=True)
    paired.add_argument('--fp32', type=Path, required=True)
    paired.add_argument('--output', type=Path, required=True)
    paired.add_argument('--substeps', type=int, help='explicit substeps for legacy receipts without this metadata')
    args = parser.parse_args()
    if args.command == 'mjwarp-endpoint-report':
        from validation.mjwarp_endpoint_report import main as report_endpoints
        return report_endpoints(args)
    if args.command == 'mjwarp-reference':
        from validation.mjwarp_reference import main as check_upstream
        return check_upstream(args=args)
    if args.command == 'mjwarp-stages':
        from validation.mjwarp_stages import main as profile_stages
        return profile_stages(args=args)
    if args.command == 'mjwarp-hessian':
        from validation.mjwarp_hessian import main as replay_hessian
        return replay_hessian(args=args)
    if args.command == 'mjwarp-newton':
        from validation.mjwarp_newton import main as replay_newton
        return replay_newton(args=args)
    if args.command == 'mjwarp-direction':
        from validation.mjwarp_direction import main as replay_direction
        return replay_direction(args=args)
    if args.command == 'mjwarp-constraint':
        from validation.mjwarp_constraint import main as replay_constraint
        return replay_constraint(args=args)
    if args.command == 'mjwarp-linesearch':
        from validation.mjwarp_linesearch import main as replay_linesearch
        return replay_linesearch(args=args)
    if args.command == 'mjwarp-solver-state':
        from validation.mjwarp_solver_state import main as replay_solver_state
        return replay_solver_state(args=args)
    if args.command == 'mjwarp-mass-solve':
        from validation.mjwarp_mass_solve import main as replay_mass
        return replay_mass(args=args)
    if args.command == 'mjwarp-acceleration':
        from validation.mjwarp_acceleration import main as replay_acceleration
        return replay_acceleration(args=args)
    if args.command == 'mjwarp-mass-factor':
        from validation.mjwarp_mass_factor import main as replay_factor
        return replay_factor(args=args)
    if args.command == 'mjwarp-mass-matrix':
        from validation.mjwarp_mass_matrix import main as replay_matrix
        return replay_matrix(args=args)
    if args.command == 'mjwarp-motion':
        from validation.mjwarp_motion import main as replay_motion
        return replay_motion(args=args)
    if args.command == 'mjwarp-postconstraint':
        from validation.mjwarp_postconstraint import main as replay_post
        return replay_post(args=args)
    if args.command == 'mjwarp-passive':
        from validation.mjwarp_passive import main as replay_passive
        return replay_passive(args=args)
    if args.command == 'mjwarp-actuation':
        from validation.mjwarp_actuation import main as replay_actuation
        return replay_actuation(args=args)
    if args.command == 'mjwarp-euler':
        from validation.mjwarp_euler import main as replay_euler
        return replay_euler(args=args)
    if args.command == 'mjwarp-transmission':
        from validation.mjwarp_transmission import main as replay_transmission
        return replay_transmission(args=args)
    if args.command == 'mjwarp-convex-collision':
        from validation.mjwarp_convex_collision import main as replay_convex_collision
        return replay_convex_collision(args=args)
    if args.command == 'mjwarp-box-manifold':
        from validation.mjwarp_box_manifold import main as replay_box_manifold
        return replay_box_manifold(args=args)
    if args.command == 'mjwarp-convex-ccd':
        from validation.mjwarp_convex_ccd import main as replay_convex_ccd
        return replay_convex_ccd(args=args)
    if args.command == 'mjwarp-convex-gjk':
        from validation.mjwarp_convex_gjk import main as replay_convex_gjk
        return replay_convex_gjk(args=args)
    if args.command == 'mjwarp-convex-simplex':
        from validation.mjwarp_convex_simplex import main as replay_convex_simplex
        return replay_convex_simplex(args=args)
    if args.command == 'mjwarp-convex-support':
        from validation.mjwarp_convex_support import main as replay_convex_support
        return replay_convex_support(args=args)
    if args.command == 'mjwarp-box':
        from validation.mjwarp_box import main as replay_box
        return replay_box(args=args)
    if args.command == 'mjwarp-plane':
        from validation.mjwarp_plane import main as replay_plane
        return replay_plane(args=args)
    if args.command == 'mjwarp-broadphase':
        from validation.mjwarp_broadphase import main as replay_broadphase
        return replay_broadphase(args=args)
    if args.command == 'mjwarp-constraint-build':
        from validation.mjwarp_constraint_build import main as replay_constraint_build
        return replay_constraint_build(args=args)
    if args.command == 'mjwarp-external-wrench':
        from validation.mjwarp_external_wrench import main as replay_external_wrench
        return replay_external_wrench(args=args)
    if args.command == 'mjwarp-fluid':
        from validation.mjwarp_fluid import main as replay_fluid
        return replay_fluid(args=args)
    if args.command == 'mjwarp-scene-inputs':
        from validation.mjwarp_scene_inputs import main as export_scene_inputs
        return export_scene_inputs(args=args)
    if args.command == 'mjwarp-complete':
        from validation.mjwarp_complete import main as replay_complete
        return replay_complete(args=args)
    if args.command == 'mjwarp-public-forward':
        from validation.mjwarp_public_forward import main as replay_public_forward
        return replay_public_forward(args=args)
    if args.command == 'mjwarp-publication':
        from validation.mjwarp_publication import main as replay_publication
        return replay_publication(args=args)
    if args.command == 'mjwarp-queries':
        from validation.mjwarp_queries import main as replay_queries
        return replay_queries(args=args)
    if args.command == 'mjwarp-camlight':
        from validation.mjwarp_camlight import main as replay_camlight
        return replay_camlight(args=args)
    if args.command == 'mjwarp-com-pos':
        from validation.mjwarp_com_pos import main as replay_com_pos
        return replay_com_pos(args=args)
    if args.command == 'mjwarp-kinematics':
        from validation.mjwarp_kinematics import main as replay_kinematics
        return replay_kinematics(args=args)
    if args.command == 'compare':
        from validation.common import save_json
        from validation.offline import compare_saved
        result = compare_saved(args)
        save_json(args.output, result)
        print(json.dumps(dict(passed=result['passed'], checks=len(result['checks']))))
        return 0 if result['passed'] else 2
    if args.command == 'summarize':
        from validation.summary import summarize
        result = summarize(args.cpu, args.gpu, args.output, substeps=args.substeps)
        print(json.dumps({key: value for key, value in result.items() if key != 'rows'}))
        return 0
    if args.command == 'summarize-precision':
        from validation.summary import summarize_precision
        result = summarize_precision(args.fp64, args.fp32, args.output, substeps=args.substeps)
        print(json.dumps(result['aggregate']))
        return 0
    if not 1 <= args.worlds <= 128 or args.warmup < 0 or args.steps < 1:
        parser.error('worlds must be 1..128, warmup >= 0 and steps > 0')
    if args.diagnose_n1 and args.worlds != 2:
        parser.error('--diagnose-n1 requires the primary run to use N=2')
    if args.execution == 'components' and args.case == 'varying_cache_miss':
        parser.error('forced cache-miss injection requires the fused execution path')
    if args.backend == 'gpu':
        from libero.gpu.host.backends.selection import resolve_engine
        try:
            args.physics_engine = resolve_engine(args.physics_engine, args.precision)
        except ValueError as error:
            parser.error(str(error))
    if args.physics_engine == 'mjwarp' and (args.backend != 'gpu' or args.precision != 'fp32' or args.case == 'varying_cache_miss'):
        parser.error('MJWarp requires --backend gpu --precision fp32 and has no native prefix-cache injection')
    if args.single_solve and (args.backend != 'gpu' or args.physics_engine != 'native' or
                             args.precision != 'fp32' or args.execution != 'fused'):
        parser.error('--single-solve requires native FP32 fused execution')
    if args.backend == 'gpu' and not args.record_reference and not (args.cpu_reference and args.baseline_reference):
        parser.error('GPU validation requires --cpu-reference and --baseline-reference, or explicit --record-reference')
    if args.backend == 'cpu':
        if args.precision != 'fp64':
            parser.error('the independent MuJoCo CPU reference uses FP64')
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'LP_NUM_THREADS'):
            os.environ[name] = '1'
    else:
        import libero.gpu.precision as scalar
        scalar.select(args.precision)
        from libero.gpu.host.build.aot import project_root
        args.build_root = project_root(args.build_root)
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
