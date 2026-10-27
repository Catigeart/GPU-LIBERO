"""GPU-event stage profile and optional same-model frozen Newton problem.

The instrumented graph uses the production PhysicsStages and controller. It
records existing resident inputs, without a second model compiler or CPU
physics. Event-instrumented timings are diagnostic, not ordinary throughput.
"""
import argparse
import hashlib
import json
from pathlib import Path
from libero.gpu.host.build.task_catalog import load_catalog

import numpy as np


def add_arguments(parser):
    parser.add_argument('--build-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', default='libero_10:4')
    parser.add_argument('--worlds', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--freeze-newton', action='store_true')
    return parser


def main(argv=None, *, args=None):
    if args is None:
        args = add_arguments(argparse.ArgumentParser(description=__doc__)).parse_args(argv)
    if not 1 <= args.worlds <= 128 or args.warmup < 0 or args.steps < 1:
        raise ValueError('worlds must be 1..128, warmup >= 0 and steps > 0')
    from libero.gpu import precision
    precision.select('fp32')
    import warp as wp
    from libero.libero import get_libero_path
    from libero.libero.envs import GpuBatchEnv
    from libero.gpu.host.backends.mjwarp.stages import PhysicsStages
    from libero.gpu.host.backends.mjwarp.build import source_files, profile_path
    from libero.gpu.host.build.compilation import compilation_scope, install_warp_guard
    from validation.common import actions, input_record, save_json
    from validation.runner import gpu_load_snapshot

    tasks = load_catalog(args.build_root)
    task = next(t for t in tasks if t['suite'] + ':' + str(t['task_id']) == args.task)
    bank = np.load(Path(task['directory']) / 'initial_states.npy', allow_pickle=False)
    args.output.mkdir(parents=True, exist_ok=False)
    install_warp_guard()
    sources = source_files()
    tool_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with compilation_scope(False) as scope:
        env = GpuBatchEnv(str(Path(get_libero_path('bddl_files')) / task['key']), args.worlds,
            physics_engine='mjwarp', precision='fp32', build_root=args.build_root,
            initialize=False, allow_compile=False, use_camera_obs=False, ignore_done=True,
            mjwarp_options=dict(fast_compile=True))
        try:
            env.reset(seed=0, init_states=bank[:args.worlds])
            stream, d = env.stream, env.data
            names = ('step1', 'position', 'velocity', 'controller', 'step2',
                     'actuation', 'acceleration', 'newton', 'integrate')
            # Event owners are created before capture. Each of the 25 physical
            # substeps has separate events; all control steps reuse this graph.
            events = {name: [(wp.Event(device=stream.device, enable_timing=True),
                              wp.Event(device=stream.device, enable_timing=True))
                             for _ in range(env.substeps)] for name in names}
            slots = dict.fromkeys(names, 0)
            with wp.ScopedStream(stream):
                iteration_trace = wp.empty((env.substeps, args.worlds), dtype=int, device=stream.device)
                row_trace = wp.empty_like(iteration_trace)
            saved = {}
            if args.freeze_newton:
                # A single boundary (last physical substep) is copied per graph
                # replay. Copies lie outside the Newton event interval.
                fields = {'input.' + name: getattr(d, name) for name in
                          ('qpos', 'qvel', 'ctrl', 'nefc', 'ne', 'nf', 'M', 'qacc_warmstart', 'qacc_smooth',
                           'qfrc_smooth', 'qacc', 'qfrc_constraint', 'nacon')}
                fields.update({'input.efc.' + name: getattr(d.efc, name) for name in
                               ('J', 'D', 'aref', 'frictionloss', 'type', 'id', 'pos',
                                'margin', 'force', 'state', 'jtdaj_adr', 'jtdaj_nrow', 'jtdaj_nblock')})
                fields.update({'input.contact.' + name: getattr(d.contact, name) for name in
                               ('worldid', 'dim', 'friction', 'dist', 'includemargin', 'efc_address')})
                fields.update({'output.' + name: getattr(d, name) for name in
                               ('qacc', 'qfrc_constraint', 'solver_niter')})
                fields.update({'output.efc.' + name: getattr(d.efc, name) for name in ('force', 'state')})
                with wp.ScopedStream(stream):
                    saved = {name: wp.empty_like(array) for name, array in fields.items()}

            def measure(name, call):
                slot = slots[name]
                slots[name] += 1
                start, end = events[name][slot]
                if name == 'newton' and slot == env.substeps - 1 and saved:
                    for key, target in saved.items():
                        if key.startswith('input.'):
                            wp.copy(target, fields[key], stream=stream)
                stream.record_event(start)
                call()
                stream.record_event(end)
                if name == 'newton':
                    wp.copy(iteration_trace, d.solver_niter, dest_offset=slot * args.worlds,
                            count=args.worlds, stream=stream)
                    wp.copy(row_trace, d.nefc, dest_offset=slot * args.worlds,
                            count=args.worlds, stream=stream)
                if name == 'newton' and slot == env.substeps - 1 and saved:
                    for key, target in saved.items():
                        if key.startswith('output.'):
                            wp.copy(target, fields[key], stream=stream)

            # Separate type; the production reference type is never monkey
            # patched. Every default method still consumes this exact Model/Data.
            class MeasuredStages(PhysicsStages):
                pass

            for name in names:
                if name != 'controller':
                    method = getattr(PhysicsStages, name)
                    def wrapped(self, method=method, name=name):
                        measure(name, lambda: method(self))
                    setattr(MeasuredStages, name, wrapped)
            env.runtime.physics = MeasuredStages(env.model, d)
            controller = env.runtime.controller
            enqueue = controller.enqueue
            controller.enqueue = lambda policy: measure('controller', lambda: enqueue(policy))
            env.runtime.prepare_control_graph()
            if slots != dict.fromkeys(names, env.substeps):
                raise RuntimeError('the graph did not capture exactly one invocation per stage/substep')
            for k in range(args.warmup):
                env.step(actions(k, args.worlds))
            counters = ('integration_count', 'newton_attempts', 'position_math', 'reset_events')
            before = {name: getattr(env.runtime, name).numpy() for name in counters}
            load_before = gpu_load_snapshot()
            rows = {name: [] for name in names}
            iterations, constraint_rows = [], []
            for k in range(args.warmup, args.warmup + args.steps):
                env.step(actions(k, args.worlds))
                for name in names:
                    rows[name].append(sum(wp.get_event_elapsed_time(start, end, synchronize=False)
                                          for start, end in events[name]))
                iterations.append(iteration_trace.numpy())
                constraint_rows.append(row_trace.numpy())
            after = {name: getattr(env.runtime, name).numpy() for name in counters}
            work = {name: (after[name] - before[name]).tolist() for name in counters}
            expected = [env.substeps * args.steps] * args.worlds
            passed = (all(work[n] == expected for n in counters[:3]) and
                      work['reset_events'] == [0] * args.worlds and not scope['events'] and
                      all(np.isfinite(getattr(d, n).numpy()).all() for n in ('qpos', 'qvel', 'qacc')))
            result = dict(passed=passed, identity=env.identity, sources=sources,
                tool_sha256=tool_digest,
                offline_profile=str(profile_path(env)),
                input=input_record(task, bank, worlds=args.worlds, warmup=args.warmup,
                                   steps=args.steps, seed=0, case='varying'),
                counters=work, runtime_compilations=list(scope['events']),
                stages={name: dict(control_ms=value, mean_ms=float(np.mean(value)),
                                   std_ms=float(np.std(value))) for name, value in rows.items()},
                shared_device_policy='concurrent workloads permitted; event timings include contention',
                gpu_load_before=load_before, gpu_load_after=gpu_load_snapshot(),
                scope='GPU event stage times across all worlds. step1/step2 are inclusive parents; do not add them to children. Diagnostic event/copy overhead is not normal throughput.')
            iterations, constraint_rows = np.asarray(iterations), np.asarray(constraint_rows)
            np.savez(args.output / 'newton-work.npz', iterations=iterations, rows=constraint_rows)
            from mujoco_warp._src import solver
            result['newton_work'] = dict(
                solves_per_world=env.substeps * args.steps,
                iterations_total_per_world=iterations.sum(axis=(0, 1)).tolist(),
                iterations_per_solve_mean=float(iterations.mean()),
                iterations_per_solve_max=int(iterations.max()),
                constraint_rows_mean=float(constraint_rows.mean()),
                constraint_rows_max=int(constraint_rows.max()),
                incremental=bool(solver._use_incremental(env.model)),
                trace='newton-work.npz')
            if saved:
                frozen = {name: array.numpy() for name, array in saved.items()}
                for name in ('M_hinit_i', 'M_colind', 'dof_parentid'):
                    frozen['model.' + name] = getattr(env.model, name).numpy()
                frozen['model.tolerance'] = env.model.opt.tolerance.numpy()
                frozen['model.meaninertia'] = env.model.stat.meaninertia.numpy()
                frozen['model.impratio_invsqrt'] = env.model.opt.impratio_invsqrt.numpy()
                np.savez(args.output / 'newton.npz', **frozen)
                result['frozen_newton'] = dict(schema='flash-mjwarp-newton-boundary-v1',
                    file='newton.npz', boundary='last Newton of final measured control step',
                    layout=dict(worlds=args.worlds, nv=env.model.nv, nv_pad=env.model.nv_pad,
                                rows=d.njmax, rows_pad=d.njmax_pad, pooled_contacts=d.naconmax,
                                mass='MJWarp CSR, indexed by M_hinit_i/M_colind', scalar='float32'),
                    sha256=hashlib.sha256((args.output / 'newton.npz').read_bytes()).hexdigest(),
                    model_source='exact compiled Model and resident Data of the upstream run',
                    solver=dict(iterations=int(env.model.opt.iterations),
                                ls_iterations=int(env.model.opt.ls_iterations),
                                cone=int(env.model.opt.cone),
                                disableflags=int(env.model.opt.disableflags)),
                    aliasing='input/output snapshots are separately owned device copies')
            if sources != source_files() or tool_digest != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
                raise RuntimeError('physics sources changed during profiling')
            save_json(args.output / 'summary.json', result)
            print(json.dumps(dict(passed=passed, stages={n: v['mean_ms'] for n, v in result['stages'].items()},
                                  counters=work, runtime_compilations=len(scope['events']))), flush=True)
            return 0 if passed else 2
        finally:
            env.close()
