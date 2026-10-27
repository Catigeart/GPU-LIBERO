"""Compare the adapter bridge, substeps and trajectory with raw MJWarp.

Use --prepare once offline. Ordinary checks forbid compilation. These checks
do not time physics and do not change any production engine/controller math.
"""
import argparse
import json
from pathlib import Path
from libero.gpu.host.build.task_catalog import load_catalog

import numpy as np


def add_arguments(parser):
    parser.add_argument('--build-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--worlds', type=int, default=2)
    parser.add_argument('--task', default='libero_10:4')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--raw-repeat', action='store_true', help='diagnose raw upstream self-variation')
    return parser


def main(argv=None, *, args=None):
    if args is None:
        parser = add_arguments(argparse.ArgumentParser(description=__doc__))
        args = parser.parse_args(argv)
    if not 1 <= args.worlds <= 128 or args.warmup < 0 or args.steps < 1:
        raise ValueError('worlds must be 1..128, warmup >= 0 and steps > 0')
    import libero.gpu.precision as scalar
    scalar.select('fp32')
    import warp as wp
    import mujoco_warp as mjw
    from libero.libero import get_libero_path
    from libero.libero.envs import GpuBatchEnv
    from libero.gpu.host.build.compilation import compilation_scope, install_warp_guard
    from validation.common import actions
    tasks = load_catalog(args.build_root)
    task = next(t for t in tasks if t['suite'] + ':' + str(t['task_id']) == args.task)
    bank = np.load(Path(task['directory']) / 'initial_states.npy')
    args.output.mkdir(parents=True, exist_ok=False)
    install_warp_guard()
    with compilation_scope(args.prepare) as scope:
        env = GpuBatchEnv(str(Path(get_libero_path('bddl_files')) / task['key']), args.worlds,
            physics_engine='mjwarp', precision='fp32', build_root=args.build_root,
            initialize=False, allow_compile=False, use_camera_obs=False, ignore_done=True,
            mjwarp_options=dict(fast_compile=True))
        try:
            env.reset(seed=0, init_states=bank[:args.worlds])
            c = env.runtime.controller
            with wp.ScopedStream(env.stream):
                c.gather()
            wp.synchronize_stream(env.stream)
            d = env.data
            robot = env.built.metadata['robot_metadata'][0]
            qi, vi, site = robot['qpos_indices'], robot['qvel_indices'], robot['eef_site_id']
            jp, jr, qv = c.jp.numpy(), c.jr.numpy(), d.qvel.numpy()
            matrix = np.zeros((args.worlds, env.model.nv, env.model.nv), np.float32)
            idx = c.mass_indices_host
            matrix[:, idx >= 0] = d.M.numpy()[:, idx[idx >= 0]]
            expected = dict(pm=matrix[:, vi][:, :, vi].reshape(args.worlds, -1),
                pj=np.concatenate((jp[:, :, vi], jr[:, :, vi]), axis=1).reshape(args.worlds, -1),
                pq=d.qpos.numpy()[:, qi], pv=qv[:, vi], bias=d.qfrc_bias.numpy()[:, vi],
                pp=d.site_xpos.numpy()[:, site], pr=d.site_xmat.numpy()[:, site].reshape(args.worlds, -1),
                pe=np.concatenate((np.einsum('wiv,wv->wi', jp.astype(np.float64), qv),
                                   np.einsum('wiv,wv->wi', jr.astype(np.float64), qv)), axis=1))
            bridge = []
            for name, value in expected.items():
                actual = c.buffers[name].numpy()
                error = float(np.max(abs(actual - value)))
                limit = 64 * np.finfo(np.float32).eps * max(1., float(np.max(abs(value)))) if name == 'pe' else 0.
                bridge.append(dict(field=name, max_error=error, limit=float(limit), passed=bool(error <= limit)))
            cases = []
            names = ('qpos', 'qvel', 'qacc', 'qacc_warmstart', 'ctrl', 'time',
                     'qfrc_smooth', 'qfrc_constraint')
            inputs = names + ('qfrc_applied', 'xfrc_applied')
            for stage in ('initial', 'after_warmup'):
                if stage == 'after_warmup' and not args.prepare:
                    env.reset(seed=0, init_states=bank[:args.worlds])
                    env.runtime._control_graph = None
                    for k in range(args.warmup):
                        env.step(actions(k, args.worlds))
                with wp.ScopedStream(env.stream):
                    raw = mjw.make_data(env.built.native_model, nworld=args.worlds,
                                        nccdmax=env.capacities['nconmax'], **env.capacities)
                    for name in inputs:
                        wp.copy(getattr(raw, name), getattr(env.data, name))
                    ctrl = wp.empty_like(env.data.ctrl)
                    c.buffers['action'].assign(actions(0, args.worlds).astype(np.float32))
                    original = c.enqueue
                    def record(policy_step):
                        original(policy_step)
                        wp.copy(ctrl, env.data.ctrl, stream=env.stream)
                    c.enqueue = record
                    try:
                        with wp.ScopedCapture(stream=env.stream) as capture:
                            env.runtime.enqueue_substep(True)
                        adapted_graph = capture.graph
                    finally:
                        c.enqueue = original
                    with wp.ScopedCapture(stream=env.stream) as capture:
                        mjw.step1(env.model, raw)
                        wp.copy(raw.ctrl, ctrl)
                        mjw.step2(env.model, raw)
                    raw_graph = capture.graph
                    if not args.prepare:
                        wp.capture_launch(adapted_graph, stream=env.stream)
                        wp.capture_launch(raw_graph, stream=env.stream)
                wp.synchronize_stream(env.stream)
                if not args.prepare:
                    checks, saved = [], {}
                    for name in names:
                        value, ref = getattr(env.data, name).numpy(), getattr(raw, name).numpy()
                        error = float(np.max(abs(value - ref)))
                        limit = 1024 * np.finfo(np.float32).eps * max(1., float(np.max(abs(ref))))
                        checks.append(dict(field=name, max_error=error, limit=float(limit), passed=bool(error <= limit)))
                        saved['adapter.' + name] = value
                        saved['upstream.' + name] = ref
                    np.savez(args.output / (stage + '.npz'), **saved)
                    cases.append(dict(stage=stage, checks=checks, passed=all(x['passed'] for x in checks)))
                del adapted_graph, raw_graph, raw
            # Record every actuator command on the GPU and replay it through
            # an independent Data instance. The reference has no OSC, observer,
            # adapter counters, or calls back into Runtime.enqueue_substep.
            from validation.mjwarp_trace import record_control, replay_control, advance_cursor
            env.reset(seed=0, init_states=bank[:args.worlds])
            total = (args.warmup + args.steps) * env.substeps
            with wp.ScopedStream(env.stream):
                raw = mjw.make_data(env.built.native_model, nworld=args.worlds,
                                    nccdmax=env.capacities['nconmax'], **env.capacities)
                for name in inputs:
                    wp.copy(getattr(raw, name), getattr(env.data, name))
                trace = wp.empty((total, args.worlds, env.model.nu), dtype=float, device=env.stream.device)
                cursor = wp.zeros(args.worlds, dtype=wp.int64, device=env.stream.device)
                start = wp.clone(env.runtime.integration_count)
                original = c.enqueue
                def record_trajectory(policy_step):
                    original(policy_step)
                    wp.launch(record_control, (args.worlds, env.model.nu),
                              [env.runtime.integration_count, start, env.data.ctrl, trace], stream=env.stream)
                c.enqueue = record_trajectory
                env.runtime._control_graph = None
                try:
                    env.runtime.prepare_control_graph()
                finally:
                    c.enqueue = original
                with wp.ScopedCapture(stream=env.stream) as capture:
                    for _ in range(env.substeps):
                        mjw.step1(env.model, raw)
                        wp.launch(replay_control, (args.worlds, env.model.nu), [cursor, trace, raw.ctrl])
                        mjw.step2(env.model, raw)
                        wp.launch(advance_cursor, args.worlds, [cursor])
                raw_graph = capture.graph
                raw_initial = {name: wp.clone(getattr(raw, name)) for name in inputs} if args.raw_repeat else {}
            trajectory = dict(preparation_only=args.prepare)
            if not args.prepare:
                for k in range(args.warmup + args.steps):
                    env.step(actions(k, args.worlds))
                with wp.ScopedStream(env.stream):
                    for _ in range(args.warmup + args.steps):
                        wp.capture_launch(raw_graph, stream=env.stream)
                wp.synchronize_stream(env.stream)
                checks, saved = [], dict(control_trace=trace.numpy())
                for name in names:
                    actual, ref = getattr(env.data, name).numpy(), getattr(raw, name).numpy()
                    error = float(np.max(abs(actual - ref)))
                    limit = 1024 * np.finfo(np.float32).eps * max(1., float(np.max(abs(ref))))
                    checks.append(dict(field=name, max_error=error, limit=float(limit), passed=bool(error <= limit)))
                    saved['adapter.' + name], saved['upstream.' + name] = actual, ref
                counts = (env.runtime.integration_count.numpy() - start.numpy()).tolist()
                raw_counts = cursor.numpy().tolist()
                checks.append(dict(field='physical_steps', adapter=counts, upstream=raw_counts,
                                   expected=total, passed=counts == raw_counts == [total] * args.worlds))
                from mujoco_warp._src.types import OverflowType
                legal = int(OverflowType.ITERATIONS | OverflowType.LS_ITERATIONS)
                checks.append(dict(field='capacity', upstream=raw.overflow.numpy().tolist(),
                                   passed=not any(int(x) & ~legal for x in raw.overflow.numpy())))
                # The newer engine's residual is a separate physical diagnostic:
                # matching upstream does not establish its numerical accuracy.
                import mujoco
                from validation.precision import balance_residual
                residuals = []
                for engine, data in [('adapter', env.data), ('upstream', raw)]:
                    matrices = data.M.numpy()
                    accel, smooth, constraint = (getattr(data, name).numpy() for name in
                                                  ('qacc', 'qfrc_smooth', 'qfrc_constraint'))
                    for world in range(args.worlds):
                        matrix = np.empty((env.model.nv, env.model.nv), np.float64)
                        host = env.built.native_model
                        mujoco.mju_sym2dense(matrix, matrices[world].astype(np.float64),
                                            host.M_rownnz, host.M_rowadr, host.M_colind)
                        residuals.append(dict(engine=engine, world=world,
                            **balance_residual(matrix, accel[world], smooth[world], constraint[world])))
                np.savez(args.output / 'trajectory.npz', **saved)
                trajectory.update(checks=checks, passed=all(x['passed'] for x in checks),
                                  force_balance=residuals, warmup=args.warmup, steps=args.steps)
                if args.raw_repeat:
                    first = {name: getattr(raw, name).numpy() for name in names}
                    with wp.ScopedStream(env.stream):
                        second = mjw.make_data(env.built.native_model, nworld=args.worlds,
                                               nccdmax=env.capacities['nconmax'], **env.capacities)
                        for name, source in raw_initial.items():
                            wp.copy(getattr(second, name), source)
                        second_cursor = wp.zeros_like(cursor)
                        with wp.ScopedCapture(stream=env.stream) as second_capture:
                            for _ in range(env.substeps):
                                mjw.step1(env.model, second)
                                wp.launch(replay_control, (args.worlds, env.model.nu), [second_cursor, trace, second.ctrl])
                                mjw.step2(env.model, second)
                                wp.launch(advance_cursor, args.worlds, [second_cursor])
                        for _ in range(args.warmup + args.steps):
                            wp.capture_launch(second_capture.graph, stream=env.stream)
                    wp.synchronize_stream(env.stream)
                    variation, repeat = [], {}
                    for name in names:
                        actual = getattr(second, name).numpy()
                        variation.append(dict(field=name, raw_self_max=float(np.max(abs(actual - first[name]))),
                                              adapter_to_repeat_max=float(np.max(abs(saved['adapter.' + name] - actual)))))
                        repeat[name] = actual
                    np.savez(args.output / 'raw-repeat.npz', **repeat)
                    trajectory['raw_self_variation'] = variation
                    from validation.mjwarp_policy import compare_replays
                    equivalent = compare_replays({name: saved['adapter.' + name] for name in names}, first, repeat)
                    trajectory['roundoff_screen_passed'] = trajectory['passed']
                    trajectory['accumulation_policy'] = dict(
                        oracle='two independent raw upstream Data/Graphs with the same frozen controls',
                        envelope='max(unchanged 1024-epsilon floor, 10 * upstream-only spread), per-world max/RMS',
                        exact_fields=['ctrl', 'time'], native_cpu_policy_changed=False)
                    trajectory['accumulation_checks'] = equivalent
                    trajectory['passed'] = all(x['passed'] for x in equivalent + checks[-2:])
                    del second_capture, second, second_cursor
            del raw_graph, raw
            env.runtime._control_graph = None  # Stop recording into the bounded trajectory trace.
            lifecycle = []
            if not args.prepare and args.worlds > 1:
                # A selected reset must hold every other world's state and
                # controller history, including its Newton warmstart.
                before = env.get_sim_state()[0].copy()
                warm = env.data.qacc_warmstart.numpy()[0].copy()
                fields = ('body_xpos', 'body_xquat', 'body_xmat', 'geom_xpos',
                    'geom_xmat', 'site_xpos', 'site_xmat', 'subtree_com',
                    'qfrc_bias', 'qfrc_smooth', 'qfrc_constraint', 'cfrc_ext',
                    'cfrc_int', 'sensordata', 'ncon', 'contact')
                public = env._snapshot(fields)
                history = {k: c.buffers[k].numpy()[0].copy() for k in
                           ('mass', 'jacobian', 'q', 'qvel', 'goal_pos', 'goal_rot', 'gripper')}
                env.reset(id=[1], init_states=bank[1:2])
                lifecycle.append(dict(check='unselected state held', passed=bool(np.array_equal(before, env.get_sim_state()[0]))))
                lifecycle.append(dict(check='unselected warmstart held', passed=bool(np.array_equal(warm, env.data.qacc_warmstart.numpy()[0]))))
                lifecycle.append(dict(check='unselected controller held', passed=all(
                    np.array_equal(value, c.buffers[k].numpy()[0]) for k, value in history.items())))
                after = env._snapshot(fields)
                lifecycle.append(dict(check='unselected public derived state held', passed=all(
                    np.array_equal(public[k][0], after[k][0]) for k in fields if k != 'contact')))
                lifecycle.append(dict(check='unselected public contacts held', passed=all(
                    np.array_equal(value, after['contact'][0][k]) for k, value in public['contact'][0].items())))
                for k in range(5, 6):
                    env.step(actions(k, args.worlds))
                lifecycle.append(dict(check='contact holds invalidated on integration', passed=not env._held_contacts))
            if not args.prepare:
                assert not scope['events']
            state_fields = {'qpos', 'qvel', 'ctrl', 'time', 'qfrc_smooth'}
            state_passed = all(x['passed'] for case in cases for x in case['checks'] if x['field'] in state_fields)
            result = dict(passed=all(x['passed'] for x in bridge + lifecycle) and state_passed and
                                  (args.prepare or trajectory['passed']),
                          substep_state_passed=state_passed,
                          substep_roundoff_screen_passed=all(x['passed'] for x in cases),
                          preparation_only=args.prepare, bridge=bridge, cases=cases, lifecycle=lifecycle,
                          trajectory=trajectory, runtime_compilations=scope['events'],
                          worlds=args.worlds, task=args.task, upstream=env.identity,
                          scope='Independent NumPy bridge and raw upstream step1/step2 replay, including the complete control trace; force-balance diagnostics are separate from adapter equivalence. No performance measurement.')
            (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(result), flush=True)
            if not result['passed']:
                raise RuntimeError('MJWarp adapter/reference comparison failed; see summary.json')
        finally:
            env.close()


if __name__ == '__main__':
    main()
