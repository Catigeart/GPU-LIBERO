"""One environment per task, explicit CPU reference or load-only GPU run."""
import json
import inspect
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from .common import (FIELDS, actions, compare_endpoints, input_record, label,
                     load_endpoint, save_json, timing_summary)


def physical_cores(worlds):
    """Select distinct allowed physical cores, never SMT siblings."""
    selected, seen = [], set()
    for cpu in sorted(os.sched_getaffinity(0)):
        root = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        identity = tuple((root / name).read_text().strip() for name in ('physical_package_id', 'core_id'))
        if identity not in seen:
            selected.append(cpu)
            seen.add(identity)
        if len(selected) == worlds:
            return selected
    raise RuntimeError(f'{worlds} distinct allowed physical CPU cores are required')


def require_cpu_control_wrapper(wrapper):
    """Reject this project's GPU-default wrapper before constructing anything."""
    if 'physics_backend' in inspect.signature(wrapper.__init__).parameters:
        raise ValueError('CPU recording requires the independent CPU LIBERO Python/check-out; '
                         'the selected ControlEnv has GPU backend routing')
    return inspect.getfile(wrapper)


def cpu_environment(bddl, core):
    from libero.libero.envs.env_wrapper import ControlEnv
    require_cpu_control_wrapper(ControlEnv)
    os.sched_setaffinity(0, {core})

    class RecordedControlEnv(ControlEnv):
        @property
        def validation_ready(self):
            return int(self.env.action_dim)

        @property
        def validation_endpoint(self):
            data = self.sim.data
            result = {field: np.array(getattr(data, field), copy=True) for field in FIELDS}
            for name in ('cfrc_ext', 'cfrc_int', 'ncon', 'nefc', 'qfrc_bias',
                         'qfrc_smooth', 'qfrc_constraint', 'qM', 'site_xpos', 'site_xmat'):
                result[name] = np.array(getattr(data, name), copy=True)
            for target, source in [('sensor', 'sensordata'), ('body_xpos', 'xpos'), ('body_xquat', 'xquat')]:
                result[target] = np.array(getattr(data, source), copy=True)
            return result

    return RecordedControlEnv(bddl_file_name=bddl, control_freq=20, horizon=1000,
        hard_reset=True, use_camera_obs=False, has_renderer=False,
        has_offscreen_renderer=False, ignore_done=True)


def gpu_status(env):
    runtime = env.runtime
    result = {name: getattr(runtime, name).numpy().tolist()
              for name in ('integration_count', 'state_epoch', 'first_error', 'reset_events')}
    result['newton_attempts'] = runtime.forward.n.n_attempt.numpy().tolist()
    result['position_math'] = runtime.forward._dataflow_reuse.math.numpy().tolist()
    return result


def gpu_endpoint(env, observations, rewards, successes):
    actual = env._physics.download_state()
    for name in ('qfrc_bias', 'qfrc_smooth', 'qfrc_constraint', 'qM',
                 'body_xpos', 'body_xquat', 'site_xpos', 'site_xmat'):
        actual[name] = env._fields[name].array.numpy().reshape(env.num_envs, -1)
    actual.update({key: value.numpy() for key, value in env._physics.bundle.post.public.items()})
    actual['ncon'] = env.runtime.forward.transport.ncon.numpy()
    actual['nefc'] = env.runtime.forward.n.ws.nefc.numpy()
    actual.update({'obs.' + key: value for key, value in observations.items()})
    actual['rewards_trace'] = np.asarray(rewards).T
    actual['successes'] = np.asarray(successes)
    return actual


def read_references(args, task, protocol):
    if args.backend == 'cpu' or args.record_reference:
        return None
    stem = label(task, args.worlds, args.case)
    values = []
    for index, folder in enumerate((args.cpu_reference, args.baseline_reference)):
        reference_stem = stem
        if index == 0 and args.case == 'varying_cache_miss' and not (folder / (stem + '.json')).exists():
            reference_stem = label(task, args.worlds, 'varying')
        receipt = json.loads((folder / (reference_stem + '.json')).read_text())
        if receipt.get('input', {}).get('signature') != protocol['signature']:
            raise ValueError('reference input signature differs: ' + str(folder / (stem + '.json')))
        if receipt.get('status') != ('recorded_cpu' if index == 0 else 'recorded_gpu'):
            raise ValueError('reference must be an explicitly recorded frozen endpoint')
        values.append(load_endpoint(folder / (reference_stem + '.npz')))
    samples = []
    for folder in args.baseline_sample_reference:
        receipt = json.loads((folder / (stem + '.json')).read_text())
        if receipt.get('status') != 'recorded_gpu' or receipt.get('input', {}).get('signature') != protocol['signature']:
            raise ValueError('additional sample is not a matching frozen GPU reference: ' + str(folder))
        samples.append(load_endpoint(folder / (stem + '.npz')))
    return (*values, samples)  # CPU, old GPU, optional OLD samples; never new.


def run_task(args, task):
    """Return 2 only for a numerical/work regression, 3 for infrastructure."""
    stem = label(task, args.worlds, args.case)
    receipt_path = args.output / (stem + '.json')
    if receipt_path.exists() or (args.output / (stem + '.npz')).exists():
        raise FileExistsError('Use a fresh output directory; existing evidence is not overwritten: ' + stem)
    env, guard = None, None
    row = dict(backend=args.backend, task=task['key'], suite=task['suite'], task_id=task['task_id'],
               worlds=args.worlds, case=args.case, performance_measured=False)
    try:
        bank = np.load(Path(task['directory']) / 'initial_states.npy', allow_pickle=False)
        if len(bank) < args.worlds:
            raise ValueError('initial state bank has fewer rows than requested worlds')
        protocol = input_record(task, bank, worlds=args.worlds, warmup=args.warmup,
                                steps=args.steps, seed=args.seed, case=args.case)
        row['input'] = protocol
        references = read_references(args, task, protocol)
        from libero.libero import get_libero_path
        bddl = str(Path(get_libero_path('bddl_files')) / task['key'])
        if args.backend == 'gpu':
            import warp as wp
            from libero.libero.envs import GpuBatchEnv
            from libero.gpu.host.build.compilation import compilation_scope, install_warp_guard
            install_warp_guard()
            guard = compilation_scope(False)
            scope = guard.__enter__()
            tick = time.perf_counter()
            env = GpuBatchEnv(bddl, args.worlds, init_states=bank, seed=args.seed,
                build_root=args.build_root, initialize=False, allow_compile=False,
                debug=False, device=args.device, use_camera_obs=False, ignore_done=True)
            row['creation_s'] = time.perf_counter() - tick
            tick = time.perf_counter()
            env.reset(seed=args.seed, init_states=bank[:args.worlds])
            row['initial_reset_s'] = time.perf_counter() - tick
            tick = time.perf_counter()
            fusion = env.prepare_substep_fusion(args.build_root / 'cmake-control',
                allow_compile=False, fast_compile=args.fast_compile, device_lto=args.device_lto)
            if args.case == 'varying_cache_miss':
                flags = np.ones(args.worlds, np.int32)
                flags[-1] = 0
                forced = wp.array(flags, dtype=wp.int32, device=env.stream.device)
                fusion.arguments.position.cached = forced.ptr
                fusion.device_arguments.assign(np.frombuffer(bytes(fusion.arguments), np.uint8).copy())
            graph = env._physics._control_graph
            row['graph_prepare_s'] = time.perf_counter() - tick
            row['build_config'] = fusion.manifest['config']
        else:
            import mujoco
            from libero.libero.envs import SubprocVectorEnv
            from libero.libero.envs.env_wrapper import ControlEnv
            row['cpu_control_source'] = require_cpu_control_wrapper(ControlEnv)
            if mujoco.__version__ != '2.3.7':
                raise ValueError('CPU semantic reference requires MuJoCo 2.3.7')
            cores = physical_cores(args.worlds)
            tick = time.perf_counter()
            env = SubprocVectorEnv([lambda core=core: cpu_environment(bddl, core) for core in cores])
            if env.get_env_attr('validation_ready') != [7] * args.worlds:
                raise ValueError('this OSC action protocol requires action_dim=7')
            row.update(creation_s=time.perf_counter() - tick, worker_cpus=cores,
                       mujoco=mujoco.__version__, python=sys.executable)

        stages = ('correctness', 'performance') if args.performance and references else ('endpoint',)
        row['runs'] = {}
        passed = True
        sequence = [actions(k, args.worlds, args.case) for k in range(args.warmup + args.steps)]
        for stage in stages:
            tick = time.perf_counter()
            if args.backend == 'gpu':
                env.reset(seed=args.seed, init_states=bank[:args.worlds])
                env._physics._control_graph, env._physics._substep_fusion = graph, fusion
            else:
                env.seed([args.seed] * args.worlds)
                env.reset()
                env.set_init_state(bank[:args.worlds])
            reset_s = time.perf_counter() - tick
            for action in sequence[:args.warmup]:
                env.step(action)
            if args.backend == 'gpu':
                wp.synchronize_stream(env.stream)
                before = gpu_status(env)
            latencies, rewards_trace = [], []
            timed = args.performance and stage != 'correctness'
            tick = time.perf_counter() if timed else None
            for action in sequence[args.warmup:]:
                start = time.perf_counter() if timed else None
                observations, rewards, successes, _ = env.step(action)
                if timed:
                    latencies.append(1000 * (time.perf_counter() - start))
                rewards_trace.append(rewards)
            if args.backend == 'gpu':
                wp.synchronize_stream(env.stream)
            elapsed = time.perf_counter() - tick if timed else None
            if args.backend == 'gpu':
                after = gpu_status(env)
                counters = {name: (np.asarray(after[name]) - before[name]).tolist()
                            for name in ('integration_count', 'newton_attempts', 'position_math', 'reset_events')}
                valid = not any(after['first_error']) and counters['integration_count'] == [args.steps * env.substeps] * args.worlds
                actual = gpu_endpoint(env, observations, rewards_trace, successes)
                stage_row = dict(reset_s=reset_s, counters=counters, status=after)
            else:
                states = env.get_env_attr('validation_endpoint')
                actual = {key: np.stack([state[key] for state in states]) for key in states[0]}
                actual.update({'obs.' + key: np.stack([obs[key] for obs in observations]) for key in observations[0]})
                actual['rewards_trace'] = np.asarray(rewards_trace).T
                actual['successes'] = np.asarray(env.check_success())
                valid, stage_row = True, dict(reset_s=reset_s)
            valid = bool(valid and all(np.isfinite(actual[field]).all() for field in FIELDS))
            if references:
                checks = compare_endpoints(actual, references[1], references[0], args.worlds,
                                           baseline_samples=references[2])
                stage_row['checks'] = checks
                valid = bool(valid and all(check['passed'] for check in checks))
            stage_row['passed'] = valid
            if timed:
                stage_row.update(timing_summary(latencies, elapsed, args.worlds))
                row['performance_measured'] = True
            row['runs'][stage] = stage_row
            np.savez(args.output / (stem + ('.npz' if stage != 'performance' else '-performance.npz')), **actual)
            if not valid:
                passed = False
                break  # Never time a failed candidate.
        if args.backend == 'gpu':
            row['runtime_compilations'] = list(scope['events'])
            if scope['events']:
                raise RuntimeError('load-only worker observed a compilation event')
        row['passed'] = passed
        row['status'] = ('recorded_' + args.backend if references is None else 'qualified') if passed else 'regression'
        save_json(receipt_path, row)
        return 0 if passed else 2
    except Exception:
        row.update(status='infrastructure_error', passed=False, error=traceback.format_exc())
        save_json(receipt_path, row)
        return 3
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            if guard is not None:
                guard.__exit__(None, None, None)
