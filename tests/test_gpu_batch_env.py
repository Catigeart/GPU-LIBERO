"""Native batch orchestration and state validation, with injected physics."""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import runpy
import sys
from threading import RLock
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'libero/libero/gpu/host/adapters/environment/batch_env.py'
spec = importlib.util.spec_from_file_location('_batch_api_tests', SOURCE)
module = importlib.util.module_from_spec(spec)
# These orchestration tests inject all physics. Load the two stdlib-only build
# helpers directly rather than importing GPU __init__ (and requiring Warp on CI).
GPU = SOURCE.parents[3]
helpers = {'libero.gpu.host.build.' + name: SimpleNamespace(**runpy.run_path(
    str(GPU / 'host/build' / (name + '.py')))) for name in ('compilation', 'startup')}
with patch.dict(sys.modules, helpers):
    spec.loader.exec_module(module)


GOAL_FIELDS = ('qpos', 'body_xpos', 'body_xquat', 'site_xpos', 'site_xmat', 'contact', 'ncon')


@contextmanager
def predicates(*args, **kwargs):
    yield


class BatchEnvTests(unittest.TestCase):
    def test_shell_clone_preserves_raw_type_and_rebinds_task_state(self):
        class Task:
            def __new__(cls):
                raise AssertionError('public constructor must not run for a cold clone')
            def success(self):
                return self.state.env is self
        shell = object.__new__(Task)
        shell.sim = object()
        shell.state = SimpleNamespace(env=shell)
        shell.callback = shell.success
        clone = module._clone_shell(shell)
        self.assertIs(type(clone), Task)
        self.assertIs(clone.sim, shell.sim)
        self.assertIsNot(clone.state, shell.state)
        self.assertTrue(clone.callback())
        self.assertIs(clone.callback.__self__, clone)

    def environment(self):
        # Inject native orchestration directly; public backend routing is tested
        # separately and must not load a physical engine in this fixture.
        env = object.__new__(module.GpuBatchEnv)
        env.num_envs = env.env_num = 4
        env.debug = True
        env.action_dim = 7
        env.substeps = 2
        env._integrations = np.zeros(4, dtype=np.int64)
        env._lock = RLock()
        env._closed = env._failed = False
        env._snapshot_cache = None
        env._goal_plans = []
        env._all = np.ones(4, dtype=np.int32)
        env._facades = [None]*4
        env.calls = []
        env.runtime = SimpleNamespace(intent=SimpleNamespace(assign=lambda x: env.calls.append('mask')))
        env._activate_all = lambda: env.runtime.intent.assign(env._all)
        env._physics = SimpleNamespace(enqueue_action=lambda x: env.calls.append(('batch', x.copy())))
        def complete():
            env.calls.append('check')
            env._integrations += env.substeps
        env._check = complete
        env._snapshot = lambda names=None: {'qpos': np.zeros((4, 2))}
        env._observations = lambda ids: {'value': np.array(ids)}
        env.shells = []
        for i in range(4):
            shell = SimpleNamespace(timestep=0, cur_time=0., done=False, control_timestep=.05)
            shell._post_action = lambda a: (float(a[0]), False, {})
            shell._check_success = lambda i=i: i == 2
            env.shells.append(shell)
        return env

    def test_one_physics_enqueue_for_four_worlds_and_ordered_outputs(self):
        env = self.environment()
        actions = np.zeros((4, 7))
        actions[:, 0] = [1, 2, 3, 4]
        with patch.dict(sys.modules, {'libero.gpu.host.adapters.observations.boundary':
                                    SimpleNamespace(boundary_predicates=predicates, GOAL_FIELDS=GOAL_FIELDS)}):
            obs, reward, success, infos = env.step(actions)
        self.assertEqual(sum(isinstance(x, tuple) for x in env.calls), 1)
        np.testing.assert_array_equal(env.calls[1][1], actions)
        np.testing.assert_array_equal(reward, [1, 2, 3, 4])
        np.testing.assert_array_equal(success, [False, False, True, False])
        self.assertEqual([i['env_id'] for i in infos], [0, 1, 2, 3])
        self.assertTrue(all(s.timestep == 1 and s.cur_time == .05 for s in env.shells))
        self.assertIsNone(env._snapshot_cache)

    def test_invalid_action_cannot_advance_any_world(self):
        env = self.environment()
        for actions in (np.zeros((3, 7)), np.full((4, 7), np.nan)):
            with self.assertRaises(ValueError):
                env.step(actions)
        self.assertFalse(env.calls)
        self.assertFalse(env._failed)

    def test_build_only_owner_requires_full_reset_before_step(self):
        env = self.environment()
        env._initialized = False
        with self.assertRaisesRegex(RuntimeError, 'reset the GPU batch'):
            env.step(np.zeros((4, 7)))
        with self.assertRaisesRegex(RuntimeError, 'reset all worlds'):
            env.reset(id=[1])
        self.assertFalse(env.calls)
        env._nq = env._nv = 1
        env.initial_states = np.zeros((4, 3))
        env._restore = lambda *a, **kw: None
        env.seed(0)
        env.reset()
        self.assertTrue(env._initialized)

    def test_failed_batch_is_quarantined(self):
        env = self.environment()
        def fail():
            raise ValueError('actual stage error')
        env._check = fail
        with self.assertRaisesRegex(ValueError, 'stage error'):
            with patch.dict(sys.modules, {'libero.gpu.host.adapters.observations.boundary':
                                        SimpleNamespace(boundary_predicates=predicates, GOAL_FIELDS=GOAL_FIELDS)}):
                env.step(np.zeros((4, 7)))
        with self.assertRaisesRegex(RuntimeError, 'rebuild'):
            env.step(np.zeros((4, 7)))

    def test_missing_world_commit_cannot_return_successful_observations(self):
        env = self.environment()
        def incomplete():
            env._integrations[:3] += env.substeps
        env._check = incomplete
        with patch.dict(sys.modules, {'libero.gpu.host.adapters.observations.boundary':
                                    SimpleNamespace(boundary_predicates=predicates, GOAL_FIELDS=GOAL_FIELDS)}):
            with self.assertRaisesRegex(RuntimeError, 'commit every'):
                env.step(np.zeros((4, 7)))
        self.assertTrue(env._failed)

    def test_formal_step_waits_without_status_checks_or_commit_verification(self):
        env = self.environment()
        env.debug = False
        del env._integrations
        env._status = SimpleNamespace(synchronize_and_check=lambda:
            self.fail('formal step must not gather or validate a status packet'))
        env.stream = object()
        env._check = module.GpuBatchEnv._check.__get__(env)
        warp = SimpleNamespace(synchronize_stream=lambda stream: env.calls.append('wait'))
        with patch.dict(sys.modules, {'warp': warp,
                'libero.gpu.host.adapters.observations.boundary':
                SimpleNamespace(boundary_predicates=predicates, GOAL_FIELDS=GOAL_FIELDS)}):
            _, rewards, _, _ = env.step(np.zeros((4, 7)))
        self.assertEqual(rewards.shape, (4,))
        self.assertIn('wait', env.calls)
        self.assertNotIn('check', env.calls)

    def test_formal_driver_captures_once_and_replays_with_each_new_action(self):
        driver_type = runpy.run_path(str(SOURCE.with_name('batched_physics.py')))['NativeBatchedPhysics']
        substeps, uploaded, launches = [], [], []
        graph = object()
        @contextmanager
        def capture(**kwargs):
            yield SimpleNamespace(graph=graph)
        controller = SimpleNamespace(controller=SimpleNamespace(action=SimpleNamespace(
            assign=lambda action: uploaded.append(action.copy()))))
        runtime = SimpleNamespace(euler=SimpleNamespace(capacity=4),
            intent=SimpleNamespace(shape=(4,)), substeps=25,
            enqueue_substep=lambda **kwargs: substeps.append(kwargs['policy_step']))
        driver = driver_type(SimpleNamespace(runtime=runtime, controller=controller, stream=object()))
        warp = SimpleNamespace(ScopedStream=predicates, ScopedCapture=capture,
            capture_launch=lambda graph, **kwargs: launches.append(graph))
        with patch.dict(sys.modules, {'warp': warp}):
            driver.enqueue_action(np.zeros((4, 7)))
            driver.enqueue_action(np.ones((4, 7)))
        self.assertEqual(substeps, [True]+[False]*24)
        self.assertEqual(launches, [graph, graph])
        np.testing.assert_array_equal(uploaded[0], np.zeros((4, 7)))
        np.testing.assert_array_equal(uploaded[1], np.ones((4, 7)))

    def test_reset_only_changes_selected_lifecycle_and_rng_is_reproducible(self):
        env = self.environment()
        env._nq = env._nv = 1
        env.initial_states = np.arange(30).reshape(10, 3).astype(float)
        writes = []
        env._restore = lambda rows, ids, reset: writes.append((rows.copy(), ids, reset))
        for shell in env.shells:
            shell.timestep, shell.cur_time, shell.done = 8, .4, True
        env.seed(9)
        env.reset(id=[3, 1])
        first = writes[-1][0].copy()
        env.seed(9)
        env.reset(id=[3, 1])
        np.testing.assert_array_equal(writes[-1][0], first)
        self.assertEqual(writes[-1][1], (3, 1))
        self.assertTrue(writes[-1][2])
        self.assertEqual([s.timestep for s in env.shells], [8, 0, 8, 0])
        with self.assertRaises(ValueError):
            env.set_init_state(np.zeros((2, 3)), id=[1, 1])

    def test_close_releases_shared_owner_once_and_rejects_future_use(self):
        env = self.environment()
        env._facades = []
        env.observer = SimpleNamespace(close=lambda: env.calls.append('observer-close'))
        env._physics.close = lambda: env.calls.append('physics-close')
        env._physics.discard_control_graph = lambda: env.calls.append('graph-discard')
        env._physics.bind_observer = lambda observer: env.calls.append(('observer-unbind', observer))
        env._status = SimpleNamespace(release=lambda: env.calls.append('status-release'))
        env.close()
        env.close()
        self.assertEqual(env.calls, ['graph-discard', ('observer-unbind', None),
                                    'observer-close', 'physics-close', 'status-release'])
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            env.seed(0)

    def test_goal_plan_prefetch_and_context_follow_requested_world_schema(self):
        env = self.environment()
        schemas = [('qpos',), ('body_xpos', 'site_xpos', 'site_xmat'),
                   ('qpos',), GOAL_FIELDS]
        env._goal_plans = [SimpleNamespace(fields_for=lambda shell, fields=fields: fields)
                           for fields in schemas]
        snapshots, contexts, order = [], [], []
        env._snapshot = lambda names=None: snapshots.append(tuple(names)) or {}
        @contextmanager
        def capture(shell, facade, *, fields):
            contexts.append(tuple(fields))
            yield
        for world, shell in enumerate(env.shells):
            shell._post_action = lambda action, world=world: order.append(('post', world)) or (0., False, {})
            shell._check_success = lambda world=world: order.append(('success', world)) or False
        with patch.dict(sys.modules, {'libero.gpu.host.adapters.observations.boundary':
                SimpleNamespace(boundary_predicates=capture, GOAL_FIELDS=GOAL_FIELDS)}):
            env.step(np.zeros((4, 7)))
        self.assertEqual(snapshots, [GOAL_FIELDS])
        self.assertEqual(contexts, schemas)
        self.assertEqual(order, [(operation, world) for world in range(4)
                                 for operation in ('post', 'success')])
        self.assertIsNone(env._snapshot_cache)

    def test_snapshot_skips_contacts_and_lazy_read_completes_real_fields_once(self):
        env = self.environment()
        env._snapshot = module.GpuBatchEnv._snapshot.__get__(env)
        env.stream = object()
        downloads = []
        class Array:
            def __init__(self, name, value):
                self.name, self.value = name, np.asarray(value)
                self.shape, self.dtype = self.value.shape, self.value.dtype
            def numpy(self): return self.value
        def copy(target, source, stream):
            self.assertIs(stream, env.stream)
            downloads.append(source.name)
            target.value[...] = source.value
        def empty(shape, dtype, device, pinned):
            self.assertEqual(device, 'cpu')
            self.assertTrue(pinned)
            return Array(None, np.empty(shape, dtype=dtype))
        env._fields = {
            'qpos': SimpleNamespace(array=Array('qpos', np.zeros((4, 2))), shape=(2,)),
            'qvel': SimpleNamespace(array=Array('qvel', np.ones((4, 2))), shape=(2,)),
            'ncon': SimpleNamespace(array=Array('ncon', [1, 0, 2, 1]), shape=()),
        }
        env.runtime.forward = SimpleNamespace(transport=SimpleNamespace(inputs={
            'geom1': Array('geom1', [[4, 8], [9, 10], [12, 13], [14, 15]]),
            'geom2': Array('geom2', [[5, 9], [10, 11], [14, 15], [16, 17]]),
        }))
        fake_warp = SimpleNamespace(empty=empty, copy=copy, synchronize_stream=lambda stream: None)
        with patch.dict(sys.modules, {'warp': fake_warp}):
            boundary_downloads = SimpleNamespace(**runpy.run_path(str(SOURCE.with_name('boundary_downloads.py'))))
        with patch.dict(sys.modules, {
                'libero.gpu.host.adapters.environment.boundary_downloads': boundary_downloads,
                'libero.gpu.host.adapters.observations.boundary': SimpleNamespace(GOAL_FIELDS=GOAL_FIELDS)}):
            env._snapshot_cache = env._snapshot(('qpos',))
            held = env._snapshot_cache['qpos']
            self.assertEqual(downloads, ['qpos'])
            boundary = module._WorldBoundary(env, 2)
            np.testing.assert_array_equal(boundary.read_fields(('qvel',))['qvel'], [1, 1])
            result = boundary.read_fields(('contact', 'ncon'))
            np.testing.assert_array_equal(result['contact']['geom1'], [12, 13])
            np.testing.assert_array_equal(result['contact']['geom2'], [14, 15])
            self.assertEqual(result['ncon'], 2)
            boundary.read_fields(('qpos', 'qvel', 'contact', 'ncon'))
            self.assertEqual(downloads, ['qpos', 'qvel', 'ncon', 'geom1', 'geom2'])
            env._sample_event(1)
            boundary.read_fields(('qvel',))
            env._fields['qpos'].array.value[:] = 17
            replacement = env._snapshot(('qpos',))['qpos']
            np.testing.assert_array_equal(held, np.zeros((4, 2)))
            np.testing.assert_array_equal(replacement, np.full((4, 2), 17))
        self.assertEqual(downloads[-2:], ['qvel', 'qpos'])
        self.assertEqual(downloads.count('qvel'), 2)

    def test_controlled_observable_edits_clear_event_cache_and_keep_stream(self):
        env = self.environment()
        env.stream = object()
        env.observer = SimpleNamespace(
            modify_observable=lambda *args, **kwargs: env.calls.append(('modify', args, kwargs)),
            add_observable=lambda *args, **kwargs: env.calls.append(('add', args, kwargs)))
        env._snapshot_cache = {'stale': True}
        env.modify_observable('sensor', 'sampling_rate', 40)
        self.assertIsNone(env._snapshot_cache)
        env.add_observable('custom')
        self.assertEqual(env.calls[0], ('modify', ('sensor', 'sampling_rate', 40), {'stream': env.stream}))
        self.assertEqual(env.calls[1], ('add', ('custom',), {'stream': env.stream}))


if __name__ == '__main__':
    unittest.main()
