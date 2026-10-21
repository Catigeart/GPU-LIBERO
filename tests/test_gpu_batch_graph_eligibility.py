"""Cold mock scheduling tests; no Warp import, GPU allocation, or compilation."""
import ast
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / 'libero/libero/gpu/host/adapters/environment/batched_physics.py'


def load_driver():
    selected = {'NativeBatchedPhysics', 'prepare_native_batch'}
    nodes = [node for node in ast.parse(SOURCE.read_text()).body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in selected]
    namespace = {'__name__': __name__, 'dataclass': dataclass, 'nullcontext': nullcontext,
                 'cold_build_scope': lambda function: function,
                 'stack_reset_states': lambda states: ('stacked', states)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace


class Observer:
    def __init__(self, worlds):
        self.nworld = worlds
        self.requires_host_sampling = False
        self.owners_current = True
        self.eligibility_reads = 0
        self.callback_events = []
        self.replay_events = []

    @property
    def graph_eligible(self):
        self.eligibility_reads += 1
        return not self.requires_host_sampling and self.owners_current

    def enable_host_sampling(self):
        self.requires_host_sampling = True

    def notify_control_graph_replay(self, **event):
        self.replay_events.append(event)


class FakeWarp:
    def __init__(self):
        self.capturing = None
        self.captures = []
        self.launches = []

    @contextmanager
    def ScopedStream(self, stream):
        yield

    @contextmanager
    def ScopedCapture(self, **options):
        graph = SimpleNamespace(operations=[], options=options)
        self.captures.append(graph)
        self.capturing = graph
        try:
            yield SimpleNamespace(graph=graph)
        finally:
            self.capturing = None

    def capture_launch(self, graph, *, stream):
        self.launches.append((graph, stream))
        for operation in graph.operations:
            operation()


class BatchGraphEligibilityTests(unittest.TestCase):
    def test_post_deferral_requires_known_device_sampler_and_keeps_final_publication(self):
        class BuiltinSampler:
            graph_eligible = True
        class Fusion:
            def __init__(self, *args, **kwargs): self.steps = []
            def enqueue_substep(self, **kwargs): self.steps.append(kwargs)
        modules = {
            'libero.gpu.host.runtime.step.fused_substep': SimpleNamespace(FusedSubstep=Fusion),
            'libero.gpu.host.adapters.observations.adapter': SimpleNamespace(GpuObservationAdapter=BuiltinSampler),
        }
        for observer, requested, expected in [
                (BuiltinSampler(), True, [False, False, True]),
                (BuiltinSampler(), False, [True, True, True]),
                (SimpleNamespace(graph_eligible=True), True, [True, True, True]),
                (None, True, [True, True, True])]:
            with self.subTest(observer=type(observer), requested=requested), patch.dict(sys.modules, modules):
                physics = self.driver(worlds=2)
                physics._observer = observer
                fusion = physics.prepare_fused_substep_graph('unused', defer_post=requested)
                self.assertEqual([s['publish_post'] for s in fusion.steps], expected)
                self.assertEqual([s['policy_step'] for s in fusion.steps], [True, False, False])

    def test_offline_branch_preparation_records_without_advancing_any_world(self):
        driver = self.driver()
        driver.bind_observer(self.observer)
        driver.prepare_substep_modules()
        self.assertEqual([o['policy_step'] for _, _, o in self.python_steps], [True, False])
        self.assertEqual(self.device_steps, [0]*8)
        self.assertIsNone(driver._control_graph)
        self.assertEqual(self.warp.launches, [])
        self.assertNotIn('force_module_load', self.warp.captures[0].options)
        driver.enqueue_action([[0.]*7 for _ in range(8)])
        self.assertEqual(self.device_steps, [3]*8)
        self.assertEqual(len(self.warp.launches), 1)

    def test_offline_branch_preparation_respects_host_sampling_fallback(self):
        driver = self.driver()
        driver.bind_observer(self.observer)
        self.observer.enable_host_sampling()
        driver.prepare_substep_modules()
        self.assertEqual(self.warp.captures, [])
        self.assertEqual(self.python_steps, [])

    def setUp(self):
        self.namespace = load_driver()
        self.warp = FakeWarp()
        self.modules = patch.dict(sys.modules, {'warp': self.warp})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def driver(self, worlds=8, *, debug=False):
        self.observer = Observer(worlds)
        self.device_steps = [0] * worlds
        self.python_steps = []
        self.uploaded = []
        self.close_events = []
        self.arena = object()
        stream = object()

        def substep(**options):
            self.python_steps.append((self.arena, worlds, options))

            def device_operation():
                for w in range(worlds):
                    self.device_steps[w] += 1

            if self.warp.capturing is not None:
                self.warp.capturing.operations.append(device_operation)
            else:
                device_operation()
                if self.observer.requires_host_sampling:
                    self.observer.callback_events.append(options['policy_step'])

        controller = SimpleNamespace(
            controller=SimpleNamespace(action=SimpleNamespace(assign=self.uploaded.append)),
            stage_action=self.uploaded.append)
        runtime = SimpleNamespace(euler=SimpleNamespace(capacity=worlds),
            intent=SimpleNamespace(shape=(worlds,)), substeps=3, enqueue_substep=substep)
        bundle = SimpleNamespace(runtime=runtime, controller=controller, stream=stream,
            close=lambda: self.close_events.append('arena'))
        self.physics = self.namespace['NativeBatchedPhysics'](bundle, debug=debug)
        return self.physics

    def test_standard_observer_captures_once_replays_each_action_and_notifies_after_launch(self):
        physics = self.driver()
        physics.bind_observer(self.observer)
        notify = self.observer.notify_control_graph_replay

        def after_launch(**event):
            self.assertEqual(self.device_steps, [3 * (len(self.observer.replay_events) + 1)] * 8)
            notify(**event)

        self.observer.notify_control_graph_replay = after_launch
        physics.prepare_control_graph()
        self.assertEqual(self.observer.replay_events, [])
        self.assertEqual(self.device_steps, [0] * 8)
        physics.enqueue_action('action1')
        physics.enqueue_action('action2')
        self.assertEqual(len(self.warp.captures), 1)
        self.assertEqual(len(self.warp.launches), 2)
        self.assertEqual(self.uploaded, ['action1', 'action2'])
        self.assertEqual(self.device_steps, [6] * 8)
        self.assertEqual([s[2]['policy_step'] for s in self.python_steps], [True, False, False])
        self.assertEqual(self.observer.callback_events, [])
        self.assertEqual(self.observer.replay_events, [dict(timestep=None, substeps=3,
            stream=physics.bundle.stream)] * 2)
        self.assertGreaterEqual(self.observer.eligibility_reads, 5)

    def test_custom_callback_after_capture_discards_graph_and_runs_each_substep(self):
        for worlds in (1, 2, 8, 64):
            with self.subTest(worlds=worlds):
                self.warp.captures.clear()
                self.warp.launches.clear()
                physics = self.driver(worlds)
                physics.bind_observer(self.observer)
                physics.enqueue_action('builtins')
                old_graph = physics._control_graph
                self.observer.enable_host_sampling()
                physics.enqueue_action('custom1')
                physics.enqueue_action('custom2')
                self.assertIsNone(physics._control_graph)
                self.assertIs(self.warp.launches[0][0], old_graph)
                self.assertEqual(len(self.warp.launches), 1)
                self.assertEqual(len(self.warp.captures), 1)
                self.assertEqual(self.observer.callback_events, [True, False, False] * 2)
                self.assertEqual(len(self.observer.replay_events), 1)
                self.assertEqual(len(self.python_steps), 9)
                self.assertTrue(all(arena is self.arena and w == worlds
                                    for arena, w, _ in self.python_steps))
                self.assertEqual(self.device_steps, [9] * worlds)

    def test_host_sampling_before_capture_never_records_callbacks_once_in_a_graph(self):
        physics = self.driver(2)
        self.observer.enable_host_sampling()
        physics.bind_observer(self.observer)
        physics.prepare_control_graph()
        self.assertEqual(self.python_steps, [])
        physics.enqueue_action('first')
        physics.enqueue_action('second')
        self.assertEqual(self.warp.captures, [])
        self.assertEqual(self.warp.launches, [])
        self.assertEqual(self.observer.callback_events, [True, False, False] * 2)
        self.assertEqual(self.observer.replay_events, [])
        self.assertEqual(self.device_steps, [6, 6])

    def test_changed_live_owners_drop_cached_graph_before_prepare_returns(self):
        physics = self.driver(2)
        physics.bind_observer(self.observer)
        physics.prepare_control_graph()
        self.observer.owners_current = False
        physics.prepare_control_graph()
        self.assertIsNone(physics._control_graph)
        physics.enqueue_action('changed-owners')
        self.assertEqual(len(self.warp.captures), 1)
        self.assertEqual(self.warp.launches, [])
        self.assertEqual(self.device_steps, [3, 3])

    def test_debug_keeps_existing_loop_and_never_notifies_replay(self):
        physics = self.driver(2, debug=True)
        self.observer.enable_host_sampling()
        physics.bind_observer(self.observer)
        physics.prepare_control_graph()
        physics.enqueue_action('debug')
        self.assertEqual(self.uploaded, ['debug'])
        self.assertEqual(self.observer.callback_events, [True, False, False])
        self.assertEqual(self.observer.replay_events, [])
        self.assertEqual(self.warp.captures, [])
        self.assertEqual(self.warp.launches, [])

    def test_rebind_unbind_and_close_release_graph_before_observer_and_arena(self):
        physics = self.driver(2)
        physics.bind_observer(self.observer)
        events = self.close_events

        class Graph:
            def __del__(graph):
                events.append(('graph', physics._observer))

        physics._control_graph = Graph()
        old_observer = self.observer
        replacement = Observer(2)
        physics.bind_observer(replacement)
        self.assertEqual(events, [('graph', old_observer)])
        physics._control_graph = Graph()
        physics.close()
        self.assertEqual(events, [('graph', old_observer), ('graph', replacement), 'arena'])
        self.assertIsNone(physics._observer)
        self.assertIsNone(physics._control_graph)

    def test_wrong_world_binding_cannot_replace_existing_graph_or_observer(self):
        physics = self.driver(2)
        physics.bind_observer(self.observer)
        physics.prepare_control_graph()
        graph = physics._control_graph
        with self.assertRaisesRegex(ValueError, 'world counts'):
            physics.bind_observer(Observer(8))
        self.assertIs(physics._control_graph, graph)
        self.assertIs(physics._observer, self.observer)

    def test_unbound_physics_only_driver_retains_existing_graph_behavior(self):
        physics = self.driver(2)
        physics.enqueue_action('pure-physics1')
        physics.enqueue_action('pure-physics2')
        self.assertEqual(len(self.warp.captures), 1)
        self.assertEqual(len(self.warp.launches), 2)
        self.assertEqual(self.device_steps, [6, 6])

    def test_native_batch_retains_exact_observation_build_context(self):
        physics = self.driver(2)
        built = SimpleNamespace(metadata={'model_sha256': 'physical-model'})
        project = {'root': Path('/cold/project'), 'events': []}
        modules = {
            'libero.gpu.host.factory': SimpleNamespace(prepare_cuda_physics=lambda *a, **kw: physics.bundle),
            'libero.gpu.host.runtime.step.recovery_final_event_environment_factory': SimpleNamespace(
                actual_decoder_record=lambda built: 'decoder',
                _native_intake_package=lambda built: ('model-spec', 'constraints')),
            'libero.gpu.host.runtime.core.product_paths': SimpleNamespace(
                product_build=lambda: nullcontext(),
                mathdx_root=lambda: self.fail('explicit SDK must be retained'),
                toolchain_scope=lambda sdk: nullcontext(), cache_root=lambda: Path('/cold/cache'),
                build_project=lambda root, **kw: nullcontext(project)),
            'libero.gpu.host.build.native': SimpleNamespace(strict_build_receipts_scope=lambda strict: nullcontext()),
            'libero.gpu.host.build.product_cache': SimpleNamespace(product_cache_scope=lambda root: nullcontext()),
            'libero.gpu.host.build.startup': SimpleNamespace(startup_phase=lambda name: nullcontext()),
        }
        self.warp.get_device = lambda device: SimpleNamespace(arch=80)
        with patch.dict(sys.modules, modules):
            result = self.namespace['prepare_native_batch'](built, states=('s0', 's1'),
                build_root='/cold/project', mathdx_root='/custom/mathdx', strict_build_receipts=True)
        self.assertEqual(result.bundle.observation_build_options, dict(build_root='/cold/project',
            mathdx_root='/custom/mathdx', strict_build_receipts=True))
        self.assertEqual(result.bundle.build_report, dict(directory='/cold/project', events=[]))
        self.assertEqual(built.metadata['decoder_record'], 'decoder')


if __name__ == '__main__':
    unittest.main()
