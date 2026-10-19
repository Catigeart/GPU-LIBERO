"""Exercise runner scheduling/admission using injected owners, without CUDA."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from validation import runner
from validation.common import FIELDS, input_record, label, save_json
from validation.summary import summarize


def state(value):
    return {key: np.full((2, 1), value, np.float64) for key in FIELDS}


class ValidationRunnerTests(unittest.TestCase):
    def test_cpu_reference_rejects_gpu_default_wrapper_before_construction(self):
        class CpuControl:
            def __init__(self, bddl_file_name):
                raise AssertionError('must not construct an environment')

        class GpuControl:
            def __init__(self, bddl_file_name, physics_backend='gpu'):
                raise AssertionError('must not construct an environment')

        self.assertEqual(runner.require_cpu_control_wrapper(CpuControl), __file__)
        with self.assertRaisesRegex(ValueError, 'independent CPU LIBERO'):
            runner.require_cpu_control_wrapper(GpuControl)

    def exercise(self, endpoint, expected_steps, expected_code):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bank = np.zeros((2, 3))
            np.save(root / 'initial_states.npy', bank)
            task = dict(suite='libero_10', task_id=0, key='libero_10/test.bddl',
                        model_sha256='fixture-model', directory=str(root))
            args = SimpleNamespace(backend='gpu', worlds=2, case='varying', warmup=5,
                steps=20, seed=0, performance=True, record_reference=False,
                output=root / 'output', cpu_reference=root / 'cpu', baseline_reference=root / 'old',
                baseline_sample_reference=[], build_root=root / 'build', device='cuda:0',
                fast_compile=True, device_lto=False)
            args.output.mkdir()
            stem = label(task, 2, 'varying')
            protocol = input_record(task, bank, worlds=2, warmup=5, steps=20, seed=0, case='varying')
            for folder, backend, value in [(args.cpu_reference, 'cpu', 0), (args.baseline_reference, 'gpu', 1e-7)]:
                folder.mkdir()
                save_json(folder / (stem + '.json'), dict(status='recorded_' + backend, input=protocol))
                np.savez(folder / (stem + '.npz'), **state(value))
            env = SimpleNamespace(stream=object(), substeps=25, num_envs=2, steps=0)
            env._physics = SimpleNamespace(_control_graph=None, _substep_fusion=None)
            fusion = SimpleNamespace(manifest={'config': {'fixture': True}})

            def reset(**kwargs):
                env.steps = 0
                env._physics._control_graph = None

            def step(action):
                env.steps += 1
                return {}, np.zeros(2), np.zeros(2, bool), []

            def prepare(*args, **kwargs):
                self.assertFalse(kwargs['allow_compile'])
                env._physics._control_graph = object()
                return fusion

            env.reset, env.step = Mock(side_effect=reset), Mock(side_effect=step)
            env.prepare_substep_fusion, env.close = Mock(side_effect=prepare), Mock()
            library = ModuleType('libero.libero')
            library.get_libero_path = lambda _: str(root)
            environments = ModuleType('libero.libero.envs')
            environments.GpuBatchEnv = Mock(return_value=env)
            warp = ModuleType('warp')
            warp.synchronize_stream = lambda _: None
            compilation = ModuleType('libero.gpu.host.build.compilation')

            @contextmanager
            def scope(allow):
                self.assertFalse(allow)
                yield {'events': []}

            compilation.compilation_scope = scope
            compilation.install_warp_guard = lambda: None

            def status(_):
                return dict(integration_count=[env.steps * 25] * 2, state_epoch=[env.steps * 25] * 2,
                            first_error=[0, 0], reset_events=[0, 0],
                            newton_attempts=[env.steps * 75] * 2, position_math=[env.steps * 25] * 2)

            with patch.dict(sys.modules, {'libero.libero': library, 'libero.libero.envs': environments,
                                          'warp': warp, 'libero.gpu.host.build.compilation': compilation}), \
                    patch.object(runner, 'gpu_status', side_effect=status), \
                    patch.object(runner, 'gpu_endpoint', return_value=state(endpoint)):
                code = runner.run_task(args, task)
            self.assertEqual(code, expected_code)
            self.assertEqual(env.step.call_count, expected_steps)
            env.prepare_substep_fusion.assert_called_once()
            env.close.assert_called_once()
            self.assertFalse(environments.GpuBatchEnv.call_args.kwargs['allow_compile'])
            receipt = json.loads((args.output / (stem + '.json')).read_text())
            self.assertEqual(receipt['performance_measured'], expected_code == 0)
            if expected_code == 0:
                self.assertEqual(receipt['status'], 'qualified')
                for stage in ('correctness', 'performance'):
                    self.assertTrue(all(c['limit'] == 3e-7 for c in receipt['runs'][stage]['checks']))
            else:
                self.assertEqual(receipt['status'], 'regression')
                self.assertNotIn('performance', receipt['runs'])

    def test_pass_reuses_one_environment_and_graph_for_one_timing(self):
        self.exercise(2e-7, 50, 0)

    def test_failed_correctness_never_starts_timing(self):
        self.exercise(1e-5, 25, 2)

    def test_summary_aggregates_total_work_and_separates_reset_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cpu, gpu = root / 'cpu', root / 'gpu'
            cpu.mkdir()
            gpu.mkdir()
            for index, gpu_s in enumerate((.1, .4)):
                stem = f'libero_10__{index:03d}_n2_varying'
                base = dict(task=str(index), suite='libero_10', task_id=index, worlds=2,
                            case='varying', passed=True, performance_measured=True,
                            input=dict(signature=str(index), steps=20))
                for folder, elapsed, status, stage in [(cpu, .2, 'recorded_cpu', 'endpoint'),
                                                       (gpu, gpu_s, 'qualified', 'performance')]:
                    stats = dict(elapsed_s=elapsed, throughput=40 / elapsed,
                                 mean_ms=elapsed * 50, std_ms=0)
                    save_json(folder / (stem + '.json'), dict(base, status=status, runs={stage: stats}))
                np.savez(cpu / (stem + '.npz'), time=np.ones(2))
                np.savez(gpu / (stem + '-performance.npz'), time=np.ones(2) * (1 if index == 0 else 2))
            result = summarize(cpu, gpu, root / 'summary')
            self.assertAlmostEqual(result['all_tasks']['gpu_speedup'], .8)
            self.assertEqual(result['time_event_consistent']['tasks'], 1)
            self.assertEqual(result['time_event_consistent']['gpu_speedup'], 2.)


if __name__ == '__main__':
    unittest.main()
