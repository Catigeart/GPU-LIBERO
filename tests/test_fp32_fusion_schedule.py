"""CPU checks for single solve/accounting and optional-source isolation."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
GPU = ROOT / 'libero/libero/gpu'


def function(path, name, *, owner=None, namespace=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body if owner is None else next(
        node.body for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner)
    node = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)
    values = {} if namespace is None else dict(namespace)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), values)
    return values[name]


class FusionScheduleTests(unittest.TestCase):
    def test_region_handoff_and_fallback_account_exactly_once(self):
        enqueue = function(GPU / 'host/backends/cuda_fp32/runtime.py',
                           'enqueue_substep', owner='Runtime')
        for ready in (False, True):
            for counted in (False, True):
                with self.subTest(ready=ready, counted=counted):
                    events = []
                    physics = NS(step1=lambda: events.append(('step1',)),
                                 solve=lambda **kw: events.append(('solve', kw['unconstrained_ready'])))

                    def integrate():
                        events.append(('integrate',))
                        return counted

                    def control(policy):
                        events.append(('control', policy))
                        return ready

                    physics.integrate = integrate
                    runtime = NS(physics=physics, controller=NS(policy_values=('keep', 'new')),
                                 control_drive=NS(enqueue=control),
                                 count=lambda kind: events.append(('count', kind)))
                    enqueue(runtime, True)
                    self.assertEqual(events, [('step1',), ('control', 'new'),
                                              ('solve', ready), ('integrate',)] +
                                     ([] if counted else [('count', 0)]))

    def solve_fixture(self, flags=0):
        events = []

        def entry(name, method='enqueue_fused'):
            return NS(**{method: lambda: events.append(name)})

        state = NS(model=NS(opt=NS(disableflags=flags, run_rne_postconstraint=False),
                            sensor_rne_postconstraint=False),
                   data=NS(**{name: entry('clear-' + name, 'zero_')
                              for name in ('act_dot', 'qfrc_actuator', 'actuator_force')}),
                   actuation_solve=entry('unconstrained', 'enqueue'),
                   actuation=entry('actuation', 'enqueue_smooth'),
                   smooth=entry('smooth', 'enqueue'), smooth_solve=entry('backsolve'),
                   public_solve=entry('factor-solve'), newton=entry('newton', 'solve'),
                   sensors=entry('sensors', 'enqueue'))
        return state, events

    def test_ready_region_skips_only_the_unconstrained_work(self):
        solve = function(GPU / 'host/backends/cuda_fp32/physics.py', 'solve', owner='Physics',
                         namespace={'types': NS(DisableBit=NS(ACTUATION=1, SENSOR=2))})
        state, events = self.solve_fixture()
        solve(state, unconstrained_ready=True)
        self.assertEqual(events, ['newton', 'sensors'])
        events.clear()
        solve(state)
        self.assertEqual(events, ['unconstrained', 'newton', 'sensors'])
        events.clear()
        solve(state, public=True)
        self.assertEqual(events, ['actuation', 'factor-solve', 'newton', 'sensors'])
        with self.assertRaisesRegex(ValueError, 'Public forward'):
            solve(state, public=True, unconstrained_ready=True)

    def test_disabled_actuation_keeps_its_clear_and_standalone_path(self):
        solve = function(GPU / 'host/backends/cuda_fp32/physics.py', 'solve', owner='Physics',
                         namespace={'types': NS(DisableBit=NS(ACTUATION=1, SENSOR=2))})
        state, events = self.solve_fixture(flags=1)
        solve(state)
        self.assertEqual(events, ['clear-act_dot', 'clear-qfrc_actuator',
                                  'clear-actuator_force', 'smooth', 'backsolve',
                                  'newton', 'sensors'])

    def test_optional_fp32_sources_do_not_invalidate_native_but_remain_watched(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            gpu = repo / 'gpu'
            names = ('gpu/cuda/step/native.cu', 'gpu/cuda/fp32/region.cu',
                     'gpu/host/backends/cuda_fp32/runtime.py',
                     'gpu/cuda/runtime/runtime.cu', 'gpu/cuda/dynamics/mjwarp_passive_math.cuh',
                     'scripts/helper.py')
            for name in names:
                path = repo / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('// source\n')
            sources = function(GPU / 'host/build/aot.py', '_source_identity_files',
                               namespace={'GPU_ROOT': gpu, 'REPO_ROOT': repo})
            self.assertEqual(set(sources()), {names[0], names[-1]})
            self.assertEqual(set(sources(include_excluded=True)), set(names))


if __name__ == '__main__':
    unittest.main()
