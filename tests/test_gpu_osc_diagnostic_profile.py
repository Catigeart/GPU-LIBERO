"""OSC cold diagnostic selection with injected plans; no GPU runtime imports."""
import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import types
import unittest
from unittest.mock import patch


GPU = Path(__file__).resolve().parents[1] / 'libero/libero/gpu'
SOURCE = GPU / 'host/bindings/osc.py'
REFERENCE = GPU / 'reference/control/osc/controller.py'


class Array:
    next_ptr = 1

    def __init__(self, shape, **kwargs):
        self.shape = shape if isinstance(shape, tuple) else (shape,)
        self.ptr = Array.next_ptr
        Array.next_ptr += 1


class Adapter:
    pass


class OSCDiagnosticProfileTests(unittest.TestCase):
    def setUp(self):
        reference = ast.parse(REFERENCE.read_text())
        workspace = next(n for n in reference.body
                         if isinstance(n, ast.ClassDef) and n.name == 'PandaOSCWorkspace')
        widths = ast.literal_eval(next(n.value for n in workspace.body
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'WIDTHS'
                                                for t in n.targets)))
        functions = {n.name: n for n in reference.body if isinstance(n, ast.FunctionDef)}
        self.control_names = tuple(a.arg for a in functions['fused_osc_control'].args.args)
        refresh_names = tuple(a.arg for a in functions['refresh_cache'].args.args)
        self.diagnostic_index = self.control_names.index('emit_diagnostics')
        self.calls, self.plans = [], {}
        device = SimpleNamespace(arch=80)
        c = SimpleNamespace(nworld=8, device=device, diagnostics=1, uncouple=1)
        for name, width in widths.items():
            setattr(c, name, Array((8, width)))
        for name in ('active', 'policy_step', 'needs_update', 'cache_valid', 'error', 'quat_valid'):
            setattr(c, name, Array(8))
        c.ranks, c.quat_cast_wxyz = Array((8, 3)), Array((8, 4))
        owner = Adapter()
        owner.controller, owner.capacity, owner.device, owner.stream = c, 8, device, object()
        owner.public = {name: Array((8, width)) for name, width in widths.items()
                        if not name.startswith('phys_') and name not in
                        ('action', 'q0', 'gains', 'ctrl_range')}
        owner.public_int = {name: Array(8) for name in ('cache_valid', 'needs_update')}
        for name in ('active', 'first_error', 'call_mask', 'refresh_mask', 'commit_mask',
                     'event', 'valid', 'written', 'consumed_parent'):
            setattr(owner, name, Array(8))
        owner.parents = SimpleNamespace(valid=Array(8), written=Array(8), expected=Array(8))
        owner._world_alias_fallback = False
        owner._config = object()  # A sealed reference configuration must stay untouched.
        owner.validate = lambda stream: self.calls.append('validate')
        owner.forward_refresh = lambda *args: self.calls.append('forward')
        self.dynamics = SimpleNamespace(
            gather_controller_inputs=lambda *args: self.calls.append('gather'),
            gather_live_bias=lambda *args: self.calls.append('bias'))
        self.owner, self.controller = owner, c

        def make(name, stages, controller, root):
            plan = SimpleNamespace(stages=stages, manifest={'name': name},
                                   launch=lambda stream: self.calls.append(name))
            self.plans[name] = plan
            return plan

        warp = SimpleNamespace(array=Array, zeros=Array, empty=Array,
                               int32='int32', float64='float64',
                               copy=lambda target, source, **kwargs:
                                   self.calls.append(('copy', target, source)))
        namespace = dict(CONTROL=self.control_names, REFRESH=refresh_names,
                         FusedRuntimeControllerAdapter=Adapter, _make=make,
                         scalar=SimpleNamespace(wp_real='float64', bits=64),
                         wp=warp, Path=Path, types=types,
                         __package__='libero.gpu.host.bindings')
        nodes = [n for n in ast.parse(SOURCE.read_text()).body
                 if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and
                 n.name in ('control_args', 'refresh_args', '_Variants', 'install')]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
        self.install = namespace['install']
        self.control_args = namespace['control_args']
        self.modules = {
            'libero.gpu.host.bindings.osc_gather': SimpleNamespace(install=lambda *a, **kw: {}),
            'libero.gpu.host.assembly.compose.backup':
                SimpleNamespace(_overlap=lambda a, b: a.ptr == b.ptr),
        }

    def prepare(self, *, debug=False, dataflow='fast', diagnostics='fast'):
        bundle = SimpleNamespace(controller=self.owner, dynamics=self.dynamics, debug=debug)
        with patch.dict(sys.modules, self.modules):
            return self.install(bundle, Path('/unused'), dataflow_profile=dataflow,
                                diagnostic_profile=diagnostics)

    def run_operands(self, policy):
        stages = self.plans[f'osc_run_policy{policy}'].stages
        return next(args for name, args, mode in stages
                    if name in ('osc_cuda::control', 'osc_cuda::control_fast', 'osc_goal_cuda::control'))

    def test_formal_fast_passes_zero_to_both_policy_plans_without_changing_workspace(self):
        config = self.owner._config
        original = self.control_args(self.controller)
        report = self.prepare()
        self.assertEqual(report['emit_diagnostics'], 0)
        self.assertEqual(report['workspace_diagnostics'], 1)
        self.assertEqual(report['effective_diagnostic_profile'], 'fast')
        self.assertEqual(report['held_diagnostics'], ['nullspace_matrix'])
        self.assertEqual(self.controller.diagnostics, 1)
        self.assertIs(self.owner._config, config)
        for policy in (0, 1):
            with self.subTest(policy=policy):
                args = self.run_operands(policy)
                self.assertEqual(args[self.diagnostic_index], 0)
                for index, value in enumerate(original):
                    if index != self.diagnostic_index:
                        self.assertIs(args[index], value)

    def test_debug_and_compatibility_keep_optional_output_enabled(self):
        for options in ({'debug': True}, {'diagnostics': 'compatibility'},
                        {'dataflow': 'compatible'}):
            with self.subTest(options=options):
                report = self.prepare(**options)
                self.assertEqual(report['emit_diagnostics'], 1)
                self.assertEqual(report['held_diagnostics'], [])
                for policy in (0, 1):
                    self.assertEqual(self.run_operands(policy)[self.diagnostic_index], 1)

    def test_alias_fallback_preserves_compatible_output_selection(self):
        self.controller.phys_mass = self.controller.mass
        report = self.prepare()
        self.assertTrue(report['alias_fallback'])
        self.assertEqual(report['effective_dataflow_profile'], 'compatible')
        self.assertEqual(report['emit_diagnostics'], 1)
        self.assertEqual(self.run_operands(0)[self.diagnostic_index], 1)

    def test_general_goals_fallback_keeps_diagnostics_and_goal_operands(self):
        c = self.controller
        c.general_goals = True
        c.goal_effective_pos, c.goal_ori_error = Array((8, 3)), Array((8, 3))
        c.goal_core_policy = Array(8)
        self.modules['libero.gpu.host.runtime.control.general_osc'] = SimpleNamespace(
            goal_stages=lambda controller: [],
            reset_goal_stage=lambda controller, mask: ('goal_reset', [mask], 'lane0'))
        report = self.prepare()
        self.assertEqual(report['effective_dataflow_profile'], 'compatible')
        self.assertEqual(report['emit_diagnostics'], 1)
        for policy in (0, 1):
            args = self.run_operands(policy)
            self.assertEqual(args[self.diagnostic_index], 1)
            self.assertIs(args[self.control_names.index('goal_pos')], c.goal_effective_pos)
            self.assertIs(args[self.control_names.index('policy_step')], c.goal_core_policy)
            self.assertIs(args[-1], c.goal_ori_error)

    def test_explicit_workspace_diagnostic_disable_is_preserved(self):
        self.controller.diagnostics = 0
        for debug in (False, True):
            with self.subTest(debug=debug):
                report = self.prepare(debug=debug, diagnostics='compatibility')
                self.assertEqual(report['workspace_diagnostics'], 0)
                self.assertEqual(report['emit_diagnostics'], 0)
        self.assertEqual(self.control_args(self.controller)[self.diagnostic_index], 0)

    def test_held_field_keeps_existing_seed_success_commit_and_publication_order(self):
        public = dict(self.owner.public)
        self.prepare()
        c = self.controller
        held_seed = ('osc_cuda::seed_double',
                     [public['nullspace_matrix'], c.nullspace_matrix, 49], 'lane0')
        self.assertIn(held_seed, self.plans['osc_seed'].stages)
        self.assertEqual(self.owner.public, public)
        self.assertEqual(len(self.plans['osc_seed'].stages), len(public)+2)
        for policy in (0, 1):
            stages = self.plans[f'osc_run_policy{policy}'].stages
            held_commit = ('osc_cuda::copy_double',
                           [self.owner.commit_mask, c.nullspace_matrix,
                            public['nullspace_matrix'], 49], 'lane0')
            self.assertIn(held_commit, stages)
            self.assertLess(next(i for i, stage in enumerate(stages)
                                 if stage[0] == 'osc_cuda::commit_mask'), stages.index(held_commit))
            self.assertEqual(stages[-1][0], 'osc_cuda::publish')
        self.owner.enqueue(policy_step=True)
        self.assertEqual(self.calls, ['validate', 'osc_plan0', 'forward', 'gather', 'bias',
            'osc_seed', ('copy', c.active, self.owner.call_mask),
            ('copy', c.needs_update, self.owner.refresh_mask), 'osc_run_policy1'])


if __name__ == '__main__':
    unittest.main()
