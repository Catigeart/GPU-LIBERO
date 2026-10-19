"""Cold allocation, sharing and capture contracts without CUDA execution."""
import ast
from functools import partial
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from libero.gpu.host.adapters.environment.facade import GpuSimFacade, SharedModelMetadata
from libero.gpu.host.build import startup


GPU = Path(__file__).resolve().parents[1] / 'libero/libero/gpu'


class Boundary:
    physics_backend = 'gpu'
    def read_fields(self, names): return {}
    def write_fields(self, *args): pass
    def forward(self): pass
    def step(self): pass
    def close(self): pass
    def update_visual_parameters(self, values): pass


class SharedMetadataTests(unittest.TestCase):
    def metadata(self):
        return dict(nbody=2, body_name_to_id={'root': 1}, body_mass=np.array([0., 1.]),
                    tex_rgb=np.arange(64, dtype=np.uint8),
                    geom_rgba=np.ones((1, 4), dtype=np.float32),
                    native_write_schema={'arrays': {'body_mass': {}}})

    def test_one_snapshot_has_detached_immutable_arrays_and_private_world_views(self):
        original = self.metadata()
        packet = SharedModelMetadata(original)
        first, second = [GpuSimFacade(Boundary(), packet) for _ in range(2)]
        self.assertIs(first.model.tex_rgb, second.model.tex_rgb)
        original['tex_rgb'][:] = 0
        np.testing.assert_array_equal(first.model.tex_rgb, np.arange(64, dtype=np.uint8))
        with self.assertRaises(ValueError):
            first.model.tex_rgb.setflags(write=True)
        first_mass = first.model.body_mass
        second_mass = second.model.body_mass
        self.assertFalse(np.shares_memory(first_mass, second_mass))
        first_mass[1] = 9.
        self.assertEqual(second_mass[1], 1.)
        self.assertEqual(packet['body_mass'][1], 1.)
        self.assertEqual(first.model.body_name2id('root'), 1)

    def test_visual_update_changes_one_world_and_preserves_other_snapshot(self):
        packet = SharedModelMetadata(self.metadata())
        first, second = [GpuSimFacade(Boundary(), packet) for _ in range(2)]
        first.update_visual_parameters({'geom_rgba': np.zeros((1, 4))})
        np.testing.assert_array_equal(first.model.geom_rgba, 0.)
        np.testing.assert_array_equal(second.model.geom_rgba, 1.)
        self.assertIsNot(first.model._metadata, second.model._metadata)

    def test_untrusted_readonly_array_is_still_copied(self):
        metadata = self.metadata()
        metadata['tex_rgb'].setflags(write=False)
        facade = GpuSimFacade(Boundary(), metadata)
        metadata['tex_rgb'].setflags(write=True)
        metadata['tex_rgb'][:] = 0
        np.testing.assert_array_equal(facade.model.tex_rgb, np.arange(64, dtype=np.uint8))

    def test_adopting_replacement_does_not_write_through_shared_snapshot(self):
        packet = SharedModelMetadata(self.metadata())
        first, second = [GpuSimFacade(Boundary(), packet) for _ in range(2)]
        updated = self.metadata()
        updated['tex_rgb'][:] = 2
        fresh = GpuSimFacade(Boundary(), SharedModelMetadata(updated))
        first.close()
        first.adopt_runtime_from(fresh)
        np.testing.assert_array_equal(first.model.tex_rgb, 2)
        np.testing.assert_array_equal(second.model.tex_rgb, np.arange(64, dtype=np.uint8))


class StartupTimingTests(unittest.TestCase):
    def test_opt_in_nested_report_and_exception_restore(self):
        class Owner:
            @startup.profile_construction
            def __init__(self, *, profile_startup=False, fail=False):
                with startup.startup_phase('model'):
                    if fail:
                        raise RuntimeError('failure')
        owner = Owner(profile_startup=True)
        self.assertEqual(set(owner.startup_timings), {'constructor', 'constructor.model'})
        self.assertGreaterEqual(owner.startup_timings['constructor'],
                                owner.startup_timings['constructor.model'])
        with self.assertRaisesRegex(RuntimeError, 'failure'):
            Owner(profile_startup=True, fail=True)
        with patch.object(startup, 'perf_counter', side_effect=AssertionError('disabled clock')):
            self.assertEqual(Owner().startup_timings, {})
        with self.assertRaisesRegex(ValueError, 'profile_startup'):
            Owner(profile_startup=1)


def node(path, name):
    return next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)


class Array:
    def __init__(self, shape, **kwargs):
        self.shape = shape if isinstance(shape, tuple) else (shape,)
        self.dtype = kwargs.get('dtype')
        self.device = kwargs.get('device')


class NewtonStartupTests(unittest.TestCase):
    def test_public_native_profile_propagates_through_cold_factories(self):
        calls = []
        class Transform:
            dynamics = euler = inserted = 1
        def derive(function, replacements, transform):
            def prepare(*args, **kwargs):
                factories = replacements['_event_cold_factories']()
                factories[1](rows=30052)
                return SimpleNamespace(runtime=object(), dynamics=object(), post=object(), close=lambda: None)
            return prepare
        pipeline = GPU/'host/assembly/pipeline/factory.py'
        namespace = dict(Path=Path, hashlib=hashlib, partial=partial, GPU_ROOT=GPU,
            first=SimpleNamespace(_derive=derive), _ColdImports=Transform,
            make_fused_adapter=object(), _newton_factory=lambda **kw: calls.append(kw),
            recovery_event_factory=SimpleNamespace(prepare_recovery_event_physics=object()),
            _recovery_loop=object(), fused_workspace_type=object(),
            dynamics=SimpleNamespace(plan_catalog=lambda obj: {}), _validate_selection=lambda obj: True)
        exec(compile(ast.Module(body=[node(pipeline, 'prepare_fused_physics')], type_ignores=[]),
                     str(pipeline), 'exec'), namespace)
        mathdx = GPU/'host/assembly/mathdx/factory.py'
        outer = dict(previous=namespace['prepare_fused_physics'], Path=Path, GPU_ROOT=GPU,
                     hashlib=hashlib, __file__=str(mathdx))
        exec(compile(ast.Module(body=[node(mathdx, 'prepare_fused_physics')], type_ignores=[]),
                     str(mathdx), 'exec'), outer)
        for profile in ('fast', 'compatible'):
            outer['prepare_fused_physics'](diagnostic_profile=profile, defer_cuda_core=True)
        self.assertEqual(calls, [{'rows': 30052, 'diagnostic_profile': 'fast'},
                                 {'rows': 30052, 'diagnostic_profile': 'compatible'}])

    def test_fast_profile_reaches_initial_solver_without_allocating_history(self):
        solver_class = node(GPU/'host/runtime/newton/solver.py', 'FwdConstraintSolver')
        solver_class.body = [n for n in solver_class.body if
                             isinstance(n, ast.FunctionDef) and n.name == '__init__']
        allocations = []
        def allocate(shape, *args, **kwargs):
            array = Array(shape, **kwargs)
            allocations.append(array)
            return array
        warp = SimpleNamespace(zeros=allocate, full=allocate, float64='f64', int32='i32')
        scope = dict(wp=warp, _f64_1=lambda a: a, _i32_1=lambda a: a,
                     NewtonFwdWorkspace=object, TAPE_SLOT_CAP=256, TRACECAP=52, NV=43)
        exec(compile(ast.Module(body=[solver_class], type_ignores=[]), '<retained-solver>', 'exec'), scope)
        class Workspace:
            def __init__(self, nworld, rows, contacts, **options):
                self.nworld, self.rows = nworld, rows
                self.maxiter, self.device = options['maxiter'], options['device']
            def __getattr__(self, name): return Array((8, 43))
            def bind_topology(self, *args): pass
        class Adapter:
            def __init__(self, ws, *, rules, solver_factory):
                self.ws, self.factory = ws, solver_factory
            def bind_inputs(self, values): pass
            def bind_outputs(self, values): pass
            def seal(self): self.solver = self.factory(self.ws)
        fn = node(GPU/'host/runtime/newton/live_copy_api.py', 'prepare_live_copy_runtime')
        model = SimpleNamespace(nv=43, nM=153, opt=SimpleNamespace(iterations=100,
            disableflags=0, tolerance=1.e-8), stat=SimpleNamespace(meaninertia=1.),
            dof_Madr=np.arange(43), dof_parentid=np.arange(43), dof_simplenum=np.ones(43))
        scope.update(Path=Path, hashlib=hashlib, SimpleNamespace=SimpleNamespace, np=np,
            mujoco=SimpleNamespace(MjModel=SimpleNamespace(from_binary_path=lambda p: model)),
            _validate_native=lambda m: 1, _AdmittedModel=lambda a: SimpleNamespace(_validate=lambda: None),
            load_candidate=lambda *a: (SimpleNamespace(workspace=SimpleNamespace(NewtonFwdWorkspace=Workspace),
                solver=SimpleNamespace(FwdConstraintSolver=scope['FwdConstraintSolver'])), {}),
            parallel_tape_solver_type=lambda cls: cls, ColdNewtonRuntimeAdapter=Adapter,
            GPU_ROOT=GPU, __file__=str(GPU/'host/runtime/newton/live_copy_api.py'))
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<cold-newton>', 'exec'), scope)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.mjb'
            path.write_bytes(b'model')
            spec = SimpleNamespace(model_sha256=hashlib.sha256(b'model').hexdigest(),
                dimensions=SimpleNamespace(nv=43, nM=153), identity_key='identity')
            for profile, slots in [('fast', 0), ('compatible', 100), ('compatibility', 100)]:
                with self.subTest(profile=profile):
                    allocations.clear()
                    adapter = scope['prepare_live_copy_runtime'](model_path=path, model_spec=spec,
                        rows=30052, contacts=5000, nworld=8, build_root=Path(tmp),
                        live_inputs={}, diagnostic_profile=profile)
                    solver = adapter.solver
                    self.assertEqual(solver.tape_slots, slots)
                    tapes = [v for k, v in vars(solver).items() if k.startswith('tape_') and k != 'tape_slots']
                    self.assertEqual(len(tapes), 10)
                    self.assertTrue(all(a.shape[1] == slots for a in tapes))
                    if slots == 0:
                        self.assertEqual(sum(np.prod(a.shape) for a in tapes), 0)
                    else:
                        self.assertGreater(sum(np.prod(a.shape) for a in tapes), 90_000_000)


if __name__ == '__main__':
    unittest.main()
