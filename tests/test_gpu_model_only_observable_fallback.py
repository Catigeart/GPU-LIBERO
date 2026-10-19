"""Real CPU observable methods over fake per-world facades; no Warp or physics.

Product factory/clone/helper/compatibility/adapter ASTs are used directly.
Only the real CPU observable methods and convert_quat body are loaded, avoiding
environment construction, dependency initialization and any JIT compilation.
"""
import ast
from collections import OrderedDict
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import runpy
import sys
from types import MethodType, ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GPU = ROOT/'libero/libero/gpu'
CPU = Path('/rjs2/lhw/libero-cpu/.venv/lib/python3.10/site-packages/robosuite')
CPU_DOMAIN = Path('/rjs2/lhw/libero-cpu/LIBERO/libero/libero/envs/bddl_base_domain.py')
ORIGINAL = runpy.run_path(str(CPU/'utils/observables.py'))


def function_node(path, name):
    return next(node for node in ast.walk(ast.parse(path.read_text()))
                if isinstance(node, ast.FunctionDef) and node.name == name)


def load_nodes(nodes, namespace):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<actual-source-ast>', 'exec'), namespace)
    return namespace


def factory_shim(shell):
    builder = function_node(GPU/'model/factory.py', 'build_task_model')
    nodes = [node for node in builder.body if
        (isinstance(node, ast.FunctionDef) and node.name == 'no_observables') or
        (isinstance(node, ast.Assign) and any(isinstance(target, ast.Attribute) and target.attr in
            ('_gpu_original_setup_observables', '_gpu_model_only_setup_observables_shim') for target in node.targets)) or
        (isinstance(node, ast.For) and isinstance(node.target, ast.Tuple) and
            [part.id for part in node.target.elts] == ['name', 'function'])]
    namespace = dict(shell=shell, OrderedDict=OrderedDict, MethodType=MethodType,
                     initialize_model=lambda self: None, no_renderer=lambda self: None,
                     no_reset=lambda self: None)
    load_nodes(nodes, namespace)


CLONE = load_nodes([function_node(GPU/'host/adapters/environment/batch_env.py', '_clone_shell')],
                   dict(deepcopy=deepcopy))['_clone_shell']
BIND = load_nodes([function_node(GPU/'host/adapters/environment/task_driver.py', '_bind_static_robot_references')],
                  {})['_bind_static_robot_references']
COMPAT = ModuleType('fallback_cold.observable_compatibility')
exec(compile((GPU/'host/adapters/observations/observable_compatibility.py').read_text(),
             '<actual-compatibility>', 'exec'), COMPAT.__dict__)
SCHEMA = runpy.run_path(str(GPU/'host/adapters/observations/schema.py'))


def reference_classes():
    namespace = dict(np=np, OrderedDict=OrderedDict, Observable=ORIGINAL['Observable'], sensor=ORIGINAL['sensor'])
    transforms = load_nodes([function_node(CPU/'utils/transform_utils.py', 'convert_quat')], dict(np=np))
    namespace['T'] = SimpleNamespace(convert_quat=transforms['convert_quat'])
    classes = [('ReferenceRobot', None, CPU/'robots/robot.py', 'setup_observables'),
               ('ReferenceArm', 'ReferenceRobot', CPU/'robots/single_arm.py', 'setup_observables'),
               ('ReferenceBase', None, CPU/'environments/base.py', '_setup_observables'),
               ('ReferenceRobotEnv', 'ReferenceBase', CPU/'environments/robot_env.py', '_setup_observables'),
               ('ReferenceDomain', 'ReferenceRobotEnv', CPU_DOMAIN, '_setup_observables')]
    for name, base, path, method in classes:
        node = ast.ClassDef(name=name, bases=[] if base is None else [ast.Name(id=base, ctx=ast.Load())],
                           keywords=[], body=[function_node(path, method)], decorator_list=[])
        load_nodes([ast.fix_missing_locations(node)], namespace)
    return namespace['ReferenceArm'], namespace['ReferenceDomain']


ARM, DOMAIN = reference_classes()
META = dict(joint_names=['arm'+str(i) for i in range(7)], qpos_indices=list(range(7)),
    qvel_indices=list(range(7)), joint_ids=list(range(7)), actuator_ids=list(range(7)),
    gripper=dict(joint_names=['finger0', 'finger1'], qpos_indices=[7, 8], qvel_indices=[7, 8], actuator_ids=[7]),
    eef_site_id=2, eef_cylinder_id=3)


def shell_fixture():
    shell = DOMAIN()
    robot = ARM()
    robot.has_gripper = True
    robot.robot_model = SimpleNamespace(naming_prefix='robot0_', dof=7, eef_name='eef')
    robot.control_freq = 20
    shell.robots = [robot]
    shell.sim = object()
    shell.control_freq = 20
    shell.use_camera_obs = False
    shell.use_object_obs = True
    shell.objects = []
    shell._obs_cache = {}
    shell.physics_backend = 'model_only'
    return shell


def facade_fixture(value):
    data = SimpleNamespace(qpos=np.arange(9, dtype=float)+value, qvel=np.arange(9, dtype=float)-value,
                           site_xpos=np.full((4, 3), value))
    data.get_body_xquat = lambda name: np.array([1., 0., 0., 0.])
    return SimpleNamespace(data=data, physics_backend='gpu')


class Array:
    def __init__(self, value):self.value=np.asarray(value).copy()
    def numpy(self):return self.value.copy()


class ModelOnlyFallback(unittest.TestCase):
    def test_factory_retains_resolved_instance_override_and_clone_owner_aliases(self):
        shell = shell_fixture()
        def instance_setup(self):return OrderedDict(instance=self)
        shell._setup_observables = MethodType(instance_setup, shell)
        factory_shim(shell)
        self.assertEqual(shell._setup_observables(), {})
        clone = CLONE(shell)
        self.assertIs(clone._gpu_original_setup_observables.__self__, clone)
        self.assertIs(clone._setup_observables, clone._gpu_model_only_setup_observables_shim)
        self.assertIs(clone._setup_observables.__self__, clone)
        clone.physics_backend = 'gpu'
        self.assertIs(COMPAT.HostObservableCompatibility(clone).observables['instance'], clone)

    def test_shim_unwrap_only_for_live_gpu_and_never_for_legitimate_empty_setup(self):
        shell = shell_fixture()
        factory_shim(shell)
        self.assertEqual(COMPAT.HostObservableCompatibility(shell).observables, {})
        shell.physics_backend = 'gpu'
        # A genuine replacement installed after factory construction wins.
        calls = []
        shell._setup_observables = lambda: calls.append('genuine empty') or {}
        self.assertEqual(COMPAT.HostObservableCompatibility(shell).observables, {})
        self.assertEqual(calls, ['genuine empty'])
        shell._setup_observables = shell._gpu_model_only_setup_observables_shim
        shell._gpu_original_setup_observables = None
        with self.assertRaisesRegex(TypeError, 'must be callable'):
            COMPAT.HostObservableCompatibility(shell)

    def test_static_helper_and_real_cpu_sensors_read_distinct_world_facades(self):
        template = shell_fixture()
        factory_shim(template)
        owners = []
        for world, shell in enumerate((template, CLONE(template))):
            shell.sim, shell.physics_backend = facade_fixture(10*(world+1)), 'gpu'
            BIND(shell.robots[0], shell.sim, META)
            compat = COMPAT.HostObservableCompatibility(shell)
            owners.append(compat)
            self.assertTrue(compat.observables['robot0_joint_pos'].is_active())
            for name, expected in [('robot0_joint_pos', shell.sim.data.qpos[:7]),
                                   ('robot0_joint_vel', shell.sim.data.qvel[:7]),
                                   ('robot0_gripper_qpos', shell.sim.data.qpos[7:9]),
                                   ('robot0_gripper_qvel', shell.sim.data.qvel[7:9]),
                                   ('robot0_eef_pos', shell.sim.data.site_xpos[2])]:
                np.testing.assert_array_equal(compat.observables[name]._sensor({}), expected)
            self.assertIs(shell.robots[0].sim, shell.sim)
            self.assertIsNot(shell.robots[0]._ref_joint_pos_indexes, META['qpos_indices'])
        self.assertIsNot(owners[0].observables['robot0_joint_pos'], owners[1].observables['robot0_joint_pos'])
        robot = SimpleNamespace()
        BIND(robot, template.sim, None)
        self.assertIs(robot.sim, template.sim)

    def test_actual_adapter_migrates_settings_clocks_and_custom_callbacks(self):
        node = next(node for node in ast.parse((GPU/'host/adapters/observations/adapter.py').read_text()).body
                    if isinstance(node, ast.ClassDef) and node.name == 'GpuObservationAdapter')
        namespace = dict(np=np, wp=SimpleNamespace(), nullcontext=nullcontext,
                         __name__='fallback_cold.adapter', __package__='fallback_cold')
        cls = load_nodes([node], namespace)['GpuObservationAdapter']
        ob = cls.__new__(cls)
        plan = SCHEMA['plan_from_metadata'](prefix='robot0_', joint_qpos_indices=range(7), joint_qvel_indices=range(7),
            gripper_qpos_indices=[7, 8], gripper_qvel_indices=[7, 8], eef_site_id=2, eef_body_id=1,
            object_body_ids=[], control_freq=20)
        ob.plan = replace(plan, specs=tuple(replace(spec, period=.25, enabled=False, active=False)
                         if spec.name=='robot0_joint_vel' else replace(spec, period=1.) for spec in plan.specs))
        ob.closed=False;ob.valid=True;ob.nworld=2;ob.device='fake';ob.scene_cache=None
        ob.host_environments={};ob.host_observables={};ob._last_sample=(.002,False)
        ob._host_sampling_required=False;ob._host_sampling_reasons=[];ob._sample_event_token=0;ob._sample_event_hook=None
        ob._pending_renderers=set();ob.raw_packer=SimpleNamespace(buffer=SimpleNamespace(device=SimpleNamespace(is_cuda=False)))
        low=SimpleNamespace(last_stream=None)
        low.observed=Array(np.arange(2*plan.width, dtype=float));low.cache=Array(np.arange(2*plan.width, dtype=float)+3)
        low.elapsed=Array(np.full((2,len(plan.specs)),.05));low.sampled=Array(np.ones((2,len(plan.specs)),dtype=np.int32))
        low.present=Array(np.ones((2,len(plan.specs)),dtype=np.int32))
        ob.lowdim=low
        template=shell_fixture();factory_shim(template)
        for world,shell in enumerate((template,CLONE(template))):
            shell.sim,shell.physics_backend=facade_fixture(world*10+10),'gpu'
            BIND(shell.robots[0],shell.sim,META);ob.host_environments[world]=shell
        modules={'fallback_cold':ModuleType('fallback_cold'),'fallback_cold.observable_compatibility':COMPAT,
                 'robosuite':SimpleNamespace(macros=SimpleNamespace(CONCATENATE_IMAGES=False))}
        with patch.dict(sys.modules,modules):
            ob._enable_host_observables()
            for world,compat in ob.host_observables.items():
                target=compat.observables['robot0_joint_pos']
                self.assertTrue(target.is_enabled() and target.is_active())
                self.assertEqual(target._sampling_timestep,1.)
                self.assertEqual(target._time_since_last_sample,.05)
                self.assertTrue(target._sampled)
                np.testing.assert_array_equal(target.obs,low.observed.numpy().reshape(2,-1)[world,:7])
                velocity=compat.observables['robot0_joint_vel']
                self.assertFalse(velocity.is_enabled() or velocity.is_active())
                self.assertEqual(velocity._sampling_timestep,.25)
            calls=[]
            @ORIGINAL['sensor'](modality='custom')
            def custom(cache):calls.append(id(cache));return np.array([1.])
            ob.add_observable(ORIGINAL['Observable']('custom',custom,sampling_rate=500))
            calls.clear()
            for _ in range(25):ob._after_gpu_sample(.002)
            self.assertEqual([calls.count(id(ob.host_observables[w].cache)) for w in range(2)],[25,25])
            for world,compat in ob.host_observables.items():
                target=compat.observables['robot0_joint_pos']
                self.assertAlmostEqual(target._time_since_last_sample,.1)
                self.assertTrue(target._sampled)
                self.assertIn('robot0_joint_pos',compat.pack())
                self.assertIn('custom',compat.pack())
                self.assertNotIn('robot0_joint_vel',compat.pack())
                self.assertEqual(compat.observables['robot0_joint_vel']._time_since_last_sample,.05)
            unselected=ob.host_observables[1]
            clocks={name:(value._time_since_last_sample,value._sampled,
                          np.asarray(value._current_observed_value).tobytes()) for name,value in unselected.observables.items()}
            cached={name:np.asarray(value).tobytes() for name,value in unselected.cache.items()}
            reset_masks=[]
            low.reset=lambda mask,stream:reset_masks.append(mask.numpy())
            ob.reset(Array([1,0]))
            np.testing.assert_array_equal(reset_masks[0],[1,0])
            for name,value in unselected.observables.items():
                self.assertEqual((value._time_since_last_sample,value._sampled,
                                  np.asarray(value._current_observed_value).tobytes()),clocks[name])
            self.assertEqual({name:np.asarray(value).tobytes() for name,value in unselected.cache.items()},cached)
            self.assertEqual(ob.host_observables[0].observables['robot0_joint_pos']._time_since_last_sample,0.)

    def test_binding_call_order_and_single_shared_assignment_implementation(self):
        driver = ast.parse((GPU/'host/adapters/environment/task_driver.py').read_text())
        install = next(node for node in ast.walk(driver) if isinstance(node, ast.FunctionDef) and node.name == '_install_shell')
        source = ast.unparse(install)
        self.assertLess(source.index('_bind_static_robot_references('), source.index('adapter = self.components.controller'))
        self.assertEqual((GPU/'host/adapters/environment/task_driver.py').read_text().count('robot._ref_joint_pos_indexes ='),1)
        batch = ast.unparse(function_node(GPU/'host/adapters/environment/batch_env.py', '_assemble'))
        self.assertIn('_bind_static_robot_references(robot, facade,',batch)
        self.assertLess(batch.index('_bind_static_robot_references('),batch.index('self.observer.bind_host_environment('))
        helper = ast.unparse(function_node(GPU/'host/adapters/environment/task_driver.py', '_bind_static_robot_references'))
        for forbidden in ('setup_references(', 'reset(', 'forward(', 'controller'):
            self.assertNotIn(forbidden,helper)


if __name__ == '__main__':
    unittest.main()
