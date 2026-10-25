"""Real original classes with synthetic static shells; no env constructors/GPU."""
import importlib.util
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
import runpy

import numpy as np
import unittest
from unittest.mock import patch

from libero.libero.envs.bddl_base_domain import TASK_MAPPING
from libero.libero.envs.base_object import OBJECTS_DICT
from libero.libero.envs.object_states.base_object_states import ObjectState, SiteObjectState
from libero.libero.envs.objects.site_object import SiteObject
from libero.libero.envs.objects.target_zones import TargetZone
import libero.libero.envs.predicates as predicates

SOURCE = Path(__file__).resolve().parents[1] / 'libero/libero/gpu/host/adapters/environment/goal_read_plan.py'
spec = importlib.util.spec_from_file_location('_goal_plan_tests', SOURCE)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
boundary = runpy.run_path(str(SOURCE.parents[1] / 'observations/boundary.py'))
boundary_predicates = boundary['boundary_predicates']


def shell_fixture(goals, *, stove=False, site_parent='basket'):
    task = TASK_MAPPING['libero_kitchen_tabletop_manipulation']
    shell = object.__new__(task)
    shell.objects_dict, shell.fixtures_dict, shell.object_sites_dict = {}, {}, {}
    shell.object_states_dict = {}
    shell.parsed_problem = {'goal_state': [list(atom) for atom in goals]}
    shell.tracking_object_states_change = []
    for name, category in [('bowl', 'akita_black_bowl'), ('plate', 'plate'),
                           ('basket', 'basket'), ('cabinet', 'wooden_cabinet')]:
        obj = object.__new__(OBJECTS_DICT[category])
        obj._name, obj._joints, obj._contact_geoms = name, ['joint'], ['geom']
        obj.object_properties = {'vis_site_names': {}, 'articulation': {}}
        shell.objects_dict[name] = obj
    if stove:
        obj = object.__new__(OBJECTS_DICT['flat_stove'])
        obj._name, obj._joints, obj._contact_geoms = 'stove', ['joint'], ['geom']
        obj.object_properties = {'vis_site_names': {'burner': ('stove_burner', False)}, 'articulation': {}}
        shell.fixtures_dict['stove'] = obj
    for name in [*shell.objects_dict, *shell.fixtures_dict]:
        shell.object_states_dict[name] = ObjectState(shell, name, name in shell.fixtures_dict)
    site = object.__new__(SiteObject if site_parent is not None else TargetZone)
    site.name, site.parent_name, site.joints = 'region', site_parent, ['cabinet_joint']
    shell.object_sites_dict['region'] = site
    shell.object_states_dict['region'] = SiteObjectState(shell, 'region', site_parent)
    if stove:
        shell.tracking_object_states_change = [shell.object_states_dict['stove']]
    return shell


class GoalReadPlanTests(unittest.TestCase):
    def test_seven_builtin_signatures_use_actual_original_bound_methods(self):
        cases = [
            ([('on', 'bowl', 'plate')], False, ('body_xpos', 'contact', 'ncon')),
            ([('on', 'bowl', 'plate')], True, ('qpos', 'body_xpos', 'contact', 'ncon')),
            ([('in', 'bowl', 'region')], False, ('body_xpos', 'site_xpos', 'site_xmat')),
            ([('in', 'bowl', 'region')], True, ('qpos', 'body_xpos', 'site_xpos', 'site_xmat')),
            ([('open', 'region')], False, ('qpos',)),
            ([('on', 'bowl', 'region')], False, ('body_xpos', 'site_xpos', 'site_xmat', 'contact', 'ncon')),
            ([('on', 'bowl', 'region')], True, ('qpos', 'body_xpos', 'site_xpos', 'site_xmat', 'contact', 'ncon')),
        ]
        for goals, stove, fields in cases:
            with self.subTest(goals=goals, stove=stove):
                shell = shell_fixture(goals, stove=stove,
                                      site_parent='cabinet' if goals[0][0] == 'open' else 'basket')
                plan = module.GoalReadPlan.from_shell(shell)
                self.assertTrue(plan.qualified, plan.reason)
                self.assertEqual(plan.fields_for(shell), fields)

    def test_workspace_site_on_and_site_in_never_need_contacts(self):
        for goal in [('on', 'bowl', 'region'), ('in', 'bowl', 'region')]:
            shell = shell_fixture([goal], site_parent=None)
            plan = module.GoalReadPlan.from_shell(shell)
            self.assertTrue(plan.qualified, plan.reason)
            self.assertNotIn('contact', plan.fields_for(shell))

    def test_unknown_descriptors_use_full_prefetch(self):
        shell = shell_fixture([('custom', 'bowl')])
        plan = module.GoalReadPlan.from_shell(shell)
        self.assertFalse(plan.qualified)
        self.assertEqual(plan.fields_for(shell), module.FULL_GOAL_FIELDS)
        self.assertEqual(plan.fields_for(object()), module.FULL_GOAL_FIELDS)

    def facade(self):
        values = {
            'qpos': np.array([0.5]), 'body_xpos': np.zeros((2, 3)),
            'body_xquat': np.array([[1., 0., 0., 0.], [1., 0., 0., 0.]]),
            'site_xpos': np.zeros((1, 3)), 'site_xmat': np.eye(3).reshape(1, 9),
            'contact': {'geom1': np.array([4, 7]), 'geom2': np.array([5, 8])},
            'ncon': np.array([1]),
        }
        reads, visuals = [], []
        def snapshot(names):
            reads.append(tuple(names))
            return {name: values[name] for name in names}
        model = SimpleNamespace(site_rgba=np.ones((1, 4)))
        return SimpleNamespace(model=model, snapshot_fields=snapshot,
            update_visual_parameters=lambda changes: visuals.append(changes)), reads, visuals

    def test_replaced_predicate_completes_reads_once_without_replaying_side_effects(self):
        shell = shell_fixture([('in', 'bowl', 'region')])
        plan = module.GoalReadPlan.from_shell(shell)
        facade, reads, visuals = self.facade()
        original_sim = shell.sim = object()
        effects = []
        def custom_predicate(source, target):
            effects.append('predicate')
            data = source.env.sim.data
            self.assertEqual(data.body_xquat[0, 0], 1.)
            self.assertEqual(data.body_xquat[0, 0], 1.)
            self.assertEqual(data.ncon, 1)
            self.assertEqual(data.contact[0].geom1, 4)
            source.env.sim.model.site_rgba[0, 3] = 0.
            return True
        with patch.dict(predicates.VALIDATE_PREDICATE_FN_DICT, {'in': custom_predicate}):
            with boundary_predicates(shell, facade, fields=plan.fields_for(shell)):
                self.assertTrue(shell._check_success())
        self.assertIs(shell.sim, original_sim)
        self.assertEqual(effects, ['predicate'])
        self.assertEqual(reads, [plan.fields, ('body_xquat',), ('contact', 'ncon')])
        self.assertEqual(len(visuals), 1)
        self.assertEqual(visuals[0]['site_rgba'][0, 3], 0.)

    def test_changed_goal_and_callback_can_read_more_than_prefetch(self):
        shell = shell_fixture([('in', 'bowl', 'region')])
        plan = module.GoalReadPlan.from_shell(shell)
        shell.parsed_problem['goal_state'] = [('open', 'region')]
        shell.object_states_dict['region'].is_open = lambda: shell.sim.data.qpos[0] > 0.
        facade, reads, _ = self.facade()
        shell.sim = object()
        with boundary_predicates(shell, facade, fields=plan.fields_for(shell)):
            self.assertTrue(shell._check_success())
        self.assertEqual(reads, [plan.fields, ('qpos',)])
        shell._check_success = MethodType(lambda self: self.sim.data.body_xquat[0, 0] == 1., shell)
        with boundary_predicates(shell, facade, fields=plan.fields_for(shell)):
            self.assertTrue(shell._check_success())
        self.assertEqual(reads[-2:], [plan.fields, ('body_xquat',)])

    def test_lazy_failure_restores_sim_and_publishes_prior_visual_changes(self):
        shell = shell_fixture([('open', 'region')])
        facade, _, visuals = self.facade()
        original_sim = shell.sim = object()
        load = facade.snapshot_fields
        def fail_contacts(names):
            if 'contact' in names:
                raise ValueError('contact transfer failed')
            return load(names)
        facade.snapshot_fields = fail_contacts
        with self.assertRaisesRegex(ValueError, 'contact transfer failed'):
            with boundary_predicates(shell, facade, fields=('qpos',)) as view:
                view.model.site_rgba[0, 3] = 0.
                view.data.contact
        self.assertIs(shell.sim, original_sim)
        self.assertEqual(visuals[0]['site_rgba'][0, 3], 0.)
        with self.assertRaises(AttributeError):
            view.data.contact

    def test_retained_view_and_unknown_fields_do_not_download_another_event(self):
        shell = shell_fixture([('open', 'region')])
        facade, reads, _ = self.facade()
        shell.sim = object()
        with boundary_predicates(shell, facade, fields=('qpos',)) as view:
            for name in ('qvel', 'unknown_field', '__array_interface__'):
                with self.assertRaises(AttributeError):
                    getattr(view.data, name)
        self.assertEqual(view.data.qpos[0], 0.5)
        with self.assertRaises(AttributeError):
            view.data.body_xpos
        self.assertEqual(reads, [('qpos',)])

    def test_contacts_validate_capacity_and_keep_published_count(self):
        shell = shell_fixture([('open', 'region')])
        facade, reads, _ = self.facade()
        shell.sim = object()
        with boundary_predicates(shell, facade, fields=('contact',)) as view:
            self.assertEqual(view.data.ncon, 1)
            self.assertEqual(view.data.contact[1].geom2, 8)
        self.assertEqual(reads, [('contact', 'ncon')])
        facade.snapshot_fields = lambda names: {'contact': {'geom1': [1], 'geom2': [2]}, 'ncon': 2}
        with self.assertRaisesRegex(ValueError, 'contact count exceeds'):
            with boundary_predicates(shell, facade):
                self.fail('invalid contact snapshot must be rejected')


if __name__ == '__main__':
    unittest.main()
