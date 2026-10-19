"""Real original classes with synthetic static shells; no env constructors/GPU."""
import importlib.util
from pathlib import Path
import sys
from types import MethodType
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

    def test_instance_override_before_or_after_planning_uses_full_schema(self):
        for name in ('reward', '_post_action', '_post_process', '_check_success', '_eval_predicate'):
            shell = shell_fixture([('in', 'bowl', 'region')])
            plan = module.GoalReadPlan.from_shell(shell)
            setattr(shell, name, MethodType(lambda self, *args: None, shell))
            self.assertEqual(plan.fields_for(shell), module.FULL_GOAL_FIELDS)
            self.assertFalse(module.GoalReadPlan.from_shell(shell).qualified)

    def test_dispatcher_alias_and_registry_owner_changes_invalidate_plan(self):
        shell = shell_fixture([('in', 'bowl', 'region')])
        plan = module.GoalReadPlan.from_shell(shell)
        globals_ = shell._eval_predicate.__func__.__globals__
        with patch.dict(globals_, {'eval_predicate_fn': lambda *args: False}):
            self.assertEqual(plan.fields_for(shell), module.FULL_GOAL_FIELDS)
            self.assertFalse(module.GoalReadPlan.from_shell(shell).qualified)
        with patch.object(predicates, 'VALIDATE_PREDICATE_FN_DICT', dict(predicates.VALIDATE_PREDICATE_FN_DICT)):
            self.assertEqual(plan.fields_for(shell), module.FULL_GOAL_FIELDS)
        with patch.dict(predicates.VALIDATE_PREDICATE_FN_DICT, {'in': lambda *args: False}):
            self.assertEqual(plan.fields_for(shell), module.FULL_GOAL_FIELDS)
            self.assertFalse(module.GoalReadPlan.from_shell(shell).qualified)

    def test_function_code_defaults_state_graph_and_tracker_changes_invalidate(self):
        shell = shell_fixture([('in', 'bowl', 'region')], stove=True)
        plan = module.GoalReadPlan.from_shell(shell)
        function = shell._check_success.__func__
        saved_code, saved_defaults = function.__code__, function.__defaults__
        try:
            function.__code__ = (lambda self: False).__code__
            self.assertEqual(plan.fields_for(shell), module.FULL_GOAL_FIELDS)
        finally:
            function.__code__ = saved_code
        try:
            function.__defaults__ = (None,)
            self.assertEqual(plan.fields_for(shell), module.FULL_GOAL_FIELDS)
        finally:
            function.__defaults__ = saved_defaults
        shell.object_states_dict['bowl'] = ObjectState(shell, 'bowl')
        self.assertEqual(plan.fields_for(shell), module.FULL_GOAL_FIELDS)
        shell = shell_fixture([('in', 'bowl', 'region')], stove=True)
        plan = module.GoalReadPlan.from_shell(shell)
        shell.tracking_object_states_change[0].update_state = lambda: None
        self.assertEqual(plan.fields_for(shell), module.FULL_GOAL_FIELDS)
        shell = shell_fixture([('in', 'bowl', 'region')])
        plan = module.GoalReadPlan.from_shell(shell)
        state = shell.object_states_dict['bowl']
        state.__class__ = type('CustomState', (type(state),), {})
        self.assertEqual(plan.fields_for(shell), module.FULL_GOAL_FIELDS)

    def test_goal_changes_and_custom_task_class_are_not_qualified(self):
        shell = shell_fixture([('in', 'bowl', 'region')])
        plan = module.GoalReadPlan.from_shell(shell)
        shell.parsed_problem['goal_state'].append(['open', 'region'])
        self.assertEqual(plan.fields_for(shell), module.FULL_GOAL_FIELDS)
        shell.__class__ = type('CustomTask', (type(shell),), {})
        self.assertEqual(plan.fields_for(shell), module.FULL_GOAL_FIELDS)
        self.assertFalse(module.GoalReadPlan.from_shell(shell).qualified)
        shell = shell_fixture([('in', 'bowl', 'region')])
        plan = module.GoalReadPlan.from_shell(shell)
        shell.__class__ = type('CustomTask', (type(shell),), {})
        self.assertEqual(plan.fields_for(shell), module.FULL_GOAL_FIELDS)


if __name__ == '__main__':
    unittest.main()
