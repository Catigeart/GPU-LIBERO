"""Reference budgets and task/timing contracts, without GPU execution."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from validation.common import actions, compare_endpoints, input_record, select_tasks, timing_summary


class ValidationToolTests(unittest.TestCase):
    def state(self, value):
        return {field: np.full((2, 1), value, np.float64)
                for field in ('qpos', 'qvel', 'qacc', 'qacc_warmstart', 'ctrl', 'time')}

    def test_frozen_budget_is_not_replaced_by_a_lucky_candidate(self):
        cpu, baseline = self.state(0), self.state(1e-7)
        first, timed = self.state(1e-12), self.state(2e-7)
        for candidate in (first, timed):
            checks = compare_endpoints(candidate, baseline, cpu, 2)
            self.assertTrue(all(c['passed'] for c in checks))
            self.assertTrue(all(c['limit'] == 3e-7 for c in checks))
        self.assertFalse(all(c['passed'] for c in compare_endpoints(self.state(1e-5), baseline, cpu, 2)))

    def test_envelope_includes_the_original_and_all_additional_old_samples(self):
        checks = compare_endpoints(self.state(2e-8), self.state(1.7e-8), self.state(0), 2,
                                   baseline_samples=[self.state(1e-11)])
        self.assertTrue(all(c['passed'] for c in checks))
        self.assertTrue(all(c['limit'] == 3 * 1.7e-8 for c in checks))
        checks = compare_endpoints(self.state(2e-8), self.state(1.7e-8), self.state(0), 2,
                                   baseline_samples=[{'qpos': np.zeros((2, 1))}])
        self.assertTrue(all(c['limit'] == 3 * 1.7e-8 for c in checks))

    def test_cache_miss_case_reuses_identical_physical_cpu_inputs(self):
        task = dict(key='task.bddl', model_sha256='model')
        bank = np.zeros((2, 3))
        kwargs = dict(worlds=2, warmup=5, steps=20, seed=0)
        a = input_record(task, bank, case='varying', **kwargs)
        b = input_record(task, bank, case='varying_cache_miss', **kwargs)
        self.assertEqual(a['signature'], b['signature'])
        self.assertNotEqual(a['case'], b['case'])

    def test_missing_shape_and_nonfinite_references_fail(self):
        ref = self.state(0)
        for broken in ({}, dict(ref, qpos=np.zeros((1, 1))), dict(ref, qpos=np.full((2, 1), np.nan))):
            with self.assertRaises(ValueError):
                compare_endpoints(ref, broken, ref, 2)
        changed = dict(ref, qpos=np.full((2, 1), np.nan))
        self.assertFalse(all(c['passed'] for c in compare_endpoints(changed, ref, ref, 2)))

    def test_world_control_step_throughput_and_selection(self):
        self.assertEqual(timing_summary([10.] * 20, .2, 2)['throughput'], 200.)
        rows = [dict(suite='libero_10', task_id=0), dict(suite='libero_spatial', task_id=1)]
        self.assertEqual(select_tasks(rows, ['libero_10:0', 'libero_10']), rows[:1])
        with self.assertRaises(ValueError):
            select_tasks(rows, ['libero_10:99'])
        np.testing.assert_array_equal(actions(3, 2)[0], actions(3, 1)[0])
        self.assertFalse(actions(3, 2, 'zero').any())


if __name__ == '__main__':
    unittest.main()
