"""Reference budgets and task/timing contracts, without GPU execution."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from validation.common import actions, compare_endpoints, input_record, select_tasks, timing_summary
from validation.summary import summarize_precision


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

    def test_fp32_budget_uses_only_frozen_reference_scale(self):
        reference = self.state(0)
        kwargs = dict(scalar_epsilon=np.finfo(np.float32).eps)
        checks = compare_endpoints(self.state(1e-5), reference, reference, 2, **kwargs)
        self.assertTrue(all(c['passed'] for c in checks))
        self.assertTrue(all(c['limit'] == 1024 * np.finfo(np.float32).eps for c in checks))
        failed = compare_endpoints(self.state(1.), reference, reference, 2, **kwargs)
        self.assertFalse(any(c['passed'] for c in failed))
        self.assertEqual([c['limit'] for c in failed], [c['limit'] for c in checks])

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

    def test_precision_summary_keeps_failed_pairs_and_checks_build_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folders = {p: root / p for p in ('fp64', 'fp32')}
            for folder in folders.values():
                folder.mkdir()
            for task in (0, 1):
                for precision, elapsed in (('fp64', .1), ('fp32', .2)):
                    ok = not (task == 1 and precision == 'fp32')
                    receipt = dict(backend='gpu', precision=precision, suite='libero_10', task_id=task,
                        worlds=2, substeps=25, case='varying', status='qualified' if ok else 'regression',
                        passed=ok, performance_measured=ok, input=dict(signature=str(task), steps=20),
                        build_config=dict(real_bits=int(precision[2:]), fast_compile=True),
                        runs=dict(performance=dict(elapsed_s=elapsed, throughput=40/elapsed,
                            counters=dict(integration_count=[500, 500], reset_events=[0, 0]),
                            status=dict(first_error=[0, 0])) if ok else {}))
                    if not ok:
                        receipt.pop('input')  # Infrastructure can fail before input recording.
                        receipt['runs'] = {}
                    (folders[precision] / f'{task}.json').write_text(json.dumps(receipt))
            result = summarize_precision(folders['fp64'], folders['fp32'], root / 'out')
            self.assertEqual(result['aggregate']['matched_tasks'], 1)
            self.assertEqual(result['aggregate']['requested_tasks'], 2)
            self.assertEqual(result['aggregate']['fp32_over_current_fp64'], .5)
            self.assertFalse(result['rows'][1]['paired_qualified'])
            path = folders['fp32'] / '0.json'
            broken = json.loads(path.read_text())
            broken['build_config']['fast_compile'] = False
            path.write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError, 'build policies differ'):
                summarize_precision(folders['fp64'], folders['fp32'], root / 'bad')


if __name__ == '__main__':
    unittest.main()
