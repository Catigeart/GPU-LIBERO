"""Independent acceleration sensitivity and force-balance screening; no GPU."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from validation.precision import sensitivity_floor,balance_residual
from validation.common import compare_endpoints,FIELDS
from validation.events import simulation_time, work_events
from validation.policy import prepare_policy

class PrecisionValidationTests(unittest.TestCase):
    def test_float_time_does_not_replace_integer_work(self):
        receipt = dict(worlds=2, substeps=25, input=dict(steps=20))
        stage = dict(counters=dict(integration_count=[500, 500], reset_events=[0, 0]),
                     status=dict(first_error=[0, 0]))
        self.assertTrue(simulation_time([1.25000894]*2, [1.25]*2, 'fp32')['passed'])
        self.assertFalse(simulation_time([1.25000894]*2, [1.25]*2, 'fp64')['passed'])
        self.assertTrue(work_events(receipt, stage)['passed'])
        stage['counters']['integration_count'][1] = 499
        self.assertFalse(work_events(receipt, stage)['passed'])
        self.assertIsNone(work_events(dict(worlds=2, input=dict(steps=20)), stage)['passed'])

    def test_policy_preserves_fp64_and_refuses_unidentified_fp32(self):
        reference = {f: np.zeros((1, 2)) for f in FIELDS}
        policy = prepare_policy('fp64', reference, 1)
        candidate = {f: v + 1e-11 for f, v in reference.items()}
        self.assertEqual(policy.evaluate(candidate, reference, reference)['checks'],
                         compare_endpoints(candidate, reference, reference, 1))
        with self.assertRaisesRegex(ValueError, 'matching MuJoCo'):
            prepare_policy('fp32', reference, 1)

    def test_single_solve_schedule_rejects_hidden_extra_forwards(self):
        receipt = dict(worlds=2, substeps=25, input=dict(steps=20),
                       build_config=dict(solves_per_substep=1))
        stage = dict(counters=dict(integration_count=[500, 500],
                     newton_attempts=[500, 500], reset_events=[0, 0]),
                     status=dict(first_error=[0, 0]))
        self.assertTrue(work_events(receipt, stage)['passed'])
        stage['counters']['newton_attempts'] = [1500, 1500]
        self.assertFalse(work_events(receipt, stage)['passed'])
        receipt['build_config']['solves_per_substep'] = 3
        self.assertTrue(work_events(receipt, stage)['passed'])

    def test_backward_error_rejects_a_wrong_acceleration(self):
        matrix=np.array([[2.,.25],[.25,1.]])
        a=np.array([.3,-.2]);smooth=matrix@a;constraint=np.zeros(2)
        self.assertTrue(balance_residual(matrix,a.astype(np.float32),smooth,constraint)['passed'])
        self.assertFalse(balance_residual(matrix,a+[.1,0],smooth,constraint)['passed'])

    def test_frozen_conditioning_and_singular_rejection(self):
        a=np.array([1.,2.])
        mild,_=sensitivity_floor(np.eye(2),a,'max')
        sensitive,condition=sensitivity_floor(np.diag([1.,1e-6]),a,'max')
        self.assertEqual(condition,1e6)
        self.assertGreater(sensitive,1000*mild)
        with self.assertRaises(ValueError):sensitivity_floor(np.diag([1.,1e-8]),a,'max')

    def test_acceleration_allowance_never_changes_state_envelopes(self):
        reference={f:np.zeros((1,2)) for f in FIELDS}
        candidate={f:v.copy() for f,v in reference.items()};candidate['qacc'][:]=.01
        floors={('qacc',0,'max'):.02,('qacc',0,'rms'):.02}
        self.assertTrue(all(c['passed'] for c in compare_endpoints(candidate,reference,reference,1,field_floors=floors)))
        candidate['qpos'][:]=.001
        self.assertFalse(all(c['passed'] for c in compare_endpoints(candidate,reference,reference,1,field_floors=floors)))

if __name__=='__main__':unittest.main()
