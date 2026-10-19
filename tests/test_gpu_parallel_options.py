"""Cold admission for collision scheduling and state-owned prefix reuse."""
import dataclasses
import unittest
from libero.gpu.host.runtime.core.algorithm_config import AlgorithmOptions, normalize_algorithm_options


class ParallelOptionsTests(unittest.TestCase):
    def test_independent_ablation_modes(self):
        for schedule in ('fused','parallel'):
            for reuse in ('off','epoch'):
                with self.subTest(schedule=schedule,reuse=reuse):
                    options=normalize_algorithm_options(dict(collision_schedule=schedule,forward_reuse=reuse))
                    self.assertEqual(options.collision_schedule,schedule)
                    self.assertEqual(options.forward_reuse,reuse)
                    self.assertEqual(options.newton,AlgorithmOptions().newton)
                    self.assertEqual(options.post,AlgorithmOptions().post)

    def test_invalid_policy_rejected_before_device_construction(self):
        for options in (dict(collision_schedule='dynamic-jit'),dict(forward_reuse='skip-newton'),
                        dict(collision_workers=0),dict(collision_workers=True),dict(collision_workers=1.5)):
            with self.subTest(options=options),self.assertRaises((ValueError,TypeError)):
                normalize_algorithm_options(options)

    def test_policy_is_frozen_and_manifest_complete(self):
        options=AlgorithmOptions(collision_schedule='parallel',collision_workers=64,forward_reuse='epoch')
        self.assertEqual(options.as_dict()['collision_workers'],64)
        with self.assertRaises(dataclasses.FrozenInstanceError):options.forward_reuse='off'


if __name__=='__main__':unittest.main()
