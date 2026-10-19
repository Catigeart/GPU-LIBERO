"""Cold topology admission for cooperative, single-RHS smooth solves."""
import unittest
import numpy as np

from libero.gpu.host.runtime.step.forward_plan import build_forward_mass_plan


class ForwardMassPlanTests(unittest.TestCase):
    def test_branch_ancestors_are_not_all_preceding_dofs(self):
        # Two children share root 0; DOF 3 belongs to the first child's chain.
        plan = build_forward_mass_plan([-1, 0, 0, 1, -1], [0, 1, 3, 5, 8], 9)
        nv, roots, width = plan[:3]
        self.assertEqual((nv, roots, width), (5, 2, 8))
        np.testing.assert_array_equal(plan[3:6], [0, 4, 5])
        np.testing.assert_array_equal(plan[6:11], [0, 1, 1, 2, 0])
        np.testing.assert_array_equal(plan[11:], [0, 1, 1, 3, 0])

    def test_width_covers_largest_component_without_task_id(self):
        for length, expected in [(6, 8), (9, 16), (21, 32), (32, 32), (33, 0), (70, 0)]:
            parent = np.arange(-1, length - 1, dtype=np.int32)
            madr = np.arange(length) * (np.arange(length) + 1) // 2
            plan = build_forward_mass_plan(parent, madr, length * (length + 1) // 2)
            self.assertEqual(plan[2], expected)
        # Large components still retain complete depths for the generic route.
        self.assertEqual(plan[5 + length - 1], length - 1)

    def test_interleaved_components_keep_general_solver(self):
        plan = build_forward_mass_plan([-1, -1, 0], [0, 1, 2], 4)
        self.assertEqual(plan[2], 0)

    def test_invalid_dependency_or_packed_width_is_rejected(self):
        for parent, madr, nm in [([-1, 2, 0], [0, 1, 3], 5),
                                  ([-1, 0], [0, 1], 2),
                                  ([0], [0], 1)]:
            with self.assertRaises(ValueError):
                build_forward_mass_plan(parent, madr, nm)


if __name__ == '__main__':
    unittest.main()
