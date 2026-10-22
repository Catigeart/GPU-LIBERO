"""Proof boundary for independent mass-subtree reductions, without CUDA."""
import unittest
import numpy as np
from libero.gpu.model.topology import contiguous_subtree_sizes

class SubtreeTopologyTests(unittest.TestCase):
    def test_contiguous_branching_and_massless_root_shape(self):
        sizes=contiguous_subtree_sizes([0,0,1,1,0,4])
        np.testing.assert_array_equal(sizes,[6,3,1,1,2,1])
        self.assertFalse(sizes.flags.writeable)
        np.testing.assert_array_equal(contiguous_subtree_sizes([0]),[1])

    def test_preceding_parent_does_not_prove_contiguous_subtrees(self):
        self.assertIsNone(contiguous_subtree_sizes([0,0,0,1]))

    def test_invalid_parent_graph_rejected_before_device_use(self):
        for parent in ([],[-1],[0,2,0],[0,1],[0,0.5]):
            with self.subTest(parent=parent),self.assertRaises(ValueError):
                contiguous_subtree_sizes(parent)

if __name__=='__main__':unittest.main()
