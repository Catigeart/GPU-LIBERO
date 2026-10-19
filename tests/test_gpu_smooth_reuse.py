"""Cold cache lifetime/alias contracts; no CUDA allocation or execution."""
from types import SimpleNamespace
import unittest

from libero.gpu.host.runtime.step.substep_binding import _validate_smooth_reuse
from libero.gpu.host.runtime.step.substep_types import SubstepArguments


class Parameters(SimpleNamespace):
    def __getattr__(self, name):
        return 0


class SmoothReuseTests(unittest.TestCase):
    def fixture(self, nv=39):
        a = SubstepArguments()
        a.state.nv = a.smooth_acceleration.nv = a.smooth_commit.width = nv
        a.fluid_passive.nbody = 3
        a.smooth_acceleration.nm = 212
        a.smooth_commit.pf, a.smooth_commit.pa = 0x10000, 0x20000
        a.smooth_acceleration.applied, a.smooth_acceleration.xfrc = 0x30000, 0x40000
        a.smooth_acceleration.scratch = 0x50000
        core = SimpleNamespace(params=Parameters(qacc=0x60000), owners=[])
        return a, core

    def test_disjoint_payloads_across_size_families(self):
        for nv in (21, 39, 57):
            with self.subTest(nv=nv):
                a, core = self.fixture(nv)
                self.assertTrue(_validate_smooth_reuse(a, core, 2)['passed'])

    def test_newton_writer_overlapping_second_world_is_rejected(self):
        a, core = self.fixture()
        core.params.canonical_qacc = a.smooth_commit.pa + 8*a.state.nv
        with self.assertRaisesRegex(ValueError, 'intervening writer'):
            _validate_smooth_reuse(a, core, 2)

    def test_scratch_cannot_overwrite_persistent_applied_force(self):
        a, core = self.fixture()
        a.smooth_acceleration.scratch = a.smooth_acceleration.applied + 8
        with self.assertRaisesRegex(ValueError, 'persistent applied-force'):
            _validate_smooth_reuse(a, core, 2)

    def test_force_and_acceleration_require_independent_publications(self):
        a, core = self.fixture()
        a.smooth_commit.pa = a.smooth_commit.pf
        with self.assertRaisesRegex(ValueError, 'publications overlap'):
            _validate_smooth_reuse(a, core, 2)

    def test_unknown_world_stride_is_rejected(self):
        a, core = self.fixture()
        a.smooth_commit.width += 1
        with self.assertRaisesRegex(ValueError, 'compact world-major'):
            _validate_smooth_reuse(a, core, 2)


if __name__ == '__main__':
    unittest.main()
