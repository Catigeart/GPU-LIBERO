"""Cold controller packet schedule: world strides, tails and alias fallback."""
import ctypes as ct
import unittest
from libero.gpu.host.runtime.step.controller_copy_plan import build_packets, access_regions, parallel_gather_allowed
from libero.gpu.host.runtime.step.substep_types import SubstepArguments, ControllerChannels, DoubleChannel, IntChannel


class ControllerCopyPlanTests(unittest.TestCase):
    def channels(self):
        c = ControllerChannels()
        c.floating_count, c.integer_count = 2, 1
        c.floating[0] = DoubleChannel(0x10000, 0x20000, 7, 3)
        c.floating[1] = DoubleChannel(0x30000, 0x40000, 4, 1)
        c.integer[0] = IntChannel(0x50000, 0x60000)
        return c

    def test_packets_keep_world_stride_and_field_tails(self):
        c = self.channels()
        p, seed, commit = build_packets(c, 2)
        self.assertEqual((seed, commit), (4, 6))
        self.assertEqual([(x.published, x.width, x.words) for x in p[:seed]],
                         [(0x10000+8*k, 7, min(2, 7-k)) for k in range(0, 7, 2)])
        self.assertEqual([(x.published, x.width) for x in p[seed+4:]], [(0x30000, 4), (0x30010, 4)])
        self.assertEqual(len(build_packets(c, 8)[0]), len(p))

    def test_cross_world_and_exact_aliases_select_ordered_path(self):
        c = self.channels()
        c.floating[0].working = 0x10000+7*8
        self.assertFalse(build_packets(c, 2)[0])
        c.floating[0].working = c.floating[0].published
        self.assertFalse(build_packets(c, 2)[0])

    def test_integer_and_float_overlap_selects_ordered_path(self):
        c = self.channels()
        c.integer[0].working = c.floating[0].published+8
        self.assertFalse(build_packets(c, 2)[0])

    def test_protected_gather_and_protocol_spans(self):
        c = self.channels()
        self.assertFalse(build_packets(c, 2, [(0x20000+13*8, 0x20000+14*8)])[0])
        self.assertFalse(build_packets(c, 2, integer_protected=[(0x60000, 0x60008)])[0])
        self.assertTrue(build_packets(c, 2, [(0x70000, 0x70008)])[0])

    def test_nonparallel_field_keeps_ordered_path(self):
        c = self.channels()
        c.floating[1].parallel_copy = 0
        self.assertFalse(build_packets(c, 2)[0])

    def test_nested_ctypes_role_exclusions(self):
        a = SubstepArguments()
        a.controller_plan.needs = 0x10000
        a.controller_solve.cache_valid = 0x20000
        a.controller_solve.needs = 0x30000
        a.controller_commit.event = a.controller_publish.event = 0x40000
        self.assertEqual(len(access_regions(a, 2)), 5)
        self.assertEqual(access_regions(a, 2, skip_copy_flags=True, skip_event=True), [])

    def wiring(self, nv=39):
        a = SubstepArguments()
        pointer = 0x100000
        for name, _ in a._fields_:
            group = getattr(a, name)
            if not isinstance(group, ct.Structure):
                continue
            for field, kind in group._fields_:
                if kind is ct.c_void_p:
                    setattr(group, field, pointer)
                    pointer += 0x100000
        g, b, c, j, v, p, m, out, solve = (a.gather_gate, a.controller_bias, a.controller_gate,
            a.controller_inputs, a.end_effector_velocity, a.end_effector_pose,
            a.controller_commit, a.controller_publish, a.controller_solve)
        j.nv, j.nq, j.nm, j.nb, b.nv, p.ns = nv, nv+7, nv*nv, 2, nv, 2
        for field in ('state', 'source', 'epoch', 'valid', 'written'):
            setattr(b, field, getattr(g, field))
        b.first = v.first = m.first = c.first = g.error
        j.active = p.active = v.active = g.active
        v.status, v.vp, v.vo = j.out_status, j.vp, j.vo
        j.pos, j.rot = p.outp, p.outm
        b.bias, b.selected = j.qbias, j.bias
        c.pv, c.written = b.parent_valid, b.parent_written
        solve.active = m.active = c.active
        solve.error = m.error = c.error
        solve.policy = c.policy
        out.mask, out.event, out.valid = m.commit, m.event, m.valid
        out.parent = b.parent_written
        return a

    def test_general_count_wiring_and_distinct_first_error_owner(self):
        for nv in (21, 39, 57):
            a = self.wiring(nv)
            self.assertTrue(parallel_gather_allowed(a, 2, index_words=7))
            a.controller_gate.first = a.controller_commit.first = 0x70000000
            self.assertTrue(parallel_gather_allowed(a, 2, index_words=7))

    def test_snapshot_aliases_and_unknown_wiring_choose_ordered_path(self):
        for field in ('needs', 'derived_ranks', 'ctrl', 'derived_data'):
            a = self.wiring()
            a.controller_gate.first = a.controller_commit.first = getattr(a.controller_solve, field)
            self.assertFalse(parallel_gather_allowed(a, 2, index_words=7))
        a = self.wiring()
        a.controller_commit.event = a.controller_publish.event = a.controller_solve.derived_key
        self.assertFalse(parallel_gather_allowed(a, 2, index_words=7))
        a = self.wiring()
        a.controller_bias.epoch += 8
        self.assertFalse(parallel_gather_allowed(a, 2, index_words=7))

    def test_cooperative_write_alias_chooses_ordered_path(self):
        a = self.wiring()
        a.end_effector_velocity.target = a.controller_inputs.vp
        self.assertFalse(parallel_gather_allowed(a, 2, index_words=7))
        a = self.wiring()
        a.controller_inputs.jac_status = a.gather_gate.active
        self.assertFalse(parallel_gather_allowed(a, 2, index_words=7))

    def test_argument_abi(self):
        self.assertEqual(ct.sizeof(SubstepArguments), 6288)


if __name__ == '__main__':
    unittest.main()
