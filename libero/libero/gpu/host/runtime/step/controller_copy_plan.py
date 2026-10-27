"""Cold copy scheduling and overlap proof for the fixed native controller."""
import libero.gpu.precision as scalar
from .substep_types import ControllerCopyPacket


def access_regions(a, worlds, *, index_words=0, skip_copy_flags=False, skip_event=False):
    """Exact byte spans touched during gather/gating, including readonly data."""
    g, p, v, b = a.controller_inputs, a.end_effector_pose, a.end_effector_velocity, a.controller_bias
    regions = []

    def add(owner, field, words, itemsize=scalar.itemsize):
        address = getattr(owner, field)
        if address and words:
            regions.append((int(address), int(address) + int(words) * itemsize))

    for field, width in [('cdof', g.nv*6), ('subtree', g.nb*3), ('pos', 3), ('rot', 9),
                         ('qpos', g.nq), ('qvel', g.nv), ('qm', g.nm), ('qbias', g.nv),
                         ('jp', 3*g.nv), ('jr', 3*g.nv), ('ep', 3), ('er', 9),
                         ('vp', 3), ('vo', 3), ('q', 7), ('qd', 7), ('J', 42), ('M', 49), ('bias', 7)]:
        add(g, field, worlds*width)
    for field, words in [('ancestor', g.nb*g.nv), ('qi', 7), ('vi', 7),
                         ('sparse', g.nv*g.nv), ('body_dof_offsets', g.nb+1),
                         ('body_dof_indices', index_words)]:
        add(g, field, words, 4)
    add(p, 'pos', worlds*p.ns*3)
    add(p, 'mat', worlds*p.ns*9)
    add(p, 'outp', worlds*3)
    add(p, 'outm', worlds*9)
    add(v, 'vp', worlds*3)
    add(v, 'vo', worlds*3)
    add(v, 'target', worlds*6)
    add(b, 'bias', worlds*b.nv)
    add(b, 'selected', worlds*7)
    add(b, 'indices', 7, 4)
    # Publication metadata must also be independent of the floating payload.
    specifications = [
        ('gather', a.gather_gate, 'mask error valid active', 'state source epoch written'),
        ('pose', p, 'active', ''), ('velocity', v, 'active status first', ''),
        ('bias', b, 'mask first valid parent_valid', 'state source epoch written parent_written'),
        ('inputs', g, 'active jac_status adapter_active adapter_status out_status', ''),
        ('plan', a.controller_plan, 'active first needs refresh call', ''),
        ('gate', a.controller_gate, 'active pv first error policy', 'written expected'),
        ('commit', a.controller_commit, 'active error first commit valid', 'event'),
        ('publish', a.controller_publish, 'mask valid', 'event parent written consumed'),
        ('solve', a.controller_solve, 'active error ranks policy needs quat_valid derived_valid derived_ranks cache_hits cache_valid', ''),
    ]
    for role, owner, int_fields, long_fields in specifications:
        for field in int_fields.split():
            if skip_copy_flags and ((role == 'plan' and field == 'needs') or
                                    (role == 'solve' and field in ('needs', 'cache_valid'))):
                continue
            add(owner, field, worlds*(3 if field in ('ranks', 'derived_ranks') else 1), 4)
        for field in long_fields.split():
            if skip_event and role in ('commit', 'publish') and field == 'event':
                continue
            add(owner, field, worlds, 8)
    return regions


def build_packets(channels, worlds, protected=(), integer_protected=()):
    """Return seed packets, then commit packets; an empty plan selects fallback.

    Proofs cover all worlds. Each field keeps its original world stride and
    packet tails; no packet crosses a field boundary. Plan size is independent
    of the number of worlds.
    """
    floating = list(channels.floating[:channels.floating_count])
    if worlds < 1 or not floating or any(not (f.parallel_copy & 1) for f in floating):
        return [], 0, 0
    spans = []
    for f in floating:
        if f.width < 1 or not f.published or not f.working:
            return [], 0, 0
        for pointer in (f.published, f.working):
            span = (int(pointer), int(pointer) + worlds*f.width*scalar.itemsize)
            if any(span[0] < end and start < span[1] for start, end in protected):
                return [], 0, 0
            spans.append(span)
    # Exact aliases also use the old path: it retains transactional visibility
    # when a public view is intentionally the working view.
    for f in channels.integer[:channels.integer_count]:
        for p in (f.published, f.working):
            if p:
                span = (int(p), int(p) + worlds*4)
                if any(span[0] < end and start < span[1] for start, end in integer_protected):
                    return [], 0, 0
                spans.append(span)
    ordered = sorted(spans)
    if any(end > next_start for (_, end), (next_start, _) in zip(ordered, ordered[1:])):
        return [], 0, 0

    def packets(seed):
        result = []
        for f in floating:
            if seed and not (f.parallel_copy & 2):
                continue
            result.extend(ControllerCopyPacket(f.published+scalar.itemsize*z, f.working+scalar.itemsize*z,
                                               f.width, min(2, f.width-z))
                          for z in range(0, f.width, 2))
        return result

    seed, commit = packets(True), packets(False)
    return seed+commit, len(seed), len(commit)


def parallel_gather_allowed(a, worlds, *, index_words=0):
    """Admit the fixed producer wiring and independent cooperative writes."""
    g, b, c, s = a.gather_gate, a.controller_bias, a.controller_gate, a.controller_solve
    j, p, v, m, out = a.controller_inputs, a.end_effector_pose, a.end_effector_velocity, a.controller_commit, a.controller_publish
    equal = [(g.state, b.state), (g.source, b.source), (g.epoch, b.epoch),
             (g.valid, b.valid), (g.written, b.written), (g.error, b.first),
             (g.active, j.active), (g.active, p.active), (g.active, v.active),
             (g.error, v.first), (v.status, j.out_status),
             (p.outp, j.pos), (p.outm, j.rot), (v.vp, j.vp), (v.vo, j.vo),
             (b.bias, j.qbias),
             (c.pv, b.parent_valid), (c.written, b.parent_written),
             (c.active, s.active), (c.active, m.active), (c.error, s.error), (c.error, m.error),
             (c.policy, s.policy), (c.first, m.first),
             (m.commit, out.mask), (m.event, out.event), (m.valid, out.valid)]
    if b.nv != j.nv or any(not x or x != y for x, y in equal):
        return False

    def span(owner, name, width, itemsize=scalar.itemsize):
        pointer = getattr(owner, name)
        return (int(pointer or 0), int(pointer or 0) + worlds*width*itemsize)

    def overlap(x, y):
        return x[0] < y[1] and y[0] < x[1]

    outputs = [span(j, k, n) for k, n in [('ep', 3), ('er', 9), ('vp', 3), ('vo', 3),
               ('q', 7), ('qd', 7), ('J', 42), ('M', 49), ('bias', 7), ('jp', 3*j.nv), ('jr', 3*j.nv)]]
    sources = [span(j, k, n) for k, n in [('cdof', j.nv*6), ('subtree', j.nb*3),
               ('pos', 3), ('rot', 9), ('qpos', j.nq), ('qvel', j.nv), ('qm', j.nm), ('qbias', j.nv)]]
    if any(not x[0] for x in outputs+sources):
        return False
    if any(overlap(x, y) for i, x in enumerate(outputs) for y in outputs[i+1:]):
        return False
    if any(overlap(x, y) for x in outputs for y in sources):
        return False
    if any(overlap(span(v, 'target', 6), x) for x in outputs+sources):
        return False
    site_outputs = [span(p, 'outp', 3), span(p, 'outm', 9)]
    if overlap(*site_outputs) or any(overlap(x, span(p, field, width*p.ns))
                                   for x in site_outputs for field, width in [('pos', 3), ('mat', 9)]):
        return False
    selected = span(b, 'selected', 7)
    if overlap(selected, span(b, 'bias', b.nv)) or overlap(selected, span(v, 'target', 6)):
        return False
    if any(overlap(selected, x) for x in outputs if x != span(j, 'bias', 7)):
        return False
    # Private metadata has one writer per role. The dynamics first error may
    # also be the controller first error: gather completes before solve/commit.
    metadata = [span(owner, field, width, size) for owner, fields, size in [
        (g, [('mask', 1), ('error', 1), ('active', 1)], 4),
        (j, [('jac_status', 1), ('adapter_active', 1), ('adapter_status', 1), ('out_status', 1)], 4),
        (b, [('mask', 1), ('parent_valid', 1)], 4), (b, [('parent_written', 1)], 8),
        (s, [('active', 1), ('error', 1), ('policy', 1), ('needs', 1), ('quat_valid', 1),
             ('derived_valid', 1), ('derived_ranks', 3), ('cache_hits', 1), ('cache_valid', 1), ('ranks', 3)], 4),
        (m, [('commit', 1), ('valid', 1)], 4), (m, [('event', 1)], 8),
        (out, [('written', 1), ('consumed', 1)], 8)] for field, width in fields]
    if span(m, 'first', 1, 4) != span(g, 'error', 1, 4):
        metadata.append(span(m, 'first', 1, 4))
    metadata = [x for x in metadata if x[0]]
    if any(overlap(x, y) for i, x in enumerate(metadata) for y in metadata[i+1:]):
        return False
    readonly = [span(g, field, 1, size) for field, size in
                [('state', 8), ('source', 8), ('epoch', 8), ('valid', 4), ('written', 8)]]
    readonly += [span(a.controller_plan, field, 1, 4) for field in ('call', 'refresh')]
    readonly.append(span(c, 'expected', 1))
    for field, count in [('qi', 7), ('vi', 7), ('sparse', j.nv*j.nv),
                         ('ancestor', j.nb*j.nv), ('body_dof_offsets', j.nb+1),
                         ('body_dof_indices', index_words)]:
        pointer = int(getattr(j, field) or 0)
        if pointer and count:
            readonly.append((pointer, pointer+4*count))
    if span(out, 'parent', 1) != span(b, 'parent_written', 1):
        readonly.append(span(out, 'parent', 1))
    # An event/first-error snapshot must not overlap a solver-written matrix,
    # cache or quaternion buffer either. This inventory is cold host metadata.
    solver = [span(s, field, width) for field, width in [
        ('mass', 49), ('jacobian', 42), ('q', 7), ('qvel', 7), ('bias', 7), ('q0', 7),
        ('ee_pos', 3), ('ee_rot', 9), ('ee_velocity', 6), ('action', 7), ('goal_pos', 3),
        ('goal_rot', 9), ('gripper', 2), ('gains', 12), ('range', 18), ('ctrl', 9), ('raw_tau', 7),
        ('mass_inv', 49), ('A_full', 36), ('A_pos', 9), ('A_ori', 9), ('lambda_full', 36),
        ('lambda_pos', 9), ('lambda_ori', 9), ('nullspace_matrix', 49), ('position_error', 3),
        ('orientation_error', 3), ('quat_input', 4), ('quat_result', 9), ('nullspace_torque', 7),
        ('clipped', 7), ('derived_key', 91)]]
    solver.append(span(s, 'derived_data', 206, 8))
    solver.append(span(s, 'quat_cast', 4, 4))
    floating = outputs+sources+site_outputs+[span(p, 'pos', 3*p.ns), span(p, 'mat', 9*p.ns),
               span(v, 'target', 6), selected, span(b, 'bias', b.nv)]+solver
    if any(overlap(x, y) for x in metadata for y in readonly+floating if y[0]):
        return False
    # The pre-entry counter snapshot cannot alias any preparation/solve input.
    event = span(m, 'event', 1)
    if any(overlap(event, x) for x in access_regions(a, worlds, index_words=index_words, skip_event=True)):
        return False
    return True
