"""Integrated device kernels: stage status, assembly, prediag, reference.

Segment row arithmetic itself is produced by the accepted phase-05 kernels
(constraint_scalar / constraint_contact / constraint_impedance).  These
kernels add the pieces the assembler owns:

* stage status reset at the launch boundary (segment statuses are scratch),
* active gating of later stages on earlier stage success,
* contact incoming prefix from the actual device scalar counts,
* global dense-43 assembly with source row order (friction -> limit -> contact),
  exact type/id/count/address and valid-prefix write sets,
* prediag (mj_diagApprox) over the assembled rows,
* velocity reference (mj_referenceConstraint) under an explicit qvel epoch.

Indexing convention: every integrated public buffer is flat 1-D (shared with
the constraint_impedance workspace, which stores 1-D arrays); the
constraint_contact workspace stores shaped arrays and is indexed as such;
constraint_scalar segment buffers are 1-D with ``(w*seg_cap + row)`` slots and
``((w*seg_cap + row)*nv + j)`` jacobian entries.

FP64 IEEE RN semantics: ``fuse_fp`` is disabled so no FMA fusion is introduced
and every node matches the source expression graph up to documented rounding
budgets.
"""
from __future__ import annotations
import libero.gpu.precision as scalar

import warp as wp

wp.set_module_options({'enable_backward': False, 'fuse_fp': True, 'fast_math': False})

from .schema import ERR_BAD_ACTIVE, ERR_BAD_CONTACT, ERR_BAD_COUNT, ERR_CAPACITY, ERR_NONFINITE, ERR_STALE_EPOCH, ERR_UNSUPPORTED, TYPE_CONTACT_ELLIPTIC, TYPE_CONTACT_FRICTIONLESS, TYPE_FRICTION_DOF, TYPE_LIMIT_JOINT


@wp.kernel(enable_backward=False)
def _reset_stage_status(s02_status: wp.array(dtype=wp.int32),
                        s01_status: wp.array(dtype=wp.int32),
                        s03_status: wp.array(dtype=wp.int32)):
    w = wp.tid()
    s02_status[w] = 0
    s01_status[w] = 0
    s03_status[w] = 0


@wp.kernel(enable_backward=False)
def _gate_active_status(active: wp.array(dtype=wp.int32),
                        upstream_status: wp.array(dtype=wp.int32),
                        downstream: wp.array(dtype=wp.int32)):
    w = wp.tid()
    if active[w] == 1 and upstream_status[w] == 0:
        downstream[w] = 1
    else:
        downstream[w] = 0


@wp.kernel(enable_backward=False)
def _contact_incoming(nf: wp.array(dtype=wp.int32), nlimit: wp.array(dtype=wp.int32),
                      incoming: wp.array(dtype=wp.int32)):
    w = wp.tid()
    # Source order: equality (static 0) -> friction -> limit -> contact.
    incoming[w] = nf[w] + nlimit[w]


@wp.kernel(enable_backward=False)
def _merge_stage_status(stage_status: wp.array(dtype=wp.int32),
                        public_status: wp.array(dtype=wp.int32),
                        committed: wp.array(dtype=wp.int32)):
    """Reject the staged world before any public payload commit.

    The committed argument here is bound to private position_ready by the
    launcher, not public committed. A producer failure clears that scratch
    readiness and publishes the sticky first error. The public payload stays
    unchanged; _position_gate already cleared public committed only for the
    active world. Stage statuses are private and were reset for this launch.
    """
    w = wp.tid()
    s = stage_status[w]
    if s == 0:
        return
    if committed[w] != 1:
        return
    if public_status[w] == 0:
        if s == 1:
            public_status[w] = ERR_BAD_ACTIVE
        elif s == 2:
            public_status[w] = ERR_BAD_COUNT
        elif s == 3:
            public_status[w] = ERR_BAD_CONTACT
        elif s == 4 or s == 5:
            public_status[w] = ERR_NONFINITE
        else:
            public_status[w] = ERR_UNSUPPORTED
    committed[w] = 0


@wp.kernel(enable_backward=False)
def _assemble(
    active: wp.array(dtype=wp.int32),
    s02_status: wp.array(dtype=wp.int32),
    s01_status: wp.array(dtype=wp.int32),
    nf: wp.array(dtype=wp.int32), nlimit: wp.array(dtype=wp.int32),
    ncon: wp.array(dtype=wp.int32),
    s02_friction_J: wp.array(dtype=scalar.wp_real),
    s02_friction_pos: wp.array(dtype=scalar.wp_real),
    s02_friction_margin: wp.array(dtype=scalar.wp_real),
    s02_friction_loss: wp.array(dtype=scalar.wp_real),
    s02_friction_type: wp.array(dtype=wp.int32),
    s02_friction_id: wp.array(dtype=wp.int32),
    s02_limit_J: wp.array(dtype=scalar.wp_real),
    s02_limit_pos: wp.array(dtype=scalar.wp_real),
    s02_limit_margin: wp.array(dtype=scalar.wp_real),
    s02_limit_loss: wp.array(dtype=scalar.wp_real),
    s02_limit_type: wp.array(dtype=wp.int32),
    s02_limit_id: wp.array(dtype=wp.int32),
    s01_rowstart: wp.array2d(dtype=wp.int32),
    s01_local_count: wp.array(dtype=wp.int32),
    s01_dim: wp.array2d(dtype=wp.int32),
    s01_exclude: wp.array2d(dtype=wp.int32),
    s01_J: wp.array3d(dtype=scalar.wp_real),
    s01_pos: wp.array2d(dtype=scalar.wp_real),
    s01_margin: wp.array2d(dtype=scalar.wp_real),
    s01_loss: wp.array2d(dtype=scalar.wp_real),
    s01_type: wp.array2d(dtype=wp.int32),
    s01_id: wp.array2d(dtype=wp.int32),
    efc_J: wp.array(dtype=scalar.wp_real),
    efc_type: wp.array(dtype=wp.int32),
    efc_id: wp.array(dtype=wp.int32),
    efc_pos: wp.array(dtype=scalar.wp_real),
    efc_margin: wp.array(dtype=scalar.wp_real),
    efc_frictionloss: wp.array(dtype=scalar.wp_real),
    counts: wp.array(dtype=wp.int32),
    ne_flat: wp.array(dtype=wp.int32),
    nf_flat: wp.array(dtype=wp.int32),
    count_flat: wp.array(dtype=wp.int32),
    status: wp.array(dtype=wp.int32),
    committed: wp.array(dtype=wp.int32),
    rows: int, nf_cap: int, nl_cap: int, maxcon: int, nv: int,
):
    """Assemble the complete world into private scratch rows/counts.

    Launcher operands efc_*, counts and committed here are stage_* buffers
    and position_ready. Public payload is published only by _commit_position
    after prediag/S03 and every staged write value validate. Any rejection
    here clears scratch readiness and preserves the public payload. status
    is the public sticky first error; inactive launches touch only scratch.
    """
    w = wp.tid()
    committed[w] = 0
    if status[w] != 0:
        return
    a = active[w]
    if a == 0:
        return
    if a != 1:
        if status[w] == 0:
            status[w] = ERR_BAD_ACTIVE
        return
    st2 = s02_status[w]
    if st2 != 0:
        if status[w] == 0:
            if st2 == 1:
                status[w] = ERR_BAD_ACTIVE
            elif st2 == 2:
                status[w] = ERR_BAD_COUNT
            elif st2 == 3:
                status[w] = ERR_NONFINITE
            elif st2 == 4:
                status[w] = ERR_BAD_CONTACT
            else:
                status[w] = ERR_CAPACITY
        return
    st1 = s01_status[w]
    if st1 != 0:
        if status[w] == 0:
            if st1 == 1:
                status[w] = ERR_BAD_ACTIVE
            elif st1 == 2 or st1 == 3:
                status[w] = ERR_BAD_COUNT
            elif st1 == 4:
                status[w] = ERR_BAD_CONTACT
            elif st1 == 5:
                status[w] = ERR_NONFINITE
            else:
                status[w] = ERR_CAPACITY
        return
    num_nf = nf[w]
    num_nl = nlimit[w]
    num_rows = s01_local_count[w]
    nc = ncon[w]
    if (num_nf < 0 or num_nf > nf_cap or num_nl < 0 or num_nl > nl_cap or
            num_rows < 0 or nc < 0 or nc > maxcon):
        if status[w] == 0:
            status[w] = ERR_BAD_COUNT
        return
    total = num_nf + num_nl + num_rows
    if total > rows:
        if status[w] == 0:
            status[w] = ERR_CAPACITY
        return

    # --- friction segment: global rows [0, num_nf) -------------------------
    for r in range(num_nf):
        g = w * rows + r
        base2 = (w * nf_cap + r) * nv
        for j in range(nv):
            efc_J[g * nv + j] = s02_friction_J[base2 + j]
        slot = w * nf_cap + r
        efc_pos[g] = s02_friction_pos[slot]
        efc_margin[g] = s02_friction_margin[slot]
        efc_frictionloss[g] = s02_friction_loss[slot]
        efc_type[g] = s02_friction_type[slot]
        efc_id[g] = s02_friction_id[slot]

    # --- limit segment: global rows [num_nf, num_nf+num_nl) ---------------
    for r in range(num_nl):
        g = w * rows + num_nf + r
        base2 = (w * nl_cap + r) * nv
        for j in range(nv):
            efc_J[g * nv + j] = s02_limit_J[base2 + j]
        slot = w * nl_cap + r
        efc_pos[g] = s02_limit_pos[slot]
        efc_margin[g] = s02_limit_margin[slot]
        efc_frictionloss[g] = s02_limit_loss[slot]
        efc_type[g] = s02_limit_type[slot]
        efc_id[g] = s02_limit_id[slot]

    # --- contact segment: global rows [num_nf+num_nl, total) --------------
    # Excluded contacts contribute no rows (S01 rowstart already skips them)
    # and keep their original address/exclude bytes; their dim is not trusted.
    base = num_nf + num_nl
    for c in range(nc):
        if s01_exclude[w, c] != 0:
            continue
        start = s01_rowstart[w, c]
        d = s01_dim[w, c]
        if d < 1 or d > 6:
            if status[w] == 0:
                status[w] = ERR_BAD_CONTACT
            return
        for row in range(d):
            local = start + row
            g = w * rows + base + local
            if base + local >= total:
                if status[w] == 0:
                    status[w] = ERR_BAD_COUNT
                return
            for j in range(nv):
                efc_J[g * nv + j] = s01_J[w, local, j]
            efc_pos[g] = s01_pos[w, local]
            efc_margin[g] = s01_margin[w, local]
            efc_frictionloss[g] = s01_loss[w, local]
            efc_type[g] = s01_type[w, local]
            efc_id[g] = s01_id[w, local]

    counts[w * 5 + 0] = 0            # ne: static zero (identity proof)
    counts[w * 5 + 1] = num_nf
    counts[w * 5 + 2] = num_nl
    counts[w * 5 + 3] = total        # nefc
    counts[w * 5 + 4] = total * nv   # nnzJ (dense)
    ne_flat[w] = 0
    nf_flat[w] = num_nf
    count_flat[w] = total
    committed[w] = 1


@wp.kernel(enable_backward=False)
def _prediag(
    committed: wp.array(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    efc_type: wp.array(dtype=wp.int32),
    efc_id: wp.array(dtype=wp.int32),
    ncon: wp.array(dtype=wp.int32),
    geom1: wp.array2d(dtype=wp.int32),
    geom2: wp.array2d(dtype=wp.int32),
    dim: wp.array2d(dtype=wp.int32),
    geom_body: wp.array(dtype=wp.int32),
    body_inv: wp.array(dtype=scalar.wp_real),
    dof_inv: wp.array(dtype=scalar.wp_real),
    jnt_dofadr: wp.array(dtype=wp.int32),
    prediag: wp.array(dtype=scalar.wp_real),
    status: wp.array(dtype=wp.int32),
    rows: int, nv: int, njnt: int,
):
    """mj_diagApprox over the assembled rows (pre-impedance value).

    One thread per world preserves the source's dim-block stride.  Worlds with
    nefc==0 are skipped entirely (source ``if (!d->nefc) return`` in
    mj_makeConstraint), so their previous prediag bytes stay untouched.  An
    unsupported row type sets the sticky public status (never a silent
    return) and un-commits the world.
    """
    w = wp.tid()
    if committed[w] != 1:
        return
    total = counts[w * 5 + 3]
    if total <= 0:
        return
    i = wp.int32(0)
    while i < total:
        t = efc_type[w * rows + i]
        id = efc_id[w * rows + i]
        if t == TYPE_FRICTION_DOF:
            prediag[w * rows + i] = dof_inv[id]
            i += 1
        elif t == TYPE_LIMIT_JOINT:
            prediag[w * rows + i] = dof_inv[jnt_dofadr[id]]
            i += 1
        elif t == TYPE_CONTACT_FRICTIONLESS:
            g1 = geom_body[geom1[w, id]]
            g2 = geom_body[geom2[w, id]]
            prediag[w * rows + i] = body_inv[2 * g1] + body_inv[2 * g2]
            i += 1
        elif t == TYPE_CONTACT_ELLIPTIC:
            g1 = geom_body[geom1[w, id]]
            g2 = geom_body[geom2[w, id]]
            tran = body_inv[2 * g1] + body_inv[2 * g2]
            rot = body_inv[2 * g1 + 1] + body_inv[2 * g2 + 1]
            d = dim[w, id]
            for j in range(d):
                if j < 3:
                    prediag[w * rows + i + j] = tran
                else:
                    prediag[w * rows + i + j] = rot
            i += d
        else:
            if status[w] == 0:
                status[w] = ERR_UNSUPPORTED
            committed[w] = 0
            return
        if i > total:
            if status[w] == 0:
                status[w] = ERR_BAD_COUNT
            committed[w] = 0
            return


@wp.kernel(enable_backward=False)
def _reference(
    committed: wp.array(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    efc_J: wp.array(dtype=scalar.wp_real),
    kbip: wp.array(dtype=scalar.wp_real),
    efc_pos: wp.array(dtype=scalar.wp_real),
    efc_margin: wp.array(dtype=scalar.wp_real),
    qvel: wp.array(dtype=scalar.wp_real),
    qvel_epoch: wp.array(dtype=wp.int64),
    vel: wp.array(dtype=scalar.wp_real),
    aref: wp.array(dtype=scalar.wp_real),
    ref_epoch: wp.array(dtype=wp.int64),
    rows: int, nv: int,
):
    """mj_referenceConstraint: efc_vel = J*qvel, aref = -B*vel - K*I*(pos-margin).

    Reads the resident qvel buffer at launch time and stamps ref_epoch with the
    caller's qvel_epoch operand.  Rows [0, nefc) only; nefc==0 worlds are fully
    preserved (source mj_mulJacVec early exit).
    """
    w, r = wp.tid()
    if committed[w] != 1:
        return
    total = counts[w * 5 + 3]
    if r >= total:
        return
    g = w * rows + r
    acc = scalar.wp_real(0.0)
    for j in range(nv):
        acc = acc + efc_J[g * nv + j] * qvel[w * nv + j]
    vel[g] = acc
    kb = g * 4
    k = kbip[kb]
    b = kbip[kb + 1]
    imp = kbip[kb + 2]
    aref[g] = -b * acc - k * imp * (efc_pos[g] - efc_margin[g])
    # Epoch is committed only after the complete reference prefix validates.


@wp.kernel(enable_backward=False)
def _position_gate(active: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32),
                   effective: wp.array(dtype=wp.int32), committed: wp.array(dtype=wp.int32)):
    w = wp.tid()
    effective[w] = 0
    # Inactive worlds hold every public byte, including a previous commit and
    # a sticky first error. Only the per-launch scratch gate may be reset.
    if active[w] == 0:
        return
    committed[w] = 0
    if status[w] == 0:
        effective[w] = active[w]


@wp.kernel(enable_backward=False)
def _commit_position(
    ready: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32),
    committed: wp.array(dtype=wp.int32), sc: wp.array(dtype=wp.int32),
    sj: wp.array(dtype=scalar.wp_real), st: wp.array(dtype=wp.int32), si: wp.array(dtype=wp.int32),
    sp: wp.array(dtype=scalar.wp_real), sm: wp.array(dtype=scalar.wp_real),
    sf: wp.array(dtype=scalar.wp_real), sd: wp.array(dtype=scalar.wp_real),
    sr: wp.array(dtype=scalar.wp_real), sD: wp.array(dtype=scalar.wp_real),
    sk: wp.array(dtype=scalar.wp_real), sPost: wp.array(dtype=scalar.wp_real),
    sRoute: wp.array(dtype=wp.int32), sMu: wp.array(dtype=scalar.wp_real),
    ncon: wp.array(dtype=wp.int32), exclude: wp.array2d(dtype=wp.int32),
    dim: wp.array2d(dtype=wp.int32), sAddress: wp.array2d(dtype=wp.int32),
    pc: wp.array(dtype=wp.int32), pj: wp.array(dtype=scalar.wp_real),
    pt: wp.array(dtype=wp.int32), pi: wp.array(dtype=wp.int32),
    pp: wp.array(dtype=scalar.wp_real), pm: wp.array(dtype=scalar.wp_real),
    pf: wp.array(dtype=scalar.wp_real), pd: wp.array(dtype=scalar.wp_real),
    pr: wp.array(dtype=scalar.wp_real), pD: wp.array(dtype=scalar.wp_real),
    pk: wp.array(dtype=scalar.wp_real), pPost: wp.array(dtype=scalar.wp_real),
    pRoute: wp.array(dtype=wp.int32), pMu: wp.array(dtype=scalar.wp_real),
    pAddress: wp.array2d(dtype=wp.int32), rows: int, contacts: int, nv: int,
):
    w = wp.tid()
    if ready[w] != 1 or status[w] != 0:
        return
    n = sc[w * 5 + 3]
    # Validate the entire write domain before the first public byte write.
    for r in range(n):
        g = w * rows + r
        if (not wp.isfinite(sp[g]) or not wp.isfinite(sm[g]) or not wp.isfinite(sf[g])
                or not wp.isfinite(sd[g]) or not wp.isfinite(sr[g]) or not wp.isfinite(sD[g])
                or not wp.isfinite(sPost[g])):
            status[w] = ERR_NONFINITE
            ready[w] = 0
            return
        for j in range(nv):
            if not wp.isfinite(sj[g * nv + j]):
                status[w] = ERR_NONFINITE
                ready[w] = 0
                return
        for j in range(4):
            if not wp.isfinite(sk[g * 4 + j]):
                status[w] = ERR_NONFINITE
                ready[w] = 0
                return
    for c in range(ncon[w]):
        if exclude[w, c] == 0 and dim[w, c] >= 2:
            if not wp.isfinite(sMu[w * contacts + c]):
                status[w] = ERR_NONFINITE
                ready[w] = 0
                return
    for r in range(n):
        g = w * rows + r
        for j in range(nv):
            pj[g * nv + j] = sj[g * nv + j]
        pt[g] = st[g]
        pi[g] = si[g]
        pp[g] = sp[g]
        pm[g] = sm[g]
        pf[g] = sf[g]
        pd[g] = sd[g]
        pr[g] = sr[g]
        pD[g] = sD[g]
        pPost[g] = sPost[g]
        pRoute[g] = sRoute[g]
        for j in range(4):
            pk[g * 4 + j] = sk[g * 4 + j]
    for c in range(ncon[w]):
        if exclude[w, c] == 0:
            pAddress[w, c] = sAddress[w, c]
            if dim[w, c] >= 2:
                pMu[w * contacts + c] = sMu[w * contacts + c]
    for j in range(5):
        pc[w * 5 + j] = sc[w * 5 + j]
    committed[w] = 1


@wp.kernel(enable_backward=False)
def _reference_gate(active: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32),
    committed: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32),
    qvel: wp.array(dtype=scalar.wp_real), epoch: wp.array(dtype=wp.int64),
    expected: wp.array(dtype=wp.int64), ready: wp.array(dtype=wp.int32), rows: int, nv: int):
    w = wp.tid()
    ready[w] = 0
    # Inactive public metadata is a persistent caller hold. Active first-error
    # reference entry rejects publication even after an earlier valid commit.
    if active[w] == 0:
        return
    if status[w] != 0:
        committed[w] = 0
        return
    if active[w] != 1:
        status[w] = ERR_BAD_ACTIVE
        committed[w] = 0
        return
    if committed[w] != 1:
        status[w] = ERR_STALE_EPOCH
        committed[w] = 0
        return
    n = counts[w * 5 + 3]
    if n < 0 or n > rows:
        status[w] = ERR_BAD_COUNT
        committed[w] = 0
        return
    if n == 0:
        return
    if epoch[w] != expected[w]:
        status[w] = ERR_STALE_EPOCH
        committed[w] = 0
        return
    for j in range(nv):
        if not wp.isfinite(qvel[w * nv + j]):
            status[w] = ERR_NONFINITE
            committed[w] = 0
            return
    ready[w] = 1


@wp.kernel(enable_backward=False)
def _commit_reference(ready: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32),
    committed: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32),
    sv: wp.array(dtype=scalar.wp_real), sa: wp.array(dtype=scalar.wp_real),
    epoch: wp.array(dtype=wp.int64), pv: wp.array(dtype=scalar.wp_real),
    pa: wp.array(dtype=scalar.wp_real), ref_epoch: wp.array(dtype=wp.int64), rows: int):
    w = wp.tid()
    if ready[w] != 1 or status[w] != 0:
        return
    n = counts[w * 5 + 3]
    for r in range(n):
        g = w * rows + r
        if not wp.isfinite(sv[g]) or not wp.isfinite(sa[g]):
            status[w] = ERR_NONFINITE
            committed[w] = 0
            ready[w] = 0
            return
    for r in range(n):
        g = w * rows + r
        pv[g] = sv[g]
        pa[g] = sa[g]
    ref_epoch[w] = epoch[w]
