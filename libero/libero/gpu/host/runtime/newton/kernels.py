"""Glue kernels between frozen S04-S08 stages for the complete fwdConstraint
and the device-resident Newton loop.

Source semantics (MuJoCo 2.3.7, engine_forward.c:388-506, engine_solver.c:
1499-1622, engine_solver.c:750-835).  Every kernel preserves the source write
order and the source arithmetic order (mju_dot four lanes then (0+2)+(1+3),
mju_addToScl, mju_subFrom).  Production launches do no allocation, upload,
download or host-side branching on device values; the 100-iteration masked
schedule only reads the static shape.
"""
from __future__ import annotations
import libero.gpu.precision as scalar

import warp as wp

wp.set_module_options({'enable_backward': False, 'fuse_fp': True,
                       'fast_math': False})

NV = 43
NM = 153
# S07 diagnostic capacity (frozen kernel.cu stride 52) and the untouched-slot
# markers used by the S07 acceptance discipline (verify_gpu seeds 97.0 / -101).
TRACECAP = 52
TAPE_UNUSED_F = 97.0
TAPE_UNUSED_TAG = -101


@wp.func
def _dot4(a: wp.array(dtype=scalar.wp_real), b: wp.array(dtype=scalar.wp_real),
          basea: int, baseb: int, n: int) -> scalar.wp_real:
    # engine_util_blas.c mju_dot: four independent lanes then (0+2)+(1+3).
    # Structure mirrors the frozen S05 _dot: the while sits outside the n>=4
    # lane priming so the induction variable stays a dynamic value.
    res = scalar.wp_real(0.0)
    i = int(0)
    s0 = scalar.wp_real(0.0)
    s1 = scalar.wp_real(0.0)
    s2 = scalar.wp_real(0.0)
    s3 = scalar.wp_real(0.0)
    if n >= 4:
        s0 = a[basea] * b[baseb]
        s1 = a[basea + 1] * b[baseb + 1]
        s2 = a[basea + 2] * b[baseb + 2]
        s3 = a[basea + 3] * b[baseb + 3]
        i = 4
    while i + 4 <= n:
        s0 = s0 + a[basea + i] * b[baseb + i]
        s1 = s1 + a[basea + i + 1] * b[baseb + i + 1]
        s2 = s2 + a[basea + i + 2] * b[baseb + i + 2]
        s3 = s3 + a[basea + i + 3] * b[baseb + i + 3]
        i += 4
    res = (s0 + s2) + (s1 + s3)
    left = n - i
    if left == 3:
        p0 = a[basea + i] * b[baseb + i]
        p1 = a[basea + i + 1] * b[baseb + i + 1]
        p2 = a[basea + i + 2] * b[baseb + i + 2]
        res = res + ((p0 + p1) + p2)
    elif left == 2:
        p0 = a[basea + i] * b[baseb + i]
        p1 = a[basea + i + 1] * b[baseb + i + 1]
        res = res + (p0 + p1)
    elif left == 1:
        res = res + a[basea + i] * b[baseb + i]
    return res


@wp.kernel(enable_backward=False)
def _jacvec_sub(active: wp.array(dtype=wp.int32),
                nefc: wp.array(dtype=wp.int32),
                J: wp.array(dtype=scalar.wp_real),
                qacc: wp.array(dtype=scalar.wp_real),
                aref: wp.array(dtype=scalar.wp_real),
                out: wp.array(dtype=scalar.wp_real),
                rows: int):
    # mj_mulJacVec then mju_subFrom: out[j] = (J*qacc)[j] - aref[j].
    # All row-outer buffers are flat world-outer; qacc is flat (w*NV base).
    w = wp.tid()
    if active[w] == 0:
        return
    base = w * rows
    qb = w * NV
    # Original mj_mulJacVec/subFrom read and write exactly nefc rows.
    # Caller-owned unused J/aref/out capacity is not part of this operation.
    for j in range(nefc[w]):
        acc = _dot4(J, qacc, (base + j) * NV, qb, NV)
        out[base + j] = acc - aref[base + j]


@wp.kernel(enable_backward=False)
def _mulM(active: wp.array(dtype=wp.int32),
          qacc: wp.array(dtype=scalar.wp_real, ndim=2),
          qM: wp.array(dtype=scalar.wp_real),
          madr: wp.array(dtype=wp.int32),
          parent: wp.array(dtype=wp.int32),
          simple: wp.array(dtype=wp.int32),
          out: wp.array(dtype=scalar.wp_real, ndim=2)):
    # mj_mulM: tree multiply with madr walk, exact source accumulation order.
    w = wp.tid()
    if active[w] == 0:
        return
    qb = w * NV
    mb = w * NM
    for i in range(NV):
        out[w, i] = scalar.wp_real(0.0)
    for i in range(NV):
        adr = madr[i]
        if simple[i] != 0:
            out[w, i] = qM[mb + adr] * qacc[w, i]
        else:
            out[w, i] = out[w, i] + qM[mb + adr] * qacc[w, i]
            j = parent[i]
            adr += 1
            while j >= 0:
                out[w, i] = out[w, i] + qM[mb + adr] * qacc[w, j]
                out[w, j] = out[w, j] + qM[mb + adr] * qacc[w, i]
                j = parent[j]
                adr += 1


@wp.kernel(enable_backward=False)
def _gauss_compute(active: wp.array(dtype=wp.int32),
                   Ma: wp.array(dtype=scalar.wp_real, ndim=2),
                   qfrc_smooth: wp.array(dtype=scalar.wp_real, ndim=2),
                   qacc: wp.array(dtype=scalar.wp_real, ndim=2),
                   qacc_smooth: wp.array(dtype=scalar.wp_real, ndim=2),
                   cost: wp.array(dtype=scalar.wp_real),
                   quad0: wp.array(dtype=scalar.wp_real),
                   add_to_cost: wp.array(dtype=wp.int32)):
    # CGupdateConstraint Gauss half / warmstart Gauss half (engine_solver.c:793-798,
    # engine_forward.c:427-432): sum 0.5*(Ma-qfrc_smooth)*(qacc-qacc_smooth),
    # source sequential accumulation over i then the source's i-wise add.
    w = wp.tid()
    if active[w] == 0:
        return
    acc = scalar.wp_real(0.0)
    for i in range(NV):
        acc = acc + scalar.wp_real(0.5) * (Ma[w, i] - qfrc_smooth[w, i]) * (qacc[w, i] - qacc_smooth[w, i])
    quad0[w] = acc
    if add_to_cost[w] != 0:
        cost[w] = cost[w] + acc


@wp.kernel(enable_backward=False)
def _count_active_cone(active: wp.array(dtype=wp.int32),
                       state: wp.array(dtype=wp.int32),
                       n: wp.array(dtype=wp.int32),
                       nactive: wp.array(dtype=wp.int32),
                       ncone: wp.array(dtype=wp.int32),
                       rows: int):
    # CGupdateConstraint counts (engine_solver.c:785-789): source row order.
    w = wp.tid()
    if active[w] == 0:
        return
    base = w * rows
    na = int(0)
    nc = int(0)
    for r in range(n[w]):
        if state[base + r] != 0:
            na += 1
        if state[base + r] == 4:
            nc += 1
    nactive[w] = na
    ncone[w] = nc


@wp.kernel(enable_backward=False)
def _save_old(active: wp.array(dtype=wp.int32),
              nefc: wp.array(dtype=wp.int32),
              state: wp.array(dtype=wp.int32),
              cost: wp.array(dtype=scalar.wp_real),
              oldstate: wp.array(dtype=wp.int32),
              oldcost: wp.array(dtype=scalar.wp_real),
              rows: int):
    # mj_solCGNewton "save old" (engine_solver.c:1548-1551): memcpy(oldstate,
    # nefc elements) and oldcost = ctx.cost.  The copy is bounded by the
    # per-world nefc exactly like the source; the row tail stays untouched.
    w = wp.tid()
    if active[w] == 0:
        return
    base = w * rows
    n = nefc[w]
    for r in range(n):
        oldstate[base + r] = state[base + r]
    oldcost[w] = cost[w]


@wp.kernel(enable_backward=False)
def _gate_ls(active: wp.array(dtype=wp.int32),
             alpha: wp.array(dtype=scalar.wp_real),
             s07_status: wp.array(dtype=wp.int32),
             stop_reason: wp.array(dtype=wp.int32),
             err_code: wp.array(dtype=wp.int32)):
    # Source order: CGsearch then "if (alpha == 0) break" (engine_solver.c:1533-1538).
    # Errors deactivate without claiming a successful stop; alpha==0 is the
    # source's no-improvement break (no commit, no stats, no iter++).
    w = wp.tid()
    if active[w] == 0:
        return
    if s07_status[w] != 0:
        if err_code[w] == 0:
            err_code[w] = 1000 + s07_status[w]
        active[w] = 0
        return
    if alpha[w] == scalar.wp_real(0.0):
        stop_reason[w] = 10
        active[w] = 0


@wp.kernel(enable_backward=False)
def _gate_status(active: wp.array(dtype=wp.int32),
                 status: wp.array(dtype=wp.int32),
                 err_base: wp.array(dtype=wp.int32),
                 err_code: wp.array(dtype=wp.int32)):
    # Deactivate on any stage status, preserving the first error code.
    w = wp.tid()
    if active[w] == 0:
        return
    if status[w] != 0:
        if err_code[w] == 0:
            err_code[w] = err_base[w] + status[w]
        active[w] = 0


@wp.kernel(enable_backward=False)
def _gate_stats(active: wp.array(dtype=wp.int32),
                out_stop: wp.array(dtype=wp.int32),
                out_reason: wp.array(dtype=wp.int32),
                s08c_status: wp.array(dtype=wp.int32),
                stop_reason: wp.array(dtype=wp.int32),
                err_code: wp.array(dtype=wp.int32)):
    # Source: saveStats (iter++) then strict tolerance break BEFORE the
    # direction write (engine_solver.c:1576-1582).  Only reason 1 (tolerance)
    # stops here; reason 2 (maxiter) is the following while guard and still
    # passes through the direction write — see _gate_maxiter.
    w = wp.tid()
    if active[w] == 0:
        return
    if s08c_status[w] != 0:
        if err_code[w] == 0:
            err_code[w] = 3000 + s08c_status[w]
        active[w] = 0
        return
    if out_stop[w] != 0 and out_reason[w] == 1:
        stop_reason[w] = 1
        active[w] = 0


@wp.kernel(enable_backward=False)
def _gate_maxiter(active: wp.array(dtype=wp.int32),
                  out_stop: wp.array(dtype=wp.int32),
                  out_reason: wp.array(dtype=wp.int32),
                  stop_reason: wp.array(dtype=wp.int32)):
    # The while(iter<maxiter) guard fires AFTER the direction write
    # (engine_solver.c:1585-1592).  Deactivate only now so the source's
    # direction write set is preserved for the maxiter exit path.
    w = wp.tid()
    if active[w] == 0:
        return
    if out_stop[w] != 0 and out_reason[w] == 2:
        stop_reason[w] = 2
        active[w] = 0


@wp.kernel(enable_backward=False)
def _direction_commit(active: wp.array(dtype=wp.int32),
                      candidate: wp.array(dtype=scalar.wp_real, ndim=2),
                      search: wp.array(dtype=scalar.wp_real, ndim=2)):
    # engine_solver.c:1528 / :1585-1586 mju_scl(search, Mgrad, -1).
    # Only the world still looping writes search: the tolerance/maxiter break
    # and the alpha==0 break leave the previous search untouched (source
    # direction write set).
    w = wp.tid()
    if active[w] == 0:
        return
    for i in range(NV):
        search[w, i] = candidate[w, i]


@wp.kernel(enable_backward=False)
def _init_direction(active: wp.array(dtype=wp.int32),
                    candidate: wp.array(dtype=scalar.wp_real, ndim=2),
                    search: wp.array(dtype=scalar.wp_real, ndim=2)):
    # engine_solver.c:1528 "start both with preconditioned gradient":
    # unconditional search = -Mgrad before the loop for every Newton world.
    w = wp.tid()
    if active[w] == 0:
        return
    for i in range(NV):
        search[w, i] = candidate[w, i]


@wp.kernel(enable_backward=False)
def _empty_fast_path(empty: wp.array(dtype=wp.int32),
                     qacc_smooth: wp.array(dtype=scalar.wp_real, ndim=2),
                     qacc: wp.array(dtype=scalar.wp_real, ndim=2),
                     warmstart: wp.array(dtype=scalar.wp_real, ndim=2),
                     qfrc_constraint: wp.array(dtype=scalar.wp_real, ndim=2),
                     solver_iter: wp.array(dtype=wp.int32)):
    # engine_forward.c:462-468: qacc=qacc_smooth, qacc_warmstart=qacc_smooth,
    # qfrc_constraint=+0, solver_iter=0.  solver_nnz and the old stats tail
    # are NOT written.  force/state/H tails are not touched either.
    w = wp.tid()
    if empty[w] == 0:
        return
    for i in range(NV):
        qacc[w, i] = qacc_smooth[w, i]
        warmstart[w, i] = qacc_smooth[w, i]
        qfrc_constraint[w, i] = scalar.wp_real(0.0)  # mju_zero writes +0
    solver_iter[w] = 0


@wp.kernel(enable_backward=False)
def _copy_qacc(active: wp.array(dtype=wp.int32),
               src: wp.array(dtype=scalar.wp_real, ndim=2),
               dst: wp.array(dtype=scalar.wp_real, ndim=2)):
    # mju_copy(d->qacc, d->qacc_warmstart/d->qacc_smooth, nv) at warmstart entry
    # and the cold branch (engine_forward.c:395 / :450).
    w = wp.tid()
    if active[w] == 0:
        return
    for i in range(NV):
        dst[w, i] = src[w, i]


@wp.kernel(enable_backward=False)
def _copy_cost(active: wp.array(dtype=wp.int32),
               src: wp.array(dtype=scalar.wp_real),
               dst: wp.array(dtype=scalar.wp_real)):
    # Preserve the warm-start evaluator total before the smooth evaluator
    # overwrites cost (engine_forward.c:418-438).
    w = wp.tid()
    if active[w] == 0:
        return
    dst[w] = src[w]


@wp.kernel(enable_backward=False)
def _zero_rows(active: wp.array(dtype=wp.int32),
               nefc: wp.array(dtype=wp.int32),
               arr: wp.array(dtype=scalar.wp_real),
               rows: int):
    # cold branch mju_zero(d->efc_force, nefc) (engine_forward.c:450):
    # exactly nefc rows become +0; rows [nefc, rows) keep their values.
    w = wp.tid()
    if active[w] == 0:
        return
    base = w * rows
    n = nefc[w]
    for r in range(n):
        arr[base + r] = scalar.wp_real(0.0)


@wp.kernel(enable_backward=False)
def _solver_iter_reset(active: wp.array(dtype=wp.int32),
                       solver_iter: wp.array(dtype=wp.int32)):
    # engine_forward.c:442: d->solver_iter = 0 after warmstart, before solver.
    w = wp.tid()
    if active[w] == 0:
        return
    solver_iter[w] = 0


@wp.kernel(enable_backward=False)
def _finalize_exit(newton_world: wp.array(dtype=wp.int32),
                   err_code: wp.array(dtype=wp.int32),
                   solver_exit: wp.array(dtype=wp.int32),
                   finalize_active: wp.array(dtype=wp.int32)):
    # solver_exit=1 only for Newton worlds that finished without error, so S08
    # _finalize writes solver_iter/solver_nnz/warmstart exactly like the source
    # exit path (engine_solver.c:1597-1610, engine_forward.c:447-448).
    # finalize_active separates the exit writes from the loop's aliveness mask.
    w = wp.tid()
    if newton_world[w] == 0 or err_code[w] != 0:
        solver_exit[w] = 0
        finalize_active[w] = 0
    else:
        solver_exit[w] = 1
        finalize_active[w] = 1


@wp.kernel(enable_backward=False)
def _seed_eval(evals: wp.array(dtype=scalar.wp_real, ndim=3),
               tags: wp.array(dtype=wp.int32, ndim=2)):
    # S07 acceptance discipline applied every iteration: seed the full
    # evaluation diagnostics with the untouched markers BEFORE the launch, so
    # slots j >= LSiter keep the markers instead of the previous iteration's
    # values (frozen verify_gpu.py seeds 97.0 / -101 for the same reason).
    w = wp.tid()
    for e in range(TRACECAP):
        tags[w, e] = TAPE_UNUSED_TAG
        for c in range(4):
            evals[w, e, c] = scalar.wp_real(TAPE_UNUSED_F)


@wp.kernel(enable_backward=False)
def _tape_commit(slot: int,
                 tape_slots: int,
                 loop_active: wp.array(dtype=wp.int32),
                 alpha: wp.array(dtype=scalar.wp_real),
                 slope: wp.array(dtype=scalar.wp_real),
                 lineslope: wp.array(dtype=scalar.wp_real),
                 improvement: wp.array(dtype=scalar.wp_real),
                 gradient: wp.array(dtype=scalar.wp_real),
                 cost: wp.array(dtype=scalar.wp_real),
                 oldcost: wp.array(dtype=scalar.wp_real),
                 quad0: wp.array(dtype=scalar.wp_real),
                 LSiter: wp.array(dtype=wp.int32),
                 LSresult: wp.array(dtype=wp.int32),
                 committed: wp.array(dtype=wp.int32),
                 route: wp.array(dtype=wp.int32),
                 nupdate: wp.array(dtype=wp.int32),
                 nactive: wp.array(dtype=wp.int32),
                 ncone: wp.array(dtype=wp.int32),
                 out_nchange: wp.array(dtype=wp.int32),
                 iteration: wp.array(dtype=wp.int32),
                 stop_reason: wp.array(dtype=wp.int32),
                 err_code: wp.array(dtype=wp.int32),
                 s07_status: wp.array(dtype=wp.int32),
                 s08b_status: wp.array(dtype=wp.int32),
                 stage_status: wp.array(dtype=wp.int32),
                 s08c_status: wp.array(dtype=wp.int32),
                 tape_f: wp.array(dtype=scalar.wp_real, ndim=3),
                 tape_i: wp.array(dtype=wp.int32, ndim=3),
                 tape_status: wp.array(dtype=wp.int32, ndim=3),
                 eval_trace: wp.array(dtype=scalar.wp_real, ndim=3),
                 eval_tag: wp.array(dtype=wp.int32, ndim=2),
                 tape_eval: wp.array(dtype=scalar.wp_real, ndim=4),
                 tape_tag: wp.array(dtype=wp.int32, ndim=3)):
    # Witness tape: one slot per masked-schedule iteration.  Written on device
    # so the production loop never downloads to decide or to observe.
    if slot >= tape_slots:
        return
    w = wp.tid()
    # record every classified Newton world for this slot (including those that
    # stopped earlier) so the tape is dense and replayable
    tape_f[w, slot, 0] = alpha[w]
    tape_f[w, slot, 1] = slope[w]
    tape_f[w, slot, 2] = lineslope[w]
    tape_f[w, slot, 3] = improvement[w]
    tape_f[w, slot, 4] = gradient[w]
    tape_f[w, slot, 5] = cost[w]
    tape_f[w, slot, 6] = oldcost[w]
    tape_f[w, slot, 7] = quad0[w]
    tape_i[w, slot, 0] = loop_active[w]
    tape_i[w, slot, 1] = LSiter[w]
    tape_i[w, slot, 2] = LSresult[w]
    tape_i[w, slot, 3] = committed[w]
    tape_i[w, slot, 4] = route[w]
    tape_i[w, slot, 5] = nupdate[w]
    tape_i[w, slot, 6] = nactive[w]
    tape_i[w, slot, 7] = ncone[w]
    tape_i[w, slot, 8] = out_nchange[w]
    tape_i[w, slot, 9] = iteration[w]
    tape_i[w, slot, 10] = stop_reason[w]
    tape_i[w, slot, 11] = err_code[w]
    tape_status[w, slot, 0] = s07_status[w]
    tape_status[w, slot, 1] = s08b_status[w]
    tape_status[w, slot, 2] = stage_status[w]
    tape_status[w, slot, 3] = s08c_status[w]
    for e in range(TRACECAP):
        for c in range(4):
            tape_eval[w, slot, e, c] = eval_trace[w, e, c]
        tape_tag[w, slot, e] = eval_tag[w, e]


@wp.kernel(enable_backward=False)
def _tape_ls_input(slot: int, tape_slots: int,
                   quad0: wp.array(dtype=scalar.wp_real),
                   tape: wp.array(dtype=scalar.wp_real, ndim=2)):
    if slot >= tape_slots:
        return
    w = wp.tid()
    tape[w, slot] = quad0[w]


@wp.kernel(enable_backward=False)
def _tape_ls_buffers(slot: int, tape_slots: int, rows: int,
                     Mv: wp.array(dtype=scalar.wp_real, ndim=2),
                     Jv: wp.array(dtype=scalar.wp_real, ndim=2),
                     quad: wp.array(dtype=scalar.wp_real, ndim=3),
                     quadGauss: wp.array(dtype=scalar.wp_real, ndim=2),
                     tape_Mv: wp.array(dtype=scalar.wp_real, ndim=3),
                     tape_Jv: wp.array(dtype=scalar.wp_real, ndim=3),
                     tape_quad: wp.array(dtype=scalar.wp_real, ndim=4),
                     tape_quadGauss: wp.array(dtype=scalar.wp_real, ndim=3)):
    if slot >= tape_slots:
        return
    w = wp.tid()
    for i in range(NV):
        tape_Mv[w, slot, i] = Mv[w, i]
    for i in range(rows):
        tape_Jv[w, slot, i] = Jv[w, i]
        for j in range(3):
            tape_quad[w, slot, i, j] = quad[w, i, j]
    for j in range(3):
        tape_quadGauss[w, slot, j] = quadGauss[w, j]


@wp.kernel(enable_backward=False)
def _validate_entry_counts(nefc: wp.array(dtype=wp.int32),
                           ne: wp.array(dtype=wp.int32),
                           nf: wp.array(dtype=wp.int32),
                           ncon: wp.array(dtype=wp.int32),
                           empty: wp.array(dtype=wp.int32),
                           newton: wp.array(dtype=wp.int32),
                           active: wp.array(dtype=wp.int32),
                           err_code: wp.array(dtype=wp.int32),
                           error_item: wp.array(dtype=wp.int32),
                           rows: int, contacts: int):
    # Invalid counts must not disappear in the nefc>0 classification mask.
    # Field ordinals: nefc=0, ncon=3, ne=1, nf=2; source check order retained.
    w = wp.tid()
    item = int(-1)
    n = nefc[w]
    if n < 0 or n > rows:
        item = 0
    elif ncon[w] < 0 or ncon[w] > contacts:
        item = 3
    elif ne[w] < 0 or ne[w] > n:
        item = 1
    elif nf[w] < 0 or nf[w] > n-ne[w]:
        item = 2
    if item >= 0:
        if err_code[w] == 0:
            err_code[w] = 4002
            error_item[w] = item
        empty[w] = 0
        newton[w] = 0
        active[w] = 0


@wp.kernel(enable_backward=False)
def _validate_contact_rows(nefc: wp.array(dtype=wp.int32),
                           ne: wp.array(dtype=wp.int32),
                           nf: wp.array(dtype=wp.int32),
                           ncon: wp.array(dtype=wp.int32),
                           typ: wp.array(dtype=wp.int32),
                           cid: wp.array(dtype=wp.int32),
                           row_dim: wp.array(dtype=wp.int32),
                           row_mu: wp.array(dtype=scalar.wp_real),
                           row_fr: wp.array(dtype=scalar.wp_real),
                           con_dim: wp.array(dtype=wp.int32),
                           con_mu: wp.array(dtype=scalar.wp_real),
                           con_fr: wp.array(dtype=scalar.wp_real),
                           newton: wp.array(dtype=wp.int32),
                           active: wp.array(dtype=wp.int32),
                           err_code: wp.array(dtype=wp.int32),
                           error_item: wp.array(dtype=wp.int32),
                           rows: int, contacts: int):
    # S04 reads contact fields, S07 reads duplicated row fields. A valid
    # original-source entry supplies the same values to both consumers.
    # Do not silently solve with a stale row copy. Code 8001 is N-owned
    # INPUT_BINDING, error_item is the first inconsistent constraint row.
    w = wp.tid()
    if newton[w] == 0 or err_code[w] != 0:
        return
    n = nefc[w]
    rb = w*rows
    cb = w*contacts
    # Leave invalid S04 structures to its frozen first-error precedence.
    i = ne[w]+nf[w]
    while i < n:
        t = typ[rb+i]
        if t == 7:
            c = cid[rb+i]
            if c < 0 or c >= ncon[w]:
                return
            d = con_dim[cb+c]
            m = con_mu[cb+c]
            if (d != 1 and d != 3 and d != 4 and d != 6) or i+d > n or m <= scalar.wp_real(0.0) or not wp.isfinite(m):
                return
            for j in range(d):
                if typ[rb+i+j] != 7 or cid[rb+i+j] != c:
                    return
            for j in range(d-1):
                f = con_fr[(cb+c)*5+j]
                if f <= scalar.wp_real(0.0) or not wp.isfinite(f):
                    return
            i += d
        elif t >= 0 and t <= 6:
            i += 1
        else:
            return
    i = ne[w]+nf[w]
    while i < n:
        if typ[rb+i] == 7:
            c = cid[rb+i]
            d = con_dim[cb+c]
            # Original S07 consumes only the block head's duplicate
            # fields and d-1 friction slots, then skips the remaining rows.
            # Unread row copies and unused friction may contain dirty bits.
            r = rb+i
            bad = row_dim[r] != d or row_mu[r] != con_mu[cb+c]
            for k in range(d-1):
                if row_fr[r*5+k] != con_fr[(cb+c)*5+k]:
                    bad = True
            if bad:
                err_code[w] = 8001
                error_item[w] = i
                active[w] = 0
                newton[w] = 0
                return
            i += d
        else:
            i += 1
