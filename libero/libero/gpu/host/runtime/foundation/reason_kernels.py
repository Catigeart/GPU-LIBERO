"""Generated source-order metadata classifiers. See SOURCE_REASON_DERIVATION ledger.
These are not physics producers and do not modify any original input/output.
"""
from __future__ import annotations
import libero.gpu.precision as scalar
import warp as wp
wp.set_module_options({"enable_backward":False,"fuse_fp":True,"fast_math":False})
NV=43
NQ=48
NJ=18
NF_CAP=9
NL_CAP=26
NM=153
OK=0
BAD_ACTIVE=1
BAD_COUNT=2
BAD_OFFSET=3
BAD_CONTACT=4
BAD_NUMERIC=5
CAPACITY=6
SAFE_MAGNITUDE=scalar.wp_real(scalar.safe_magnitude)
from libero.gpu.reference.position.body_kinematics.kernel import Model, Data, Diagnostic
from libero.gpu.reference.newton.newton_direction.kernel import _factor


@wp.kernel(enable_backward=False)
def _reason_body(m: Model, d: Data, g: Diagnostic, reason_code: wp.array(dtype=wp.int32), reason_kind: wp.array(dtype=wp.int32)):
    w = wp.tid()
    reason_code[w] = 0
    reason_kind[w] = 0
    if d.active[w] == 0:
        return
    else:
        pass
    if d.active[w] != 1:
        reason_code[w] = 1
        reason_kind[w] = 1
        return
    else:
        pass
    for k in range(m.nq):
        if not wp.isfinite(d.qpos[w, k]):
            reason_code[w] = 4
            reason_kind[w] = 5
            return
        else:
            pass
    for k in range(m.nm * 3):
        if not wp.isfinite(d.mocap_pos[w, k]):
            reason_code[w] = 4
            reason_kind[w] = 5
            return
        else:
            pass
    for k in range(m.nm * 4):
        if not wp.isfinite(d.mocap_quat[w, k]):
            reason_code[w] = 4
            reason_kind[w] = 5
            return
        else:
            pass
    for j in range(m.nj):
        if m.jtype[j] == 3:
            qa = m.qadr[j]
            angle = d.qpos[w, qa] - m.qpos0[qa]
            if angle != scalar.wp_real(0.0):
                z = angle * scalar.wp_real(0.5)
                if not wp.isfinite(z) or wp.abs(z) > scalar.wp_real(4.0):
                    reason_code[w] = 2
                    if not wp.isfinite(z):
                        reason_kind[w] = 5
                    elif wp.abs(z) > scalar.wp_real(4.0):
                        reason_kind[w] = 6
                    else:
                        reason_kind[w] = 8
                    return
                else:
                    pass
            else:
                pass
        else:
            pass

@wp.kernel(enable_backward=False)
def _reason_contact(ncon: wp.array(dtype=wp.int32), incoming: wp.array(dtype=wp.int32), capacity: wp.array(dtype=wp.int32), active: wp.array(dtype=wp.int32), geom1: wp.array2d(dtype=wp.int32), geom2: wp.array2d(dtype=wp.int32), pos: wp.array3d(dtype=scalar.wp_real), frame: wp.array3d(dtype=scalar.wp_real), dist: wp.array2d(dtype=scalar.wp_real), margin: wp.array2d(dtype=scalar.wp_real), dim: wp.array2d(dtype=wp.int32), exclude: wp.array2d(dtype=wp.int32), cdof: wp.array3d(dtype=scalar.wp_real), subtree: wp.array3d(dtype=scalar.wp_real), geom_body: wp.array(dtype=wp.int32), rowstart: wp.array2d(dtype=wp.int32), local_count: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32), maxcon: int, maxrows: int, nv: int, nbody: int, ngeom: int, reason_code: wp.array(dtype=wp.int32), reason_kind: wp.array(dtype=wp.int32)):
    w = wp.tid()
    reason_code[w] = 0
    reason_kind[w] = 0
    if active[w] == 0:
        return
    else:
        pass
    if active[w] != 1:
        reason_code[w] = BAD_ACTIVE
        reason_kind[w] = 1
        return
    else:
        pass
    count = ncon[w]
    if count < 0 or count > maxcon:
        reason_code[w] = BAD_COUNT
        if count < 0:
            reason_kind[w] = 2
        elif count > maxcon:
            reason_kind[w] = 3
        else:
            reason_kind[w] = 8
        return
    else:
        pass
    base = incoming[w]
    cap = capacity[w]
    if base < 0 or cap < base:
        reason_code[w] = BAD_OFFSET
        reason_kind[w] = 2
        return
    else:
        pass
    total = int(0)
    for c in range(count):
        if exclude[w, c] != 0:
            continue
        else:
            pass
        d = dim[w, c]
        g1 = geom1[w, c]
        g2 = geom2[w, c]
        if d != 1 and d != 3 and (d != 4) and (d != 6) or g1 < 0 or g1 >= ngeom or (g2 < 0) or (g2 >= ngeom):
            reason_code[w] = BAD_CONTACT
            reason_kind[w] = 4
            return
        else:
            pass
        b1 = geom_body[g1]
        b2 = geom_body[g2]
        if b1 < 0 or b1 >= nbody or b2 < 0 or (b2 >= nbody):
            reason_code[w] = BAD_CONTACT
            reason_kind[w] = 4
            return
        else:
            pass
        for k in range(3):
            if not wp.isfinite(pos[w, c, k]) or wp.abs(pos[w, c, k]) > SAFE_MAGNITUDE:
                reason_code[w] = BAD_NUMERIC
                if not wp.isfinite(pos[w, c, k]):
                    reason_kind[w] = 5
                elif wp.abs(pos[w, c, k]) > SAFE_MAGNITUDE:
                    reason_kind[w] = 6
                else:
                    reason_kind[w] = 8
                return
            else:
                pass
        for k in range(9):
            if not wp.isfinite(frame[w, c, k]) or wp.abs(frame[w, c, k]) > SAFE_MAGNITUDE:
                reason_code[w] = BAD_NUMERIC
                if not wp.isfinite(frame[w, c, k]):
                    reason_kind[w] = 5
                elif wp.abs(frame[w, c, k]) > SAFE_MAGNITUDE:
                    reason_kind[w] = 6
                else:
                    reason_kind[w] = 8
                return
            else:
                pass
        if not wp.isfinite(dist[w, c]) or not wp.isfinite(margin[w, c]) or wp.abs(dist[w, c]) > SAFE_MAGNITUDE or (wp.abs(margin[w, c]) > SAFE_MAGNITUDE):
            reason_code[w] = BAD_NUMERIC
            if not wp.isfinite(dist[w, c]):
                reason_kind[w] = 5
            elif not wp.isfinite(margin[w, c]):
                reason_kind[w] = 5
            elif wp.abs(dist[w, c]) > SAFE_MAGNITUDE:
                reason_kind[w] = 6
            elif wp.abs(margin[w, c]) > SAFE_MAGNITUDE:
                reason_kind[w] = 6
            else:
                reason_kind[w] = 8
            return
        else:
            pass
        if nv > 0:
            total += d
        else:
            pass
    if total > maxrows or base > 2147483647 - total or base + total > cap:
        reason_code[w] = CAPACITY
        reason_kind[w] = 3
        return
    else:
        pass
    if total > 0:
        for j in range(nv):
            for k in range(6):
                if not wp.isfinite(cdof[w, j, k]) or wp.abs(cdof[w, j, k]) > SAFE_MAGNITUDE:
                    reason_code[w] = BAD_NUMERIC
                    if not wp.isfinite(cdof[w, j, k]):
                        reason_kind[w] = 5
                    elif wp.abs(cdof[w, j, k]) > SAFE_MAGNITUDE:
                        reason_kind[w] = 6
                    else:
                        reason_kind[w] = 8
                    return
                else:
                    pass
        for b in range(nbody):
            for k in range(3):
                if not wp.isfinite(subtree[w, b, k]) or wp.abs(subtree[w, b, k]) > SAFE_MAGNITUDE:
                    reason_code[w] = BAD_NUMERIC
                    if not wp.isfinite(subtree[w, b, k]):
                        reason_kind[w] = 5
                    elif wp.abs(subtree[w, b, k]) > SAFE_MAGNITUDE:
                        reason_kind[w] = 6
                    else:
                        reason_kind[w] = 8
                    return
                else:
                    pass
    else:
        pass

@wp.kernel(enable_backward=False)
def _reason_scalar(qpos: wp.array(dtype=scalar.wp_real), active: wp.array(dtype=wp.int32), incoming: wp.array(dtype=wp.int32), capacity: wp.array(dtype=wp.int32), frictionloss: wp.array(dtype=scalar.wp_real), jnt_type: wp.array(dtype=wp.int32), limited: wp.array(dtype=wp.int32), qposadr: wp.array(dtype=wp.int32), dofadr: wp.array(dtype=wp.int32), ranges: wp.array(dtype=scalar.wp_real), margins: wp.array(dtype=scalar.wp_real), friction_J: wp.array(dtype=scalar.wp_real), friction_pos: wp.array(dtype=scalar.wp_real), friction_margin: wp.array(dtype=scalar.wp_real), friction_loss: wp.array(dtype=scalar.wp_real), friction_type: wp.array(dtype=wp.int32), friction_id: wp.array(dtype=wp.int32), friction_global_adr: wp.array(dtype=wp.int32), limit_J: wp.array(dtype=scalar.wp_real), limit_pos: wp.array(dtype=scalar.wp_real), limit_margin: wp.array(dtype=scalar.wp_real), limit_loss: wp.array(dtype=scalar.wp_real), limit_type: wp.array(dtype=wp.int32), limit_id: wp.array(dtype=wp.int32), limit_global_adr: wp.array(dtype=wp.int32), nf_delta: wp.array(dtype=wp.int32), nlimit_delta: wp.array(dtype=wp.int32), nefc_delta: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32), reason_code: wp.array(dtype=wp.int32), reason_kind: wp.array(dtype=wp.int32)):
    w = wp.tid()
    reason_code[w] = 0
    reason_kind[w] = 0
    if status[w] != 0:
        return
    else:
        pass
    if active[w] == 0:
        return
    else:
        pass
    if active[w] != 1:
        reason_code[w] = 1
        reason_kind[w] = 1
        return
    else:
        pass
    offset = incoming[w]
    cap = capacity[w]
    if offset < 0 or cap < 0 or offset > cap:
        reason_code[w] = 2
        if offset < 0:
            reason_kind[w] = 2
        elif cap < 0:
            reason_kind[w] = 2
        elif offset > cap:
            reason_kind[w] = 3
        else:
            reason_kind[w] = 8
        return
    else:
        pass
    for k in range(NQ):
        if not wp.isfinite(qpos[w * NQ + k]):
            reason_code[w] = 3
            reason_kind[w] = 5
            return
        else:
            pass
    nf = wp.int32(0)
    for i in range(NV):
        if frictionloss[i] > scalar.wp_real(0.0):
            nf += 1
        else:
            pass
    nl = wp.int32(0)
    for j in range(NJ):
        if limited[j] != 0:
            typ = jnt_type[j]
            if typ != 2 and typ != 3:
                reason_code[w] = 4
                reason_kind[w] = 4
                return
            else:
                pass
            value = qpos[w * NQ + qposadr[j]]
            margin = margins[j]
            lower = scalar.wp_real(-1.0) * (ranges[2 * j] - value)
            if lower < margin:
                nl += 1
            else:
                pass
            upper = scalar.wp_real(1.0) * (ranges[2 * j + 1] - value)
            if upper < margin:
                nl += 1
            else:
                pass
        else:
            pass
    need = nf + nl
    if nf > NF_CAP or nl > NL_CAP or need > cap - offset:
        reason_code[w] = 5
        reason_kind[w] = 3
        return
    else:
        pass

@wp.kernel(enable_backward=False)
def _reason_update(active: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32), ne: wp.array(dtype=wp.int32), nf: wp.array(dtype=wp.int32), ncon: wp.array(dtype=wp.int32), flag: wp.array(dtype=wp.int32), jar: wp.array(dtype=scalar.wp_real), D: wp.array(dtype=scalar.wp_real), R: wp.array(dtype=scalar.wp_real), loss: wp.array(dtype=scalar.wp_real), typ: wp.array(dtype=wp.int32), cid: wp.array(dtype=wp.int32), J: wp.array(dtype=scalar.wp_real), mu: wp.array(dtype=scalar.wp_real), fr: wp.array(dtype=scalar.wp_real), dim: wp.array(dtype=wp.int32), force: wp.array(dtype=scalar.wp_real), state: wp.array(dtype=wp.int32), H: wp.array(dtype=scalar.wp_real), cost: wp.array(dtype=scalar.wp_real), status: wp.array(dtype=wp.int32), rows: int, contacts: int, nv: int, reason_code: wp.array(dtype=wp.int32), reason_kind: wp.array(dtype=wp.int32)):
    w = wp.tid()
    reason_code[w] = 0
    reason_kind[w] = 0
    if active[w] == 0:
        return
    else:
        pass
    if active[w] != 1:
        reason_code[w] = 1
        reason_kind[w] = 1
        return
    else:
        pass
    n = count[w]
    nc = ncon[w]
    eq = ne[w]
    fcount = nf[w]
    if n < 0 or n > rows or nc < 0 or (nc > contacts) or (flag[w] != 0 and flag[w] != 1):
        reason_code[w] = 2
        if n < 0:
            reason_kind[w] = 2
        elif n > rows:
            reason_kind[w] = 3
        elif nc < 0:
            reason_kind[w] = 2
        elif nc > contacts:
            reason_kind[w] = 3
        elif flag[w] != 0 and flag[w] != 1:
            reason_kind[w] = 4
        else:
            reason_kind[w] = 8
        return
    else:
        pass
    if eq < 0 or eq > n:
        reason_code[w] = 2
        reason_kind[w] = 2
        return
    else:
        pass
    if fcount < 0 or fcount > n - eq:
        reason_code[w] = 2
        reason_kind[w] = 2
        return
    else:
        pass
    rb = w * rows
    cb = w * contacts
    i = eq + fcount
    while i < n:
        t = typ[rb + i]
        if t == 7:
            c = cid[rb + i]
            if c < 0 or c >= nc:
                reason_code[w] = 3
                reason_kind[w] = 2
                return
            else:
                pass
            d = dim[cb + c]
            if d != 1 and d != 3 and (d != 4) and (d != 6) or i + d > n or mu[cb + c] <= scalar.wp_real(0.0) or (not wp.isfinite(mu[cb + c])):
                reason_code[w] = 4
                if d != 1 and d != 3 and (d != 4) and (d != 6):
                    reason_kind[w] = 4
                elif i + d > n:
                    reason_kind[w] = 2
                elif mu[cb + c] <= scalar.wp_real(0.0):
                    reason_kind[w] = 6
                elif not wp.isfinite(mu[cb + c]):
                    reason_kind[w] = 5
                else:
                    reason_kind[w] = 8
                return
            else:
                pass
            for j in range(d):
                if typ[rb + i + j] != 7 or cid[rb + i + j] != c:
                    reason_code[w] = 5
                    reason_kind[w] = 4
                    return
                else:
                    pass
            for j in range(d - 1):
                if fr[(cb + c) * 5 + j] <= scalar.wp_real(0.0) or not wp.isfinite(fr[(cb + c) * 5 + j]):
                    reason_code[w] = 6
                    if fr[(cb + c) * 5 + j] <= scalar.wp_real(0.0):
                        reason_kind[w] = 6
                    elif not wp.isfinite(fr[(cb + c) * 5 + j]):
                        reason_kind[w] = 5
                    else:
                        reason_kind[w] = 8
                    return
                else:
                    pass
            i += d
        elif t == 0 or t == 1 or t == 2 or (t == 3) or (t == 4) or (t == 5) or (t == 6):
            i += 1
        else:
            reason_code[w] = 7
            reason_kind[w] = 4
            return
    for r in range(n):
        if not wp.isfinite(jar[rb + r]) or not wp.isfinite(D[rb + r]) or (not wp.isfinite(R[rb + r])) or (not wp.isfinite(loss[rb + r])):
            reason_code[w] = 8
            reason_kind[w] = 5
            return
        else:
            pass
        for j in range(nv):
            if not wp.isfinite(J[(rb + r) * nv + j]):
                reason_code[w] = 9
                reason_kind[w] = 5
                return
            else:
                pass

@wp.kernel(enable_backward=False)
def _reason_hessian(count: wp.array(dtype=wp.int32), ncon: wp.array(dtype=wp.int32), active: wp.array(dtype=wp.int32), kind: wp.array(dtype=wp.int32), state: wp.array(dtype=wp.int32), oldstate: wp.array(dtype=wp.int32), D: wp.array(dtype=scalar.wp_real), J: wp.array(dtype=scalar.wp_real), qm: wp.array(dtype=scalar.wp_real), H: wp.array(dtype=scalar.wp_real), efc_id: wp.array(dtype=wp.int32), contact_dim: wp.array(dtype=wp.int32), contact_H: wp.array(dtype=scalar.wp_real), ncone: wp.array(dtype=wp.int32), nnz: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32), route: wp.array(dtype=wp.int32), rows: int, contacts: int, reason_code: wp.array(dtype=wp.int32), reason_kind: wp.array(dtype=wp.int32)):
    w = wp.tid()
    reason_code[w] = 0
    reason_kind[w] = 0
    if active[w] == 0:
        return
    else:
        pass
    if active[w] != 1 or (kind[w] != 1 and kind[w] != 2 and (kind[w] != 3)):
        reason_code[w] = 1
        if active[w] != 1:
            reason_kind[w] = 1
        elif kind[w] != 1 and kind[w] != 2 and (kind[w] != 3):
            reason_kind[w] = 4
        else:
            reason_kind[w] = 8
        return
    else:
        pass
    nr = count[w]
    nc = ncon[w]
    if nr < 0 or nr > rows or nc < 0 or (nc > contacts):
        reason_code[w] = 2
        if nr < 0:
            reason_kind[w] = 2
        elif nr > rows:
            reason_kind[w] = 3
        elif nc < 0:
            reason_kind[w] = 2
        elif nc > contacts:
            reason_kind[w] = 3
        else:
            reason_kind[w] = 8
        return
    else:
        pass
    if kind[w] != 1 and nnz[w] != NV * NV:
        reason_code[w] = 3
        reason_kind[w] = 2
        return
    else:
        pass
    if kind[w] == 2 or kind[w] == 3:
        for i in range(NV):
            for j in range(NV):
                if (kind[w] == 2 or ncone[w] != 0 or j <= i) and (not wp.isfinite(H[w * NV * NV + i * NV + j])):
                    reason_code[w] = 4
                    reason_kind[w] = 5
                    return
                else:
                    pass
    else:
        pass
    cones = int(0)
    for r in range(nr):
        s = state[w * rows + r]
        if s < 0 or s > 4:
            reason_code[w] = 3
            reason_kind[w] = 4
            return
        else:
            pass
        if kind[w] == 3 and (oldstate[w * rows + r] < 0 or oldstate[w * rows + r] > 4):
            reason_code[w] = 3
            reason_kind[w] = 4
            return
        else:
            pass
        if s == 4:
            cones += 1
        else:
            pass
        reads_d = s == 1 and kind[w] == 1
        if kind[w] == 3 and (oldstate[w * rows + r] == 1) != (s == 1):
            reads_d = True
        else:
            pass
        if reads_d and (D[w * rows + r] < scalar.wp_real(0.0) or not wp.isfinite(D[w * rows + r])):
            reason_code[w] = 4
            if reads_d and D[w * rows + r] < scalar.wp_real(0.0):
                reason_kind[w] = 6
            elif reads_d and (not wp.isfinite(D[w * rows + r])):
                reason_kind[w] = 5
            else:
                reason_kind[w] = 8
            return
        else:
            pass
        reads_j = s == 4
        if kind[w] == 1 and s == 1 and (D[w * rows + r] != scalar.wp_real(0.0)):
            reads_j = True
        else:
            pass
        if kind[w] == 3 and (oldstate[w * rows + r] == 1) != (s == 1):
            reads_j = True
        else:
            pass
        if reads_j:
            for k in range(NV):
                if not wp.isfinite(J[(w * rows + r) * NV + k]):
                    reason_code[w] = 4
                    reason_kind[w] = 5
                    return
                else:
                    pass
        else:
            pass
    if cones != ncone[w]:
        reason_code[w] = 3
        reason_kind[w] = 2
        return
    else:
        pass
    r = int(0)
    while r < nr:
        if state[w * rows + r] == 4:
            con = efc_id[w * rows + r]
            if con < 0 or con >= nc:
                reason_code[w] = 3
                reason_kind[w] = 2
                return
            else:
                pass
            dim = contact_dim[w * contacts + con]
            if dim != 1 and dim != 3 and (dim != 4) and (dim != 6) or r + dim > nr:
                reason_code[w] = 3
                if dim != 1 and dim != 3 and (dim != 4) and (dim != 6):
                    reason_kind[w] = 4
                elif r + dim > nr:
                    reason_kind[w] = 2
                else:
                    reason_kind[w] = 8
                return
            else:
                pass
            for z in range(dim):
                if state[w * rows + r + z] != 4 or efc_id[w * rows + r + z] != con:
                    reason_code[w] = 3
                    reason_kind[w] = 4
                    return
                else:
                    pass
            for z in range(dim * dim):
                if not wp.isfinite(contact_H[(w * contacts + con) * 36 + z]):
                    reason_code[w] = 4
                    reason_kind[w] = 5
                    return
                else:
                    pass
            r += dim
        else:
            r += 1
    if kind[w] == 1:
        for z in range(NM):
            if not wp.isfinite(qm[w * NM + z]):
                reason_code[w] = 4
                reason_kind[w] = 5
                return
            else:
                pass
    else:
        pass

@wp.kernel(enable_backward=False)
def _reason_direction(Ma: wp.array(dtype=scalar.wp_real, ndim=2), smooth: wp.array(dtype=scalar.wp_real, ndim=2), constraint: wp.array(dtype=scalar.wp_real, ndim=2), H: wp.array(dtype=scalar.wp_real, ndim=3), Hcone: wp.array(dtype=scalar.wp_real, ndim=3), ncone: wp.array(dtype=wp.int32), nefc: wp.array(dtype=wp.int32), active: wp.array(dtype=wp.int32), upstream_status: wp.array(dtype=wp.int32), grad: wp.array(dtype=scalar.wp_real, ndim=2), Mgrad: wp.array(dtype=scalar.wp_real, ndim=2), candidate: wp.array(dtype=scalar.wp_real, ndim=2), status: wp.array(dtype=wp.int32), scratch_grad: wp.array(dtype=scalar.wp_real, ndim=2), scratch_x: wp.array(dtype=scalar.wp_real, ndim=2), scratch_fwd: wp.array(dtype=scalar.wp_real, ndim=2), reason_code: wp.array(dtype=wp.int32), reason_kind: wp.array(dtype=wp.int32)):
    w = wp.tid()
    reason_code[w] = 0
    reason_kind[w] = 0
    if upstream_status[w] != 0:
        reason_code[w] = upstream_status[w]
        reason_kind[w] = 8
        return
    else:
        pass
    if active[w] == 0:
        return
    else:
        pass
    if active[w] != 1:
        reason_code[w] = 1
        reason_kind[w] = 1
        return
    else:
        pass
    cone = ncone[w]
    n = nefc[w]
    if n < 0 or cone < 0 or cone > n:
        reason_code[w] = 2
        reason_kind[w] = 2
        return
    else:
        pass
    for i in range(NV):
        if not wp.isfinite(Ma[w, i]) or not wp.isfinite(smooth[w, i]) or (not wp.isfinite(constraint[w, i])):
            reason_code[w] = 3
            reason_kind[w] = 5
            return
        else:
            pass
        for j in range(i + 1):
            value = _factor(H, Hcone, w, cone, i, j)
            if not wp.isfinite(value):
                reason_code[w] = 3
                reason_kind[w] = 5
                return
            else:
                pass
    for i in range(NV):
        if _factor(H, Hcone, w, cone, i, i) == scalar.wp_real(0.0):
            reason_code[w] = 4
            reason_kind[w] = 6
            return
        else:
            pass

@wp.kernel(enable_backward=False)
def _reason_warm(active: wp.array(dtype=wp.int32), nefc: wp.array(dtype=wp.int32), warm_cost: wp.array(dtype=scalar.wp_real), smooth_cost: wp.array(dtype=scalar.wp_real), warm_qacc: wp.array(dtype=scalar.wp_real, ndim=2), smooth_qacc: wp.array(dtype=scalar.wp_real, ndim=2), smooth_force: wp.array(dtype=scalar.wp_real, ndim=2), smooth_state: wp.array(dtype=wp.int32, ndim=2), smooth_qfrc: wp.array(dtype=scalar.wp_real, ndim=2), qacc: wp.array(dtype=scalar.wp_real, ndim=2), force: wp.array(dtype=scalar.wp_real, ndim=2), state: wp.array(dtype=wp.int32, ndim=2), qfrc: wp.array(dtype=scalar.wp_real, ndim=2), choice: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32), nv: int, capacity: int, reason_code: wp.array(dtype=wp.int32), reason_kind: wp.array(dtype=wp.int32)):
    w = wp.tid()
    reason_code[w] = 0
    reason_kind[w] = 0
    if status[w] != 0:
        return
    else:
        pass
    if active[w] == 0:
        return
    else:
        pass
    if active[w] != 1:
        reason_code[w] = 1
        reason_kind[w] = 1
        return
    else:
        pass
    n = nefc[w]
    if n <= 0 or n > capacity:
        reason_code[w] = 2
        if n <= 0:
            reason_kind[w] = 2
        elif n > capacity:
            reason_kind[w] = 3
        else:
            reason_kind[w] = 8
        return
    else:
        pass
    if not wp.isfinite(warm_cost[w]) or not wp.isfinite(smooth_cost[w]):
        reason_code[w] = 3
        reason_kind[w] = 5
        return
    else:
        pass
    for i in range(nv):
        if not wp.isfinite(warm_qacc[w, i]) or not wp.isfinite(smooth_qacc[w, i]) or (not wp.isfinite(smooth_qfrc[w, i])):
            reason_code[w] = 3
            reason_kind[w] = 5
            return
        else:
            pass
    for j in range(n):
        if not wp.isfinite(smooth_force[w, j]):
            reason_code[w] = 3
            reason_kind[w] = 5
            return
        else:
            pass

@wp.kernel(enable_backward=False)
def _reason_step(active: wp.array(dtype=wp.int32), nefc: wp.array(dtype=wp.int32), alpha: wp.array(dtype=scalar.wp_real), search: wp.array(dtype=scalar.wp_real, ndim=2), mv: wp.array(dtype=scalar.wp_real, ndim=2), jv: wp.array(dtype=scalar.wp_real, ndim=2), qacc: wp.array(dtype=scalar.wp_real, ndim=2), ma: wp.array(dtype=scalar.wp_real, ndim=2), jaref: wp.array(dtype=scalar.wp_real, ndim=2), committed: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32), nv: int, capacity: int, reason_code: wp.array(dtype=wp.int32), reason_kind: wp.array(dtype=wp.int32)):
    w = wp.tid()
    reason_code[w] = 0
    reason_kind[w] = 0
    if status[w] != 0:
        return
    else:
        pass
    if active[w] == 0:
        return
    else:
        pass
    if active[w] != 1:
        reason_code[w] = 1
        reason_kind[w] = 1
        return
    else:
        pass
    n = nefc[w]
    if n <= 0 or n > capacity:
        reason_code[w] = 2
        if n <= 0:
            reason_kind[w] = 2
        elif n > capacity:
            reason_kind[w] = 3
        else:
            reason_kind[w] = 8
        return
    else:
        pass

@wp.kernel(enable_backward=False)
def _reason_stats(active: wp.array(dtype=wp.int32), nefc: wp.array(dtype=wp.int32), stage: wp.array(dtype=wp.int32), committed: wp.array(dtype=wp.int32), oldstate: wp.array(dtype=wp.int32, ndim=2), newstate: wp.array(dtype=wp.int32, ndim=2), grad: wp.array(dtype=scalar.wp_real, ndim=2), oldcost: wp.array(dtype=scalar.wp_real), newcost: wp.array(dtype=scalar.wp_real), meaninertia: wp.array(dtype=scalar.wp_real), tolerance: wp.array(dtype=scalar.wp_real), lineslope: wp.array(dtype=scalar.wp_real), nactive: wp.array(dtype=wp.int32), nupdate: wp.array(dtype=wp.int32), neval: wp.array(dtype=wp.int32), iteration: wp.array(dtype=wp.int32), solver_iter_base: wp.array(dtype=wp.int32), maxiter: wp.array(dtype=wp.int32), stats_f: wp.array(dtype=scalar.wp_real, ndim=3), stats_i: wp.array(dtype=wp.int32, ndim=3), out_nchange: wp.array(dtype=wp.int32), out_improvement: wp.array(dtype=scalar.wp_real), out_gradient: wp.array(dtype=scalar.wp_real), out_stop: wp.array(dtype=wp.int32), out_reason: wp.array(dtype=wp.int32), out_iter: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32), nv: int, capacity: int, reason_code: wp.array(dtype=wp.int32), reason_kind: wp.array(dtype=wp.int32)):
    w = wp.tid()
    reason_code[w] = 0
    reason_kind[w] = 0
    if status[w] != 0:
        return
    else:
        pass
    if active[w] == 0:
        return
    else:
        pass
    if active[w] != 1:
        reason_code[w] = 1
        reason_kind[w] = 1
        return
    else:
        pass
    n = nefc[w]
    it = iteration[w]
    base = solver_iter_base[w]
    if n <= 0 or n > capacity or it < 0 or (base < 0) or (base > 2147483647 - it) or (it >= maxiter[w]) or (maxiter[w] < 0) or (maxiter[w] > 1000000):
        reason_code[w] = 2
        if n <= 0:
            reason_kind[w] = 2
        elif n > capacity:
            reason_kind[w] = 3
        elif it < 0:
            reason_kind[w] = 2
        elif base < 0:
            reason_kind[w] = 2
        elif base > 2147483647 - it:
            reason_kind[w] = 2
        elif it >= maxiter[w]:
            reason_kind[w] = 2
        elif maxiter[w] < 0:
            reason_kind[w] = 2
        elif maxiter[w] > 1000000:
            reason_kind[w] = 2
        else:
            reason_kind[w] = 8
        return
    else:
        pass
    if stage[w] != 1 or committed[w] != 1:
        reason_code[w] = 4
        reason_kind[w] = 4
        return
    else:
        pass
    inertia = meaninertia[w]
    tol = tolerance[w]
    if not wp.isfinite(oldcost[w]) or not wp.isfinite(newcost[w]) or (not wp.isfinite(inertia)) or (not wp.isfinite(tol)) or (not wp.isfinite(lineslope[w])) or (inertia <= scalar.wp_real(0.0)) or (tol < scalar.wp_real(0.0)):
        reason_code[w] = 3
        if not wp.isfinite(oldcost[w]):
            reason_kind[w] = 5
        elif not wp.isfinite(newcost[w]):
            reason_kind[w] = 5
        elif not wp.isfinite(inertia):
            reason_kind[w] = 5
        elif not wp.isfinite(tol):
            reason_kind[w] = 5
        elif not wp.isfinite(lineslope[w]):
            reason_kind[w] = 5
        elif inertia <= scalar.wp_real(0.0):
            reason_kind[w] = 6
        elif tol < scalar.wp_real(0.0):
            reason_kind[w] = 6
        else:
            reason_kind[w] = 8
        return
    else:
        pass
    if nactive[w] < 0 or nupdate[w] < 0 or neval[w] < 0:
        reason_code[w] = 2
        reason_kind[w] = 2
        return
    else:
        pass
    nchange = wp.int32(0)
    for j in range(n):
        if oldstate[w, j] != newstate[w, j]:
            nchange += 1
        else:
            pass
    for i in range(nv):
        if not wp.isfinite(grad[w, i]):
            reason_code[w] = 3
            reason_kind[w] = 5
            return
        else:
            pass
