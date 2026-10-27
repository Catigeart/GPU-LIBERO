"""P06-C complete ordered collision device chain.

Stage order (device counts drive everything; nothing dynamic is downloaded
to decide scheduling):

  1 bounds       - covariance-aligned frame (mju_eig3 algorithm) + body AABBs
  2 broadphase   - special pairs + FP32-key SAP + signature-sorted body pairs
  3 stagebuild   - signature dedup, predefined-pair merge, exclude, route
                   decision (tree when body_geomnum product > 1 and midphase
                   enabled, else flat), body-pair segment ids
  4 invoke       - libero.gpu.reference.collision.collision_tree.CollisionTreeWorkspace (C04,
                   Warp): ordered geom requests + bvh_active
  5 dispatch     - libero.gpu.reference.collision.collision_dispatch.CollisionDispatchWorkspace
                   (C05, Warp): calls, margin/gap, mid_count
  6 detect       - detectors._detect_kernel: raw contacts or explicit
                   UNSUPPORTED_ROUTE first_error
  7 finalize     - libero.gpu.reference.collision.contact_finalize.ContactFinalizeWorkspace
                   (C06, Warp): ordered contacts, BoxBox dedup, segment sort

First-error protocol: per-world sticky (stage, code, item) written on device
in source check order (each stage kernel is one thread per world). Any stage
error blocks the finalize commit for that world; unwritten outputs keep the
dirty prefill. Unimplemented routes never return a clean zero-contact set.
"""
from __future__ import annotations
import libero.gpu.precision as scalar

import numpy as np
import warp as wp

from .errors import ERR_BAD_ACTIVE, ERR_BAD_COUNT, ERR_CAPACITY, ERR_INTERNAL, ERR_NONFINITE, ERR_UNSUPPORTED_ROUTE, ERR_UPSTREAM_MODULE, STAGE_SAP, STAGE_STAGEBUILD, STAGE_TREE, STAGE_DISPATCH, STAGE_DETECT, STAGE_FINALIZE

wp.set_module_options({'enable_backward': False, 'fuse_fp': True, 'fast_math': False})

MINVAL = scalar.wp_real(1.0e-15)


# --------------------------------------------------------------------- bounds


@wp.func
def _q2m(out: wp.array(dtype=scalar.wp_real), o: wp.int32,
         w: scalar.wp_real, x: scalar.wp_real, y: scalar.wp_real, z: scalar.wp_real):
    # mju_quat2Mat, engine_util_spatial.c:144-180 (row-major, exact form).
    # null-quat identity branch first (original res[0..8] = identity).
    if w == scalar.wp_real(1.0) and x == scalar.wp_real(0.0) and y == scalar.wp_real(0.0) and z == scalar.wp_real(0.0):
        out[o] = scalar.wp_real(1.0)
        out[o + 1] = scalar.wp_real(0.0)
        out[o + 2] = scalar.wp_real(0.0)
        out[o + 3] = scalar.wp_real(0.0)
        out[o + 4] = scalar.wp_real(1.0)
        out[o + 5] = scalar.wp_real(0.0)
        out[o + 6] = scalar.wp_real(0.0)
        out[o + 7] = scalar.wp_real(0.0)
        out[o + 8] = scalar.wp_real(1.0)
    else:
        q00 = w * w
        q01 = w * x
        q02 = w * y
        q03 = w * z
        q11 = x * x
        q12 = x * y
        q13 = x * z
        q22 = y * y
        q23 = y * z
        q33 = z * z
        out[o] = q00 + q11 - q22 - q33
        out[o + 4] = q00 - q11 + q22 - q33
        out[o + 8] = q00 - q11 - q22 + q33
        out[o + 1] = scalar.wp_real(2.0) * (q12 - q03)
        out[o + 2] = scalar.wp_real(2.0) * (q13 + q02)
        out[o + 3] = scalar.wp_real(2.0) * (q12 + q03)
        out[o + 5] = scalar.wp_real(2.0) * (q23 - q01)
        out[o + 6] = scalar.wp_real(2.0) * (q13 - q02)
        out[o + 7] = scalar.wp_real(2.0) * (q23 + q01)


@wp.func
def _mulq(ar: wp.array(dtype=scalar.wp_real), ai: wp.int32,
          br: wp.array(dtype=scalar.wp_real), bi: wp.int32,
          dst: wp.array(dtype=scalar.wp_real), di: wp.int32):
    a0 = ar[ai]
    a1 = ar[ai + 1]
    a2 = ar[ai + 2]
    a3 = ar[ai + 3]
    b0 = br[bi]
    b1 = br[bi + 1]
    b2 = br[bi + 2]
    b3 = br[bi + 3]
    dst[di] = a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3
    dst[di + 1] = a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2
    dst[di + 2] = a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1
    dst[di + 3] = a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0


@wp.func
def _norm4(x: wp.array(dtype=scalar.wp_real), o: wp.int32) -> bool:
    n = wp.sqrt(x[o] * x[o] + x[o + 1] * x[o + 1] + x[o + 2] * x[o + 2] + x[o + 3] * x[o + 3])
    if not (n == n) or n > scalar.wp_real(scalar.norm_limit):
        return False
    if n < MINVAL:
        x[o] = scalar.wp_real(1.0)
        x[o + 1] = scalar.wp_real(0.0)
        x[o + 2] = scalar.wp_real(0.0)
        x[o + 3] = scalar.wp_real(0.0)
    else:
        inv = scalar.wp_real(1.0) / n
        x[o] *= inv
        x[o + 1] *= inv
        x[o + 2] *= inv
        x[o + 3] *= inv
    return True


@wp.func
def _eig3(cov: wp.array(dtype=scalar.wp_real), co: wp.int32,
          eig: wp.array(dtype=scalar.wp_real), eo: wp.int32,
          frame: wp.array(dtype=scalar.wp_real), fo: wp.int32,
          quat: wp.array(dtype=scalar.wp_real), qo: wp.int32,
          tmp: wp.array(dtype=scalar.wp_real), tm: wp.int32,
          scratch: wp.array(dtype=scalar.wp_real), sc: wp.int32) -> wp.int32:
    """Port of the accepted C02 c02_eig3 (500-iter quaternion Jacobi,
    strict offdiagonal pivot ties, three strict eigenvalue sort passes).
    Returns iteration count or -1 on nonfinite."""
    quat[qo] = scalar.wp_real(1.0)
    quat[qo + 1] = scalar.wp_real(0.0)
    quat[qo + 2] = scalar.wp_real(0.0)
    quat[qo + 3] = scalar.wp_real(0.0)
    off_tolerance = scalar.wp_real(1.0e-12)
    if scalar.bits == 32:
        norm = scalar.wp_real(0.0)
        for row in range(3):
            row_sum = scalar.wp_real(0.0)
            for col in range(3):
                row_sum += wp.abs(cov[co + 3 * row + col])
            norm = wp.max(norm, row_sum)
        off_tolerance = wp.max(off_tolerance, scalar.wp_real(8.0 * scalar.epsilon) * norm)
    iters = wp.int32(0)
    converged = wp.int32(0)
    for it in range(500):
        iters = it
        _q2m(frame, fo, quat[qo], quat[qo + 1], quat[qo + 2], quat[qo + 3])
        # D = frame^T * cov * frame
        for i in range(3):
            for j in range(3):
                acc = scalar.wp_real(0.0)
                for k in range(3):
                    acc += frame[fo + 3 * k + i] * cov[co + 3 * k + j]
                scratch[sc + 3 * i + j] = acc
        for i in range(3):
            for j in range(3):
                acc = scalar.wp_real(0.0)
                for k in range(3):
                    acc += scratch[sc + 3 * i + k] * frame[fo + 3 * k + j]
                tmp[tm + 3 * i + j] = acc
        eig[eo] = tmp[tm]
        eig[eo + 1] = tmp[tm + 4]
        eig[eo + 2] = tmp[tm + 8]
        rk = wp.int32(0)
        ck = wp.int32(1)
        rotk = wp.int32(2)
        d01 = tmp[tm + 1]
        if d01 < scalar.wp_real(0.0):
            d01 = -d01
        d02 = tmp[tm + 2]
        if d02 < scalar.wp_real(0.0):
            d02 = -d02
        d12 = tmp[tm + 5]
        if d12 < scalar.wp_real(0.0):
            d12 = -d12
        if d01 > d02 and d01 > d12:
            rk = 0
            ck = 1
            rotk = 2
        elif d02 > d12:
            rk = 0
            ck = 2
            rotk = 1
        else:
            rk = 1
            ck = 2
            rotk = 0
        off = tmp[tm + 3 * rk + ck]
        aoff = off
        if aoff < scalar.wp_real(0.0):
            aoff = -aoff
        if aoff < off_tolerance:
            converged = wp.int32(1)
            break
        tau = (tmp[tm + 4 * ck] - tmp[tm + 4 * rk]) / (scalar.wp_real(2.0) * off)
        t = scalar.wp_real(0.0)
        if tau >= scalar.wp_real(0.0):
            t = scalar.wp_real(1.0) / (tau + wp.sqrt(scalar.wp_real(1.0) + tau * tau))
        else:
            t = scalar.wp_real(-1.0) / (-tau + wp.sqrt(scalar.wp_real(1.0) + tau * tau))
        c = scalar.wp_real(1.0) / wp.sqrt(scalar.wp_real(1.0) + t * t)
        if scalar.bits == 64 and c > scalar.wp_real(1.0) - scalar.wp_real(1.0e-12):
            converged = wp.int32(1)
            break
        # express rotation as quaternion (engine_util_solve.c:770-774)
        #   tmp[rotk+1] = (tau >= 0 ? -sqrt(0.5-0.5*c) : +sqrt(0.5-0.5*c));
        #   if (rotk == 1) tmp[rotk+1] = -tmp[rotk+1];
        tmp[tm] = scalar.wp_real(0.0)
        tmp[tm + 1] = scalar.wp_real(0.0)
        tmp[tm + 2] = scalar.wp_real(0.0)
        tmp[tm + 3] = scalar.wp_real(0.0)
        s = scalar.wp_real(0.0)
        if scalar.bits == 32:
            s = wp.abs(t) * c / wp.sqrt(scalar.wp_real(2.0) * (scalar.wp_real(1.0) + c))
        else:
            s = wp.sqrt(scalar.wp_real(0.5) - scalar.wp_real(0.5) * c)
        if tau >= scalar.wp_real(0.0):
            s = -s
        if rotk == 1:
            s = -s
        tmp[tm + rotk + 1] = s
        tmp[tm] = wp.sqrt(scalar.wp_real(1.0) - s * s)
        if not _norm4(tmp, tm):
            return wp.int32(-1)
        # quat = quat * tmp  (mulquat writes into scratch then copy)
        _mulq(quat, qo, tmp, tm, scratch, sc)
        for k in range(4):
            quat[qo + k] = scratch[sc + k]
        if not _norm4(quat, qo):
            return wp.int32(-1)
    # three strict eigenvalue sort passes (adjacent swap j1=j%2 pattern)
    for j in range(3):
        j1 = j % 2
        if eig[eo + j1] < eig[eo + j1 + 1]:
            tv = eig[eo + j1]
            eig[eo + j1] = eig[eo + j1 + 1]
            eig[eo + j1 + 1] = tv
            tmp[tm] = scalar.wp_real(0.707106781186548)
            tmp[tm + 1] = scalar.wp_real(0.0)
            tmp[tm + 2] = scalar.wp_real(0.0)
            tmp[tm + 3] = scalar.wp_real(0.0)
            tmp[tm + (j1 + 2) % 3 + 1] = tmp[tm]
            _mulq(quat, qo, tmp, tm, scratch, sc)
            for k in range(4):
                quat[qo + k] = scratch[sc + k]
            if not _norm4(quat, qo):
                return wp.int32(-1)
    _q2m(frame, fo, quat[qo], quat[qo + 1], quat[qo + 2], quat[qo + 3])
    if converged == 0:
        return wp.int32(-2)  # R01-32: non-converging Jacobi must not pass
    return iters


@wp.kernel(enable_backward=False)
def _bounds_kernel(
        xpos: wp.array(dtype=scalar.wp_real), active: wp.array(dtype=wp.int32),
        gates: wp.array(dtype=wp.int32), collision_mode: wp.array(dtype=wp.int32),
        geom_rbound: wp.array(dtype=scalar.wp_real), geom_margin: wp.array(dtype=scalar.wp_real),
        body_geomadr: wp.array(dtype=wp.int32), body_geomnum: wp.array(dtype=wp.int32),
        geom_bodyid: wp.array(dtype=wp.int32),
        aabb: wp.array(dtype=scalar.wp_real), frame: wp.array(dtype=scalar.wp_real),
        entered: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32),
        status: wp.array(dtype=wp.int32),
        scratch: wp.array(dtype=scalar.wp_real),
        ngeom: int, nbody: int):
    """C02-equivalent bounds: cen/cov over nonworld geoms (ascending geom id),
    eig3, then per-body AABB projecting onto frame rows with rbound then margin."""
    w = wp.tid()
    status[w] = 0
    entered[w] = 0
    count[w] = 0
    if active[w] == 0:
        status[w] = 1
        return
    if active[w] != 1:
        status[w] = 2
        return
    if gates[0] != 0 or gates[1] != 0 or gates[2] == 0 or nbody < 2 or collision_mode[0] == 1:
        return
    gp = w * ngeom * 3
    cx = scalar.wp_real(0.0)
    cy = scalar.wp_real(0.0)
    cz = scalar.wp_real(0.0)
    cnt = wp.int32(0)
    for i in range(ngeom):
        if geom_bodyid[i] != 0:
            px = xpos[gp + 3 * i]
            py = xpos[gp + 3 * i + 1]
            pz = xpos[gp + 3 * i + 2]
            if not (px == px and py == py and pz == pz):
                status[w] = 3
                return
            cx += px
            cy += py
            cz += pz
            cnt += 1
    if cnt == 0:
        # accepted C02: entered stays 0 on the zero-nonworld-geom early exit
        return
    cx /= scalar.wp_real(cnt)
    cy /= scalar.wp_real(cnt)
    cz /= scalar.wp_real(cnt)
    # scratch layout per world (34 doubles):
    #   [0:9] cov, [9:18] eig3 D/tmp, [18:27] eig3 work, [27:30] eigval,
    #   [30:34] quat
    sc = w * 34
    for k in range(9):
        scratch[sc + k] = scalar.wp_real(0.0)
    for i in range(ngeom):
        if geom_bodyid[i] != 0:
            dx = xpos[gp + 3 * i] - cx
            dy = xpos[gp + 3 * i + 1] - cy
            dz = xpos[gp + 3 * i + 2] - cz
            scratch[sc] += dx * dx
            scratch[sc + 1] += dx * dy
            scratch[sc + 2] += dx * dz
            scratch[sc + 3] += dx * dy
            scratch[sc + 4] += dy * dy
            scratch[sc + 5] += dy * dz
            scratch[sc + 6] += dx * dz
            scratch[sc + 7] += dy * dz
            scratch[sc + 8] += dz * dz
    # original: cov[i] /= cnt (engine_collision_driver.c:729); division, not
    # multiplication by a reciprocal (R01-33 bit-exactness).
    for k in range(9):
        scratch[sc + k] = scratch[sc + k] / scalar.wp_real(cnt)
    iters = _eig3(scratch, sc, scratch, sc + 27, frame, w * 9,
                  scratch, sc + 30, scratch, sc + 9, scratch, sc + 18)
    if iters < 0:
        status[w] = 5
        return
    entered[w] = 1
    count[w] = cnt
    for b in range(nbody):
        if b == 0 or body_geomnum[b] == 0:
            for k in range(6):
                aabb[w * nbody * 6 + 6 * b + k] = scalar.wp_real(0.0)
            continue
        first = wp.bool(True)
        for i in range(body_geomnum[b]):
            geom = body_geomadr[b] + i
            for j in range(3):
                cen = (xpos[gp + 3 * geom] * frame[w * 9 + 3 * j]
                       + xpos[gp + 3 * geom + 1] * frame[w * 9 + 3 * j + 1]
                       + xpos[gp + 3 * geom + 2] * frame[w * 9 + 3 * j + 2])
                lo = cen - geom_rbound[geom] - geom_margin[geom]
                hi = cen + geom_rbound[geom] + geom_margin[geom]
                idx = w * nbody * 6 + 6 * b + 2 * j
                if first:
                    aabb[idx] = lo
                    aabb[idx + 1] = hi
                else:
                    if aabb[idx] > lo:
                        aabb[idx] = lo
                    if aabb[idx + 1] < hi:
                        aabb[idx + 1] = hi
            first = False
    return


# ----------------------------------------------------------------- broadphase


@wp.func
def _can_collide(geom_contype: wp.array(dtype=wp.int32),
                 geom_conaffinity: wp.array(dtype=wp.int32),
                 geomadr: wp.int32, geomnum: wp.int32) -> wp.int32:
    for g in range(geomnum):
        ind = geomadr + g
        if geom_contype[ind] != 0 or geom_conaffinity[ind] != 0:
            return wp.int32(1)
    return wp.int32(0)


@wp.kernel(enable_backward=False)
def _broadphase_kernel(
        aabb: wp.array(dtype=scalar.wp_real), active: wp.array(dtype=wp.int32),
        gates: wp.array(dtype=wp.int32), collision_mode: wp.array(dtype=wp.int32),
        body_weldid: wp.array(dtype=wp.int32), body_parentid: wp.array(dtype=wp.int32),
        body_geomadr: wp.array(dtype=wp.int32), body_geomnum: wp.array(dtype=wp.int32),
        geom_bodyid: wp.array(dtype=wp.int32), geom_type: wp.array(dtype=wp.int32),
        geom_contype: wp.array(dtype=wp.int32), geom_conaffinity: wp.array(dtype=wp.int32),
        can: wp.array(dtype=wp.int32), plane: wp.array(dtype=wp.int32),
        filterparent_disabled: wp.array(dtype=wp.int32),
        out_pair: wp.array(dtype=wp.int32), out_npair: wp.array(dtype=wp.int32),
        status: wp.array(dtype=wp.int32),
        sortbuf: wp.array(dtype=wp.float32), sorttag: wp.array(dtype=wp.int32),
        activebuf: wp.array(dtype=wp.int32),
        nbody: int, pair_capacity: int):
    """mj_broadphase port: special pairs first (b1 outer order), then
    FP32-key SAP with stable-equal insertion sort, y/z prune, single-geom
    affinity filter, final signature sort (ascending, duplicates kept)."""
    w = wp.tid()
    status[w] = 0
    out_npair[w] = 0
    if active[w] == 0:
        return
    if active[w] != 1:
        status[w] = 2
        return
    if gates[0] != 0 or gates[1] != 0 or gates[2] == 0 or nbody < 2 or collision_mode[0] == 1:
        return
    npair = wp.int32(0)
    cap = pair_capacity
    ab = w * nbody * 6
    # ---- special pairs: world-with-geoms or welded plane/hfield body vs all
    for b1 in range(nbody):
        if can[b1] == 0:
            continue
        special = wp.bool(False)
        if b1 == 0 and body_geomnum[b1] > 0:
            special = True
        if body_weldid[b1] == 0 and plane[b1] != 0:
            special = True
        if not special:
            continue
        for b2 in range(nbody):
            w2 = body_weldid[b2]
            # body_pair_filter(weld1=0, parent_weld1=0, w2, parent_weld2, disabled):
            # same-weld check (0 == w2) discards only the world-welded body;
            # the weldparent clause requires weldbody1 != 0 so it never applies
            # on this special-pair path (original body_pair_filter).
            if w2 == 0:
                continue
            if npair >= cap:
                status[w] = 1
                return
            if b1 < b2:
                out_pair[w * pair_capacity + npair] = (b1 << 16) + b2
            else:
                out_pair[w * pair_capacity + npair] = (b2 << 16) + b1
            npair += 1
    # ---- nonworld geom count (original returns early when zero)
    cnt = wp.int32(0)
    for b in range(1, nbody):
        cnt += body_geomnum[b]
    if cnt == 0:
        out_npair[w] = npair
        return
    # ---- SAP on axis0 with FP32 keys (original casts endpoints to float)
    ne = wp.int32(0)
    for b in range(1, nbody):
        if can[b] != 0:
            sortbuf[w * 2 * nbody + ne] = wp.float32(aabb[ab + 6 * b])
            sorttag[w * 2 * nbody + ne] = b
            ne += 1
            sortbuf[w * 2 * nbody + ne] = wp.float32(aabb[ab + 6 * b + 1])
            sorttag[w * 2 * nbody + ne] = b + 65536
            ne += 1
    # stable insertion sort on FP32 keys: swap only when strictly '<'
    for i in range(1, ne):
        keyv = sortbuf[w * 2 * nbody + i]
        keyt = sorttag[w * 2 * nbody + i]
        j = i
        while j > 0:
            if not keyv < sortbuf[w * 2 * nbody + j - 1]:
                break
            sortbuf[w * 2 * nbody + j] = sortbuf[w * 2 * nbody + j - 1]
            sorttag[w * 2 * nbody + j] = sorttag[w * 2 * nbody + j - 1]
            j -= 1
        sortbuf[w * 2 * nbody + j] = keyv
        sorttag[w * 2 * nbody + j] = keyt
    nactive = wp.int32(0)
    for i in range(ne):
        tag = sorttag[w * 2 * nbody + i]
        if (tag & 65536) == 0:
            b2 = tag
            for j in range(nactive):
                b1 = activebuf[w * nbody + j]
                w1 = body_weldid[b1]
                w2 = body_weldid[b2]
                p1 = body_weldid[body_parentid[w1]]
                p2 = body_weldid[body_parentid[w2]]
                # body_pair_filter(w1,p1,w2,p2,disabled)
                skip = wp.bool(False)
                if w1 == w2:
                    skip = True
                if filterparent_disabled[0] == 0 and w1 != 0 and w2 != 0:
                    if w1 == p2 or w2 == p1:
                        skip = True
                if skip:
                    continue
                a1 = ab + 6 * b1
                a2 = ab + 6 * b2
                if (aabb[a1 + 2] > aabb[a2 + 3] or aabb[a1 + 3] < aabb[a2 + 2]
                        or aabb[a1 + 4] > aabb[a2 + 5] or aabb[a1 + 5] < aabb[a2 + 4]):
                    continue
                # add_pair with single-geom affinity filter
                if npair >= cap:
                    status[w] = 1
                    return
                add = wp.bool(True)
                if body_geomnum[b1] == 1 and body_geomnum[b2] == 1:
                    g1 = body_geomadr[b1]
                    g2 = body_geomadr[b2]
                    if ((geom_contype[g1] & geom_conaffinity[g2]) == 0
                            and (geom_contype[g2] & geom_conaffinity[g1]) == 0):
                        add = False
                if add:
                    if b1 < b2:
                        out_pair[w * pair_capacity + npair] = (b1 << 16) + b2
                    else:
                        out_pair[w * pair_capacity + npair] = (b2 << 16) + b1
                    npair += 1
            activebuf[w * nbody + nactive] = b2
            nactive += 1
        else:
            b1 = tag - 65536
            for j in range(nactive):
                if activebuf[w * nbody + j] == b1:
                    for k in range(j, nactive - 1):
                        activebuf[w * nbody + k] = activebuf[w * nbody + k + 1]
                    nactive -= 1
                    break
    # ---- final signature sort (ascending)
    for i in range(1, npair):
        key = out_pair[w * pair_capacity + i]
        j = i
        while j > 0:
            if not key < out_pair[w * pair_capacity + j - 1]:
                break
            out_pair[w * pair_capacity + j] = out_pair[w * pair_capacity + j - 1]
            j -= 1
        out_pair[w * pair_capacity + j] = key
    out_npair[w] = npair


# ---------------------------------------------------------------- stagebuild


@wp.kernel(enable_backward=False)
def _stagebuild_kernel(
        pair: wp.array(dtype=wp.int32), npair_arr: wp.array(dtype=wp.int32),
        active: wp.array(dtype=wp.int32),
        body_geomnum: wp.array(dtype=wp.int32),
        pair_geom1: wp.array(dtype=wp.int32), pair_geom2: wp.array(dtype=wp.int32),
        pair_signature: wp.array(dtype=wp.int32),
        exclude_signature: wp.array(dtype=wp.int32),
        midphase_disabled: wp.array(dtype=wp.int32),
        gates: wp.array(dtype=wp.int32), collision_mode: wp.array(dtype=wp.int32),
        out_stage: wp.array(dtype=wp.int32), out_stage_count: wp.array(dtype=wp.int32),
        out_stage_route: wp.array(dtype=wp.int32), out_merged: wp.array(dtype=wp.int32),
        out_seg_count: wp.array(dtype=wp.int32),
        out_seg_pair_start: wp.array(dtype=wp.int32),
        out_seg_pair_end: wp.array(dtype=wp.int32),
        out_seg_geom_stage: wp.array(dtype=wp.int32),
        out_trailing_start: wp.array(dtype=wp.int32),
        out_trailing_end: wp.array(dtype=wp.int32),
        out_pairadr: wp.array(dtype=wp.int32),
        status: wp.array(dtype=wp.int32), first_item: wp.array(dtype=wp.int32),
        nbody: int, pair_capacity: int, stage_slots: int, npair_static: int,
        nexclude: int):
    """mj_collision driver loop: signature dedup, predefined-pair merge,
    exclude, route decision. Stage tuple: (b1, b2, merged, start, end, route)
    matching C04's stage ABI; route 1=tree, 2=flat.

    Driver gates (engine_collision_driver.c:390-405): contact/constraint
    disabled, nconmax==0 or nbody<2 emit no stages; mjCOL_PAIR emits only
    predefined pairs (segments with geom_stage=-1); mjCOL_DYNAMIC skips the
    predefined merge. Every consumed merge range is recorded as a segment so
    pair invocations survive body-pair exclusion (R01-19).
    """
    w = wp.tid()
    status[w] = 0
    out_stage_count[w] = 0
    out_seg_count[w] = 0
    out_pairadr[w] = 0
    first_item[w] = -1
    if active[w] == 0:
        return
    if active[w] != 1:
        status[w] = 1
        return
    # driver gates: no collision work at all
    if gates[0] != 0 or gates[1] != 0 or gates[2] == 0 or nbody < 2:
        out_trailing_start[w] = 0
        out_trailing_end[w] = 0
        return
    if collision_mode[0] == 1:
        # Source disabled gates precede the predefined-only branch.
        out_trailing_start[w] = 0
        out_trailing_end[w] = npair_static
        return
    n = npair_arr[w]
    if n < 0 or n > pair_capacity:
        status[w] = 2
        return
    scount = wp.int32(0)   # geom stages (C04 stage ids)
    segcount = wp.int32(0) # all segments incl. pair-only
    last_signature = wp.int32(-1)
    pairadr = wp.int32(0)
    exadr = wp.int32(0)
    mode = collision_mode[0]
    for i in range(n):
        packed = pair[w * pair_capacity + i]
        b1 = (packed >> 16) & 65535
        b2 = packed & 65535
        signature = ((b1 + 1) << 16) + (b2 + 1)
        if signature == last_signature:
            continue
        last_signature = signature
        merged = wp.int32(0)
        startadr = pairadr
        # merge predefined pairs (engine_collision_driver.c:433-442);
        # mjCOL_ALL only, and only when pairs exist
        if npair_static != 0 and mode == 0:
            while pairadr < npair_static and pair_signature[pairadr] <= signature:
                if pair_signature[pairadr] == signature:
                    merged = 1
                pairadr += 1
        # exclude
        skip = wp.bool(False)
        while exadr < nexclude and exclude_signature[exadr] < signature:
            exadr += 1
        if exadr < nexclude and exclude_signature[exadr] == signature:
            skip = True
        # body pair geometry stage
        want_geom = wp.bool(False)
        if not skip and body_geomnum[b1] != 0 and body_geomnum[b2] != 0:
            want_geom = True
        # pair range for this body pair (invoked even if the body pair is
        # excluded or empty — original calls pairs before the exclude check)
        if pairadr > startadr or want_geom:
            if segcount >= stage_slots:
                status[w] = 3
                first_item[w] = segcount
                return
            seg_s = w * stage_slots + segcount
            out_seg_pair_start[seg_s] = startadr
            out_seg_pair_end[seg_s] = pairadr
            if want_geom:
                route = wp.int32(2)
                if midphase_disabled[0] == 0 and body_geomnum[b1] * body_geomnum[b2] > 1:
                    route = 1
                so = (w * stage_slots + scount) * 6
                out_stage[so] = b1
                out_stage[so + 1] = b2
                out_stage[so + 2] = merged
                out_stage[so + 3] = startadr
                out_stage[so + 4] = pairadr
                out_stage[so + 5] = route
                out_stage_route[w * stage_slots + scount] = route
                out_merged[w * stage_slots + scount] = merged
                out_seg_geom_stage[seg_s] = scount
                scount += 1
            else:
                out_seg_geom_stage[seg_s] = -1
            segcount += 1
    # trailing flush of predefined pairs (engine_collision_driver.c:481-485)
    out_trailing_start[w] = pairadr
    out_trailing_end[w] = npair_static if (npair_static != 0 and mode == 0) else pairadr
    out_pairadr[w] = pairadr
    out_stage_count[w] = scount
    out_seg_count[w] = segcount


# ------------------------------------------------------------- call adapters


@wp.kernel(enable_backward=False)
def _req_to_dispatch_kernel(
        out_req: wp.array(dtype=wp.int32), req_count: wp.array(dtype=wp.int32),
        active: wp.array(dtype=wp.int32),
        d_raw: wp.array(dtype=wp.int32), d_usermargin: wp.array(dtype=scalar.wp_real),
        d_counts: wp.array(dtype=wp.int32), d_active: wp.array(dtype=wp.int32),
        d_origin: wp.array(dtype=wp.int32),
        seg_count: wp.array(dtype=wp.int32),
        seg_pair_start: wp.array(dtype=wp.int32), seg_pair_end: wp.array(dtype=wp.int32),
        seg_geom_stage: wp.array(dtype=wp.int32),
        trailing_start: wp.array(dtype=wp.int32), trailing_end: wp.array(dtype=wp.int32),
        pair_geom1: wp.array(dtype=wp.int32), pair_geom2: wp.array(dtype=wp.int32),
        req_slots: int, d_slots: int, npair_static: int, stage_slots: int):
    """Build the ordered dispatch request list: for each body-pair segment
    first its predefined-pair calls (ipair, g2=-1, user=0), then its geom
    requests from C04 (g1, g2, user=0); then the trailing pair flush.
    Order matches engine_collision_driver.c:439/463/482."""
    w = wp.tid()
    d_active[w] = active[w]
    d_counts[w] = 0
    if active[w] == 0:
        return
    rc = req_count[w]
    if rc < 0 or rc > req_slots:
        d_counts[w] = -1
        return
    segs = seg_count[w]
    req = wp.int32(0)
    overflow = wp.int32(0)
    for s in range(segs):
        base = w * stage_slots + s
        ps = seg_pair_start[base]
        pe = seg_pair_end[base]
        for p in range(ps, pe):
            if req >= d_slots:
                overflow = 1
                break
            d_raw[(w * d_slots + req) * 3] = p
            d_raw[(w * d_slots + req) * 3 + 1] = -1
            d_raw[(w * d_slots + req) * 3 + 2] = 0
            d_usermargin[w * d_slots + req] = scalar.wp_real(0.0)
            d_origin[(w * d_slots + req) * 2] = -req - 1
            d_origin[(w * d_slots + req) * 2 + 1] = -1
            req += 1
        if overflow != 0:
            break
        gs = seg_geom_stage[base]
        if gs >= 0:
            for r in range(rc):
                if out_req[(w * req_slots + r) * 4] == gs:
                    if req >= d_slots:
                        overflow = 1
                        break
                    g1 = out_req[(w * req_slots + r) * 4 + 1]
                    g2 = out_req[(w * req_slots + r) * 4 + 2]
                    d_raw[(w * d_slots + req) * 3] = g1
                    d_raw[(w * d_slots + req) * 3 + 1] = g2
                    d_raw[(w * d_slots + req) * 3 + 2] = 0
                    d_usermargin[w * d_slots + req] = scalar.wp_real(0.0)
                    d_origin[(w * d_slots + req) * 2] = gs
                    d_origin[(w * d_slots + req) * 2 + 1] = r
                    req += 1
            if overflow != 0:
                break
    if overflow == 0:
        ts = trailing_start[w]
        te = trailing_end[w]
        for p in range(ts, te):
            if req >= d_slots:
                overflow = 1
                break
            d_raw[(w * d_slots + req) * 3] = p
            d_raw[(w * d_slots + req) * 3 + 1] = -1
            d_raw[(w * d_slots + req) * 3 + 2] = 0
            d_usermargin[w * d_slots + req] = scalar.wp_real(0.0)
            d_origin[(w * d_slots + req) * 2] = -req - 1
            d_origin[(w * d_slots + req) * 2 + 1] = -1
            req += 1
    if overflow != 0:
        d_counts[w] = -2  # request capacity exceeded: explicit error
        return
    d_counts[w] = req


@wp.kernel(enable_backward=False)
def _dispatch_to_calls_kernel(
        out_i: wp.array(dtype=wp.int32), out_f: wp.array(dtype=scalar.wp_real),
        call_index: wp.array(dtype=wp.int32), call_count: wp.array(dtype=wp.int32),
        active: wp.array(dtype=wp.int32), d_status: wp.array(dtype=wp.int32),
        d_origin: wp.array(dtype=wp.int32),
        stage_route: wp.array(dtype=wp.int32), stage_slots: int,
        c_g1: wp.array(dtype=wp.int32), c_g2: wp.array(dtype=wp.int32),
        c_t1: wp.array(dtype=wp.int32), c_t2: wp.array(dtype=wp.int32),
        c_margin: wp.array(dtype=scalar.wp_real),
        c_ipair: wp.array(dtype=wp.int32), c_user: wp.array(dtype=wp.int32),
        c_segment: wp.array(dtype=wp.int32), c_sort: wp.array(dtype=wp.int32),
        c_route_slot: wp.array(dtype=wp.int32),
        c_count: wp.array(dtype=wp.int32), pairs: int, d_slots: int):
    """C05 call list -> detector/finalize call records; segment = origin stage."""
    w = wp.tid()
    c_count[w] = 0
    if active[w] == 0:
        return
    if d_status[w] != 0:
        c_count[w] = -1
        return
    nc = call_count[w]
    if nc < 0 or nc > pairs:
        c_count[w] = -1
        return
    c_count[w] = nc
    for j in range(nc):
        q = call_index[w * pairs + j]
        b = (w * d_slots + q) * 12
        c_g1[w * pairs + j] = out_i[b + 2]
        c_g2[w * pairs + j] = out_i[b + 3]
        c_t1[w * pairs + j] = out_i[b + 4]
        c_t2[w * pairs + j] = out_i[b + 5]
        c_margin[w * pairs + j] = out_f[(w * d_slots + q) * 2]
        c_ipair[w * pairs + j] = out_i[b + 7]
        c_user[w * pairs + j] = 0
        c_route_slot[w * pairs + j] = q
        seg = d_origin[(w * d_slots + q) * 2]
        c_segment[w * pairs + j] = seg
        sortv = wp.int32(0)
        if seg >= 0 and seg < stage_slots:
            sortv = stage_route[w * stage_slots + seg]
            if sortv != 1:
                sortv = 0
        c_sort[w * pairs + j] = sortv


@wp.kernel(enable_backward=False)
def _calls_to_finalize_kernel(
        c_g1: wp.array(dtype=wp.int32), c_g2: wp.array(dtype=wp.int32),
        c_t1: wp.array(dtype=wp.int32), c_t2: wp.array(dtype=wp.int32),
        c_ipair: wp.array(dtype=wp.int32), c_user: wp.array(dtype=wp.int32),
        c_segment: wp.array(dtype=wp.int32), c_sort: wp.array(dtype=wp.int32),
        c_count: wp.array(dtype=wp.int32), c_margin: wp.array(dtype=scalar.wp_real),
        det_raw: wp.array(dtype=scalar.wp_real), det_raw_count: wp.array(dtype=wp.int32),
        det_status: wp.array(dtype=wp.int32),
        f_raw: wp.array(dtype=scalar.wp_real), f_raw_count: wp.array(dtype=wp.int32),
        f_call_count: wp.array(dtype=wp.int32), f_active: wp.array(dtype=wp.int32),
        f_types: wp.array(dtype=wp.int32), f_segment: wp.array(dtype=wp.int32),
        f_sort: wp.array(dtype=wp.int32),
        p_route: wp.array(dtype=wp.int32), p_usermargin: wp.array(dtype=scalar.wp_real),
        p_counts: wp.array(dtype=wp.int32),
        f_status: wp.array(dtype=wp.int32),
        active: wp.array(dtype=wp.int32),
        nworld: int, pairs: int, max_raw: int, raw_f: int):
    """Detect outputs -> C06 device buffers (route replication included)."""
    w = wp.tid()
    f_active[w] = active[w]
    f_call_count[w] = 0
    f_status[w] = 0
    # params.counts is one scalar per world: every raw slot is a params row
    p_counts[w] = 0
    if active[w] == 0:
        return
    nc = c_count[w]
    if nc < 0:
        f_status[w] = 1  # upstream dispatch error: no finalize commit
        return
    for j in range(nc):
        if det_status[w * pairs + j] != 0:
            f_active[w] = 0
            f_status[w] = 1
            return
    p_counts[w] = pairs * max_raw
    for j in range(pairs):
        slot = w * pairs + j
        for r in range(max_raw):
            src = (slot * max_raw + r) * raw_f
            dst = (slot * max_raw + r) * raw_f
            for k in range(raw_f):
                f_raw[dst + k] = det_raw[src + k]
            # route replication: (g1, g2, ipair, user) per raw row
            p_route[(slot * max_raw + r) * 4] = c_g1[slot]
            p_route[(slot * max_raw + r) * 4 + 1] = c_g2[slot]
            p_route[(slot * max_raw + r) * 4 + 2] = c_ipair[slot]
            p_route[(slot * max_raw + r) * 4 + 3] = c_user[slot]
            p_usermargin[slot * max_raw + r] = c_margin[slot]
        f_raw_count[slot] = det_raw_count[slot]
        f_types[slot * 2] = c_t1[slot]
        f_types[slot * 2 + 1] = c_t2[slot]
        f_segment[slot] = c_segment[slot]
        f_sort[slot] = c_sort[slot]
    f_call_count[w] = nc


@wp.kernel(enable_backward=False)
def _chain_error_kernel(
        bounds_status: wp.array(dtype=wp.int32),
        bp_status: wp.array(dtype=wp.int32),
        sb_status: wp.array(dtype=wp.int32),
        tree_status: wp.array(dtype=wp.int32),
        dispatch_status: wp.array(dtype=wp.int32),
        detect_fe_code: wp.array(dtype=wp.int32),
        detect_fe_item: wp.array(dtype=wp.int32),
        fin_status: wp.array(dtype=wp.int32), fin_ncon: wp.array(dtype=wp.int32),
        fin_warnings: wp.array(dtype=wp.int32), fin_capacity: int,
        active: wp.array(dtype=wp.int32),
        fe_stage: wp.array(dtype=wp.int32), fe_code: wp.array(dtype=wp.int32),
        fe_item: wp.array(dtype=wp.int32)):
    """Chain-level sticky first_error and finalize commit guard.

    Runs after every stage including finalize. Any upstream nonzero status
    becomes the world's first_error (original first-check order: bounds ->
    broadphase -> stagebuild -> tree -> dispatch -> detect -> finalize) and
    blocks the finalize commit: fin_status is overwritten with the chain
    error code and fin_ncon with 0, so a caller that only reads the finalize
    record cannot mistake an errored world for a clean zero-contact success
    (R01-20/R01-36). R02-1: finalize warnings (contact_capacity overflow)
    is also an explicit first_error.
    """
    w = wp.tid()
    if fe_code[w] != 0:
        fin_status[w] = fe_code[w]
        fin_ncon[w] = 0
        return
    if active[w] == 0:
        return
    if active[w] != 1:
        fe_stage[w] = 1
        fe_code[w] = 1  # ERR_BAD_ACTIVE
        fe_item[w] = w
        fin_status[w] = 1
        return
    if bounds_status[w] != 0:
        fe_stage[w] = 1
        fe_code[w] = 9
        fe_item[w] = 0
        fin_status[w] = 9
        fin_ncon[w] = 0
        return
    if bp_status[w] != 0:
        fe_stage[w] = 2
        fe_code[w] = 3 if bp_status[w] == 1 else 9
        fe_item[w] = 0
        fin_status[w] = fe_code[w]
        fin_ncon[w] = 0
        return
    if sb_status[w] != 0:
        fe_stage[w] = 3
        fe_code[w] = 3 if sb_status[w] == 3 else 9
        fe_item[w] = 0
        fin_status[w] = fe_code[w]
        fin_ncon[w] = 0
        return
    if tree_status[w] != 0:
        fe_stage[w] = 4
        fe_code[w] = 3 if (tree_status[w] == 5 or tree_status[w] == 6 or tree_status[w] == 9) else 9
        fe_item[w] = 0
        fin_status[w] = fe_code[w]
        fin_ncon[w] = 0
        return
    if dispatch_status[w] != 0:
        fe_stage[w] = 5
        fe_code[w] = 3 if dispatch_status[w] == 4 else 9
        fe_item[w] = 0
        fin_status[w] = fe_code[w]
        fin_ncon[w] = 0
        return
    if detect_fe_code[w] != 0:
        fe_stage[w] = 6
        fe_code[w] = detect_fe_code[w]
        fe_item[w] = detect_fe_item[w]
        fin_status[w] = detect_fe_code[w]
        fin_ncon[w] = 0
        return
    if fin_status[w] != 0:
        fe_stage[w] = 7
        fe_code[w] = 8
        fe_item[w] = 0
        fin_ncon[w] = 0
        return
    # R02-1/R03-1: finalize contact_capacity overflow (warnings != 0) is an
    # explicit capacity error. ncon == capacity is a legal exactly-full state
    # (every contact written, none dropped) — only warnings signals overflow.
    # A redundant ncon > capacity check is kept as a safety net (R03-1: >= is wrong).
    if fin_warnings[w] != 0 or fin_ncon[w] > fin_capacity:
        fe_stage[w] = 7
        fe_code[w] = 3  # ERR_CAPACITY
        fe_item[w] = 0
        fin_status[w] = 3
        fin_ncon[w] = 0
        return


@wp.kernel(enable_backward=False)
def _effective_active_kernel(active: wp.array(dtype=wp.int32),
        fe_code: wp.array(dtype=wp.int32), effective: wp.array(dtype=wp.int32)):
    w = wp.tid()
    effective[w] = active[w]
    if fe_code[w] != 0:
        effective[w] = 0


@wp.kernel(enable_backward=False)
def _reset_error_kernel(reset: wp.array(dtype=wp.int32),
        fe_stage: wp.array(dtype=wp.int32), fe_code: wp.array(dtype=wp.int32),
        fe_item: wp.array(dtype=wp.int32)):
    w = wp.tid()
    if reset[w] == 1:
        fe_stage[w] = 0
        fe_code[w] = 0
        fe_item[w] = -1


@wp.kernel(enable_backward=False)
def _restore_failed_contacts_kernel(active: wp.array(dtype=wp.int32),
        fe_code: wp.array(dtype=wp.int32), before_f: wp.array(dtype=scalar.wp_real),
        before_i: wp.array(dtype=wp.int32), public_f: wp.array(dtype=scalar.wp_real),
        public_i: wp.array(dtype=wp.int32), capacity: int):
    w = wp.tid()
    if active[w] == 0 or fe_code[w] != 0:
        for j in range(capacity):
            for k in range(65):
                ix = (w * capacity + j) * 65 + k
                public_f[ix] = before_f[ix]
            for k in range(5):
                ix = (w * capacity + j) * 5 + k
                public_i[ix] = before_i[ix]


@wp.kernel(enable_backward=False)
def _body_counters_kernel(call_count: wp.array(dtype=wp.int32),
        call_segment: wp.array(dtype=wp.int32), call_ipair: wp.array(dtype=wp.int32),
        raw_count: wp.array(dtype=wp.int32), effective: wp.array(dtype=wp.int32),
        collision_mode: wp.array(dtype=wp.int32), gates: wp.array(dtype=wp.int32),
        broad: wp.array(dtype=wp.int32), narrow: wp.array(dtype=wp.int32),
        pairs: int, npair: int, nbody: int):
    w = wp.tid()
    broad[w] = 0
    narrow[w] = 0
    if effective[w] != 1 or gates[0] != 0 or gates[1] != 0 or gates[2] == 0 or nbody < 2:
        return
    nc = call_count[w]
    if nc < 0 or nc > pairs:
        return
    if collision_mode[0] == 1:
        broad[w] = npair
    for j in range(nc):
        slot = w * pairs + j
        if call_ipair[slot] >= 0:
            if collision_mode[0] == 1:
                broad[w] += 1
                if raw_count[slot] > 0:
                    narrow[w] += 1
        else:
            segment = call_segment[slot]
            first = wp.bool(True)
            for k in range(j):
                kslot = w * pairs + k
                if call_ipair[kslot] < 0 and call_segment[kslot] == segment:
                    first = False
            if first:
                broad[w] += 1
                entered = wp.bool(False)
                for k in range(j, nc):
                    kslot = w * pairs + k
                    if call_ipair[kslot] < 0 and call_segment[kslot] == segment and raw_count[kslot] > 0:
                        entered = True
                if entered:
                    narrow[w] += 1


@wp.kernel(enable_backward=False)
def _stage_error_kernel(status: wp.array(dtype=wp.int32),
        active: wp.array(dtype=wp.int32), stage_id: int, cap_a: int, cap_b: int,
        cap_c: int, fe_stage: wp.array(dtype=wp.int32),
        fe_code: wp.array(dtype=wp.int32), fe_item: wp.array(dtype=wp.int32),
        effective: wp.array(dtype=wp.int32)):
    w = wp.tid()
    if active[w] == 1 and fe_code[w] == 0 and status[w] != 0:
        fe_stage[w] = stage_id
        code = status[w]
        fe_code[w] = 3 if (code == cap_a or code == cap_b or code == cap_c) else 9
        fe_item[w] = 0
    effective[w] = active[w]
    if fe_code[w] != 0:
        effective[w] = 0


@wp.kernel(enable_backward=False)
def _detect_error_kernel(status: wp.array(dtype=wp.int32),
        item: wp.array(dtype=wp.int32), active: wp.array(dtype=wp.int32),
        fe_stage: wp.array(dtype=wp.int32), fe_code: wp.array(dtype=wp.int32),
        fe_item: wp.array(dtype=wp.int32), effective: wp.array(dtype=wp.int32)):
    w = wp.tid()
    if active[w] == 1 and fe_code[w] == 0 and status[w] != 0:
        fe_stage[w] = 6
        fe_code[w] = status[w]
        fe_item[w] = item[w]
    effective[w] = active[w]
    if fe_code[w] != 0:
        effective[w] = 0


@wp.kernel(enable_backward=False)
def _input_error_kernel(active: wp.array(dtype=wp.int32),
        fe_stage: wp.array(dtype=wp.int32), fe_code: wp.array(dtype=wp.int32),
        fe_item: wp.array(dtype=wp.int32)):
    w = wp.tid()
    if fe_code[w] == 0 and active[w] != 0 and active[w] != 1:
        fe_stage[w] = 1
        fe_code[w] = 1
        fe_item[w] = w


@wp.kernel(enable_backward=False)
def _restore_inactive_i32(active: wp.array(dtype=wp.int32),
        before: wp.array(dtype=wp.int32), public: wp.array(dtype=wp.int32),
        stride: int):
    i = wp.tid()
    if active[i // stride] == 0:
        public[i] = before[i]


@wp.kernel(enable_backward=False)
def _restore_inactive_f64(active: wp.array(dtype=wp.int32),
        before: wp.array(dtype=scalar.wp_real), public: wp.array(dtype=scalar.wp_real),
        stride: int):
    i = wp.tid()
    if active[i // stride] == 0:
        public[i] = before[i]


@wp.kernel(enable_backward=False)
def _restore_inactive_f32(active: wp.array(dtype=wp.int32),
        before: wp.array(dtype=wp.float32), public: wp.array(dtype=wp.float32),
        stride: int):
    i = wp.tid()
    if active[i // stride] == 0:
        public[i] = before[i]


class CollisionChainWorkspace:
    """Complete ordered collision chain over one fixed model, N worlds.

    prepare() admits the static model; upload() writes dynamic poses outside
    the production interval; launch() runs the whole device chain with no
    host count round-trips; download() is diagnostic only.
    """

    def __init__(self, static: dict, nworld: int, *, device='cuda:0',
                 slots: int | None = None, call_capacity: int | None = None,
                 pair_capacity: int | None = None, stage_slots: int | None = None,
                 req_slots: int | None = None, contact_capacity: int | None = None,
                 max_device_bytes: int | None = None,
                 borrow_detector_raw: bool = False, detector_backend: str = 'legacy'):
        from libero.gpu.reference.collision.collision_dispatch.kernel import CollisionDispatchWorkspace
        from libero.gpu.reference.collision.collision_tree.kernel import CollisionTreeWorkspace
        from libero.gpu.reference.collision.contact_finalize.kernel import ContactFinalizeWorkspace

        if type(nworld) is not int or nworld < 1:
            raise ValueError('nworld must be a positive integer')
        if type(borrow_detector_raw) is not bool:
            raise ValueError("borrow_detector_raw must be bool")
        if detector_backend not in ('legacy', 'handwritten_only'):
            raise ValueError('unknown detector_backend ABI')
        self._detector_backend = detector_backend
        self.borrow_detector_raw = borrow_detector_raw
        self.nworld = nworld
        self.device = device
        self._device_input_binding = None
        self._launch_started = False
        self._contact_field_views = None
        self.nbody = len(static['body_geomnum'])
        self.ngeom = len(static['geom_size'])
        self.npair_static = len(static['pair_geom1'])
        self.nexclude = len(static['exclude_signature'])

        def i32(name, shape=None):
            a = np.asarray(static[name])
            if a.dtype != np.int32:
                raise ValueError(f'{name} must be int32')
            if shape is not None and a.shape != shape:
                raise ValueError(f'{name} shape {a.shape} != {shape}')
            return np.ascontiguousarray(a)

        def f64(name, shape=None):
            a = np.asarray(static[name])
            if a.dtype != np.float64:
                raise ValueError(f'{name} must be float64')
            if shape is not None and a.shape != shape:
                raise ValueError(f'{name} shape')
            return np.ascontiguousarray(a)

        # ---- capacities (host preparation decisions; runtime overflow is a
        # device error, never a silent truncation)
        def capacity(value, default, name):
            if value is None:
                value = default
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 1 <= int(value) < 2**31:
                raise ValueError('positive int32 capacity required: ' + name)
            return int(value)
        self.slots = capacity(slots, self.ngeom * 4 + 16, 'slots')
        self.call_capacity = capacity(call_capacity, self.slots, 'call_capacity')
        self.pair_capacity = capacity(pair_capacity, (self.nbody * (self.nbody - 1)) // 2 + 8, 'pair_capacity')
        self.stage_slots = capacity(stage_slots, self.pair_capacity, 'stage_slots')
        self.req_slots = capacity(req_slots, self.slots, 'req_slots')
        self.contact_capacity = capacity(contact_capacity, 512, 'contact_capacity')
        if self.call_capacity > self.slots:
            raise ValueError('call_capacity <= slots required')
        # All kernel offsets are int32. Reject overflow and a conservative memory
        # upper bound before allocation, including complete preservation shadows.
        lengths = (nworld * self.call_capacity * 50 * 26,
                   nworld * self.slots * 12, nworld * self.contact_capacity * 65,
                   nworld * self.req_slots * 4, nworld * self.stage_slots * 6)
        if max(lengths) >= 2**31:
            raise ValueError('int32 device offset overflow')
        self.preflight_device_bytes = nworld * (
            self.call_capacity * 50 * 2048 + self.slots * 512 +
            self.contact_capacity * 4096 + self.req_slots * 2048 +
            self.stage_slots * 512 + self.nbody * 512 + self.ngeom * 1024)
        if max_device_bytes is None:
            from libero.gpu.host.runtime.core.memory_budget import resolve_memory_budget
            max_device_bytes = resolve_memory_budget(device=device)['collision_budget_bytes']
        if type(max_device_bytes) is not int or max_device_bytes <= 0 or self.preflight_device_bytes > max_device_bytes:
            raise ValueError('private device byte budget exceeded before allocation')

        # ---- device statics
        self.body_geomadr = wp.array(i32('body_geomadr', (self.nbody,)), dtype=wp.int32, device=device)
        self.body_geomnum = wp.array(i32('body_geomnum', (self.nbody,)), dtype=wp.int32, device=device)
        self.body_weldid = wp.array(i32('body_weldid', (self.nbody,)), dtype=wp.int32, device=device)
        self.body_parentid = wp.array(i32('body_parentid', (self.nbody,)), dtype=wp.int32, device=device)
        self.geom_bodyid = wp.array(i32('geom_bodyid', (self.ngeom,)), dtype=wp.int32, device=device)
        self.geom_type = wp.array(i32('geom_type', (self.ngeom,)), dtype=wp.int32, device=device)
        self.geom_contype = wp.array(i32('geom_contype', (self.ngeom,)), dtype=wp.int32, device=device)
        self.geom_conaffinity = wp.array(i32('geom_conaffinity', (self.ngeom,)), dtype=wp.int32, device=device)
        self.geom_rbound = wp.array(f64('geom_rbound', (self.ngeom,)), dtype=scalar.wp_real, device=device)
        self.geom_margin = wp.array(f64('geom_margin', (self.ngeom,)), dtype=scalar.wp_real, device=device)
        self.filterparent_disabled = wp.array(
            np.array([int(static.get('filterparent_disabled', 0))], np.int32),
            dtype=wp.int32, device=device)
        self.midphase_disabled = wp.array(
            np.array([int(static.get('midphase_disabled', 0))], np.int32),
            dtype=wp.int32, device=device)
        can = np.zeros(self.nbody, np.int32)
        plane = np.zeros(self.nbody, np.int32)
        gtype = np.asarray(static['geom_type'])
        gcont = np.asarray(static['geom_contype'])
        gaff = np.asarray(static['geom_conaffinity'])
        for b in range(self.nbody):
            start, count = int(static['body_geomadr'][b]), int(static['body_geomnum'][b])
            if count:
                sl = slice(start, start + count)
                can[b] = int(np.any(gcont[sl] != 0) or np.any(gaff[sl] != 0))
                plane[b] = int(np.any((gtype[sl] == 0) | (gtype[sl] == 1)))
        self.can = wp.array(can, dtype=wp.int32, device=device)
        self.plane = wp.array(plane, dtype=wp.int32, device=device)
        self.pair_signature = wp.array(
            np.ascontiguousarray(static.get('pair_signature', np.zeros(0, np.int32)), dtype=np.int32).ravel()
            if self.npair_static else np.zeros(1, np.int32),
            dtype=wp.int32, device=device)
        self.exclude_signature = wp.array(
            np.ascontiguousarray(static['exclude_signature'], dtype=np.int32).ravel()
            if self.nexclude else np.zeros(1, np.int32),
            dtype=wp.int32, device=device)

        # ---- chain buffers
        N = nworld
        self.xpos = wp.zeros(N * self.ngeom * 3, dtype=scalar.wp_real, device=device)
        self.xmat = wp.zeros(N * self.ngeom * 9, dtype=scalar.wp_real, device=device)
        self.active = wp.zeros(N, dtype=wp.int32, device=device)
        self.aabb = wp.zeros(N * self.nbody * 6, dtype=scalar.wp_real, device=device)
        self.frame = wp.zeros(N * 9, dtype=scalar.wp_real, device=device)
        self.bounds_entered = wp.zeros(N, dtype=wp.int32, device=device)
        self.bounds_count = wp.zeros(N, dtype=wp.int32, device=device)
        self.bounds_status = wp.zeros(N, dtype=wp.int32, device=device)
        self.bounds_scratch = wp.zeros(N * 34, dtype=scalar.wp_real, device=device)
        self.out_pair = wp.zeros(N * self.pair_capacity, dtype=wp.int32, device=device)
        self.out_npair = wp.zeros(N, dtype=wp.int32, device=device)
        self.bp_status = wp.zeros(N, dtype=wp.int32, device=device)
        self.sortbuf = wp.zeros(N * 2 * self.nbody, dtype=wp.float32, device=device)
        self.sorttag = wp.zeros(N * 2 * self.nbody, dtype=wp.int32, device=device)
        self.activebuf = wp.zeros(N * self.nbody, dtype=wp.int32, device=device)
        self.stage = wp.zeros(N * self.stage_slots * 6, dtype=wp.int32, device=device)
        self.stage_count = wp.zeros(N, dtype=wp.int32, device=device)
        self.stage_merged = wp.zeros(N * self.stage_slots, dtype=wp.int32, device=device)
        self.stage_route = wp.zeros(N * self.stage_slots, dtype=wp.int32, device=device)
        self.seg_count = wp.zeros(N, dtype=wp.int32, device=device)
        self.seg_pair_start = wp.zeros(N * self.stage_slots, dtype=wp.int32, device=device)
        self.seg_pair_end = wp.zeros(N * self.stage_slots, dtype=wp.int32, device=device)
        self.seg_geom_stage = wp.zeros(N * self.stage_slots, dtype=wp.int32, device=device)
        self.trailing_start = wp.zeros(N, dtype=wp.int32, device=device)
        self.trailing_end = wp.zeros(N, dtype=wp.int32, device=device)
        self.sb_pairadr = wp.zeros(N, dtype=wp.int32, device=device)
        self.bodypair_broad = wp.zeros(N, dtype=wp.int32, device=device)
        self.bodypair_narrow = wp.zeros(N, dtype=wp.int32, device=device)
        self.sb_status = wp.zeros(N, dtype=wp.int32, device=device)
        self.sb_first = wp.zeros(N, dtype=wp.int32, device=device)
        # driver gates: [contact_disabled, constraint_disabled, nconmax] and
        # collision mode (0=ALL, 1=PAIR, 2=DYNAMIC)
        self.gates = wp.array(
            np.array([int(static.get('contact_disabled', 0)),
                      int(static.get('constraint_disabled', 0)),
                      int(static.get('nconmax', 5000))], np.int32),
            dtype=wp.int32, device=device)
        self.collision_mode = wp.array(
            np.array([int(static.get('collision_mode', 0))], np.int32),
            dtype=wp.int32, device=device)

        # ---- C04 tree/flat invocation (accepted Warp module)
        tree_static = {
            'body_bvhadr': i32('body_bvhadr', (self.nbody,)),
            'body_bvhnum': i32('body_bvhnum', (self.nbody,)),
            'body_geomadr': i32('body_geomadr', (self.nbody,)),
            'body_geomnum': i32('body_geomnum', (self.nbody,)),
            'bvh_child': i32('bvh_child'),
            'bvh_geomid': i32('bvh_geomid'),
            'bvh_aabb': f64('bvh_aabb'),
            'geom_aabb': f64('geom_aabb'),
            'geom_rbound': f64('geom_rbound', (self.ngeom,)),
            'geom_type': i32('geom_type', (self.ngeom,)),
            'pair_geom1': i32('pair_geom1') if self.npair_static else np.zeros(1, np.int32),
            'pair_geom2': i32('pair_geom2') if self.npair_static else np.zeros(1, np.int32),
        }
        self.tree = CollisionTreeWorkspace(
            tree_static, N, self.stage_slots, self.req_slots,
            self.req_slots * 4 + 16, self.req_slots * 2 + 16,
            self.req_slots + 16, device=device)

        # ---- C05 dispatch (accepted Warp module)
        disp_static = {
            'geom_type': i32('geom_type', (self.ngeom,)),
            'geom_contype': i32('geom_contype', (self.ngeom,)),
            'geom_conaffinity': i32('geom_conaffinity', (self.ngeom,)),
            'geom_priority': i32('geom_priority', (self.ngeom,)),
            'geom_condim': i32('geom_condim', (self.ngeom,)),
            'geom_margin': f64('geom_margin', (self.ngeom,)),
            'geom_gap': f64('geom_gap', (self.ngeom,)),
            'geom_rbound': f64('geom_rbound', (self.ngeom,)),
            'pair_geom1': i32('pair_geom1') if self.npair_static else np.zeros(1, np.int32),
            'pair_geom2': i32('pair_geom2') if self.npair_static else np.zeros(1, np.int32),
            'pair_dim': i32('pair_dim') if self.npair_static else np.zeros(1, np.int32),
            'pair_margin': f64('pair_margin') if self.npair_static else np.zeros(1, np.float64),
            'pair_gap': f64('pair_gap') if self.npair_static else np.zeros(1, np.float64),
            'override': i32('override', (1,)),
            'o_margin': f64('o_margin', (1,)),
            'dispatch_present': i32('dispatch_present', (8, 8)),
        }
        self.dispatch = CollisionDispatchWorkspace(
            N, self.slots, self.call_capacity, disp_static, device=device)

        # ---- detect
        from .detectors import DetectorWorkspace
        det_static = {
            'mpr_iterations': int(static['mpr_iterations']),
            'mpr_tolerance': float(static['mpr_tolerance']),
            'multiccd': int(static['multiccd']),
            'geom_size': f64('geom_size', (self.ngeom, 3)),
            'geom_dataid': i32('geom_dataid', (self.ngeom,)),
            'geom_rbound': f64('geom_rbound', (self.ngeom,)),
            'mesh_vert': np.ascontiguousarray(static.get('mesh_vert', np.zeros(0, np.float32)), dtype=np.float32).ravel(),
            'mesh_vertadr': i32('mesh_vertadr') if len(static.get('mesh_vertadr', [])) else np.zeros(0, np.int32),
            'mesh_vertnum': i32('mesh_vertnum') if len(static.get('mesh_vertnum', [])) else np.zeros(0, np.int32),
            'mesh_graphadr': i32('mesh_graphadr') if len(static.get('mesh_graphadr', [])) else np.zeros(0, np.int32),
            'mesh_graph': np.ascontiguousarray(static.get('mesh_graph', np.zeros(0, np.int32)), dtype=np.int32).ravel(),
        }
        self.detect = DetectorWorkspace(N, self.call_capacity, det_static, device=device,
            prepare_baseline=(detector_backend == 'legacy'))
        # All consumers use the upstream prepared GPU frames, with retained owners.
        self.detect.xpos, self.detect.xmat = self.xpos, self.xmat
        self.tree.geom_pos, self.tree.geom_mat = self.xpos, self.xmat
        self.dispatch.xpos, self.dispatch.xmat = self.xpos, self.xmat
        # per-call records bridging dispatch -> detect -> finalize
        self.dispatch_origin = wp.zeros(N * self.slots * 2, dtype=wp.int32, device=device)
        self.call_ipair = wp.zeros(N * self.call_capacity, dtype=wp.int32, device=device)
        self.call_user = wp.zeros(N * self.call_capacity, dtype=wp.int32, device=device)
        self.call_segment = wp.zeros(N * self.call_capacity, dtype=wp.int32, device=device)
        self.call_sort = wp.zeros(N * self.call_capacity, dtype=wp.int32, device=device)
        self.call_route_slot = wp.zeros(N * self.call_capacity, dtype=wp.int32, device=device)

        # ---- C06 finalize (accepted Warp module)
        fin_static = {
            'geom_priority': i32('geom_priority', (self.ngeom,)),
            'geom_condim': i32('geom_condim', (self.ngeom,)),
            'geom_margin': f64('geom_margin', (self.ngeom,)),
            'geom_gap': f64('geom_gap', (self.ngeom,)),
            'geom_friction': f64('geom_friction', (self.ngeom, 3)),
            'geom_solmix': f64('geom_solmix', (self.ngeom,)),
            'geom_solref': f64('geom_solref', (self.ngeom, 2)),
            'geom_solimp': f64('geom_solimp', (self.ngeom, 5)),
            'pair_geom1': i32('pair_geom1') if self.npair_static else np.zeros(1, np.int32),
            'pair_geom2': i32('pair_geom2') if self.npair_static else np.zeros(1, np.int32),
            'pair_margin': f64('pair_margin') if self.npair_static else np.zeros(1, np.float64),
            'pair_gap': f64('pair_gap') if self.npair_static else np.zeros(1, np.float64),
            'pair_dim': i32('pair_dim') if self.npair_static else np.zeros(1, np.int32),
            'pair_friction': f64('pair_friction') if self.npair_static else np.zeros((1, 5), np.float64),
            'pair_solref': f64('pair_solref') if self.npair_static else np.zeros((1, 2), np.float64),
            'pair_solreffriction': f64('pair_solreffriction') if self.npair_static else np.zeros((1, 2), np.float64),
            'pair_solimp': f64('pair_solimp') if self.npair_static else np.zeros((1, 5), np.float64),
            'o_margin': f64('o_margin', (1,)),
            'o_solref': f64('o_solref', (2,)),
            'o_solimp': f64('o_solimp', (5,)),
            'override': i32('override', (1,)),
        }
        self.finalize = ContactFinalizeWorkspace(
            N, self.call_capacity, self.contact_capacity, fin_static,
            i32('geom_type', (self.ngeom,)), device=device)

        # Optional native-source-only storage: the bridge copies identical
        # offsets and dedup only reads this input. Public detector raw and all
        # original output shadows remain separate and fully allocated. The
        # standalone finalizer constructor/default is unchanged.
        if self.borrow_detector_raw:
            self.finalize.raw = self.detect.raw
            self._borrowed_raw_owner = self.detect

        # ---- chain first_error
        self.fe_stage = wp.zeros(N, dtype=wp.int32, device=device)
        self.fe_code = wp.zeros(N, dtype=wp.int32, device=device)
        self.fe_item = wp.full(N, -1, dtype=wp.int32, device=device)
        self.effective_active = wp.zeros(N, dtype=wp.int32, device=device)
        self.reset_mask = wp.zeros(N, dtype=wp.int32, device=device)
        self.before_contact_f = wp.zeros(N * self.contact_capacity * 65, dtype=scalar.wp_real, device=device)
        self.before_contact_i = wp.zeros(N * self.contact_capacity * 5, dtype=wp.int32, device=device)
        self.outputs = {}
        chain_names = (
            'aabb', 'frame', 'bounds_entered', 'bounds_count', 'bounds_status',
            'bounds_scratch', 'out_pair', 'out_npair', 'bp_status', 'sortbuf',
            'sorttag', 'activebuf', 'stage', 'stage_count', 'stage_merged',
            'stage_route', 'seg_count', 'seg_pair_start', 'seg_pair_end',
            'seg_geom_stage', 'trailing_start', 'trailing_end', 'sb_pairadr',
            'bodypair_broad', 'bodypair_narrow', 'sb_status', 'sb_first',
            'dispatch_origin', 'call_ipair', 'call_user', 'call_segment', 'call_sort', 'call_route_slot')
        for name in chain_names:
            self.outputs['chain_' + name] = getattr(self, name)
        module_names = {
            'tree': ('out_req', 'out_visit', 'out_stack', 'out_leaf', 'out_active',
                     'req_count', 'visit_count', 'stack_count', 'leaf_count',
                     'status', 'first_stage'),
            'dispatch': ('out_i', 'out_f', 'call_index', 'call_count', 'mid_count',
                         'status', 'counts', 'raw', 'usermargin'),
            'detect': ('raw', 'raw_count', 'det_status', 'native_raw', 'native_count',
                       'native_status', 'native_diag', 'native_frame0',
                       'native_framewritten', 'call_g1', 'call_g2', 'call_type1',
                       'call_type2', 'call_margin', 'call_count'),
            'finalize': ('contact_f', 'contact_i', 'ncon', 'attempts', 'appended',
                         'warnings', 'status', 'compact_count', 'keep_index',
                         'dedup_status'),
        }
        for module, names in module_names.items():
            obj = getattr(self, module)
            for name in names:
                self.outputs[module + '_' + name] = getattr(obj, name)
        self.outputs['finalize_compact_raw'] = self.finalize.params.raw
        self.outputs['params_f'] = self.finalize.params.out_f
        self.outputs['params_i'] = self.finalize.params.out_i
        self.before_outputs = {}
        for name, arr in self.outputs.items():
            arr.zero_()
            if arr.size % N:
                raise ValueError('output world stride')
            if name == 'finalize_contact_f':
                shadow = self.before_contact_f
            elif name == 'finalize_contact_i':
                shadow = self.before_contact_i
            else:
                shadow = wp.empty_like(arr)
            self.before_outputs[name] = shadow

    def bind_device_inputs(self, owner, paths, *, stream):
        """Prepare five retained source edges without host pose/count copies.

        paths maps xpos/xmat/xipos/ximat/active to actual attributes of owner.
        The active source must be the caller's device producer/epoch gate.
        Binding is immutable and must precede any warm-up or Graph capture.
        """
        from .shared import DeviceInputBinding, DeviceBindingError
        if self._launch_started or self._device_input_binding is not None:
            raise DeviceBindingError('bind once before first collision launch')
        binding = DeviceInputBinding(self, owner, paths, stream)
        self._device_input_binding = binding
        return binding.manifest(self, allow_unsealed=(self._detector_backend == 'handwritten_only'))

    def seal_handwritten_backend(self):
        from .shared import DeviceBindingError
        if self._detector_backend != 'handwritten_only' or self._device_input_binding is None:
            raise DeviceBindingError('handwritten_only retained binding required')
        return self._device_input_binding.seal_backend(self)

    def validate_device_bindings(self, stream=None):
        """Validate actual owners/arguments before eager or caller Graph replay."""
        binding = self._device_input_binding
        if binding is not None:
            binding.validate(self, binding.stream if stream is None else stream)
        if self._contact_field_views is not None:
            self._contact_field_views.validate()
        return True

    def contact_field_views(self):
        """Expose authentic strided fields; no field is presented as contiguous."""
        from .shared import ContactFieldViews
        if self._contact_field_views is None:
            self._contact_field_views = ContactFieldViews(self)
        self._contact_field_views.validate()
        return self._contact_field_views

    def resident_metadata(self):
        """Inspect reachable actual allocation roots; no counts/poses download."""
        from .shared import resident_metadata
        self.validate_device_bindings()
        return resident_metadata(self)

    def capture_graph(self):
        """Capture this bound C chain after explicit caller cold preparation."""
        from .shared import DeviceBindingError
        binding = self._device_input_binding
        if binding is None:
            raise DeviceBindingError('retained external inputs required for this Graph API')
        self.validate_device_bindings(binding.stream)
        with wp.ScopedCapture(stream=binding.stream) as capture:
            self.launch(stream=binding.stream)
        graph = capture.graph
        binding.graphs.append(graph)
        return graph

    def launch_graph(self, graph):
        """Replay a retained C Graph after validating its frozen source graph."""
        from .shared import DeviceBindingError
        binding = self._device_input_binding
        if binding is None or not any(graph is g for g in binding.graphs):
            raise DeviceBindingError('Graph was not captured by this bound workspace')
        self.validate_device_bindings(binding.stream)
        wp.capture_launch(graph, stream=binding.stream)

    def upload(self, xpos, xmat, active, *, xipos=None, ximat=None,
               contact_f_prefill=None, contact_i_prefill=None,
               preserve_output=False):
        """Upload dynamic inputs outside the production interval.

        xipos/ximat are the body inertial poses used by the C04 tree OBB
        tests (R01-23: real inertial frames, not zeros). preserve_output=False
        installs the contact dirty prefill; preserve_output=True leaves the
        current contact_f/contact_i device contents for a second upload cycle.
        """
        if self.borrow_detector_raw:
            raise ValueError("borrowed raw mode requires source device binding, not host upload")
        if self._device_input_binding is not None:
            from .shared import DeviceBindingError
            raise DeviceBindingError('shared GPU inputs are written by their source producer')
        specs = ((xpos, (self.nworld, self.ngeom, 3), np.float64, self.xpos),
                 (xmat, (self.nworld, self.ngeom, 9), np.float64, self.xmat),
                 (active, (self.nworld,), np.int32, self.active))
        for value, shape, dtype, dst in specs:
            a = np.asarray(value)
            if a.shape != shape or a.dtype != dtype:
                raise ValueError(f'chain input {a.shape}/{a.dtype} != {shape}/{dtype}')
            if dtype == np.float64 and (not np.all(np.isfinite(a)) or np.any(np.abs(a) > 1e100)):
                raise ValueError('unsupported_numeric_range')
            if dtype == np.int32 and not np.all((a == 0) | (a == 1)):
                raise ValueError('active must be 0/1')
            dst.assign(np.ascontiguousarray(a).ravel())
        self.detect.upload(xpos, xmat, active)
        # C04 poses: body inertial frames for OBB tests (R01-23)
        n, g, b = self.nworld, self.ngeom, self.nbody
        if xipos is None or ximat is None:
            raise ValueError('actual body inertial xipos/ximat are required')
        for value, shape in ((xipos,(n,b,3)),(ximat,(n,b,9))):
            a=np.asarray(value)
            if a.dtype!=np.dtype('float64') or a.shape!=shape or not np.isfinite(a).all() or np.any(np.abs(a)>1e100):
                raise ValueError('body inertial input dtype/shape/numeric range')
        self.tree.upload(
            np.zeros((n, self.stage_slots, 6), np.int32),
            np.zeros(n, np.int32), active,
            xipos, ximat, xpos, xmat)
        # C06 prefill
        n, p, c = self.nworld, self.call_capacity, self.contact_capacity
        if contact_f_prefill is None:
            contact_f_prefill = np.zeros((n, c, 65), np.float64)
        if contact_i_prefill is None:
            contact_i_prefill = np.zeros((n, c, 5), np.int32)
        self.finalize.upload(
            np.zeros((n, p, 50, 10), np.float64), np.zeros((n, p), np.int32),
            np.zeros(n, np.int32), active,
            np.zeros((n, p, 4), np.int32), np.zeros((n, p), np.float64),
            np.zeros((n, p, 2), np.int32), np.zeros((n, p), np.int32),
            np.zeros((n, p), np.int32),
            contact_f_prefill, contact_i_prefill, preserve_output=preserve_output)
        # C05 receives the prepared aliased GPU poses. Its request buffers
        # are device bridge outputs; upload does not overwrite their held bytes.
        self.dispatch.active.assign(np.ascontiguousarray(active).ravel())

    def launch(self, stream=None):
        if self.borrow_detector_raw and (self._borrowed_raw_owner is not self.detect or self.finalize.raw is not self.detect.raw):
            raise ValueError("borrowed detector/finalizer raw identity changed")
        N = self.nworld
        binding = self._device_input_binding
        stream = stream or (binding.stream if binding is not None else wp.get_stream(self.device))
        self.validate_device_bindings(stream)
        self._launch_started = True
        for name, arr in self.outputs.items():
            wp.copy(self.before_outputs[name], arr, stream=stream)
        wp.launch(_input_error_kernel, dim=N, inputs=[
            self.active, self.fe_stage, self.fe_code, self.fe_item], device=self.device, stream=stream)
        wp.launch(_effective_active_kernel, dim=N, inputs=[
            self.active, self.fe_code, self.effective_active], device=self.device, stream=stream)
        wp.launch(_bounds_kernel, dim=N, inputs=[
            self.xpos, self.effective_active, self.gates, self.collision_mode,
            self.geom_rbound, self.geom_margin,
            self.body_geomadr, self.body_geomnum, self.geom_bodyid,
            self.aabb, self.frame, self.bounds_entered, self.bounds_count,
            self.bounds_status, self.bounds_scratch, self.ngeom, self.nbody],
            device=self.device, stream=stream)
        self._stage_guard(self.bounds_status, 1, (), stream)
        wp.launch(_broadphase_kernel, dim=N, inputs=[
            self.aabb, self.effective_active, self.gates, self.collision_mode,
            self.body_weldid, self.body_parentid,
            self.body_geomadr, self.body_geomnum, self.geom_bodyid, self.geom_type,
            self.geom_contype, self.geom_conaffinity, self.can, self.plane,
            self.filterparent_disabled, self.out_pair, self.out_npair,
            self.bp_status, self.sortbuf, self.sorttag, self.activebuf,
            self.nbody, self.pair_capacity],
            device=self.device, stream=stream)
        self._stage_guard(self.bp_status, 2, (1,), stream)
        wp.launch(_stagebuild_kernel, dim=N, inputs=[
            self.out_pair, self.out_npair, self.effective_active, self.body_geomnum,
            self.dispatch.static['pair_geom1'], self.dispatch.static['pair_geom2'],
            self.pair_signature, self.exclude_signature, self.midphase_disabled,
            self.gates, self.collision_mode,
            self.stage, self.stage_count, self.stage_route, self.stage_merged,
            self.seg_count, self.seg_pair_start, self.seg_pair_end,
            self.seg_geom_stage, self.trailing_start, self.trailing_end,
            self.sb_pairadr, self.sb_status, self.sb_first,
            self.nbody, self.pair_capacity, self.stage_slots,
            self.npair_static, self.nexclude],
            device=self.device, stream=stream)
        self._stage_guard(self.sb_status, 3, (3,), stream)
        # copy stages into C04 (device-to-device), seed route mask
        wp.copy(self.tree.stage, self.stage, stream=stream)
        wp.copy(self.tree.stage_count, self.stage_count, stream=stream)
        wp.copy(self.tree.active_in, self.effective_active, stream=stream)
        self.tree.launch(stream=stream)
        self._stage_guard(self.tree.status, 4, (5, 6, 9), stream)
        wp.launch(_req_to_dispatch_kernel, dim=N, inputs=[
            self.tree.out_req, self.tree.req_count, self.effective_active,
            self.dispatch.raw, self.dispatch.usermargin,
            self.dispatch.counts, self.dispatch.active, self.dispatch_origin,
            self.seg_count, self.seg_pair_start, self.seg_pair_end,
            self.seg_geom_stage, self.trailing_start, self.trailing_end,
            self.dispatch.static['pair_geom1'], self.dispatch.static['pair_geom2'],
            self.req_slots, self.slots, self.npair_static, self.stage_slots],
            device=self.device, stream=stream)
        self.dispatch.launch(stream=stream)
        self._stage_guard(self.dispatch.status, 5, (4,), stream)
        wp.launch(_dispatch_to_calls_kernel, dim=N, inputs=[
            self.dispatch.out_i, self.dispatch.out_f,
            self.dispatch.call_index, self.dispatch.call_count,
            self.dispatch.active, self.dispatch.status,
            self.dispatch_origin,
            self.stage_route, self.stage_slots,
            self.detect.call_g1, self.detect.call_g2,
            self.detect.call_type1, self.detect.call_type2,
            self.detect.call_margin,
            self.call_ipair, self.call_user,
            self.call_segment, self.call_sort, self.call_route_slot,
            self.detect.call_count, self.call_capacity, self.slots],
            device=self.device, stream=stream)
        # clear stale detect first_error from a previous launch (R01-22)
        wp.launch(_reset_error_kernel, dim=N, inputs=[self.effective_active, self.detect.fe_stage, self.detect.fe_code, self.detect.fe_item], device=self.device, stream=stream)
        wp.copy(self.detect.active, self.effective_active, stream=stream)
        self.detect.launch(stream=stream)
        wp.launch(_detect_error_kernel, dim=N, inputs=[
            self.detect.fe_code, self.detect.fe_item, self.active,
            self.fe_stage, self.fe_code, self.fe_item, self.effective_active],
            device=self.device, stream=stream)
        wp.launch(_calls_to_finalize_kernel, dim=N, inputs=[
            self.detect.call_g1, self.detect.call_g2,
            self.detect.call_type1, self.detect.call_type2,
            self.call_ipair, self.call_user,
            self.call_segment, self.call_sort,
            self.detect.call_count, self.detect.call_margin,
            self.detect.raw, self.detect.raw_count, self.detect.det_status,
            self.finalize.raw, self.finalize.raw_count,
            self.finalize.call_count, self.finalize.active,
            self.finalize.types, self.finalize.segment, self.finalize.sort_enabled,
            self.finalize.params.route, self.finalize.params.usermargin,
            self.finalize.params.counts, self.finalize.status,
            self.effective_active, N, self.call_capacity, 50, 10],
            device=self.device, stream=stream)
        self.finalize.launch(stream=stream)
        # chain-level sticky first_error + finalize commit guard (R01-4/20/21/22)
        wp.launch(_chain_error_kernel, dim=N, inputs=[
            self.bounds_status, self.bp_status, self.sb_status,
            self.tree.status, self.dispatch.status,
            self.detect.fe_code, self.detect.fe_item,
            self.finalize.status, self.finalize.ncon,
            self.finalize.warnings, self.contact_capacity,
            self.active,
            self.fe_stage, self.fe_code, self.fe_item],
            device=self.device, stream=stream)
        # The final stage can create the first error (including capacity).
        # Publish the same device mask as all preceding stage guards before
        # any downstream accounting or the next captured launch.
        wp.launch(_effective_active_kernel, dim=N, inputs=[
            self.active, self.fe_code, self.effective_active], device=self.device, stream=stream)

        wp.launch(_restore_failed_contacts_kernel, dim=N, inputs=[
            self.active, self.fe_code, self.before_contact_f, self.before_contact_i,
            self.finalize.contact_f, self.finalize.contact_i, self.contact_capacity],
            device=self.device, stream=stream)
        wp.launch(_body_counters_kernel, dim=N, inputs=[
            self.detect.call_count, self.call_segment, self.call_ipair,
            self.detect.raw_count, self.effective_active, self.collision_mode,
            self.gates, self.bodypair_broad, self.bodypair_narrow,
            self.call_capacity, self.npair_static, self.nbody],
            device=self.device, stream=stream)
        for name, arr in self.outputs.items():
            restore = {wp.int32: _restore_inactive_i32,
                       scalar.wp_real: _restore_inactive_f64,
                       wp.float32: _restore_inactive_f32}[arr.dtype]
            wp.launch(restore, dim=arr.size, inputs=[
                self.active, self.before_outputs[name], arr, arr.size // N],
                device=self.device, stream=stream)

    def _stage_guard(self, status, stage, capacity_codes, stream):
        codes = (*capacity_codes, -999, -999, -999)[:3]
        wp.launch(_stage_error_kernel, dim=self.nworld, inputs=[
            status, self.active, stage, *codes, self.fe_stage, self.fe_code,
            self.fe_item, self.effective_active], device=self.device, stream=stream)

    def reset_errors(self, mask, *, stream=None):
        """Explicit boundary reset; ordinary upload/launch never clears first_error."""
        a = np.asarray(mask)
        if a.dtype != np.int32 or a.shape != (self.nworld,) or not np.all((a == 0) | (a == 1)):
            raise ValueError('reset mask schema')
        binding = self._device_input_binding
        stream = stream or (binding.stream if binding is not None else wp.get_stream(self.device))
        self.validate_device_bindings(stream)
        # Warp array.assign uses the currently scoped stream. A reset upload
        # and its consumer must have the same ordering even on a nondefault one.
        with wp.ScopedStream(stream):
            self.reset_mask.assign(a)
        wp.launch(_reset_error_kernel, dim=self.nworld, inputs=[
            self.reset_mask, self.fe_stage, self.fe_code, self.fe_item],
            device=self.device, stream=stream)

    def download(self):
        out = self.finalize.download()
        raw = self.detect.raw_count.numpy().reshape(self.nworld, self.call_capacity)
        seg = self.call_segment.numpy().reshape(self.nworld, self.call_capacity)
        cc = self.detect.call_count.numpy()
        out.update({
            'bounds_status': self.bounds_status.numpy(),
            'bp_status': self.bp_status.numpy(),
            'bp_npair': self.out_npair.numpy(),
            'stage_count': self.stage_count.numpy(),
            'sb_status': self.sb_status.numpy(),
            'bodypair_broad': self.bodypair_broad.numpy(),
            'bodypair_narrow': self.bodypair_narrow.numpy(),
            'tree_req_count': self.tree.req_count.numpy(),
            'tree_status': self.tree.status.numpy(),
            'bvh_active': self.tree.out_active.numpy().reshape(self.nworld, -1),
            'dispatch_status': self.dispatch.status.numpy(),
            'dispatch_call_count': self.dispatch.call_count.numpy(),
            'dispatch_mid_count': self.dispatch.mid_count.numpy(),
            'dispatch_request_count': self.dispatch.counts.numpy(),
            'det_status': self.detect.det_status.numpy().reshape(self.nworld, self.call_capacity),
            'raw_count': raw,
            'detect_fe_code': self.detect.fe_code.numpy(),
            'detect_fe_item': self.detect.fe_item.numpy(),
            'fe_stage': self.fe_stage.numpy(),
            'fe_code': self.fe_code.numpy(),
            'fe_item': self.fe_item.numpy(),
        })
        n, p, g = self.nworld, self.call_capacity, self.ngeom
        out.update({
            'raw': self.detect.raw.numpy().reshape(n, p, 50, 10),
            'call_g1': self.detect.call_g1.numpy().reshape(n, p),
            'call_g2': self.detect.call_g2.numpy().reshape(n, p),
            'call_type1': self.detect.call_type1.numpy().reshape(n, p),
            'call_type2': self.detect.call_type2.numpy().reshape(n, p),
            'call_margin': self.detect.call_margin.numpy().reshape(n, p),
            'call_count': self.detect.call_count.numpy(),
            'call_ipair': self.call_ipair.numpy().reshape(n, p),
            'call_segment': self.call_segment.numpy().reshape(n, p),
            'call_sort': self.call_sort.numpy().reshape(n, p),
            'tree_requests': self.tree.out_req.numpy().reshape(n, self.req_slots, 4),
            'dispatch_requests': self.dispatch.raw.numpy().reshape(n, self.slots, 3),
            'dispatch_origin': self.dispatch_origin.numpy().reshape(n, self.slots, 2),
            'stage_rows': self.stage.numpy().reshape(n, self.stage_slots, 6),
            'stage_route': self.stage_route.numpy().reshape(n, self.stage_slots),
            'native_status': self.detect.native_status.numpy().reshape(n, p),
            'native_diag': self.detect.native_diag.numpy().reshape(n, p, 16),
            'native_framewritten': self.detect.native_framewritten.numpy().reshape(n, p),
            'before_raw': self.before_outputs['detect_raw'].numpy().reshape(n, p, 50, 10),
            'effective_active': self.effective_active.numpy(),
            'input_active': self.active.numpy(),
            'input_xpos': self.xpos.numpy().reshape(n, g, 3),
            'input_xmat': self.xmat.numpy().reshape(n, g, 9),
            'input_xipos': self.tree.xipos.numpy().reshape(n, self.nbody, 3),
            'input_ximat': self.tree.ximat.numpy().reshape(n, self.nbody, 9),
            'detector_input_xpos': self.detect.xpos.numpy().reshape(n, g, 3),
            'detector_input_xmat': self.detect.xmat.numpy().reshape(n, g, 9),
            'before_contact_f': self.before_contact_f.numpy().reshape(n, self.contact_capacity, 65),
            'before_contact_i': self.before_contact_i.numpy().reshape(n, self.contact_capacity, 5),
        })
        for name, arr in self.outputs.items():
            out['all_' + name] = arr.numpy().reshape(n, -1)
            out['before_all_' + name] = self.before_outputs[name].numpy().reshape(n, -1)
        return out
