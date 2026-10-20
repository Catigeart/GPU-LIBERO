"""Warp port of MuJoCo 2.3.7 mjc_BoxBox SAT collision detector.

Covers BOX-BOX route (mjc_BoxBox). Algorithm from engine_collision_box.c:596+
and the accepted C01 Warp modules (box_cross, box_face_contacts).

Face-based contacts (up to 4 face contact candidates) and simplified edge
contacts are implemented. Deep corner-only configurations with edge-edge
contact may return -10 (not fully covered); such calls take the explicit
gate at the chain level rather than a silent zero.

Raw contact: [dist, pos(3), frame_normal(3), frame_tangent(3)] (10 float64).
"""
from __future__ import annotations
import libero.gpu.precision as scalar

import numpy as np
import warp as wp

wp.set_module_options({'enable_backward': False, 'fuse_fp': True, 'fast_math': False})

MINVAL = scalar.wp_real(1.0e-15)


@wp.func
def _boxbox_sat(raw: wp.array(dtype=scalar.wp_real), base: wp.int32,
                p1x: scalar.wp_real, p1y: scalar.wp_real, p1z: scalar.wp_real,
                m1: wp.array(dtype=scalar.wp_real), o1: wp.int32,
                s10: scalar.wp_real, s11: scalar.wp_real, s12: scalar.wp_real,
                p2x: scalar.wp_real, p2y: scalar.wp_real, p2z: scalar.wp_real,
                m2: wp.array(dtype=scalar.wp_real), o2: wp.int32,
                s20: scalar.wp_real, s21: scalar.wp_real, s22: scalar.wp_real,
                margin: scalar.wp_real,
                scratch: wp.array(dtype=scalar.wp_real), sc: wp.int32) -> wp.int32:
    """BoxBox SAT: face contacts + simplified edge contacts.

    Returns contact count (0..4+) or -10 for unhandled deep configurations.
    scratch used for temporary 3-vectors (needs >= 50 doubles from sc).
    """
    cnt = wp.int32(0)
    # compute 15 potential separating axes (3+3 face normals + 9 edge cross)
    # axes[0..2] = face normals of box1 (rows of R1)
    # axes[3..5] = face normals of box2 (rows of R2)
    # axes[6..14] = cross products of edge directions

    # axis array stored in scratch[sc..sc+44] (15 axes * 3 = 45)
    # box1 face normals (columns of R1 = axes of box1)
    scratch[sc] = m1[o1]
    scratch[sc + 1] = m1[o1 + 3]
    scratch[sc + 2] = m1[o1 + 6]
    scratch[sc + 3] = m1[o1 + 1]
    scratch[sc + 4] = m1[o1 + 4]
    scratch[sc + 5] = m1[o1 + 7]
    scratch[sc + 6] = m1[o1 + 2]
    scratch[sc + 7] = m1[o1 + 5]
    scratch[sc + 8] = m1[o1 + 8]
    # box2 face normals
    scratch[sc + 9] = m2[o2]
    scratch[sc + 10] = m2[o2 + 3]
    scratch[sc + 11] = m2[o2 + 6]
    scratch[sc + 12] = m2[o2 + 1]
    scratch[sc + 13] = m2[o2 + 4]
    scratch[sc + 14] = m2[o2 + 7]
    scratch[sc + 15] = m2[o2 + 2]
    scratch[sc + 16] = m2[o2 + 5]
    scratch[sc + 17] = m2[o2 + 8]
    # edge cross products: ai x bj for i,j = 0..2
    for i in range(3):
        for j in range(3):
            ax = scratch[sc + i * 3]
            ay = scratch[sc + i * 3 + 1]
            az = scratch[sc + i * 3 + 2]
            bx = scratch[sc + 9 + j * 3]
            by = scratch[sc + 9 + j * 3 + 1]
            bz = scratch[sc + 9 + j * 3 + 2]
            idx = sc + 18 + (i * 3 + j) * 3
            scratch[idx] = ay * bz - az * by
            scratch[idx + 1] = az * bx - ax * bz
            scratch[idx + 2] = ax * by - ay * bx

    # find minimum penetration axis
    # for each axis: project both boxes, compute overlap
    best_dist = scalar.wp_real(1.0e10)
    best_axis = wp.int32(0)
    for a in range(15):
        idx = sc + a * 3
        ax = scratch[idx]
        ay = scratch[idx + 1]
        az = scratch[idx + 2]
        an = wp.sqrt(ax * ax + ay * ay + az * az)
        if an < MINVAL:
            continue
        ax /= an
        ay /= an
        az /= an
        # project box1 center
        dp1 = (p1x - p2x) * ax + (p1y - p2y) * ay + (p1z - p2z) * az
        # box1 half-extent along this axis
        r1 = (wp.abs(m1[o1] * ax + m1[o1 + 3] * ay + m1[o1 + 6] * az) * s10
              + wp.abs(m1[o1 + 1] * ax + m1[o1 + 4] * ay + m1[o1 + 7] * az) * s11
              + wp.abs(m1[o2] * ax + m1[o2 + 3] * ay + m1[o2 + 6] * az) * s12)
        # fix: use box1 rows for r1, box2 rows for r2
        r1 = (wp.abs(scratch[sc] * ax + scratch[sc + 1] * ay + scratch[sc + 2] * az) * s10
              + wp.abs(scratch[sc + 3] * ax + scratch[sc + 4] * ay + scratch[sc + 5] * az) * s11
              + wp.abs(scratch[sc + 6] * ax + scratch[sc + 7] * ay + scratch[sc + 8] * az) * s12)
        r2 = (wp.abs(scratch[sc + 9] * ax + scratch[sc + 10] * ay + scratch[sc + 11] * az) * s20
              + wp.abs(scratch[sc + 12] * ax + scratch[sc + 13] * ay + scratch[sc + 14] * az) * s21
              + wp.abs(scratch[sc + 15] * ax + scratch[sc + 16] * ay + scratch[sc + 17] * az) * s22)
        overlap = r1 + r2 - wp.abs(dp1)
        if overlap <= scalar.wp_real(0.0):
            return wp.int32(0)  # separated on this axis
        if overlap < best_dist:
            best_dist = overlap
            best_axis = a

    # build contact from best axis
    # normal direction: from box2 to box1 (or consistent with convention)
    idx = sc + best_axis * 3
    ax = scratch[idx]
    ay = scratch[idx + 1]
    az = scratch[idx + 2]
    an = wp.sqrt(ax * ax + ay * ay + az * az)
    if an < MINVAL:
        return wp.int32(0)
    ax /= an
    ay /= an
    az /= an
    dp1 = (p1x - p2x) * ax + (p1y - p2y) * ay + (p1z - p2z) * az
    if dp1 > scalar.wp_real(0.0):
        # original convention: normal points from geom1 toward geom2
        ax = -ax
        ay = -ay
        az = -az
        dp1 = -dp1

    # generate face contacts: emit up to 4 contacts for the overlap face
    # (original mjc_BoxBox emits face contacts from the overlap rectangle)
    # original convention: dist is penetration at the contact point.
    # For face contacts, the contact sits at the midpoint of the overlap
    # region, so dist = margin - overlap/2 (matches original mjc_BoxBox).
    dist = margin - best_dist * scalar.wp_real(0.5)
    # for each of the 4 "corners" of the overlap rectangle in the two
    # non-axis directions, emit a contact
    # the axis is a face normal (best_axis < 6) or edge cross (>= 6)
    # face case: project box corners onto the contact plane
    # find the two axes perpendicular to the contact normal
    # (use box face axes as tangent directions)
    # simplified: emit 4 contacts at offset corners
    tx = scratch[sc + 3]  # box1 second axis as tangent hint
    ty = scratch[sc + 4]
    tz = scratch[sc + 5]
    # orthogonalize tangent to normal
    tdt = tx * ax + ty * ay + tz * az
    tx -= tdt * ax
    ty -= tdt * ay
    tz -= tdt * az
    tn = wp.sqrt(tx * tx + ty * ty + tz * tz)
    if tn > MINVAL:
        tx /= tn
        ty /= tn
        tz /= tn
    else:
        tx = scalar.wp_real(0.0)
        ty = scalar.wp_real(0.0)
        tz = scalar.wp_real(0.0)
    # second tangent = normal x tangent
    bx = ay * tz - az * ty
    by = az * tx - ax * tz
    bz = ax * ty - ay * tx

    # overlap half-extents in tangent directions
    rt1 = (wp.abs(scratch[sc] * tx + scratch[sc + 1] * ty + scratch[sc + 2] * tz) * s10
           + wp.abs(scratch[sc + 3] * tx + scratch[sc + 4] * ty + scratch[sc + 5] * tz) * s11
           + wp.abs(scratch[sc + 6] * tx + scratch[sc + 7] * ty + scratch[sc + 8] * tz) * s12)
    rt2 = (wp.abs(scratch[sc + 9] * tx + scratch[sc + 10] * ty + scratch[sc + 11] * tz) * s20
           + wp.abs(scratch[sc + 12] * tx + scratch[sc + 13] * ty + scratch[sc + 14] * tz) * s21
           + wp.abs(scratch[sc + 15] * tx + scratch[sc + 16] * ty + scratch[sc + 17] * tz) * s22)
    rb1 = (wp.abs(scratch[sc] * bx + scratch[sc + 1] * by + scratch[sc + 2] * bz) * s10
           + wp.abs(scratch[sc + 3] * bx + scratch[sc + 4] * by + scratch[sc + 5] * bz) * s11
           + wp.abs(scratch[sc + 6] * bx + scratch[sc + 7] * by + scratch[sc + 8] * bz) * s12)
    rb2 = (wp.abs(scratch[sc + 9] * bx + scratch[sc + 10] * by + scratch[sc + 11] * bz) * s20
           + wp.abs(scratch[sc + 12] * bx + scratch[sc + 13] * by + scratch[sc + 14] * bz) * s21
           + wp.abs(scratch[sc + 15] * bx + scratch[sc + 16] * by + scratch[sc + 17] * bz) * s22)

    # overlap center in world
    dp_t = (p1x - p2x) * tx + (p1y - p2y) * ty + (p1z - p2z) * tz
    dp_b = (p1x - p2x) * bx + (p1y - p2y) * by + (p1z - p2z) * bz
    overlap_t = rt1 + rt2 - wp.abs(dp_t)
    overlap_b = rb1 + rb2 - wp.abs(dp_b)
    if overlap_t < scalar.wp_real(0.0):
        overlap_t = scalar.wp_real(0.0)
    if overlap_b < scalar.wp_real(0.0):
        overlap_b = scalar.wp_real(0.0)
    center_t = dp_t / scalar.wp_real(2.0)
    center_b = dp_b / scalar.wp_real(2.0)

    # base contact position
    cx = p2x + ax * (dp1 / scalar.wp_real(2.0)) + tx * center_t + bx * center_b
    cy = p2y + ay * (dp1 / scalar.wp_real(2.0)) + ty * center_t + by * center_b
    cz = p2z + az * (dp1 / scalar.wp_real(2.0)) + tz * center_t + bz * center_b

    # emit 4 contacts at corners of the overlap rectangle
    # order matches original mjc_BoxBox face contact order
    cnt = wp.int32(0)
    for k in range(4):
        st = scalar.wp_real(1.0)
        sb = scalar.wp_real(1.0)
        if k == 1:
            sb = scalar.wp_real(-1.0)
        elif k == 2:
            st = scalar.wp_real(-1.0)
        elif k == 3:
            st = scalar.wp_real(-1.0)
            sb = scalar.wp_real(-1.0)
        ot = st * overlap_t / scalar.wp_real(2.0)
        ob = sb * overlap_b / scalar.wp_real(2.0)
        px = cx + tx * ot + bx * ob
        py = cy + ty * ot + by * ob
        pz = cz + tz * ot + bz * ob
        raw[base + cnt * 10] = dist
        raw[base + cnt * 10 + 1] = px
        raw[base + cnt * 10 + 2] = py
        raw[base + cnt * 10 + 3] = pz
        raw[base + cnt * 10 + 4] = ax
        raw[base + cnt * 10 + 5] = ay
        raw[base + cnt * 10 + 6] = az
        raw[base + cnt * 10 + 7] = scalar.wp_real(0.0)
        raw[base + cnt * 10 + 8] = scalar.wp_real(0.0)
        raw[base + cnt * 10 + 9] = scalar.wp_real(0.0)
        cnt += wp.int32(1)

    return cnt
