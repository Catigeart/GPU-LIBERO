"""Warp ports of the pinned MuJoCo 2.3.7 narrow-phase detectors.

Algorithm order follows the frozen sources in evidence CAS:
  engine_collision_primitive.c  sha256 3e94a3c81c32d27cd737bc5f3ada61e4a9779dcf94d9dff60b61d0b6c071383c
      PlaneSphere/Capsule/Cylinder/Box, SphereSphere/Capsule/Cylinder,
      CapsuleCapsule
  engine_collision_box.c        sha256 fd038c185abfae66213bcabd0db34450a5fb596481fa5d1bb567fdff62a587b2
      SphereBox
  engine_collision_convex.c     sha256 87fb8624056d0a8a02258225aa58f664452a334ce4048583d3f095418523c945
      PlaneConvex (support + mesh extra vertices)

Raw contact layout per slot (10 float64), matching scripts/gpu_contact_finalize
RAW_F: [dist, pos.x, pos.y, pos.z, frame.n.x, frame.n.y, frame.n.z,
        frame.t.x, frame.t.y, frame.t.z]. frame.n is the original
con.frame[0..2] unit normal; frame.t is the original con.frame[3..5] y-hint
before mju_makeFrame. BoxBox exact-position dedup downstream compares pos.

An unimplemented route (DET_UNSUPPORTED) raises ERR_UNSUPPORTED_ROUTE and
writes no contacts. There is no path that silently returns zero contacts for
an unimplemented route; a legal detector returning 0 contacts is a distinct,
recorded outcome (the call still counts at the dispatch/mid layer).
"""
from __future__ import annotations
import libero.gpu.precision as scalar

import numpy as np
import warp as wp

# Import new detector implementations at module level so Warp can link them
from .native import prepare_native, launch_native

wp.set_module_options({'enable_backward': False, 'fuse_fp': True, 'fast_math': False})

MAX_RAW = 50
RAW_F = 10
MINVAL = scalar.wp_real(1.0e-15)

DET_NONE = 0
DET_PLANE_SPHERE = 1
DET_PLANE_CAPSULE = 2
DET_PLANE_CYLINDER = 3
DET_PLANE_BOX = 4
DET_PLANE_CONVEX = 5
DET_SPHERE_SPHERE = 6
DET_SPHERE_CAPSULE = 7
DET_SPHERE_CYLINDER = 8
DET_SPHERE_BOX = 9
DET_CAPSULE_CAPSULE = 10
DET_CONVEX_MPR = 11
DET_BOXBOX_SAT = 12
DET_CAPSULE_BOX_NATIVE = 13
DET_UNSUPPORTED = -1

DETECTOR_NAME = {
    DET_NONE: 'none',
    DET_PLANE_SPHERE: 'plane_sphere',
    DET_PLANE_CAPSULE: 'plane_capsule',
    DET_PLANE_CYLINDER: 'plane_cylinder',
    DET_PLANE_BOX: 'plane_box',
    DET_PLANE_CONVEX: 'plane_convex',
    DET_SPHERE_SPHERE: 'sphere_sphere',
    DET_SPHERE_CAPSULE: 'sphere_capsule',
    DET_SPHERE_CYLINDER: 'sphere_cylinder',
    DET_SPHERE_BOX: 'sphere_box',
    DET_CAPSULE_CAPSULE: 'capsule_capsule',
    DET_CONVEX_MPR: 'convex_mpr',
    DET_BOXBOX_SAT: 'boxbox_native',
    DET_CAPSULE_BOX_NATIVE: 'capsule_box_native',
    DET_UNSUPPORTED: 'unsupported_gate',
}

_NAME_TO_ID = {
    'plane_sphere': DET_PLANE_SPHERE,
    'plane_capsule': DET_PLANE_CAPSULE,
    'plane_cylinder': DET_PLANE_CYLINDER,
    'plane_box': DET_PLANE_BOX,
    'plane_mesh': DET_PLANE_CONVEX,
    'plane_ellipsoid': DET_PLANE_CONVEX,
    'sphere_sphere': DET_SPHERE_SPHERE,
    'sphere_capsule': DET_SPHERE_CAPSULE,
    'sphere_cylinder': DET_SPHERE_CYLINDER,
    'sphere_box': DET_SPHERE_BOX,
    'capsule_capsule': DET_CAPSULE_CAPSULE,
    'convex_mpr': DET_CONVEX_MPR,
    'boxbox_native': DET_BOXBOX_SAT,
    'capsule_box_native': DET_CAPSULE_BOX_NATIVE,
}


def build_route_ids() -> np.ndarray:
    """8x8 int32 detector ids with mirrored lower triangle.

    present + warp_impl -> detector id; present but unimplemented ->
    DET_UNSUPPORTED; absent slot -> DET_NONE.
    """
    from .routes import DEFAULT_PRESENT, detector_status, route_key
    ids = np.zeros((8, 8), dtype=np.int32)
    for r in range(8):
        for c in range(8):
            key = route_key(r, c)
            if r != key[0]:
                ids[r, c] = ids[key[0], key[1]]
                continue
            status, impl = detector_status(r, c)
            if status == 'absent_table_slot':
                ids[r, c] = DET_NONE
            elif status in ('warp_impl', 'native_cuda_impl'):
                ids[r, c] = _NAME_TO_ID[impl]
            else:
                ids[r, c] = DET_UNSUPPORTED
    return ids


@wp.func
def _emit(raw: wp.array(dtype=scalar.wp_real), base: wp.int32, r: wp.int32,
          dist: scalar.wp_real, px: scalar.wp_real, py: scalar.wp_real, pz: scalar.wp_real,
          nx: scalar.wp_real, ny: scalar.wp_real, nz: scalar.wp_real,
          tx: scalar.wp_real, ty: scalar.wp_real, tz: scalar.wp_real) -> wp.int32:
    if r >= MAX_RAW:
        return wp.int32(0)
    o = base + r * RAW_F
    raw[o] = dist
    raw[o + 1] = px
    raw[o + 2] = py
    raw[o + 3] = pz
    raw[o + 4] = nx
    raw[o + 5] = ny
    raw[o + 6] = nz
    raw[o + 7] = tx
    raw[o + 8] = ty
    raw[o + 9] = tz
    return wp.int32(1)


@wp.func
def _norm3(x: scalar.wp_real, y: scalar.wp_real, z: scalar.wp_real) -> scalar.wp_real:
    return wp.sqrt(x * x + y * y + z * z)


@wp.func
def _sign(x: scalar.wp_real) -> scalar.wp_real:
    # mju_sign: -1 / 0 / +1 (engine_util_misc.c:757-765)
    if x < scalar.wp_real(0.0):
        return scalar.wp_real(-1.0)
    elif x > scalar.wp_real(0.0):
        return scalar.wp_real(1.0)
    return scalar.wp_real(0.0)


@wp.func
def _rot(mx: wp.array(dtype=scalar.wp_real), o: wp.int32,
         vx: scalar.wp_real, vy: scalar.wp_real, vz: scalar.wp_real,
         out: wp.array(dtype=scalar.wp_real), oo: wp.int32):
    # mju_rotVecMat: res = row-major mat * vec
    out[oo] = mx[o] * vx + mx[o + 1] * vy + mx[o + 2] * vz
    out[oo + 1] = mx[o + 3] * vx + mx[o + 4] * vy + mx[o + 5] * vz
    out[oo + 2] = mx[o + 6] * vx + mx[o + 7] * vy + mx[o + 8] * vz


@wp.func
def _rot_t(mx: wp.array(dtype=scalar.wp_real), o: wp.int32,
           vx: scalar.wp_real, vy: scalar.wp_real, vz: scalar.wp_real,
           out: wp.array(dtype=scalar.wp_real), oo: wp.int32):
    # mju_rotVecMatT: res = row-major mat^T * vec
    out[oo] = mx[o] * vx + mx[o + 3] * vy + mx[o + 6] * vz
    out[oo + 1] = mx[o + 1] * vx + mx[o + 4] * vy + mx[o + 7] * vz
    out[oo + 2] = mx[o + 2] * vx + mx[o + 5] * vy + mx[o + 8] * vz


# ---------------------------------------------------------------------------
# Detector bodies. Each writes raw contacts at base and returns the count.
# Geom1 is the type-swapped lower-type geom (as stored by dispatch out_i).
# ---------------------------------------------------------------------------


@wp.func
def _plane_sphere(raw: wp.array(dtype=scalar.wp_real), base: wp.int32,
                  p1x: scalar.wp_real, p1y: scalar.wp_real, p1z: scalar.wp_real,
                  m1: wp.array(dtype=scalar.wp_real), o1: wp.int32,
                  p2x: scalar.wp_real, p2y: scalar.wp_real, p2z: scalar.wp_real,
                  r2: scalar.wp_real, margin: scalar.wp_real) -> wp.int32:
    nx = m1[o1 + 2]
    ny = m1[o1 + 5]
    nz = m1[o1 + 8]
    cdist = (p2x - p1x) * nx + (p2y - p1y) * ny + (p2z - p1z) * nz
    if cdist > margin + r2:
        return wp.int32(0)
    dist = cdist - r2
    scl = -dist * scalar.wp_real(0.5) - r2
    return _emit(raw, base, 0, dist,
                 p2x + nx * scl, p2y + ny * scl, p2z + nz * scl,
                 nx, ny, nz, scalar.wp_real(0.0), scalar.wp_real(0.0), scalar.wp_real(0.0))


@wp.func
def _sphere_sphere_r(raw: wp.array(dtype=scalar.wp_real), base: wp.int32,
                     p1x: scalar.wp_real, p1y: scalar.wp_real, p1z: scalar.wp_real,
                     m1: wp.array(dtype=scalar.wp_real), o1: wp.int32, r1: scalar.wp_real,
                     p2x: scalar.wp_real, p2y: scalar.wp_real, p2z: scalar.wp_real,
                     m2: wp.array(dtype=scalar.wp_real), o2: wp.int32, r2: scalar.wp_real,
                     margin: scalar.wp_real) -> wp.int32:
    dx = p1x - p2x
    dy = p1y - p2y
    dz = p1z - p2z
    cdist_sqr = dx * dx + dy * dy + dz * dz
    min_dist = margin + r1 + r2
    if cdist_sqr > min_dist * min_dist:
        return wp.int32(0)
    dist = wp.sqrt(cdist_sqr) - r1 - r2
    fx = p2x - p1x
    fy = p2y - p1y
    fz = p2z - p1z
    length = _norm3(fx, fy, fz)
    if length < MINVAL:
        ax = m1[o1 + 2]
        ay = m1[o1 + 5]
        az = m1[o1 + 8]
        bx = m2[o2 + 2]
        by = m2[o2 + 5]
        bz = m2[o2 + 8]
        fx = ay * bz - az * by
        fy = az * bx - ax * bz
        fz = ax * by - ay * bx
        length = _norm3(fx, fy, fz)
        if length < MINVAL:
            fx = scalar.wp_real(1.0)
            fy = scalar.wp_real(0.0)
            fz = scalar.wp_real(0.0)
        else:
            inv = scalar.wp_real(1.0) / length
            fx *= inv
            fy *= inv
            fz *= inv
    else:
        inv = scalar.wp_real(1.0) / length
        fx *= inv
        fy *= inv
        fz *= inv
    scl = r1 + dist * scalar.wp_real(0.5)
    return _emit(raw, base, 0, dist,
                 p1x + fx * scl, p1y + fy * scl, p1z + fz * scl,
                 fx, fy, fz, scalar.wp_real(0.0), scalar.wp_real(0.0), scalar.wp_real(0.0))


@wp.func
def _plane_capsule(raw: wp.array(dtype=scalar.wp_real), base: wp.int32,
                   p1x: scalar.wp_real, p1y: scalar.wp_real, p1z: scalar.wp_real,
                   m1: wp.array(dtype=scalar.wp_real), o1: wp.int32,
                   p2x: scalar.wp_real, p2y: scalar.wp_real, p2z: scalar.wp_real,
                   m2: wp.array(dtype=scalar.wp_real), o2: wp.int32,
                   r2: scalar.wp_real, hl2: scalar.wp_real, margin: scalar.wp_real) -> wp.int32:
    # mjc_PlaneCapsule: two sphere-plane tests at segment endpoints, frames
    # aligned with the capsule axis (con.frame[3..5] = axis hint).
    ax = m2[o2 + 2]
    ay = m2[o2 + 5]
    az = m2[o2 + 8]
    sx = hl2 * ax
    sy = hl2 * ay
    sz = hl2 * az
    n1 = _plane_sphere(raw, base, p1x, p1y, p1z, m1, o1,
                       p2x + sx, p2y + sy, p2z + sz, r2, margin)
    n2 = wp.int32(0)
    if n1 > 0:
        o = base + 7
        raw[o] = ax
        raw[o + 1] = ay
        raw[o + 2] = az
    else:
        n2 = _plane_sphere(raw, base, p1x, p1y, p1z, m1, o1,
                           p2x - sx, p2y - sy, p2z - sz, r2, margin)
        if n2 > 0:
            o = base + 7
            raw[o] = ax
            raw[o + 1] = ay
            raw[o + 2] = az
        return n2
    n2 = _plane_sphere(raw, base + RAW_F, p1x, p1y, p1z, m1, o1,
                       p2x - sx, p2y - sy, p2z - sz, r2, margin)
    if n2 > 0:
        o = base + RAW_F + 7
        raw[o] = ax
        raw[o + 1] = ay
        raw[o + 2] = az
    return n1 + n2


@wp.func
def _plane_cylinder(raw: wp.array(dtype=scalar.wp_real), base: wp.int32,
                    p1x: scalar.wp_real, p1y: scalar.wp_real, p1z: scalar.wp_real,
                    m1: wp.array(dtype=scalar.wp_real), o1: wp.int32,
                    p2x: scalar.wp_real, p2y: scalar.wp_real, p2z: scalar.wp_real,
                    m2: wp.array(dtype=scalar.wp_real), o2: wp.int32,
                    rad: scalar.wp_real, hl: scalar.wp_real, margin: scalar.wp_real) -> wp.int32:
    # mjc_PlaneCylinder (engine_collision_primitive.c:97-199)
    nx = m1[o1 + 2]
    ny = m1[o1 + 5]
    nz = m1[o1 + 8]
    ax = m2[o2 + 2]
    ay = m2[o2 + 5]
    az = m2[o2 + 8]
    prjaxis = nx * ax + ny * ay + nz * az
    if prjaxis > scalar.wp_real(0.0):
        ax = -ax
        ay = -ay
        az = -az
        prjaxis = -prjaxis
    vx = p2x - p1x
    vy = p2y - p1y
    vz = p2z - p1z
    dist0 = vx * nx + vy * ny + vz * nz
    # vec = axis*prjaxis - normal  (scl axis, subFrom normal)
    vx = ax * prjaxis - nx
    vy = ay * prjaxis - ny
    vz = az * prjaxis - nz
    len_sqr = vx * vx + vy * vy + vz * vz
    if len_sqr >= MINVAL * MINVAL:
        scl = rad / wp.sqrt(len_sqr)
        vx *= scl
        vy *= scl
        vz *= scl
    else:
        vx = m2[o2] * rad
        vy = m2[o2 + 3] * rad
        vz = m2[o2 + 6] * rad
    prjvec = vx * nx + vy * ny + vz * nz
    ax = ax * hl
    ay = ay * hl
    az = az * hl
    prjaxis = prjaxis * hl
    cnt = wp.int32(0)
    # first point
    if dist0 + prjaxis + prjvec <= margin:
        dist = dist0 + prjaxis + prjvec
        cnt = _emit(raw, base, 0, dist,
                    p2x + vx + ax - nx * dist * scalar.wp_real(0.5),
                    p2y + vy + ay - ny * dist * scalar.wp_real(0.5),
                    p2z + vz + az - nz * dist * scalar.wp_real(0.5),
                    nx, ny, nz, scalar.wp_real(0.0), scalar.wp_real(0.0), scalar.wp_real(0.0))
    else:
        return wp.int32(0)
    # second point
    if dist0 - prjaxis + prjvec <= margin:
        dist = dist0 - prjaxis + prjvec
        cnt += _emit(raw, base, 1, dist,
                     p2x + vx - ax - nx * dist * scalar.wp_real(0.5),
                     p2y + vy - ay - ny * dist * scalar.wp_real(0.5),
                     p2z + vz - az - nz * dist * scalar.wp_real(0.5),
                     nx, ny, nz, scalar.wp_real(0.0), scalar.wp_real(0.0), scalar.wp_real(0.0))
    # triangle points on the side closer to the plane
    prjvec1 = -prjvec * scalar.wp_real(0.5)
    if dist0 + prjaxis + prjvec1 <= margin:
        # vec1 = normalize(cross(vec, axis)) * rad * sqrt(3)/2
        # R01-29: the original always constructs both side contacts; only the
        # normalization is guarded (mju_normalize3 falls back to (1,0,0)).
        cx = vy * az - vz * ay
        cy = vz * ax - vx * az
        cz = vx * ay - vy * ax
        clen = _norm3(cx, cy, cz)
        if clen >= MINVAL:
            inv = scalar.wp_real(1.0) / clen
            cx *= inv
            cy *= inv
            cz *= inv
        else:
            cx = scalar.wp_real(1.0)
            cy = scalar.wp_real(0.0)
            cz = scalar.wp_real(0.0)
        scl = rad * wp.sqrt(scalar.wp_real(3.0)) * scalar.wp_real(0.5)
        cx *= scl
        cy *= scl
        cz *= scl
        dist = dist0 + prjaxis + prjvec1
        # point A
        cnt += _emit(raw, base, cnt, dist,
                     p2x + cx + ax - vx * scalar.wp_real(0.5) - nx * dist * scalar.wp_real(0.5),
                     p2y + cy + ay - vy * scalar.wp_real(0.5) - ny * dist * scalar.wp_real(0.5),
                     p2z + cz + az - vz * scalar.wp_real(0.5) - nz * dist * scalar.wp_real(0.5),
                     nx, ny, nz, scalar.wp_real(0.0), scalar.wp_real(0.0), scalar.wp_real(0.0))
        # point B
        cnt += _emit(raw, base, cnt, dist,
                     p2x - cx + ax - vx * scalar.wp_real(0.5) - nx * dist * scalar.wp_real(0.5),
                     p2y - cy + ay - vy * scalar.wp_real(0.5) - ny * dist * scalar.wp_real(0.5),
                     p2z - cz + az - vz * scalar.wp_real(0.5) - nz * dist * scalar.wp_real(0.5),
                     nx, ny, nz, scalar.wp_real(0.0), scalar.wp_real(0.0), scalar.wp_real(0.0))
    return cnt


@wp.func
def _plane_box(raw: wp.array(dtype=scalar.wp_real), base: wp.int32,
               p1x: scalar.wp_real, p1y: scalar.wp_real, p1z: scalar.wp_real,
               m1: wp.array(dtype=scalar.wp_real), o1: wp.int32,
               p2x: scalar.wp_real, p2y: scalar.wp_real, p2z: scalar.wp_real,
               m2: wp.array(dtype=scalar.wp_real), o2: wp.int32,
               s20: scalar.wp_real, s21: scalar.wp_real, s22: scalar.wp_real,
               margin: scalar.wp_real) -> wp.int32:
    # mjc_PlaneBox: up to 4 contacts from the 8 corners below the margin,
    # skipping corners that point up (ldist > 0).
    nx = m1[o1 + 2]
    ny = m1[o1 + 5]
    nz = m1[o1 + 8]
    dist = (p2x - p1x) * nx + (p2y - p1y) * ny + (p2z - p1z) * nz
    cnt = wp.int32(0)
    for i in range(8):
        sx = s20
        if (i & 1) == 0:
            sx = -s20
        sy = s21
        if (i & 2) == 0:
            sy = -s21
        sz = s22
        if (i & 4) == 0:
            sz = -s22
        cx = m2[o2] * sx + m2[o2 + 1] * sy + m2[o2 + 2] * sz
        cy = m2[o2 + 3] * sx + m2[o2 + 4] * sy + m2[o2 + 5] * sz
        cz = m2[o2 + 6] * sx + m2[o2 + 7] * sy + m2[o2 + 8] * sz
        ldist = nx * cx + ny * cy + nz * cz
        if dist + ldist > margin or ldist > scalar.wp_real(0.0):
            continue
        d = dist + ldist
        cnt += _emit(raw, base, cnt, d,
                     p2x + cx - nx * d * scalar.wp_real(0.5),
                     p2y + cy - ny * d * scalar.wp_real(0.5),
                     p2z + cz - nz * d * scalar.wp_real(0.5),
                     nx, ny, nz, scalar.wp_real(0.0), scalar.wp_real(0.0), scalar.wp_real(0.0))
        if cnt >= 4:
            return wp.int32(4)
    return cnt


@wp.func
def _sphere_capsule(raw: wp.array(dtype=scalar.wp_real), base: wp.int32,
                    p1x: scalar.wp_real, p1y: scalar.wp_real, p1z: scalar.wp_real,
                    m1: wp.array(dtype=scalar.wp_real), o1: wp.int32, r1: scalar.wp_real,
                    p2x: scalar.wp_real, p2y: scalar.wp_real, p2z: scalar.wp_real,
                    m2: wp.array(dtype=scalar.wp_real), o2: wp.int32,
                    r2: scalar.wp_real, hl2: scalar.wp_real, margin: scalar.wp_real) -> wp.int32:
    # mjc_SphereCapsule: clip to segment, then sphere-sphere with capsule radius.
    ax = m2[o2 + 2]
    ay = m2[o2 + 5]
    az = m2[o2 + 8]
    vx = p1x - p2x
    vy = p1y - p2y
    vz = p1z - p2z
    x = ax * vx + ay * vy + az * vz
    # mju_clip order: min first, then max (engine_util_misc.c:744-752)
    if x < -hl2:
        x = -hl2
    elif x > hl2:
        x = hl2
    qx = ax * x + p2x
    qy = ay * x + p2y
    qz = az * x + p2z
    return _sphere_sphere_r(raw, base, p1x, p1y, p1z, m1, o1, r1,
                            qx, qy, qz, m2, o2, r2, margin)


@wp.func
def _sphere_cylinder(raw: wp.array(dtype=scalar.wp_real), base: wp.int32,
                     p1x: scalar.wp_real, p1y: scalar.wp_real, p1z: scalar.wp_real,
                     m1: wp.array(dtype=scalar.wp_real), o1: wp.int32, r1: scalar.wp_real,
                     p2x: scalar.wp_real, p2y: scalar.wp_real, p2z: scalar.wp_real,
                     m2: wp.array(dtype=scalar.wp_real), o2: wp.int32,
                     rad: scalar.wp_real, hl: scalar.wp_real, margin: scalar.wp_real) -> wp.int32:
    # mjc_SphereCylinder (engine_collision_primitive.c:318-389)
    ax = m2[o2 + 2]
    ay = m2[o2 + 5]
    az = m2[o2 + 8]
    vx = p1x - p2x
    vy = p1y - p2y
    vz = p1z - p2z
    x = ax * vx + ay * vy + az * vz
    apx = ax * x
    apy = ay * x
    apz = az * x
    ppx = vx - apx
    ppy = vy - apy
    ppz = vz - apz
    p_proj_sqr = ppx * ppx + ppy * ppy + ppz * ppz
    absx = x
    if absx < scalar.wp_real(0.0):
        absx = -absx
    collide_side = absx < hl
    collide_cap = p_proj_sqr < rad * rad
    if collide_side and collide_cap:
        dist_cap = hl - absx
        dist_radius = rad - wp.sqrt(p_proj_sqr)
        if dist_cap < dist_radius:
            collide_side = False
        else:
            collide_cap = False
    if collide_side:
        return _sphere_sphere_r(raw, base, p1x, p1y, p1z, m1, o1, r1,
                                apx + p2x, apy + p2y, apz + p2z,
                                m2, o2, rad, margin)
    if collide_cap:
        # _PlaneSphere with the cap plane, then flip the frame normal
        if x > scalar.wp_real(0.0):
            cx = p2x + ax * hl
            cy = p2y + ay * hl
            cz = p2z + az * hl
            mnx = m2[o2 + 2]
            mny = m2[o2 + 5]
            mnz = m2[o2 + 8]
        else:
            cx = p2x - ax * hl
            cy = p2y - ay * hl
            cz = p2z - az * hl
            mnx = -m2[o2 + 2]
            mny = -m2[o2 + 5]
            mnz = -m2[o2 + 8]
        # plane-sphere with plane normal (mn) at cap center, sphere = geom1
        cdist = (p1x - cx) * mnx + (p1y - cy) * mny + (p1z - cz) * mnz
        if cdist > margin + r1:
            return wp.int32(0)
        dist = cdist - r1
        scl = -dist * scalar.wp_real(0.5) - r1
        # flip frame normal because PLANE < SPHERE < CYLINDER ordering
        return _emit(raw, base, 0, dist,
                     p1x + mnx * scl, p1y + mny * scl, p1z + mnz * scl,
                     -mnx, -mny, -mnz,
                     scalar.wp_real(0.0), scalar.wp_real(0.0), scalar.wp_real(0.0))
    # corner collision: point sphere at the rim corner
    plen = wp.sqrt(p_proj_sqr)
    scale = rad / plen
    hsign = hl
    if x <= scalar.wp_real(0.0):
        hsign = -hl
    qx = ppx * scale + ax * hsign + p2x
    qy = ppy * scale + ay * hsign + p2y
    qz = ppz * scale + az * hsign + p2z
    return _sphere_sphere_r(raw, base, p1x, p1y, p1z, m1, o1, r1,
                            qx, qy, qz, m2, o2, scalar.wp_real(0.0), margin)


@wp.func
def _sphere_box(raw: wp.array(dtype=scalar.wp_real), base: wp.int32,
                p1x: scalar.wp_real, p1y: scalar.wp_real, p1z: scalar.wp_real,
                m1: wp.array(dtype=scalar.wp_real), o1: wp.int32, r1: scalar.wp_real,
                p2x: scalar.wp_real, p2y: scalar.wp_real, p2z: scalar.wp_real,
                m2: wp.array(dtype=scalar.wp_real), o2: wp.int32,
                s20: scalar.wp_real, s21: scalar.wp_real, s22: scalar.wp_real,
                margin: scalar.wp_real) -> wp.int32:
    # _SphereBox (engine_collision_box.c:39-94)
    tx = p1x - p2x
    ty = p1y - p2y
    tz = p1z - p2z
    # center = mat2^T * tmp
    cx = m2[o2] * tx + m2[o2 + 3] * ty + m2[o2 + 6] * tz
    cy = m2[o2 + 1] * tx + m2[o2 + 4] * ty + m2[o2 + 7] * tz
    cz = m2[o2 + 2] * tx + m2[o2 + 5] * ty + m2[o2 + 8] * tz
    # clamp center to box (mju_clampVec: clamp only where limit[i] > 0)
    clx = cx
    if s20 > scalar.wp_real(0.0):
        if clx > s20:
            clx = s20
        elif clx < -s20:
            clx = -s20
    cly = cy
    if s21 > scalar.wp_real(0.0):
        if cly > s21:
            cly = s21
        elif cly < -s21:
            cly = -s21
    clz = cz
    if s22 > scalar.wp_real(0.0):
        if clz > s22:
            clz = s22
        elif clz < -s22:
            clz = -s22
    nx = clx - cx
    ny = cly - cy
    nz = clz - cz
    dist = _norm3(nx, ny, nz)
    if dist - r1 > margin:
        return wp.int32(0)
    if dist <= MINVAL:
        # sphere center inside the box: nearest face
        closest = (s20 + s21 + s22) * scalar.wp_real(2.0)
        k = wp.int32(0)
        for i in range(6):
            axis = i // 2
            sign = scalar.wp_real(-1.0)
            if (i % 2) == 1:
                sign = scalar.wp_real(1.0)
            sz = s20
            if axis == 1:
                sz = s21
            elif axis == 2:
                sz = s22
            cval = cx
            if axis == 1:
                cval = cy
            elif axis == 2:
                cval = cz
            gap = sign * sz - cval
            if gap < scalar.wp_real(0.0):
                gap = -gap
            if closest > gap:
                closest = gap
                k = i
        nnx = scalar.wp_real(0.0)
        nny = scalar.wp_real(0.0)
        nnz = scalar.wp_real(0.0)
        sgn = scalar.wp_real(1.0)
        if (k % 2) == 1:
            sgn = scalar.wp_real(-1.0)
        axis = k // 2
        if axis == 0:
            nnx = sgn
        elif axis == 1:
            nny = sgn
        else:
            nnz = sgn
        # pos = center + nearest * (r1 - closest) / 2
        scl = (r1 - closest) * scalar.wp_real(0.5)
        px = cx + nnx * scl
        py = cy + nny * scl
        pz = cz + nnz * scl
        # frame = mat2 * nearest
        fx = m2[o2] * nnx + m2[o2 + 1] * nny + m2[o2 + 2] * nnz
        fy = m2[o2 + 3] * nnx + m2[o2 + 4] * nny + m2[o2 + 5] * nnz
        fz = m2[o2 + 6] * nnx + m2[o2 + 7] * nny + m2[o2 + 8] * nnz
        # world pos
        wx = m2[o2] * px + m2[o2 + 1] * py + m2[o2 + 2] * pz + p2x
        wy = m2[o2 + 3] * px + m2[o2 + 4] * py + m2[o2 + 5] * pz + p2y
        wz = m2[o2 + 6] * px + m2[o2 + 7] * py + m2[o2 + 8] * pz + p2z
        return _emit(raw, base, 0, dist - r1, wx, wy, wz, fx, fy, fz,
                     scalar.wp_real(0.0), scalar.wp_real(0.0), scalar.wp_real(0.0))
    # normalize the clamp direction
    inv = scalar.wp_real(1.0) / dist
    ux = nx * inv
    uy = ny * inv
    uz = nz * inv
    # deepest = center + u*r1? original: deepest=center, then deepest += tmp*size1[0]
    # where tmp was the normalized (clamped-center) direction
    # pos = 0.5*clamped + 0.5*deepest
    dx = cx + ux * r1
    dy = cy + uy * r1
    dz = cz + uz * r1
    px = clx * scalar.wp_real(0.5) + dx * scalar.wp_real(0.5)
    py = cly * scalar.wp_real(0.5) + dy * scalar.wp_real(0.5)
    pz = clz * scalar.wp_real(0.5) + dz * scalar.wp_real(0.5)
    fx = m2[o2] * ux + m2[o2 + 1] * uy + m2[o2 + 2] * uz
    fy = m2[o2 + 3] * ux + m2[o2 + 4] * uy + m2[o2 + 5] * uz
    fz = m2[o2 + 6] * ux + m2[o2 + 7] * uy + m2[o2 + 8] * uz
    wx = m2[o2] * px + m2[o2 + 1] * py + m2[o2 + 2] * pz + p2x
    wy = m2[o2 + 3] * px + m2[o2 + 4] * py + m2[o2 + 5] * pz + p2y
    wz = m2[o2 + 6] * px + m2[o2 + 7] * py + m2[o2 + 8] * pz + p2z
    return _emit(raw, base, 0, dist - r1, wx, wy, wz, fx, fy, fz,
                 scalar.wp_real(0.0), scalar.wp_real(0.0), scalar.wp_real(0.0))


@wp.func
def _capsule_capsule(raw: wp.array(dtype=scalar.wp_real), base: wp.int32,
                     p1x: scalar.wp_real, p1y: scalar.wp_real, p1z: scalar.wp_real,
                     m1: wp.array(dtype=scalar.wp_real), o1: wp.int32,
                     r1: scalar.wp_real, hl1: scalar.wp_real,
                     p2x: scalar.wp_real, p2y: scalar.wp_real, p2z: scalar.wp_real,
                     m2: wp.array(dtype=scalar.wp_real), o2: wp.int32,
                     r2: scalar.wp_real, hl2: scalar.wp_real, margin: scalar.wp_real) -> wp.int32:
    # mjc_CapsuleCapsule (engine_collision_primitive.c:392-485)
    a1x = m1[o1 + 2] * hl1
    a1y = m1[o1 + 5] * hl1
    a1z = m1[o1 + 8] * hl1
    a2x = m2[o2 + 2] * hl2
    a2y = m2[o2 + 5] * hl2
    a2z = m2[o2 + 8] * hl2
    dfx = p1x - p2x
    dfy = p1y - p2y
    dfz = p1z - p2z
    ma = a1x * a1x + a1y * a1y + a1z * a1z
    mb = -(a1x * a2x + a1y * a2y + a1z * a2z)
    mc = a2x * a2x + a2y * a2y + a2z * a2z
    u = -(a1x * dfx + a1y * dfy + a1z * dfz)
    v = a2x * dfx + a2y * dfy + a2z * dfz
    det = ma * mc - mb * mb
    cnt = wp.int32(0)
    if wp.abs(det) >= MINVAL:
        x1 = (mc * u - mb * v) / det
        x2 = (ma * v - mb * u) / det
        if x1 > scalar.wp_real(1.0):
            x1 = scalar.wp_real(1.0)
            x2 = (v - mb) / mc
        elif x1 < scalar.wp_real(-1.0):
            x1 = scalar.wp_real(-1.0)
            x2 = (v + mb) / mc
        if x2 > scalar.wp_real(1.0):
            x2 = scalar.wp_real(1.0)
            x1 = (u - mb) / ma
            if x1 > scalar.wp_real(1.0):
                x1 = scalar.wp_real(1.0)
            elif x1 < scalar.wp_real(-1.0):
                x1 = scalar.wp_real(-1.0)
        elif x2 < scalar.wp_real(-1.0):
            x2 = scalar.wp_real(-1.0)
            x1 = (u + mb) / ma
            if x1 > scalar.wp_real(1.0):
                x1 = scalar.wp_real(1.0)
            elif x1 < scalar.wp_real(-1.0):
                x1 = scalar.wp_real(-1.0)
        return _sphere_sphere_r(raw, base,
                                p1x + a1x * x1, p1y + a1y * x1, p1z + a1z * x1,
                                m1, o1, r1,
                                p2x + a2x * x2, p2y + a2y * x2, p2z + a2z * x2,
                                m2, o2, r2, margin)
    # parallel axes: four candidate endpoint tests
    x2 = (v - mb) / mc
    if x2 > scalar.wp_real(1.0):
        x2 = scalar.wp_real(1.0)
    elif x2 < scalar.wp_real(-1.0):
        x2 = scalar.wp_real(-1.0)
    n1 = _sphere_sphere_r(raw, base,
                          p1x + a1x, p1y + a1y, p1z + a1z, m1, o1, r1,
                          p2x + a2x * x2, p2y + a2y * x2, p2z + a2z * x2,
                          m2, o2, r2, margin)
    x2 = (v + mb) / mc
    if x2 > scalar.wp_real(1.0):
        x2 = scalar.wp_real(1.0)
    elif x2 < scalar.wp_real(-1.0):
        x2 = scalar.wp_real(-1.0)
    n2 = wp.int32(0)
    if n1 + n2 < 2:
        n2 = _sphere_sphere_r(raw, base + n1 * RAW_F,
                              p1x - a1x, p1y - a1y, p1z - a1z, m1, o1, r1,
                              p2x + a2x * x2, p2y + a2y * x2, p2z + a2z * x2,
                              m2, o2, r2, margin)
    if n1 + n2 >= 2:
        return n1 + n2
    x1 = (u - mb) / ma
    if x1 > scalar.wp_real(1.0):
        x1 = scalar.wp_real(1.0)
    elif x1 < scalar.wp_real(-1.0):
        x1 = scalar.wp_real(-1.0)
    n3 = wp.int32(0)
    if n1 + n2 < 2:
        n3 = _sphere_sphere_r(raw, base + (n1 + n2) * RAW_F,
                              p1x + a1x * x1, p1y + a1y * x1, p1z + a1z * x1,
                              m1, o1, r1,
                              p2x + a2x, p2y + a2y, p2z + a2z, m2, o2, r2, margin)
    if n1 + n2 + n3 >= 2:
        return n1 + n2 + n3
    x1 = (u + mb) / ma
    if x1 > scalar.wp_real(1.0):
        x1 = scalar.wp_real(1.0)
    elif x1 < scalar.wp_real(-1.0):
        x1 = scalar.wp_real(-1.0)
    n4 = wp.int32(0)
    if n1 + n2 + n3 < 2:
        n4 = _sphere_sphere_r(raw, base + (n1 + n2 + n3) * RAW_F,
                              p1x + a1x * x1, p1y + a1y * x1, p1z + a1z * x1,
                              m1, o1, r1,
                              p2x - a2x, p2y - a2y, p2z - a2z, m2, o2, r2, margin)
    return n1 + n2 + n3 + n4


@wp.func
def _mesh_support(mesh_vert: wp.array(dtype=wp.float32),
                  mesh_vertadr: wp.array(dtype=wp.int32),
                  mesh_vertnum: wp.array(dtype=wp.int32),
                  mesh_graphadr: wp.array(dtype=wp.int32),
                  mesh_graph: wp.array(dtype=wp.int32),
                  mid: wp.int32,
                  dx: scalar.wp_real, dy: scalar.wp_real, dz: scalar.wp_real,
                  scratch: wp.array(dtype=wp.int32), so: wp.int32) -> wp.int32:
    """Support vertex local index of mesh mid in local direction (d).

    Exhaustive search when the hull graph is absent, hill-climb otherwise
    (mjccd_support, engine_collision_convex.c:42-135). Returns -1 on failure.
    """
    adr = mesh_vertadr[mid]
    num = mesh_vertnum[mid]
    ga = mesh_graphadr[mid]
    best = wp.int32(-1)
    tmp = scalar.wp_real(-1.0e10)
    if ga < 0:
        for ii in range(num):
            vi = 3 * (adr + ii)
            vdot = dx * scalar.wp_real(mesh_vert[vi]) + dy * scalar.wp_real(mesh_vert[vi + 1]) + dz * scalar.wp_real(mesh_vert[vi + 2])
            if vdot > tmp:
                tmp = vdot
                best = ii
        scratch[so] = best
        return best
    nv = mesh_graph[ga]
    cur = wp.int32(0)
    g0 = mesh_graph[ga + 2 + nv]
    vi = 3 * (adr + g0)
    tmp = dx * scalar.wp_real(mesh_vert[vi]) + dy * scalar.wp_real(mesh_vert[vi + 1]) + dz * scalar.wp_real(mesh_vert[vi + 2])
    best = wp.int32(0)
    change = wp.int32(1)
    steps = wp.int32(0)
    while change != 0:
        steps += 1
        if steps > num + 1:
            return wp.int32(-2)  # watchdog: explicit error, never a silent shrink
        change = wp.int32(0)
        ei = mesh_graph[ga + 2 + cur]
        loc = mesh_graph[ga + 2 + 2 * nv + ei]
        while loc >= 0:
            gi = mesh_graph[ga + 2 + nv + loc]
            vi2 = 3 * (adr + gi)
            vdot = dx * scalar.wp_real(mesh_vert[vi2]) + dy * scalar.wp_real(mesh_vert[vi2 + 1]) + dz * scalar.wp_real(mesh_vert[vi2 + 2])
            if vdot > tmp:
                tmp = vdot
                cur = loc
                change = wp.int32(1)
            ei += 1
            loc = mesh_graph[ga + 2 + 2 * nv + ei]
    scratch[so] = cur
    return mesh_graph[ga + 2 + nv + cur]


@wp.func
def _plane_convex(raw: wp.array(dtype=scalar.wp_real), base: wp.int32,
                  p1x: scalar.wp_real, p1y: scalar.wp_real, p1z: scalar.wp_real,
                  m1: wp.array(dtype=scalar.wp_real), o1: wp.int32,
                  p2x: scalar.wp_real, p2y: scalar.wp_real, p2z: scalar.wp_real,
                  m2: wp.array(dtype=scalar.wp_real), o2: wp.int32,
                  s20: scalar.wp_real, s21: scalar.wp_real, s22: scalar.wp_real,
                  t2: wp.int32, d2: wp.int32,
                  mesh_vert: wp.array(dtype=wp.float32),
                  mesh_vertadr: wp.array(dtype=wp.int32),
                  mesh_vertnum: wp.array(dtype=wp.int32),
                  mesh_graphadr: wp.array(dtype=wp.int32),
                  mesh_graph: wp.array(dtype=wp.int32),
                  rbound2: scalar.wp_real,
                  scratch: wp.array(dtype=wp.int32), so: wp.int32,
                  margin: scalar.wp_real) -> wp.int32:
    """mjc_PlaneConvex: support contact plus up to 3 extra mesh vertices.

    Ellipsoid (dataid < 0) returns exactly the support contact. Mesh extra
    vertices follow the frozen threshold/separation rules
    (maxplanemesh=3, tolplanemesh=0.3).
    """
    nx = m1[o1 + 2]
    ny = m1[o1 + 5]
    nz = m1[o1 + 8]
    # support point in -normal direction, geom2 local frame
    ldx = -(m2[o2] * nx + m2[o2 + 3] * ny + m2[o2 + 6] * nz)
    ldy = -(m2[o2 + 1] * nx + m2[o2 + 4] * ny + m2[o2 + 7] * nz)
    ldz = -(m2[o2 + 2] * nx + m2[o2 + 5] * ny + m2[o2 + 8] * nz)
    sx = scalar.wp_real(0.0)
    sy = scalar.wp_real(0.0)
    sz = scalar.wp_real(0.0)
    best_local = wp.int32(-1)
    if t2 == 4:
        # ellipsoid support: res_i = dir_i*size_i; mju_normalize3(res) (with
        # (1,0,0) fallback); res_i *= size_i (engine_collision_convex.c:76-85)
        rx = ldx * s20
        ry = ldy * s21
        rz = ldz * s22
        nlen = _norm3(rx, ry, rz)
        if nlen >= MINVAL:
            inv = scalar.wp_real(1.0) / nlen
            rx *= inv
            ry *= inv
            rz *= inv
        else:
            # mju_normalize3 fallback: (1,0,0), then res_i *= size_i
            rx = scalar.wp_real(1.0)
            ry = scalar.wp_real(0.0)
            rz = scalar.wp_real(0.0)
        sx = rx * s20
        sy = ry * s21
        sz = rz * s22
    elif t2 == 7:
        if d2 < 0:
            # R01-31: original warns and uses a zero support point
            best_local = wp.int32(-1)
        else:
            best = _mesh_support(mesh_vert, mesh_vertadr, mesh_vertnum,
                                 mesh_graphadr, mesh_graph, d2,
                                 ldx, ldy, ldz, scratch, so)
            if best == -2:
                return wp.int32(-6)  # watchdog: explicit error
            if best < 0:
                # R01-31: original warns and uses a zero support point
                best_local = wp.int32(-1)
            else:
                # R01-27: best is the GLOBAL vertex id; the hull-local index
                # for vert_edgeadr access is the one _mesh_support stored.
                best_local = scratch[so]
                vi = 3 * (mesh_vertadr[d2] + best)
                sx = scalar.wp_real(mesh_vert[vi])
                sy = scalar.wp_real(mesh_vert[vi + 1])
                sz = scalar.wp_real(mesh_vert[vi + 2])
    elif t2 == 6:
        # box support in local direction
        sx = _sign(ldx) * s20
        sy = _sign(ldy) * s21
        sz = _sign(ldz) * s22
    elif t2 == 5:
        # cylinder support
        tmp = wp.sqrt(ldx * ldx + ldy * ldy)
        if tmp > MINVAL:
            sx = ldx / tmp * s20
            sy = ldy / tmp * s20
        sz = _sign(ldz) * s21
    else:
        return wp.int32(-4)
    # world support point
    wx = m2[o2] * sx + m2[o2 + 1] * sy + m2[o2 + 2] * sz + p2x
    wy = m2[o2 + 3] * sx + m2[o2 + 4] * sy + m2[o2 + 5] * sz + p2y
    wz = m2[o2 + 6] * sx + m2[o2 + 7] * sy + m2[o2 + 8] * sz + p2z
    difx = wx - p1x
    dify = wy - p1y
    difz = wz - p1z
    dist = difx * nx + dify * ny + difz * nz
    if dist > margin:
        return wp.int32(0)
    cnt = _emit(raw, base, 0, dist,
                wx - nx * dist * scalar.wp_real(0.5),
                wy - ny * dist * scalar.wp_real(0.5),
                wz - nz * dist * scalar.wp_real(0.5),
                nx, ny, nz, scalar.wp_real(0.0), scalar.wp_real(0.0), scalar.wp_real(0.0))
    # extra mesh vertices below the margin (maxplanemesh = 3)
    if t2 != 7 or d2 < 0:
        return cnt
    maxplanemesh = 3
    tolplanemesh = scalar.wp_real(0.3)
    # threshold along locdir relative to geom2 center
    threshold = (p2x - p1x) * nx + (p2y - p1y) * ny + (p2z - p1z) * nz - margin
    ga = mesh_graphadr[d2]
    firstx = raw[base + 1]
    firsty = raw[base + 2]
    firstz = raw[base + 3]
    if best_local < 0:
        return cnt
    nvert = mesh_vertnum[d2]
    adr = mesh_vertadr[d2]
    if ga < 0:
        for i in range(nvert):
            # original: for (...; count < maxplanemesh; i++) with count
            # starting at 1, so at most 2 extra contacts (3 total)
            if cnt >= maxplanemesh:
                break
            if i == best_local:
                continue
            vi = 3 * (adr + i)
            vdot = ldx * scalar.wp_real(mesh_vert[vi]) + ldy * scalar.wp_real(mesh_vert[vi + 1]) + ldz * scalar.wp_real(mesh_vert[vi + 2])
            if vdot > threshold:
                # addplanemesh: world point, skip if too close to first contact
                px = m2[o2] * scalar.wp_real(mesh_vert[vi]) + m2[o2 + 1] * scalar.wp_real(mesh_vert[vi + 1]) + m2[o2 + 2] * scalar.wp_real(mesh_vert[vi + 2]) + p2x
                py = m2[o2 + 3] * scalar.wp_real(mesh_vert[vi]) + m2[o2 + 4] * scalar.wp_real(mesh_vert[vi + 1]) + m2[o2 + 5] * scalar.wp_real(mesh_vert[vi + 2]) + p2y
                pz = m2[o2 + 6] * scalar.wp_real(mesh_vert[vi]) + m2[o2 + 7] * scalar.wp_real(mesh_vert[vi + 1]) + m2[o2 + 8] * scalar.wp_real(mesh_vert[vi + 2]) + p2z
                ddx = px - firstx
                ddy = py - firsty
                ddz = pz - firstz
                if wp.sqrt(ddx * ddx + ddy * ddy + ddz * ddz) < tolplanemesh * rbound2:
                    continue
                d2i = (px - p1x) * nx + (py - p1y) * ny + (pz - p1z) * nz
                cnt += _emit(raw, base, cnt, d2i,
                             px - nx * d2i * scalar.wp_real(0.5),
                             py - ny * d2i * scalar.wp_real(0.5),
                             pz - nz * d2i * scalar.wp_real(0.5),
                             nx, ny, nz, scalar.wp_real(0.0), scalar.wp_real(0.0), scalar.wp_real(0.0))
        return cnt
    # graph neighborhood of the best vertex (local hull id, R01-27)
    nv = mesh_graph[ga]
    ei = mesh_graph[ga + 2 + best_local]
    loc = mesh_graph[ga + 2 + 2 * nv + ei]
    while loc >= 0 and cnt < maxplanemesh:
        gi = mesh_graph[ga + 2 + nv + loc]
        vi = 3 * (adr + gi)
        vdot = ldx * scalar.wp_real(mesh_vert[vi]) + ldy * scalar.wp_real(mesh_vert[vi + 1]) + ldz * scalar.wp_real(mesh_vert[vi + 2])
        if vdot > threshold:
            px = m2[o2] * scalar.wp_real(mesh_vert[vi]) + m2[o2 + 1] * scalar.wp_real(mesh_vert[vi + 1]) + m2[o2 + 2] * scalar.wp_real(mesh_vert[vi + 2]) + p2x
            py = m2[o2 + 3] * scalar.wp_real(mesh_vert[vi]) + m2[o2 + 4] * scalar.wp_real(mesh_vert[vi + 1]) + m2[o2 + 5] * scalar.wp_real(mesh_vert[vi + 2]) + p2y
            pz = m2[o2 + 6] * scalar.wp_real(mesh_vert[vi]) + m2[o2 + 7] * scalar.wp_real(mesh_vert[vi + 1]) + m2[o2 + 8] * scalar.wp_real(mesh_vert[vi + 2]) + p2z
            ddx = px - firstx
            ddy = py - firsty
            ddz = pz - firstz
            if wp.sqrt(ddx * ddx + ddy * ddy + ddz * ddz) >= tolplanemesh * rbound2:
                d2i = (px - p1x) * nx + (py - p1y) * ny + (pz - p1z) * nz
                cnt += _emit(raw, base, cnt, d2i,
                             px - nx * d2i * scalar.wp_real(0.5),
                             py - ny * d2i * scalar.wp_real(0.5),
                             pz - nz * d2i * scalar.wp_real(0.5),
                             nx, ny, nz, scalar.wp_real(0.0), scalar.wp_real(0.0), scalar.wp_real(0.0))
        ei += 1
        loc = mesh_graph[ga + 2 + 2 * nv + ei]
    return cnt


@wp.kernel(enable_backward=False)
def _detect_kernel(
        call_g1: wp.array(dtype=wp.int32), call_g2: wp.array(dtype=wp.int32),
        call_type1: wp.array(dtype=wp.int32), call_type2: wp.array(dtype=wp.int32),
        call_margin: wp.array(dtype=scalar.wp_real), call_count: wp.array(dtype=wp.int32),
        active: wp.array(dtype=wp.int32),
        xpos: wp.array(dtype=scalar.wp_real), xmat: wp.array(dtype=scalar.wp_real),
        geom_size: wp.array(dtype=scalar.wp_real), geom_dataid: wp.array(dtype=wp.int32),
        geom_rbound: wp.array(dtype=scalar.wp_real),
        mesh_vert: wp.array(dtype=wp.float32), mesh_vertadr: wp.array(dtype=wp.int32),
        mesh_vertnum: wp.array(dtype=wp.int32), mesh_graphadr: wp.array(dtype=wp.int32),
        mesh_graph: wp.array(dtype=wp.int32),
        route_id: wp.array(dtype=wp.int32),
        raw: wp.array(dtype=scalar.wp_real), raw_count: wp.array(dtype=wp.int32),
        det_status: wp.array(dtype=wp.int32), scratch_i: wp.array(dtype=wp.int32),
        scratch_f: wp.array(dtype=scalar.wp_real),
        native_raw: wp.array(dtype=scalar.wp_real), native_count: wp.array(dtype=wp.int32),
        native_status: wp.array(dtype=wp.int32), native_frame0: wp.array(dtype=scalar.wp_real),
        native_framewritten: wp.array(dtype=wp.int32),
        native_diag: wp.array(dtype=wp.int32), raw_bits: wp.array(dtype=scalar.wp_bits),
        native_raw_bits: wp.array(dtype=scalar.wp_bits), native_frame0_bits: wp.array(dtype=scalar.wp_bits),
        multiccd: int,
        fe_stage: wp.array(dtype=wp.int32), fe_code: wp.array(dtype=wp.int32),
        fe_item: wp.array(dtype=wp.int32),
        ngeom: int, pairs: int):
    """One thread per world; calls processed in source call order so the
    first_error keeps original first-check order. All raw rows and per-slot
    counts/statuses for active worlds are reset. Unwritten raw rows are
    preserved, and only this launch's counts can reach finalize (R01-21)."""
    w = wp.tid()
    base_calls = w * pairs
    if fe_code[w] != 0:
        return
    if active[w] == 0:
        return
    for j in range(pairs):
        slot = base_calls + j
        det_status[slot] = 0
        raw_count[slot] = 0
        # Raw fields not written by this launch retain their exact prior bytes.
    if active[w] == 0:
        return
    if active[w] != 1:
        det_status[base_calls] = 1
        fe_stage[w] = 6
        fe_code[w] = 1
        fe_item[w] = w
        return
    nc = call_count[w]
    if nc < 0 or nc > pairs:
        det_status[base_calls] = 2
        fe_stage[w] = 6
        fe_code[w] = 2
        fe_item[w] = 0
        return
    for j in range(nc):
        slot = base_calls + j
        g1 = call_g1[slot]
        g2 = call_g2[slot]
        t1 = call_type1[slot]
        t2 = call_type2[slot]
        rbase = slot * MAX_RAW * RAW_F
        if g1 < 0 or g1 >= ngeom or g2 < 0 or g2 >= ngeom or t1 < 0 or t1 > 7 or t2 < 0 or t2 > 7:
            det_status[slot] = 8
            fe_stage[w] = 6
            fe_code[w] = 8
            fe_item[w] = j
            return
        det = route_id[t1 * 8 + t2]
        if det == DET_NONE:
            det_status[slot] = 3
            fe_stage[w] = 6
            fe_code[w] = 8
            fe_item[w] = j
            return
        if det == DET_UNSUPPORTED:
            det_status[slot] = 4
            fe_stage[w] = 6
            fe_code[w] = 4  # ERR_UNSUPPORTED_ROUTE
            fe_item[w] = j
            return
        o1 = g1 * 9
        o2 = g2 * 9
        p1x = xpos[w * ngeom * 3 + g1 * 3]
        p1y = xpos[w * ngeom * 3 + g1 * 3 + 1]
        p1z = xpos[w * ngeom * 3 + g1 * 3 + 2]
        p2x = xpos[w * ngeom * 3 + g2 * 3]
        p2y = xpos[w * ngeom * 3 + g2 * 3 + 1]
        p2z = xpos[w * ngeom * 3 + g2 * 3 + 2]
        m1 = w * ngeom * 9
        m2 = w * ngeom * 9
        s10 = geom_size[g1 * 3]
        s11 = geom_size[g1 * 3 + 1]
        s12 = geom_size[g1 * 3 + 2]
        s20 = geom_size[g2 * 3]
        s21 = geom_size[g2 * 3 + 1]
        s22 = geom_size[g2 * 3 + 2]
        margin = call_margin[slot]
        cnt = wp.int32(0)
        if det == DET_PLANE_SPHERE:
            cnt = _plane_sphere(raw, rbase, p1x, p1y, p1z, xmat, m1 + o1,
                                p2x, p2y, p2z, s20, margin)
        elif det == DET_PLANE_CAPSULE:
            cnt = _plane_capsule(raw, rbase, p1x, p1y, p1z, xmat, m1 + o1,
                                 p2x, p2y, p2z, xmat, m2 + o2,
                                 s20, s21, margin)
        elif det == DET_PLANE_CYLINDER:
            cnt = _plane_cylinder(raw, rbase, p1x, p1y, p1z, xmat, m1 + o1,
                                  p2x, p2y, p2z, xmat, m2 + o2,
                                  s20, s21, margin)
        elif det == DET_PLANE_BOX:
            cnt = _plane_box(raw, rbase, p1x, p1y, p1z, xmat, m1 + o1,
                             p2x, p2y, p2z, xmat, m2 + o2,
                             s20, s21, s22, margin)
        elif det == DET_PLANE_CONVEX:
            cnt = _plane_convex(raw, rbase, p1x, p1y, p1z, xmat, m1 + o1,
                                p2x, p2y, p2z, xmat, m2 + o2,
                                s20, s21, s22, t2, geom_dataid[g2],
                                mesh_vert, mesh_vertadr, mesh_vertnum,
                                mesh_graphadr, mesh_graph, geom_rbound[g2],
                                scratch_i, slot, margin)
        elif det == DET_SPHERE_SPHERE:
            cnt = _sphere_sphere_r(raw, rbase, p1x, p1y, p1z, xmat, m1 + o1, s10,
                                   p2x, p2y, p2z, xmat, m2 + o2, s20, margin)
        elif det == DET_SPHERE_CAPSULE:
            cnt = _sphere_capsule(raw, rbase, p1x, p1y, p1z, xmat, m1 + o1, s10,
                                  p2x, p2y, p2z, xmat, m2 + o2, s20, s21, margin)
        elif det == DET_SPHERE_CYLINDER:
            cnt = _sphere_cylinder(raw, rbase, p1x, p1y, p1z, xmat, m1 + o1, s10,
                                   p2x, p2y, p2z, xmat, m2 + o2, s20, s21, margin)
        elif det == DET_SPHERE_BOX:
            cnt = _sphere_box(raw, rbase, p1x, p1y, p1z, xmat, m1 + o1, s10,
                              p2x, p2y, p2z, xmat, m2 + o2,
                              s20, s21, s22, margin)
        elif det == DET_CAPSULE_CAPSULE:
            cnt = _capsule_capsule(raw, rbase, p1x, p1y, p1z, xmat, m1 + o1,
                                   s10, s11, p2x, p2y, p2z, xmat, m2 + o2,
                                   s20, s21, margin)
        elif det == DET_CONVEX_MPR or det == DET_BOXBOX_SAT or det == DET_CAPSULE_BOX_NATIVE:
            ns = native_status[slot]
            if ns != 0:
                det_status[slot] = 100 + ns
                fe_stage[w] = 6
                fe_code[w] = 9 if ns == 6 else 8
                fe_item[w] = j
                return
            cnt = native_count[slot]
            if cnt < 0 or cnt > MAX_RAW:
                det_status[slot] = 8
                fe_stage[w] = 6
                fe_code[w] = 2
                fe_item[w] = j
                return
            # Source control domains are checked before consuming native data.
            # The integer views alias the actual buffers and preserve NaN payload
            # identity when checking the unexported raw tail. Finite held values
            # accept numerical equality, including both zero signs.
            flag = native_framewritten[slot]
            db = slot * 16
            invalid = int(0)
            if (flag != 0 and flag != 1) or native_diag[db + 4] != cnt or native_diag[db + 5] != flag:
                invalid = 1
            for k in range(7,16):
                if native_diag[db + k] != 0:
                    invalid = 1
            if det == DET_BOXBOX_SAT:
                for k in range(4):
                    if native_diag[db + k] != 0:
                        invalid = 1
                branch = native_diag[db + 6]
                # Original BoxBox face1..11 reaches both con[0].frame stores
                # before return n, even at n=0. Reset code0 also describes
                # earlier SAT returns; edges12..20 may return before stores.
                if branch < -1 or branch > 20 or (flag == 1 and branch < 0) or (cnt > 0 and flag != 1) or (branch >= 1 and branch <= 11 and flag != 1):
                    invalid = 1
            elif det == DET_CAPSULE_BOX_NATIVE:
                # Original CapsuleBox appends at most two SphereBox contacts.
                # It uses no MPR or BoxBox SAT diagnostics. A frame store
                # exists exactly when at least one contact was appended.
                if t1 != 3 or t2 != 6 or cnt > 2 or native_diag[db + 6] != 0:
                    invalid = 1
                if (cnt == 0 and flag != 0) or (cnt > 0 and flag != 1):
                    invalid = 1
                for k in range(4):
                    if native_diag[db + k] != 0:
                        invalid = 1
            else:
                if flag != 0 or native_diag[db + 6] != 0 or cnt > (5 if multiccd != 0 else 1):
                    invalid = 1
                total = int(0)
                for k in range(3):
                    phase = native_diag[db + k]
                    if phase < 0 or phase > 4096:
                        invalid = 1
                    else:
                        total += phase
                attempts = 5 if multiccd != 0 and cnt > 0 else 1
                if native_diag[db + 3] < 1 or native_diag[db + 3] > 2 * attempts + total:
                    invalid = 1
            for k in range(cnt * RAW_F):
                if not wp.isfinite(native_raw[rbase + k]):
                    invalid = 1
            for k in range(cnt * RAW_F,MAX_RAW * RAW_F):
                if native_raw_bits[rbase + k] != raw_bits[rbase + k] and native_raw[rbase + k] != raw[rbase + k]:
                    invalid = 1
            if flag == 1:
                for k in range(6):
                    if not wp.isfinite(native_frame0[slot * 6 + k]):
                        invalid = 1
                    if cnt > 0 and native_frame0[slot * 6 + k] != native_raw[rbase + 4 + k]:
                        invalid = 1
                    # BoxBox source mju_zero3(frame + 3) has the singleton
                    # numeric domain {0}. Both FP64 zero signs represent it.
                    # The preceding copy and held-float relations accept
                    # numeric equality. Actual forbidden stores and physical
                    # alias identity require independent evidence.
                    if k >= 3 and native_frame0[slot * 6 + k] != scalar.wp_real(0.0):
                        invalid = 1
            if invalid != 0:
                det_status[slot] = 8
                fe_stage[w] = 6
                fe_code[w] = 8
                fe_item[w] = j
                return
            for k in range(cnt * RAW_F):
                raw[rbase + k] = native_raw[rbase + k]
            if cnt == 0 and flag == 1:
                for k in range(6):
                    raw[rbase + 4 + k] = native_frame0[slot * 6 + k]
        if cnt < 0:
            # internal/watchdog/binding errors from detector bodies
            det_status[slot] = 10 - cnt
            fe_stage[w] = 6
            fe_code[w] = 8
            fe_item[w] = j
            return
        raw_count[slot] = cnt


from .mesh_descriptor import validate_mesh_static, build_mesh_descriptor


class DetectorWorkspace:
    """Device-resident detector stage: dispatch calls in, raw contacts out.

    One thread per world keeps original call order for first_error. The
    unsupported gate writes no contacts and marks ERR_UNSUPPORTED_ROUTE.
    """

    def __init__(self, nworld: int, pairs: int, static: dict, *, device='cuda:0', prepare_baseline=True):
        if min(nworld, pairs) < 1:
            raise ValueError('N and pairs must be positive')
        self.nworld = nworld
        self.pairs = pairs
        self.device = device
        self.ngeom = len(static['geom_size'])
        self.mesh_descriptor = build_mesh_descriptor(static)

        def exact(name, dtype, shape):
            a = np.asarray(static[name])
            if a.dtype != dtype or a.shape != shape:
                raise ValueError(f'{name} expected {shape} {dtype}, got {a.shape} {a.dtype}')
            return np.ascontiguousarray(a).ravel()

        self.geom_size = wp.array(exact('geom_size', np.float64, (self.ngeom, 3)),
                                 dtype=scalar.wp_real, device=device)
        self.geom_dataid = wp.array(exact('geom_dataid', np.int32, (self.ngeom,)),
                                   dtype=wp.int32, device=device)
        self.geom_rbound = wp.array(exact('geom_rbound', np.float64, (self.ngeom,)),
                                   dtype=scalar.wp_real, device=device)
        nvert = len(static['mesh_vert'])
        ngraph = len(static['mesh_graph'])
        nmesh = len(static['mesh_vertadr'])
        self.mesh_vert = wp.array(np.ascontiguousarray(static['mesh_vert'], dtype=np.float32),
                                 dtype=wp.float32, device=device)
        self.mesh_vertadr = wp.array(exact('mesh_vertadr', np.int32, (nmesh,)),
                                    dtype=wp.int32, device=device)
        self.mesh_vertnum = wp.array(exact('mesh_vertnum', np.int32, (nmesh,)),
                                    dtype=wp.int32, device=device)
        self.mesh_graphadr = wp.array(exact('mesh_graphadr', np.int32, (nmesh,)),
                                     dtype=wp.int32, device=device)
        self.mesh_graph = wp.array(np.ascontiguousarray(static['mesh_graph'], dtype=np.int32),
                                  dtype=wp.int32, device=device)
        if nvert == 0:
            self.mesh_vert = wp.zeros(1, dtype=wp.float32, device=device)
        if ngraph == 0:
            self.mesh_graph = wp.zeros(1, dtype=wp.int32, device=device)
        self.mesh_edge_end = wp.array(self.mesh_descriptor['edge_ends'] if ngraph else np.full(1,-1,dtype=np.int32), dtype=wp.int32, device=device)
        self._mesh_descriptor_owners=(self.mesh_vertadr,self.mesh_vertnum,self.mesh_graphadr,self.mesh_graph,self.mesh_edge_end)
        self.nvert, self.ngraph, self.nmesh = nvert, ngraph, nmesh
        self.mpr_iterations = int(static['mpr_iterations'])
        self.mpr_tolerance = float(static['mpr_tolerance'])
        self.multiccd = int(static['multiccd'])
        if not isinstance(prepare_baseline,bool):raise ValueError('prepare_baseline must be bool')
        self.native = prepare_native(device) if prepare_baseline else None
        self.route_id = wp.array(build_route_ids().ravel(), dtype=wp.int32, device=device)
        n = nworld * pairs
        self.call_g1 = wp.zeros(n, dtype=wp.int32, device=device)
        self.call_g2 = wp.zeros(n, dtype=wp.int32, device=device)
        self.call_type1 = wp.zeros(n, dtype=wp.int32, device=device)
        self.call_type2 = wp.zeros(n, dtype=wp.int32, device=device)
        self.call_margin = wp.zeros(n, dtype=scalar.wp_real, device=device)
        self.call_count = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.active = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.xpos = wp.zeros(nworld * self.ngeom * 3, dtype=scalar.wp_real, device=device)
        self.xmat = wp.zeros(nworld * self.ngeom * 9, dtype=scalar.wp_real, device=device)
        self.raw = wp.zeros(n * MAX_RAW * RAW_F, dtype=scalar.wp_real, device=device)
        self.raw_count = wp.zeros(n, dtype=wp.int32, device=device)
        self.det_status = wp.zeros(n, dtype=wp.int32, device=device)
        self.native_raw = wp.zeros(n * MAX_RAW * RAW_F, dtype=scalar.wp_real, device=device)
        self.native_count = wp.zeros(n, dtype=wp.int32, device=device)
        self.native_status = wp.zeros(n, dtype=wp.int32, device=device)
        self.native_diag = wp.zeros(n * 16, dtype=wp.int32, device=device)
        self.native_frame0 = wp.zeros(n * 6, dtype=scalar.wp_real, device=device)
        self.native_framewritten = wp.zeros(n, dtype=wp.int32, device=device)
        self.raw_bits = self.raw.view(scalar.wp_bits)
        self.native_raw_bits = self.native_raw.view(scalar.wp_bits)
        self.native_frame0_bits = self.native_frame0.view(scalar.wp_bits)
        self.scratch_i = wp.zeros(n, dtype=wp.int32, device=device)
        self.scratch_f = wp.zeros(n * 64, dtype=scalar.wp_real, device=device)
        self.fe_stage = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.fe_code = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.fe_item = wp.zeros(nworld, dtype=wp.int32, device=device)

    def upload(self, xpos, xmat, active):
        specs = ((xpos, (self.nworld, self.ngeom, 3), np.float64, self.xpos),
                 (xmat, (self.nworld, self.ngeom, 9), np.float64, self.xmat),
                 (active, (self.nworld,), np.int32, self.active))
        for value, shape, dtype, dst in specs:
            a = np.asarray(value)
            if a.shape != shape or a.dtype != dtype:
                raise ValueError(f'detector input {a.shape}/{a.dtype} != {shape}/{dtype}')
            if dtype == np.float64 and (not np.all(np.isfinite(a)) or np.any(np.abs(a) > 1e100)):
                raise ValueError('unsupported_numeric_range')
            dst.assign(np.ascontiguousarray(a).ravel())

    def upload_calls(self, call_g1, call_g2, call_type1, call_type2,
                     call_margin, call_count):
        n = self.nworld * self.pairs
        specs = ((call_g1, (self.nworld, self.pairs), np.int32, self.call_g1),
                 (call_g2, (self.nworld, self.pairs), np.int32, self.call_g2),
                 (call_type1, (self.nworld, self.pairs), np.int32, self.call_type1),
                 (call_type2, (self.nworld, self.pairs), np.int32, self.call_type2),
                 (call_margin, (self.nworld, self.pairs), np.float64, self.call_margin),
                 (call_count, (self.nworld,), np.int32, self.call_count))
        for value, shape, dtype, dst in specs:
            a = np.asarray(value)
            if a.shape != shape or a.dtype != dtype:
                raise ValueError('call array schema')
            dst.assign(np.ascontiguousarray(a).ravel())

    def prepare_native_baseline(self):
        """Explicit cold preparation, before capture/launch; never compile in launch."""
        if self.native is None:self.native=prepare_native(self.device)
        return self.native

    def launch(self, stream=None):
        if self.native is None:raise RuntimeError('native baseline not prepared; call prepare_native_baseline before capture')
        stream = stream or wp.get_stream(self.device)
        wp.copy(self.native_raw, self.raw, stream=stream)
        launch_native(self, stream=stream)
        wp.launch(_detect_kernel, dim=self.nworld, inputs=[
            self.call_g1, self.call_g2, self.call_type1, self.call_type2,
            self.call_margin, self.call_count, self.active,
            self.xpos, self.xmat,
            self.geom_size, self.geom_dataid, self.geom_rbound,
            self.mesh_vert, self.mesh_vertadr, self.mesh_vertnum,
            self.mesh_graphadr, self.mesh_graph,
            self.route_id,
            self.raw, self.raw_count, self.det_status, self.scratch_i,
            self.scratch_f,
            self.native_raw, self.native_count, self.native_status,
            self.native_frame0, self.native_framewritten,
            self.native_diag,self.raw_bits,self.native_raw_bits,self.native_frame0_bits,self.multiccd,
            self.fe_stage, self.fe_code, self.fe_item,
            self.ngeom, self.pairs],
            device=self.device, stream=stream)

    def download(self):
        n, p = self.nworld, self.pairs
        return {
            'raw': self.raw.numpy().reshape(n, p, MAX_RAW, RAW_F),
            'raw_count': self.raw_count.numpy().reshape(n, p),
            'det_status': self.det_status.numpy().reshape(n, p),
            'fe_stage': self.fe_stage.numpy(),
            'fe_code': self.fe_code.numpy(),
            'fe_item': self.fe_item.numpy(),
        }
