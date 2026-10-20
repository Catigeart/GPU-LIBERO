"""Warp port of the MuJoCo 2.3.7 MPR convex-pair detector (mjc_Convex).

SOURCE BINDING (frozen):
  Algorithm order and numerical association: the accepted C07 CUDA port
      libero/libero/gpu/reference/collision/mesh_convex/kernel.cu (305 lines)
    which is itself pinned to MuJoCo 2.3.7
      engine_collision_convex.c:32-239   mjccd_support + mjc_MPRIteration
      engine_collision_convex.c:278-364  mjc_Convex body
    and pinned libccd 7931e764 src/mpr.c, src/support.c, src/vec3.c.
  Stages map to kernel.cu as:
      discover          (initial portal)        kernel.cu:127-162
      refine            (portal nesting test)    kernel.cu:163-175
      find_penetration  (portal refinement)      kernel.cu:203-252
      segment_dist2 / triangle_dist2            kernel.cu:177-202
        (pinned libccd vec3.c:137-216, branch tests and strict tie order kept)
      contact generation                        kernel.cu:276-281
  Support functions follow mjccd_support (engine_collision_convex.c:42-207)
  generalized from the C07 mesh support to the convex subset:
      cylinder arm  engine_collision_convex.c:87-99
      box arm       engine_collision_convex.c:101-105
      mesh arm      kernel.cu mesh_support:53-91 (graph hill-climbing with a
                    watchdog, exhaustive vertex scan when mesh_graphadr < 0)

COVERED ROUTES (all mjc_Convex in the frozen mjCOLLISIONFUNC table):
  CYLINDER-CYLINDER (5,5), CYLINDER-BOX (5,6), CYLINDER-MESH (5,7),
  BOX-MESH (6,7), MESH-MESH (7,7) -- geom type order is the original enum
  PLANE=0 HFIELD=1 SPHERE=2 CAPSULE=3 ELLIPSOID=4 CYLINDER=5 BOX=6 MESH=7.
  BOX-BOX is mjc_BoxBox (separate detector) and is rejected here.  PLANE /
  HFIELD / SPHERE / CAPSULE / ELLIPSOID pairs are out of domain and are
  explicitly rejected; this module never widens the model's geom set.

RAW CONTACT (detectors._emit pattern, RAW_F = 10 float64 per slot):
  raw[0] = dist (= margin - depth), raw[1:4] = pos,
  raw[4:7] = frame normal (unit), raw[7:10] = frame tangent hint (zeros).
  At most one raw row is written per call (kernel.cu mesh_mpr_kernel
  semantics).  mjc_Convex's optional MULTICCD perturbation loop
  (engine_collision_convex.c:299-362) is not part of the C07 reference
  kernel and is not reproduced here.

_convex_mpr RETURN: contact count 0 or 1, or a negative error code
  -1  pair / geom type outside the five routes above
  -2  watchdog exhausted: hill-climbing exceeded HILL_WATCHDOG, or an MPR
      stage loop exceeded STAGE_WATCHDOG (fail-closed, never a silent
      partial portal)
  -3  mesh binding invalid (MESH geom with dataid < 0)
  -4  mesh support search failed (no support vertex found)
Separation is not an error: it returns 0 and writes no raw rows.

DELIBERATE BOUNDARIES (kept identical to the C07 reference kernel):
  - mjc_fixNormal (engine_collision_convex.c:875, applied by
    mjc_MPRIteration after the contact fill) is a no-op when neither geom
    is SPHERE/CAPSULE/ELLIPSOID/CYLINDER and is absent from kernel.cu;
    this port writes the raw MPR direction exactly as kernel.cu:276-280.
  - kernel.cu's 256-entry support-id recording budget (its reason 7) is a
    diagnostic sink of the C07 replay harness, not part of the MPR
    algorithm; support calls are not capped here.
  - maxiter / tolerance default to the MuJoCo 2.3.7 mjOption values
    (mpr_iterations = 50, mpr_tolerance = 1e-6) and can be overridden per
    call, matching kernel.cu's model.maxiter / model.tolerance inputs.
"""
from __future__ import annotations
import libero.gpu.precision as scalar

import warp as wp

wp.set_module_options({'enable_backward': False, 'fuse_fp': True, 'fast_math': False})

# float.h DBL_EPSILON: scale of kernel.cu zero()/eq()
DBL_EPSILON = scalar.wp_real(scalar.epsilon)
# mjMINVAL (mujoco mjtnum.h): cylinder XY-plane guard in mjccd_support
MINVAL = scalar.wp_real(1.0e-15)
# hill-climbing watchdog (kernel.cu mesh_support loop); exhaustion returns -2
HILL_WATCHDOG = 1024
# kernel.cu Query.watchdog for the discover / refine / penetrate loops
STAGE_WATCHDOG = 4096
# MuJoCo 2.3.7 mjOption defaults (kernel.cu model.maxiter / model.tolerance)
MPR_MAXITER = wp.int32(50)
MPR_TOLERANCE = scalar.wp_real(1.0e-6)

try:
    from .detectors import _emit
except ImportError:  # standalone load outside the package keeps the same pattern
    @wp.func
    def _emit(raw: wp.array(dtype=scalar.wp_real), base: wp.int32, r: wp.int32,
              dist: scalar.wp_real, px: scalar.wp_real, py: scalar.wp_real, pz: scalar.wp_real,
              nx: scalar.wp_real, ny: scalar.wp_real, nz: scalar.wp_real,
              tx: scalar.wp_real, ty: scalar.wp_real, tz: scalar.wp_real) -> wp.int32:
        # detectors._emit pattern: RAW_F = 10 layout, r >= 50 writes nothing
        if r >= 50:
            return wp.int32(0)
        o = base + r * 10
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


# ---------------------------------------------------------------------------
# kernel.cu struct V helpers (operation order identical to kernel.cu:26-38)
# ---------------------------------------------------------------------------


@wp.func
def _add(a: scalar.vec3_real, b: scalar.vec3_real) -> scalar.vec3_real:
    return scalar.vec3_real(a[0] + b[0], a[1] + b[1], a[2] + b[2])


@wp.func
def _sub(a: scalar.vec3_real, b: scalar.vec3_real) -> scalar.vec3_real:
    return scalar.vec3_real(a[0] - b[0], a[1] - b[1], a[2] - b[2])


@wp.func
def _scale(a: scalar.vec3_real, b: scalar.wp_real) -> scalar.vec3_real:
    return scalar.vec3_real(a[0] * b, a[1] * b, a[2] * b)


@wp.func
def _dot(a: scalar.vec3_real, b: scalar.vec3_real) -> scalar.wp_real:
    d = a[0] * b[0]
    d += a[1] * b[1]
    d += a[2] * b[2]
    return d


@wp.func
def _cross(a: scalar.vec3_real, b: scalar.vec3_real) -> scalar.vec3_real:
    return scalar.vec3_real(a[1] * b[2] - a[2] * b[1],
                    a[2] * b[0] - a[0] * b[2],
                    a[0] * b[1] - a[1] * b[0])


@wp.func
def _norm(a: scalar.vec3_real) -> scalar.vec3_real:
    k = scalar.wp_real(1.0) / wp.sqrt(_dot(a, a))
    return _scale(a, k)


@wp.func
def _zero(a: scalar.wp_real) -> bool:
    return wp.abs(a) < DBL_EPSILON


@wp.func
def _eq(a: scalar.wp_real, b: scalar.wp_real) -> bool:
    d = wp.abs(a - b)
    if wp.abs(d) < DBL_EPSILON:
        return True
    aa = wp.abs(a)
    bb = wp.abs(b)
    mx = aa
    if bb > aa:
        mx = bb
    return d < DBL_EPSILON * mx


@wp.func
def _veq(a: scalar.vec3_real, b: scalar.vec3_real) -> bool:
    return _eq(a[0], b[0]) and _eq(a[1], b[1]) and _eq(a[2], b[2])


@wp.func
def _sign(x: scalar.wp_real) -> scalar.wp_real:
    # mju_sign, engine_util_misc.c:757-765
    if x < scalar.wp_real(0.0):
        return scalar.wp_real(-1.0)
    if x > scalar.wp_real(0.0):
        return scalar.wp_real(1.0)
    return scalar.wp_real(0.0)


@wp.func
def _zero3() -> scalar.vec3_real:
    return scalar.vec3_real(scalar.wp_real(0.0), scalar.wp_real(0.0), scalar.wp_real(0.0))


# ---------------------------------------------------------------------------
# Support functions
# ---------------------------------------------------------------------------


@wp.func
def _convex_support(geom_type: wp.int32,
                    s0: scalar.wp_real, s1: scalar.wp_real, s2: scalar.wp_real,
                    ldx: scalar.wp_real, ldy: scalar.wp_real, ldz: scalar.wp_real,
                    mesh_vert: wp.array(dtype=wp.float32),
                    mesh_vertadr: wp.array(dtype=wp.int32),
                    mesh_vertnum: wp.array(dtype=wp.int32),
                    mesh_graphadr: wp.array(dtype=wp.int32),
                    mesh_graph: wp.array(dtype=wp.int32),
                    mid: wp.int32):
    """Support point of one convex geom in LOCAL coordinates.

    geom_type: 5 = CYLINDER, 6 = BOX, 7 = MESH (original enum order).
    (ldx, ldy, ldz) is the direction already rotated into the geom frame
    (mju_rotVecMatT).  Returns (point, err): err is 0 on success, -1 for a
    geom type outside cylinder/box/mesh, -2 when the mesh hill-climbing
    watchdog is exhausted, -3 for an invalid mesh binding, -4 when the mesh
    search finds no support vertex.
    """
    if geom_type == 6:
        # box: res[i] = mju_sign(dir[i]) * size[i]  (convex.c:101-105)
        return scalar.vec3_real(_sign(ldx) * s0, _sign(ldy) * s1, _sign(ldz) * s2), wp.int32(0)
    if geom_type == 5:
        # cylinder (convex.c:87-99): circle support in XY, signed half in Z
        tmp = wp.sqrt(ldx * ldx + ldy * ldy)
        sx = scalar.wp_real(0.0)
        sy = scalar.wp_real(0.0)
        if tmp > MINVAL:
            sx = ldx / tmp * s0
            sy = ldy / tmp * s0
        return scalar.vec3_real(sx, sy, _sign(ldz) * s1), wp.int32(0)
    if geom_type == 7:
        if mid < 0:
            return _zero3(), wp.int32(-3)
        adr = mesh_vertadr[mid]
        num = mesh_vertnum[mid]
        ga = mesh_graphadr[mid]
        best = wp.int32(-1)
        tmp = scalar.wp_real(-1.0e10)
        if ga < 0:
            # exhaustive scan when the hull graph is absent
            for ii in range(num):
                vi = 3 * (adr + ii)
                vdot = (ldx * scalar.wp_real(mesh_vert[vi])
                        + ldy * scalar.wp_real(mesh_vert[vi + 1])
                        + ldz * scalar.wp_real(mesh_vert[vi + 2]))
                if vdot > tmp:
                    tmp = vdot
                    best = ii
        else:
            # graph hill-climbing (kernel.cu mesh_support:65-84)
            nv = mesh_graph[ga]
            g0 = mesh_graph[ga + 2 + nv]
            vi = 3 * (adr + g0)
            tmp = (ldx * scalar.wp_real(mesh_vert[vi])
                   + ldy * scalar.wp_real(mesh_vert[vi + 1])
                   + ldz * scalar.wp_real(mesh_vert[vi + 2]))
            cur = wp.int32(0)
            change = wp.int32(1)
            steps = wp.int32(0)
            while change != 0:
                steps += 1
                if steps > HILL_WATCHDOG:
                    # watchdog: explicit error, never a silent partial answer
                    return _zero3(), wp.int32(-2)
                change = wp.int32(0)
                ei = mesh_graph[ga + 2 + cur]
                loc = mesh_graph[ga + 2 + 2 * nv + ei]
                while loc >= 0:
                    gi = mesh_graph[ga + 2 + nv + loc]
                    vi2 = 3 * (adr + gi)
                    vdot = (ldx * scalar.wp_real(mesh_vert[vi2])
                            + ldy * scalar.wp_real(mesh_vert[vi2 + 1])
                            + ldz * scalar.wp_real(mesh_vert[vi2 + 2]))
                    if vdot > tmp:
                        tmp = vdot
                        cur = loc
                        change = wp.int32(1)
                    ei += 1
                    loc = mesh_graph[ga + 2 + 2 * nv + ei]
            best = mesh_graph[ga + 2 + nv + cur]
        if best < 0:
            return _zero3(), wp.int32(-4)
        vi = 3 * (adr + best)
        return scalar.vec3_real(scalar.wp_real(mesh_vert[vi]),
                        scalar.wp_real(mesh_vert[vi + 1]),
                        scalar.wp_real(mesh_vert[vi + 2])), wp.int32(0)
    return _zero3(), wp.int32(-1)


@wp.func
def _geom_support_world(gtype: wp.int32,
                        s0: scalar.wp_real, s1: scalar.wp_real, s2: scalar.wp_real,
                        px: scalar.wp_real, py: scalar.wp_real, pz: scalar.wp_real,
                        m: wp.array(dtype=scalar.wp_real), mo: wp.int32,
                        mid: wp.int32,
                        dx: scalar.wp_real, dy: scalar.wp_real, dz: scalar.wp_real,
                        mesh_vert: wp.array(dtype=wp.float32),
                        mesh_vertadr: wp.array(dtype=wp.int32),
                        mesh_vertnum: wp.array(dtype=wp.int32),
                        mesh_graphadr: wp.array(dtype=wp.int32),
                        mesh_graph: wp.array(dtype=wp.int32),
                        margin: scalar.wp_real):
    """One geom's world support point for a world direction.

    kernel.cu mesh_support:53-91: rotate the direction into the geom frame,
    take the local support, add (local_dir * margin) / 2 per axis (MuJoCo's
    association, convex.c:197-199), rotate back and add the geom position.
    Returns (point, err).
    """
    # dir = rotT(direction, mat)
    ldx = m[mo] * dx + m[mo + 3] * dy + m[mo + 6] * dz
    ldy = m[mo + 1] * dx + m[mo + 4] * dy + m[mo + 7] * dz
    ldz = m[mo + 2] * dx + m[mo + 5] * dy + m[mo + 8] * dz
    res, err = _convex_support(gtype, s0, s1, s2, ldx, ldy, ldz,
                               mesh_vert, mesh_vertadr, mesh_vertnum,
                               mesh_graphadr, mesh_graph, mid)
    if err != 0:
        return _zero3(), err
    sx = res[0] + (ldx * margin) / scalar.wp_real(2.0)
    sy = res[1] + (ldy * margin) / scalar.wp_real(2.0)
    sz = res[2] + (ldz * margin) / scalar.wp_real(2.0)
    # add(rot(res, mat), pos)
    wx = (m[mo] * sx + m[mo + 1] * sy + m[mo + 2] * sz) + px
    wy = (m[mo + 3] * sx + m[mo + 4] * sy + m[mo + 5] * sz) + py
    wz = (m[mo + 6] * sx + m[mo + 7] * sy + m[mo + 8] * sz) + pz
    return scalar.vec3_real(wx, wy, wz), wp.int32(0)


@wp.func
def _minkowski_support(dx: scalar.wp_real, dy: scalar.wp_real, dz: scalar.wp_real,
                       p1x: scalar.wp_real, p1y: scalar.wp_real, p1z: scalar.wp_real,
                       m1: wp.array(dtype=scalar.wp_real), o1: wp.int32,
                       s10: scalar.wp_real, s11: scalar.wp_real, s12: scalar.wp_real,
                       t1: wp.int32, d1: wp.int32,
                       p2x: scalar.wp_real, p2y: scalar.wp_real, p2z: scalar.wp_real,
                       m2: wp.array(dtype=scalar.wp_real), o2: wp.int32,
                       s20: scalar.wp_real, s21: scalar.wp_real, s22: scalar.wp_real,
                       t2: wp.int32, d2: wp.int32,
                       mesh_vert: wp.array(dtype=wp.float32),
                       mesh_vertadr: wp.array(dtype=wp.int32),
                       mesh_vertnum: wp.array(dtype=wp.int32),
                       mesh_graphadr: wp.array(dtype=wp.int32),
                       mesh_graph: wp.array(dtype=wp.int32),
                       margin: scalar.wp_real):
    """Minkowski-difference support S{v, v1, v2} for a world direction.

    kernel.cu support():92-103 -- v1 = support(geom1, dir),
    v2 = support(geom2, -dir), v = v1 - v2.  Returns (v, v1, v2, err).
    """
    v1, err = _geom_support_world(t1, s10, s11, s12, p1x, p1y, p1z, m1, o1, d1,
                                  dx, dy, dz,
                                  mesh_vert, mesh_vertadr, mesh_vertnum,
                                  mesh_graphadr, mesh_graph, margin)
    if err != 0:
        return _zero3(), _zero3(), _zero3(), err
    v2, err = _geom_support_world(t2, s20, s21, s22, p2x, p2y, p2z, m2, o2, d2,
                                  -dx, -dy, -dz,
                                  mesh_vert, mesh_vertadr, mesh_vertnum,
                                  mesh_graphadr, mesh_graph, margin)
    if err != 0:
        return _zero3(), _zero3(), _zero3(), err
    return _sub(v1, v2), v1, v2, wp.int32(0)


# ---------------------------------------------------------------------------
# Portal helpers (kernel.cu:104-124)
# ---------------------------------------------------------------------------


@wp.func
def _portal_dir(p1v: scalar.vec3_real, p2v: scalar.vec3_real, p3v: scalar.vec3_real) -> scalar.vec3_real:
    # norm(cross(sub(p2.v, p1.v), sub(p3.v, p1.v)))
    return _norm(_cross(_sub(p2v, p1v), _sub(p3v, p1v)))


@wp.func
def _encapsulates(p1v: scalar.vec3_real, dir: scalar.vec3_real) -> bool:
    x = _dot(dir, p1v)
    return _zero(x) or x > scalar.wp_real(0.0)


@wp.func
def _can_encapsulate(sv: scalar.vec3_real, dir: scalar.vec3_real) -> bool:
    x = _dot(sv, dir)
    return _zero(x) or x > scalar.wp_real(0.0)


@wp.func
def _reach_tolerance(p1v: scalar.vec3_real, p2v: scalar.vec3_real, p3v: scalar.vec3_real,
                     sv: scalar.vec3_real, dir: scalar.vec3_real, tol: scalar.wp_real) -> bool:
    d1 = _dot(p1v, dir)
    d2 = _dot(p2v, dir)
    d3 = _dot(p3v, dir)
    d4 = _dot(sv, dir)
    x = d4 - d1
    y = d4 - d2
    z = d4 - d3
    if y < x:
        x = y
    if z < x:
        x = z
    return _eq(x, tol) or x < tol


@wp.func
def _expand_slot(sv: scalar.vec3_real, p0v: scalar.vec3_real, p1v: scalar.vec3_real,
                 p2v: scalar.vec3_real, p3v: scalar.vec3_real) -> wp.int32:
    """Which portal slot kernel.cu expand() replaces with s (1, 2 or 3)."""
    v4v0 = _cross(sv, p0v)
    d = _dot(p1v, v4v0)
    if d > scalar.wp_real(0.0):
        d = _dot(p2v, v4v0)
        if d > scalar.wp_real(0.0):
            return wp.int32(1)
        return wp.int32(3)
    d = _dot(p3v, v4v0)
    if d > scalar.wp_real(0.0):
        return wp.int32(2)
    return wp.int32(1)


# ---------------------------------------------------------------------------
# Distance helpers (kernel.cu:177-202, pinned libccd vec3.c:137-216)
# ---------------------------------------------------------------------------


@wp.func
def _segment_dist2(P: scalar.vec3_real, A: scalar.vec3_real, B: scalar.vec3_real):
    d = _sub(B, A)
    a = _sub(A, P)
    t = scalar.wp_real(-1.0) * _dot(a, d)
    t = t / _dot(d, d)
    if t < scalar.wp_real(0.0) or _zero(t):
        return _dot(_sub(A, P), _sub(A, P)), A
    if t > scalar.wp_real(1.0) or _eq(t, scalar.wp_real(1.0)):
        return _dot(_sub(B, P), _sub(B, P)), B
    witness = _add(_scale(d, t), A)
    diff = _sub(witness, P)
    return _dot(diff, diff), witness


@wp.func
def _triangle_dist2(P: scalar.vec3_real, A: scalar.vec3_real, B: scalar.vec3_real, C: scalar.vec3_real):
    d1 = _sub(B, A)
    d2 = _sub(C, A)
    a = _sub(A, P)
    u = _dot(a, a)
    vv = _dot(d1, d1)
    w = _dot(d2, d2)
    pp = _dot(a, d1)
    q = _dot(a, d2)
    r = _dot(d1, d2)
    det = w * vv - r * r
    s = scalar.wp_real(-1.0)
    t = scalar.wp_real(-1.0)
    if not _zero(det):
        s = (q * r - w * pp) / det
        t = (-s * r - q) / w
    if ((_zero(s) or s > scalar.wp_real(0.0)) and (_eq(s, scalar.wp_real(1.0)) or s < scalar.wp_real(1.0))
            and (_zero(t) or t > scalar.wp_real(0.0)) and (_eq(t, scalar.wp_real(1.0)) or t < scalar.wp_real(1.0))
            and (_eq(t + s, scalar.wp_real(1.0)) or t + s < scalar.wp_real(1.0))):
        witness = _add(_add(A, _scale(d1, s)), _scale(d2, t))
        delta = _sub(witness, P)
        return _dot(delta, delta), witness
    dist, witness = _segment_dist2(P, A, B)
    dist2, witness2 = _segment_dist2(P, A, C)
    if dist2 < dist:
        dist = dist2
        witness = witness2
    dist2, witness2 = _segment_dist2(P, B, C)
    if dist2 < dist:
        dist = dist2
        witness = witness2
    return dist, witness


@wp.func
def _find_pos(p0v: scalar.vec3_real, p0v1: scalar.vec3_real, p0v2: scalar.vec3_real,
              p1v: scalar.vec3_real, p1v1: scalar.vec3_real, p1v2: scalar.vec3_real,
              p2v: scalar.vec3_real, p2v1: scalar.vec3_real, p2v2: scalar.vec3_real,
              p3v: scalar.vec3_real, p3v1: scalar.vec3_real, p3v2: scalar.vec3_real) -> scalar.vec3_real:
    """Barycentric witness position, kernel.cu find_pos:203-222."""
    dir = _portal_dir(p1v, p2v, p3v)
    b0 = _dot(_cross(p1v, p2v), p3v)
    b1 = _dot(_cross(p3v, p2v), p0v)
    b2 = _dot(_cross(p0v, p1v), p3v)
    b3 = _dot(_cross(p2v, p1v), p0v)
    tot = b0 + b1 + b2 + b3
    if _zero(tot) or tot < scalar.wp_real(0.0):
        b0 = scalar.wp_real(0.0)
        b1 = _dot(_cross(p2v, p3v), dir)
        b2 = _dot(_cross(p3v, p1v), dir)
        b3 = _dot(_cross(p1v, p2v), dir)
        tot = b1 + b2 + b3
    inv = scalar.wp_real(1.0) / tot
    p1 = _zero3()
    p2 = _zero3()
    p1 = _add(p1, _scale(p0v1, b0))
    p2 = _add(p2, _scale(p0v2, b0))
    p1 = _add(p1, _scale(p1v1, b1))
    p2 = _add(p2, _scale(p1v2, b1))
    p1 = _add(p1, _scale(p2v1, b2))
    p2 = _add(p2, _scale(p2v2, b2))
    p1 = _add(p1, _scale(p3v1, b3))
    p2 = _add(p2, _scale(p3v2, b3))
    p1 = _scale(p1, inv)
    p2 = _scale(p2, inv)
    return _scale(_add(p1, p2), scalar.wp_real(0.5))


# ---------------------------------------------------------------------------
# mjc_Convex MPR entry (kernel.cu mpr:241-253 + contact fill :276-281)
# ---------------------------------------------------------------------------


@wp.func
def _convex_mpr(raw: wp.array(dtype=scalar.wp_real), base: wp.int32,
                p1x: scalar.wp_real, p1y: scalar.wp_real, p1z: scalar.wp_real,
                m1: wp.array(dtype=scalar.wp_real), o1: wp.int32,
                s10: scalar.wp_real, s11: scalar.wp_real, s12: scalar.wp_real,
                t1: wp.int32, d1: wp.int32,
                p2x: scalar.wp_real, p2y: scalar.wp_real, p2z: scalar.wp_real,
                m2: wp.array(dtype=scalar.wp_real), o2: wp.int32,
                s20: scalar.wp_real, s21: scalar.wp_real, s22: scalar.wp_real,
                t2: wp.int32, d2: wp.int32,
                mesh_vert: wp.array(dtype=wp.float32),
                mesh_vertadr: wp.array(dtype=wp.int32),
                mesh_vertnum: wp.array(dtype=wp.int32),
                mesh_graphadr: wp.array(dtype=wp.int32),
                mesh_graph: wp.array(dtype=wp.int32),
                margin: scalar.wp_real,
                maxiter: wp.int32 = MPR_MAXITER,
                tolerance: scalar.wp_real = MPR_TOLERANCE) -> wp.int32:
    """mjc_Convex MPR for one convex pair; writes raw rows at base.

    (p1*, m1/o1, s10..s12, t1, d1) is geom1's world position, row-major
    rotation matrix, size, type and mesh dataid; (p2*, ...) is geom2's.
    t1/t2 use the original enum (CYLINDER=5 BOX=6 MESH=7).  Returns the
    contact count (0 or 1) or a negative error code (see module docstring).
    """
    # ---- domain gate: exactly the five mjc_Convex routes ----
    ta = t1
    tb = t2
    if ta > tb:
        ta = t2
        tb = t1
    key = ta * 8 + tb
    allowed = wp.int32(0)
    if key == 45 or key == 46 or key == 47 or key == 55 or key == 63:
        allowed = wp.int32(1)
    if allowed == 0:
        return wp.int32(-1)
    if t1 == 7 and d1 < 0:
        return wp.int32(-3)
    if t2 == 7 and d2 < 0:
        return wp.int32(-3)

    z3 = _zero3()

    # ---- discover: initial portal (kernel.cu:127-162) ----
    p0v = scalar.vec3_real(p1x - p2x, p1y - p2y, p1z - p2z)
    p0v1 = scalar.vec3_real(p1x, p1y, p1z)
    p0v2 = scalar.vec3_real(p2x, p2y, p2z)
    if _veq(p0v, z3):
        p0v = _add(p0v, scalar.vec3_real(DBL_EPSILON * scalar.wp_real(10.0),
                                 scalar.wp_real(0.0), scalar.wp_real(0.0)))
    ndir = _norm(_scale(p0v, scalar.wp_real(-1.0)))
    s_v, s_v1, s_v2, err = _minkowski_support(
        ndir[0], ndir[1], ndir[2],
        p1x, p1y, p1z, m1, o1, s10, s11, s12, t1, d1,
        p2x, p2y, p2z, m2, o2, s20, s21, s22, t2, d2,
        mesh_vert, mesh_vertadr, mesh_vertnum, mesh_graphadr, mesh_graph, margin)
    if err != 0:
        return err
    p1v, p1v1, p1v2 = s_v, s_v1, s_v2
    d = _dot(p1v, ndir)
    if _zero(d) or d < scalar.wp_real(0.0):
        return wp.int32(0)  # separated
    ndir = _cross(p0v, p1v)
    if _zero(_dot(ndir, ndir)):
        if _veq(p1v, z3):
            # touch: normal undefined, kernel.cu emits no contact
            return wp.int32(0)
        # segment case (kernel.cu mpr rc==2): one contact
        pos = _scale(_add(p1v1, p1v2), scalar.wp_real(0.5))
        depth = wp.sqrt(_dot(p1v, p1v))
        nrm = _norm(p1v)
        return _emit(raw, base, wp.int32(0), margin - depth,
                     pos[0], pos[1], pos[2], nrm[0], nrm[1], nrm[2],
                     scalar.wp_real(0.0), scalar.wp_real(0.0), scalar.wp_real(0.0))
    ndir = _norm(ndir)
    s_v, s_v1, s_v2, err = _minkowski_support(
        ndir[0], ndir[1], ndir[2],
        p1x, p1y, p1z, m1, o1, s10, s11, s12, t1, d1,
        p2x, p2y, p2z, m2, o2, s20, s21, s22, t2, d2,
        mesh_vert, mesh_vertadr, mesh_vertnum, mesh_graphadr, mesh_graph, margin)
    if err != 0:
        return err
    p2v, p2v1, p2v2 = s_v, s_v1, s_v2
    d = _dot(p2v, ndir)
    if _zero(d) or d < scalar.wp_real(0.0):
        return wp.int32(0)  # separated
    va = _sub(p1v, p0v)
    vb = _sub(p2v, p0v)
    ndir = _norm(_cross(va, vb))
    d = _dot(ndir, p0v)
    if d > scalar.wp_real(0.0):
        p1v, p1v1, p1v2, p2v, p2v1, p2v2 = p2v, p2v1, p2v2, p1v, p1v1, p1v2
        ndir = _scale(ndir, scalar.wp_real(-1.0))
    psize = wp.int32(3)
    discover_n = wp.int32(0)
    p3v = z3
    p3v1 = z3
    p3v2 = z3
    while psize < 4:
        discover_n += 1
        if discover_n > STAGE_WATCHDOG:
            return wp.int32(-2)
        s_v, s_v1, s_v2, err = _minkowski_support(
            ndir[0], ndir[1], ndir[2],
            p1x, p1y, p1z, m1, o1, s10, s11, s12, t1, d1,
            p2x, p2y, p2z, m2, o2, s20, s21, s22, t2, d2,
            mesh_vert, mesh_vertadr, mesh_vertnum, mesh_graphadr, mesh_graph, margin)
        if err != 0:
            return err
        p3v, p3v1, p3v2 = s_v, s_v1, s_v2
        d = _dot(p3v, ndir)
        if _zero(d) or d < scalar.wp_real(0.0):
            return wp.int32(0)  # separated
        cont = wp.int32(0)
        va = _cross(p1v, p3v)
        d = _dot(va, p0v)
        if d < scalar.wp_real(0.0) and not _zero(d):
            p2v, p2v1, p2v2 = p3v, p3v1, p3v2
            cont = wp.int32(1)
        if cont == 0:
            va = _cross(p3v, p2v)
            d = _dot(va, p0v)
            if d < scalar.wp_real(0.0) and not _zero(d):
                p1v, p1v1, p1v2 = p3v, p3v1, p3v2
                cont = wp.int32(1)
        if cont != 0:
            va = _sub(p1v, p0v)
            vb = _sub(p2v, p0v)
            ndir = _norm(_cross(va, vb))
        else:
            psize = wp.int32(4)

    # ---- refine: portal must encapsulate the origin (kernel.cu:163-175) ----
    refine_n = wp.int32(0)
    refine_done = wp.int32(0)
    while refine_done == 0:
        refine_n += 1
        if refine_n > STAGE_WATCHDOG:
            return wp.int32(-2)
        ndir = _portal_dir(p1v, p2v, p3v)
        if _encapsulates(p1v, ndir):
            refine_done = wp.int32(1)
        else:
            s_v, s_v1, s_v2, err = _minkowski_support(
                ndir[0], ndir[1], ndir[2],
                p1x, p1y, p1z, m1, o1, s10, s11, s12, t1, d1,
                p2x, p2y, p2z, m2, o2, s20, s21, s22, t2, d2,
                mesh_vert, mesh_vertadr, mesh_vertnum, mesh_graphadr,
                mesh_graph, margin)
            if err != 0:
                return err
            if not _can_encapsulate(s_v, ndir):
                return wp.int32(0)  # separated
            if _reach_tolerance(p1v, p2v, p3v, s_v, ndir, tolerance):
                return wp.int32(0)  # separated
            slot = _expand_slot(s_v, p0v, p1v, p2v, p3v)
            if slot == 1:
                p1v, p1v1, p1v2 = s_v, s_v1, s_v2
            elif slot == 2:
                p2v, p2v1, p2v2 = s_v, s_v1, s_v2
            else:
                p3v, p3v1, p3v2 = s_v, s_v1, s_v2

    # ---- find_penetration (kernel.cu:223-240) ----
    pene_n = wp.int32(0)
    iter = wp.int32(0)
    pene_done = wp.int32(0)
    depth = scalar.wp_real(0.0)
    nrm = z3
    pos = z3
    while pene_done == 0:
        pene_n += 1
        if pene_n > STAGE_WATCHDOG:
            return wp.int32(-2)
        ndir = _portal_dir(p1v, p2v, p3v)
        s_v, s_v1, s_v2, err = _minkowski_support(
            ndir[0], ndir[1], ndir[2],
            p1x, p1y, p1z, m1, o1, s10, s11, s12, t1, d1,
            p2x, p2y, p2z, m2, o2, s20, s21, s22, t2, d2,
            mesh_vert, mesh_vertadr, mesh_vertnum, mesh_graphadr, mesh_graph,
            margin)
        if err != 0:
            return err
        tolerance_hit = _reach_tolerance(p1v, p2v, p3v, s_v, ndir, tolerance)
        if tolerance_hit or iter > maxiter:
            dist2, witness = _triangle_dist2(z3, p1v, p2v, p3v)
            depth = wp.sqrt(dist2)
            if _zero(depth):
                nrm = z3
            else:
                nrm = _norm(witness)
            pos = _find_pos(p0v, p0v1, p0v2,
                            p1v, p1v1, p1v2, p2v, p2v1, p2v2, p3v, p3v1, p3v2)
            pene_done = wp.int32(1)
        else:
            slot = _expand_slot(s_v, p0v, p1v, p2v, p3v)
            if slot == 1:
                p1v, p1v1, p1v2 = s_v, s_v1, s_v2
            elif slot == 2:
                p2v, p2v1, p2v2 = s_v, s_v1, s_v2
            else:
                p3v, p3v1, p3v2 = s_v, s_v1, s_v2
            iter += 1

    # ---- contact generation (kernel.cu:276-281) ----
    if _veq(nrm, z3):
        return wp.int32(0)  # contact found but normal undefined
    return _emit(raw, base, wp.int32(0), margin - depth,
                 pos[0], pos[1], pos[2], nrm[0], nrm[1], nrm[2],
                 scalar.wp_real(0.0), scalar.wp_real(0.0), scalar.wp_real(0.0))


# ---------------------------------------------------------------------------
# Batched kernel: one thread per pair.  pose is 25 float64 per pair in
# kernel.cu order [margin, pos1(3), mat1(9), pos2(3), mat2(9)].
# out_status: 0 ok, 3 pair-index / mesh-dataid binding error, 4 nonfinite
#             pose or bad MPR parameter, otherwise the negative code from
#             _convex_mpr (-1 for a pair outside the five routes, etc.).
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def convex_mpr_kernel(
        pose: wp.array(dtype=scalar.wp_real),
        pair_g1: wp.array(dtype=wp.int32),
        pair_g2: wp.array(dtype=wp.int32),
        geom_type: wp.array(dtype=wp.int32),
        geom_dataid: wp.array(dtype=wp.int32),
        geom_size: wp.array(dtype=scalar.wp_real),
        mesh_vert: wp.array(dtype=wp.float32),
        mesh_vertadr: wp.array(dtype=wp.int32),
        mesh_vertnum: wp.array(dtype=wp.int32),
        mesh_graphadr: wp.array(dtype=wp.int32),
        mesh_graph: wp.array(dtype=wp.int32),
        ngeom: int, nmesh: int,
        maxiter: wp.int32, tolerance: scalar.wp_real,
        raw: wp.array(dtype=scalar.wp_real),
        out_count: wp.array(dtype=wp.int32),
        out_status: wp.array(dtype=wp.int32)):
    w = wp.tid()
    a = pair_g1[w]
    b = pair_g2[w]
    if a < 0 or b < 0 or a >= ngeom or b >= ngeom:
        out_status[w] = 3
        return
    t1 = geom_type[a]
    t2 = geom_type[b]
    ta = t1
    tb = t2
    if ta > tb:
        ta = t2
        tb = t1
    key = ta * 8 + tb
    if not (key == 45 or key == 46 or key == 47 or key == 55 or key == 63):
        out_status[w] = -1
        return
    d1 = geom_dataid[a]
    d2 = geom_dataid[b]
    if t1 == 7 and (d1 < 0 or d1 >= nmesh):
        out_status[w] = -3
        return
    if t2 == 7 and (d2 < 0 or d2 >= nmesh):
        out_status[w] = -3
        return
    if maxiter < 0 or not (tolerance == tolerance) or tolerance < scalar.wp_real(0.0):
        out_status[w] = 4
        return
    po = 25 * w
    for i in range(25):
        if not wp.isfinite(pose[po + i]):
            out_status[w] = 4
            return
    margin = pose[po]
    base = w * 10
    cnt = _convex_mpr(
        raw, base,
        pose[po + 1], pose[po + 2], pose[po + 3], pose, po + 4,
        geom_size[a * 3], geom_size[a * 3 + 1], geom_size[a * 3 + 2], t1, d1,
        pose[po + 13], pose[po + 14], pose[po + 15], pose, po + 16,
        geom_size[b * 3], geom_size[b * 3 + 1], geom_size[b * 3 + 2], t2, d2,
        mesh_vert, mesh_vertadr, mesh_vertnum, mesh_graphadr, mesh_graph,
        margin, maxiter, tolerance)
    if cnt < 0:
        out_count[w] = 0
        out_status[w] = cnt
        return
    out_count[w] = cnt
    out_status[w] = 0
