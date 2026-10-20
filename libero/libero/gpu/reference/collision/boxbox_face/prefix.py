"""Independent FP64 device prefix of MuJoCo 2.3.7 mjc_BoxBox.

Source: src/engine/engine_collision_box.c:596-659 at c9246e1f5006379d599e0bcddf159a8616d31441
(SHA-256 fd038c185abfae66213bcabd0db34450a5fb596481fa5d1bb567fdff62a587b2),
and src/engine/engine_util_blas.c helpers (Apache-2.0). This stops before
cross-axis SAT; a nonseparated face prefix is not a collision decision.
"""
from __future__ import annotations
import libero.gpu.precision as scalar

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": True})

# FP64 output [N,P,55] is specified by the schema in doc/phase_02/boxbox_face.
# State [N,P,5]: evaluated, separated, first_axis, code, status.
COUNT_ERROR = 1
GEOM_ID_ERROR = 2
GEOM_TYPE_ERROR = 3
NONFINITE_ERROR = 4


@wp.kernel
def _face_prefix(
    size: wp.array(dtype=scalar.wp_real, ndim=2),
    kind: wp.array(dtype=wp.int32, ndim=1),
    xpos: wp.array(dtype=scalar.wp_real, ndim=3),
    xmat: wp.array(dtype=scalar.wp_real, ndim=3),
    pair_ids: wp.array(dtype=wp.int32, ndim=3),
    pair_count: wp.array(dtype=wp.int32, ndim=1),
    margin: wp.array(dtype=scalar.wp_real, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=1),
    out: wp.array(dtype=scalar.wp_real, ndim=3),
    route: wp.array(dtype=wp.int32, ndim=3),
    state: wp.array(dtype=wp.int32, ndim=3),
    geom_count: int,
    pair_capacity: int,
):
    t = wp.tid()
    w = t // pair_capacity
    q = t - w * pair_capacity
    for j in range(55):
        out[w, q, j] = scalar.wp_real(0.0)
    route[w, q, 0] = -1
    route[w, q, 1] = -1
    state[w, q, 0] = 0
    state[w, q, 1] = 0
    state[w, q, 2] = -1
    state[w, q, 3] = -1
    state[w, q, 4] = 0
    if active[w] == 0:
        return
    count = pair_count[w]
    if count < 0 or count > pair_capacity:
        state[w, q, 4] = COUNT_ERROR
        return
    if q >= count:
        return
    g1 = pair_ids[w, q, 0]
    g2 = pair_ids[w, q, 1]
    if g1 < 0 or g2 < 0 or g1 >= geom_count or g2 >= geom_count:
        state[w, q, 4] = GEOM_ID_ERROR
        return
    if kind[g1] != 6 or kind[g2] != 6:
        state[w, q, 4] = GEOM_TYPE_ERROR
        return
    route[w, q, 0] = g1
    route[w, q, 1] = g2
    if not wp.isfinite(margin[w, q]) or margin[w, q] < scalar.wp_real(0.0):
        state[w, q, 4] = NONFINITE_ERROR
        route[w, q, 0] = -1
        route[w, q, 1] = -1
        return
    for j in range(3):
        if not wp.isfinite(size[g1, j]) or not wp.isfinite(size[g2, j]) or size[g1, j] <= scalar.wp_real(0.0) or size[g2, j] <= scalar.wp_real(0.0) or not wp.isfinite(xpos[w, g1, j]) or not wp.isfinite(xpos[w, g2, j]):
            state[w, q, 4] = NONFINITE_ERROR
            route[w, q, 0] = -1
            route[w, q, 1] = -1
            return
    for j in range(9):
        if not wp.isfinite(xmat[w, g1, j]) or not wp.isfinite(xmat[w, g2, j]):
            state[w, q, 4] = NONFINITE_ERROR
            route[w, q, 0] = -1
            route[w, q, 1] = -1
            return

    # Both differences are evaluated independently, as in mju_sub3 calls.
    for j in range(3):
        v0 = xpos[w, g2, 0] - xpos[w, g1, 0]
        v1 = xpos[w, g2, 1] - xpos[w, g1, 1]
        v2 = xpos[w, g2, 2] - xpos[w, g1, 2]
        out[w, q, j] = (xmat[w, g1, j] * v0 + xmat[w, g1, 3 + j] * v1) + xmat[w, g1, 6 + j] * v2
        u0 = xpos[w, g1, 0] - xpos[w, g2, 0]
        u1 = xpos[w, g1, 1] - xpos[w, g2, 1]
        u2 = xpos[w, g1, 2] - xpos[w, g2, 2]
        out[w, q, 3 + j] = (xmat[w, g2, j] * u0 + xmat[w, g2, 3 + j] * u1) + xmat[w, g2, 6 + j] * u2

    # mju_mulMatTMat zeroes rot and skips numerically zero factors. Its
    # addToScl uses a sequential multiply followed by add for each cell.
    for i in range(3):
        for j in range(3):
            factor = xmat[w, g1, 3 * i + j]
            if factor != scalar.wp_real(0.0):
                for k in range(3):
                    idx = 6 + 3 * j + k
                    out[w, q, idx] = out[w, q, idx] + xmat[w, g2, 3 * i + k] * factor
    for i in range(3):
        for j in range(3):
            value = out[w, q, 6 + 3 * i + j]
            out[w, q, 15 + 3 * j + i] = value
            out[w, q, 24 + 3 * i + j] = wp.abs(value)
            out[w, q, 33 + 3 * j + i] = wp.abs(value)
    for j in range(3):
        out[w, q, 42 + j] = (out[w, q, 24 + 3 * j] * size[g2, 0] + out[w, q, 25 + 3 * j] * size[g2, 1]) + out[w, q, 26 + 3 * j] * size[g2, 2]
        out[w, q, 45 + j] = (out[w, q, 24 + j] * size[g1, 0] + out[w, q, 27 + j] * size[g1, 1]) + out[w, q, 30 + j] * size[g1, 2]
    penetration = margin[w, q]
    for i in range(3):
        penetration = penetration + (size[g1, i] * scalar.wp_real(3.0) + size[g2, i] * scalar.wp_real(3.0))
    code = int(-1)
    for i in range(3):
        c1 = (-wp.abs(out[w, q, i]) + size[g1, i]) + out[w, q, 42 + i]
        c2 = (-wp.abs(out[w, q, 3 + i]) + size[g2, i]) + out[w, q, 45 + i]
        out[w, q, 48 + i] = c1
        out[w, q, 51 + i] = c2
        state[w, q, 0] = i + 1
        if c1 < -margin[w, q] or c2 < -margin[w, q]:
            state[w, q, 1] = 1
            state[w, q, 2] = i
            break
        if c1 < penetration:
            penetration = c1
            side = int(0)
            if out[w, q, i] < scalar.wp_real(0.0):
                side = 1
            code = i + 3 * side
        if c2 < penetration:
            penetration = c2
            side = int(0)
            if out[w, q, 3 + i] < scalar.wp_real(0.0):
                side = 1
            code = i + 3 * side + 6
    out[w, q, 54] = penetration
    state[w, q, 3] = code


def validate_host_inputs(geom_size, geom_type, geom_xpos, geom_xmat,
                         pair_geom_ids, pair_count, margin, active_world):
    """Prepare boundary domain check; no normalization or pose correction."""
    size = np.asarray(geom_size)
    kind = np.asarray(geom_type)
    xpos = np.asarray(geom_xpos)
    xmat = np.asarray(geom_xmat)
    pair = np.asarray(pair_geom_ids)
    count = np.asarray(pair_count)
    gap = np.asarray(margin)
    active = np.asarray(active_world)
    if size.dtype != np.float64 or size.ndim != 2 or size.shape[1] != 3:
        raise ValueError('geom_size schema')
    n, g = xpos.shape[:2] if xpos.ndim == 3 else (0, 0)
    p = pair.shape[1] if pair.ndim == 3 else 0
    expected = ((kind, np.int32, (g,)), (xpos, np.float64, (n,g,3)),
                (xmat, np.float64, (n,g,9)), (pair, np.int32, (n,p,2)),
                (count, np.int32, (n,)), (gap, np.float64, (n,p)),
                (active, np.int32, (n,)))
    if n < 1 or p < 1 or size.shape != (g,3) or any(a.dtype != dtype or a.shape != shape for a,dtype,shape in expected):
        raise ValueError('BoxBoxFace input shape/dtype')
    if np.any((active != 0) & (active != 1)):
        raise ValueError('active_world must be 0/1')
    # Dynamic count/ID errors are intentionally handled on device. Only
    # addressed payload is checked here; sparse unused geom slots may be NaN.
    for w in range(n):
        if not active[w] or count[w] < 0 or count[w] > p:
            continue
        for q in range(int(count[w])):
            a,b = pair[w,q]
            if not (0 <= a < g and 0 <= b < g):
                continue
            if kind[a] != 6 or kind[b] != 6:
                raise ValueError('non-box geom')
            if np.any(~np.isfinite(size[[a,b]])) or np.any(size[[a,b]] <= 0) or np.any(~np.isfinite(xpos[w,[a,b]])) or np.any(~np.isfinite(xmat[w,[a,b]])) or not np.isfinite(gap[w,q]) or gap[w,q] < 0:
                raise ValueError('nonfinite/invalid addressed payload')
            for mat in xmat[w,[a,b]].reshape(2,3,3):
                if not np.allclose(mat.T @ mat, np.eye(3), rtol=0, atol=1e-9) or np.linalg.det(mat) <= 0:
                    raise ValueError('bad rotation matrix')
    return n,g,p


class BoxBoxFaceWorkspace:
    """One static geom table and preallocated N-world device buffers."""

    def __init__(self, geom_size, geom_type, n: int, p: int, device: str = 'cuda:0'):
        raw_size = np.asarray(geom_size)
        raw_kind = np.asarray(geom_type)
        if raw_size.dtype != np.float64 or raw_kind.dtype != np.int32:
            raise ValueError('static geom_size/geom_type must be FP64/int32')
        size = np.ascontiguousarray(raw_size)
        kind = np.ascontiguousarray(raw_kind)
        if size.ndim != 2 or size.shape[1] != 3 or kind.shape != (len(size),) or n < 1 or p < 1:
            raise ValueError('invalid static schema')
        self.n, self.g, self.p, self.device = n, len(size), p, device
        self.host_geom_size = size.copy()
        self.host_geom_type = kind.copy()
        self.geom_size = wp.array(size, dtype=scalar.wp_real, device=device)
        self.geom_type = wp.array(kind, dtype=wp.int32, device=device)
        self.geom_xpos = wp.empty((n,self.g,3), dtype=scalar.wp_real, device=device)
        self.geom_xmat = wp.empty((n,self.g,9), dtype=scalar.wp_real, device=device)
        self.pair_geom_ids = wp.empty((n,p,2), dtype=wp.int32, device=device)
        self.pair_count = wp.empty((n,), dtype=wp.int32, device=device)
        self.margin = wp.empty((n,p), dtype=scalar.wp_real, device=device)
        self.active_world = wp.empty((n,), dtype=wp.int32, device=device)
        self.out = wp.empty((n,p,55), dtype=scalar.wp_real, device=device)
        self.route = wp.empty((n,p,2), dtype=wp.int32, device=device)
        self.state = wp.empty((n,p,5), dtype=wp.int32, device=device)

    def upload(self, geom_xpos, geom_xmat, pair_geom_ids, pair_count, margin, active_world,
               *, validate: bool = True):
        arrays = tuple(np.ascontiguousarray(a) for a in (geom_xpos, geom_xmat, pair_geom_ids,
                                                          pair_count, margin, active_world))
        if validate:
            validate_host_inputs(self.host_geom_size, self.host_geom_type, *arrays)
        for name, value in zip(('geom_xpos','geom_xmat','pair_geom_ids','pair_count','margin','active_world'), arrays):
            getattr(self, name).assign(value)

    def launch(self, stream=None):
        wp.launch(_face_prefix, dim=self.n*self.p,
                  inputs=[self.geom_size,self.geom_type,self.geom_xpos,self.geom_xmat,
                          self.pair_geom_ids,self.pair_count,self.margin,self.active_world,
                          self.out,self.route,self.state,self.g,self.p],
                  device=self.device, stream=stream)
