"""Dense elliptic contact instantiation from a frozen original-C boundary.

MuJoCo 2.3.7 provenance: engine_core_constraint.c:157-248, 773-880 and
engine_support.c:208-257, 474-563. The output J is a local contact segment;
efc_address and outgoing nefc use the incoming global offset. No CPU contact
or Jacobian values are used in production.
"""

from __future__ import annotations
import libero.gpu.precision as scalar

import numpy as np
import warp as wp

wp.set_module_options({'enable_backward': False, 'fuse_fp': True, 'fast_math': False})

OK = 0
BAD_ACTIVE = 1
BAD_COUNT = 2
BAD_OFFSET = 3
BAD_CONTACT = 4
BAD_NUMERIC = 5
CAPACITY = 6
SAFE_MAGNITUDE = scalar.wp_real(scalar.safe_magnitude)


class ContactStatic:
    """Validated, immutable topology and the exact dense mj_jac ancestor route."""

    fields = ('geom_bodyid', 'body_rootid', 'body_parentid', 'body_dofnum',
              'body_dofadr', 'dof_parentid')

    def __init__(self, arrays, model_sha256: str, *, jacobian_mode='dense', cone_mode='elliptic'):
        if (not isinstance(model_sha256, str) or len(model_sha256) != 64 or
                any(c not in '0123456789abcdef' for c in model_sha256)):
            raise ValueError('full MJB SHA256 required')
        if jacobian_mode != 'dense' or cone_mode != 'elliptic':
            raise ValueError('only dense elliptic contact construction is supported')
        if set(arrays) not in (set(self.fields), set(self.fields) | {'body_simple'}):
            raise ValueError('static field set')
        self.arrays = {}
        for name in self.fields:
            a = np.asarray(arrays[name])
            if a.dtype != np.int32 or a.ndim != 1:
                raise ValueError(f'{name}: int32 vector required')
            self.arrays[name] = np.ascontiguousarray(a)
        self.ngeom = len(self.arrays['geom_bodyid'])
        self.nbody = len(self.arrays['body_rootid'])
        self.nv = len(self.arrays['dof_parentid'])
        if self.ngeom < 1 or self.nbody < 1:
            raise ValueError('empty model')
        a = self.arrays
        if (len(a['body_parentid']) != self.nbody or
            len(a['body_dofnum']) != self.nbody or
            len(a['body_dofadr']) != self.nbody or
            np.any((a['geom_bodyid'] < 0) | (a['geom_bodyid'] >= self.nbody)) or
            a['body_rootid'][0] != 0 or a['body_parentid'][0] != 0 or
            a['body_dofnum'][0] != 0):
            raise ValueError('invalid body topology')
        mask = np.zeros((self.nbody, max(self.nv, 1)), dtype=np.int32)
        for body in range(self.nbody):
            seen = set()
            p = body
            last = p
            while p:
                if p in seen or p < 0 or p >= self.nbody:
                    raise ValueError('body ancestor cycle')
                seen.add(p)
                last = p
                p = int(a['body_parentid'][p])
            if a['body_rootid'][body] != last:
                raise ValueError('body root')
            p = body
            while p and a['body_dofnum'][p] == 0:
                p = int(a['body_parentid'][p])
            if p:
                dof = int(a['body_dofadr'][p] + a['body_dofnum'][p] - 1)
                seen_dof = set()
                while dof >= 0:
                    if dof >= self.nv or dof in seen_dof:
                        raise ValueError('DOF ancestor cycle/range')
                    seen_dof.add(dof)
                    mask[body, dof] = 1
                    dof = int(a['dof_parentid'][dof])
        self.ancestor_mask = mask
        self.model_sha256 = model_sha256


@wp.kernel(enable_backward=False)
def _preflight(
    ncon: wp.array(dtype=wp.int32), incoming: wp.array(dtype=wp.int32),
    capacity: wp.array(dtype=wp.int32), active: wp.array(dtype=wp.int32),
    geom1: wp.array2d(dtype=wp.int32), geom2: wp.array2d(dtype=wp.int32),
    pos: wp.array3d(dtype=scalar.wp_real), frame: wp.array3d(dtype=scalar.wp_real),
    dist: wp.array2d(dtype=scalar.wp_real), margin: wp.array2d(dtype=scalar.wp_real),
    dim: wp.array2d(dtype=wp.int32), exclude: wp.array2d(dtype=wp.int32),
    cdof: wp.array3d(dtype=scalar.wp_real), subtree: wp.array3d(dtype=scalar.wp_real),
    geom_body: wp.array(dtype=wp.int32), rowstart: wp.array2d(dtype=wp.int32),
    local_count: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32),
    maxcon: int, maxrows: int, nv: int, nbody: int, ngeom: int,
):
    w = wp.tid()
    status[w] = OK
    if active[w] == 0:
        return
    if active[w] != 1:
        status[w] = BAD_ACTIVE
        return
    count = ncon[w]
    if count < 0 or count > maxcon:
        status[w] = BAD_COUNT
        return
    base = incoming[w]
    cap = capacity[w]
    if base < 0 or cap < base:
        status[w] = BAD_OFFSET
        return
    total = int(0)
    for c in range(count):
        rowstart[w, c] = total
        if exclude[w, c] != 0:
            continue
        d = dim[w, c]
        g1 = geom1[w, c]
        g2 = geom2[w, c]
        if ((d != 1 and d != 3 and d != 4 and d != 6) or
            g1 < 0 or g1 >= ngeom or g2 < 0 or g2 >= ngeom):
            status[w] = BAD_CONTACT
            return
        b1 = geom_body[g1]
        b2 = geom_body[g2]
        if b1 < 0 or b1 >= nbody or b2 < 0 or b2 >= nbody:
            status[w] = BAD_CONTACT
            return
        for k in range(3):
            if not wp.isfinite(pos[w, c, k]) or wp.abs(pos[w, c, k]) > SAFE_MAGNITUDE:
                status[w] = BAD_NUMERIC
                return
        for k in range(9):
            if not wp.isfinite(frame[w, c, k]) or wp.abs(frame[w, c, k]) > SAFE_MAGNITUDE:
                status[w] = BAD_NUMERIC
                return
        if (not wp.isfinite(dist[w, c]) or not wp.isfinite(margin[w, c]) or
            wp.abs(dist[w, c]) > SAFE_MAGNITUDE or
            wp.abs(margin[w, c]) > SAFE_MAGNITUDE):
            status[w] = BAD_NUMERIC
            return
        if nv > 0:
            total += d
    if total > maxrows or base > 2147483647 - total or base + total > cap:
        status[w] = CAPACITY
        return
    if total > 0:
        for j in range(nv):
            for k in range(6):
                if not wp.isfinite(cdof[w, j, k]) or wp.abs(cdof[w, j, k]) > SAFE_MAGNITUDE:
                    status[w] = BAD_NUMERIC
                    return
        for b in range(nbody):
            for k in range(3):
                if (not wp.isfinite(subtree[w, b, k]) or
                    wp.abs(subtree[w, b, k]) > SAFE_MAGNITUDE):
                    status[w] = BAD_NUMERIC
                    return
    local_count[w] = total


@wp.func
def _body_component(
    cdof: wp.array3d(dtype=scalar.wp_real), subtree: wp.array3d(dtype=scalar.wp_real),
    pos: wp.array3d(dtype=scalar.wp_real), roots: wp.array(dtype=wp.int32),
    ancestor: wp.array2d(dtype=wp.int32),
    w: int, c: int, body: int, j: int, component: int, rotation: int,
) -> scalar.wp_real:
    if ancestor[body, j] == 0:
        return scalar.wp_real(0.0)
    if rotation == 1:
        return cdof[w, j, component]
    root = roots[body]
    ox = pos[w, c, 0] - subtree[w, root, 0]
    oy = pos[w, c, 1] - subtree[w, root, 1]
    oz = pos[w, c, 2] - subtree[w, root, 2]
    ax = cdof[w, j, 0]
    ay = cdof[w, j, 1]
    az = cdof[w, j, 2]
    if component == 0:
        return cdof[w, j, 3] + (ay * oz - az * oy)
    if component == 1:
        return cdof[w, j, 4] + (az * ox - ax * oz)
    return cdof[w, j, 5] + (ax * oy - ay * ox)


@wp.kernel(enable_backward=False)
def _jacobian_rows(
    ncon: wp.array(dtype=wp.int32), active: wp.array(dtype=wp.int32),
    geom1: wp.array2d(dtype=wp.int32), geom2: wp.array2d(dtype=wp.int32),
    pos: wp.array3d(dtype=scalar.wp_real), frame: wp.array3d(dtype=scalar.wp_real),
    dim: wp.array2d(dtype=wp.int32), exclude: wp.array2d(dtype=wp.int32),
    cdof: wp.array3d(dtype=scalar.wp_real), subtree: wp.array3d(dtype=scalar.wp_real),
    geom_body: wp.array(dtype=wp.int32), roots: wp.array(dtype=wp.int32),
    ancestor: wp.array2d(dtype=wp.int32), rowstart: wp.array2d(dtype=wp.int32),
    status: wp.array(dtype=wp.int32), J: wp.array3d(dtype=scalar.wp_real),
    jac1p: wp.array4d(dtype=scalar.wp_real), jac2p: wp.array4d(dtype=scalar.wp_real),
    difp: wp.array4d(dtype=scalar.wp_real), jac1r: wp.array4d(dtype=scalar.wp_real),
    jac2r: wp.array4d(dtype=scalar.wp_real), difr: wp.array4d(dtype=scalar.wp_real),
    maxcon: int, nv: int,
):
    w, c, j = wp.tid()
    if status[w] != OK or active[w] != 1 or c >= ncon[w] or j >= nv or exclude[w, c] != 0:
        return
    b1 = geom_body[geom1[w, c]]
    b2 = geom_body[geom2[w, c]]
    d = dim[w, c]
    start = rowstart[w, c]
    for k in range(3):
        one_p = _body_component(cdof, subtree, pos, roots, ancestor,
                                w, c, b1, j, k, 0)
        two_p = _body_component(cdof, subtree, pos, roots, ancestor,
                                w, c, b2, j, k, 0)
        jac1p[w, c, k, j] = one_p
        jac2p[w, c, k, j] = two_p
        difp[w, c, k, j] = two_p - one_p
        if d > 3:
            one_r = _body_component(cdof, subtree, pos, roots, ancestor,
                                    w, c, b1, j, k, 1)
            two_r = _body_component(cdof, subtree, pos, roots, ancestor,
                                    w, c, b2, j, k, 1)
            jac1r[w, c, k, j] = one_r
            jac2r[w, c, k, j] = two_r
            difr[w, c, k, j] = two_r - one_r
    for row in range(6):
        if row >= d:
            break
        rotation = int(0)
        frame_row = row
        if row >= 3:
            rotation = 1
            frame_row = row - 3
        value = scalar.wp_real(0.0)
        for k in range(3):
            coeff = frame[w, c, frame_row * 3 + k]
            if coeff != scalar.wp_real(0.0):
                difference = difp[w, c, k, j]
                if rotation == 1:
                    difference = difr[w, c, k, j]
                value = value + coeff * difference
        J[w, start + row, j] = value


@wp.kernel(enable_backward=False)
def _metadata(
    ncon: wp.array(dtype=wp.int32), incoming: wp.array(dtype=wp.int32),
    active: wp.array(dtype=wp.int32), dim: wp.array2d(dtype=wp.int32),
    exclude: wp.array2d(dtype=wp.int32), dist: wp.array2d(dtype=scalar.wp_real),
    margin: wp.array2d(dtype=scalar.wp_real), rowstart: wp.array2d(dtype=wp.int32),
    local_count: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32),
    out_address: wp.array2d(dtype=wp.int32), out_exclude: wp.array2d(dtype=wp.int32),
    out_pos: wp.array2d(dtype=scalar.wp_real), out_margin: wp.array2d(dtype=scalar.wp_real),
    out_frictionloss: wp.array2d(dtype=scalar.wp_real),
    out_type: wp.array2d(dtype=wp.int32), out_id: wp.array2d(dtype=wp.int32),
    outgoing: wp.array(dtype=wp.int32), nv: int,
):
    w, c = wp.tid()
    if status[w] != OK or active[w] != 1:
        return
    if c == 0:
        outgoing[w] = incoming[w] + local_count[w]
    if c >= ncon[w] or exclude[w, c] != 0:
        return
    if nv == 0:
        out_address[w, c] = -1
        out_exclude[w, c] = 3
        return
    start = rowstart[w, c]
    out_address[w, c] = incoming[w] + start
    d = dim[w, c]
    kind = int(7)
    if d == 1:
        kind = 5
    for row in range(6):
        if row >= d:
            break
        r = start + row
        value = scalar.wp_real(0.0)
        gap = scalar.wp_real(0.0)
        if row == 0:
            value = dist[w, c]
            gap = margin[w, c]
        out_pos[w, r] = value
        out_margin[w, r] = gap
        out_frictionloss[w, r] = scalar.wp_real(0.0)
        out_type[w, r] = kind
        out_id[w, r] = c


class ContactWorkspace:
    """Preallocated resident buffers. upload/download are outside production."""

    INPUTS = ('ncon', 'incoming', 'capacity', 'active', 'geom1', 'geom2', 'pos',
              'frame', 'dist', 'margin', 'dim', 'exclude', 'cdof', 'subtree')
    OUTPUTS = ('outgoing', 'out_address', 'out_exclude', 'J', 'out_pos',
               'out_margin', 'out_frictionloss', 'out_type', 'out_id')

    def __init__(self, static: ContactStatic, nworld: int, maxcon: int,
                 maxrows: int, *, device='cuda:0'):
        if nworld <= 0 or maxcon <= 0 or maxrows <= 0:
            raise ValueError('positive N/C/R required')
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError('CUDA required')
        self.static = static
        self.nworld, self.maxcon, self.maxrows = nworld, maxcon, maxrows
        N, C, R, V, B = nworld, maxcon, maxrows, max(static.nv, 1), static.nbody
        F, I = scalar.wp_real, wp.int32
        self.geom_body = wp.array(static.arrays['geom_bodyid'], dtype=I, device=self.device)
        self.roots = wp.array(static.arrays['body_rootid'], dtype=I, device=self.device)
        self.ancestor = wp.array(static.ancestor_mask, dtype=I, device=self.device)
        shapes = {
            'ncon': (N,), 'incoming': (N,), 'capacity': (N,), 'active': (N,),
            'geom1': (N, C), 'geom2': (N, C), 'pos': (N, C, 3),
            'frame': (N, C, 9), 'dist': (N, C), 'margin': (N, C),
            'dim': (N, C), 'exclude': (N, C),
            'cdof': (N, V, 6), 'subtree': (N, B, 3),
            'rowstart': (N, C), 'local_count': (N,), 'status': (N,),
            'outgoing': (N,), 'out_address': (N, C), 'out_exclude': (N, C),
            'J': (N, R, V), 'out_pos': (N, R), 'out_margin': (N, R),
            'out_frictionloss': (N, R), 'out_type': (N, R), 'out_id': (N, R),
            'jac1p': (N, C, 3, V), 'jac2p': (N, C, 3, V),
            'difp': (N, C, 3, V), 'jac1r': (N, C, 3, V),
            'jac2r': (N, C, 3, V), 'difr': (N, C, 3, V),
        }
        floats = {'pos', 'frame', 'dist', 'margin', 'cdof', 'subtree',
                  'J', 'out_pos', 'out_margin', 'out_frictionloss',
                  'jac1p', 'jac2p', 'difp', 'jac1r', 'jac2r', 'difr'}
        self.shapes = shapes
        self.dtypes = {name: np.dtype('float64' if name in floats else 'int32')
                       for name in shapes}
        for name, shape in shapes.items():
            setattr(self, name, wp.zeros(shape, dtype=F if name in floats else I,
                                         device=self.device))

    def upload(self, **arrays):
        if set(arrays) != set(self.INPUTS + self.OUTPUTS):
            raise ValueError('complete input and seeded output set required')
        prepared = {}
        for name, value in arrays.items():
            raw = np.asarray(value)
            if raw.dtype != self.dtypes[name] or raw.shape != self.shapes[name]:
                raise ValueError(f'{name}: expected {self.shapes[name]} {self.dtypes[name]}, got {raw.shape} {raw.dtype}')
            prepared[name] = np.ascontiguousarray(raw)
        for name in self.INPUTS + self.OUTPUTS:
            getattr(self, name).assign(prepared[name])

    def launch(self, stream=None):
        """Enqueue the three real production kernels, with no host array access."""
        s = self.static
        wp.launch(_preflight, dim=self.nworld, inputs=[
            self.ncon, self.incoming, self.capacity, self.active,
            self.geom1, self.geom2, self.pos, self.frame, self.dist, self.margin,
            self.dim, self.exclude, self.cdof, self.subtree, self.geom_body,
            self.rowstart, self.local_count, self.status,
            self.maxcon, self.maxrows, s.nv, s.nbody, s.ngeom],
            device=self.device, stream=stream)
        wp.launch(_jacobian_rows, dim=(self.nworld, self.maxcon, max(s.nv, 1)), inputs=[
            self.ncon, self.active, self.geom1, self.geom2, self.pos, self.frame,
            self.dim, self.exclude, self.cdof, self.subtree, self.geom_body,
            self.roots, self.ancestor, self.rowstart, self.status, self.J,
            self.jac1p, self.jac2p, self.difp, self.jac1r, self.jac2r, self.difr,
            self.maxcon, s.nv], device=self.device, stream=stream)
        wp.launch(_metadata, dim=(self.nworld, self.maxcon), inputs=[
            self.ncon, self.incoming, self.active, self.dim, self.exclude,
            self.dist, self.margin, self.rowstart, self.local_count, self.status,
            self.out_address, self.out_exclude, self.out_pos, self.out_margin,
            self.out_frictionloss, self.out_type, self.out_id, self.outgoing, s.nv],
            device=self.device, stream=stream)

    def download(self):
        return {name: getattr(self, name).numpy() for name in self.OUTPUTS + ('status',)}

    def download_diagnostics(self):
        """Download stage nodes only outside the production trace."""
        return {name: getattr(self, name).numpy() for name in
                ('jac1p', 'jac2p', 'difp', 'jac1r', 'jac2r', 'difr')}
