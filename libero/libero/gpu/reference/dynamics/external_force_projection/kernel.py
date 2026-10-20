"""D02 source-ordered xfrc projection; MuJoCo 2.3.7 engine_support.c.

Each CUDA thread owns one independent world, preserving the native body order,
zero-body skip, row order in mju_mulMatTVec, and force-before-torque additions.
The input boundary is xipos/subtree_com/cdof, never a CPU Jacobian or qforce.
"""
from __future__ import annotations
import libero.gpu.precision as scalar

import numpy as np
import warp as wp

wp.set_module_options({'enable_backward': False, 'fuse_fp': True})

STATUS_OK = 0
STATUS_ACTIVE = 1
STATUS_COUNT = 2
STATUS_BODY = 3
STATUS_SEED = 4
STATUS_XFRC = 5
STATUS_KINEMATIC = 6
STATUS_ARITHMETIC = 7


class StaticTopology:
    def __init__(self, fields: dict[str, np.ndarray]):
        needed = {'body_parentid', 'body_rootid', 'body_dofadr',
                  'body_dofnum', 'dof_parentid'}
        if set(fields) != needed:
            raise ValueError('static topology field set')
        self.fields = {}
        for name in needed:
            raw = np.asarray(fields[name])
            if raw.dtype != np.int32 or raw.ndim != 1:
                raise TypeError(f'{name}: expected original validated int32 vector')
            self.fields[name] = np.ascontiguousarray(raw)
        nbody = len(self.fields['body_parentid'])
        nv = len(self.fields['dof_parentid'])
        if nbody < 2 or nv < 1 or any(len(self.fields[name]) != nbody for name in
                                      ('body_rootid', 'body_dofadr', 'body_dofnum')):
            raise ValueError('static topology shape')
        parent = self.fields['body_parentid']
        root = self.fields['body_rootid']
        adr = self.fields['body_dofadr']
        num = self.fields['body_dofnum']
        dp = self.fields['dof_parentid']
        if parent[0] != 0 or num[0] != 0 or root[0] != 0:
            raise ValueError('world body topology')
        mask = np.zeros((nbody, nv), np.int32)
        for body in range(1, nbody):
            cursor = body
            seen = set()
            last = body
            while cursor:
                if cursor in seen or cursor < 0 or cursor >= nbody:
                    raise ValueError('body ancestor cycle/range')
                seen.add(cursor)
                last = cursor
                cursor = int(parent[cursor])
            if root[body] != last:
                raise ValueError('body root mismatch')
            cursor = body
            while cursor and num[cursor] == 0:
                cursor = int(parent[cursor])
            if not cursor:
                continue
            if num[cursor] < 0 or adr[cursor] < 0 or adr[cursor]+num[cursor] > nv:
                raise ValueError('body DOF range')
            dof = int(adr[cursor] + num[cursor] - 1)
            seen = set()
            while dof >= 0:
                if dof in seen or dof >= nv or dp[dof] < -1:
                    raise ValueError('DOF ancestor cycle/range')
                seen.add(dof)
                mask[body, dof] = 1
                dof = int(dp[dof])
        self.ancestor_mask = mask
        self.nbody = nbody
        self.nv = nv


@wp.kernel(enable_backward=False)
def _project(seed: wp.array(dtype=scalar.wp_real, ndim=2),
             xfrc: wp.array(dtype=scalar.wp_real, ndim=3),
             xipos: wp.array(dtype=scalar.wp_real, ndim=3),
             subtree_com: wp.array(dtype=scalar.wp_real, ndim=3),
             cdof: wp.array(dtype=scalar.wp_real, ndim=3),
             body_ids: wp.array(dtype=wp.int32, ndim=2),
             body_count: wp.array(dtype=wp.int32),
             active: wp.array(dtype=wp.int32),
             body_rootid: wp.array(dtype=wp.int32),
             ancestor: wp.array(dtype=wp.int32, ndim=2),
             scratch: wp.array(dtype=scalar.wp_real, ndim=2),
             stages: wp.array(dtype=scalar.wp_real, ndim=4),
             qfrc: wp.array(dtype=scalar.wp_real, ndim=2),
             status: wp.array(dtype=wp.int32),
             nbody: int, nv: int, diagnostic: int):
    w = wp.tid()
    status[w] = STATUS_OK
    if active[w] != 0 and active[w] != 1:
        status[w] = STATUS_ACTIVE
        return
    if active[w] == 0:
        return
    if body_count[w] != nbody-1:
        status[w] = STATUS_COUNT
        return
    for slot in range(nbody-1):
        if body_ids[w, slot] != slot+1:
            status[w] = STATUS_BODY
            return
    for j in range(nv):
        if not wp.isfinite(seed[w, j]):
            status[w] = STATUS_SEED
            return
    for body in range(1, nbody):
        for k in range(6):
            if not wp.isfinite(xfrc[w, body, k]):
                status[w] = STATUS_XFRC
                return
    for body in range(1, nbody):
        nonzero = False
        for k in range(6):
            if xfrc[w, body, k] != scalar.wp_real(0.0):
                nonzero = True
        if nonzero:
            root = body_rootid[body]
            for k in range(3):
                if not wp.isfinite(xipos[w, body, k]) or not wp.isfinite(subtree_com[w, root, k]):
                    status[w] = STATUS_KINEMATIC
                    return
            for j in range(nv):
                if ancestor[body, j] != 0:
                    for k in range(6):
                        if not wp.isfinite(cdof[w, j, k]):
                            status[w] = STATUS_KINEMATIC
                            return

    # The source loop leaves a zero-force target unchanged, including signed
    # zero.  Scratch is preallocated; qfrc is committed after all checks.
    for j in range(nv):
        scratch[w, j] = seed[w, j]
    for body in range(1, nbody):
        nonzero = False
        for k in range(6):
            if xfrc[w, body, k] != scalar.wp_real(0.0):
                nonzero = True
        if not nonzero:
            continue
        root = body_rootid[body]
        ox = xipos[w, body, 0] - subtree_com[w, root, 0]
        oy = xipos[w, body, 1] - subtree_com[w, root, 1]
        oz = xipos[w, body, 2] - subtree_com[w, root, 2]
        if not wp.isfinite(ox) or not wp.isfinite(oy) or not wp.isfinite(oz):
            status[w] = STATUS_ARITHMETIC
            return
        for j in range(nv):
            jp0 = scalar.wp_real(0.0)
            jp1 = scalar.wp_real(0.0)
            jp2 = scalar.wp_real(0.0)
            jr0 = scalar.wp_real(0.0)
            jr1 = scalar.wp_real(0.0)
            jr2 = scalar.wp_real(0.0)
            if ancestor[body, j] != 0:
                jr0 = cdof[w, j, 0]
                jr1 = cdof[w, j, 1]
                jr2 = cdof[w, j, 2]
                jp0 = cdof[w, j, 3] + (jr1*oz - jr2*oy)
                jp1 = cdof[w, j, 4] + (jr2*ox - jr0*oz)
                jp2 = cdof[w, j, 5] + (jr0*oy - jr1*ox)
                if not wp.isfinite(jp0) or not wp.isfinite(jp1) or not wp.isfinite(jp2):
                    status[w] = STATUS_ARITHMETIC
                    return
            if diagnostic != 0:
                stages[w, body, 0, j] = ox
                stages[w, body, 1, j] = oy
                stages[w, body, 2, j] = oz
                stages[w, body, 3, j] = jp0
                stages[w, body, 4, j] = jp1
                stages[w, body, 5, j] = jp2
                stages[w, body, 6, j] = jr0
                stages[w, body, 7, j] = jr1
                stages[w, body, 8, j] = jr2
            # mju_mulMatTVec zeros qforce and skips exactly-zero rows.  Its
            # row reductions are independent for each DOF, so DOF is outer.
            force = scalar.wp_real(0.0)
            if xfrc[w, body, 0] != scalar.wp_real(0.0):
                force = force + jp0*xfrc[w, body, 0]
            if xfrc[w, body, 1] != scalar.wp_real(0.0):
                force = force + jp1*xfrc[w, body, 1]
            if xfrc[w, body, 2] != scalar.wp_real(0.0):
                force = force + jp2*xfrc[w, body, 2]
            if not wp.isfinite(force):
                status[w] = STATUS_ARITHMETIC
                return
            scratch[w, j] = scratch[w, j] + force
            if not wp.isfinite(scratch[w, j]):
                status[w] = STATUS_ARITHMETIC
                return
            if diagnostic != 0:
                stages[w, body, 9, j] = force
                stages[w, body, 10, j] = scratch[w, j]
            torque = scalar.wp_real(0.0)
            if xfrc[w, body, 3] != scalar.wp_real(0.0):
                torque = torque + jr0*xfrc[w, body, 3]
            if xfrc[w, body, 4] != scalar.wp_real(0.0):
                torque = torque + jr1*xfrc[w, body, 4]
            if xfrc[w, body, 5] != scalar.wp_real(0.0):
                torque = torque + jr2*xfrc[w, body, 5]
            if not wp.isfinite(torque):
                status[w] = STATUS_ARITHMETIC
                return
            scratch[w, j] = scratch[w, j] + torque
            if not wp.isfinite(scratch[w, j]):
                status[w] = STATUS_ARITHMETIC
                return
            if diagnostic != 0:
                stages[w, body, 11, j] = torque
                stages[w, body, 12, j] = scratch[w, j]
    for j in range(nv):
        qfrc[w, j] = scratch[w, j]


class ExternalForceWorkspace:
    """Prepared static topology and preallocated CUDA buffers for N worlds."""

    def __init__(self, topology: StaticTopology, nworld: int, *, device='cuda:0', diagnostic=False):
        if nworld <= 0:
            raise ValueError('N must be positive')
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError('CUDA required')
        self.topology = topology
        self.nworld = nworld
        self.nbody = topology.nbody
        self.nv = topology.nv
        self.diagnostic = bool(diagnostic)
        N, B, V = nworld, self.nbody, self.nv
        F, I = scalar.wp_real, wp.int32
        self.body_rootid = wp.array(topology.fields['body_rootid'], dtype=I, device=self.device)
        self.ancestor_mask = wp.array(topology.ancestor_mask, dtype=I, device=self.device)
        self.seed_qfrc = wp.empty((N, V), dtype=F, device=self.device)
        self.xfrc_applied = wp.empty((N, B, 6), dtype=F, device=self.device)
        self.xipos = wp.empty((N, B, 3), dtype=F, device=self.device)
        self.subtree_com = wp.empty((N, B, 3), dtype=F, device=self.device)
        self.cdof = wp.empty((N, V, 6), dtype=F, device=self.device)
        self.body_ids = wp.empty((N, B-1), dtype=I, device=self.device)
        self.body_count = wp.empty(N, dtype=I, device=self.device)
        self.active_world = wp.empty(N, dtype=I, device=self.device)
        self.qfrc = wp.empty((N, V), dtype=F, device=self.device)
        self.scratch = wp.empty((N, V), dtype=F, device=self.device)
        self.stages = wp.empty((N, B, 13, V) if self.diagnostic else (N, 1, 1, 1),
                               dtype=F, device=self.device)
        self.status = wp.empty(N, dtype=I, device=self.device)

    def upload(self, **arrays):
        """Strict transfer boundary. Call only before production/capture/replay."""
        for name, value in arrays.items():
            if name not in ('seed_qfrc', 'xfrc_applied', 'xipos', 'subtree_com',
                            'cdof', 'body_ids', 'body_count', 'active_world', 'qfrc'):
                raise KeyError(name)
            target = getattr(self, name)
            raw = np.asarray(value)
            dtype = np.dtype('int32') if name in ('body_ids', 'body_count', 'active_world') else np.dtype('float64')
            if raw.dtype != dtype or raw.shape != target.shape:
                raise TypeError(f'{name}: expected {target.shape}/{dtype}, got {raw.shape}/{raw.dtype}')
            target.assign(np.ascontiguousarray(raw))

    def launch(self, *, stream=None):
        """One production kernel, with no allocation, host read or transfers."""
        wp.launch(_project, dim=self.nworld, inputs=[
            self.seed_qfrc, self.xfrc_applied, self.xipos, self.subtree_com,
            self.cdof, self.body_ids, self.body_count, self.active_world,
            self.body_rootid, self.ancestor_mask, self.scratch, self.stages,
            self.qfrc, self.status, self.nbody, self.nv, int(self.diagnostic)],
            device=self.device, stream=stream)

    def download(self):
        return {'qfrc': self.qfrc.numpy(), 'status': self.status.numpy()}

    def download_stages(self):
        if not self.diagnostic:
            raise ValueError('diagnostic buffer was not prepared')
        return self.stages.numpy()
