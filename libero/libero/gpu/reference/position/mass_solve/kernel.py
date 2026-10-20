"""MuJoCo 2.3.7 sparse mj_factorM + mj_solveM, two frozen models.

Apache-2.0 source: src/engine/engine_core_smooth.c:1018-1212,
commit c9246e1f5006379d599e0bcddf159a8616d31441. The code below
transcribes the native loop order; it is not an external CPU dependency.
"""
import libero.gpu.precision as scalar

import hashlib
import numpy as np
import warp as wp

wp.set_module_options({'enable_backward': False, 'fuse_fp': True, 'fast_math': False})

_TOPOLOGIES = {
    (43, 153): {
        'nq': 48, 'nmocap': 0,
        'dof_Madr': '3e27466135207e6124bbf87bad323aa0795dd99f58cb135885cce724e1f7b27d',
        'dof_parentid': 'a57334b5f8e08a96647a0de0fc27eaac9e1fbfa1c76f007f5308360f2f4fd76a',
        'dof_simplenum': '58f218aeb259309eeb545345a8f5c715987967b0ec724806d448394da578d0b0',
    },
    (11, 36): {
        'nq': 13, 'nmocap': 1,
        'dof_Madr': '728df6bba5d83727b1e8d9202490986a9befa66ee3f946fbbdc9175f6f9c0290',
        'dof_parentid': '6f7d52dc4fa192cb00204477a83f005f629b8e9506a854b9adf7f485dd56a8f5',
        'dof_simplenum': '85759b3811ff7dc47b03792ac85317be51431a3f9e01dcafce317ed736a391b0',
    },
}


@wp.func
def _clear(w: int, nv: int, nm: int,
           ld: wp.array(dtype=scalar.wp_real, ndim=2),
           inv: wp.array(dtype=scalar.wp_real, ndim=2),
           sqrt_inv: wp.array(dtype=scalar.wp_real, ndim=2),
           solution: wp.array(dtype=scalar.wp_real, ndim=3),
           pivot: wp.array(dtype=wp.int32, ndim=2),
           warning_count: wp.array(dtype=wp.int32),
           warning_last: wp.array(dtype=wp.int32),
           preserve_events: int):
    for a in range(nm):
        ld[w, a] = scalar.wp_real(0.0)
    for i in range(nv):
        inv[w, i] = scalar.wp_real(0.0)
        sqrt_inv[w, i] = scalar.wp_real(0.0)
        if preserve_events == 0:
            pivot[w, i] = 0
        for r in range(4):
            solution[w, r, i] = scalar.wp_real(0.0)
    if preserve_events == 0:
        warning_count[w] = 0
        warning_last[w] = -1


@wp.kernel(enable_backward=False)
def _factor_solve(qm: wp.array(dtype=scalar.wp_real, ndim=2),
                  rhs: wp.array(dtype=scalar.wp_real, ndim=3),
                  count: wp.array(dtype=wp.int32),
                  active: wp.array(dtype=wp.int32),
                  madr: wp.array(dtype=wp.int32),
                  parent: wp.array(dtype=wp.int32),
                  ld: wp.array(dtype=scalar.wp_real, ndim=2),
                  inv: wp.array(dtype=scalar.wp_real, ndim=2),
                  sqrt_inv: wp.array(dtype=scalar.wp_real, ndim=2),
                  solution: wp.array(dtype=scalar.wp_real, ndim=3),
                  pivot: wp.array(dtype=wp.int32, ndim=2),
                  status: wp.array(dtype=wp.int32),
                  warning_count: wp.array(dtype=wp.int32),
                  warning_last: wp.array(dtype=wp.int32),
                  nv: int, nm: int):
    w = wp.tid()
    status[w] = 0
    _clear(w, nv, nm, ld, inv, sqrt_inv, solution, pivot,
           warning_count, warning_last, 0)
    if active[w] == 0:
        return
    if active[w] != 1:
        status[w] = 1
        return
    n = count[w]
    if n < 1 or n > 4:
        status[w] = 2
        return
    for a in range(nm):
        if not wp.isfinite(qm[w, a]):
            status[w] = 3
            return
    for r in range(4):
        for i in range(nv):
            if not wp.isfinite(rhs[w, r, i]):
                status[w] = 3
                return
    bad = wp.int32(0)
    for a in range(nm):
        ld[w, a] = qm[w, a]
    for k in range(nv - 1, -1, -1):
        kk = madr[k]
        if ld[w, kk] < scalar.wp_real(1.0e-15):
            pivot[w, k] = 1
            warning_count[w] = warning_count[w] + 1
            warning_last[w] = k
            ld[w, kk] = scalar.wp_real(1.0e-15)
        ki = kk + 1
        i = parent[k]
        while i >= 0:
            tmp = ld[w, ki] / ld[w, kk]
            if not wp.isfinite(tmp):
                bad = 1
            cnt = nm - madr[i]
            if i < nv - 1:
                cnt = madr[i + 1] - madr[i]
            for a in range(cnt):
                dest = madr[i] + a
                src = ki + a
                ld[w, dest] = ld[w, dest] + ld[w, src] * (-tmp)
                if not wp.isfinite(ld[w, dest]):
                    bad = 1
            ld[w, ki] = tmp
            ki += 1
            i = parent[i]
    for i in range(nv):
        diag = ld[w, madr[i]]
        inv[w, i] = scalar.wp_real(1.0) / diag
        sqrt_inv[w, i] = scalar.wp_real(1.0) / wp.sqrt(diag)
        if not wp.isfinite(inv[w, i]) or not wp.isfinite(sqrt_inv[w, i]):
            bad = 1
    for r in range(4):
        if r < n:
            for i in range(nv):
                solution[w, r, i] = rhs[w, r, i]
    if n == 1:
        for i in range(nv - 1, -1, -1):
            if solution[w, 0, i] != scalar.wp_real(0.0):
                a = madr[i] + 1
                j = parent[i]
                while j >= 0:
                    solution[w, 0, j] = solution[w, 0, j] - ld[w, a] * solution[w, 0, i]
                    if not wp.isfinite(solution[w, 0, j]):
                        bad = 1
                    a += 1
                    j = parent[j]
        for i in range(nv):
            solution[w, 0, i] = solution[w, 0, i] * inv[w, i]
            if not wp.isfinite(solution[w, 0, i]):
                bad = 1
        for i in range(nv):
            a = madr[i] + 1
            j = parent[i]
            while j >= 0:
                solution[w, 0, i] = solution[w, 0, i] - ld[w, a] * solution[w, 0, j]
                if not wp.isfinite(solution[w, 0, i]):
                    bad = 1
                a += 1
                j = parent[j]
    else:
        for i in range(nv - 1, -1, -1):
            a = madr[i] + 1
            j = parent[i]
            while j >= 0:
                for r in range(4):
                    if r < n:
                        tmp = solution[w, r, i]
                        if tmp != scalar.wp_real(0.0):
                            solution[w, r, j] = solution[w, r, j] - ld[w, a] * tmp
                            if not wp.isfinite(solution[w, r, j]):
                                bad = 1
                a += 1
                j = parent[j]
        for i in range(nv):
            for r in range(4):
                if r < n:
                    solution[w, r, i] = solution[w, r, i] * inv[w, i]
                    if not wp.isfinite(solution[w, r, i]):
                        bad = 1
        for i in range(nv):
            a = madr[i] + 1
            j = parent[i]
            while j >= 0:
                for r in range(4):
                    if r < n:
                        solution[w, r, i] = solution[w, r, i] - ld[w, a] * solution[w, r, j]
                        if not wp.isfinite(solution[w, r, i]):
                            bad = 1
                a += 1
                j = parent[j]
    if bad != 0:
        _clear(w, nv, nm, ld, inv, sqrt_inv, solution, pivot,
               warning_count, warning_last, 1)
        status[w] = 4
    elif warning_count[w] != 0:
        status[w] = 5


class MassSolveWorkspace:
    """Prepared CUDA workspace for either frozen sparse layout.

    status 0 completes the unmodified arithmetic path; 5 completes after
    native pivot repair and solves that modified system. 1/2/3 reject mask,
    count, and nonfinite input; 4 rejects nonfinite device arithmetic and
    clears every numerical output. status 0 alone does not certify SPD.
    """

    def __init__(self, model, capacity, device='cuda:0'):
        self.nv, self.nM = int(model.nv), int(model.nM)
        topology = _TOPOLOGIES.get((self.nv, self.nM))
        if topology is None or capacity <= 0:
            raise ValueError('unsupported model/capacity')
        for name in ('nq', 'nmocap'):
            if hasattr(model, name) and int(getattr(model, name)) != topology[name]:
                raise ValueError('unsupported model identity: '+name)
        for name in ('dof_Madr', 'dof_parentid', 'dof_simplenum'):
            raw = np.asarray(getattr(model, name))
            if raw.shape != (self.nv,) or raw.dtype.kind not in 'iu':
                raise TypeError('model integer schema: '+name)
            if any(int(x) < np.iinfo(np.int32).min or int(x) > np.iinfo(np.int32).max
                   for x in raw):
                raise ValueError('lossy model integer conversion: '+name)
            a = np.ascontiguousarray(raw, dtype=np.int32)
            if hashlib.sha256(a.tobytes()).hexdigest() != topology[name]:
                raise ValueError('unsupported model topology: '+name)
            setattr(self, name, wp.array(a, dtype=wp.int32, device=device))
        self.capacity, self.device = capacity, device
        self.qm = wp.zeros((capacity, self.nM), dtype=scalar.wp_real, device=device)
        self.rhs = wp.zeros((capacity, 4, self.nv), dtype=scalar.wp_real, device=device)
        self.count = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.active = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.ld = wp.zeros((capacity, self.nM), dtype=scalar.wp_real, device=device)
        self.inv = wp.zeros((capacity, self.nv), dtype=scalar.wp_real, device=device)
        self.sqrt_inv = wp.zeros((capacity, self.nv), dtype=scalar.wp_real, device=device)
        self.solution = wp.zeros((capacity, 4, self.nv), dtype=scalar.wp_real, device=device)
        self.pivot = wp.zeros((capacity, self.nv), dtype=wp.int32, device=device)
        self.status = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.warning_count = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.warning_last = wp.zeros(capacity, dtype=wp.int32, device=device)

    def launch(self, stream=None):
        wp.launch(_factor_solve, dim=self.capacity,
                  inputs=[self.qm, self.rhs, self.count, self.active,
                          self.dof_Madr, self.dof_parentid, self.ld,
                          self.inv, self.sqrt_inv, self.solution,
                          self.pivot, self.status, self.warning_count,
                          self.warning_last, self.nv, self.nM], device=self.device,
                  stream=stream)
