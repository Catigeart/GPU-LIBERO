"""S02 fixed-MJB scalar constraint production kernel.

One CUDA thread owns one world and commits both local segments after checking
the complete required row count. The source ordering is serialized within that
thread. This adapter owns only its two segments; contact/equality rows belong
to later assembly.
"""
from __future__ import annotations
import libero.gpu.precision as scalar

import hashlib

import numpy as np
import warp as wp

wp.set_module_options({'enable_backward': False, 'fuse_fp': True, 'fast_math': False})

NV, NQ, NJ, NF_CAP, NL_CAP = 43, 48, 18, 9, 26
_FIXED_STATIC_SHA = {
    'dof_frictionloss': '4d0f6b2a64c9435b80fe32a0beff4cee9931b12207138a9d5295805afe6a9a58',
    'jnt_type': '4c0e0d380d48f2a6a7f66baa70f16e9bef2974dcffe93545a1da175cd5c9f2ce',
    'jnt_limited': '9fa4b56dad88df916bdb60474e25f570af8f4620424a9648f058b373cfbdc38f',
    'jnt_qposadr': '38ae4abac73ad2b845e1908b24e91cd00b1d3df40b5c6f885075fe46adb9ab35',
    'jnt_dofadr': '13c9448fa632713f92cc44c1c525c473e12ce7105bc7b23c30d8fcdf8b90a02b',
    'jnt_range': '043664b7a7266553f2d7d50e22bb94502e3dbca01c993b91831ea12dd98509fe',
    'jnt_margin': '81c611f35bff79491538b2f7cf201c7597a661a5c549633541c62bdc8af1613f',
}


@wp.kernel(enable_backward=False)
def _produce(qpos: wp.array(dtype=scalar.wp_real), active: wp.array(dtype=wp.int32),
             incoming: wp.array(dtype=wp.int32), capacity: wp.array(dtype=wp.int32),
             frictionloss: wp.array(dtype=scalar.wp_real), jnt_type: wp.array(dtype=wp.int32),
             limited: wp.array(dtype=wp.int32), qposadr: wp.array(dtype=wp.int32),
             dofadr: wp.array(dtype=wp.int32), ranges: wp.array(dtype=scalar.wp_real),
             margins: wp.array(dtype=scalar.wp_real),
             friction_J: wp.array(dtype=scalar.wp_real), friction_pos: wp.array(dtype=scalar.wp_real),
             friction_margin: wp.array(dtype=scalar.wp_real), friction_loss: wp.array(dtype=scalar.wp_real),
             friction_type: wp.array(dtype=wp.int32), friction_id: wp.array(dtype=wp.int32),
             friction_global_adr: wp.array(dtype=wp.int32),
             limit_J: wp.array(dtype=scalar.wp_real), limit_pos: wp.array(dtype=scalar.wp_real),
             limit_margin: wp.array(dtype=scalar.wp_real), limit_loss: wp.array(dtype=scalar.wp_real),
             limit_type: wp.array(dtype=wp.int32), limit_id: wp.array(dtype=wp.int32),
             limit_global_adr: wp.array(dtype=wp.int32),
             nf_delta: wp.array(dtype=wp.int32), nlimit_delta: wp.array(dtype=wp.int32),
             nefc_delta: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32)):
    w = wp.tid()
    # Status is sticky so a later inactive launch cannot hide the first error.
    if status[w] != 0:
        return
    if active[w] == 0:
        return
    if active[w] != 1:
        status[w] = 1
        return
    offset = incoming[w]
    cap = capacity[w]
    if offset < 0 or cap < 0 or offset > cap:
        status[w] = 2
        return
    for k in range(NQ):
        if not wp.isfinite(qpos[w * NQ + k]):
            status[w] = 3
            return

    nf = wp.int32(0)
    for i in range(NV):
        if frictionloss[i] > scalar.wp_real(0.0):
            nf += 1
    nl = wp.int32(0)
    for j in range(NJ):
        if limited[j] != 0:
            typ = jnt_type[j]
            if typ != 2 and typ != 3:
                status[w] = 4
                return
            value = qpos[w * NQ + qposadr[j]]
            margin = margins[j]
            lower = scalar.wp_real(-1.0) * (ranges[2 * j] - value)
            if lower < margin:
                nl += 1
            upper = scalar.wp_real(1.0) * (ranges[2 * j + 1] - value)
            if upper < margin:
                nl += 1
    need = nf + nl
    if nf > NF_CAP or nl > NL_CAP or need > cap - offset:
        status[w] = 5
        return

    fr = wp.int32(0)
    for i in range(NV):
        loss = frictionloss[i]
        if loss > scalar.wp_real(0.0):
            base = (w * NF_CAP + fr) * NV
            for k in range(NV):
                friction_J[base + k] = scalar.wp_real(0.0)
            friction_J[base + i] = scalar.wp_real(1.0)
            slot = w * NF_CAP + fr
            friction_pos[slot] = scalar.wp_real(0.0)
            friction_margin[slot] = scalar.wp_real(0.0)
            friction_loss[slot] = loss
            friction_type[slot] = 1
            friction_id[slot] = i
            friction_global_adr[slot] = offset + fr
            fr += 1

    lr = wp.int32(0)
    for j in range(NJ):
        if limited[j] != 0:
            value = qpos[w * NQ + qposadr[j]]
            margin = margins[j]
            adr = dofadr[j]
            # Literal source order: lower side=-1, then upper side=+1.
            lower = scalar.wp_real(-1.0) * (ranges[2 * j] - value)
            if lower < margin:
                base = (w * NL_CAP + lr) * NV
                for k in range(NV):
                    limit_J[base + k] = scalar.wp_real(0.0)
                limit_J[base + adr] = scalar.wp_real(1.0)
                slot = w * NL_CAP + lr
                limit_pos[slot] = lower
                limit_margin[slot] = margin
                limit_loss[slot] = scalar.wp_real(0.0)
                limit_type[slot] = 3
                limit_id[slot] = j
                limit_global_adr[slot] = offset + nf + lr
                lr += 1
            upper = scalar.wp_real(1.0) * (ranges[2 * j + 1] - value)
            if upper < margin:
                base = (w * NL_CAP + lr) * NV
                for k in range(NV):
                    limit_J[base + k] = scalar.wp_real(0.0)
                limit_J[base + adr] = scalar.wp_real(-1.0)
                slot = w * NL_CAP + lr
                limit_pos[slot] = upper
                limit_margin[slot] = margin
                limit_loss[slot] = scalar.wp_real(0.0)
                limit_type[slot] = 3
                limit_id[slot] = j
                limit_global_adr[slot] = offset + nf + lr
                lr += 1
    nf_delta[w] = fr
    nlimit_delta[w] = lr
    nefc_delta[w] = fr + lr


@wp.kernel(enable_backward=False)
def _probe_dist(qpos: wp.array(dtype=scalar.wp_real), qposadr: wp.array(dtype=wp.int32),
                ranges: wp.array(dtype=scalar.wp_real), joint: wp.array(dtype=wp.int32),
                side: wp.array(dtype=wp.int32), out: wp.array(dtype=scalar.wp_real)):
    """Diagnostic only: expose the identical source-order arithmetic at equality."""
    w = wp.tid()
    j = joint[w]
    value = qpos[w * NQ + qposadr[j]]
    if side[w] == -1:
        out[w] = scalar.wp_real(-1.0) * (ranges[2 * j] - value)
    else:
        out[w] = scalar.wp_real(1.0) * (ranges[2 * j + 1] - value)


def _require_array(value, shape, dtype, label):
    arr = np.asarray(value)
    if arr.shape != shape or arr.dtype != np.dtype(dtype):
        raise ValueError(f'{label} requires {shape}/{np.dtype(dtype)}, got {arr.shape}/{arr.dtype}')
    return np.ascontiguousarray(arr)


class ScalarWorkspace:
    """Preallocated N-world workspace; launch performs only the CUDA kernel."""

    _STATIC = {
        'dof_frictionloss': ((43,), np.float64),
        'jnt_type': ((18,), np.int32),
        'jnt_limited': ((18,), np.uint8),
        'jnt_qposadr': ((18,), np.int32),
        'jnt_dofadr': ((18,), np.int32),
        'jnt_range': ((18, 2), np.float64),
        'jnt_margin': ((18,), np.float64),
    }

    def __init__(self, nworld: int, static: dict, static_sha: dict, device='cuda:0'):
        if nworld < 1:
            raise ValueError('positive N world required')
        if set(static) != set(self._STATIC) or set(static_sha) != set(self._STATIC):
            raise ValueError('static model fields')
        if static_sha != _FIXED_STATIC_SHA:
            raise ValueError('static digest must identify frozen MJB scalar read set')
        checked = {}
        for key, (shape, dtype) in self._STATIC.items():
            arr = _require_array(static[key], shape, dtype, key)
            if hashlib.sha256(arr.tobytes()).hexdigest() != static_sha[key]:
                raise ValueError(f'fixed MJB static digest {key}')
            checked[key] = arr
        if (np.count_nonzero(checked['dof_frictionloss'] > 0) != 9 or
            np.count_nonzero(checked['jnt_limited']) != 13 or
            np.any(~np.isin(checked['jnt_type'][checked['jnt_limited'] != 0], [2, 3])) or
            np.any(~np.isfinite(checked['dof_frictionloss'])) or
            np.any(~np.isfinite(checked['jnt_range'])) or
            np.any(~np.isfinite(checked['jnt_margin'])) or
            np.any(checked['jnt_qposadr'] < 0) or np.any(checked['jnt_qposadr'] >= NQ) or
            np.any(checked['jnt_dofadr'] < 0) or np.any(checked['jnt_dofadr'] >= NV)):
            raise ValueError('unsupported fixed static model')
        self.nworld = nworld
        self.device = device
        self.static = {}
        for key, arr in checked.items():
            if key == 'jnt_limited':
                arr = arr.astype(np.int32)
            self.static[key] = wp.array(arr.ravel(), dtype=scalar.wp_real if arr.dtype == np.float64 else wp.int32, device=device)
        self.qpos = wp.zeros(nworld * NQ, dtype=scalar.wp_real, device=device)
        self.active = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.incoming = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.capacity = wp.zeros(nworld, dtype=wp.int32, device=device)
        for seg, rows in (('friction', NF_CAP), ('limit', NL_CAP)):
            for key in ('J', 'pos', 'margin', 'loss'):
                setattr(self, f'{seg}_{key}', wp.zeros(nworld * rows * (NV if key == 'J' else 1), dtype=scalar.wp_real, device=device))
            for key in ('type', 'id', 'global_adr'):
                setattr(self, f'{seg}_{key}', wp.zeros(nworld * rows, dtype=wp.int32, device=device))
        for key in ('nf_delta', 'nlimit_delta', 'nefc_delta', 'status'):
            setattr(self, key, wp.zeros(nworld, dtype=wp.int32, device=device))

    def upload_inputs(self, qpos, active, incoming, capacity):
        for key, value, shape, dtype in (
            ('qpos', qpos, (self.nworld, NQ), np.float64),
            ('active', active, (self.nworld,), np.int32),
            ('incoming', incoming, (self.nworld,), np.int32),
            ('capacity', capacity, (self.nworld,), np.int32)):
            arr = _require_array(value, shape, dtype, key)
            getattr(self, key).assign(arr.ravel())

    def upload_outputs(self, values: dict):
        """Prepare old output bytes outside production for preservation tests."""
        for key, value in values.items():
            if key in ('qpos', 'active', 'incoming', 'capacity') or not hasattr(self, key):
                raise ValueError(f'unknown output {key}')
            arr = np.asarray(value)
            target = getattr(self, key)
            dtype = np.int32 if key in ('nf_delta', 'nlimit_delta', 'nefc_delta', 'status') or key.endswith(('type', 'id', 'adr')) else np.float64
            if arr.dtype != dtype or arr.size != target.size:
                raise ValueError(f'output schema {key}')
            target.assign(np.ascontiguousarray(arr).ravel())

    def launch(self, stream=None):
        s = self.static
        wp.launch(_produce, dim=self.nworld, inputs=[self.qpos, self.active, self.incoming, self.capacity,
            s['dof_frictionloss'], s['jnt_type'], s['jnt_limited'], s['jnt_qposadr'],
            s['jnt_dofadr'], s['jnt_range'], s['jnt_margin'],
            self.friction_J, self.friction_pos, self.friction_margin, self.friction_loss,
            self.friction_type, self.friction_id, self.friction_global_adr,
            self.limit_J, self.limit_pos, self.limit_margin, self.limit_loss,
            self.limit_type, self.limit_id, self.limit_global_adr,
            self.nf_delta, self.nlimit_delta, self.nefc_delta, self.status],
            device=self.device, stream=stream)

    def download(self) -> dict:
        out = {}
        for seg, rows in (('friction', NF_CAP), ('limit', NL_CAP)):
            for key in ('J', 'pos', 'margin', 'loss', 'type', 'id', 'global_adr'):
                shape = (self.nworld, rows, NV) if key == 'J' else (self.nworld, rows)
                out[f'{seg}_{key}'] = getattr(self, f'{seg}_{key}').numpy().reshape(shape).copy()
        for key in ('nf_delta', 'nlimit_delta', 'nefc_delta', 'status'):
            out[key] = getattr(self, key).numpy().copy()
        return out
