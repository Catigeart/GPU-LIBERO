"""Device-only MuJoCo 2.3.7 mj_makeImpedance for fixed-model row types.

One thread owns each world, preserving the original three ordered passes and
avoiding races between a contact normal row and its friction rows.  Type 1/3/5/7
are supported; the pinned model has no equality, tendon, or pyramid rows.
"""
from __future__ import annotations
import libero.gpu.precision as scalar

import numpy as np
import warp as wp
import hashlib

wp.set_module_options({'enable_backward': False, 'fuse_fp': True, 'fast_math': False})

# SHA256 of the four original-model float64 arrays in sealed static_model.npz.
# Local diagnostic cases vary contact rows, not these model arrays.
FIXED_STATIC_SHA256 = {
    'dof_solref': '3a51b5cfd4ed59fc256e10ff95668f53808d6c3d6285e69b85f45477b6668d50',
    'dof_solimp': '3d5ff332a4210168e5ee911f9d443f5019371006905232fa018f1184fff56687',
    'jnt_solref': 'fd23518305cd031f74079325f4349194a1cec0fda80de3594c9c303455377a9c',
    'jnt_solimp': 'e27ba3e0e41de43a80a5399be5db810b36b8062dc87e6ca71f05cb146f3f91be',
}


@wp.kernel(enable_backward=False)
def _impedance(
    count: wp.array(dtype=wp.int32), ncon: wp.array(dtype=wp.int32),
    ne: wp.array(dtype=wp.int32), nf: wp.array(dtype=wp.int32), active: wp.array(dtype=wp.int32),
    typ: wp.array(dtype=wp.int32), eid: wp.array(dtype=wp.int32),
    pos: wp.array(dtype=scalar.wp_real), margin: wp.array(dtype=scalar.wp_real),
    prediag: wp.array(dtype=scalar.wp_real),
    cdim: wp.array(dtype=wp.int32), cref: wp.array(dtype=scalar.wp_real),
    creff: wp.array(dtype=scalar.wp_real), cimp: wp.array(dtype=scalar.wp_real),
    cfric: wp.array(dtype=scalar.wp_real),
    dof_ref: wp.array(dtype=scalar.wp_real), dof_imp: wp.array(dtype=scalar.wp_real),
    jnt_ref: wp.array(dtype=scalar.wp_real), jnt_imp: wp.array(dtype=scalar.wp_real),
    R: wp.array(dtype=scalar.wp_real), D: wp.array(dtype=scalar.wp_real),
    KBIP: wp.array(dtype=scalar.wp_real), post: wp.array(dtype=scalar.wp_real),
    mu: wp.array(dtype=scalar.wp_real), route: wp.array(dtype=wp.int32),
    status: wp.array(dtype=wp.int32),
    rows: int, contacts: int, nv: int, njnt: int,
    timestep: scalar.wp_real, impratio: scalar.wp_real, refsafe: int):
    w = wp.tid()
    status[w] = 0
    if active[w] == 0:
        return
    if active[w] != 1:
        status[w] = 1
        return
    n = count[w]; nc = ncon[w]
    if n < 0 or n > rows or nc < 0 or nc > contacts or ne[w] < 0 or nf[w] < 0 or ne[w] + nf[w] > n:
        status[w] = 2
        return
    rb = w * rows; cb = w * contacts
    # Validate full blocks before any numeric output is written.  A bad later
    # row cannot turn an earlier partial update into a successful world.
    i = wp.int32(0); boundary = wp.int32(0)
    while i < n:
        if i == ne[w]+nf[w]: boundary = 1
        t = typ[rb+i]; id = eid[rb+i]; dim = wp.int32(1)
        if t == 1:
            if id < 0 or id >= nv: status[w] = 3; return
        elif t == 3:
            if id < 0 or id >= njnt: status[w] = 3; return
        elif t == 5 or t == 7:
            if id < 0 or id >= nc: status[w] = 3; return
            if cdim[cb+id] < 1 or cdim[cb+id] > 6: status[w] = 4; return
            if t == 7 and cdim[cb+id] < 2: status[w] = 4; return
            if t == 7: dim = cdim[cb+id]
        else:
            status[w] = 3; return
        if i + dim > n: status[w] = 4; return
        for j in range(dim):
            q = rb+i+j
            if typ[q] != t or eid[q] != id or not wp.isfinite(prediag[q]) or not wp.isfinite(pos[q]) or not wp.isfinite(margin[q]):
                status[w] = 4; return
        if t == 1:
            for j in range(2):
                if not wp.isfinite(dof_ref[id*2+j]): status[w] = 5; return
            for j in range(5):
                if not wp.isfinite(dof_imp[id*5+j]): status[w] = 5; return
            pwr = dof_imp[id*5+4]
            if pwr > scalar.wp_real(1.0) and pwr != scalar.wp_real(2.0):status[w] = 6;return
        elif t == 3:
            for j in range(2):
                if not wp.isfinite(jnt_ref[id*2+j]): status[w] = 5; return
            for j in range(5):
                if not wp.isfinite(jnt_imp[id*5+j]): status[w] = 5; return
            pwr = jnt_imp[id*5+4]
            if pwr > scalar.wp_real(1.0) and pwr != scalar.wp_real(2.0):status[w] = 6;return
        else:
            for j in range(2):
                if not wp.isfinite(cref[(cb+id)*2+j]) or not wp.isfinite(creff[(cb+id)*2+j]):
                    status[w] = 5; return
            for j in range(5):
                if not wp.isfinite(cimp[(cb+id)*5+j]) or not wp.isfinite(cfric[(cb+id)*5+j]):
                    status[w] = 5; return
            pwr = cimp[(cb+id)*5+4]
            if pwr > scalar.wp_real(1.0) and pwr != scalar.wp_real(2.0):status[w] = 6;return
            if t == 7:
                for j in range(1, dim-1):
                    if cfric[(cb+id)*5+j] == scalar.wp_real(0.0):status[w] = 5;return
        i += dim
    if ne[w]+nf[w] == n: boundary = 1
    if boundary == 0: status[w] = 4; return
    # Original pass 1: solparam -> pos/dim -> impedance -> R and KBIP.
    i = wp.int32(0)
    while i < n:
        t = typ[rb+i]; id = eid[rb+i]; dim = wp.int32(1)
        r0 = scalar.wp_real(0.0); r1 = scalar.wp_real(0.0)
        f0 = scalar.wp_real(0.0); f1 = scalar.wp_real(0.0)
        p0 = scalar.wp_real(0.0); p1 = scalar.wp_real(0.0); width = scalar.wp_real(0.0)
        mid = scalar.wp_real(0.0); power = scalar.wp_real(0.0)
        if t == 1:
            r0 = dof_ref[id*2]; r1 = dof_ref[id*2+1]
            p0 = dof_imp[id*5]; p1 = dof_imp[id*5+1]
            width = dof_imp[id*5+2]; mid = dof_imp[id*5+3]; power = dof_imp[id*5+4]
        elif t == 3:
            r0 = jnt_ref[id*2]; r1 = jnt_ref[id*2+1]
            p0 = jnt_imp[id*5]; p1 = jnt_imp[id*5+1]
            width = jnt_imp[id*5+2]; mid = jnt_imp[id*5+3]; power = jnt_imp[id*5+4]
        else:
            r0 = cref[(cb+id)*2]; r1 = cref[(cb+id)*2+1]
            f0 = creff[(cb+id)*2]; f1 = creff[(cb+id)*2+1]
            p0 = cimp[(cb+id)*5]; p1 = cimp[(cb+id)*5+1]
            width = cimp[(cb+id)*5+2]; mid = cimp[(cb+id)*5+3]; power = cimp[(cb+id)*5+4]
            if t == 7: dim = cdim[cb+id]
        # getsolparam has two format checks, refsafe, then the five clamps.
        if (r0 > scalar.wp_real(0.0)) != (r1 > scalar.wp_real(0.0)):
            r0 = scalar.wp_real(0.02); r1 = scalar.wp_real(1.0)
        if refsafe != 0 and r0 > scalar.wp_real(0.0): r0 = wp.max(r0, scalar.wp_real(2.0)*timestep)
        if (f0 > scalar.wp_real(0.0)) != (f1 > scalar.wp_real(0.0)):
            f0 = scalar.wp_real(0.0); f1 = scalar.wp_real(0.0)
        if refsafe != 0 and f0 > scalar.wp_real(0.0): f0 = wp.max(f0, scalar.wp_real(2.0)*timestep)
        p0 = wp.min(scalar.wp_real(0.9999), wp.max(scalar.wp_real(0.0001), p0))
        p1 = wp.min(scalar.wp_real(0.9999), wp.max(scalar.wp_real(0.0001), p1))
        width = wp.max(scalar.wp_real(0.0), width)
        mid = wp.min(scalar.wp_real(0.9999), wp.max(scalar.wp_real(0.0001), mid))
        power = wp.max(scalar.wp_real(1.0), power)
        imp = scalar.wp_real(0.0); impP = scalar.wp_real(0.0); branch = wp.int32(0)
        if p0 == p1 or width <= scalar.wp_real(1.0e-15):
            imp = scalar.wp_real(0.5)*(p0+p1)
        else:
            x = (pos[rb+i]-margin[rb+i])/width
            sgn = scalar.wp_real(1.0)
            if x < scalar.wp_real(0.0): x = -x; sgn = scalar.wp_real(-1.0)
            if x >= scalar.wp_real(1.0) or x <= scalar.wp_real(0.0):
                if x >= scalar.wp_real(1.0): imp = p1; branch = 1
                else: imp = p0; branch = 2
            else:
                y = scalar.wp_real(0.0); yP = scalar.wp_real(0.0)
                if power == scalar.wp_real(1.0):
                    y = x; yP = scalar.wp_real(1.0); branch = 3
                elif power == scalar.wp_real(2.0):
                    # Fixed model mju_pow operands are exponent 1 or 2.
                    # The frozen original-C probe establishes output I/P
                    # equality, but not internal pow-node equality everywhere.
                    if x <= mid:
                        branch = 4
                        a = scalar.wp_real(1.0)/mid
                        y = a*(x*x)
                        yP = power*a*x
                    else:
                        branch = 5
                        z = scalar.wp_real(1.0)-x
                        b = scalar.wp_real(1.0)/(scalar.wp_real(1.0)-mid)
                        y = scalar.wp_real(1.0)-b*(z*z)
                        yP = power*b*z
                else:
                    status[w] = 6; return
                imp = p0+y*(p1-p0)
                impP = yP*sgn*(p1-p0)/width
        for j in range(dim):
            q = rb+i+j; tp = typ[q]
            route[q] = branch
            v = (scalar.wp_real(1.0)-imp)*prediag[q]/imp
            R[q] = wp.max(scalar.wp_real(1.0e-15), v)
            ellip_friction = (tp == 7 and j > 0)
            rr0 = r0; rr1 = r1
            if ellip_friction and (f0 != scalar.wp_real(0.0) or f1 != scalar.wp_real(0.0)):
                rr0 = f0; rr1 = f1
            k = scalar.wp_real(0.0)
            if tp == 1 or ellip_friction:
                k = scalar.wp_real(0.0)
            elif rr0 > scalar.wp_real(0.0):
                den = p1*p1*rr0*rr0*rr1*rr1
                k = scalar.wp_real(1.0)/wp.max(scalar.wp_real(1.0e-15), den)
            else:
                k = -rr0/wp.max(scalar.wp_real(1.0e-15), p1*p1)
            bval = scalar.wp_real(0.0)
            if rr1 > scalar.wp_real(0.0):
                bval = scalar.wp_real(2.0)/wp.max(scalar.wp_real(1.0e-15), p1*rr0)
            else:
                bval = -rr1/wp.max(scalar.wp_real(1.0e-15), p1)
            KBIP[q*4] = k; KBIP[q*4+1] = bval; KBIP[q*4+2] = imp; KBIP[q*4+3] = impP
        i += dim
    # Original pass 2: frictional contacts adjust R and write contact.mu.
    i = ne[w]+nf[w]
    while i < n:
        t = typ[rb+i]
        if t == 7:
            id = eid[rb+i]; dim = cdim[cb+id]
            if dim < 2 or i+dim > n: status[w] = 4; return
            q = rb+i
            R[q+1] = R[q]/wp.max(scalar.wp_real(1.0e-15), impratio)
            fbase = (cb+id)*5
            muf = cfric[fbase]*wp.sqrt(R[q+1]/R[q])
            mu[cb+id] = muf
            for j in range(1, dim-1):
                fj = cfric[fbase+j]
                R[q+j+1] = R[q+1]*cfric[fbase]*cfric[fbase]/(fj*fj)
            i += dim
        else:
            i += 1
    # Original passes 3 and 4.  Preserve all inactive and unused capacity.
    for j in range(n):
        q = rb+j
        D[q] = scalar.wp_real(1.0)/R[q]
    for j in range(n):
        q = rb+j; imp = KBIP[q*4+2]
        post[q] = R[q]*imp/(scalar.wp_real(1.0)-imp)


def _assign(dst, value, shape, dtype):
    a = np.asarray(value)
    if a.shape != shape or a.dtype != np.dtype(dtype):
        raise ValueError(f'input schema {a.shape}/{a.dtype} expected {shape}/{np.dtype(dtype)}')
    dst.assign(np.ascontiguousarray(a).ravel())


class ImpedanceWorkspace:
    """Preallocated resident buffers.  upload() is preparation; launch() is production."""

    def __init__(self, worlds, rows, contacts, static, device='cuda:0', option_domain='fixed_model'):
        if worlds < 1 or rows < 1 or contacts < 1:
            raise ValueError('positive capacities required')
        if option_domain not in ('fixed_model', 'local_diagnostic'):
            raise ValueError('static option domain')
        expect = dict(dof_solref=(43, 2), dof_solimp=(43, 5), jnt_solref=(18, 2), jnt_solimp=(18, 5))
        prepared = {}
        for key, shape in expect.items():
            raw = static[key]
            if not isinstance(raw, np.ndarray): raise ValueError('static schema '+key)
            a = np.asarray(raw)
            if a.shape != shape or a.dtype != np.float64: raise ValueError('static schema '+key)
            if not np.isfinite(a).all(): raise ValueError('static domain '+key)
            prepared[key] = np.ascontiguousarray(a).copy()
            if hashlib.sha256(prepared[key].tobytes()).hexdigest() != FIXED_STATIC_SHA256[key]:
                raise ValueError('fixed model source '+key)
        scalars = {}
        for key, dtype in (('timestep', np.float64), ('impratio', np.float64), ('disableflags', np.int32)):
            raw = static[key]
            if not isinstance(raw, (np.ndarray, np.generic)):
                raise ValueError('static schema '+key)
            a = np.asarray(raw)
            if a.shape != () or a.dtype != np.dtype(dtype):
                raise ValueError('static schema '+key)
            scalars[key] = a.item()
        timestep, impratio, flags = scalars['timestep'], scalars['impratio'], scalars['disableflags']
        if not np.isfinite(timestep) or not np.isfinite(impratio) or timestep <= 0 or impratio <= 0:
            raise ValueError('static option domain')
        if option_domain == 'fixed_model':
            if timestep != 0.002 or impratio != 20.0 or flags != 0:
                raise ValueError('fixed option domain')
        elif flags not in (0, 1 << 11):
            raise ValueError('local diagnostic option domain')
        self.worlds, self.rows, self.contacts, self.device = worlds, rows, contacts, device
        self.nv, self.njnt = 43, 18
        self.option_domain = option_domain
        self.static = {key: wp.array(a.ravel(), dtype=scalar.wp_real, device=device)
                       for key, a in prepared.items()}
        self.timestep = timestep
        self.impratio = impratio
        self.disableflags = flags
        self.refsafe = flags & (1 << 11) == 0
        self.last_launch_scalars = None
        for key, size, dtype in (
            ('count', worlds, wp.int32), ('ncon', worlds, wp.int32), ('ne', worlds, wp.int32),
            ('nf', worlds, wp.int32), ('active', worlds, wp.int32), ('status', worlds, wp.int32),
            ('typ', worlds*rows, wp.int32), ('eid', worlds*rows, wp.int32),
            ('pos', worlds*rows, scalar.wp_real), ('margin', worlds*rows, scalar.wp_real),
            ('prediag', worlds*rows, scalar.wp_real), ('cdim', worlds*contacts, wp.int32),
            ('cref', worlds*contacts*2, scalar.wp_real), ('creff', worlds*contacts*2, scalar.wp_real),
            ('cimp', worlds*contacts*5, scalar.wp_real), ('cfric', worlds*contacts*5, scalar.wp_real),
            ('R', worlds*rows, scalar.wp_real), ('D', worlds*rows, scalar.wp_real),
            ('KBIP', worlds*rows*4, scalar.wp_real), ('post', worlds*rows, scalar.wp_real),
            ('mu', worlds*contacts, scalar.wp_real), ('route', worlds*rows, wp.int32)):
            setattr(self, key, wp.zeros(size, dtype=dtype, device=device))

    def upload(self, data, old=-97.5, active=None):
        n = self.worlds; r = self.rows; c = self.contacts
        if active is None: active = np.ones(n, np.int32)
        fields = (
            ('count', data['nefc'], (n,), np.int32), ('ncon', data['ncon'], (n,), np.int32),
            ('ne', data['ne'], (n,), np.int32), ('nf', data['nf'], (n,), np.int32),
            ('active', active, (n,), np.int32),
            ('typ', data['et'], (n,r), np.int32), ('eid', data['eid'], (n,r), np.int32),
            ('pos', data['epos'], (n,r), np.float64), ('margin', data['margin'], (n,r), np.float64),
            ('prediag', data['prediag'], (n,r), np.float64), ('cdim', data['c_dim'], (n,c), np.int32),
            ('cref', data['c_solref'], (n,c,2), np.float64),
            ('creff', data['c_reff'], (n,c,2), np.float64),
            ('cimp', data['c_solimp'], (n,c,5), np.float64),
            ('cfric', data['c_friction'], (n,c,5), np.float64),
            ('mu', data.get('mu_seed', data['mu']), (n,c), np.float64))
        for key, value, shape, dtype in fields: _assign(getattr(self, key), value, shape, dtype)
        for key, shape in (('R', (n,r)), ('D', (n,r)), ('KBIP', (n,r,4)), ('post', (n,r))):
            _assign(getattr(self, key), np.full(shape, old, np.float64), shape, np.float64)
        _assign(self.route, np.full((n,r), -99, np.int32), (n,r), np.int32)

    def launch(self, stream=None):
        s = self.static
        # These exact by-value operands are passed to Warp.  The verifier records
        # them outside the hot launch, including the values captured by a graph.
        timestep = scalar.wp_real(self.timestep)
        impratio = scalar.wp_real(self.impratio)
        refsafe = int(self.refsafe)
        self.last_launch_scalars = dict(timestep=np.asarray(self.timestep, np.float64),
                                        impratio=np.asarray(self.impratio, np.float64),
                                        disableflags=np.asarray(self.disableflags, np.int32),
                                        refsafe=np.asarray(refsafe, np.int32))
        wp.launch(_impedance, dim=self.worlds, inputs=[
            self.count, self.ncon, self.ne, self.nf, self.active,
            self.typ, self.eid, self.pos, self.margin, self.prediag,
            self.cdim, self.cref, self.creff, self.cimp, self.cfric,
            s['dof_solref'], s['dof_solimp'], s['jnt_solref'], s['jnt_solimp'],
            self.R, self.D, self.KBIP, self.post, self.mu, self.route, self.status,
            self.rows, self.contacts, self.nv, self.njnt,
            timestep, impratio, refsafe],
            device=self.device, stream=stream)
