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
                    # Keep a GPU pow node for the original mju_pow; its CPU/libm
                    # nonzero-neighborhood certificate remains unverified.
                    if x <= mid:
                        branch = 4
                        a = scalar.wp_real(1.0)/mid
                        y = a*wp.pow(x, power)
                        yP = power*a*x
                    else:
                        branch = 5
                        z = scalar.wp_real(1.0)-x
                        b = scalar.wp_real(1.0)/(scalar.wp_real(1.0)-mid)
                        y = scalar.wp_real(1.0)-b*wp.pow(z, power)
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

