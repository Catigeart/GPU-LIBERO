"""MuJoCo 2.3.7 mj_crbSkip(m,d,1), source c9246e1, Apache-2.0.

Arithmetic follows engine_core_smooth.c:961-1013, engine_util_spatial.c:406-416,
and the AVX-compatible mju_dot reduction in engine_util_blas.c:631-685.
"""
import libero.gpu.precision as scalar

from dataclasses import dataclass

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False, "fuse_fp": True, "fast_math": False})


@dataclass(frozen=True)
class CRBModel:
    nbody: int
    nv: int
    nM: int
    body_parent: object
    dof_body: object
    dof_parent: object
    dof_madr: object
    dof_simple: object
    dof_m0: object
    dof_armature: object

    @classmethod
    def prepare(cls, model, device="cuda"):
        def dimension(name):
            raw = np.asarray(getattr(model, name))
            if raw.shape != () or not np.issubdtype(raw.dtype, np.integer):
                raise ValueError(f"{name}: invalid scalar integer dtype/shape")
            value = int(raw)
            if value < 0 or value >= (1 << 31):
                raise ValueError(f"{name}: outside int32")
            return value
        nb, nv, nm = (dimension(name) for name in ("nbody", "nv", "nM"))
        if nb < 1 or nv < 0 or nm < nv:
            raise ValueError("invalid dimensions")
        def integer(name, size):
            a = np.asarray(getattr(model, name))
            if a.shape != (size,) or not np.issubdtype(a.dtype, np.integer):
                raise ValueError(f"{name}: invalid shape/dtype")
            if np.any(a < -(1 << 31)) or np.any(a >= (1 << 31)):
                raise ValueError(f"{name}: outside int32")
            return a.astype(np.int32)
        def number(name, size):
            a = np.asarray(getattr(model, name))
            if a.shape != (size,) or a.dtype != np.float64 or not np.all(np.isfinite(a)):
                raise ValueError(f"{name}: invalid shape/dtype/value")
            return np.ascontiguousarray(a)
        bp = integer("body_parentid", nb)
        db = integer("dof_bodyid", nv)
        dp = integer("dof_parentid", nv)
        ma = integer("dof_Madr", nv)
        si = integer("dof_simplenum", nv)
        m0 = number("dof_M0", nv)
        arm = number("dof_armature", nv)
        if bp[0] != 0 or np.any(bp[1:] < 0) or np.any(bp[1:] >= np.arange(1, nb)):
            raise ValueError("invalid body topology")
        if np.any(db <= 0) or np.any(db >= nb) or np.any(si < 0):
            raise ValueError("invalid dof body/simple")
        if nv:
            if ma[0] != 0 or np.any(ma[1:] <= ma[:-1]) or ma[-1] >= nm:
                raise ValueError("invalid sparse row starts")
        for i in range(nv):
            if dp[i] >= i or dp[i] < -1:
                raise ValueError("dof parent must precede child")
            count = 0
            j = i
            while j >= 0:
                count += 1
                j = int(dp[j])
            end = ma[i + 1] if i + 1 < nv else nm
            if end - ma[i] != count:
                raise ValueError("sparse row length differs from ancestor chain")
        return cls(nb, nv, nm,
                   wp.array(bp, dtype=wp.int32, device=device),
                   wp.array(db, dtype=wp.int32, device=device),
                   wp.array(dp, dtype=wp.int32, device=device),
                   wp.array(ma, dtype=wp.int32, device=device),
                   wp.array(si, dtype=wp.int32, device=device),
                   wp.array(m0, dtype=scalar.wp_real, device=device),
                   wp.array(arm, dtype=scalar.wp_real, device=device))


@wp.func
def _dot6(cdof: wp.array2d(dtype=scalar.wp_real), w: int, j: int,
          t0: scalar.wp_real, t1: scalar.wp_real, t2: scalar.wp_real,
          t3: scalar.wp_real, t4: scalar.wp_real, t5: scalar.wp_real) -> scalar.wp_real:
    p0 = cdof[w, 6*j] * t0
    p1 = cdof[w, 6*j+1] * t1
    p2 = cdof[w, 6*j+2] * t2
    p3 = cdof[w, 6*j+3] * t3
    p4 = cdof[w, 6*j+4] * t4
    p5 = cdof[w, 6*j+5] * t5
    return ((p0+p2)+(p1+p3))+(p4+p5)


@wp.kernel
def _crb(active: wp.array(dtype=wp.int32),
         cinert: wp.array2d(dtype=scalar.wp_real),
         cdof: wp.array2d(dtype=scalar.wp_real),
         body_parent: wp.array(dtype=wp.int32),
         dof_body: wp.array(dtype=wp.int32),
         dof_parent: wp.array(dtype=wp.int32),
         dof_madr: wp.array(dtype=wp.int32),
         dof_simple: wp.array(dtype=wp.int32),
         dof_m0: wp.array(dtype=scalar.wp_real),
         dof_armature: wp.array(dtype=scalar.wp_real),
         nbody: int, nv: int, nm: int,
         crb: wp.array2d(dtype=scalar.wp_real),
         qm: wp.array2d(dtype=scalar.wp_real),
         status: wp.array(dtype=wp.int32)):
    w = wp.tid()
    status[w] = 0
    for k in range(nbody*10):
        crb[w, k] = scalar.wp_real(0.0)
    for k in range(nm):
        qm[w, k] = scalar.wp_real(0.0)
    if active[w] == 0:
        return
    if active[w] != 1:
        status[w] = 1
        return
    for k in range(nbody*10):
        if not wp.isfinite(cinert[w, k]):
            status[w] = 2
            return
    for k in range(nv*6):
        if not wp.isfinite(cdof[w, k]):
            status[w] = 2
            return
    for k in range(nbody*10):
        crb[w, k] = cinert[w, k]
    for ii in range(nbody-1):
        i = nbody-1-ii
        p = body_parent[i]
        if p > 0:
            for k in range(10):
                crb[w, 10*p+k] += crb[w, 10*i+k]
    for ii in range(nv):
        i = nv-1-ii
        adr = dof_madr[i]
        if dof_simple[i] != 0:
            qm[w, adr] = dof_m0[i]
        else:
            qm[w, adr] = dof_armature[i]
            b = 10*dof_body[i]
            v0 = cdof[w, 6*i]
            v1 = cdof[w, 6*i+1]
            v2 = cdof[w, 6*i+2]
            v3 = cdof[w, 6*i+3]
            v4 = cdof[w, 6*i+4]
            v5 = cdof[w, 6*i+5]
            t0 = crb[w,b]*v0 + crb[w,b+3]*v1 + crb[w,b+4]*v2 - crb[w,b+8]*v4 + crb[w,b+7]*v5
            t1 = crb[w,b+3]*v0 + crb[w,b+1]*v1 + crb[w,b+5]*v2 + crb[w,b+8]*v3 - crb[w,b+6]*v5
            t2 = crb[w,b+4]*v0 + crb[w,b+5]*v1 + crb[w,b+2]*v2 - crb[w,b+7]*v3 + crb[w,b+6]*v4
            t3 = crb[w,b+8]*v1 - crb[w,b+7]*v2 + crb[w,b+9]*v3
            t4 = crb[w,b+6]*v2 - crb[w,b+8]*v0 + crb[w,b+9]*v4
            t5 = crb[w,b+7]*v0 - crb[w,b+6]*v1 + crb[w,b+9]*v5
            j = i
            while j >= 0:
                qm[w, adr] += _dot6(cdof, w, j, t0, t1, t2, t3, t4, t5)
                j = dof_parent[j]
                adr += 1
    for k in range(nbody*10):
        if not wp.isfinite(crb[w, k]):
            status[w] = 3
    for k in range(nm):
        if not wp.isfinite(qm[w, k]):
            status[w] = 3
    if status[w] == 3:
        for k in range(nbody*10):
            crb[w, k] = scalar.wp_real(0.0)
        for k in range(nm):
            qm[w, k] = scalar.wp_real(0.0)


def launch(model, active, cinert, cdof, crb, qm, status, stream=None):
    n = active.shape[0]
    schemas = ((active, (n,), wp.int32), (cinert, (n, model.nbody*10), scalar.wp_real),
               (cdof, (n, model.nv*6), scalar.wp_real),
               (crb, (n, model.nbody*10), scalar.wp_real),
               (qm, (n, model.nM), scalar.wp_real), (status, (n,), wp.int32))
    for a, shape, dtype in schemas:
        if a.shape != shape or a.dtype != dtype or not a.device.is_cuda:
            raise ValueError(f"invalid device array: expected {shape}/{dtype}")
    wp.launch(_crb, dim=n,
              inputs=[active, cinert, cdof, model.body_parent, model.dof_body,
                      model.dof_parent, model.dof_madr, model.dof_simple,
                      model.dof_m0, model.dof_armature, model.nbody, model.nv, model.nM],
              outputs=[crb, qm, status], stream=stream)
