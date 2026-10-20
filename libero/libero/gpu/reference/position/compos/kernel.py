"""GPU implementation of MuJoCo 2.3.7 mj_comPos.

Source: MuJoCo c9246e1, Apache-2.0, engine_core_smooth.c:179-265 and
engine_util_spatial.c:371-435. Scalar expression order follows those files.
"""
import libero.gpu.precision as scalar

from dataclasses import dataclass

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False, "fuse_fp": True, "fast_math": False})


@dataclass(frozen=True)
class ComposModel:
    nbody: int
    njnt: int
    nv: int
    mass: object
    mass_subtree: object
    inertia: object
    parent: object
    root: object
    jtype: object
    jbody: object
    jdof: object
    # None means the optional compiled-model cache was absent. Otherwise each
    # entry records (body, mj_comPos mass hex, body_subtreemass hex).
    cached_mass_differences: tuple[tuple[int, str, str], ...] | None = None

    @classmethod
    def prepare(cls, model, device="cuda"):
        nbody, njnt, nv = int(model.nbody), int(model.njnt), int(model.nv)
        if nbody < 1 or njnt < 0 or nv < 0:
            raise ValueError("invalid model dimensions")
        def integer(name, count):
            raw = np.asarray(getattr(model, name))
            if raw.shape != (count,) or not np.issubdtype(raw.dtype, np.integer):
                raise ValueError(f"{name}: expected integer vector of length {count}")
            if np.any(raw < -(1 << 31)) or np.any(raw >= (1 << 31)):
                raise ValueError(f"{name}: outside int32")
            return raw.astype(np.int32)
        def number(name, shape):
            raw = np.asarray(getattr(model, name))
            if raw.shape != shape or raw.dtype != np.float64 or not np.all(np.isfinite(raw)):
                raise ValueError(f"{name}: expected finite FP64 {shape}")
            return np.ascontiguousarray(raw)
        parent = integer("body_parentid", nbody)
        root = integer("body_rootid", nbody)
        jtype = integer("jnt_type", njnt)
        jbody = integer("jnt_bodyid", njnt)
        jdof = integer("jnt_dofadr", njnt)
        mass = number("body_mass", (nbody,))
        inertia = number("body_inertia", (nbody, 3))
        if np.any(mass < 0) or np.any(inertia < 0):
            raise ValueError("negative mass or principal inertia outside supported model domain")
        if parent[0] != 0 or root[0] != 0:
            raise ValueError("invalid world topology")
        if np.any(parent[1:] >= np.arange(1, nbody)) or np.any(parent[1:] < 0):
            raise ValueError("body order is not parent before child")
        if np.any(root < 0) or np.any(root >= nbody):
            raise ValueError("invalid body root")
        for i in range(1, nbody):
            expected = i if parent[i] == 0 else root[parent[i]]
            if root[i] != expected:
                raise ValueError("inconsistent body root")
        if np.any(jbody <= 0) or np.any(jbody >= nbody):
            raise ValueError("invalid joint body")
        if np.any((jtype < 0) | (jtype > 3)):
            raise ValueError("unsupported joint type")
        coverage = np.zeros(nv, dtype=np.int32)
        for typ, adr in zip(jtype, jdof):
            width = 6 if typ == 0 else (3 if typ == 1 else 1)
            if adr < 0 or adr + width > nv:
                raise ValueError("joint dof range outside nv")
            coverage[adr:adr + width] += 1
        if np.any(coverage != 1):
            raise ValueError("joint dof ranges must cover nv exactly once")
        # mj_comPos clears its temporary mass array, then adds each body's own
        # mass immediately before passing the subtotal to its parent. The
        # compiled body_subtreemass cache uses a different rounding order.
        mass_subtree = np.zeros(nbody, dtype=np.float64)
        for i in range(nbody - 1, -1, -1):
            mass_subtree[i] += mass[i]
            if i:
                mass_subtree[parent[i]] += mass_subtree[i]
        if not np.all(np.isfinite(mass_subtree)):
            raise ValueError("subtree mass overflow")
        cached_mass_differences = None
        if hasattr(model, "body_subtreemass"):
            frozen = number("body_subtreemass", (nbody,))
            different = np.flatnonzero(mass_subtree.view(np.uint64) != frozen.view(np.uint64))
            cached_mass_differences = tuple(
                (int(i), float(mass_subtree[i]).hex(), float(frozen[i]).hex())
                for i in different
            )
        return cls(nbody, njnt, nv,
                   wp.array(mass, dtype=scalar.wp_real, device=device),
                   wp.array(mass_subtree, dtype=scalar.wp_real, device=device),
                   wp.array(inertia.reshape(-1), dtype=scalar.wp_real, device=device),
                   wp.array(parent, dtype=wp.int32, device=device),
                   wp.array(root, dtype=wp.int32, device=device),
                   wp.array(jtype, dtype=wp.int32, device=device),
                   wp.array(jbody, dtype=wp.int32, device=device),
                   wp.array(jdof, dtype=wp.int32, device=device),
                   cached_mass_differences)


@wp.kernel
def _compos(active: wp.array(dtype=wp.int32),
            xipos: wp.array2d(dtype=scalar.wp_real),
            ximat: wp.array2d(dtype=scalar.wp_real),
            xanchor: wp.array2d(dtype=scalar.wp_real),
            xmat: wp.array2d(dtype=scalar.wp_real),
            xaxis: wp.array2d(dtype=scalar.wp_real),
            mass: wp.array(dtype=scalar.wp_real),
            mass_subtree: wp.array(dtype=scalar.wp_real),
            inertia: wp.array(dtype=scalar.wp_real),
            parent: wp.array(dtype=wp.int32),
            root: wp.array(dtype=wp.int32),
            jtype: wp.array(dtype=wp.int32),
            jbody: wp.array(dtype=wp.int32),
            jdof: wp.array(dtype=wp.int32),
            nbody: wp.int32, njnt: wp.int32, nv: wp.int32,
            subtree_com: wp.array2d(dtype=scalar.wp_real),
            cinert: wp.array2d(dtype=scalar.wp_real),
            cdof: wp.array2d(dtype=scalar.wp_real),
            status: wp.array(dtype=wp.int32)):
    w = wp.tid()
    status[w] = 0
    for k in range(nbody * 3):
        subtree_com[w, k] = scalar.wp_real(0.0)
    for k in range(nbody * 10):
        cinert[w, k] = scalar.wp_real(0.0)
    for k in range(nv * 6):
        cdof[w, k] = scalar.wp_real(0.0)
    if active[w] == 0:
        return
    if active[w] != 1:
        status[w] = 1
        return
    for k in range(nbody * 3):
        if not wp.isfinite(xipos[w, k]):
            status[w] = 2
            return
    for k in range(nbody * 9):
        if not wp.isfinite(ximat[w, k]) or not wp.isfinite(xmat[w, k]):
            status[w] = 2
            return
    for k in range(njnt * 3):
        if not wp.isfinite(xanchor[w, k]) or not wp.isfinite(xaxis[w, k]):
            status[w] = 2
            return

    for ii in range(nbody):
        i = nbody - 1 - ii
        mi = mass[i]
        for c in range(3):
            subtree_com[w, 3 * i + c] += xipos[w, 3 * i + c] * mi
        if i > 0:
            p = parent[i]
            for c in range(3):
                subtree_com[w, 3 * p + c] += subtree_com[w, 3 * i + c]
        if mass_subtree[i] < scalar.wp_real(1.0e-15):
            for c in range(3):
                subtree_com[w, 3 * i + c] = xipos[w, 3 * i + c]
        else:
            inv = scalar.wp_real(1.0) / mass_subtree[i]
            for c in range(3):
                subtree_com[w, 3 * i + c] *= inv

    for i in range(1, nbody):
        b = 10 * i
        a = 9 * i
        r = root[i]
        dx = xipos[w, 3 * i] - subtree_com[w, 3 * r]
        dy = xipos[w, 3 * i + 1] - subtree_com[w, 3 * r + 1]
        dz = xipos[w, 3 * i + 2] - subtree_com[w, 3 * r + 2]
        ix = inertia[3 * i]
        iy = inertia[3 * i + 1]
        iz = inertia[3 * i + 2]
        mi = mass[i]
        t0 = ximat[w, a] * ix
        t1 = ximat[w, a + 3] * ix
        t2 = ximat[w, a + 6] * ix
        t3 = ximat[w, a + 1] * iy
        t4 = ximat[w, a + 4] * iy
        t5 = ximat[w, a + 7] * iy
        t6 = ximat[w, a + 2] * iz
        t7 = ximat[w, a + 5] * iz
        t8 = ximat[w, a + 8] * iz
        cinert[w, b] = ximat[w, a] * t0 + ximat[w, a + 1] * t3 + ximat[w, a + 2] * t6
        cinert[w, b + 1] = ximat[w, a + 3] * t1 + ximat[w, a + 4] * t4 + ximat[w, a + 5] * t7
        cinert[w, b + 2] = ximat[w, a + 6] * t2 + ximat[w, a + 7] * t5 + ximat[w, a + 8] * t8
        cinert[w, b + 3] = ximat[w, a] * t1 + ximat[w, a + 1] * t4 + ximat[w, a + 2] * t7
        cinert[w, b + 4] = ximat[w, a] * t2 + ximat[w, a + 1] * t5 + ximat[w, a + 2] * t8
        cinert[w, b + 5] = ximat[w, a + 3] * t2 + ximat[w, a + 4] * t5 + ximat[w, a + 5] * t8
        cinert[w, b] += mi * (dy * dy + dz * dz)
        cinert[w, b + 1] += mi * (dx * dx + dz * dz)
        cinert[w, b + 2] += mi * (dx * dx + dy * dy)
        cinert[w, b + 3] -= mi * dx * dy
        cinert[w, b + 4] -= mi * dx * dz
        cinert[w, b + 5] -= mi * dy * dz
        cinert[w, b + 6] = mi * dx
        cinert[w, b + 7] = mi * dy
        cinert[w, b + 8] = mi * dz
        cinert[w, b + 9] = mi

    for j in range(njnt):
        bi = jbody[j]
        da = jdof[j] * 6
        r = root[bi]
        ox = subtree_com[w, 3 * r] - xanchor[w, 3 * j]
        oy = subtree_com[w, 3 * r + 1] - xanchor[w, 3 * j + 1]
        oz = subtree_com[w, 3 * r + 2] - xanchor[w, 3 * j + 2]
        typ = jtype[j]
        skip = 0
        if typ == 0:
            cdof[w, da + 3] = scalar.wp_real(1.0)
            cdof[w, da + 10] = scalar.wp_real(1.0)
            cdof[w, da + 17] = scalar.wp_real(1.0)
            skip = 18
        if typ == 0 or typ == 1:
            for k in range(3):
                ax = xmat[w, 9 * bi + k]
                ay = xmat[w, 9 * bi + k + 3]
                az = xmat[w, 9 * bi + k + 6]
                d = da + skip + 6 * k
                cdof[w, d] = ax
                cdof[w, d + 1] = ay
                cdof[w, d + 2] = az
                cdof[w, d + 3] = ay * oz - az * oy
                cdof[w, d + 4] = az * ox - ax * oz
                cdof[w, d + 5] = ax * oy - ay * ox
        elif typ == 2:
            cdof[w, da + 3] = xaxis[w, 3 * j]
            cdof[w, da + 4] = xaxis[w, 3 * j + 1]
            cdof[w, da + 5] = xaxis[w, 3 * j + 2]
        else:
            ax = xaxis[w, 3 * j]
            ay = xaxis[w, 3 * j + 1]
            az = xaxis[w, 3 * j + 2]
            cdof[w, da] = ax
            cdof[w, da + 1] = ay
            cdof[w, da + 2] = az
            cdof[w, da + 3] = ay * oz - az * oy
            cdof[w, da + 4] = az * ox - ax * oz
            cdof[w, da + 5] = ax * oy - ay * ox
    for k in range(nbody * 3):
        if not wp.isfinite(subtree_com[w, k]):
            status[w] = 3
    for k in range(nbody * 10):
        if not wp.isfinite(cinert[w, k]):
            status[w] = 3
    for k in range(nv * 6):
        if not wp.isfinite(cdof[w, k]):
            status[w] = 3
    if status[w] == 3:
        for k in range(nbody * 3):
            subtree_com[w, k] = scalar.wp_real(0.0)
        for k in range(nbody * 10):
            cinert[w, k] = scalar.wp_real(0.0)
        for k in range(nv * 6):
            cdof[w, k] = scalar.wp_real(0.0)


def launch(model: ComposModel, active, xipos, ximat, xanchor, xmat, xaxis,
           subtree_com, cinert, cdof, status, stream=None):
    """Launch only; all arrays must be preallocated on the device."""
    n = active.shape[0]
    schemas = ((active, (n,), wp.int32), (xipos, (n, model.nbody * 3), scalar.wp_real),
               (ximat, (n, model.nbody * 9), scalar.wp_real),
               (xanchor, (n, model.njnt * 3), scalar.wp_real),
               (xmat, (n, model.nbody * 9), scalar.wp_real),
               (xaxis, (n, model.njnt * 3), scalar.wp_real),
               (subtree_com, (n, model.nbody * 3), scalar.wp_real),
               (cinert, (n, model.nbody * 10), scalar.wp_real),
               (cdof, (n, model.nv * 6), scalar.wp_real), (status, (n,), wp.int32))
    for array, shape, dtype in schemas:
        if array.shape != shape or array.dtype != dtype or not array.device.is_cuda:
            raise ValueError(f"device array schema mismatch: expected {shape}/{dtype}")
    wp.launch(_compos, dim=active.shape[0], inputs=[active, xipos, ximat, xanchor,
              xmat, xaxis, model.mass, model.mass_subtree, model.inertia,
              model.parent, model.root, model.jtype, model.jbody, model.jdof,
              model.nbody, model.njnt, model.nv],
              outputs=[subtree_com, cinert, cdof, status], stream=stream)
