"""MuJoCo 2.3.7 mj_comVel, source c9246e1, Apache-2.0.

Follows engine_core_smooth.c:1261-1321 and engine_util_spatial.c:339-369,435-446.
"""
import libero.gpu.precision as scalar

from dataclasses import dataclass

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False, "fuse_fp": True, "fast_math": False})


@dataclass(frozen=True)
class ComVelModel:
    nbody: int
    nv: int
    body_parent: object
    body_dofadr: object
    body_dofnum: object
    dof_jntid: object
    jnt_type: object

    @classmethod
    def prepare(cls, model, device="cuda"):
        nb, nv, nj = (int(getattr(model, n)) for n in ("nbody", "nv", "njnt"))
        if nb < 1 or nv < 0 or nj < 0:
            raise ValueError("invalid dimensions")
        def integer(name, size):
            a = np.asarray(getattr(model, name))
            if a.shape != (size,) or not np.issubdtype(a.dtype, np.integer):
                raise ValueError(f"{name}: wrong shape/dtype")
            if np.any(a < -(1 << 31)) or np.any(a >= (1 << 31)):
                raise ValueError(f"{name}: outside int32")
            return a.astype(np.int32)
        bp = integer("body_parentid", nb)
        ba = integer("body_dofadr", nb)
        bn = integer("body_dofnum", nb)
        dj = integer("dof_jntid", nv)
        jt = integer("jnt_type", nj)
        if bp[0] != 0 or bn[0] != 0 or np.any(bp[1:] < 0) or np.any(bp[1:] >= np.arange(1, nb)):
            raise ValueError("invalid body topology")
        if np.any(bn < 0) or np.any((bn > 0) & ((ba < 0) | (ba + bn > nv))):
            raise ValueError("invalid body dof range")
        if np.any(dj < 0) or np.any(dj >= nj) or np.any((jt < 0) | (jt > 3)):
            raise ValueError("invalid joint mapping/type")
        coverage = np.zeros(nv, dtype=np.int32)
        for adr, count in zip(ba, bn):
            if count:
                coverage[adr:adr+count] += 1
        if np.any(coverage != 1):
            raise ValueError("body dof ranges do not cover nv exactly once")
        for i in range(1, nb):
            adr, count = int(ba[i]), int(bn[i])
            j = 0
            while j < count:
                typ = int(jt[dj[adr+j]])
                width = 6 if typ == 0 else (3 if typ == 1 else 1)
                if j + width > count or np.any(dj[adr+j:adr+j+width] != dj[adr+j]):
                    raise ValueError("joint width/mapping does not fit body dofs")
                j += width
        return cls(nb, nv, *(wp.array(x, dtype=wp.int32, device=device)
                              for x in (bp, ba, bn, dj, jt)))


@wp.func
def _cross_motion(cvel: wp.array2d(dtype=scalar.wp_real), cdof: wp.array2d(dtype=scalar.wp_real),
                  w: int, bi: int, dof: int, out: wp.array2d(dtype=scalar.wp_real)):
    c = 6*bi
    d = 6*dof
    v0, v1, v2 = cvel[w,c], cvel[w,c+1], cvel[w,c+2]
    v3, v4, v5 = cvel[w,c+3], cvel[w,c+4], cvel[w,c+5]
    a0, a1, a2 = cdof[w,d], cdof[w,d+1], cdof[w,d+2]
    a3, a4, a5 = cdof[w,d+3], cdof[w,d+4], cdof[w,d+5]
    out[w,d] = -v2*a1 + v1*a2
    out[w,d+1] = v2*a0 - v0*a2
    out[w,d+2] = -v1*a0 + v0*a1
    out[w,d+3] = -v2*a4 + v1*a5
    out[w,d+4] = v2*a3 - v0*a5
    out[w,d+5] = -v1*a3 + v0*a4
    out[w,d+3] += -v5*a1 + v4*a2
    out[w,d+4] += v5*a0 - v3*a2
    out[w,d+5] += -v4*a0 + v3*a1


@wp.func
def _add_three(cvel: wp.array2d(dtype=scalar.wp_real), cdof: wp.array2d(dtype=scalar.wp_real),
               qvel: wp.array2d(dtype=scalar.wp_real), w: int, bi: int, start: int):
    # mju_mulMatTVec first zeros tmp, then skips exact-zero qvel and adds rows.
    t0, t1, t2 = scalar.wp_real(0.0), scalar.wp_real(0.0), scalar.wp_real(0.0)
    t3, t4, t5 = scalar.wp_real(0.0), scalar.wp_real(0.0), scalar.wp_real(0.0)
    for k in range(3):
        q = qvel[w,start+k]
        if q != scalar.wp_real(0.0):
            d = 6*(start+k)
            t0 += cdof[w,d]*q
            t1 += cdof[w,d+1]*q
            t2 += cdof[w,d+2]*q
            t3 += cdof[w,d+3]*q
            t4 += cdof[w,d+4]*q
            t5 += cdof[w,d+5]*q
    c = 6*bi
    cvel[w,c] += t0
    cvel[w,c+1] += t1
    cvel[w,c+2] += t2
    cvel[w,c+3] += t3
    cvel[w,c+4] += t4
    cvel[w,c+5] += t5


@wp.kernel
def _comvel(active: wp.array(dtype=wp.int32),
            cdof: wp.array2d(dtype=scalar.wp_real), qvel: wp.array2d(dtype=scalar.wp_real),
            body_parent: wp.array(dtype=wp.int32), body_dofadr: wp.array(dtype=wp.int32),
            body_dofnum: wp.array(dtype=wp.int32), dof_jntid: wp.array(dtype=wp.int32),
            jnt_type: wp.array(dtype=wp.int32), nbody: int, nv: int,
            cvel: wp.array2d(dtype=scalar.wp_real), cdof_dot: wp.array2d(dtype=scalar.wp_real),
            status: wp.array(dtype=wp.int32)):
    w = wp.tid()
    status[w] = 0
    for k in range(nbody*6):
        cvel[w,k] = scalar.wp_real(0.0)
    for k in range(nv*6):
        cdof_dot[w,k] = scalar.wp_real(0.0)
    if active[w] == 0:
        return
    if active[w] != 1:
        status[w] = 1
        return
    for k in range(nv*6):
        if not wp.isfinite(cdof[w,k]):
            status[w] = 2
            return
    for k in range(nv):
        if not wp.isfinite(qvel[w,k]):
            status[w] = 2
            return
    for i in range(1,nbody):
        c = 6*i
        p = 6*body_parent[i]
        for k in range(6):
            cvel[w,c+k] = cvel[w,p+k]
        bda = body_dofadr[i]
        count = body_dofnum[i]
        j = int(0)
        while j < count:
            dof = bda+j
            typ = jnt_type[dof_jntid[dof]]
            if typ == 0:
                # Three translational cdof_dot rows stay exact zero.
                _add_three(cvel,cdof,qvel,w,i,dof)
                j += 3
                dof = bda+j
                for k in range(3):
                    _cross_motion(cvel,cdof,w,i,dof+k,cdof_dot)
                _add_three(cvel,cdof,qvel,w,i,dof)
                j += 3
            elif typ == 1:
                for k in range(3):
                    _cross_motion(cvel,cdof,w,i,dof+k,cdof_dot)
                _add_three(cvel,cdof,qvel,w,i,dof)
                j += 3
            else:
                _cross_motion(cvel,cdof,w,i,dof,cdof_dot)
                # n=1 mju_scl, including exact-zero velocity multiplications.
                q = qvel[w,dof]
                for k in range(6):
                    cvel[w,c+k] += cdof[w,6*dof+k]*q
                j += 1
    for k in range(nbody*6):
        if not wp.isfinite(cvel[w,k]):
            status[w] = 3
    for k in range(nv*6):
        if not wp.isfinite(cdof_dot[w,k]):
            status[w] = 3
    if status[w] == 3:
        for k in range(nbody*6):
            cvel[w,k] = scalar.wp_real(0.0)
        for k in range(nv*6):
            cdof_dot[w,k] = scalar.wp_real(0.0)


def launch(model, active, cdof, qvel, cvel, cdof_dot, status, stream=None):
    n = active.shape[0]
    schemas = ((active,(n,),wp.int32),(cdof,(n,model.nv*6),scalar.wp_real),
               (qvel,(n,model.nv),scalar.wp_real),(cvel,(n,model.nbody*6),scalar.wp_real),
               (cdof_dot,(n,model.nv*6),scalar.wp_real),(status,(n,),wp.int32))
    for a, shape, dtype in schemas:
        if a.shape != shape or a.dtype != dtype or not a.device.is_cuda:
            raise ValueError(f"invalid device array: expected {shape}/{dtype}")
    wp.launch(_comvel,dim=n,
              inputs=[active,cdof,qvel,model.body_parent,model.body_dofadr,
                      model.body_dofnum,model.dof_jntid,model.jnt_type,model.nbody,model.nv],
              outputs=[cvel,cdof_dot,status],stream=stream)
