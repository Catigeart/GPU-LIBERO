"""Count-only new D atomic candidates; original fixed functions unchanged.

Generated with scripts/gpu_d_runtime/generate_count_kernels.py.
Numerical/model qualification is independent of the original fixed licence.
"""
import libero.gpu.precision as scalar
import warp as wp
wp.set_module_options({'enable_backward': False, 'fuse_fp': True, 'fast_math': False})
PI = scalar.wp_real(3.14159265358979323846)
V6 = wp.types.vector(length=6,dtype=scalar.wp_real)
STATUS_OK = 0
STATUS_ACTIVE = 1
STATUS_INPUT = 2
STATUS_ARITHMETIC = 3


@wp.kernel(enable_backward=False)
def joint_passive_count(qpos: wp.array(dtype=scalar.wp_real, ndim=2), qvel: wp.array(dtype=scalar.wp_real, ndim=2), active: wp.array(dtype=wp.int32), jtype: wp.array(dtype=wp.int32), qadr: wp.array(dtype=wp.int32), dadr: wp.array(dtype=wp.int32), stiffness: wp.array(dtype=scalar.wp_real), spring: wp.array(dtype=scalar.wp_real), damping: wp.array(dtype=scalar.wp_real), out: wp.array(dtype=scalar.wp_real, ndim=2), status: wp.array(dtype=wp.int32), nv: int, nq: int, njnt: int):
    w = wp.tid()
    status[w] = 0
    for k in range(nv):
        out[w, k] = scalar.wp_real(0.0)
    if active[w] == 0:
        return
    if active[w] != 1:
        status[w] = 1
        return
    for k in range(nq):
        if not wp.isfinite(qpos[w, k]):
            status[w] = 2
            return
    for k in range(nv):
        if not wp.isfinite(qvel[w, k]):
            status[w] = 2
            return
    for i in range(njnt):
        t = jtype[i]
        s = stiffness[i]
        q = qadr[i]
        d = dadr[i]
        if t == 0:
            for axis in range(3):
                x = qpos[w, q + axis] - spring[q + axis]
                out[w, d + axis] = out[w, d + axis] - s * x
        elif t == 2 or t == 3:
            x = qpos[w, q] - spring[q]
            out[w, d] = out[w, d] - s * x
    for k in range(nv):
        out[w, k] = out[w, k] - damping[k] * qvel[w, k]

@wp.func
def _cross(a: scalar.vec3_real, b: scalar.vec3_real):
    return scalar.vec3_real(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

@wp.func
def _rot(m: wp.array(dtype=scalar.wp_real, ndim=3), w: int, body: int, v: scalar.vec3_real):
    return scalar.vec3_real(m[w, body, 0] * v[0] + m[w, body, 1] * v[1] + m[w, body, 2] * v[2], m[w, body, 3] * v[0] + m[w, body, 4] * v[1] + m[w, body, 5] * v[2], m[w, body, 6] * v[0] + m[w, body, 7] * v[1] + m[w, body, 8] * v[2])

@wp.func
def _rot_t(m: wp.array(dtype=scalar.wp_real, ndim=3), w: int, body: int, v: scalar.vec3_real):
    return scalar.vec3_real(m[w, body, 0] * v[0] + m[w, body, 3] * v[1] + m[w, body, 6] * v[2], m[w, body, 1] * v[0] + m[w, body, 4] * v[1] + m[w, body, 7] * v[2], m[w, body, 2] * v[0] + m[w, body, 5] * v[1] + m[w, body, 8] * v[2])

@wp.kernel(enable_backward=False)
def fluid_count(cvel: wp.array(dtype=scalar.wp_real, ndim=3), subtree_com: wp.array(dtype=scalar.wp_real, ndim=3), xipos: wp.array(dtype=scalar.wp_real, ndim=3), ximat: wp.array(dtype=scalar.wp_real, ndim=3), cdof: wp.array(dtype=scalar.wp_real, ndim=3), active: wp.array(dtype=wp.int32), rootid: wp.array(dtype=wp.int32), mass: wp.array(dtype=scalar.wp_real), box: wp.array(dtype=scalar.wp_real, ndim=2), ancestor: wp.array(dtype=wp.int32, ndim=2), wind: wp.array(dtype=scalar.wp_real), opt: wp.array(dtype=scalar.wp_real), passive_disabled: int, diagnostic: int, stages: wp.array(dtype=scalar.wp_real, ndim=3), qfrc: wp.array(dtype=scalar.wp_real, ndim=2), status: wp.array(dtype=wp.int32), nbody: int, nv: int):
    w = wp.tid()
    if diagnostic != 0:
        for body in range(nbody):
            for k in range(33 + 7 * nv):
                stages[w, body, k] = scalar.wp_real(0.0)
    if active[w] == 0:
        return
    if status[w] != 0:
        return
    for j in range(nv):
        if not wp.isfinite(qfrc[w, j]):
            status[w] = 2
            return
    for i in range(nbody):
        for k in range(6):
            if not wp.isfinite(cvel[w, i, k]):
                status[w] = 2
                return
        for k in range(3):
            if not wp.isfinite(subtree_com[w, i, k]) or not wp.isfinite(xipos[w, i, k]):
                status[w] = 2
                return
        for k in range(9):
            if not wp.isfinite(ximat[w, i, k]):
                status[w] = 2
                return
    for j in range(nv):
        for k in range(6):
            if not wp.isfinite(cdof[w, j, k]):
                status[w] = 2
                return
    if passive_disabled != 0:
        return
    if opt[0] <= scalar.wp_real(0.0) and opt[1] <= scalar.wp_real(0.0):
        return
    for body in range(1, nbody):
        if mass[body] < scalar.wp_real(1e-15):
            continue
        root = rootid[body]
        offset = scalar.vec3_real(xipos[w, body, 0] - subtree_com[w, root, 0], xipos[w, body, 1] - subtree_com[w, root, 1], xipos[w, body, 2] - subtree_com[w, root, 2])
        ang = scalar.vec3_real(cvel[w, body, 0], cvel[w, body, 1], cvel[w, body, 2])
        lin = scalar.vec3_real(cvel[w, body, 3], cvel[w, body, 4], cvel[w, body, 5])
        local_ang = _rot_t(ximat, w, body, ang)
        object_lin = _rot_t(ximat, w, body, lin - _cross(offset, ang))
        local_wind = _rot_t(ximat, w, body, scalar.vec3_real(wind[0], wind[1], wind[2]))
        local_lin = object_lin - local_wind
        for r in range(3):
            if not wp.isfinite(offset[r]) or not wp.isfinite(local_ang[r]) or (not wp.isfinite(object_lin[r])) or (not wp.isfinite(local_wind[r])) or (not wp.isfinite(local_lin[r])):
                status[w] = 2
                return
        if diagnostic != 0:
            for r in range(3):
                stages[w, body, r] = box[body, r]
                stages[w, body, 3 + r] = local_ang[r]
                stages[w, body, 6 + r] = object_lin[r]
                stages[w, body, 9 + r] = scalar.wp_real(0.0)
                stages[w, body, 12 + r] = local_wind[r]
                stages[w, body, 15 + r] = local_ang[r]
                stages[w, body, 18 + r] = local_lin[r]
        (x, y, z) = (box[body, 0], box[body, 1], box[body, 2])
        (visc, density) = (opt[0], opt[1])
        torque = scalar.vec3_real(scalar.wp_real(0.0), scalar.wp_real(0.0), scalar.wp_real(0.0))
        force = scalar.vec3_real(scalar.wp_real(0.0), scalar.wp_real(0.0), scalar.wp_real(0.0))
        if visc > scalar.wp_real(0.0):
            diam = (x + y + z) / scalar.wp_real(3.0)
            ang_scl = -PI * diam * diam * diam * visc
            lin_scl = -scalar.wp_real(3.0) * PI * diam * visc
            torque = local_ang * ang_scl
            force = local_lin * lin_scl
        if density > scalar.wp_real(0.0):
            force[0] = force[0] - scalar.wp_real(0.5) * density * y * z * wp.abs(local_lin[0]) * local_lin[0]
            force[1] = force[1] - scalar.wp_real(0.5) * density * x * z * wp.abs(local_lin[1]) * local_lin[1]
            force[2] = force[2] - scalar.wp_real(0.5) * density * x * y * wp.abs(local_lin[2]) * local_lin[2]
            torque[0] = torque[0] - density * x * (y * y * y * y + z * z * z * z) * wp.abs(local_ang[0]) * local_ang[0] / scalar.wp_real(64.0)
            torque[1] = torque[1] - density * y * (x * x * x * x + z * z * z * z) * wp.abs(local_ang[1]) * local_ang[1] / scalar.wp_real(64.0)
            torque[2] = torque[2] - density * z * (x * x * x * x + y * y * y * y) * wp.abs(local_ang[2]) * local_ang[2] / scalar.wp_real(64.0)
        world_torque = _rot(ximat, w, body, torque)
        world_force = _rot(ximat, w, body, force)
        for r in range(3):
            if not wp.isfinite(torque[r]) or not wp.isfinite(force[r]) or (not wp.isfinite(world_torque[r])) or (not wp.isfinite(world_force[r])):
                status[w] = 2
                return
        if diagnostic != 0:
            for r in range(3):
                stages[w, body, 21 + r] = torque[r]
                stages[w, body, 24 + r] = force[r]
                stages[w, body, 27 + r] = world_torque[r]
                stages[w, body, 30 + r] = world_force[r]
        for j in range(nv):
            jp = scalar.vec3_real(scalar.wp_real(0.0), scalar.wp_real(0.0), scalar.wp_real(0.0))
            jr = scalar.vec3_real(scalar.wp_real(0.0), scalar.wp_real(0.0), scalar.wp_real(0.0))
            if ancestor[body, j] != 0:
                jr = scalar.vec3_real(cdof[w, j, 0], cdof[w, j, 1], cdof[w, j, 2])
                jp = scalar.vec3_real(cdof[w, j, 3], cdof[w, j, 4], cdof[w, j, 5]) + _cross(jr, offset)
            for r in range(3):
                if not wp.isfinite(jp[r]) or not wp.isfinite(jr[r]):
                    status[w] = 2
                    return
            if diagnostic != 0:
                for r in range(3):
                    stages[w, body, 33 + nv * r + j] = jp[r]
                    stages[w, body, 33 + 3 * nv + nv * r + j] = jr[r]
            qforce = scalar.wp_real(0.0)
            for r in range(3):
                if world_force[r] != scalar.wp_real(0.0):
                    qforce = qforce + jp[r] * world_force[r]
            qfrc[w, j] = qfrc[w, j] + qforce
            qtorque = scalar.wp_real(0.0)
            for r in range(3):
                if world_torque[r] != scalar.wp_real(0.0):
                    qtorque = qtorque + jr[r] * world_torque[r]
            qfrc[w, j] = qfrc[w, j] + qtorque
            if not wp.isfinite(qfrc[w, j]):
                status[w] = 2
                return
            if diagnostic != 0:
                stages[w, body, 33 + 6 * nv + j] = qfrc[w, j]

@wp.kernel(enable_backward=False)
def actuation_count(length: wp.array(dtype=scalar.wp_real, ndim=2), moment: wp.array(dtype=scalar.wp_real, ndim=3), velocity: wp.array(dtype=scalar.wp_real, ndim=2), ctrl: wp.array(dtype=scalar.wp_real, ndim=2), active: wp.array(dtype=wp.int32), vel_status: wp.array(dtype=wp.int32), gain: wp.array(dtype=scalar.wp_real), bias0: wp.array(dtype=scalar.wp_real), bias1: wp.array(dtype=scalar.wp_real), bias2: wp.array(dtype=scalar.wp_real), bias_type: wp.array(dtype=wp.int32), ctrl_lo: wp.array(dtype=scalar.wp_real), ctrl_hi: wp.array(dtype=scalar.wp_real), force_lo: wp.array(dtype=scalar.wp_real), force_hi: wp.array(dtype=scalar.wp_real), force_limited: wp.array(dtype=wp.int32), disable_flags: wp.int32, force: wp.array(dtype=scalar.wp_real, ndim=2), force_pre: wp.array(dtype=scalar.wp_real, ndim=2), ctrl_used: wp.array(dtype=scalar.wp_real, ndim=2), ctrl_route: wp.array(dtype=wp.int32, ndim=2), force_route: wp.array(dtype=wp.int32, ndim=2), qfrc_actuator: wp.array(dtype=scalar.wp_real, ndim=2), status: wp.array(dtype=wp.int32), warn_badctrl_count: wp.array(dtype=wp.int32), warn_badctrl_lastinfo: wp.array(dtype=wp.int32), warning_number: wp.array(dtype=wp.int32), warning_lastinfo: wp.array(dtype=wp.int32), nv: int, nu: int):
    w = wp.tid()
    status[w] = STATUS_OK
    warn_badctrl_count[w] = 0
    warn_badctrl_lastinfo[w] = -1
    for j in range(nv):
        qfrc_actuator[w, j] = scalar.wp_real(0.0)
    for i in range(nu):
        force[w, i] = scalar.wp_real(0.0)
        force_pre[w, i] = scalar.wp_real(0.0)
        ctrl_used[w, i] = scalar.wp_real(0.0)
        ctrl_route[w, i] = 0
        force_route[w, i] = 0
    if active[w] == 0:
        return
    if active[w] != 1:
        status[w] = STATUS_ACTIVE
        return
    if vel_status[w] != 0:
        status[w] = STATUS_INPUT
        return
    if disable_flags & 1024 != 0:
        return
    bad_index = int(-1)
    for i in range(nu):
        u = ctrl[w, i]
        if disable_flags & 128 == 0:
            if u < ctrl_lo[i]:
                u = ctrl_lo[i]
                ctrl_route[w, i] = -1
            elif u > ctrl_hi[i]:
                u = ctrl_hi[i]
                ctrl_route[w, i] = 1
        if bad_index < 0 and (u != u or u > scalar.wp_real(10000000000.0) or u < scalar.wp_real(-10000000000.0)):
            bad_index = i
    if bad_index >= 0:
        warn_badctrl_count[w] = 1
        warn_badctrl_lastinfo[w] = bad_index
        warning_number[w] = warning_number[w] + 1
        warning_lastinfo[w] = bad_index
    for i in range(nu):
        u = scalar.wp_real(0.0)
        if bad_index < 0:
            u = ctrl[w, i]
            if ctrl_route[w, i] == -1:
                u = ctrl_lo[i]
            elif ctrl_route[w, i] == 1:
                u = ctrl_hi[i]
        ctrl_used[w, i] = u
        f = gain[i] * u
        if bias_type[i] == 1:
            f = f + (bias0[i] + bias1[i] * length[w, i] + bias2[i] * velocity[w, i])
        force_pre[w, i] = f
        if force_limited[i] != 0:
            if f < force_lo[i]:
                f = force_lo[i]
                force_route[w, i] = -1
            elif f > force_hi[i]:
                f = force_hi[i]
                force_route[w, i] = 1
        force[w, i] = f
    for i in range(nu):
        f = force[w, i]
        if f != scalar.wp_real(0.0):
            for j in range(nv):
                qfrc_actuator[w, j] = qfrc_actuator[w, j] + moment[w, i, j] * f
    for j in range(nv):
        if not wp.isfinite(qfrc_actuator[w, j]):
            status[w] = STATUS_ARITHMETIC
            return

@wp.func
def _dofmul(a: wp.array2d(dtype=scalar.wp_real), q: wp.array2d(dtype=scalar.wp_real), w: int, adr: int, count: int, k: int) -> scalar.wp_real:
    if count == 1:
        return a[w, 6 * adr + k] * q[w, adr]
    out = scalar.wp_real(0.0)
    if count > 1:
        for i in range(count):
            scale = q[w, adr + i]
            if scale != scalar.wp_real(0.0):
                out += a[w, 6 * (adr + i) + k] * scale
    return out

@wp.func
def _inert(i: wp.array2d(dtype=scalar.wp_real), v: wp.array2d(dtype=scalar.wp_real), w: int, body: int) -> V6:
    b = 10 * body
    a = 6 * body
    (v0, v1, v2) = (v[w, a], v[w, a + 1], v[w, a + 2])
    (v3, v4, v5) = (v[w, a + 3], v[w, a + 4], v[w, a + 5])
    f0 = i[w, b] * v0 + i[w, b + 3] * v1 + i[w, b + 4] * v2 - i[w, b + 8] * v4 + i[w, b + 7] * v5
    f1 = i[w, b + 3] * v0 + i[w, b + 1] * v1 + i[w, b + 5] * v2 + i[w, b + 8] * v3 - i[w, b + 6] * v5
    f2 = i[w, b + 4] * v0 + i[w, b + 5] * v1 + i[w, b + 2] * v2 - i[w, b + 7] * v3 + i[w, b + 6] * v4
    f3 = i[w, b + 8] * v1 - i[w, b + 7] * v2 + i[w, b + 9] * v3
    f4 = i[w, b + 6] * v2 - i[w, b + 8] * v0 + i[w, b + 9] * v4
    f5 = i[w, b + 7] * v0 - i[w, b + 6] * v1 + i[w, b + 9] * v5
    return V6(f0, f1, f2, f3, f4, f5)

@wp.func
def _inert_acc(i: wp.array2d(dtype=scalar.wp_real), v: wp.array2d(dtype=scalar.wp_real), w: int, body: int) -> V6:
    return _inert(i, v, w, body)

@wp.kernel(enable_backward=False)
def post_count(active: wp.array(dtype=wp.int32), qvel: wp.array2d(dtype=scalar.wp_real), qacc: wp.array2d(dtype=scalar.wp_real), cinert: wp.array2d(dtype=scalar.wp_real), cvel: wp.array2d(dtype=scalar.wp_real), cdof: wp.array2d(dtype=scalar.wp_real), cdof_dot: wp.array2d(dtype=scalar.wp_real), xipos: wp.array3d(dtype=scalar.wp_real), com: wp.array3d(dtype=scalar.wp_real), site_pos: wp.array3d(dtype=scalar.wp_real), site_mat: wp.array3d(dtype=scalar.wp_real), xfrc: wp.array2d(dtype=scalar.wp_real), efc: wp.array2d(dtype=scalar.wp_real), ncon: wp.array(dtype=wp.int32), nefc: wp.array(dtype=wp.int32), nequal: wp.array(dtype=wp.int32), cdim: wp.array2d(dtype=wp.int32), caddr: wp.array2d(dtype=wp.int32), cgeom1: wp.array2d(dtype=wp.int32), cgeom2: wp.array2d(dtype=wp.int32), cframe: wp.array2d(dtype=scalar.wp_real), cpos: wp.array2d(dtype=scalar.wp_real), parent: wp.array(dtype=wp.int32), dofadr: wp.array(dtype=wp.int32), dofnum: wp.array(dtype=wp.int32), root: wp.array(dtype=wp.int32), geom_body: wp.array(dtype=wp.int32), gravity: wp.array(dtype=scalar.wp_real), ccap: int, ecap: int, cacc: wp.array2d(dtype=scalar.wp_real), ext: wp.array2d(dtype=scalar.wp_real), internal: wp.array3d(dtype=scalar.wp_real), decoded: wp.array3d(dtype=scalar.wp_real), status: wp.array(dtype=wp.int32), nbody: int, nv: int, nsite: int, ngeom: int):
    w = wp.tid()
    status[w] = 0
    for j in range(nbody):
        for k in range(6):
            cacc[w, 6 * j + k] = scalar.wp_real(0.0)
            ext[w, 6 * j + k] = scalar.wp_real(0.0)
            internal[w, j, k] = scalar.wp_real(0.0)
    for j in range(ccap):
        for k in range(6):
            decoded[w, j, k] = scalar.wp_real(0.0)
    if active[w] == 0:
        return
    if active[w] != 1:
        status[w] = 1
        return
    (nc, ne) = (ncon[w], nefc[w])
    if nc < 0 or nc > ccap or ne < 0 or (ne > ecap):
        status[w] = 2
        return
    if nequal[w] != 0:
        status[w] = 6
        return
    for j in range(nv):
        if not wp.isfinite(qvel[w, j]) or not wp.isfinite(qacc[w, j]):
            status[w] = 4
            return
    for j in range(nv * 6):
        if not wp.isfinite(cdof[w, j]) or not wp.isfinite(cdof_dot[w, j]):
            status[w] = 4
            return
    for j in range(nbody):
        for k in range(10):
            if not wp.isfinite(cinert[w, 10 * j + k]):
                status[w] = 4
                return
        for k in range(6):
            if not wp.isfinite(cvel[w, 6 * j + k]) or not wp.isfinite(xfrc[w, 6 * j + k]):
                status[w] = 4
                return
        for k in range(3):
            if not wp.isfinite(xipos[w, j, k]) or not wp.isfinite(com[w, j, k]):
                status[w] = 4
                return
    for j in range(ne):
        if not wp.isfinite(efc[w, j]):
            status[w] = 4
            return
    for j in range(nsite):
        for k in range(3):
            if not wp.isfinite(site_pos[w, j, k]):
                status[w] = 4
                return
        for k in range(9):
            if not wp.isfinite(site_mat[w, j, k]):
                status[w] = 4
                return
    for j in range(nc):
        (ad, dim) = (caddr[w, j], cdim[w, j])
        if ad >= 0:
            if dim < 1 or dim > 6:
                status[w] = 3
                return
            if ad > ne - dim or cgeom1[w, j] < 0 or cgeom1[w, j] >= ngeom or (cgeom2[w, j] < 0) or (cgeom2[w, j] >= ngeom):
                status[w] = 3
                return
        if ad >= 0:
            for k in range(9):
                if not wp.isfinite(cframe[w, 9 * j + k]):
                    status[w] = 4
                    return
            for k in range(3):
                if not wp.isfinite(cpos[w, 3 * j + k]):
                    status[w] = 4
                    return
    for k in range(3):
        cacc[w, 3 + k] = gravity[k] * scalar.wp_real(-1.0)
    for j in range(1, nbody):
        nonzero = False
        for k in range(6):
            if xfrc[w, 6 * j + k] != scalar.wp_real(0.0):
                nonzero = True
        if nonzero:
            dx = com[w, root[j], 0] - xipos[w, j, 0]
            dy = com[w, root[j], 1] - xipos[w, j, 1]
            dz = com[w, root[j], 2] - xipos[w, j, 2]
            (fx, fy, fz) = (xfrc[w, 6 * j], xfrc[w, 6 * j + 1], xfrc[w, 6 * j + 2])
            (tx, ty, tz) = (xfrc[w, 6 * j + 3], xfrc[w, 6 * j + 4], xfrc[w, 6 * j + 5])
            ext[w, 6 * j] += tx - (dy * fz - dz * fy)
            ext[w, 6 * j + 1] += ty - (dz * fx - dx * fz)
            ext[w, 6 * j + 2] += tz - (dx * fy - dy * fx)
            ext[w, 6 * j + 3] += fx
            ext[w, 6 * j + 4] += fy
            ext[w, 6 * j + 5] += fz
    for j in range(nc):
        decode_address = caddr[w, j]
        if decode_address >= 0:
            for k in range(cdim[w, j]):
                decoded[w, j, k] = efc[w, decode_address + k]
            f0 = cframe[w, 9 * j] * decoded[w, j, 0] + cframe[w, 9 * j + 3] * decoded[w, j, 1] + cframe[w, 9 * j + 6] * decoded[w, j, 2]
            f1 = cframe[w, 9 * j + 1] * decoded[w, j, 0] + cframe[w, 9 * j + 4] * decoded[w, j, 1] + cframe[w, 9 * j + 7] * decoded[w, j, 2]
            f2 = cframe[w, 9 * j + 2] * decoded[w, j, 0] + cframe[w, 9 * j + 5] * decoded[w, j, 1] + cframe[w, 9 * j + 8] * decoded[w, j, 2]
            t0 = cframe[w, 9 * j] * decoded[w, j, 3] + cframe[w, 9 * j + 3] * decoded[w, j, 4] + cframe[w, 9 * j + 6] * decoded[w, j, 5]
            t1 = cframe[w, 9 * j + 1] * decoded[w, j, 3] + cframe[w, 9 * j + 4] * decoded[w, j, 4] + cframe[w, 9 * j + 7] * decoded[w, j, 5]
            t2 = cframe[w, 9 * j + 2] * decoded[w, j, 3] + cframe[w, 9 * j + 5] * decoded[w, j, 4] + cframe[w, 9 * j + 8] * decoded[w, j, 5]
            for side in range(2):
                geom_index = cgeom1[w, j] + side * (cgeom2[w, j] - cgeom1[w, j])
                body = geom_body[geom_index]
                if body > 0:
                    dx = com[w, root[body], 0] - cpos[w, 3 * j]
                    dy = com[w, root[body], 1] - cpos[w, 3 * j + 1]
                    dz = com[w, root[body], 2] - cpos[w, 3 * j + 2]
                    if side == 0:
                        ext[w, 6 * body] -= t0 - (dy * f2 - dz * f1)
                        ext[w, 6 * body + 1] -= t1 - (dz * f0 - dx * f2)
                        ext[w, 6 * body + 2] -= t2 - (dx * f1 - dy * f0)
                        ext[w, 6 * body + 3] -= f0
                        ext[w, 6 * body + 4] -= f1
                        ext[w, 6 * body + 5] -= f2
                    else:
                        ext[w, 6 * body] += t0 - (dy * f2 - dz * f1)
                        ext[w, 6 * body + 1] += t1 - (dz * f0 - dx * f2)
                        ext[w, 6 * body + 2] += t2 - (dx * f1 - dy * f0)
                        ext[w, 6 * body + 3] += f0
                        ext[w, 6 * body + 4] += f1
                        ext[w, 6 * body + 5] += f2
    for j in range(1, nbody):
        (adr, count) = (dofadr[j], dofnum[j])
        for k in range(6):
            first = _dofmul(cdof_dot, qvel, w, adr, count, k)
            cacc[w, 6 * j + k] = cacc[w, 6 * parent[j] + k] + first
        for k in range(6):
            second = _dofmul(cdof, qacc, w, adr, count, k)
            cacc[w, 6 * j + k] += second
        inertial = _inert_acc(cinert, cacc, w, j)
        corr = _inert(cinert, cvel, w, j)
        a = 6 * j
        (cv0, cv1, cv2) = (cvel[w, a], cvel[w, a + 1], cvel[w, a + 2])
        (cv3, cv4, cv5) = (cvel[w, a + 3], cvel[w, a + 4], cvel[w, a + 5])
        x0 = -cv2 * corr[1] + cv1 * corr[2]
        x1 = cv2 * corr[0] - cv0 * corr[2]
        x2 = -cv1 * corr[0] + cv0 * corr[1]
        x3 = -cv2 * corr[4] + cv1 * corr[5]
        x4 = cv2 * corr[3] - cv0 * corr[5]
        x5 = -cv1 * corr[3] + cv0 * corr[4]
        x0 += -cv5 * corr[4] + cv4 * corr[5]
        x1 += cv5 * corr[3] - cv3 * corr[5]
        x2 += -cv4 * corr[3] + cv3 * corr[4]
        internal[w, j, 0] = inertial[0] + x0 - ext[w, a]
        internal[w, j, 1] = inertial[1] + x1 - ext[w, a + 1]
        internal[w, j, 2] = inertial[2] + x2 - ext[w, a + 2]
        internal[w, j, 3] = inertial[3] + x3 - ext[w, a + 3]
        internal[w, j, 4] = inertial[4] + x4 - ext[w, a + 4]
        internal[w, j, 5] = inertial[5] + x5 - ext[w, a + 5]
    for idx in range(nbody - 1):
        j = nbody - 1 - idx
        p = parent[j]
        for k in range(6):
            internal[w, p, k] += internal[w, j, k]
    for j in range(nbody):
        for k in range(6):
            if not wp.isfinite(cacc[w, 6 * j + k]) or not wp.isfinite(ext[w, 6 * j + k]) or (not wp.isfinite(internal[w, j, k])):
                status[w] = 5
    if status[w] == 5:
        for j in range(nbody):
            for k in range(6):
                cacc[w, 6 * j + k] = scalar.wp_real(0.0)
                ext[w, 6 * j + k] = scalar.wp_real(0.0)
                internal[w, j, k] = scalar.wp_real(0.0)

@wp.kernel(enable_backward=False)
def force_sensor_count(cfrc: wp.array(dtype=scalar.wp_real, ndim=3), com: wp.array(dtype=scalar.wp_real, ndim=3), site_pos: wp.array(dtype=scalar.wp_real, ndim=3), site_mat: wp.array(dtype=scalar.wp_real, ndim=3), active: wp.array(dtype=wp.int32), sensor_type: wp.array(dtype=wp.int32), sensor_site: wp.array(dtype=wp.int32), sensor_adr: wp.array(dtype=wp.int32), site_body: wp.array(dtype=wp.int32), body_root: wp.array(dtype=wp.int32), result: wp.array(dtype=scalar.wp_real, ndim=2), status: wp.array(dtype=wp.int32), nbody: int, nsite: int):
    w = wp.tid()
    status[w] = 0
    for k in range(6):
        result[w, k] = scalar.wp_real(0.0)
    if active[w] == 0:
        return
    if active[w] != 1:
        status[w] = 1
        return
    for i in range(nbody):
        for k in range(6):
            if not wp.isfinite(cfrc[w, i, k]):
                status[w] = 2
                return
        for k in range(3):
            if not wp.isfinite(com[w, i, k]):
                status[w] = 2
                return
    for i in range(nsite):
        for k in range(3):
            if not wp.isfinite(site_pos[w, i, k]):
                status[w] = 2
                return
        for k in range(9):
            if not wp.isfinite(site_mat[w, i, k]):
                status[w] = 2
                return
    for s in range(2):
        site = sensor_site[s]
        body = site_body[site]
        root = body_root[body]
        dx = site_pos[w, site, 0] - com[w, root, 0]
        dy = site_pos[w, site, 1] - com[w, root, 1]
        dz = site_pos[w, site, 2] - com[w, root, 2]
        fx = cfrc[w, body, 3]
        fy = cfrc[w, body, 4]
        fz = cfrc[w, body, 5]
        cx = dy * fz - dz * fy
        cy = dz * fx - dx * fz
        cz = dx * fy - dy * fx
        tx = cfrc[w, body, 0] - cx
        ty = cfrc[w, body, 1] - cy
        tz = cfrc[w, body, 2] - cz
        if sensor_type[s] == 4:
            (v0, v1, v2) = (fx, fy, fz)
        else:
            (v0, v1, v2) = (tx, ty, tz)
        adr = sensor_adr[s]
        result[w, adr] = site_mat[w, site, 0] * v0 + site_mat[w, site, 3] * v1 + site_mat[w, site, 6] * v2
        result[w, adr + 1] = site_mat[w, site, 1] * v0 + site_mat[w, site, 4] * v1 + site_mat[w, site, 7] * v2
        result[w, adr + 2] = site_mat[w, site, 2] * v0 + site_mat[w, site, 5] * v1 + site_mat[w, site, 8] * v2
