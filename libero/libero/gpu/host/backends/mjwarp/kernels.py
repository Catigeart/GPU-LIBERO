"""Device-only array bridges; physical equations remain in upstream MJWarp."""
import warp as wp

wp.set_module_options({'enable_backward': False})


@wp.kernel
def site_points(site: int, positions: wp.array2d(dtype=wp.vec3), points: wp.array(dtype=wp.vec3)):
    w = wp.tid()
    points[w] = positions[w, site]


@wp.kernel
def gather(site: int, qi: wp.array(dtype=int), vi: wp.array(dtype=int), mi: wp.array(dtype=int),
           qp: wp.array2d(dtype=float), qv: wp.array2d(dtype=float), bias: wp.array2d(dtype=float),
           mass: wp.array2d(dtype=float), jp: wp.array3d(dtype=float), jr: wp.array3d(dtype=float),
           pos: wp.array2d(dtype=wp.vec3), rot: wp.array2d(dtype=wp.mat33),
           om: wp.array2d(dtype=float), oj: wp.array2d(dtype=float), oq: wp.array2d(dtype=float),
           ov: wp.array2d(dtype=float), ob: wp.array2d(dtype=float), op: wp.array2d(dtype=float),
           ore: wp.array2d(dtype=float), oe: wp.array2d(dtype=float)):
    w, x = wp.tid()
    if x < 49:
        address = mi[x]
        value = float(0.)
        if address >= 0:
            value = mass[w, address]
        om[w, x] = value
    if x < 42:
        axis = x // 7
        dof = vi[x % 7]
        value = float(0.)
        if axis < 3:
            value = jp[w, axis, dof]
        else:
            value = jr[w, axis - 3, dof]
        oj[w, x] = value
    if x < 7:
        oq[w, x] = qp[w, qi[x]]
        ov[w, x] = qv[w, vi[x]]
        ob[w, x] = bias[w, vi[x]]
    if x < 3:
        op[w, x] = pos[w, site][x]
    if x < 9:
        ore[w, x] = rot[w, site][x // 3, x % 3]
    if x < 6:
        value = float(0.)
        for k in range(qv.shape[1]):
            if x < 3:
                value += jp[w, x, k] * qv[w, k]
            else:
                value += jr[w, x - 3, k] * qv[w, k]
        oe[w, x] = value


@wp.kernel
def scatter_control(ids: wp.array(dtype=int), source: wp.array2d(dtype=float), target: wp.array2d(dtype=float)):
    w, i = wp.tid()
    target[w, ids[i]] = source[w, i]


@wp.kernel
def policy(value: int, flags: wp.array(dtype=int)):
    flags[wp.tid()] = value


@wp.kernel
def advance_count(integrations: wp.array(dtype=wp.int64), epochs: wp.array(dtype=wp.int64),
                  attempts: wp.array(dtype=wp.int64), positions: wp.array(dtype=wp.int64)):
    w = wp.tid()
    integrations[w] += wp.int64(1)
    epochs[w] += wp.int64(1)
    attempts[w] += wp.int64(1)
    positions[w] += wp.int64(1)


@wp.kernel
def state_written(mask: wp.array(dtype=int), epochs: wp.array(dtype=wp.int64),
                  resets: wp.array(dtype=wp.int64), reset: int):
    w = wp.tid()
    if mask[w] != 0:
        epochs[w] += wp.int64(1)
        resets[w] += wp.int64(reset)


@wp.kernel
def forward_count(attempts: wp.array(dtype=wp.int64), positions: wp.array(dtype=wp.int64)):
    w = wp.tid()
    attempts[w] += wp.int64(1)
    positions[w] += wp.int64(1)


@wp.kernel
def masked_float(mask: wp.array(dtype=int), source: wp.array2d(dtype=float), target: wp.array2d(dtype=float)):
    w, j = wp.tid()
    if mask[w] != 0:
        target[w, j] = source[w, j]


@wp.kernel
def bootstrap(mask: wp.array(dtype=int), pos: wp.array2d(dtype=float), rot: wp.array2d(dtype=float),
              gp: wp.array2d(dtype=float), gr: wp.array2d(dtype=float), grip: wp.array2d(dtype=float),
              error: wp.array(dtype=int), needs: wp.array(dtype=int), valid: wp.array(dtype=int)):
    w = wp.tid()
    if mask[w] != 0:
        for j in range(3):
            gp[w, j] = pos[w, j]
        for j in range(9):
            gr[w, j] = rot[w, j]
        grip[w, 0] = float(0.)
        grip[w, 1] = float(0.)
        error[w] = 0
        # The first OSC invocation consumes the current step1 inputs as well.
        needs[w] = 1
        valid[w] = 1


