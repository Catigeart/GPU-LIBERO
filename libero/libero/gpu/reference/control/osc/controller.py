"""Device-resident Panda OSC for the first staged GPU migration.

This module uses fixed-size FP64 algebra on CUDA and reproduces the deliberate
FP32 quaternion conversion in robosuite 1.4. A world owns its cached controller
values and targets. The caller uploads *physical* snapshots and action buffers
at the explicit action boundary; :meth:`PandaOSCWorkspace.launch` only reads
device arrays and enqueues Warp kernels. The outputs include intermediate
matrices from the production kernel for stage-by-stage verification.

The small-matrix inverse, one-sided Jacobi SVD and quaternion conversion were
adapted from the independent controller in LIBERO_failed. See PROVENANCE.md.
"""

from __future__ import annotations
import libero.gpu.precision as scalar

import warp as wp

wp.set_module_options({"enable_backward": False, "fuse_fp": True})

Mat77 = wp.types.matrix(shape=(7, 7), dtype=scalar.wp_real)
Mat67 = wp.types.matrix(shape=(6, 7), dtype=scalar.wp_real)
Mat76 = wp.types.matrix(shape=(7, 6), dtype=scalar.wp_real)
Mat66 = wp.types.matrix(shape=(6, 6), dtype=scalar.wp_real)
Vec7 = wp.types.vector(length=7, dtype=scalar.wp_real)
Vec6 = wp.types.vector(length=6, dtype=scalar.wp_real)
Vec7i = wp.types.vector(length=7, dtype=wp.int32)

# Error bits are sticky until explicitly reset, including between graph replays.
INVALID_INPUT = 1
SINGULAR_MASS = 2
SVD_NOT_CONVERGED = 4
NONFINITE_OUTPUT = 8


@wp.func
def _inverse7(a: Mat77):
    """Partial-pivoted LU and seven triangular solves; no SPD assumption."""
    lu = a
    perm = Vec7i()
    valid = int(1)
    for i in range(7):
        perm[i] = i
    for k in range(7):
        pivot = k
        largest = wp.abs(lu[k, k])
        for i in range(k + 1, 7):
            if wp.abs(lu[i, k]) > largest:
                largest = wp.abs(lu[i, k])
                pivot = i
        if largest == scalar.wp_real(0.0) or not wp.isfinite(largest):
            valid = 0
        if valid == 1:
            if pivot != k:
                for j in range(7):
                    tmp = lu[k, j]
                    lu[k, j] = lu[pivot, j]
                    lu[pivot, j] = tmp
                ip = perm[k]
                perm[k] = perm[pivot]
                perm[pivot] = ip
            for i in range(k + 1, 7):
                lu[i, k] = lu[i, k] / lu[k, k]
                for j in range(k + 1, 7):
                    lu[i, j] = lu[i, j] - lu[i, k] * lu[k, j]
    result = Mat77()
    if valid == 1:
        for column in range(7):
            x = Vec7()
            for i in range(7):
                val = scalar.wp_real(0.0)
                if perm[i] == column:
                    val = scalar.wp_real(1.0)
                for j in range(i):
                    val = val - lu[i, j] * x[j]
                x[i] = val
            for reverse in range(7):
                i = 6 - reverse
                val = x[i]
                for j in range(i + 1, 7):
                    val = val - lu[i, j] * x[j]
                x[i] = val / lu[i, i]
            for i in range(7):
                result[i, column] = x[i]
    return result, valid


@wp.func
def _pinv6(a: Mat66, n: int):
    """One-sided Jacobi SVD, n=3/6, with NumPy/Numba's 1e-15 cutoff.

    Unlike eigendecomposition of A.T @ A this does not square the condition
    number.  Scaling avoids overflow/underflow for finite nonzero inputs.
    ``B=A/scale @ V`` is orthogonalized column by column, then the inverse is
    accumulated as V[:,k] B[:,k].T / (sigma[k]**2 * scale).
    """
    scale = scalar.wp_real(0.0)
    for i in range(n):
        for j in range(n):
            scale = wp.max(scale, wp.abs(a[i, j]))
    result = Mat66()
    rank = int(0)
    converged = int(1)
    if scale > scalar.wp_real(0.0):
        b = Mat66()
        v = Mat66()
        for i in range(n):
            v[i, i] = scalar.wp_real(1.0)
            for j in range(n):
                b[i, j] = a[i, j] / scale
        converged = int(0)
        for sweep in range(64):
            rotations = int(0)
            for p in range(n):
                for q in range(p + 1, n):
                    alpha = scalar.wp_real(0.0)
                    beta = scalar.wp_real(0.0)
                    gamma = scalar.wp_real(0.0)
                    for i in range(n):
                        alpha = alpha + b[i, p] * b[i, p]
                        beta = beta + b[i, q] * b[i, q]
                        gamma = gamma + b[i, p] * b[i, q]
                    threshold = scalar.wp_real(2.220446049250313e-16) * wp.sqrt(alpha) * wp.sqrt(beta)
                    if wp.abs(gamma) > threshold:
                        zeta = (beta - alpha) / (scalar.wp_real(2.0) * gamma)
                        sign_zeta = scalar.wp_real(1.0)
                        if zeta < scalar.wp_real(0.0):
                            sign_zeta = scalar.wp_real(-1.0)
                        t = sign_zeta / (wp.abs(zeta) + wp.sqrt(scalar.wp_real(1.0) + zeta * zeta))
                        c = scalar.wp_real(1.0) / wp.sqrt(scalar.wp_real(1.0) + t * t)
                        s = c * t
                        for i in range(n):
                            bp = b[i, p]
                            bq = b[i, q]
                            b[i, p] = c * bp - s * bq
                            b[i, q] = s * bp + c * bq
                            vp = v[i, p]
                            vq = v[i, q]
                            v[i, p] = c * vp - s * vq
                            v[i, q] = s * vp + c * vq
                        rotations = rotations + 1
            if rotations == 0:
                converged = 1
                break
        # Relative column orthogonality is a more useful convergence check
        # than demanding that the final sweep contains no roundoff rotations.
        max_corr = scalar.wp_real(0.0)
        norms = Vec6()
        max_norm = scalar.wp_real(0.0)
        for p in range(n):
            for i in range(n):
                norms[p] = norms[p] + b[i, p] * b[i, p]
            max_norm = wp.max(max_norm, wp.sqrt(norms[p]))
        for p in range(n):
            for q in range(p + 1, n):
                gamma = scalar.wp_real(0.0)
                for i in range(n):
                    gamma = gamma + b[i, p] * b[i, q]
                denom = wp.sqrt(norms[p]) * wp.sqrt(norms[q])
                # Discarded singular columns carry no contribution to the
                # truncated inverse; their direction is undefined at rank
                # deficiency and must not create false convergence failures.
                if wp.sqrt(norms[p]) > scalar.wp_real(1.0e-15) * max_norm and wp.sqrt(norms[q]) > scalar.wp_real(1.0e-15) * max_norm:
                    max_corr = wp.max(max_corr, wp.abs(gamma) / denom)
        if max_corr < scalar.wp_real(2.0e-15):
            converged = 1
        for k in range(n):
            if wp.sqrt(norms[k]) > scalar.wp_real(1.0e-15) * max_norm:
                rank = rank + 1
                weight = (scalar.wp_real(1.0) / norms[k]) / scale
                for i in range(n):
                    for j in range(n):
                        result[i, j] = result[i, j] + v[i, k] * b[j, k] * weight
    return result, rank, converged


@wp.func
def _clip_preserve_nan(value: scalar.wp_real, low: scalar.wp_real, high: scalar.wp_real):
    # NumPy clip propagates NaN; Warp clamp uses fmin/fmax and would replace
    # it by a bound. Keep the original CPU command and controller semantics.
    if wp.isnan(value):
        return value
    return wp.clamp(value, low, high)


@wp.func
def _rotation_delta(x: scalar.wp_real, y: scalar.wp_real, z: scalar.wp_real):
    """Match robosuite 1.4 axisangle2quat -> quat2mat float32 rounding."""
    angle = wp.sqrt(x * x + y * y + z * z)
    qraw = scalar.vec4_real(scalar.wp_real(0.0), scalar.wp_real(0.0), scalar.wp_real(0.0), scalar.wp_real(1.0))
    q = wp.vec4f(wp.float32(1.0), wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    if angle != scalar.wp_real(0.0):
        f = wp.sin(angle * scalar.wp_real(0.5)) / angle
        qraw = scalar.vec4_real(x * f, y * f, z * f, wp.cos(angle * scalar.wp_real(0.5)))
        q = wp.vec4f(wp.float32(qraw[3]), wp.float32(qraw[0]), wp.float32(qraw[1]), wp.float32(qraw[2]))
    q_cast = q
    # The pinned SciPy/OpenBLAS sdot tail accumulates rounded float products
    # in double, then returns float. A float accumulator differs by an ulp
    # for valid actions and measurably changes the controller target.
    norm_accum = scalar.wp_real(0.0)
    for i in range(4):
        norm_accum = norm_accum + scalar.wp_real(q[i] * q[i])
    norm = wp.float32(norm_accum)
    factor = wp.sqrt(scalar.wp_real(2.0) / scalar.wp_real(norm))
    for i in range(4):
        q[i] = wp.float32(scalar.wp_real(q[i]) * factor)
    r = scalar.mat33_real()
    r[0, 0] = scalar.wp_real(1.0) - scalar.wp_real(q[2] * q[2]) - scalar.wp_real(q[3] * q[3])
    r[1, 1] = scalar.wp_real(1.0) - scalar.wp_real(q[1] * q[1]) - scalar.wp_real(q[3] * q[3])
    r[2, 2] = scalar.wp_real(1.0) - scalar.wp_real(q[1] * q[1]) - scalar.wp_real(q[2] * q[2])
    r[0, 1] = scalar.wp_real(q[1] * q[2] - q[3] * q[0])
    r[0, 2] = scalar.wp_real(q[1] * q[3] + q[2] * q[0])
    r[1, 0] = scalar.wp_real(q[1] * q[2] + q[3] * q[0])
    r[1, 2] = scalar.wp_real(q[2] * q[3] - q[1] * q[0])
    r[2, 0] = scalar.wp_real(q[1] * q[3] - q[2] * q[0])
    r[2, 1] = scalar.wp_real(q[2] * q[3] + q[1] * q[0])
    return r, qraw, q_cast


@wp.kernel(enable_backward=False)
def refresh_cache(
    active: wp.array(dtype=wp.int32),
    needs_update: wp.array(dtype=wp.int32),
    cache_valid: wp.array(dtype=wp.int32),
    error: wp.array(dtype=wp.int32),
    phys_mass: wp.array2d(dtype=scalar.wp_real),
    phys_jacobian: wp.array2d(dtype=scalar.wp_real),
    phys_q: wp.array2d(dtype=scalar.wp_real),
    phys_qvel: wp.array2d(dtype=scalar.wp_real),
    phys_ee_pos: wp.array2d(dtype=scalar.wp_real),
    phys_ee_rot: wp.array2d(dtype=scalar.wp_real),
    phys_ee_velocity: wp.array2d(dtype=scalar.wp_real),
    mass: wp.array2d(dtype=scalar.wp_real),
    jacobian: wp.array2d(dtype=scalar.wp_real),
    q: wp.array2d(dtype=scalar.wp_real),
    qvel: wp.array2d(dtype=scalar.wp_real),
    ee_pos: wp.array2d(dtype=scalar.wp_real),
    ee_rot: wp.array2d(dtype=scalar.wp_real),
    ee_velocity: wp.array2d(dtype=scalar.wp_real),
):
    world = wp.tid()
    if active[world] == 0 or error[world] != 0:
        return
    if needs_update[world] != 0:
        for i in range(49):
            mass[world, i] = phys_mass[world, i]
        for i in range(42):
            jacobian[world, i] = phys_jacobian[world, i]
        for i in range(7):
            q[world, i] = phys_q[world, i]
            qvel[world, i] = phys_qvel[world, i]
        for i in range(3):
            ee_pos[world, i] = phys_ee_pos[world, i]
        for i in range(9):
            ee_rot[world, i] = phys_ee_rot[world, i]
        for i in range(6):
            ee_velocity[world, i] = phys_ee_velocity[world, i]
        cache_valid[world] = 1
        needs_update[world] = 0
    elif cache_valid[world] == 0:
        error[world] = INVALID_INPUT


@wp.func
def osc_device(
    world: int,
    mass: wp.array2d(dtype=scalar.wp_real),
    jacobian: wp.array2d(dtype=scalar.wp_real),
    q: wp.array2d(dtype=scalar.wp_real),
    qvel: wp.array2d(dtype=scalar.wp_real),
    phys_bias: wp.array2d(dtype=scalar.wp_real),
    q0: wp.array2d(dtype=scalar.wp_real),
    ee_pos: wp.array2d(dtype=scalar.wp_real),
    ee_rot: wp.array2d(dtype=scalar.wp_real),
    ee_velocity: wp.array2d(dtype=scalar.wp_real),
    action: wp.array2d(dtype=scalar.wp_real),
    goal_pos: wp.array2d(dtype=scalar.wp_real),
    goal_rot: wp.array2d(dtype=scalar.wp_real),
    gripper: wp.array2d(dtype=scalar.wp_real),
    active: wp.array(dtype=wp.int32),
    gains: wp.array2d(dtype=scalar.wp_real),
    ctrl_range: wp.array2d(dtype=scalar.wp_real),
    ctrl: wp.array2d(dtype=scalar.wp_real),
    raw_tau: wp.array2d(dtype=scalar.wp_real),
    error: wp.array(dtype=wp.int32),
    ranks: wp.array2d(dtype=wp.int32),
    policy_step: wp.array(dtype=wp.int32),
    needs_update: wp.array(dtype=wp.int32),
    mass_inv: wp.array2d(dtype=scalar.wp_real),
    A_full: wp.array2d(dtype=scalar.wp_real),
    A_pos: wp.array2d(dtype=scalar.wp_real),
    A_ori: wp.array2d(dtype=scalar.wp_real),
    lambda_full: wp.array2d(dtype=scalar.wp_real),
    lambda_pos: wp.array2d(dtype=scalar.wp_real),
    lambda_ori: wp.array2d(dtype=scalar.wp_real),
    nullspace_matrix: wp.array2d(dtype=scalar.wp_real),
    position_error: wp.array2d(dtype=scalar.wp_real),
    orientation_error: wp.array2d(dtype=scalar.wp_real),
    quat2mat_input: wp.array2d(dtype=scalar.wp_real),
    quat_cast_wxyz: wp.array2d(dtype=wp.float32),
    quat2mat_result: wp.array2d(dtype=scalar.wp_real),
    quat_valid: wp.array(dtype=wp.int32),
    nullspace_torque: wp.array2d(dtype=scalar.wp_real),
    torques_clipped: wp.array2d(dtype=scalar.wp_real),
    emit_diagnostics: int,
    uncouple: int,
):
    if active[world] == 0 or error[world] != 0:
        return
    m = Mat77()
    j = Mat67()
    valid = int(1)
    for row in range(7):
        if not wp.isfinite(phys_bias[world, row]) or not wp.isfinite(q0[world, row]):
            valid = 0
        for column in range(7):
            m[row, column] = mass[world, row * 7 + column]
            if not wp.isfinite(m[row, column]):
                valid = 0
    for row in range(6):
        for column in range(7):
            j[row, column] = jacobian[world, row * 7 + column]
            if not wp.isfinite(j[row, column]):
                valid = 0
    # The CPU OSC accepts non-finite pose/velocity/joint cache entries in its
    # elementwise and dot-product arithmetic. Preserve their propagation into
    # raw/clipped torque; native actuation handles NaN ctrl with BADCTRL. M/J
    # remain separate: their inverse/SVD failure semantics are not this path.
    if valid == 0:
        error[world] = 1
        return
    inverse, mass_valid = _inverse7(m)
    if mass_valid == 0:
        error[world] = 2
        return
    for row in range(7):
        for column in range(7):
            mass_inv[world, row * 7 + column] = inverse[row, column]
    # JM^-1 and its transpose are reused for all three task-space matrices
    # and the nullspace projector; no explicit 7x7 projector is materialized.
    jw = Mat67()
    for row in range(6):
        for column in range(7):
            for k in range(7):
                jw[row, column] = jw[row, column] + j[row, k] * inverse[k, column]
    a = Mat66()
    ap = Mat66()
    ar = Mat66()
    for row in range(6):
        for column in range(6):
            for k in range(7):
                a[row, column] = a[row, column] + jw[row, k] * j[column, k]
    for row in range(3):
        for column in range(3):
            ap[row, column] = a[row, column]
            ar[row, column] = a[row + 3, column + 3]
    lf, rank_full, ok_full = _pinv6(a, 6)
    lp, rank_pos, ok_pos = _pinv6(ap, 3)
    lr, rank_ori, ok_ori = _pinv6(ar, 3)
    ranks[world, 0] = rank_full
    ranks[world, 1] = rank_pos
    ranks[world, 2] = rank_ori
    if ok_full == 0 or ok_pos == 0 or ok_ori == 0:
        error[world] = 4
        return
    for row in range(6):
        for column in range(6):
            A_full[world, row * 6 + column] = a[row, column]
            lambda_full[world, row * 6 + column] = lf[row, column]
    for row in range(3):
        for column in range(3):
            A_pos[world, row * 3 + column] = ap[row, column]
            A_ori[world, row * 3 + column] = ar[row, column]
            lambda_pos[world, row * 3 + column] = lp[row, column]
            lambda_ori[world, row * 3 + column] = lr[row, column]
    # N = I - (M^-1 J.T Lambda) J is a production intermediate and is used
    # below to compute the actual nullspace torque, as in robosuite.
    minv_jt = Mat76()
    jbar = Mat76()
    nullspace = Mat77()
    for row in range(7):
        for column in range(6):
            for k in range(7):
                minv_jt[row, column] = minv_jt[row, column] + inverse[row, k] * j[column, k]
    for row in range(7):
        for column in range(6):
            for k in range(6):
                jbar[row, column] = jbar[row, column] + minv_jt[row, k] * lf[k, column]
    for row in range(7):
        for column in range(7):
            value = scalar.wp_real(0.0)
            if row == column:
                value = scalar.wp_real(1.0)
            for k in range(6):
                value = value - jbar[row, k] * j[k, column]
            nullspace[row, column] = value
            if emit_diagnostics != 0:
                nullspace_matrix[world, row * 7 + column] = value
    gp = scalar.vec3_real()
    gr = scalar.mat33_real()
    current = scalar.mat33_real()
    for row in range(3):
        gp[row] = goal_pos[world, row]
        for column in range(3):
            gr[row, column] = goal_rot[world, row * 3 + column]
            current[row, column] = ee_rot[world, row * 3 + column]
    if policy_step[world] != 0:
        for i in range(3):
            gp[i] = ee_pos[world, i] + scalar.wp_real(0.05) * _clip_preserve_nan(action[world, i], scalar.wp_real(-1.0), scalar.wp_real(1.0))
        dx = scalar.wp_real(0.5) * _clip_preserve_nan(action[world, 3], scalar.wp_real(-1.0), scalar.wp_real(1.0))
        dy = scalar.wp_real(0.5) * _clip_preserve_nan(action[world, 4], scalar.wp_real(-1.0), scalar.wp_real(1.0))
        dz = scalar.wp_real(0.5) * _clip_preserve_nan(action[world, 5], scalar.wp_real(-1.0), scalar.wp_real(1.0))
        if dx != scalar.wp_real(0.0) or dy != scalar.wp_real(0.0) or dz != scalar.wp_real(0.0):
            delta_rot, qraw, q_cast = _rotation_delta(dx, dy, dz)
            gr = delta_rot * current
            quat_valid[world] = 1
            for i in range(4):
                quat2mat_input[world, i] = qraw[i]
                quat_cast_wxyz[world, i] = q_cast[i]
            for i in range(3):
                for k in range(3):
                    quat2mat_result[world, i * 3 + k] = delta_rot[i, k]
        else:
            quat_valid[world] = 0
    else:
        quat_valid[world] = 0
    er = scalar.vec3_real()
    for column in range(3):
        c = scalar.vec3_real(current[0, column], current[1, column], current[2, column])
        d = scalar.vec3_real(gr[0, column], gr[1, column], gr[2, column])
        er = er + wp.cross(c, d) * scalar.wp_real(0.5)
    desired = Vec6()
    wrench = Vec6()
    for i in range(3):
        position_error[world, i] = gp[i] - ee_pos[world, i]
        orientation_error[world, i] = er[i]
        desired[i] = gains[world, i] * position_error[world, i] - gains[world, i + 6] * ee_velocity[world, i]
        desired[i + 3] = gains[world, i + 3] * er[i] - gains[world, i + 9] * ee_velocity[world, i + 3]
    if uncouple != 0:
        for row in range(3):
            for column in range(3):
                wrench[row] = wrench[row] + lp[row, column] * desired[column]
                wrench[row + 3] = wrench[row + 3] + lr[row, column] * desired[column + 3]
    else:
        for row in range(6):
            for column in range(6):
                wrench[row] = wrench[row] + lf[row, column] * desired[column]
    pose_tau = Vec7()
    for row in range(7):
        for column in range(7):
            pd = scalar.wp_real(10.0) * (q0[world, column] - q[world, column]) - scalar.wp_real(6.324555320336759) * qvel[world, column]
            pose_tau[row] = pose_tau[row] + m[row, column] * pd
    tau = Vec7()
    for row in range(7):
        nt = scalar.wp_real(0.0)
        for column in range(7):
            nt = nt + nullspace[column, row] * pose_tau[column]
        nullspace_torque[world, row] = nt
        for column in range(6):
            tau[row] = tau[row] + j[column, row] * wrench[column]
        tau[row] = tau[row] + phys_bias[world, row]
        tau[row] = tau[row] + nt
    for i in range(3):
        goal_pos[world, i] = gp[i]
        for k in range(3):
            goal_rot[world, i * 3 + k] = gr[i, k]
    for i in range(7):
        raw_tau[world, i] = tau[i]
        torques_clipped[world, i] = _clip_preserve_nan(tau[i], ctrl_range[world, 2 * i], ctrl_range[world, 2 * i + 1])
        ctrl[world, i] = torques_clipped[world, i]
    sign = scalar.wp_real(0.0)
    if wp.isnan(action[world, 6]):
        sign = action[world, 6]
    elif action[world, 6] > scalar.wp_real(0.0):
        sign = scalar.wp_real(1.0)
    elif action[world, 6] < scalar.wp_real(0.0):
        sign = scalar.wp_real(-1.0)
    for i in range(2):
        direction = scalar.wp_real(-1.0)
        if i == 1:
            direction = scalar.wp_real(1.0)
        value = _clip_preserve_nan(gripper[world, i] + scalar.wp_real(0.01) * direction * sign, scalar.wp_real(-1.0), scalar.wp_real(1.0))
        gripper[world, i] = value
        low = ctrl_range[world, 2 * (i + 7)]
        high = ctrl_range[world, 2 * (i + 7) + 1]
        ctrl[world, i + 7] = (high + low) * scalar.wp_real(0.5) + (high - low) * scalar.wp_real(0.5) * value
    needs_update[world] = 1


@wp.kernel(enable_backward=False)
def fused_osc_control(
    mass: wp.array2d(dtype=scalar.wp_real),
    jacobian: wp.array2d(dtype=scalar.wp_real),
    q: wp.array2d(dtype=scalar.wp_real),
    qvel: wp.array2d(dtype=scalar.wp_real),
    phys_bias: wp.array2d(dtype=scalar.wp_real),
    q0: wp.array2d(dtype=scalar.wp_real),
    ee_pos: wp.array2d(dtype=scalar.wp_real),
    ee_rot: wp.array2d(dtype=scalar.wp_real),
    ee_velocity: wp.array2d(dtype=scalar.wp_real),
    action: wp.array2d(dtype=scalar.wp_real),
    goal_pos: wp.array2d(dtype=scalar.wp_real),
    goal_rot: wp.array2d(dtype=scalar.wp_real),
    gripper: wp.array2d(dtype=scalar.wp_real),
    active: wp.array(dtype=wp.int32),
    gains: wp.array2d(dtype=scalar.wp_real),
    ctrl_range: wp.array2d(dtype=scalar.wp_real),
    ctrl: wp.array2d(dtype=scalar.wp_real),
    raw_tau: wp.array2d(dtype=scalar.wp_real),
    error: wp.array(dtype=wp.int32),
    ranks: wp.array2d(dtype=wp.int32),
    policy_step: wp.array(dtype=wp.int32),
    needs_update: wp.array(dtype=wp.int32),
    mass_inv: wp.array2d(dtype=scalar.wp_real),
    A_full: wp.array2d(dtype=scalar.wp_real),
    A_pos: wp.array2d(dtype=scalar.wp_real),
    A_ori: wp.array2d(dtype=scalar.wp_real),
    lambda_full: wp.array2d(dtype=scalar.wp_real),
    lambda_pos: wp.array2d(dtype=scalar.wp_real),
    lambda_ori: wp.array2d(dtype=scalar.wp_real),
    nullspace_matrix: wp.array2d(dtype=scalar.wp_real),
    position_error: wp.array2d(dtype=scalar.wp_real),
    orientation_error: wp.array2d(dtype=scalar.wp_real),
    quat2mat_input: wp.array2d(dtype=scalar.wp_real),
    quat_cast_wxyz: wp.array2d(dtype=wp.float32),
    quat2mat_result: wp.array2d(dtype=scalar.wp_real),
    quat_valid: wp.array(dtype=wp.int32),
    nullspace_torque: wp.array2d(dtype=scalar.wp_real),
    torques_clipped: wp.array2d(dtype=scalar.wp_real),
    emit_diagnostics: int,
    uncouple: int,
):
    world = wp.tid()
    osc_device(world, mass, jacobian, q, qvel, phys_bias, q0, ee_pos, ee_rot, ee_velocity,
               action, goal_pos, goal_rot, gripper, active, gains, ctrl_range,
               ctrl, raw_tau, error, ranks, policy_step, needs_update,
               mass_inv, A_full, A_pos, A_ori, lambda_full, lambda_pos,
               lambda_ori, nullspace_matrix, position_error, orientation_error,
               quat2mat_input, quat_cast_wxyz, quat2mat_result, quat_valid,
               nullspace_torque, torques_clipped, emit_diagnostics, uncouple)


@wp.kernel(enable_backward=False)
def reset_controller(
    mask: wp.array(dtype=wp.int32),
    ee_pos: wp.array2d(dtype=scalar.wp_real),
    ee_rot: wp.array2d(dtype=scalar.wp_real),
    q: wp.array2d(dtype=scalar.wp_real),
    goal_pos: wp.array2d(dtype=scalar.wp_real),
    goal_rot: wp.array2d(dtype=scalar.wp_real),
    q0: wp.array2d(dtype=scalar.wp_real),
    gripper: wp.array2d(dtype=scalar.wp_real),
    error: wp.array(dtype=wp.int32),
    cache_valid: wp.array(dtype=wp.int32),
    needs_update: wp.array(dtype=wp.int32),
    clear_gripper: int,
    update_initial_joints: int,
):
    world = wp.tid()
    if mask[world] != 0:
        for i in range(3):
            goal_pos[world, i] = ee_pos[world, i]
        for i in range(9):
            goal_rot[world, i] = ee_rot[world, i]
        if update_initial_joints != 0:
            for i in range(7):
                q0[world, i] = q[world, i]
        if clear_gripper != 0:
            for i in range(2):
                gripper[world, i] = scalar.wp_real(0.0)
        error[world] = 0
        cache_valid[world] = 0
        needs_update[world] = 1


class PandaOSCWorkspace:
    """Preallocated device inputs, controller state, outputs, and diagnostics.

    Matrix buffers are row-major flattened per world. ``phys_*`` inputs are
    boundary data supplied by the caller. ``phys_bias`` is read live on every
    control call. The other corresponding unprefixed buffers are the persistent
    cache, refreshed on device only when ``needs_update`` is true.
    ``policy_step`` selects goal updates per world.
    ``launch`` uses no NumPy, host solve, allocation, or device-host transfer.
    """

    WIDTHS = {
        "phys_mass": 49, "phys_jacobian": 42, "phys_q": 7,
        "phys_qvel": 7, "phys_bias": 7, "phys_ee_pos": 3,
        "phys_ee_rot": 9, "phys_ee_velocity": 6,
        "mass": 49, "jacobian": 42, "q": 7, "qvel": 7,
        "ee_pos": 3, "ee_rot": 9, "ee_velocity": 6,
        "q0": 7, "action": 7, "goal_pos": 3, "goal_rot": 9,
        "gripper": 2, "gains": 12, "ctrl_range": 18,
        "ctrl": 9, "torques_raw": 7, "torques_clipped": 7,
        "mass_inv": 49, "A_full": 36, "A_pos": 9, "A_ori": 9,
        "lambda_full": 36, "lambda_pos": 9, "lambda_ori": 9,
        "nullspace_matrix": 49,
        "position_error": 3, "orientation_error": 3,
        "quat2mat_input": 4, "quat2mat_result": 9,
        "nullspace_torque": 7,
    }

    def __init__(self, nworld: int, device="cuda:0", *, uncouple=True,
                 diagnostics=True):
        if type(nworld) is not int or nworld < 1:
            raise ValueError("phase 1 supports N=1/4/20/128")
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError("PandaOSCWorkspace requires actual CUDA")
        self.nworld = nworld
        self.uncouple = int(bool(uncouple))
        self.diagnostics = int(bool(diagnostics))
        for name, width in self.WIDTHS.items():
            setattr(self, name, wp.zeros((nworld, width), dtype=scalar.wp_real, device=self.device))
        self.quat_cast_wxyz = wp.zeros((nworld, 4), dtype=wp.float32, device=self.device)
        for name in ("active", "policy_step", "needs_update", "cache_valid", "error", "quat_valid"):
            setattr(self, name, wp.zeros(nworld, dtype=wp.int32, device=self.device))
        self.ranks = wp.zeros((nworld, 3), dtype=wp.int32, device=self.device)

    def launch(self, *, stream=None):
        """Enqueue one control call with resident buffers; also Graph capturable."""
        refresh_names = (
            "active", "needs_update", "cache_valid", "error",
            "phys_mass", "phys_jacobian", "phys_q", "phys_qvel",
            "phys_ee_pos", "phys_ee_rot", "phys_ee_velocity",
            "mass", "jacobian", "q", "qvel", "ee_pos",
            "ee_rot", "ee_velocity",
        )
        wp.launch(refresh_cache, dim=self.nworld,
                  inputs=[getattr(self, name) for name in refresh_names],
                  device=self.device, stream=stream)
        control_names = (
            "mass", "jacobian", "q", "qvel", "phys_bias", "q0", "ee_pos",
            "ee_rot", "ee_velocity", "action", "goal_pos", "goal_rot",
            "gripper", "active", "gains", "ctrl_range", "ctrl",
            "torques_raw", "error", "ranks", "policy_step", "needs_update",
            "mass_inv", "A_full", "A_pos", "A_ori", "lambda_full",
            "lambda_pos", "lambda_ori", "nullspace_matrix", "position_error",
            "orientation_error", "quat2mat_input", "quat_cast_wxyz",
            "quat2mat_result", "quat_valid", "nullspace_torque", "torques_clipped",
        )
        wp.launch(fused_osc_control, dim=self.nworld,
                  inputs=[getattr(self, name) for name in control_names]
                         + [self.diagnostics, self.uncouple],
                  device=self.device, stream=stream)

    def reset_selected(self, mask, *, clear_gripper=False,
                       update_initial_joints=False, stream=None):
        """Reset selected worlds from resident physical inputs only."""
        if mask.device != self.device or mask.dtype != wp.int32 or mask.shape != (self.nworld,):
            raise ValueError("mask must be a device int32 array of shape (N,)")
        wp.launch(reset_controller, dim=self.nworld,
                  inputs=[mask, self.phys_ee_pos, self.phys_ee_rot, self.phys_q,
                          self.goal_pos, self.goal_rot, self.q0, self.gripper,
                          self.error, self.cache_valid, self.needs_update,
                          int(clear_gripper), int(update_initial_joints)],
                  device=self.device, stream=stream)
