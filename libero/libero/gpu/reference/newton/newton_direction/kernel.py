"""MuJoCo 2.3.7 dense Newton CGupdateGradient and mju_cholSolve order.

Source: engine_solver.c:807-830 and engine_util_solve.c:67-93.
Each CUDA thread owns one independent nv=43 world.  All inputs are already
resident; launch does not allocate, upload, download, or call a CPU solver.
"""
from __future__ import annotations
import libero.gpu.precision as scalar

import numpy as np
import warp as wp

wp.set_module_options({'enable_backward': False, 'fuse_fp': True, 'fast_math': False})
NV = 43


@wp.func
def _factor(H: wp.array(dtype=scalar.wp_real, ndim=3),
            Hcone: wp.array(dtype=scalar.wp_real, ndim=3),
            w: int, cone: int, i: int, j: int) -> scalar.wp_real:
    if cone > 0:
        return Hcone[w, i, j]
    return H[w, i, j]


@wp.func
def _source_dot(H: wp.array(dtype=scalar.wp_real, ndim=3),
                Hcone: wp.array(dtype=scalar.wp_real, ndim=3),
                x: wp.array(dtype=scalar.wp_real, ndim=2),
                w: int, cone: int, row: int, count: int) -> scalar.wp_real:
    # The original mju_dot uses 4 AVX lanes, then (0+2)+(1+3).
    res = scalar.wp_real(0.0)
    k = int(0)
    if count >= 4:
        a = _factor(H, Hcone, w, cone, row, 0) * x[w, 0]
        b = _factor(H, Hcone, w, cone, row, 1) * x[w, 1]
        c = _factor(H, Hcone, w, cone, row, 2) * x[w, 2]
        d = _factor(H, Hcone, w, cone, row, 3) * x[w, 3]
        k = int(4)
        while k <= count - 4:
            a = a + _factor(H, Hcone, w, cone, row, k) * x[w, k]
            b = b + _factor(H, Hcone, w, cone, row, k+1) * x[w, k+1]
            c = c + _factor(H, Hcone, w, cone, row, k+2) * x[w, k+2]
            d = d + _factor(H, Hcone, w, cone, row, k+3) * x[w, k+3]
            k += 4
        res = (a + c) + (b + d)
    left = count - k
    if left == 3:
        p0 = _factor(H, Hcone, w, cone, row, k) * x[w, k]
        p1 = _factor(H, Hcone, w, cone, row, k+1) * x[w, k+1]
        p2 = _factor(H, Hcone, w, cone, row, k+2) * x[w, k+2]
        res = res + ((p0 + p1) + p2)
    elif left == 2:
        p0 = _factor(H, Hcone, w, cone, row, k) * x[w, k]
        p1 = _factor(H, Hcone, w, cone, row, k+1) * x[w, k+1]
        res = res + (p0 + p1)
    elif left == 1:
        res = res + _factor(H, Hcone, w, cone, row, k) * x[w, k]
    return res


@wp.kernel(enable_backward=False)
def _direction(Ma: wp.array(dtype=scalar.wp_real, ndim=2),
               smooth: wp.array(dtype=scalar.wp_real, ndim=2),
               constraint: wp.array(dtype=scalar.wp_real, ndim=2),
               H: wp.array(dtype=scalar.wp_real, ndim=3),
               Hcone: wp.array(dtype=scalar.wp_real, ndim=3),
               ncone: wp.array(dtype=wp.int32),
               nefc: wp.array(dtype=wp.int32),
               active: wp.array(dtype=wp.int32),
               upstream_status: wp.array(dtype=wp.int32),
               grad: wp.array(dtype=scalar.wp_real, ndim=2),
               Mgrad: wp.array(dtype=scalar.wp_real, ndim=2),
               candidate: wp.array(dtype=scalar.wp_real, ndim=2),
               status: wp.array(dtype=wp.int32),
               scratch_grad: wp.array(dtype=scalar.wp_real, ndim=2),
               scratch_x: wp.array(dtype=scalar.wp_real, ndim=2),
               scratch_fwd: wp.array(dtype=scalar.wp_real, ndim=2)):
    w = wp.tid()
    if upstream_status[w] != 0:
        status[w] = upstream_status[w]
        return
    status[w] = 0
    if active[w] == 0:
        return
    if active[w] != 1:
        status[w] = 1
        return
    cone = ncone[w]
    n = nefc[w]
    if n < 0 or cone < 0 or cone > n:
        status[w] = 2
        return
    # Do not inspect upper triangle or the unselected factor.
    for i in range(NV):
        if not wp.isfinite(Ma[w, i]) or not wp.isfinite(smooth[w, i]) or not wp.isfinite(constraint[w, i]):
            status[w] = 3
            return
        for j in range(i+1):
            value = _factor(H, Hcone, w, cone, i, j)
            if not wp.isfinite(value):
                status[w] = 3
                return
    # The first-error contract gives every selected nonfinite operand
    # priority over any zero diagonal, regardless of its row position.
    for i in range(NV):
        if _factor(H, Hcone, w, cone, i, i) == scalar.wp_real(0.0):
            status[w] = 4
            return
    for i in range(NV):
        value = (Ma[w, i] - smooth[w, i]) - constraint[w, i]
        if not wp.isfinite(value):
            status[w] = 5
            return
        scratch_grad[w, i] = value
        scratch_x[w, i] = value  # original mju_copy before substitution
    for i in range(NV):
        value = scratch_x[w, i]
        if i > 0:
            value = value - _source_dot(H, Hcone, scratch_x, w, cone, i, i)
        value = value / _factor(H, Hcone, w, cone, i, i)
        if not wp.isfinite(value):
            status[w] = 5
            return
        scratch_x[w, i] = value
        scratch_fwd[w, i] = value
    for reverse in range(NV):
        i = NV - 1 - reverse
        value = scratch_x[w, i]
        for j in range(i+1, NV):
            value = value - _factor(H, Hcone, w, cone, j, i) * scratch_x[w, j]
        value = value / _factor(H, Hcone, w, cone, i, i)
        if not wp.isfinite(value):
            status[w] = 5
            return
        scratch_x[w, i] = value
    for i in range(NV):
        grad[w, i] = scratch_grad[w, i]
        Mgrad[w, i] = scratch_x[w, i]
        # Warp's unary neg lowers to 0.0 - x, which maps +0 to +0.  The
        # original mju_scl multiplies by -1 and maps +0 to -0.  copysign
        # flips the sign bit exactly, including both signed zero values.
        sign = wp.copysign(scalar.wp_real(1.0), scratch_x[w, i])
        candidate[w, i] = wp.copysign(scratch_x[w, i], -sign)


class NewtonDirectionWorkspace:
    """Prepared nv=43 dense workspace; caller owns input update and inspection."""

    def __init__(self, capacity: int, device: str = 'cuda:0'):
        if capacity < 1:
            raise ValueError('capacity must be positive')
        self.capacity = int(capacity)
        self.device = device
        self.stream = wp.get_stream(device)
        n = self.capacity
        for key in ('Ma', 'smooth', 'constraint', 'grad', 'Mgrad', 'candidate',
                    'scratch_grad', 'scratch_x', 'scratch_fwd'):
            setattr(self, key, wp.empty((n, NV), dtype=scalar.wp_real, device=device))
        for key in ('H', 'Hcone'):
            setattr(self, key, wp.empty((n, NV, NV), dtype=scalar.wp_real, device=device))
        for key in ('ncone', 'nefc', 'active', 'upstream_status', 'status'):
            setattr(self, key, wp.empty(n, dtype=wp.int32, device=device))

    def upload(self, **fields):
        """Prepare phase only.  No upload is hidden in launch()."""
        shapes = {k: (self.capacity, NV) for k in ('Ma', 'smooth', 'constraint')}
        shapes.update({k: (self.capacity, NV, NV) for k in ('H', 'Hcone')})
        shapes.update({k: (self.capacity,) for k in ('ncone', 'nefc', 'active', 'upstream_status')})
        if set(fields) != set(shapes):
            raise TypeError(f'input fields: {set(fields) ^ set(shapes)}')
        prepared = {}
        for key, shape in shapes.items():
            a = np.asarray(fields[key])
            expected = np.float64 if key not in ('ncone', 'nefc', 'active', 'upstream_status') else np.int32
            if a.shape != shape or a.dtype != expected:
                raise TypeError(f'{key} requires {shape} {expected}')
            prepared[key] = np.ascontiguousarray(a)
        for key, a in prepared.items():
            getattr(self, key).assign(a)

    def launch(self):
        wp.launch(_direction, dim=self.capacity,
                  inputs=[self.Ma, self.smooth, self.constraint, self.H, self.Hcone,
                          self.ncone, self.nefc, self.active, self.upstream_status],
                  outputs=[self.grad, self.Mgrad, self.candidate, self.status,
                           self.scratch_grad, self.scratch_x, self.scratch_fwd],
                  device=self.device, stream=self.stream)

    def download(self):
        return {k: getattr(self, k).numpy() for k in ('grad', 'Mgrad', 'candidate', 'status', 'scratch_fwd')}
