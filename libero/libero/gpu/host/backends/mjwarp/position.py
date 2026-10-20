# Copyright 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Pinned ordinary position schedule with independent matrix/factor leaves."""
from mujoco_warp._src import forward, smooth


class Position:
    def __init__(self, model, data, *, mass_matrix=None, mass_factor=None):
        self.model, self.data = model, data
        self.mass_matrix, self.mass_factor = mass_matrix, mass_factor

    def position(self):
        m, d = self.model, self.data
        matrix, factor = self.mass_matrix, self.mass_factor
        matrix_cuda = matrix is not None and matrix.selected == 'cuda'
        factor_cuda = factor is not None and factor.selected == 'cuda'
        if not matrix_cuda and not factor_cuda:
            forward.fwd_position(m, d)
            return
        # Both selectors admit the ordinary Euler/no-sleep caller before any
        # optional library load. Keep the pinned order and every public output.
        forward.fwd_kinematics(m, d)
        if matrix_cuda:
            matrix.enqueue()
        else:
            smooth.crb(m, d)
        smooth.tendon_armature(m, d)
        if factor_cuda:
            factor.position_binding.enqueue()
        else:
            smooth.factor_m(m, d)
        if m.opt.run_collision_detection:
            forward.collision_driver.collision(m, d)
        forward.constraint.make_constraint(m, d)
        smooth.transmission(m, d)
