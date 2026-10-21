# Copyright 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Pinned ordinary position schedule with independent matrix/factor leaves."""
from mujoco_warp._src import forward, smooth


class Position:
    def __init__(self, model, data, *, mass_matrix=None, mass_factor=None, transmission=None, kinematics=None, com_pos=None, camlight=None, constraint_build=None, collision=None):
        self.model, self.data = model, data
        self.mass_matrix, self.mass_factor = mass_matrix, mass_factor
        self.transmission = transmission
        self.kinematics = kinematics
        self.com_pos = com_pos
        self.camlight = camlight
        self.constraint_build = constraint_build
        self.collision = collision

    def position(self,*,factorize=True):
        m, d = self.model, self.data
        matrix, factor = self.mass_matrix, self.mass_factor
        matrix_cuda = matrix is not None and matrix.selected == 'cuda'
        factor_cuda = factor is not None and factor.selected == 'cuda'
        transmission_cuda = self.transmission is not None and self.transmission.selected['transmission']=='cuda'
        kinematics_cuda = self.kinematics is not None and self.kinematics.selected=='cuda'
        com_cuda = self.com_pos is not None and self.com_pos.selected=='cuda'
        cam_cuda = self.camlight is not None and self.camlight.selected=='cuda'
        constraint_cuda = self.constraint_build is not None and self.constraint_build.selected == 'cuda'
        collision_cuda = self.collision is not None and self.collision.selected == 'cuda'
        if not matrix_cuda and not factor_cuda and not transmission_cuda and not kinematics_cuda and not com_cuda and not cam_cuda and not constraint_cuda and not collision_cuda:
            if factorize:forward.fwd_position(m,d)
            else:forward.fwd_position(m,d,factorize=False)
            return
        # All selected leaves admit the ordinary Euler/no-sleep caller before any
        # optional library load. Keep the pinned order and every public output.
        if cam_cuda:
            if self.kinematics is not None:self.kinematics.forward(com_pos=self.com_pos,camlight=self.camlight)
            else:
                smooth.kinematics(m,d)
                if com_cuda:self.com_pos.enqueue()
                else:smooth.com_pos(m,d)
                self.camlight.enqueue()
                smooth.flex(m,d)
                smooth.tendon(m,d)
        elif com_cuda:
            if self.kinematics is not None:self.kinematics.forward(com_pos=self.com_pos)
            else:
                smooth.kinematics(m,d)
                self.com_pos.enqueue()
                smooth.camlight(m,d)
                smooth.flex(m,d)
                smooth.tendon(m,d)
        elif kinematics_cuda:self.kinematics.forward()
        else:forward.fwd_kinematics(m, d)
        if matrix_cuda:
            matrix.enqueue()
        else:
            smooth.crb(m, d)
        smooth.tendon_armature(m, d)
        if factorize:
            if factor_cuda:factor.position_binding.enqueue()
            else:smooth.factor_m(m,d)
        if m.opt.run_collision_detection:
            if collision_cuda:self.collision.enqueue()
            else:forward.collision_driver.collision(m, d)
        if constraint_cuda:self.constraint_build.enqueue()
        else:forward.constraint.make_constraint(m, d)
        if transmission_cuda:self.transmission.enqueue_transmission()
        else:smooth.transmission(m, d)
