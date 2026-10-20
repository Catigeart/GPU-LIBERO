# Copyright 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Retained rigid-body contributions from pinned passive.py::passive.

MJWarp c384ac51df67eb92ab7657cc472ff7494dc036bf; only names are qualified.
The caller admits no tendon/flex, then preserves gravity/fluid/adhesion order.
"""
import warp as wp
from mujoco_warp._src import passive,types


def other_forces(m,d):
    gravity_enabled = not (m.opt.disableflags & types.DisableBit.GRAVITY)
    d.qfrc_gravcomp.zero_()
    if gravity_enabled:
      wp.launch(
        passive._gravity_force,
        dim=(d.nworld, m.nbody - 1, m.nv),
        inputs=[
          m.opt.gravity,
          m.body_parentid,
          m.body_rootid,
          m.body_mass,
          m.body_gravcomp,
          m.dof_bodyid,
          m.body_isdofancestor,
          d.xipos,
          d.subtree_com,
          d.cdof,
        ],
        outputs=[d.qfrc_gravcomp],
      )

    if m.has_fluid:
      passive._fluid(m, d)

    d.qfrc_adhesion.zero_()
    if m.flg_adhesion and (not (m.opt.disableflags & types.DisableBit.CONTACT)) and m.nv > 0:
      wp.launch(
        passive._qfrc_adhesion,
        dim=d.naconmax,
        inputs=[
          m.body_parentid,
          m.body_rootid,
          m.body_weldid,
          m.body_dofnum,
          m.body_dofadr,
          m.dof_bodyid,
          m.geom_bodyid,
          m.body_isdofancestor,
          d.subtree_com,
          d.cdof,
          d.contact.pos,
          d.contact.frame,
          d.contact.geom,
          d.contact.worldid,
          d.contact.adhesion,
          d.nacon,
        ],
        outputs=[
          d.qfrc_adhesion,
        ],
      )
