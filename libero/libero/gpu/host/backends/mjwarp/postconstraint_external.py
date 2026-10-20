# Copyright 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Unchanged external-wrench prefix of pinned smooth.rne_postconstraint.

Source: MJWarp c384ac51df67eb92ab7657cc472ff7494dc036bf. Only kernel names
are qualified with smooth.; no force mathematics or scheduling is replaced.
This literal host schedule provides an explicit boundary for the CUDA suffix.
"""
import warp as wp
from mujoco_warp._src import smooth, types


def assemble_external(m, d):
    # cfrc_ext = perturb
    wp.launch(
      smooth._cfrc_ext,
      dim=(d.nworld, m.nbody),
      inputs=[m.body_rootid, d.xfrc_applied, d.xipos, d.subtree_com],
      outputs=[d.cfrc_ext],
    )

    # Equality constraint forces - only if model has equality constraints
    if m.neq > 0:
      # Allocate inline counters and count from efc data
      ne_connect = wp.zeros((d.nworld,), dtype=int)
      ne_weld = wp.zeros((d.nworld,), dtype=int)

      wp.launch(
        smooth._count_equality_constraints,
        dim=(d.nworld, d.njmax),  # TODO(team): launch over max equality constraints
        inputs=[m.eq_type, d.ne, d.efc.type, d.efc.id],
        outputs=[ne_connect, ne_weld],
      )

      wp.launch(
        smooth._cfrc_ext_equality,
        dim=(d.nworld, m.neq),
        inputs=[
          m.body_rootid,
          m.site_bodyid,
          m.site_pos,
          m.site_quat,
          m.eq_obj1id,
          m.eq_obj2id,
          m.eq_objtype,
          m.eq_data,
          d.xpos,
          d.xquat,
          d.xmat,
          d.subtree_com,
          d.efc.id,
          d.efc.force,
          ne_connect,
          ne_weld,
        ],
        outputs=[d.cfrc_ext],
      )

    # cfrc_ext += contacts
    wp.launch(
      smooth._cfrc_ext_contact,
      dim=(d.naconmax,),
      inputs=[
        m.opt.cone,
        m.body_rootid,
        m.geom_bodyid,
        d.subtree_com,
        d.contact.pos,
        d.contact.frame,
        d.contact.friction,
        d.contact.dim,
        d.contact.geom,
        d.contact.efc_address,
        d.contact.worldid,
        d.contact.adhesion,
        d.efc.force,
        d.njmax,
        d.nacon,
      ],
      outputs=[d.cfrc_ext],
    )

    # cfrc_ext += spatial tendons
    if m.ntendon > 0:
      ten_frc = wp.zeros((d.nworld, m.ntendon), dtype=float)

      wp.launch(
        smooth._cfrc_ext_tendon_constraint,
        dim=(d.nworld, d.njmax),
        inputs=[
          m.eq_type,
          m.eq_obj1id,
          m.eq_obj2id,
          m.eq_data,
          m.tendon_range,
          m.tendon_length0,
          d.nefc,
          d.ten_length,
          d.efc.type,
          d.efc.id,
          d.efc.force,
        ],
        outputs=[ten_frc],
      )

      wp.launch(
        smooth._cfrc_ext_tendon_actuator,
        dim=(d.nworld, m.nu),
        inputs=[
          m.actuator_trntype,
          m.actuator_trnid,
          m.actuator_gear,
          d.actuator_force,
        ],
        outputs=[ten_frc],
      )

      ten_bias_coef = wp.zeros((d.nworld, m.ntendon), dtype=float)
      if m.nJten > 0:
        ten_Jdot = wp.zeros((d.nworld, m.nJten), dtype=float)
        wp.launch(
          smooth._tendon_dot,
          dim=(d.nworld, m.ntendon),
          inputs=[
            m.body_parentid,
            m.body_rootid,
            m.body_dofnum,
            m.body_dofadr,
            m.jnt_type,
            m.jnt_dofadr,
            m.dof_jntid,
            m.site_bodyid,
            m.tendon_adr,
            m.tendon_num,
            m.ten_J_rownnz,
            m.ten_J_rowadr,
            m.ten_J_colind,
            m.tendon_armature,
            m.wrap_type,
            m.wrap_objid,
            m.wrap_prm,
            d.site_xpos,
            d.subtree_com,
            d.cdof,
            d.cvel,
            d.cdof_dot,
          ],
          outputs=[ten_Jdot],
        )

        wp.launch(
          smooth._tendon_bias_coef,
          dim=(d.nworld, m.ntendon, m.max_ten_J_rownnz),
          inputs=[m.ten_J_rownnz, m.ten_J_rowadr, m.ten_J_colind, m.tendon_armature, d.qvel, ten_Jdot],
          outputs=[ten_bias_coef],
        )

      dsbl_spring = bool(m.opt.disableflags & types.DisableBit.SPRING)
      dsbl_damper = bool(m.opt.disableflags & types.DisableBit.DAMPER)

      wp.launch(
        smooth._cfrc_ext_spatial_tendon,
        dim=(d.nworld, m.ntendon),
        inputs=[
          m.body_rootid,
          m.geom_bodyid,
          m.site_bodyid,
          m.tendon_adr,
          m.tendon_num,
          m.ten_J_rownnz,
          m.ten_J_rowadr,
          m.ten_J_colind,
          m.tendon_stiffness,
          m.tendon_stiffnesspoly,
          m.tendon_damping,
          m.tendon_dampingpoly,
          m.tendon_armature,
          m.tendon_lengthspring,
          m.wrap_type,
          m.wrap_objid,
          m.wrap_prm,
          d.qacc,
          d.subtree_com,
          d.ten_wrapadr,
          d.ten_wrapnum,
          d.ten_J,
          d.ten_length,
          d.wrap_obj,
          d.wrap_xpos,
          d.ten_velocity,
          ten_bias_coef,
          dsbl_spring,
          dsbl_damper,
          ten_frc,
        ],
        outputs=[d.cfrc_ext],
      )
