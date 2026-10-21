# Copyright 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Literal retained _advance prefix/tail from the pinned MJWarp forward.py.

Only the intervening velocity/position kernels are replaced. History remains
before time/overflow publication; warmstart always copies public Data.qacc.
"""
import warp as wp
from mujoco_warp._src import history,sleep
from mujoco_warp._src.forward import _next_activation,_next_time_builder,fwd_velocity
from mujoco_warp._src.types import EnableBit,DisableBit


def advance_activation(m, d):
    wp.launch(
        _next_activation, dim=(d.nworld, m.nactuator),
        inputs=[
            m.opt.timestep, m.jnt_type, m.actuator_trntype, m.actuator_dyntype, m.actuator_gaintype, m.actuator_biastype,
            m.actuator_actadr, m.actuator_actnum, m.actuator_trnid,
            m.actuator_dynprm, m.actuator_gainprm, m.actuator_biasprm,
            m.actuator_actlimited, m.actuator_actrange, m.actuator_gear,
            d.act, d.act_dot, d.actuator_length, d.actuator_velocity, 1.0, True],
        outputs=[d.act])


def publish_advance(m, d):
    history.insert_ctrl_history(m, d)
    wp.launch(
        _next_time_builder(int(m.opt.warn_overflow)), dim=d.nworld,
        inputs=[
            m.opt.timestep, m.is_sparse, d.nefc, d.time,
            d.efc.J_rownnz, d.efc.J_rowadr, d.nworld, d.naconmax, d.njmax,
            d.njmax_nnz, d.nacon, d.ncollision],
        outputs=[d.time, d.overflow])
    wp.copy(d.qacc_warmstart, d.qacc)
    sleep_enabled = bool(m.opt.enableflags & EnableBit.SLEEP) and (not bool(m.opt.disableflags & DisableBit.ISLAND))
    if sleep_enabled:
        sleep.sleep(m, d)
        fwd_velocity(m, d)
        sleep.update_sleep(m, d)
