# Copyright 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Rigid passive schedule with independent joint/sum and fluid selection."""
import warp as wp
from mujoco_warp._src import passive,types
from libero.gpu.host.runtime.dynamics.mjwarp_passive import PassiveForces
from .passive_other import other_forces


def caller_unsupported_reason(model,data):
    if model.opt.integrator!=types.IntegratorType.EULER:return 'passive caller requires Euler'
    if model.opt.enableflags & types.EnableBit.SLEEP:return 'active-body compaction'
    if min(model.nv,model.nq,model.njnt)<1:return 'empty passive topology'
    if model.ntendon:return 'tendon passive forces retained upstream'
    if any(getattr(model,key) for key in ('nflex','nflexedge','nflexelem','nflexbend_interp','nflexintcell')):
        return 'flex passive forces retained upstream'
    return None


class Passive:
    def __init__(self,model,data,stream,*,implementation='upstream',library=None,fluid=None,queries=None):
        if implementation not in ('upstream','cuda'):raise ValueError('passive must be upstream or cuda')
        self.model,self.data=model,data;self.requested=implementation
        self.fluid=fluid;self.queries=queries
        self.fallback=caller_unsupported_reason(model,data) if implementation=='cuda' else None
        self.selected='upstream' if self.fallback else implementation
        self.binding=PassiveForces(model,data,library,stream) if self.selected=='cuda' else None

    @property
    def uses_cuda(self):
        return self.binding is not None or (self.queries is not None and self.queries.gravity is not None) or (self.fluid is not None and self.fluid.selected=='cuda')

    @property
    def identity(self):
        return dict(requested=self.requested,selected=self.selected,fallback=self.fallback,
            other_contributions=dict(fluid=self.fluid.selected if self.fluid is not None else 'upstream',gravity=self.queries.selected if self.queries is not None else 'upstream',adhesion='upstream'),
            **(self.binding.identity if self.binding is not None else {}))

    def enqueue(self):
        m,d=self.model,self.data
        if not self.uses_cuda:passive.passive(m,d);return
        flags=self.binding.flags if self.binding is not None else dict(
            disable_spring=bool(m.opt.disableflags & types.DisableBit.SPRING),
            disable_damper=bool(m.opt.disableflags & types.DisableBit.DAMPER))
        if flags['disable_spring'] and flags['disable_damper']:
            # Pinned early return also skips other contributions and callback;
            # qfrc_adhesion is held rather than cleared in that branch.
            for key in ('qfrc_spring','qfrc_damper','qfrc_gravcomp','qfrc_fluid','qfrc_passive'):
                getattr(d,key).zero_()
            return
        if self.binding is not None:self.binding.enqueue(0)
        else:
            wp.launch(passive._spring_damper_dof_passive,dim=(d.nworld,m.njnt),
                inputs=[m.opt.disableflags,m.qpos_spring,m.jnt_type,m.jnt_qposadr,m.jnt_dofadr,m.jnt_stiffness,
                        m.jnt_stiffnesspoly,m.dof_damping,m.dof_dampingpoly,d.qpos,d.qvel],
                outputs=[d.qfrc_spring,d.qfrc_damper])
        other_forces(m,d,fluid=self.fluid,gravity=self.queries.gravity if self.queries is not None else None)
        if self.binding is not None:self.binding.enqueue(1)
        else:
            gravity_enabled=not (m.opt.disableflags & types.DisableBit.GRAVITY)
            wp.launch(passive._qfrc_passive_kernel(m.has_fluid,m.flg_adhesion,gravity_enabled),dim=(d.nworld,m.nv),
                inputs=[m.jnt_actgravcomp,m.dof_jntid,d.qfrc_spring,d.qfrc_damper,d.qfrc_gravcomp,d.qfrc_fluid,d.qfrc_adhesion],
                outputs=[d.qfrc_passive])
        if m.callback.passive:m.callback.passive(m,d)
