# Copyright 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Qualified rigid passive caller; only joint forces and final sum are replaced."""
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
    def __init__(self,model,data,stream,*,implementation='upstream',library=None):
        if implementation not in ('upstream','cuda'):raise ValueError('passive must be upstream or cuda')
        self.model,self.data=model,data;self.requested=implementation
        self.fallback=caller_unsupported_reason(model,data) if implementation=='cuda' else None
        self.selected='upstream' if self.fallback else implementation
        self.binding=PassiveForces(model,data,library,stream) if self.selected=='cuda' else None

    @property
    def identity(self):
        return dict(requested=self.requested,selected=self.selected,fallback=self.fallback,
            other_contributions='upstream',**(self.binding.identity if self.binding is not None else {}))

    def enqueue(self):
        m,d=self.model,self.data
        if self.binding is None:passive.passive(m,d);return
        if self.binding.flags['disable_spring'] and self.binding.flags['disable_damper']:
            # Pinned early return also skips other contributions and callback;
            # qfrc_adhesion is held rather than cleared in that branch.
            for key in ('qfrc_spring','qfrc_damper','qfrc_gravcomp','qfrc_fluid','qfrc_passive'):
                getattr(d,key).zero_()
            return
        self.binding.enqueue(0)
        other_forces(m,d)
        self.binding.enqueue(1)
        if m.callback.passive:m.callback.passive(m,d)
