# Copyright 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Pinned postconstraint caller with independent external-wrench/RNE leaves."""
from mujoco_warp._src import smooth
from libero.gpu.host.runtime.dynamics.mjwarp_motion import MotionTopology
from libero.gpu.host.runtime.dynamics.mjwarp_postconstraint import PostconstraintRNE
from .velocity import caller_unsupported_reason
from .postconstraint_external import assemble_external


def upstream_suffix(m,d):
    """Literal pinned suffix, also used with independently selected CUDA inputs."""
    smooth._rne_cacc_world(m,d)
    smooth._rne_cacc_forward(m,d,flg_acc=True)
    smooth._rne_cfrc(m,d,flg_cfrc_ext=True)
    smooth._rne_cfrc_backward(m,d)


class Postconstraint:
    def __init__(self,model,data,stream,*,implementation='upstream',library=None,topology=None,external_wrench=None):
        if implementation not in ('upstream','cuda'):
            raise ValueError('postconstraint must be upstream or cuda')
        self.model,self.data=model,data
        self.external_wrench=external_wrench
        self.requested=implementation
        self.fallback=caller_unsupported_reason(model,data) if implementation=='cuda' else None
        self.selected='upstream' if self.fallback else implementation
        self.topology=self.binding=None
        if self.selected=='cuda':
            self.topology=topology if topology is not None else MotionTopology(model,data,stream)
            self.binding=PostconstraintRNE(model,data,library,stream,self.topology)

    @property
    def identity(self):
        return dict(requested=self.requested,selected=self.selected,fallback=self.fallback,
            external_wrenches=self.external_wrench.selected if self.external_wrench is not None else 'upstream',
            external_wrench=self.external_wrench.identity if self.external_wrench is not None else None,
            **(self.binding.identity if self.binding is not None else {}))

    def enqueue(self):
        external_cuda=self.external_wrench is not None and self.external_wrench.selected=='cuda'
        if self.binding is None and not external_cuda:smooth.rne_postconstraint(self.model,self.data);return
        if self.external_wrench is None:assemble_external(self.model,self.data)
        else:self.external_wrench.enqueue()
        if self.binding is None:upstream_suffix(self.model,self.data)
        else:self.binding.enqueue()
