"""Pinned postconstraint caller with independently selected RNE suffix."""
from mujoco_warp._src import smooth
from libero.gpu.host.runtime.dynamics.mjwarp_motion import MotionTopology
from libero.gpu.host.runtime.dynamics.mjwarp_postconstraint import PostconstraintRNE
from .velocity import caller_unsupported_reason
from .postconstraint_external import assemble_external


class Postconstraint:
    def __init__(self,model,data,stream,*,implementation='upstream',library=None,topology=None):
        if implementation not in ('upstream','cuda'):
            raise ValueError('postconstraint must be upstream or cuda')
        self.model,self.data=model,data
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
            external_wrenches='upstream',**(self.binding.identity if self.binding is not None else {}))

    def enqueue(self):
        if self.binding is None:smooth.rne_postconstraint(self.model,self.data);return
        assemble_external(self.model,self.data)
        self.binding.enqueue()
