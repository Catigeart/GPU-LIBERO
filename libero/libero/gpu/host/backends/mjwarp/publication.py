# Copyright 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Stateless/no-history Euler suffix, with whole-suffix fallback."""
import warp as wp
from mujoco_warp._src import types
from .euler_other import advance_activation,publish_advance
from libero.gpu.host.runtime.dynamics.mjwarp_publication import PublicationState


def publication_product(root,*,fast_compile=False):
    from libero.gpu.host.build.project_paths import project_paths
    from libero.gpu.host.build.compilation import MissingCompiledProduct
    folder=project_paths(root).cmake/('sm80-dev' if fast_compile else 'sm80-release');path=folder/'mjwarp_publication_fp32.so';target='flash_mjwarp_publication_fp32'
    if not path.is_file():raise MissingCompiledProduct('Build '+target+' offline in '+str(folder))
    return path


def unsupported_reason(m):
    if m.na:return 'activation state remains upstream'
    if m.nhistory:return 'history remains upstream'
    if m.opt.enableflags & types.EnableBit.SLEEP and not (m.opt.disableflags & types.DisableBit.ISLAND):return 'sleep remains upstream'
    return None


class Publication:
    def __init__(self,m,d,stream,*,implementation='upstream',library=None):
        if implementation not in ('upstream','cuda'):raise ValueError('publication must be upstream or cuda')
        self.model,self.data=m,d;self.requested=implementation
        self.fallback=unsupported_reason(m) if implementation=='cuda' else None;self.selected='upstream' if self.fallback else implementation
        self.binding=PublicationState(m,d,library,stream) if self.selected=='cuda' else None
    @property
    def identity(self):return dict(requested=self.requested,selected=self.selected,fallback=self.fallback,activation='empty' if self.binding is not None else 'upstream',history='empty' if self.binding is not None else 'upstream',**(self.binding.identity if self.binding is not None else {}))
    def activation(self):
        if self.binding is None:advance_activation(self.model,self.data)
    def publish(self):
        if self.binding is None:publish_advance(self.model,self.data)
        else:
            self.binding.enqueue();wp.copy(self.data.qacc_warmstart,self.data.qacc)
