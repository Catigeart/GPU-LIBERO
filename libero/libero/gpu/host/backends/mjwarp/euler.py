# Copyright 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Independent Euler numerical selection; existing mass factor work is reused."""
from types import SimpleNamespace
import warp as wp
from mujoco_warp._src import smooth,types
from libero.gpu.host.runtime.dynamics.mjwarp_euler import EulerNumerics,caller_unsupported_reason
from .euler_other import advance_activation,publish_advance


def euler_product(root,*,fast_compile=False):
    from libero.gpu.host.build.project_paths import project_paths
    from libero.gpu.host.build.compilation import MissingCompiledProduct
    folder=project_paths(root).cmake/('sm80-dev' if fast_compile else 'sm80-release')
    target='flash_mjwarp_euler_fp32';library=folder/'mjwarp_euler_fp32.so'
    if not library.is_file():
        raise MissingCompiledProduct('Build '+target+' offline in '+str(folder))
    return library


class Euler:
    def __init__(self,model,data,stream,*,mass_factor,implementation='upstream',library=None,publication=None):
        if implementation not in ('upstream','cuda'):raise ValueError('euler must be upstream or cuda')
        self.model,self.data,self.stream,self.mass_factor=model,data,stream,mass_factor
        self.publication=publication
        self.requested=implementation
        self.fallback=caller_unsupported_reason(model,data) if implementation=='cuda' else None
        self.selected='upstream' if self.fallback else implementation
        self.binding=self.workspace=None;self.owns_workspace=False
        self.damped=not bool(model.opt.disableflags & (types.DisableBit.EULERDAMP|types.DisableBit.DAMPER))
        if self.selected=='cuda':
            if self.damped:
                self.workspace=mass_factor.euler_workspace
                if self.workspace is None:
                    self.workspace=SimpleNamespace(matrix=wp.empty_like(data.M),factor=wp.empty_like(data.qLD),
                        diagonal=wp.empty_like(data.qLDiagInv),acceleration=wp.empty_like(data.qacc),damping=wp.empty_like(data.qvel))
                    self.owns_workspace=True
            self.binding=EulerNumerics(model,data,library,stream,workspace=self.workspace)

    @property
    def identity(self):
        return dict(requested=self.requested,selected=self.selected,fallback=self.fallback,damped=self.damped,
            factor=self.mass_factor.selected,workspace='owned' if self.owns_workspace else 'mass-factor-shared' if self.workspace is not None else 'none',
            owned_workspace_bytes=sum(a.capacity for a in vars(self.workspace).values()) if self.owns_workspace else 0,
            **(self.binding.identity if self.binding is not None else {}))

    def integrate(self):
        if self.binding is None:self.mass_factor.integrate();return
        m,d,w=self.model,self.data,self.workspace
        if self.damped:
            self.binding.enqueue(0)
            wp.copy(w.matrix,d.M)
            self.binding.enqueue(1)
            if self.mass_factor.euler_binding is not None:self.mass_factor.euler_binding.enqueue()
            else:smooth.factor_solve_i(m,d,w.matrix,w.factor,w.diagonal,w.acceleration,d.efc.Ma)
        if self.publication is None:advance_activation(m,d)
        else:self.publication.activation()
        self.binding.enqueue(2)
        self.binding.enqueue(3)
        if self.publication is None:publish_advance(m,d)
        else:self.publication.publish()
