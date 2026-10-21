# Copyright 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Pinned ordinary position/Euler schedules with explicit mass-factor leaves."""
from types import SimpleNamespace
import warp as wp
from mujoco_warp._src import forward,smooth,types
from libero.gpu.host.runtime.dynamics.mjwarp_mass_factor import MassFactorPlan,unsupported_reason


def caller_unsupported_reason(model,data):
    if model.opt.integrator!=types.IntegratorType.EULER:return 'mass-factor caller requires Euler'
    if model.opt.enableflags & types.EnableBit.SLEEP:return 'active-DOF compaction'
    return unsupported_reason(model,data)


def factor_product(root,*,fast_compile=False):
    import subprocess
    from libero.gpu.host.build.project_paths import project_paths
    from libero.gpu.host.build.compilation import MissingCompiledProduct
    folder=project_paths(root).cmake/('sm80-dev' if fast_compile else 'sm80-release')
    target='flash_mjwarp_mass_factor_fp32';library=folder/'mjwarp_mass_factor_fp32.so'
    result=subprocess.run(['ninja','-C',str(folder),'-n',target],capture_output=True,text=True)
    if not library.is_file() or result.returncode or 'ninja: no work to do.' not in result.stdout:
        raise MissingCompiledProduct('Build '+target+' offline in '+str(folder))
    return library


class MassFactors:
    """One factor plan, public position factors, and private Euler scratch."""
    def __init__(self,model,data,stream,*,implementation='upstream',library=None):
        if implementation not in ('upstream','cuda'):raise ValueError('invalid mass factor implementation')
        self.model,self.data,self.stream=model,data,stream
        self.requested=implementation
        self.fallback=caller_unsupported_reason(model,data) if implementation=='cuda' else None
        self.selected='upstream' if self.fallback else implementation
        self.plan=self.position_binding=self.public_binding=self.euler_binding=self.euler_workspace=None
        self.damped=not bool(model.opt.disableflags & (types.DisableBit.EULERDAMP|types.DisableBit.DAMPER))
        if self.selected=='cuda':
            self.plan=MassFactorPlan(model,data,library,stream)
            self.position_binding=self.plan.bind(data.M,data.qLD,data.qLDiagInv)
            if self.damped:
                # All substeps on this stream execute serially. These private
                # buffers are overwritten before each use and survive the Graph.
                w=SimpleNamespace(matrix=wp.empty_like(data.M),factor=wp.empty_like(data.qLD),
                    diagonal=wp.empty_like(data.qLDiagInv),acceleration=wp.empty_like(data.qacc),
                    damping=wp.empty_like(data.qvel))
                self.euler_workspace=w
                self.euler_binding=self.plan.bind(w.matrix,w.factor,w.diagonal,w.acceleration,data.efc.Ma)

    @property
    def identity(self):
        result=dict(requested=self.requested,selected=self.selected,fallback=self.fallback,
                    euler_damping=self.damped)
        if self.plan is not None:
            result.update(self.plan.identity)
            result['euler_workspace_bytes']=(sum(a.capacity for a in vars(self.euler_workspace).values())
                if self.euler_workspace is not None else 0)
        return result

    def prepare_public(self):
        if self.plan is not None and self.public_binding is None:
            d=self.data
            self.public_binding=self.plan.bind(d.M,d.qLD,d.qLDiagInv,d.qacc_smooth,d.qfrc_smooth)

    def position(self):
        # Compatibility entry; the runtime uses the common position schedule.
        from .position import Position
        Position(self.model,self.data,mass_factor=self).position()

    def integrate(self):
        m,d=self.model,self.data
        if self.selected!='cuda':
            if m.opt.integrator in (types.IntegratorType.IMPLICITFAST,types.IntegratorType.IMPLICIT):forward.implicit(m,d)
            elif m.opt.integrator==types.IntegratorType.DISCRETE:forward.discrete(m,d)
            else:forward.euler(m,d)
            return
        if not self.damped:forward._advance(m,d,d.qacc);return
        w=self.euler_workspace
        wp.launch(forward._compute_damping_deriv,dim=(d.nworld,m.nv),
            inputs=[m.dof_damping,m.dof_dampingpoly,d.qvel],outputs=[w.damping])
        wp.copy(w.matrix,d.M)
        wp.launch(forward._euler_damp_qfrc,dim=(d.nworld,m.nv),
            inputs=[m.opt.timestep,m.M_rownnz,m.M_rowadr,w.damping],outputs=[w.matrix])
        self.euler_binding.enqueue()
        # _advance still copies d.qacc to warmstart; do not publish the private
        # damping-adjusted acceleration as Newton's public qacc.
        forward._advance(m,d,w.acceleration)
