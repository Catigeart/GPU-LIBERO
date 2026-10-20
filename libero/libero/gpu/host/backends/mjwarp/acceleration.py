# Copyright 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Explicit pinned acceleration schedule, with separately qualified CUDA leaves."""
from mujoco_warp._src import forward,types
import warp as wp

from libero.gpu.host.runtime.dynamics.mjwarp_mass_solve import MassSolvePlan,unsupported_reason


def caller_unsupported_reason(model,data):
    if model.opt.integrator==types.IntegratorType.DISCRETE:return 'discrete effective metric'
    if model.opt.enableflags & types.EnableBit.SLEEP:return 'active-DOF compaction'
    return unsupported_reason(model,data)


def mass_product(root,*,fast_compile=False,component='mass_solve'):
    import subprocess
    from libero.gpu.host.build.project_paths import project_paths
    from libero.gpu.host.build.compilation import MissingCompiledProduct
    folder=project_paths(root).cmake/('sm80-dev' if fast_compile else 'sm80-release')
    if component not in ('mass_solve','smooth_force'):raise ValueError('invalid acceleration product')
    target='flash_mjwarp_'+component+'_fp32'
    library=folder/('mjwarp_'+component+'_fp32.so')
    result=subprocess.run(['ninja','-C',str(folder),'-n',target],capture_output=True,text=True)
    if not library.is_file() or result.returncode or 'ninja: no work to do.' not in result.stdout:
        raise MissingCompiledProduct('Build '+target+' offline in '+str(folder))
    return library


class Acceleration:
    """Select complete acceleration, mass-only CUDA, or whole upstream fallback."""

    def __init__(self,model,data,stream,*,implementation='upstream',library=None,
                 force_implementation='upstream',force_library=None):
        if implementation not in ('upstream','cuda'):raise ValueError('invalid mass solve implementation')
        if force_implementation not in ('upstream','cuda'):raise ValueError('invalid force implementation')
        if force_implementation=='cuda' and implementation!='cuda':
            raise ValueError('CUDA acceleration requires CUDA mass solve')
        self.model,self.data,self.stream=model,data,stream
        self.requested=implementation
        self.fallback=caller_unsupported_reason(model,data) if implementation=='cuda' else None
        self.selected='upstream' if self.fallback else implementation
        self.plan=MassSolvePlan(model,data,library,stream) if self.selected=='cuda' else None
        self.binding=self.plan.bind(data.qacc_smooth,data.qfrc_smooth) if self.plan is not None else None
        self.force_requested=force_implementation
        self.force=None
        if force_implementation=='cuda' and self.selected=='cuda':
            from libero.gpu.host.runtime.dynamics.mjwarp_smooth_force import SmoothForce
            self.force=SmoothForce(model,data,force_library,stream)

    @property
    def identity(self):
        result=dict(requested=self.requested,selected=self.selected,fallback=self.fallback)
        if self.plan is not None:result.update(self.plan.identity)
        if self.force_requested=='cuda':
            result['forces']=dict(requested='cuda',selected='cuda' if self.force is not None else 'upstream',
                                  fallback=self.fallback)
            if self.force is not None:result['forces'].update(self.force.identity)
        return result

    def enqueue(self):
        m,d=self.model,self.data
        if self.selected=='upstream':forward.fwd_acceleration(m,d);return
        if self.force is not None:
            self.force.enqueue()
            self.binding.enqueue()
            return
        # Exact pinned dense/full-DOF/non-discrete scheduling. No duplicate
        # upstream solve_m is hidden behind a call to fwd_acceleration.
        wp.launch(forward._qfrc_smooth(False),dim=(d.nworld,m.nv),
            inputs=[m.body_treeid,m.dof_bodyid,d.qfrc_applied,d.tree_awake,d.qfrc_bias,d.qfrc_passive,d.qfrc_actuator],
            outputs=[d.qfrc_smooth])
        forward.xfrc_accumulate(m,d,d.qfrc_smooth)
        self.binding.enqueue()
