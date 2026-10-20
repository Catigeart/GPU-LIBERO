# Copyright 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Pinned stateless actuation schedule with explicit CUDA numerical leaves."""
from mujoco_warp._src import forward,history,types
from libero.gpu.host.runtime.dynamics.mjwarp_actuation import ActuatorForces,qualify


def actuation_product(root,*,fast_compile=False):
    import subprocess
    from libero.gpu.host.build.project_paths import project_paths
    from libero.gpu.host.build.compilation import MissingCompiledProduct
    folder=project_paths(root).cmake/('sm80-dev' if fast_compile else 'sm80-release')
    target='flash_mjwarp_actuation_fp32';library=folder/'mjwarp_actuation_fp32.so'
    result=subprocess.run(['ninja','-C',str(folder),'-n',target],capture_output=True,text=True)
    if not library.is_file() or result.returncode or 'ninja: no work to do.' not in result.stdout:
        raise MissingCompiledProduct('Build '+target+' offline in '+str(folder))
    return library


class Actuation:
    def __init__(self,model,data,stream,*,implementation='upstream',library=None,admission=None):
        if implementation not in ('upstream','cuda'):raise ValueError('actuation must be upstream or cuda')
        self.model,self.data=model,data;self.requested=implementation
        if implementation=='cuda':
            admission=qualify(model,data) if admission is None else admission
            if admission['model']!=id(model) or admission['data']!=id(data):raise ValueError('Actuation admission owner differs')
            self.fallback=admission['reason']
        else:self.fallback=None
        self.selected='upstream' if self.fallback else implementation
        self.binding=self.control=None
        if self.selected=='cuda':
            import warp as wp
            # Delay interpolation stays upstream; stable workspace ownership
            # replaces only its per-capture allocation, not its mathematics.
            self.control=wp.empty_like(data.ctrl) if model.nhistory>0 else data.ctrl
            self.binding=ActuatorForces(model,data,library,stream,admission=admission,control=self.control)

    @property
    def identity(self):
        return dict(requested=self.requested,selected=self.selected,fallback=self.fallback,
            control='upstream-history' if self.model.nhistory>0 else 'direct',
            control_workspace_bytes=self.control.capacity if self.binding is not None and self.model.nhistory>0 else 0,
            **(self.binding.identity if self.binding is not None else {}))

    def enqueue(self):
        m,d=self.model,self.data
        if self.binding is None:forward.fwd_actuation(m,d);return
        if m.opt.disableflags & types.DisableBit.ACTUATION:
            d.act_dot.zero_();d.qfrc_actuator.zero_();d.actuator_force.zero_();return
        if m.nhistory>0:history.read_ctrl_delayed(m,d,self.control)
        self.binding.enqueue(0)
        if m.callback.act_dyn:m.callback.act_dyn(m,d)
        if m.callback.act_gain:m.callback.act_gain(m,d)
        if m.callback.act_bias:m.callback.act_bias(m,d)
        self.binding.enqueue(1)
        self.binding.enqueue(2)
