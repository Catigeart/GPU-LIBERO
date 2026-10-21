# Copyright 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Pinned rigid kinematics boundary; retain com_pos/camera/flex/tendon schedules."""
from mujoco_warp._src import forward,smooth
from libero.gpu.host.runtime.dynamics.mjwarp_kinematics import KinematicFrames,caller_unsupported_reason


def kinematics_product(root,*,fast_compile=False):
    import subprocess
    from libero.gpu.host.build.project_paths import project_paths
    from libero.gpu.host.build.compilation import MissingCompiledProduct
    folder=project_paths(root).cmake/('sm80-dev' if fast_compile else 'sm80-release')
    target='flash_mjwarp_kinematics_fp32';library=folder/'mjwarp_kinematics_fp32.so'
    result=subprocess.run(['ninja','-C',str(folder),'-n',target],capture_output=True,text=True)
    if not library.is_file() or result.returncode or 'ninja: no work to do.' not in result.stdout:
        raise MissingCompiledProduct('Build '+target+' offline in '+str(folder))
    return library


class Kinematics:
    def __init__(self,model,data,stream,*,implementation='upstream',library=None,topology=None):
        if implementation not in ('upstream','cuda'):raise ValueError('kinematics must be upstream or cuda')
        self.model,self.data=model,data;self.requested=implementation
        self.fallback=caller_unsupported_reason(model,data) if implementation=='cuda' else None
        self.selected='upstream' if self.fallback else implementation;self.binding=None
        if self.selected=='cuda':self.binding=KinematicFrames(model,data,library,stream,topology=topology)

    @property
    def identity(self):
        return dict(requested=self.requested,selected=self.selected,fallback=self.fallback,
            **(self.binding.identity if self.binding is not None else {}))

    def enqueue(self):
        if self.binding is None:smooth.kinematics(self.model,self.data);return
        for phase in range(5):self.binding.enqueue(phase)

    def forward(self, *, com_pos=None, camlight=None):
        m,d=self.model,self.data
        com_cuda = com_pos is not None and com_pos.selected == 'cuda'
        cam_cuda = camlight is not None and camlight.selected == 'cuda'
        if self.binding is None and not com_cuda and not cam_cuda:forward.fwd_kinematics(m,d);return
        self.enqueue()
        if com_cuda:com_pos.enqueue()
        else:smooth.com_pos(m,d)
        if cam_cuda:camlight.enqueue()
        else:smooth.camlight(m,d)
        smooth.flex(m,d)
        smooth.tendon(m,d)
