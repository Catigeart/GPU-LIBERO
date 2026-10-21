# Copyright 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Independent selectors for position transmission and velocity-stage CSR dot."""
import warp as wp
from mujoco_warp._src import forward,smooth
from libero.gpu.host.runtime.dynamics.mjwarp_transmission import TransmissionNumerics,qualify


def transmission_product(root,*,fast_compile=False):
    import subprocess
    from libero.gpu.host.build.project_paths import project_paths
    from libero.gpu.host.build.compilation import MissingCompiledProduct
    folder=project_paths(root).cmake/('sm80-dev' if fast_compile else 'sm80-release')
    target='flash_mjwarp_transmission_fp32';library=folder/'mjwarp_transmission_fp32.so'
    result=subprocess.run(['ninja','-C',str(folder),'-n',target],capture_output=True,text=True)
    if not library.is_file() or result.returncode or 'ninja: no work to do.' not in result.stdout:
        raise MissingCompiledProduct('Build '+target+' offline in '+str(folder))
    return library


class Transmission:
    def __init__(self,model,data,stream,*,transmission='upstream',actuator_velocity='upstream',library=None,admission=None):
        self.model,self.data=model,data
        self.admission=qualify(model,data,transmission=transmission,actuator_velocity=actuator_velocity) if admission is None else admission
        if self.admission['model']!=id(model) or self.admission['data']!=id(data):raise ValueError('Transmission admission owner differs')
        if self.admission['requested']!=dict(transmission=transmission,actuator_velocity=actuator_velocity):raise ValueError('Transmission admission selectors differ')
        self.selected=self.admission['selected'];self.binding=None
        if 'cuda' in self.selected.values():self.binding=TransmissionNumerics(model,data,library,stream,admission=self.admission)

    @property
    def identity(self):
        return dict(implementations={k:dict(requested=self.admission['requested'][k],selected=v,fallback=self.admission['reasons'][k]) for k,v in self.selected.items()},
            **(self.binding.identity if self.binding is not None else {}))

    def enqueue_transmission(self):
        if self.selected['transmission']=='cuda':self.binding.enqueue(0)
        else:smooth.transmission(self.model,self.data)

    def enqueue_velocity(self):
        if self.selected['actuator_velocity']=='cuda':self.binding.enqueue(1)
        else:
            m,d=self.model,self.data
            wp.launch(forward._actuator_velocity,dim=(d.nworld,m.nactuator),
                inputs=[d.qvel,d.moment_rownnz,d.moment_rowadr,d.moment_colind,d.actuator_moment],outputs=[d.actuator_velocity])
