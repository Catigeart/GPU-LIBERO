# Copyright 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Current rigid queries; optional sensor domain retains a visible fallback."""
from mujoco_warp._src import sensor,forward,types
from libero.gpu.host.runtime.dynamics.mjwarp_queries import Jacobian,ForceSensors,GravityProjection


def queries_product(root,*,fast_compile=False):
    import subprocess
    from libero.gpu.host.build.project_paths import project_paths
    from libero.gpu.host.build.compilation import MissingCompiledProduct
    folder=project_paths(root).cmake/('sm80-dev' if fast_compile else 'sm80-release')
    path=folder/'mjwarp_queries_fp32.so';target='flash_mjwarp_queries_fp32'
    check=subprocess.run(['ninja','-C',str(folder),'-n',target],capture_output=True,text=True)
    if not path.is_file() or check.returncode or 'ninja: no work to do.' not in check.stdout:raise MissingCompiledProduct('Build '+target+' offline in '+str(folder))
    return path


def sensor_reason(m):
    import numpy as np
    if m.nhistory:return 'sensor history/delay remains upstream'
    if np.any(~np.isin(m.sensor_type.numpy(),[4,5])):return 'only FORCE/TORQUE sensors are qualified'
    return None


class Queries:
    def __init__(self,m,d,stream,*,implementation='upstream',library=None):
        if implementation not in ('upstream','cuda'):raise ValueError('queries must be upstream or cuda')
        self.model,self.data,self.stream,self.library=m,d,stream,library;self.selected=implementation
        self.sensor_fallback=sensor_reason(m) if implementation=='cuda' else None
        self.sensors=ForceSensors(m,d,library,stream) if implementation=='cuda' and self.sensor_fallback is None else None
        self.gravity=GravityProjection(m,d,library,stream) if implementation=='cuda' else None
    @property
    def identity(self):
        return dict(selected=self.selected,jacobian=self.selected,gravity=self.selected,
            sensors='cuda' if self.sensors is not None else 'upstream',sensor_fallback=self.sensor_fallback,
            **(self.gravity.identity if self.gravity is not None else {}))
    def jacobian(self,point,body,jp,jr):
        return Jacobian(self.model,self.data,point,body,jp,jr,self.library,self.stream) if self.selected=='cuda' else None
    def sensor_pos(self):
        if self.sensors is None:sensor.sensor_pos(self.model,self.data)
        else:self._callback(types.Stage.POS)
    def sensor_vel(self):
        if self.sensors is None:sensor.sensor_vel(self.model,self.data)
        else:self._callback(types.Stage.VEL)
    def sensor_acc(self):
        if self.sensors is None:sensor.sensor_acc(self.model,self.data,skip_rne_postconstraint=True)
        else:
            self.sensors.enqueue();self._callback(types.Stage.ACC)
    def _callback(self,stage):
        m=self.model
        if not (m.opt.disableflags & types.DisableBit.SENSOR) and m.callback.sensor:m.callback.sensor(m,self.data,stage)
    def energy_pos(self):
        if self.model.opt.enableflags & types.EnableBit.ENERGY:forward._energy_pos(self.model,self.data)
        else:self.data.energy.zero_()
    def energy_vel(self):
        if self.model.opt.enableflags & types.EnableBit.ENERGY:forward._energy_vel(self.model,self.data)
