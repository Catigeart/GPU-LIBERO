# Copyright 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Pinned velocity schedule with independently selected com_vel and bias RNE."""
import warp as wp
from mujoco_warp._src import forward, smooth, passive, types
from libero.gpu.host.runtime.dynamics.mjwarp_motion import MotionTopology
from libero.gpu.host.runtime.dynamics.mjwarp_velocity import ComVelocity
from libero.gpu.host.runtime.dynamics.mjwarp_bias import Bias


def caller_unsupported_reason(model,data):
    if model.opt.integrator!=types.IntegratorType.EULER:return 'motion caller requires Euler'
    if model.opt.enableflags & types.EnableBit.SLEEP:return 'active-body compaction'
    if model.nv<1 or model.nbody<2:return 'empty motion topology'
    return None


def velocity_product(root,*,arch,component='velocity',fast_compile=False):
    from libero.gpu.host.build.project_paths import cmake_directory
    from libero.gpu.host.build.compilation import MissingCompiledProduct
    if component not in ('velocity','bias','postconstraint','passive'):raise ValueError('invalid motion product')
    folder=cmake_directory(root, arch, fast_compile=fast_compile)
    target=f'flash_mjwarp_{component}_fp32';library=folder/f'mjwarp_{component}_fp32.so'
    if not library.is_file():
        raise MissingCompiledProduct('Build '+target+' offline in '+str(folder))
    return library


class Velocity:
    def __init__(self,model,data,stream,*,com_vel='upstream',bias='upstream',velocity_library=None,bias_library=None,passive_driver=None,transmission=None):
        if com_vel not in ('upstream','cuda') or bias not in ('upstream','cuda'):
            raise ValueError('com_vel and bias must be upstream or cuda')
        self.model,self.data=model,data
        self.passive_driver=passive_driver
        self.transmission=transmission
        self.requested=dict(com_vel=com_vel,bias=bias)
        self.fallback=caller_unsupported_reason(model,data) if 'cuda' in self.requested.values() else None
        self.selected={k:'upstream' if self.fallback else v for k,v in self.requested.items()}
        self.topology=MotionTopology(model,data,stream) if 'cuda' in self.selected.values() else None
        self.com_vel=ComVelocity(model,data,velocity_library,stream,self.topology) if self.selected['com_vel']=='cuda' else None
        self.bias=Bias(model,data,bias_library,stream,self.topology) if self.selected['bias']=='cuda' else None

    @property
    def identity(self):
        result={k:dict(requested=self.requested[k],selected=self.selected[k],fallback=self.fallback,
                **(getattr(self,k).identity if getattr(self,k) is not None else {})) for k in self.requested}
        if self.passive_driver is not None:result['passive']=self.passive_driver.identity
        return result

    def enqueue(self):
        m,d=self.model,self.data
        velocity_cuda=self.transmission is not None and self.transmission.selected['actuator_velocity']=='cuda'
        passive_cuda=self.passive_driver is not None and getattr(self.passive_driver,'uses_cuda',self.passive_driver.selected=='cuda')
        if self.com_vel is None and self.bias is None and not passive_cuda and not velocity_cuda:
            forward.fwd_velocity(m,d);return
        if velocity_cuda:self.transmission.enqueue_velocity()
        else:
            wp.launch(forward._actuator_velocity,dim=(d.nworld,m.nactuator),
                inputs=[d.qvel,d.moment_rownnz,d.moment_rowadr,d.moment_colind,d.actuator_moment],outputs=[d.actuator_velocity])
        wp.launch(forward._tendon_velocity,dim=(d.nworld,m.ntendon),
            inputs=[m.ten_J_rownnz,m.ten_J_rowadr,m.ten_J_colind,d.qvel,d.ten_J],outputs=[d.ten_velocity])
        if self.com_vel is None:smooth.com_vel(m,d)
        else:self.com_vel.enqueue()
        if self.passive_driver is None:passive.passive(m,d)
        else:self.passive_driver.enqueue()
        if self.bias is None:smooth.rne(m,d)
        else:self.bias.enqueue()
        smooth.tendon_bias(m,d,d.qfrc_bias)
