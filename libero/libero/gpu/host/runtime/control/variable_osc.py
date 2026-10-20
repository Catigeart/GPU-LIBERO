"""Cold variable-impedance OSC storage and original action prefix boundary."""
import libero.gpu.precision as scalar
import numpy as np
import warp as wp
from libero.gpu.host.assembly.pipeline.osc import FusedRuntimeControllerAdapter

def prepare_variable_osc(c,cfg):
    mode={"fixed":0,"variable_kp":1,"variable":2}[cfg['impedance_mode']]
    c.variable_mode=mode
    if mode==0:return c
    delta=3 if cfg['type']=='OSC_POSITION' else 6
    c.raw_action_dim=6*mode+delta+int(getattr(c,'gripper_dof',1))
    def bound(value):
        a=np.asarray(value,np.float64)
        if a.shape==(2,):a=np.broadcast_to(a[:,None],(2,6))
        if a.shape!=(2,6) or not np.isfinite(a).all() or np.any(a[0]>a[1]):raise ValueError('ordered finite six-dimensional impedance limits required')
        return a.copy()
    kp=bound(cfg.get('kp_limits',(0,300)))
    damping=bound(cfg.get('damping_ratio_limits',(0,100)))
    if np.any(kp[0]<0):raise ValueError('nonnegative variable OSC kp required')
    c.WIDTHS=dict(c.WIDTHS)
    c.WIDTHS.update(current_gains=12,phys_action=c.raw_action_dim,phys_impedance_limits=24,phys_initial_gains=12)
    c.current_gains=c.gains
    c.phys_action=wp.zeros((c.nworld,c.raw_action_dim),dtype=scalar.wp_real,device=c.device)
    c.phys_impedance_limits=wp.array(np.concatenate([*kp,*damping]),dtype=scalar.wp_real,device=c.device)
    c.phys_initial_gains=wp.array(c.gains.numpy()[0],dtype=scalar.wp_real,device=c.device)
    lo,hi=(np.r_[kp[0],-np.ones(delta)],np.r_[kp[1],np.ones(delta)])
    if mode==2:lo,hi=np.r_[damping[0],lo],np.r_[damping[1],hi]
    c.control_limits=(lo,hi)
    return c

class RuntimeConfiguredOSCControllerAdapter(FusedRuntimeControllerAdapter):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        c=self.controller
        self.action_dim=c.raw_action_dim
        self.control_limits=tuple(np.r_[x,(-1. if i==0 else 1.)] for i,x in enumerate(c.control_limits))
        self._variable_config=(c.variable_mode,c.raw_action_dim,bool(getattr(c,'position_only',False)))
        self.public['current_gains'].assign(c.gains.numpy())

    def validate(self,stream=None):
        super().validate(stream);c=self.controller
        if self._variable_config!=(c.variable_mode,c.raw_action_dim,bool(getattr(c,'position_only',False))):
            raise ValueError('prepared variable OSC configuration changed')

    def stage_action(self,action):
        self.validate();c=self.controller;a=np.asarray(action)
        if a.dtype!=np.float64 or a.shape!=(self.capacity,self.action_dim) or not np.isfinite(a).all():
            raise ValueError('finite actual variable OSC action shape required')
        packed=np.zeros((self.capacity,7),np.float64);offset=6*c.variable_mode
        delta=3 if c.position_only else 6
        packed[:,:delta]=a[:,offset:offset+delta];packed[:,6]=a[:,-1]
        with wp.ScopedStream(self.stream):
            c.action.assign(packed);c.phys_action.assign(np.ascontiguousarray(a))
