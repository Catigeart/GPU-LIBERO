"""Cold non-default OSC goal semantics; original default math is retained."""
import libero.gpu.precision as scalar
import math
import numpy as np
import warp as wp
from libero.gpu.host.assembly.pipeline.osc import FusedRuntimeControllerAdapter
from .variable_osc import prepare_variable_osc

def needs_general_goals(cfg):
    d=3 if cfg['type']=='OSC_POSITION' else 6
    expected=[.05]*3+([.5]*3 if d==6 else [])
    return (cfg.get('interpolation') is not None or cfg.get('position_limits') is not None or
        cfg.get('orientation_limits') is not None or cfg.get('control_delta',True) is not True or
        not np.all(np.asarray(cfg.get('input_min',-1))==-1) or not np.all(np.asarray(cfg.get('input_max',1))==1) or
        not np.array_equal(np.broadcast_to(np.asarray(cfg['output_min']),(d,)),-np.asarray(expected)) or
        not np.array_equal(np.broadcast_to(np.asarray(cfg['output_max']),(d,)),np.asarray(expected)))

def prepare_general_osc(c,cfg,*,timestep,policy_freq):
    if not needs_general_goals(cfg):return c
    prepare_variable_osc(c,cfg)
    c.general_goals=True;c.delta_dim=d=3 if cfg['type']=='OSC_POSITION' else 6
    c.raw_action_dim=6*c.variable_mode+d+int(getattr(c,'gripper_dof',1))
    c.relative_goals=int(bool(cfg.get('control_delta',True)))
    interpolation=cfg.get('interpolation')
    if interpolation not in (None,'linear'):raise NotImplementedError('GPU OSC interpolation: '+str(interpolation))
    c.interpolate_position=int(interpolation=='linear')
    c.interpolate_orientation=int(interpolation=='linear' and d==6)
    c.interpolation_steps=max(1,int(math.ceil(float(cfg.get('ramp_ratio',.2))/float(timestep)/float(policy_freq))))
    def vec(x):
        a=np.broadcast_to(np.asarray(x,np.float64),(d,)).copy()
        if not np.isfinite(a).all():raise ValueError('finite OSC action limits required')
        return a
    low,high=vec(cfg.get('input_min',-1)),vec(cfg.get('input_max',1))
    if np.any(low>=high):raise ValueError('ordered nondegenerate OSC input ranges required')
    output_low,output_high=vec(cfg['output_min']),vec(cfg['output_max'])
    def bounds(value,orientation=False):
        if value is None:return np.zeros((2,3)) if orientation else np.array([[-np.inf]*3,[np.inf]*3])
        result=np.asarray(value,np.float64)
        if result.shape!=(2,3) or np.isnan(result).any():raise ValueError('OSC goal limits must have shape (2,3)')
        if orientation and not np.isfinite(result).all():raise ValueError('finite orientation limits required')
        return result.copy()
    pos=bounds(cfg.get('position_limits'));ori=bounds(cfg.get('orientation_limits'),True)
    c.position_limited=int(cfg.get('position_limits') is not None)
    c.orientation_limited=int(bool(ori.any()))
    c.phys_goal_config=wp.array(np.r_[low,high,output_low,output_high,pos.reshape(-1),ori.reshape(-1)],dtype=scalar.wp_real,device=c.device)
    c.WIDTHS=dict(c.WIDTHS)
    c.WIDTHS.update(phys_action=c.raw_action_dim,phys_goal_config=4*d+12,
        current_gains=12,phys_initial_gains=12,goal_effective_pos=3,goal_ori_error=3,
        interp_previous_pos=3,interp_start_pos=3,interp_start_ori=3,interp_target_ori=3,interp_step=2,goal_delta_rot=9)
    c.current_gains=c.gains
    if not hasattr(c,'phys_action'):c.phys_action=wp.zeros((c.nworld,c.raw_action_dim),dtype=scalar.wp_real,device=c.device)
    c.phys_initial_gains=wp.array(c.gains.numpy()[0],dtype=scalar.wp_real,device=c.device)
    for name in ('goal_effective_pos','goal_ori_error','interp_previous_pos','interp_start_pos','interp_start_ori','interp_target_ori'):
        setattr(c,name,wp.zeros((c.nworld,3),dtype=scalar.wp_real,device=c.device))
    c.interp_step=wp.zeros((c.nworld,2),dtype=scalar.wp_real,device=c.device)
    c.goal_delta_rot=wp.zeros((c.nworld,9),dtype=scalar.wp_real,device=c.device)
    c.goal_rotate=wp.zeros(c.nworld,dtype=wp.int32,device=c.device)
    c.goal_core_policy=wp.zeros(c.nworld,dtype=wp.int32,device=c.device)
    if c.variable_mode:
        prefix=c.control_limits[0][:-d],c.control_limits[1][:-d]
        c.control_limits=(np.r_[prefix[0],low],np.r_[prefix[1],high])
    else:c.control_limits=low,high
    return c

class RuntimeGeneralOSCControllerAdapter(FusedRuntimeControllerAdapter):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        c=self.controller;self.action_dim=c.raw_action_dim
        self.control_limits=tuple(np.r_[x,(-1. if i==0 else 1.)] for i,x in enumerate(c.control_limits))
        from libero.gpu.host.runtime.dynamics.runtime_adapter import _token
        self._goal_owners={name:_token(getattr(c,name)) for name in ('goal_rotate','goal_core_policy')}
        self._goal_config=self._configuration()
        self.public['current_gains'].assign(c.gains.numpy())

    def _configuration(self):
        c=self.controller
        return (c.variable_mode,c.raw_action_dim,c.delta_dim,c.relative_goals,c.interpolate_position,
            c.interpolate_orientation,c.interpolation_steps,c.position_limited,c.orientation_limited,c.general_goals)

    def validate(self,stream=None):
        super().validate(stream)
        if self._goal_config!=self._configuration():raise ValueError('prepared OSC goal policy changed')
        from libero.gpu.host.runtime.dynamics.runtime_adapter import _token
        for name,tok in self._goal_owners.items():
            if _token(getattr(self.controller,name))!=tok:raise ValueError('prepared OSC goal owner changed: '+name)

    def stage_action(self,action):
        self.validate();c=self.controller;a=np.asarray(action)
        if a.dtype!=np.float64 or a.shape!=(self.capacity,self.action_dim) or not np.isfinite(a).all():
            raise ValueError('finite actual configured OSC action shape required')
        packed=np.zeros((self.capacity,7),np.float64);offset=6*c.variable_mode
        packed[:,:c.delta_dim]=a[:,offset:offset+c.delta_dim];packed[:,6]=a[:,-1]
        with wp.ScopedStream(self.stream):c.action.assign(packed);c.phys_action.assign(np.ascontiguousarray(a))

def goal_stages(c):
    return [('osc_goal_cuda::begin',[c.active,c.policy_step,c.error,c.phys_action,c.ee_pos,c.ee_rot,
        c.phys_goal_config,c.goal_pos,c.goal_rot,c.goal_delta_rot,c.goal_rotate,c.delta_dim,c.raw_action_dim,
        c.variable_mode,c.relative_goals,c.position_limited],'lane0'),
        ('osc_goal_cuda::product',[c.active,c.error,c.goal_rotate,c.goal_delta_rot,c.ee_rot,c.goal_rot],'block'),
        ('osc_goal_cuda::finish',[c.active,c.policy_step,c.error,c.goal_rotate,c.phys_goal_config,
        c.goal_pos,c.goal_rot,c.ee_rot,c.interp_previous_pos,c.interp_start_pos,c.interp_start_ori,
        c.interp_target_ori,c.interp_step,c.goal_effective_pos,c.goal_ori_error,c.delta_dim,
        c.orientation_limited,c.interpolate_position,c.interpolate_orientation,c.interpolation_steps],'lane0')]

def reset_goal_stage(c,mask):
    return ('osc_goal_cuda::reset_history',[mask,c.interp_previous_pos,c.interp_start_pos,
        c.interp_start_ori,c.interp_target_ori,c.interp_step,c.phys_initial_gains,c.gains],'lane0')
