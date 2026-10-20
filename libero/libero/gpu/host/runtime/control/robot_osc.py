"""Real robot actuator/gripper boundary over the unchanged seven-arm OSC core."""
import libero.gpu.precision as scalar
import numpy as np
import warp as wp
from libero.gpu.host.paths import GPU_ROOT
from libero.gpu.host.build.native import NativePlan
from libero.gpu.host.runtime.dynamics.runtime_adapter import _array, _token
from libero.gpu.host.assembly.pipeline.osc import FusedRuntimeControllerAdapter
from .position_osc import RuntimePositionOSCControllerAdapter
from .variable_osc import RuntimeConfiguredOSCControllerAdapter
from .general_osc import RuntimeGeneralOSCControllerAdapter

def formatter(grip):
    ids=grip.get('actuator_ids',[]);ng=len(ids);dof=int(grip.get('dof',0));name=grip.get('class')
    if ng==0 and dof==0:return 0,0.
    if name=='PandaGripper' and ng==2 and dof==1:return 1,float(grip.get('speed',.01))
    if name in ('RethinkGripper','Robotiq140Gripper') and ng==2 and dof==1:return 2,float(grip.get('speed',.01))
    if name in ('Robotiq85Gripper','RobotiqThreeFingerDexterousGripper') and dof in (1,ng):return 3,float(grip.get('speed',.01))
    if name in ('JacoThreeFingerGripper','JacoThreeFingerDexterousGripper') and dof in (1,ng):return 4,float(grip.get('speed',.005))
    if name=='RobotiqThreeFingerGripper' and ng==4 and dof==1:return 5,float(grip.get('speed',.01))
    if name in ('PandaGripperBase','RethinkGripperBase','Robotiq85GripperBase','Robotiq140GripperBase','JacoThreeFingerGripperBase','RobotiqThreeFingerGripperBase') and ng==dof:return 0,0.
    raise NotImplementedError('GPU OSC gripper formatter: '+str(name))

def prepare_robot_osc(c,robot,nu):
    arm=list(robot['actuator_ids']);g=robot.get('gripper',{});gid=list(g.get('actuator_ids',[]))
    n=len(robot['joint_ids'])
    if n not in (6,7) or len(arm)!=n:raise NotImplementedError('OSC core requires six or seven real scalar arm joints')
    ids=arm+gid
    if len(ids)!=nu or len(set(ids))!=nu or set(ids)!=set(range(nu)):
        raise ValueError('actual single-arm controller actuators must cover the native control root')
    mode,speed=formatter(g)
    c.actual_nu=int(nu);c.gripper_dof=int(g.get('dof',0));c.actual_ng=len(gid)
    c.actual_joint_dim=n;c.formatter_mode=mode;c.formatter_speed=speed
    ranges=np.asarray(robot['actuator_ctrlrange']+g.get('actuator_ctrlrange',[]),np.float64)
    if ranges.shape!=(nu,2) or not np.isfinite(ranges).all():raise ValueError('finite actual actuator ranges required')
    c.robot_projection=not (n==7 and arm==list(range(7)) and gid==[7,8] and nu==9 and mode==1 and speed==.01)
    internal=np.r_[ranges[:n],np.tile([-1.,1.],(9-n,1))]
    if not c.robot_projection:internal=ranges
    c.ctrl_range.assign(np.tile(internal.reshape(-1),(c.nworld,1)))
    if not c.robot_projection:return c
    c.WIDTHS=dict(c.WIDTHS)
    c.WIDTHS.update(ctrl=nu,gripper=max(1,c.actual_ng),internal_ctrl=9,internal_gripper=2,
        phys_gripper_action=max(1,c.gripper_dof),phys_actual_ranges=2*nu,phys_actual_ids=nu)
    c.internal_ctrl=c.ctrl;c.internal_gripper=c.gripper
    c.ctrl=wp.zeros((c.nworld,nu),dtype=scalar.wp_real,device=c.device)
    c.gripper=wp.zeros((c.nworld,max(1,c.actual_ng)),dtype=scalar.wp_real,device=c.device)
    c.phys_gripper_action=wp.zeros((c.nworld,max(1,c.gripper_dof)),dtype=scalar.wp_real,device=c.device)
    c.phys_actual_ranges=wp.array(ranges.reshape(-1),dtype=scalar.wp_real,device=c.device)
    c.phys_actual_ids=wp.array(ids,dtype=wp.int32,device=c.device)
    return c

class _RobotBoundary:
    def __init__(self,*args,**kwargs):
        build_root=kwargs.pop('build_root')
        super().__init__(*args,**kwargs)
        c=self.controller
        self.action_dim=6*getattr(c,'variable_mode',0)+(3 if c.position_only else 6)+c.gripper_dof
        lo,hi=c.control_limits
        self.control_limits=np.r_[lo,-np.ones(c.gripper_dof)],np.r_[hi,np.ones(c.gripper_dof)]
        self._robot_config=(c.actual_nu,c.actual_joint_dim,c.actual_ng,c.gripper_dof,c.formatter_mode,c.formatter_speed,c.robot_projection)
        self._robot_init_mask=wp.empty(self.capacity,dtype=wp.int32,device=self.device)
        self._robot_init_ctrl=wp.empty((self.capacity,c.actual_nu),dtype=scalar.wp_real,device=self.device)
        self._robot_init_tokens=(_token(self._robot_init_mask),_token(self._robot_init_ctrl))
        self._robot_init_plan=NativePlan('osc_robot_reset_public',
            [GPU_ROOT/'cuda/control/robot_osc.cuh'],
            [('osc_robot_cuda::reset_public',[self._robot_init_mask,self.first_error,
                self._robot_init_ctrl,self.public['ctrl'],self.valid,self.written,self.event,c.actual_nu],'lane0')],
            self.capacity,build_root,device=self.device)

    def validate(self,stream=None):
        super().validate(stream);c=self.controller
        if self._robot_config!=(c.actual_nu,c.actual_joint_dim,c.actual_ng,c.gripper_dof,c.formatter_mode,c.formatter_speed,c.robot_projection):
            raise ValueError('prepared robot formatter configuration changed')
        if self._robot_init_tokens!=(_token(self._robot_init_mask),_token(self._robot_init_ctrl)):
            raise ValueError('prepared robot reset owner changed')

    def initialize_ctrl_from_reset(self,mask,initial_ctrl,*,stream=None):
        self.validate(stream)
        _array(mask,(self.capacity,),wp.int32,self.device)
        _array(initial_ctrl,(self.capacity,self.controller.actual_nu),scalar.wp_real,self.device)
        wp.copy(self._robot_init_mask,mask,stream=self.stream)
        wp.copy(self._robot_init_ctrl,initial_ctrl,stream=self.stream)
        self._robot_init_plan.launch(self.stream)

    def stage_action(self,action):
        self.validate();c=self.controller;a=np.asarray(action)
        if a.dtype!=np.float64 or a.shape!=(self.capacity,self.action_dim) or not np.isfinite(a).all():
            raise ValueError('finite original robot controller action shape required')
        mode=getattr(c,'variable_mode',0);d=3 if c.position_only else 6
        packed=np.zeros((self.capacity,7),np.float64);packed[:,:d]=a[:,6*mode:6*mode+d]
        grip=np.zeros((self.capacity,max(1,c.gripper_dof)),np.float64)
        if c.gripper_dof:grip[:,:c.gripper_dof]=a[:,-c.gripper_dof:];packed[:,6]=grip[:,0]
        with wp.ScopedStream(self.stream):
            c.action.assign(packed);c.phys_gripper_action.assign(grip)
            if hasattr(c,'phys_action'):c.phys_action.assign(np.ascontiguousarray(a))

class RobotPoseAdapter(_RobotBoundary,FusedRuntimeControllerAdapter):pass
class RobotPositionAdapter(_RobotBoundary,RuntimePositionOSCControllerAdapter):pass
class RobotVariableAdapter(_RobotBoundary,RuntimeConfiguredOSCControllerAdapter):pass
class RobotGeneralAdapter(_RobotBoundary,RuntimeGeneralOSCControllerAdapter):pass

def robot_osc_adapter_class(c):
    if getattr(c,'general_goals',False):return RobotGeneralAdapter
    if getattr(c,'variable_mode',0):return RobotVariableAdapter
    if c.position_only:return RobotPositionAdapter
    return RobotPoseAdapter

def project_stage(c):
    return ('osc_robot_cuda::project',[c.active,c.error,c.torques_clipped,c.phys_gripper_action,
        c.gripper,c.ctrl,c.phys_actual_ids,c.phys_actual_ranges,c.actual_nu,c.actual_joint_dim,c.actual_ng,c.gripper_dof,
        c.formatter_mode,c.formatter_speed],'lane0')

def reset_stage(c,mask,clear):
    return ('osc_robot_cuda::reset',[mask,c.gripper,c.actual_ng,clear],'lane0')
