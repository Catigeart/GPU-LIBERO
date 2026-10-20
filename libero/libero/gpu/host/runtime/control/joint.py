"""Resident robosuite single-arm joint controllers with handwritten CUDA math.

Only cold configuration and explicit action boundary use NumPy. Every controller
history field participates in the enclosing adapter's successful-world commit.
"""
import libero.gpu.precision as scalar
from pathlib import Path
import math
import numpy as np
import warp as wp
from libero.gpu.host.paths import GPU_ROOT
from libero.gpu.host.build.native import NativePlan
from libero.gpu.host.runtime.dynamics import runtime_adapter as original

JOINT_FAMILIES = {"JOINT_TORQUE": 0, "JOINT_POSITION": 1, "JOINT_VELOCITY": 2}

def _vector(value, count, name, *, finite=True):
    result = np.broadcast_to(np.asarray(value, dtype=np.float64), (count,)).copy()
    if finite and not np.isfinite(result).all(): raise ValueError("nonfinite controller " + name)
    return result

def _bounds(value, count, name, fallback):
    if value is None: return np.asarray(fallback, np.float64).copy()
    a = np.asarray(value, np.float64)
    if a.shape == (2,): a = np.broadcast_to(a[:, None], (2, count))
    if a.shape != (2, count): raise ValueError("controller bounds shape: " + name)
    if np.isnan(a).any() or np.any(a[0] > a[1]): raise ValueError("controller bounds order: " + name)
    return a

class JointControllerWorkspace:
    def __init__(self, nworld, *, robot, nu, timestep, policy_freq, device="cuda:0"):
        self.device = wp.get_device(device)
        if not self.device.is_cuda: raise ValueError("joint controller requires actual CUDA")
        self.nworld, self.nu = int(nworld), int(nu)
        if self.nworld < 1 or self.nu < 1: raise ValueError("positive actual controller dimensions required")
        cfg = dict(robot["controller_config"])
        self.name, self.family = cfg["type"], JOINT_FAMILIES[cfg["type"]]
        self.joint_dim = n = len(robot["joint_ids"])
        if not 1 <= n <= 32: raise ValueError("GPU single-arm joint dimension must be 1..32")
        arm = list(robot["actuator_ids"])
        if len(arm) != n: raise ValueError("one native actuator per scalar arm joint required")
        grip = robot.get("gripper", {})
        grip_ids = list(grip.get("actuator_ids", [])); self.ng = len(grip_ids)
        self.gripper_dof = gdof = int(grip.get("dof", 0))
        gripclass = grip.get("class")
        if self.ng == 0 and gdof == 0: self.gripmode, self.speed = 0, 0.
        elif gripclass == "PandaGripper" and self.ng == 2 and gdof == 1:
            self.gripmode, self.speed = 1, float(grip.get("speed", .01))
        elif gripclass in ("RethinkGripper","Robotiq140Gripper") and self.ng == 2 and gdof == 1:
            self.gripmode,self.speed=2,float(grip.get("speed",.01))
        elif gripclass in ("Robotiq85Gripper","RobotiqThreeFingerDexterousGripper") and gdof in (1,self.ng):
            self.gripmode,self.speed=3,float(grip.get("speed",.01))
        elif gripclass in ("JacoThreeFingerGripper","JacoThreeFingerDexterousGripper") and gdof in (1,self.ng):
            self.gripmode,self.speed=4,float(grip.get("speed",.005))
        elif gripclass == "RobotiqThreeFingerGripper" and self.ng == 4 and gdof == 1:
            self.gripmode,self.speed=5,float(grip.get("speed",.01))
        elif gripclass in ("PandaGripperBase","RethinkGripperBase","Robotiq85GripperBase",
            "Robotiq140GripperBase","JacoThreeFingerGripperBase","RobotiqThreeFingerGripperBase") and self.ng == gdof:
            self.gripmode, self.speed = 0, 0.
        else: raise NotImplementedError("GPU gripper formatter not implemented: " + str(gripclass))
        ids = arm + grip_ids
        if len(ids) != self.nu or len(set(ids)) != self.nu or set(ids) != set(range(self.nu)):
            raise ValueError("actual controller actuators must cover the native single-arm control root")
        self.impedance = {"fixed": 0, "variable_kp": 1, "variable": 2}[cfg.get("impedance_mode", "fixed")]
        if self.family != 1 and self.impedance: raise ValueError("impedance modes belong to joint-position only")
        self.action_dim = n * (1 + self.impedance) + gdof
        self.interpolation = cfg.get("interpolation")
        if self.interpolation not in (None, "linear"): raise NotImplementedError("GPU interpolation: " + str(self.interpolation))
        self.total_steps = max(1, int(math.ceil(float(cfg.get("ramp_ratio", .2)) / float(timestep) / float(policy_freq))))
        inlo = _vector(cfg.get("input_min", -1), n, "input_min")
        inhi = _vector(cfg.get("input_max", 1), n, "input_max")
        if np.any(inlo >= inhi): raise ValueError("ordered nondegenerate input action range required")
        outlo = _vector(cfg["output_min"], n, "output_min")
        outhi = _vector(cfg["output_max"], n, "output_max")
        ranges = np.asarray(robot["actuator_ctrlrange"] + grip.get("actuator_ctrlrange", []), np.float64)
        if ranges.shape != (self.nu, 2) or not np.isfinite(ranges).all() or np.any(ranges[:, 0] > ranges[:, 1]):
            raise ValueError("finite ordered native actuator bounds required")
        kp = _vector(cfg.get("kp", 0), n, "kp")
        if np.any(kp < 0): raise ValueError("nonnegative gain required")
        kd = 2 * np.sqrt(kp) * _vector(cfg.get("damping_ratio", 1), n, "damping_ratio")
        unlimited = np.array([np.full(n, -np.inf), np.full(n, np.inf)])
        target = _bounds(cfg.get(("torque_limits", "qpos_limits", "velocity_limits")[self.family]), n,
                         "target", ranges[:n].T if self.family == 0 else unlimited)
        kplim = _bounds(cfg.get("kp_limits", [0, 300]), n, "kp", unlimited)
        damplim = _bounds(cfg.get("damping_ratio_limits", [0, 100]), n, "damping", unlimited)
        if np.any(kplim[0] < 0) or np.any(damplim[0] < 0): raise ValueError("nonnegative variable impedance gains required")
        params = np.concatenate([inlo, inhi, outlo, outhi, kp, kd, *target, *kplim, *damplim])
        self.config = wp.array(params, dtype=scalar.wp_real, device=self.device)
        self.actuator_ids = wp.array(ids, dtype=wp.int32, device=self.device)
        self.actuator_ranges = wp.array(ranges.reshape(-1), dtype=scalar.wp_real, device=self.device)
        low, high = inlo, inhi
        if self.impedance == 1: low, high = np.r_[kplim[0], low], np.r_[kplim[1], high]
        if self.impedance == 2: low, high = np.r_[damplim[0], kplim[0], low], np.r_[damplim[1], kplim[1], high]
        self.control_limits = (low.copy(), high.copy())
        self.action_limits = (np.r_[low, -np.ones(gdof)], np.r_[high, np.ones(gdof)])
        self.uncouple, self.diagnostics = 1, 0
        self.WIDTHS = {"phys_mass":n*n,"phys_q":n,"phys_qvel":n,"phys_bias":n,
            "phys_jacobian":6*n,"phys_ee_pos":3,"phys_ee_rot":9,"phys_ee_velocity":6,
            "mass":n*n,"q":n,"qvel":n,"q0":n,"action":self.action_dim,
            "goal":n,"interp_start":n,"current_gains":2*n,"last_err":n,"summed_err":n,
            "derivative_ring":5*n,"history":4,"drive":n,"product":n,"gripper":max(1,self.ng),
            "ctrl":self.nu,"torques_raw":n,"gains":2*n,"ctrl_range":2*self.nu}
        for name,width in self.WIDTHS.items():
            setattr(self,name,wp.zeros((self.nworld,width),dtype=scalar.wp_real,device=self.device))
        self.gains.assign(np.tile(np.r_[kp,kd],(self.nworld,1)))
        self.current_gains.assign(np.tile(np.r_[kp,kd],(self.nworld,1)))
        self.history.assign(np.tile([-1.,0.,0.,0.],(self.nworld,1)))
        for name in ("active","policy_step","needs_update","cache_valid","error","quat_valid"):
            setattr(self,name,wp.zeros(self.nworld,dtype=wp.int32,device=self.device))
        self.ranks = wp.zeros((self.nworld,3),dtype=wp.int32,device=self.device)
        self.quat_cast_wxyz = wp.zeros((self.nworld,4),dtype=wp.float32,device=self.device)
        self._plans = None

    def install(self, build_root):
        root=Path(build_root);here=GPU_ROOT/"cuda/control";n=self.joint_dim
        def make(name,stages,heavy=False):
            return NativePlan(name,[here/"joint_runtime.cuh"],stages,self.nworld,root/name,
                device=self.device,mathdx=heavy,bridges=[here/"joint_bridge.cu"] if heavy else [],
                defines={"JOINT_DIM":n,"DX_SM":self.device.arch*10})
        refresh=("joint_cuda::refresh",[self.active,self.needs_update,self.cache_valid,self.error,
            self.phys_mass,self.phys_q,self.phys_qvel,self.mass,self.q,self.qvel,n],"lane0")
        pre=("joint_cuda::prepare",[self.active,self.policy_step,self.cache_valid,self.error,self.action,
            self.q,self.qvel,self.config,self.goal,self.interp_start,self.current_gains,self.last_err,
            self.summed_err,self.derivative_ring,self.history,self.drive,n,self.action_dim,self.family,
            self.impedance,int(self.interpolation is not None),self.total_steps],"lane0")
        stages=[refresh,pre]
        if self.family==1:stages.append(("joint_cuda::matvec",[self.active,self.error,self.mass,self.drive,self.product],"block"))
        stages.append(("joint_cuda::finish",[self.active,self.error,self.needs_update,self.action,self.phys_bias,
            self.drive,self.product,self.torques_raw,self.ctrl,self.actuator_ranges,self.actuator_ids,self.gripper,
            self.history,n,self.nu,self.action_dim,self.family,self.ng,self.gripper_dof,self.gripmode,self.speed],"lane0"))
        self._run=make("joint_run",stages,self.family==1)
        self._refresh=make("joint_refresh",[refresh])
        self._reset={clear:make("joint_reset%d"%clear,[("joint_cuda::reset",[self.active,self.phys_q,self.config,
            self.goal,self.interp_start,self.current_gains,self.last_err,self.summed_err,self.derivative_ring,
            self.history,self.gripper,self.error,self.cache_valid,self.needs_update,n,self.ng,self.family,clear],"lane0")]) for clear in (0,1)}
        self._plans=(self._run,self._refresh,*self._reset.values())
        return self

    def launch(self,*,stream=None):
        if self._plans is None:raise RuntimeError("joint native plans must be installed during cold preparation")
        self._run.launch(stream)

    def refresh(self,*,stream=None):self._refresh.launch(stream)

    def reset_selected(self,mask,*,clear_gripper=False,update_initial_joints=False,stream=None):
        original._array(mask,(self.nworld,),wp.int32,self.device)
        wp.copy(self.active,mask,stream=stream)
        self._reset[int(bool(clear_gripper))].launch(stream)

class RuntimeJointControllerAdapter(original.RuntimeControllerAdapter):
    def __init__(self,*args,build_root,**kwargs):
        super().__init__(*args,**kwargs)
        c=self.controller
        self.action_dim=c.action_dim
        self.control_limits=tuple(x.copy() for x in c.action_limits)
        self._joint_config=(c.family,c.impedance,c.interpolation,c.total_steps,c.joint_dim,c.nu,
            c.ng,c.gripper_dof,c.gripmode,c.speed)
        self._joint_tokens={name:original._token(getattr(c,name)) for name in ("config","actuator_ids","actuator_ranges")}
        c.install(Path(build_root)/"workspace")
        self._reset_public=NativePlan("joint_reset_public",[GPU_ROOT/"cuda/control/joint_runtime.cuh"],
            [("joint_cuda::reset_public",[self.call_mask,self.first_error,self._reset_source if hasattr(self,"_reset_source") else c.ctrl,
                self.public["ctrl"],self.valid,self.written,self.event,c.nu],"lane0")],self.capacity,
            Path(build_root)/"reset_public",device=self.device)
        self.public["current_gains"].assign(c.current_gains.numpy())
        self.public["history"].assign(c.history.numpy())

    def validate(self,stream=None):
        super().validate(stream)
        c=self.controller
        if self._joint_config!=(c.family,c.impedance,c.interpolation,c.total_steps,c.joint_dim,c.nu,
                c.ng,c.gripper_dof,c.gripmode,c.speed):raise ValueError("prepared joint configuration changed")
        for name,tok in self._joint_tokens.items():
            if original._token(getattr(c,name))!=tok:raise ValueError("joint static owner changed: "+name)

    def stage_action(self,action):
        self.validate();a=np.asarray(action)
        if a.dtype!=np.float64 or a.shape!=(self.capacity,self.action_dim) or not np.isfinite(a).all():
            raise ValueError("action must be finite float64 (worlds,actual controller action_dim)")
        with wp.ScopedStream(self.stream):self.controller.action.assign(np.ascontiguousarray(a))

    def initialize_ctrl_from_reset(self,mask,initial_ctrl,*,stream=None):
        self.validate(stream);c=self.controller
        original._array(mask,(self.capacity,),wp.int32,self.device)
        original._array(initial_ctrl,(self.capacity,c.nu),scalar.wp_real,self.device)
        wp.copy(self.call_mask,mask,stream=self.stream)
        wp.copy(c.ctrl,initial_ctrl,stream=self.stream)
        self._reset_public.launch(self.stream)

    def bootstrap_from_current_physics(self,mask,*,stream=None):
        self.validate(stream);original._array(mask,(self.capacity,),wp.int32,self.device)
        c,st,n=self.controller,self.stream,self.capacity
        self.forward_refresh(mask,st);self.forward_refresh(mask,st);self.gather(mask,st);self.live_bias(mask,st)
        wp.copy(self.call_mask,mask,stream=st)
        wp.launch(original._control_gate,n,inputs=[self.call_mask,self.parents.valid,self.parents.written,
            self.parents.expected,self.first_error,c.error,c.policy_step,0],device=self.device,stream=st)
        c.reset_selected(self.call_mask,clear_gripper=True,update_initial_joints=True,stream=st)
        c.refresh(stream=st)
        wp.launch(original._bootstrap_finish,n,inputs=[self.call_mask,c.error,c.cache_valid,c.needs_update,self.first_error],device=self.device,stream=st)
        for k,a in self.public.items():wp.launch(original._copy_admitted,n,inputs=[self.call_mask,getattr(c,k),a,a.shape[1]],device=self.device,stream=st)
        for k,a in self.public_int.items():wp.launch(original._copy_i32,n,inputs=[self.call_mask,getattr(c,k),a],device=self.device,stream=st)
