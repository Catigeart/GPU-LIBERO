"""Independent device buffers and transport for the existing FP32 CUDA OSC."""
import ctypes as ct
from pathlib import Path
import subprocess

import numpy as np
from . import device as cuda
from .bridges import Bridges,Gather
from .fused_control_prepare import FusedControlPrepare
from libero.gpu.host.runtime.dynamics.mjwarp_queries import Jacobian

from libero.gpu.host.build.compilation import MissingCompiledProduct
from libero.gpu.host.build.project_paths import project_paths
from libero.gpu.host.runtime.step.substep_types import ControllerSolveData


def controller_product(root, *, fast_compile=False):
    folder = project_paths(root).cmake / ('sm80-dev' if fast_compile else 'sm80-release')
    target = 'flash_mjwarp_controller_fp32'
    library = folder / 'mjwarp_controller_fp32.so'
    check = subprocess.run(['ninja', '-C', str(folder), '-n', target], capture_output=True, text=True)
    if not library.is_file() or check.returncode or 'ninja: no work to do.' not in check.stdout:
        raise MissingCompiledProduct('Build ' + target + ' offline in ' + str(folder))
    return library


class Controller:
    WIDTHS = dict(mass=49, jacobian=42, q=7, qvel=7, bias=7, q0=7, ee_pos=3, ee_rot=9,
                  ee_velocity=6, action=7, goal_pos=3, goal_rot=9, gripper=2, gains=12,
                  range=18, ctrl=9, raw_tau=7, mass_inv=49, A_full=36, A_pos=9, A_ori=9,
                  lambda_full=36, lambda_pos=9, lambda_ori=9, nullspace_matrix=49,
                  position_error=3, orientation_error=3, quat_input=4, quat_result=9,
                  nullspace_torque=7, clipped=7, pm=49, pj=42, pq=7, pv=7, pp=3, pr=9, pe=6)

    def __init__(self, host, model, data, metadata, stream, library, *, queries=None):
        self.model, self.data, self.stream = model, data, stream
        self.n = data.nworld
        robots = metadata['robot_metadata']
        if len(robots) != 1:
            raise NotImplementedError('MJWarp OSC requires one Panda robot')
        robot = robots[0]
        cfg, grip = robot['controller_config'], robot['gripper']
        if (len(robot['qpos_indices']) != 7 or cfg['type'] != 'OSC_POSE' or
                cfg.get('impedance_mode', 'fixed') != 'fixed' or not cfg.get('control_delta', True) or
                cfg.get('position_limits') is not None or cfg.get('orientation_limits') is not None or
                cfg.get('interpolation') is not None or grip['class'] != 'PandaGripper' or
                grip['speed'] != .01 or host.nu != 9 or cfg.get('input_min') != -1 or
                cfg.get('input_max') != 1 or cfg.get('output_max') != [.05, .05, .05, .5, .5, .5]):
            raise NotImplementedError('MJWarp frontend currently supports the default LIBERO Panda OSC profile')
        dev = stream.device
        self.buffers = {name: cuda.zeros((self.n, width), dtype=cuda.float32, device=dev)
                        for name, width in self.WIDTHS.items()}
        for name in ('active', 'error', 'policy', 'needs', 'quat_valid', 'cache_valid'):
            self.buffers[name] = cuda.zeros(self.n, dtype=cuda.int32, device=dev)
        self.buffers['ranks'] = cuda.zeros((self.n, 3), dtype=cuda.int32, device=dev)
        self.buffers['quat_cast'] = cuda.zeros((self.n, 4), dtype=cuda.float32, device=dev)
        self.buffers['active'].fill_(1)
        kp = np.broadcast_to(np.asarray(cfg['kp']), (6,)).copy()
        kd = 2 * np.sqrt(kp) * np.asarray(cfg.get('damping_ratio', 1))
        self.buffers['gains'].assign(np.tile(np.r_[kp, kd].astype(np.float32), (self.n, 1)))
        ranges = robot['actuator_ctrlrange'] + grip['actuator_ctrlrange']
        self.buffers['range'].assign(np.tile(np.asarray(ranges, np.float32).reshape(-1), (self.n, 1)))
        self.buffers['q0'].assign(np.tile(np.asarray(robot['init_qpos'], np.float32), (self.n, 1)))
        self.qi = cuda.array(robot['qpos_indices'], dtype=int, device=dev)
        self.vi = cuda.array(robot['qvel_indices'], dtype=int, device=dev)
        self.ids = cuda.array(robot['actuator_ids'] + grip['actuator_ids'], dtype=int, device=dev)
        self.site = robot['eef_site_id']
        self.points = cuda.zeros(self.n, dtype=cuda.vec3, device=dev)
        self.body = cuda.array(np.full(self.n, host.site_bodyid[self.site], np.int32), dtype=int, device=dev)
        self.jp = cuda.zeros((self.n, 3, model.nv), dtype=float, device=dev)
        self.jr = cuda.zeros_like(self.jp)
        rows, cols = model.M_hinit_i.numpy().reshape(-1), model.M_colind.numpy().reshape(-1)
        mass_indices = np.full((model.nv, model.nv), -1, np.int32)
        for k, (i, j) in enumerate(zip(rows, cols)):
            mass_indices[i, j] = mass_indices[j, i] = k
        self.mass_indices_host = mass_indices
        dofs = np.asarray(robot['qvel_indices'])
        self.mi = cuda.array(mass_indices[np.ix_(dofs, dofs)].reshape(-1), dtype=int, device=dev)
        args = ControllerSolveData()
        for name, array in self.buffers.items():
            setattr(args, name, array.ptr)
        args.diagnostics = 0
        args.uncouple = int(cfg.get('uncouple_pos_ori', True))
        self.library_path = Path(library)
        self.library = ct.CDLL(str(library))
        for symbol, expected in [('scalar_bytes', 4), ('arguments_bytes', ct.sizeof(args))]:
            f = getattr(self.library, 'flash_mjwarp_controller_' + symbol)
            f.restype = ct.c_size_t
            if f() != expected:
                raise ValueError('MJWarp controller ABI differs: ' + symbol)
        self.args = cuda.array(np.frombuffer(bytes(args), np.uint8).copy(), dtype=cuda.uint8, device=dev)
        self.launch = self.library.flash_mjwarp_controller_launch
        self.launch.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p]
        self.launch.restype = ct.c_int
        self.publish=self.library.flash_mjwarp_controller_publish
        self.publish.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_void_p]
        self.publish.restype=ct.c_int
        self.bridge=Bridges(Path(library).parent/'cuda_bridges_fp32.so',stream)
        self.jacobian=Jacobian(model,data,self.points,self.body,self.jp,self.jr,Path(library).parent/'mjwarp_queries_fp32.so',stream)
        self.policy_values=[cuda.full(self.n,value,dtype=int,device=dev) for value in (0,1)]
        self.preparation=FusedControlPrepare(self,Path(library).parent/'fp32_fused_control_prepare.so')

    def _gather_arguments(self):
        return Gather(worlds=self.n,site=self.site,nv=self.model.nv,nq=self.model.nq,nmass=self.model.nC,nsites=self.model.nsite,
            qi=self.qi.ptr,vi=self.vi.ptr,mi=self.mi.ptr,qpos=self.data.qpos.ptr,qvel=self.data.qvel.ptr,
            bias=self.data.qfrc_bias.ptr,mass=self.data.M.ptr,jp=self.jp.ptr,jr=self.jr.ptr,
            position=self.data.site_xpos.ptr,rotation=self.data.site_xmat.ptr,
            **{k:self.buffers[v].ptr for k,v in [('out_mass','pm'),('out_jac','pj'),('out_q','pq'),('out_v','pv'),
                ('out_bias','bias'),('out_pos','pp'),('out_rot','pr'),('out_vel','pe')]})

    def gather(self,policy_source=None):
        self.preparation.enqueue(policy_source)

    def enqueue(self,policy_step):
        self.gather(self.policy_values[int(policy_step)])
        error=self.publish(self.args.ptr,self.n,self.ids.ptr,self.data.ctrl.ptr,self.model.nu,self.stream.cuda_stream)
        if error:raise RuntimeError('CUDA OSC launch failed: '+str(error))

    def bootstrap(self,mask):
        self.gather()
        for source,target in [('pm','mass'),('pj','jacobian'),('pq','q'),('pv','qvel'),('pp','ee_pos'),('pr','ee_rot'),('pe','ee_velocity')]:
            self.bridge.call('masked',self.n,self.WIDTHS[target],mask.ptr,self.buffers[source].ptr,self.buffers[target].ptr)
        self.bridge.call('bootstrap',self.n,mask.ptr,*[self.buffers[k].ptr for k in
            ('pp','pr','goal_pos','goal_rot','gripper','error','needs','cache_valid')])
