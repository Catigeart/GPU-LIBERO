"""Bind MJWarp device arrays to the existing separately compiled FP32 OSC."""
import ctypes as ct
from pathlib import Path
import subprocess

import numpy as np
import warp as wp
import mujoco_warp as mjw

from libero.gpu.host.build.compilation import MissingCompiledProduct
from libero.gpu.host.build.project_paths import project_paths
from libero.gpu.host.runtime.step.substep_types import ControllerSolveData
from . import kernels


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
        self.buffers = {name: wp.zeros((self.n, width), dtype=wp.float32, device=dev)
                        for name, width in self.WIDTHS.items()}
        for name in ('active', 'error', 'policy', 'needs', 'quat_valid', 'cache_valid'):
            self.buffers[name] = wp.zeros(self.n, dtype=wp.int32, device=dev)
        self.buffers['ranks'] = wp.zeros((self.n, 3), dtype=wp.int32, device=dev)
        self.buffers['quat_cast'] = wp.zeros((self.n, 4), dtype=wp.float32, device=dev)
        self.buffers['active'].fill_(1)
        kp = np.broadcast_to(np.asarray(cfg['kp']), (6,)).copy()
        kd = 2 * np.sqrt(kp) * np.asarray(cfg.get('damping_ratio', 1))
        self.buffers['gains'].assign(np.tile(np.r_[kp, kd].astype(np.float32), (self.n, 1)))
        ranges = robot['actuator_ctrlrange'] + grip['actuator_ctrlrange']
        self.buffers['range'].assign(np.tile(np.asarray(ranges, np.float32).reshape(-1), (self.n, 1)))
        self.buffers['q0'].assign(np.tile(np.asarray(robot['init_qpos'], np.float32), (self.n, 1)))
        self.qi = wp.array(robot['qpos_indices'], dtype=int, device=dev)
        self.vi = wp.array(robot['qvel_indices'], dtype=int, device=dev)
        self.ids = wp.array(robot['actuator_ids'] + grip['actuator_ids'], dtype=int, device=dev)
        self.site = robot['eef_site_id']
        self.points = wp.zeros(self.n, dtype=wp.vec3, device=dev)
        self.body = wp.array(np.full(self.n, host.site_bodyid[self.site], np.int32), dtype=int, device=dev)
        self.jp = wp.zeros((self.n, 3, model.nv), dtype=float, device=dev)
        self.jr = wp.zeros_like(self.jp)
        self.jacobian=queries.jacobian(self.points,self.body,self.jp,self.jr) if queries is not None else None
        rows, cols = model.M_hinit_i.numpy().reshape(-1), model.M_colind.numpy().reshape(-1)
        mass_indices = np.full((model.nv, model.nv), -1, np.int32)
        for k, (i, j) in enumerate(zip(rows, cols)):
            mass_indices[i, j] = mass_indices[j, i] = k
        self.mass_indices_host = mass_indices
        dofs = np.asarray(robot['qvel_indices'])
        self.mi = wp.array(mass_indices[np.ix_(dofs, dofs)].reshape(-1), dtype=int, device=dev)
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
        self.args = wp.array(np.frombuffer(bytes(args), np.uint8).copy(), dtype=wp.uint8, device=dev)
        self.launch = self.library.flash_mjwarp_controller_launch
        self.launch.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p]
        self.launch.restype = ct.c_int

    def gather(self):
        wp.launch(kernels.site_points, self.n, [self.site, self.data.site_xpos, self.points], stream=self.stream)
        if self.jacobian is None:mjw.jac(self.model, self.data, self.jp, self.jr, self.points, self.body)
        else:self.jacobian.enqueue()
        wp.launch(kernels.gather, (self.n, 49), inputs=[self.site, self.qi, self.vi, self.mi,
            self.data.qpos, self.data.qvel, self.data.qfrc_bias, self.data.M, self.jp, self.jr,
            self.data.site_xpos, self.data.site_xmat] + [self.buffers[k] for k in
            ('pm', 'pj', 'pq', 'pv', 'bias', 'pp', 'pr', 'pe')], stream=self.stream)

    def enqueue(self, policy_step):
        wp.launch(kernels.policy, self.n, [int(policy_step), self.buffers['policy']], stream=self.stream)
        self.gather()
        code = self.launch(self.args.ptr, self.n, self.stream.cuda_stream)
        if code:
            raise RuntimeError('MJWarp CUDA OSC launch failed: ' + str(code))
        wp.launch(kernels.scatter_control, (self.n, 9),
                  [self.ids, self.buffers['ctrl'], self.data.ctrl], stream=self.stream)

    def bootstrap(self, mask):
        self.gather()
        for src, dst in [('pm', 'mass'), ('pj', 'jacobian'), ('pq', 'q'), ('pv', 'qvel'),
                         ('pp', 'ee_pos'), ('pr', 'ee_rot'), ('pe', 'ee_velocity')]:
            wp.launch(kernels.masked_float, self.buffers[dst].shape,
                      [mask, self.buffers[src], self.buffers[dst]], stream=self.stream)
        wp.launch(kernels.bootstrap, self.n, [mask] + [self.buffers[k] for k in
            ('pp', 'pr', 'goal_pos', 'goal_rot', 'gripper', 'error', 'needs', 'cache_valid')], stream=self.stream)
