"""D07 CUDA chain: original mj_jacSite read set to the accepted N18 adapter.

The producer follows MuJoCo 2.3.7 engine_support.c:208-257,312-314
(Apache-2.0, SHA e532f0275e7e8c46e95271e0033e6b6319dc3c832fe52521c77efb3966947a77).
The adapter is the merged N18 implementation.  Its Jacobian allocations are
also the producer's output, so the producer-to-adapter edge has no copy.
"""
from __future__ import annotations
import libero.gpu.precision as scalar

from types import SimpleNamespace
from typing import Mapping

import numpy as np
import warp as wp

from libero.gpu.reference.position.jacobian.operator import StaticModel
from libero.gpu.reference.control.osc_inputs.kernel import OSCInputsWorkspace

wp.set_module_options({'enable_backward': False, 'fuse_fp': True, 'fast_math': False})


@wp.kernel(enable_backward=False)
def _site_jacobian(cdof: wp.array(dtype=scalar.wp_real, ndim=3),
                   subtree_com: wp.array(dtype=scalar.wp_real, ndim=3),
                   site_pos: wp.array(dtype=scalar.wp_real, ndim=2),
                   active: wp.array(dtype=wp.int32),
                   ancestor_mask: wp.array(dtype=wp.int32, ndim=2),
                   root: int, body: int,
                   jacp: wp.array(dtype=scalar.wp_real, ndim=3),
                   jacr: wp.array(dtype=scalar.wp_real, ndim=3),
                   status: wp.array(dtype=wp.int32)):
    w = wp.tid()
    status[w] = 0
    for row in range(3):
        for k in range(43):
            jacp[w, row, k] = scalar.wp_real(0.0)
            jacr[w, row, k] = scalar.wp_real(0.0)
    if active[w] == 0:
        return
    if active[w] != 1:
        status[w] = 4
        return
    ox = site_pos[w, 0] - subtree_com[w, root, 0]
    oy = site_pos[w, 1] - subtree_com[w, root, 1]
    oz = site_pos[w, 2] - subtree_com[w, root, 2]
    if not wp.isfinite(ox) or not wp.isfinite(oy) or not wp.isfinite(oz):
        status[w] = 5
        return
    for k in range(43):
        if ancestor_mask[body, k] == 1:
            ax = cdof[w, k, 0]
            ay = cdof[w, k, 1]
            az = cdof[w, k, 2]
            lx = cdof[w, k, 3]
            ly = cdof[w, k, 4]
            lz = cdof[w, k, 5]
            if (not wp.isfinite(ax) or not wp.isfinite(ay) or not wp.isfinite(az) or
                not wp.isfinite(lx) or not wp.isfinite(ly) or not wp.isfinite(lz)):
                status[w] = 5
                break
            cx = ay * oz - az * oy
            cy = az * ox - ax * oz
            cz = ax * oy - ay * ox
            vx = lx + cx
            vy = ly + cy
            vz = lz + cz
            if not wp.isfinite(vx) or not wp.isfinite(vy) or not wp.isfinite(vz):
                status[w] = 5
                break
            jacr[w, 0, k] = ax
            jacr[w, 1, k] = ay
            jacr[w, 2, k] = az
            jacp[w, 0, k] = vx
            jacp[w, 1, k] = vy
            jacp[w, 2, k] = vz
    if status[w] != 0:
        for row in range(3):
            for k in range(43):
                jacp[w, row, k] = scalar.wp_real(0.0)
                jacr[w, row, k] = scalar.wp_real(0.0)


@wp.kernel(enable_backward=False)
def _merge_status(producer: wp.array(dtype=wp.int32),
                  adapter: wp.array(dtype=wp.int32),
                  xvelp: wp.array(dtype=scalar.wp_real, ndim=2),
                  xvelr: wp.array(dtype=scalar.wp_real, ndim=2),
                  chain: wp.array(dtype=wp.int32)):
    w = wp.tid()
    chain[w] = adapter[w]
    if producer[w] != 0:
        chain[w] = 10 + producer[w]
        return
    if adapter[w] == 0:
        for row in range(3):
            if not wp.isfinite(xvelp[w,row]) or not wp.isfinite(xvelr[w,row]):
                chain[w] = 20
                return


class ChainWorkspace:
    """Static model preparation once; launch reads and writes only CUDA buffers."""

    DYNAMIC = {'cdof': (43, 6), 'subtree_com': (38, 3),
               'qpos': (48,), 'qvel': (43,), 'site_pos': (3,),
               'site_mat': (9,), 'qM': (153,), 'qfrc_bias': (43,),
               'active': ()}
    OUTPUT = ('jacp', 'jacr', 'ee_pos', 'ee_mat', 'xvelp', 'xvelr',
              'joint_pos', 'joint_vel', 'J', 'M', 'bias', 'jac_status',
              'adapter_status', 'status')

    def __init__(self, fields: Mapping[str, np.ndarray], capacity: int,
                 *, model_mjb_sha256: str, device: str = 'cuda:0'):
        if not wp.get_device(device).is_cuda:
            raise ValueError('D07 production requires a CUDA device')
        original_capacity = np.asarray(capacity)
        if (original_capacity.shape != () or original_capacity.dtype.kind not in 'iu' or
                int(original_capacity) <= 0 or int(original_capacity) >= 2**31):
            raise ValueError('capacity must be a positive int32-range integer scalar')
        capacity = int(original_capacity)
        if model_mjb_sha256 != '3eea791e54a911d83159c7891050be5447468c62efe026611f91414273156b42':
            raise ValueError('D07 fixed model SHA mismatch')
        raw = {name: np.asarray(value) for name, value in fields.items()}
        for key, tail in [('nq', 48), ('nv', 43), ('nM', 153),
                          ('nbody', 38), ('nsite', 31), ('eef_site_id', 13)]:
            if raw[key].shape != () or raw[key].dtype != np.int32 or int(raw[key]) != tail:
                raise ValueError(f'{key} schema or fixed value')
        self.static = StaticModel(raw, model_mjb_sha256=model_mjb_sha256)
        if self.static.device_static_sha256 != '59bdae2e4f2b1b5ed6162d3271da1f3569319137ade194d97d7751c3e59575a5':
            raise ValueError('fixed MJB topology/read-set digest mismatch')
        site = int(raw['eef_site_id'])
        self.body = int(self.static.fields['site_bodyid'][site])
        self.root = int(self.static.fields['body_rootid'][self.body])
        model = SimpleNamespace(**{key: raw[key] for key in
                                   ('nq', 'nv', 'nM', 'nsite', 'eef_site_id',
                                    'qpos_index', 'qvel_index', 'dof_parentid', 'dof_Madr')})
        self.osc = OSCInputsWorkspace(model, np.int32(capacity), device=device)
        self.capacity, self.device = capacity, device
        self.producer_jacp = self.osc.jacp
        self.producer_jacr = self.osc.jacr
        self.cdof = wp.zeros((capacity, 43, 6), dtype=scalar.wp_real, device=device)
        self.subtree_com = wp.zeros((capacity, 38, 3), dtype=scalar.wp_real, device=device)
        self.ancestor_mask = wp.array(self.static.ancestor_mask, dtype=wp.int32, device=device)
        self.jac_status = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.status = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.static_device_snapshot = {
            name: getattr(self.osc,name).numpy() for name in
            ('qpos_index','qvel_index','sparse_map')}
        self.static_device_snapshot['ancestor_mask'] = self.ancestor_mask.numpy()

    def upload(self, **arrays):
        """Strictly typed host staging outside production and Graph capture."""
        for name, value in arrays.items():
            if name not in self.DYNAMIC:
                raise KeyError(name)
            shape = (self.capacity,) + self.DYNAMIC[name]
            dtype = np.dtype('int32' if name == 'active' else 'float64')
            value = np.asarray(value)
            if value.shape != shape or value.dtype != dtype:
                raise ValueError(f'{name}: expected {shape} {dtype}, got {value.shape} {value.dtype}')
            target = getattr(self, name) if name in ('cdof', 'subtree_com') else getattr(self.osc, name)
            target.assign(np.ascontiguousarray(value))

    def launch(self, stream=None):
        """Three meaningful kernels, no host read, allocation, or transfer."""
        wp.launch(_site_jacobian, dim=self.capacity,
                  inputs=[self.cdof, self.subtree_com, self.osc.site_pos,
                          self.osc.active, self.ancestor_mask, self.root,
                          self.body, self.producer_jacp, self.producer_jacr, self.jac_status],
                  device=self.device, stream=stream)
        self.osc.launch(stream=stream)
        wp.launch(_merge_status, dim=self.capacity,
                  inputs=[self.jac_status, self.osc.status, self.osc.xvelp,
                          self.osc.xvelr, self.status],
                  device=self.device, stream=stream)

    def download(self):
        result = {name: getattr(self.osc, name).numpy() for name in self.OUTPUT
                  if name not in ('jac_status', 'adapter_status', 'status')}
        result['jac_status'] = self.jac_status.numpy()
        result['adapter_status'] = self.osc.status.numpy()
        result['status'] = self.status.numpy()
        return result
