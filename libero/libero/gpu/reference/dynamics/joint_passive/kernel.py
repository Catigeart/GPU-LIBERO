"""Scalar joint springs and all DOF dampers; MuJoCo 2.3.7 passive prefix.

Apache-2.0 source: engine_passive.c:mj_passive lines 37-91. This scoped
translation uses the fixed model's zero free/ball rotational stiffness.
"""
import libero.gpu.precision as scalar

import numpy as np
import warp as wp

wp.set_module_options({'enable_backward': False, 'fuse_fp': True})


@wp.kernel(enable_backward=False)
def _passive(qpos: wp.array(dtype=scalar.wp_real, ndim=2),
             qvel: wp.array(dtype=scalar.wp_real, ndim=2),
             active: wp.array(dtype=wp.int32),
             jtype: wp.array(dtype=wp.int32),
             qadr: wp.array(dtype=wp.int32),
             dadr: wp.array(dtype=wp.int32),
             stiffness: wp.array(dtype=scalar.wp_real),
             spring: wp.array(dtype=scalar.wp_real),
             damping: wp.array(dtype=scalar.wp_real),
             out: wp.array(dtype=scalar.wp_real, ndim=2),
             status: wp.array(dtype=wp.int32)):
    w = wp.tid()
    status[w] = 0
    for k in range(43):
        out[w, k] = scalar.wp_real(0.0)
    if active[w] == 0:
        return
    if active[w] != 1:
        status[w] = 1
        return
    for k in range(48):
        if not wp.isfinite(qpos[w, k]):
            status[w] = 2
            return
    for k in range(43):
        if not wp.isfinite(qvel[w, k]):
            status[w] = 2
            return
    # Preserve original joint-before-dof sequence, including zero springs.
    for i in range(18):
        t = jtype[i]
        s = stiffness[i]
        q = qadr[i]
        d = dadr[i]
        if t == 0:  # mjJNT_FREE: only translational part; rotational k=0.
            for axis in range(3):
                x = qpos[w, q + axis] - spring[q + axis]
                out[w, d + axis] = out[w, d + axis] - s * x
        elif t == 2 or t == 3:  # slide / hinge
            x = qpos[w, q] - spring[q]
            out[w, d] = out[w, d] - s * x
    for k in range(43):
        out[w, k] = out[w, k] - damping[k] * qvel[w, k]


class JointPassiveWorkspace:
    def __init__(self, model, capacity, device='cuda:0'):
        if (int(model.nq), int(model.nv), int(model.njnt), int(model.ntendon),
            int(model.nplugin)) != (48, 43, 18, 0, 0):
            raise ValueError('unsupported model shape or passive plugin/tendon')
        if int(model.opt_disableflags) != 0 or np.any(model.body_gravcomp):
            raise ValueError('unsupported passive branch')
        # Fluid options belong to the later passive suffix. They may remain
        # enabled in the original MJB; this prefix never reads them.
        if capacity <= 0:
            raise ValueError('invalid capacity')
        for i in range(18):
            if model.jnt_type[i] in (0, 1) and model.jnt_stiffness[i] != 0:
                raise ValueError('nonzero free/ball rotational spring unsupported')
            if model.jnt_type[i] not in (0, 2, 3):
                raise ValueError('unsupported joint type')
        self.capacity, self.device = capacity, device
        self.qpos = wp.zeros((capacity, 48), scalar.wp_real, device=device)
        self.qvel = wp.zeros((capacity, 43), scalar.wp_real, device=device)
        self.active = wp.zeros(capacity, wp.int32, device=device)
        self.out = wp.zeros((capacity, 43), scalar.wp_real, device=device)
        self.status = wp.zeros(capacity, wp.int32, device=device)
        for name, value, dtype in (
            ('jtype', model.jnt_type, wp.int32),
            ('qadr', model.jnt_qposadr, wp.int32),
            ('dadr', model.jnt_dofadr, wp.int32),
            ('stiffness', model.jnt_stiffness, scalar.wp_real),
            ('spring', model.qpos_spring, scalar.wp_real),
            ('damping', model.dof_damping, scalar.wp_real),
        ):
            arr = np.ascontiguousarray(value, dtype=np.int32 if dtype == wp.int32 else np.float64)
            setattr(self, name, wp.array(arr, dtype=dtype, device=device))

    def launch(self, stream=None):
        wp.launch(_passive, dim=self.capacity, inputs=[
            self.qpos, self.qvel, self.active, self.jtype, self.qadr,
            self.dadr, self.stiffness, self.spring, self.damping,
            self.out, self.status], device=self.device, stream=stream)
