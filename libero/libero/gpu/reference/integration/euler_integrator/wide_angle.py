"""Independent Euler full-angle diagnostic candidate.

Original accepted bounded kernel and guards remain unchanged. Only finite
half-angle >4 uses CUDA double sin/cos; accepted-domain Taylor remains exact.
Unscaled norm/overflow guards and atomic public commit/held policy are inherited.
Not production-qualified until actual native/source-prior independent acceptance.
"""
import libero.gpu.precision as scalar
import warp as wp
from .kernel import _prepare, _source_solve_labels, _commit
from .certified_trig import sin45, cos44
from .generic_workspace import GenericEulerWorkspace
wp.set_module_options({'enable_backward':False,'fuse_fp':True,'fast_math':False})

@wp.kernel(enable_backward=False)
def _integrate_wide(active: wp.array(dtype=wp.int32),
               qpos: wp.array(dtype=scalar.wp_real, ndim=2),
               qvel: wp.array(dtype=scalar.wp_real, ndim=2),
               time: wp.array(dtype=scalar.wp_real),
               qacc: wp.array(dtype=scalar.wp_real, ndim=2),
               solution: wp.array(dtype=scalar.wp_real, ndim=3),
               mass_status: wp.array(dtype=wp.int32),
               damped: wp.array(dtype=wp.int32),
               jtype: wp.array(dtype=wp.int32),
               padr: wp.array(dtype=wp.int32),
               vadr: wp.array(dtype=wp.int32),
               scratch_pos: wp.array(dtype=scalar.wp_real, ndim=2),
               scratch_vel: wp.array(dtype=scalar.wp_real, ndim=2),
               used_acc: wp.array(dtype=scalar.wp_real, ndim=2),
               scratch_time: wp.array(dtype=scalar.wp_real),
               axis: wp.array(dtype=scalar.wp_real, ndim=3),
               angle: wp.array(dtype=scalar.wp_real, ndim=2),
               increment: wp.array(dtype=scalar.wp_real, ndim=3),
               pre_norm: wp.array(dtype=scalar.wp_real, ndim=3),
               axis_fallback: wp.array(dtype=wp.int32, ndim=2),
               zero_angle: wp.array(dtype=wp.int32, ndim=2),
               quat_fallback: wp.array(dtype=wp.int32, ndim=2),
               status: wp.array(dtype=wp.int32),
               dt: scalar.wp_real, nq: int, nv: int, njnt: int):
    w = wp.tid()
    if active[w] != 1 or status[w] != 0:
        return
    if damped[w] != 0 and mass_status[w] != 0 and mass_status[w] != 5:
        status[w] = 10 + mass_status[w]
        return
    for i in range(nv):
        acc = qacc[w, i]
        if damped[w] != 0:
            acc = solution[w, 0, i]
        used_acc[w, i] = acc
        scratch_vel[w, i] = qvel[w, i] + dt*acc
        if not wp.isfinite(scratch_vel[w, i]):
            status[w] = 3
            return
    for i in range(nq):
        scratch_pos[w, i] = qpos[w, i]
    qi = wp.int32(0)
    for j in range(njnt):
        pa = padr[j]
        va = vadr[j]
        kind = jtype[j]
        if kind == 0:  # mjJNT_FREE
            for k in range(3):
                scratch_pos[w, pa+k] = scratch_pos[w, pa+k] + dt*scratch_vel[w, va+k]
            pa += 3
            va += 3
        elif kind == 2 or kind == 3:  # slide / hinge
            scratch_pos[w, pa] = scratch_pos[w, pa] + dt*scratch_vel[w, va]
        if kind == 0 or kind == 1:  # free / ball
            x = scratch_vel[w, va]
            y = scratch_vel[w, va+1]
            z = scratch_vel[w, va+2]
            # The original unscaled norm graph is part of the support domain.
            # A finite velocity may overflow a square or the sum, then a
            # reciprocal can hide that overflow in apparently finite output.
            xx=x*x
            yy=y*y
            zz=z*z
            sum_xy=xx+yy
            sum_xyz=sum_xy+zz
            if not wp.isfinite(xx) or not wp.isfinite(yy) or not wp.isfinite(zz) or not wp.isfinite(sum_xy) or not wp.isfinite(sum_xyz):
                status[w]=3
                return
            length = wp.sqrt(sum_xyz)
            if not wp.isfinite(length):
                status[w]=3
                return
            axis_fallback[w,qi]=0
            if length < scalar.wp_real(1.0e-15):
                axis_fallback[w,qi]=1
                x = scalar.wp_real(1.0)
                y = scalar.wp_real(0.0)
                z = scalar.wp_real(0.0)
            else:
                inv = scalar.wp_real(1.0)/length
                x = x*inv
                y = y*inv
                z = z*inv
            a = dt*length
            if not wp.isfinite(a) or not wp.isfinite(x) or not wp.isfinite(y) or not wp.isfinite(z):
                status[w]=3
                return
            axis[w, qi, 0] = x
            axis[w, qi, 1] = y
            axis[w, qi, 2] = z
            angle[w, qi] = a
            zero_angle[w,qi]=0
            c = scalar.wp_real(1.0)
            s = scalar.wp_real(0.0)
            if a != scalar.wp_real(0.0):
                half = a*scalar.wp_real(0.5)
                if not wp.isfinite(half):
                    status[w] = 3
                    return
                if wp.abs(half) <= scalar.wp_real(4.0):
                    s = sin45(half)
                    c = cos44(half)
                else:
                    # CUDA binary64 full-range libdevice (no fast intrinsics).
                    s = wp.sin(half)
                    c = wp.cos(half)
                if not wp.isfinite(s) or not wp.isfinite(c):
                    status[w]=3
                    return
            else:
                zero_angle[w,qi]=1
            b0 = c
            b1 = x*s
            b2 = y*s
            b3 = z*s
            if not wp.isfinite(b0) or not wp.isfinite(b1) or not wp.isfinite(b2) or not wp.isfinite(b3):
                status[w]=3
                return
            if a == scalar.wp_real(0.0):
                b1 = scalar.wp_real(0.0)
                b2 = scalar.wp_real(0.0)
                b3 = scalar.wp_real(0.0)
            increment[w, qi, 0] = b0
            increment[w, qi, 1] = b1
            increment[w, qi, 2] = b2
            increment[w, qi, 3] = b3
            a0 = scratch_pos[w, pa]
            a1 = scratch_pos[w, pa+1]
            a2 = scratch_pos[w, pa+2]
            a3 = scratch_pos[w, pa+3]
            t0 = a0*b0 - a1*b1 - a2*b2 - a3*b3
            t1 = a0*b1 + a1*b0 + a2*b3 - a3*b2
            t2 = a0*b2 - a1*b3 + a2*b0 + a3*b1
            t3 = a0*b3 + a1*b2 - a2*b1 + a3*b0
            if not wp.isfinite(t0) or not wp.isfinite(t1) or not wp.isfinite(t2) or not wp.isfinite(t3):
                status[w]=3
                return
            pre_norm[w, qi, 0] = t0
            pre_norm[w, qi, 1] = t1
            pre_norm[w, qi, 2] = t2
            pre_norm[w, qi, 3] = t3
            q0sq=t0*t0
            q1sq=t1*t1
            q2sq=t2*t2
            q3sq=t3*t3
            qsum01=q0sq+q1sq
            qsum012=qsum01+q2sq
            qsum=qsum012+q3sq
            if not wp.isfinite(q0sq) or not wp.isfinite(q1sq) or not wp.isfinite(q2sq) or not wp.isfinite(q3sq) or not wp.isfinite(qsum01) or not wp.isfinite(qsum012) or not wp.isfinite(qsum):
                status[w]=3
                return
            qlength = wp.sqrt(qsum)
            if not wp.isfinite(qlength):
                status[w]=3
                return
            quat_fallback[w,qi]=0
            if qlength < scalar.wp_real(1.0e-15):
                quat_fallback[w,qi]=1
                t0 = scalar.wp_real(1.0)
                t1 = scalar.wp_real(0.0)
                t2 = scalar.wp_real(0.0)
                t3 = scalar.wp_real(0.0)
            else:
                qinv = scalar.wp_real(1.0)/qlength
                t0 = t0*qinv
                t1 = t1*qinv
                t2 = t2*qinv
                t3 = t3*qinv
            if not wp.isfinite(t0) or not wp.isfinite(t1) or not wp.isfinite(t2) or not wp.isfinite(t3):
                status[w]=3
                return
            scratch_pos[w, pa] = t0
            scratch_pos[w, pa+1] = t1
            scratch_pos[w, pa+2] = t2
            scratch_pos[w, pa+3] = t3
            qi += 1
    scratch_time[w] = time[w] + dt
    if not wp.isfinite(scratch_time[w]):
        status[w] = 3
        return
    for i in range(nq):
        if not wp.isfinite(scratch_pos[w, i]):
            status[w] = 3
            return


class WideAngleEulerWorkspace(GenericEulerWorkspace):
    qualification_status="pending_independent_full_angle_acceptance"
    def launch(self, stream=None):
        m=self.mass
        wp.launch(_prepare,dim=self.capacity,inputs=[self.active,self.qpos,self.qvel,self.time,
            self.qM,self.qfrc_smooth,self.qfrc_constraint,self.qacc,self.dof_damping,m.dof_Madr,
            m.qm,m.rhs,m.count,m.active,self.damped,self.status,self.dt,self.nq,self.nv,self.nM],
            device=self.device,stream=stream)
        m.launch(stream=stream)
        wp.launch(_source_solve_labels,dim=self.capacity,inputs=[self.active,self.damped,
            m.status,m.rhs,m.ld,m.dof_Madr,m.dof_parentid,m.dof_simplenum,self.zero_work,
            self.zero_skip,self.nv],device=self.device,stream=stream)
        wp.launch(_integrate_wide,dim=self.capacity,inputs=[self.active,self.qpos,self.qvel,self.time,
            self.qacc,m.solution,m.status,self.damped,self.jnt_type,self.jnt_qposadr,self.jnt_dofadr,
            self.qpos_after,self.qvel_after,self.qacc_used,self.time_after,self.axis,self.angle,self.increment,
            self.pre_norm,self.axis_fallback,self.zero_angle,self.quat_fallback,
            self.status,self.dt,self.nq,self.nv,self.njnt],device=self.device,stream=stream)
        wp.launch(_commit,dim=self.capacity,inputs=[self.active,self.qpos,self.qvel,self.time,
            self.qH,self.qHDiagInv,m.ld,m.inv,m.qm,m.rhs,m.sqrt_inv,m.solution,
            m.pivot,m.warning_count,m.warning_last,self.damped,self.qpos_after,self.qvel_after,
            self.qacc_used,self.time_after,self.axis,self.angle,self.increment,self.pre_norm,
            self.axis_fallback,self.zero_angle,self.quat_fallback,
            self.zero_work,self.zero_skip,self.status,self.nq,self.nv,self.nM,self.nquat],device=self.device,stream=stream)
