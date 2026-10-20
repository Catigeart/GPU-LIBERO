"""MuJoCo 2.3.7 semi-implicit Euler on frozen N17/N06 sparse layouts.

The mass workspace owns its buffers; Euler only fills them through device
kernels.  Every production launch has three Euler kernels and the accepted
N06 factor/solve kernel.  A failed world never commits partial state.
"""
import libero.gpu.precision as scalar

import hashlib
import numpy as np
import warp as wp
from libero.gpu.reference.position.mass_solve.kernel import MassSolveWorkspace
from .certified_trig import sin45, cos44

wp.set_module_options({'enable_backward': False, 'fuse_fp': True,
                       'fast_math': False})

_JOINT_TOPOLOGIES={
    (48,43,153,18):{
        'nu':9,'nsensordata':6,
        'jnt_type':'4c0e0d380d48f2a6a7f66baa70f16e9bef2974dcffe93545a1da175cd5c9f2ce',
        'jnt_qposadr':'38ae4abac73ad2b845e1908b24e91cd00b1d3df40b5c6f885075fe46adb9ab35',
        'jnt_dofadr':'13c9448fa632713f92cc44c1c525c473e12ce7105bc7b23c30d8fcdf8b90a02b'},
    (13,11,36,4):{
        'nu':0,'nsensordata':0,
        'jnt_type':'05dc715f8e536c7c410383b0d77eb649831df1dfad839486e54ffc948ea3e3b2',
        'jnt_qposadr':'5fa93bdcd2715a0d5267fb2fd4aaa78a23ddb967b5ba7d6df47702921b096e53',
        'jnt_dofadr':'53468ebc113f40d8632767eefcd86745a491d078543a3c56ee537b64e4f149c1'}
}


@wp.kernel(enable_backward=False)
def _prepare(active: wp.array(dtype=wp.int32),
             qpos: wp.array(dtype=scalar.wp_real, ndim=2),
             qvel: wp.array(dtype=scalar.wp_real, ndim=2),
             time: wp.array(dtype=scalar.wp_real),
             qm: wp.array(dtype=scalar.wp_real, ndim=2),
             smooth: wp.array(dtype=scalar.wp_real, ndim=2),
             constraint: wp.array(dtype=scalar.wp_real, ndim=2),
             qacc: wp.array(dtype=scalar.wp_real, ndim=2),
             damping: wp.array(dtype=scalar.wp_real),
             madr: wp.array(dtype=wp.int32),
             mass_qm: wp.array(dtype=scalar.wp_real, ndim=2),
             mass_rhs: wp.array(dtype=scalar.wp_real, ndim=3),
             mass_count: wp.array(dtype=wp.int32),
             mass_active: wp.array(dtype=wp.int32),
             damped: wp.array(dtype=wp.int32),
             status: wp.array(dtype=wp.int32),
             dt: scalar.wp_real, nq: int, nv: int, nm: int):
    w = wp.tid()
    status[w] = 0
    damped[w] = 0
    mass_active[w] = 0
    mass_count[w] = 1
    # MhB/RHS are diagnostic device outputs too; clear them even when an
    # inactive world carries NaN payload or this slot was active previously.
    for a in range(nm):
        mass_qm[w,a]=scalar.wp_real(0.0)
    for r in range(4):
        for i in range(nv):
            mass_rhs[w,r,i]=scalar.wp_real(0.0)
    if active[w] == 0:
        return
    if active[w] != 1:
        status[w] = 1
        return
    if not wp.isfinite(time[w]):
        status[w] = 2
        return
    for i in range(nq):
        if not wp.isfinite(qpos[w, i]):
            status[w] = 2
            return
    for i in range(nv):
        if not wp.isfinite(qvel[w, i]) or not wp.isfinite(qacc[w, i]) or not wp.isfinite(smooth[w, i]) or not wp.isfinite(constraint[w, i]):
            status[w] = 2
            return
        if damping[i] > scalar.wp_real(0.0):
            damped[w] = 1
    if damped[w] == 0:
        return
    for a in range(nm):
        if not wp.isfinite(qm[w, a]):
            status[w] = 2
            return
        mass_qm[w, a] = qm[w, a]
    for i in range(nv):
        a = madr[i]
        mass_qm[w, a] = mass_qm[w, a] + dt*damping[i]
        mass_rhs[w, 0, i] = smooth[w, i] + constraint[w, i]
    mass_active[w] = 1


@wp.kernel(enable_backward=False)
def _source_solve_labels(active: wp.array(dtype=wp.int32),
                         damped: wp.array(dtype=wp.int32),
                         mass_status: wp.array(dtype=wp.int32),
                         rhs: wp.array(dtype=scalar.wp_real, ndim=3),
                         ld: wp.array(dtype=scalar.wp_real, ndim=2),
                         madr: wp.array(dtype=wp.int32),
                         parent: wp.array(dtype=wp.int32),
                         simple: wp.array(dtype=wp.int32),
                         work: wp.array(dtype=scalar.wp_real, ndim=2),
                         zero_skip: wp.array(dtype=wp.int32, ndim=2), nv: int):
    """Device diagnostic replay of source first triangular pass.

    N06's internal zero-skip is not instrumented; this independent device
    replay records the branch implied by its actual LD and RHS buffers.
    """
    w=wp.tid()
    for i in range(nv):
        zero_skip[w,i]=0
        work[w,i]=scalar.wp_real(0.0)
    if active[w]!=1 or damped[w]==0 or (mass_status[w]!=0 and mass_status[w]!=5):
        return
    for i in range(nv):
        work[w,i]=rhs[w,0,i]
    for i in range(nv-1,-1,-1):
        if simple[i]!=0:
            continue
        if work[w,i]==scalar.wp_real(0.0):
            zero_skip[w,i]=1
        else:
            a=madr[i]+1
            j=parent[i]
            while j>=0:
                work[w,j]=work[w,j]-ld[w,a]*work[w,i]
                a+=1
                j=parent[j]


@wp.kernel(enable_backward=False)
def _integrate(active: wp.array(dtype=wp.int32),
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
                if not wp.isfinite(half) or wp.abs(half) > scalar.wp_real(4.0):
                    status[w] = 4
                    return
                s = sin45(half)
                c = cos44(half)
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


@wp.kernel(enable_backward=False)
def _commit(active: wp.array(dtype=wp.int32),
            qpos: wp.array(dtype=scalar.wp_real, ndim=2),
            qvel: wp.array(dtype=scalar.wp_real, ndim=2),
            time: wp.array(dtype=scalar.wp_real),
            qH: wp.array(dtype=scalar.wp_real, ndim=2),
            qHDiagInv: wp.array(dtype=scalar.wp_real, ndim=2),
            mass_ld: wp.array(dtype=scalar.wp_real, ndim=2),
            mass_inv: wp.array(dtype=scalar.wp_real, ndim=2),
            mass_qm: wp.array(dtype=scalar.wp_real, ndim=2),
            mass_rhs: wp.array(dtype=scalar.wp_real, ndim=3),
            mass_sqrt_inv: wp.array(dtype=scalar.wp_real, ndim=2),
            mass_solution: wp.array(dtype=scalar.wp_real, ndim=3),
            mass_pivot: wp.array(dtype=wp.int32, ndim=2),
            mass_warning_count: wp.array(dtype=wp.int32),
            mass_warning_last: wp.array(dtype=wp.int32),
            damped: wp.array(dtype=wp.int32),
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
            zero_work: wp.array(dtype=scalar.wp_real, ndim=2),
            zero_skip: wp.array(dtype=wp.int32, ndim=2),
            status: wp.array(dtype=wp.int32), nq: int, nv: int, nm: int, nquat: int):
    w = wp.tid()
    if active[w] != 1 or status[w] != 0:
        for a in range(nm):
            mass_qm[w,a]=scalar.wp_real(0.0)
            mass_ld[w,a]=scalar.wp_real(0.0)
        for i in range(nv):
            mass_inv[w,i]=scalar.wp_real(0.0)
            mass_sqrt_inv[w,i]=scalar.wp_real(0.0)
            mass_pivot[w,i]=0
            zero_work[w,i]=scalar.wp_real(0.0)
            zero_skip[w,i]=0
            for r in range(4):
                mass_rhs[w,r,i]=scalar.wp_real(0.0)
                mass_solution[w,r,i]=scalar.wp_real(0.0)
        mass_warning_count[w]=0
        mass_warning_last[w]=-1
        for i in range(nq):
            scratch_pos[w, i] = scalar.wp_real(0.0)
        for i in range(nv):
            scratch_vel[w, i] = scalar.wp_real(0.0)
            used_acc[w, i] = scalar.wp_real(0.0)
        scratch_time[w] = scalar.wp_real(0.0)
        for i in range(nquat):
            angle[w, i] = scalar.wp_real(0.0)
            axis_fallback[w,i]=0
            zero_angle[w,i]=0
            quat_fallback[w,i]=0
            for k in range(3):
                axis[w, i, k] = scalar.wp_real(0.0)
            for k in range(4):
                increment[w, i, k] = scalar.wp_real(0.0)
                pre_norm[w, i, k] = scalar.wp_real(0.0)
        return
    for i in range(nq):
        qpos[w, i] = scratch_pos[w, i]
    for i in range(nv):
        qvel[w, i] = scratch_vel[w, i]
    time[w] = scratch_time[w]
    if damped[w] != 0:
        for a in range(nm):
            qH[w, a] = mass_ld[w, a]
        for i in range(nv):
            qHDiagInv[w, i] = mass_inv[w, i]


class EulerWorkspace:
    """Device-resident batched Euler state for the two accepted sparse layouts.

    ``upload`` validates original host dtype/shape before transfer. ``launch``
    reads only device buffers and is safe for Warp graph capture/replay. Status:
    0 success, 1 bad active, 2 nonfinite input, 3 nonfinite arithmetic,
    4 half-angle outside [-4,4], 10+N06 error.  N06 status 5 records native
    pivot repair and still commits the repaired-system result.
    """

    def __init__(self, model, capacity, device='cuda:0'):
        if int(model.na) != 0 or int(model.nplugin) != 0 or int(model.opt.integrator) != 0:
            raise ValueError('Euler/na=0/nplugin=0 required')
        self.mass = MassSolveWorkspace(model, capacity, device)
        self.nq, self.nv, self.nM, self.njnt = map(int, (model.nq,model.nv,model.nM,model.njnt))
        expected=_JOINT_TOPOLOGIES.get((self.nq,self.nv,self.nM,self.njnt))
        if expected is None:
            raise ValueError('unsupported Euler joint dimensions')
        self.capacity, self.device = int(capacity), device
        self.nu=int(getattr(model,'nu',0))
        self.nsensordata=int(getattr(model,'nsensordata',0))
        if self.nu!=expected['nu'] or self.nsensordata!=expected['nsensordata']:
            raise ValueError('unsupported preserved-state dimensions')
        self.dt = float(model.opt.timestep)
        if not np.isfinite(self.dt):
            raise ValueError('nonfinite timestep')
        for name, length, dtype in (('dof_damping',self.nv,np.float64),
                                    ('jnt_type',self.njnt,np.int32),
                                    ('jnt_qposadr',self.njnt,np.int32),
                                    ('jnt_dofadr',self.njnt,np.int32)):
            raw = np.asarray(getattr(model,name))
            if raw.shape != (length,) or (raw.dtype != np.float64 if dtype == np.float64 else raw.dtype.kind not in 'iu'):
                raise TypeError('static model schema: '+name)
            if dtype == np.int32 and (np.min(raw) < np.iinfo(np.int32).min or np.max(raw) > np.iinfo(np.int32).max):
                raise ValueError('static integer range: '+name)
            canonical=np.ascontiguousarray(raw,dtype=dtype)
            if name=='dof_damping':
                if not np.isfinite(canonical).all():
                    raise ValueError('nonfinite damping')
            elif hashlib.sha256(canonical.tobytes()).hexdigest()!=expected[name]:
                raise ValueError('unsupported Euler joint topology: '+name)
            setattr(self,name,wp.array(canonical,dtype=scalar.wp_real if dtype==np.float64 else wp.int32,device=device))
        jt=np.asarray(model.jnt_type,dtype=np.int32)
        qp=np.asarray(model.jnt_qposadr,dtype=np.int32)
        qv=np.asarray(model.jnt_dofadr,dtype=np.int32)
        for j,t in enumerate(jt):
            if t not in (0,1,2,3):raise ValueError('invalid joint type')
            spanq=7 if t==0 else 4 if t==1 else 1
            spanv=6 if t==0 else 3 if t==1 else 1
            if qp[j]<0 or qp[j]+spanq>self.nq or qv[j]<0 or qv[j]+spanv>self.nv:
                raise ValueError('joint address out of bounds')
        self.model_identity = hashlib.sha256(b''.join(np.asarray(getattr(model,n)).tobytes() for n in
            ('dof_Madr','dof_parentid','dof_simplenum','dof_damping','jnt_type','jnt_qposadr','jnt_dofadr'))+
            np.float64(self.dt).tobytes()+np.int32(self.nu).tobytes()+
            np.int32(self.nsensordata).tobytes()).hexdigest()
        f64 = scalar.wp_real
        self.active = wp.zeros(capacity,dtype=wp.int32,device=device)
        for name, shape in (('qpos',(capacity,self.nq)),('qvel',(capacity,self.nv)),
                            ('time',(capacity,)),('qM',(capacity,self.nM)),
                            ('qfrc_smooth',(capacity,self.nv)),('qfrc_constraint',(capacity,self.nv)),
                            ('qacc',(capacity,self.nv)),('qH',(capacity,self.nM)),
                            ('qHDiagInv',(capacity,self.nv)),('qpos_after',(capacity,self.nq)),
                            ('qvel_after',(capacity,self.nv)),('qacc_used',(capacity,self.nv)),
                            ('time_after',(capacity,))):
            setattr(self,name,wp.zeros(shape,dtype=f64,device=device))
        nquat = sum(int(t) in (0,1) for t in np.asarray(model.jnt_type))
        self.nquat = nquat
        self.axis = wp.zeros((capacity,nquat,3),dtype=f64,device=device)
        self.angle = wp.zeros((capacity,nquat),dtype=f64,device=device)
        self.increment = wp.zeros((capacity,nquat,4),dtype=f64,device=device)
        self.pre_norm = wp.zeros((capacity,nquat,4),dtype=f64,device=device)
        self.damped = wp.zeros(capacity,dtype=wp.int32,device=device)
        self.status = wp.zeros(capacity,dtype=wp.int32,device=device)
        self.qacc_warmstart=wp.zeros((capacity,self.nv),dtype=f64,device=device)
        self.ctrl=wp.zeros((capacity,self.nu),dtype=f64,device=device) if self.nu else None
        self.sensordata=wp.zeros((capacity,self.nsensordata),dtype=f64,device=device) if self.nsensordata else None
        self.axis_fallback=wp.zeros((capacity,nquat),dtype=wp.int32,device=device)
        self.zero_angle=wp.zeros((capacity,nquat),dtype=wp.int32,device=device)
        self.quat_fallback=wp.zeros((capacity,nquat),dtype=wp.int32,device=device)
        self.zero_skip=wp.zeros((capacity,self.nv),dtype=wp.int32,device=device)
        self.zero_work=wp.zeros((capacity,self.nv),dtype=f64,device=device)

    def upload(self, **fields):
        required = ('active','qpos','qvel','time','qM','qfrc_smooth',
                    'qfrc_constraint','qacc','qH','qHDiagInv',
                    'ctrl','qacc_warmstart','sensordata')
        if set(fields) != set(required):
            raise ValueError('dynamic fields mismatch')
        for name in required:
            arr = np.asarray(fields[name])
            dst = getattr(self,name)
            dtype = np.int32 if name=='active' else np.float64
            if name=='ctrl' and self.nu==0:
                if arr.dtype!=dtype or arr.shape!=(self.capacity,0):
                    raise TypeError((name,'empty schema'))
                continue
            if name=='sensordata' and self.nsensordata==0:
                if arr.dtype!=dtype or arr.shape!=(self.capacity,0):
                    raise TypeError((name,'empty schema'))
                continue
            if arr.dtype != dtype or tuple(arr.shape) != tuple(dst.shape):
                raise TypeError((name,'dtype/shape',arr.dtype,arr.shape,dst.shape))
            wp.copy(dst,wp.array(np.ascontiguousarray(arr),dtype=wp.int32 if name=='active' else scalar.wp_real,device=self.device))

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
        wp.launch(_integrate,dim=self.capacity,inputs=[self.active,self.qpos,self.qvel,self.time,
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
