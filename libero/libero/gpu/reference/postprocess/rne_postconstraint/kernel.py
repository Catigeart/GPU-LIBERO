"""Fixed elliptic mj_rnePostConstraint source order, followed by accepted N10.

Source: MuJoCo 2.3.7 Apache-2.0 engine_core_smooth.c:1446-1592,
engine_support.c:mj_contactForce, engine_util_spatial.c, engine_sensor.c.
The production call receives already solved efc_force and does no CPU math.
"""
from __future__ import annotations
import libero.gpu.precision as scalar
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import warp as wp

_n10_path=(GPU_ROOT / 'reference/postprocess/force_sensor/kernel.py')
_n10_spec=importlib.util.spec_from_file_location('d03_accepted_n10_kernel',_n10_path)
_n10=importlib.util.module_from_spec(_n10_spec)
sys.modules[_n10_spec.name]=_n10
_n10_spec.loader.exec_module(_n10)
ForceSensorWorkspace=_n10.ForceSensorWorkspace

wp.set_module_options({'enable_backward': False, 'fuse_fp': True, 'fast_math': False})
V6 = wp.types.vector(length=6, dtype=scalar.wp_real)


@wp.func
def _dofmul(a: wp.array2d(dtype=scalar.wp_real), q: wp.array2d(dtype=scalar.wp_real),
            w: int, adr: int, count: int, k: int) -> scalar.wp_real:
    if count == 1:
        return a[w, 6*adr+k]*q[w, adr]
    out = scalar.wp_real(0.0)
    if count > 1:
        for i in range(count):
            scale = q[w, adr+i]
            if scale != scalar.wp_real(0.0):
                out += a[w, 6*(adr+i)+k]*scale
    return out


@wp.func
def _inert(i: wp.array2d(dtype=scalar.wp_real), v: wp.array2d(dtype=scalar.wp_real),
           w: int, body: int) -> V6:
    b = 10*body
    a = 6*body
    v0,v1,v2 = v[w,a],v[w,a+1],v[w,a+2]
    v3,v4,v5 = v[w,a+3],v[w,a+4],v[w,a+5]
    f0 = i[w,b]*v0+i[w,b+3]*v1+i[w,b+4]*v2-i[w,b+8]*v4+i[w,b+7]*v5
    f1 = i[w,b+3]*v0+i[w,b+1]*v1+i[w,b+5]*v2+i[w,b+8]*v3-i[w,b+6]*v5
    f2 = i[w,b+4]*v0+i[w,b+5]*v1+i[w,b+2]*v2-i[w,b+7]*v3+i[w,b+6]*v4
    f3 = i[w,b+8]*v1-i[w,b+7]*v2+i[w,b+9]*v3
    f4 = i[w,b+6]*v2-i[w,b+8]*v0+i[w,b+9]*v4
    f5 = i[w,b+7]*v0-i[w,b+6]*v1+i[w,b+9]*v5
    return V6(f0,f1,f2,f3,f4,f5)


@wp.func
def _inert_acc(i: wp.array2d(dtype=scalar.wp_real), v: wp.array2d(dtype=scalar.wp_real),
               w: int, body: int) -> V6:
    # Separate function because cacc and cvel use different device buffers.
    return _inert(i,v,w,body)


@wp.kernel(enable_backward=False)
def _rne(active: wp.array(dtype=wp.int32),
         qvel: wp.array2d(dtype=scalar.wp_real), qacc: wp.array2d(dtype=scalar.wp_real),
         cinert: wp.array2d(dtype=scalar.wp_real), cvel: wp.array2d(dtype=scalar.wp_real),
         cdof: wp.array2d(dtype=scalar.wp_real), cdof_dot: wp.array2d(dtype=scalar.wp_real),
         xipos: wp.array3d(dtype=scalar.wp_real), com: wp.array3d(dtype=scalar.wp_real),
         site_pos: wp.array3d(dtype=scalar.wp_real), site_mat: wp.array3d(dtype=scalar.wp_real),
         xfrc: wp.array2d(dtype=scalar.wp_real), efc: wp.array2d(dtype=scalar.wp_real),
         ncon: wp.array(dtype=wp.int32), nefc: wp.array(dtype=wp.int32),
         nequal: wp.array(dtype=wp.int32),
         cdim: wp.array2d(dtype=wp.int32), caddr: wp.array2d(dtype=wp.int32),
         cgeom1: wp.array2d(dtype=wp.int32), cgeom2: wp.array2d(dtype=wp.int32),
         cframe: wp.array2d(dtype=scalar.wp_real), cpos: wp.array2d(dtype=scalar.wp_real),
         parent: wp.array(dtype=wp.int32), dofadr: wp.array(dtype=wp.int32),
         dofnum: wp.array(dtype=wp.int32), root: wp.array(dtype=wp.int32),
         geom_body: wp.array(dtype=wp.int32), gravity: wp.array(dtype=scalar.wp_real),
         ccap: int, ecap: int,
         cacc: wp.array2d(dtype=scalar.wp_real), ext: wp.array2d(dtype=scalar.wp_real),
         internal: wp.array3d(dtype=scalar.wp_real),
         decoded: wp.array3d(dtype=scalar.wp_real), status: wp.array(dtype=wp.int32)):
    w = wp.tid()
    status[w] = 0
    for j in range(38):
        for k in range(6):
            cacc[w,6*j+k] = scalar.wp_real(0.0)
            ext[w,6*j+k] = scalar.wp_real(0.0)
            internal[w,j,k] = scalar.wp_real(0.0)
    for j in range(ccap):
        for k in range(6):
            decoded[w,j,k] = scalar.wp_real(0.0)
    if active[w] == 0:
        return
    if active[w] != 1:
        status[w] = 1
        return
    nc, ne = ncon[w], nefc[w]
    if nc < 0 or nc > ccap or ne < 0 or ne > ecap:
        status[w] = 2
        return
    if nequal[w] != 0:
        status[w] = 6
        return
    for j in range(43):
        if not wp.isfinite(qvel[w,j]) or not wp.isfinite(qacc[w,j]):
            status[w] = 4
            return
    for j in range(43*6):
        if not wp.isfinite(cdof[w,j]) or not wp.isfinite(cdof_dot[w,j]):
            status[w] = 4
            return
    for j in range(38):
        for k in range(10):
            if not wp.isfinite(cinert[w,10*j+k]):
                status[w] = 4
                return
        for k in range(6):
            if not wp.isfinite(cvel[w,6*j+k]) or not wp.isfinite(xfrc[w,6*j+k]):
                status[w] = 4
                return
        for k in range(3):
            if not wp.isfinite(xipos[w,j,k]) or not wp.isfinite(com[w,j,k]):
                status[w] = 4
                return
    for j in range(ne):
        if not wp.isfinite(efc[w,j]):
            status[w] = 4
            return
    for j in range(31):
        for k in range(3):
            if not wp.isfinite(site_pos[w,j,k]):
                status[w] = 4
                return
        for k in range(9):
            if not wp.isfinite(site_mat[w,j,k]):
                status[w] = 4
                return
    for j in range(nc):
        ad, dim = caddr[w,j], cdim[w,j]
        if ad >= 0:
            if dim < 1 or dim > 6:
                status[w] = 3
                return
            # ne-dim is bounded once dim is validated; ad+dim could overflow.
            if ad > ne-dim or cgeom1[w,j] < 0 or cgeom1[w,j] >= 270 or \
               cgeom2[w,j] < 0 or cgeom2[w,j] >= 270:
                status[w] = 3
                return
        if ad >= 0:
            for k in range(9):
                if not wp.isfinite(cframe[w,9*j+k]):
                    status[w] = 4
                    return
            for k in range(3):
                if not wp.isfinite(cpos[w,3*j+k]):
                    status[w] = 4
                    return
    # Original mju_zero(cacc,6), followed by linear -gravity.
    for k in range(3):
        cacc[w,3+k] = gravity[k]*scalar.wp_real(-1.0)
    # Applied wrench is force:torque at xipos; output is torque:force at root COM.
    for j in range(1,38):
        nonzero = False
        for k in range(6):
            if xfrc[w,6*j+k] != scalar.wp_real(0.0):
                nonzero = True
        if nonzero:
            dx = com[w,root[j],0]-xipos[w,j,0]
            dy = com[w,root[j],1]-xipos[w,j,1]
            dz = com[w,root[j],2]-xipos[w,j,2]
            fx,fy,fz = xfrc[w,6*j],xfrc[w,6*j+1],xfrc[w,6*j+2]
            tx,ty,tz = xfrc[w,6*j+3],xfrc[w,6*j+4],xfrc[w,6*j+5]
            ext[w,6*j] += tx-(dy*fz-dz*fy)
            ext[w,6*j+1] += ty-(dz*fx-dx*fz)
            ext[w,6*j+2] += tz-(dx*fy-dy*fx)
            ext[w,6*j+3] += fx
            ext[w,6*j+4] += fy
            ext[w,6*j+5] += fz
    # Fixed elliptic mj_contactForce: copy dim local force:torque entries.
    for j in range(nc):
        decode_address = caddr[w,j]
        if decode_address >= 0:
            for k in range(cdim[w,j]):
                decoded[w,j,k] = efc[w,decode_address+k]
            f0 = cframe[w,9*j]*decoded[w,j,0]+cframe[w,9*j+3]*decoded[w,j,1]+cframe[w,9*j+6]*decoded[w,j,2]
            f1 = cframe[w,9*j+1]*decoded[w,j,0]+cframe[w,9*j+4]*decoded[w,j,1]+cframe[w,9*j+7]*decoded[w,j,2]
            f2 = cframe[w,9*j+2]*decoded[w,j,0]+cframe[w,9*j+5]*decoded[w,j,1]+cframe[w,9*j+8]*decoded[w,j,2]
            t0 = cframe[w,9*j]*decoded[w,j,3]+cframe[w,9*j+3]*decoded[w,j,4]+cframe[w,9*j+6]*decoded[w,j,5]
            t1 = cframe[w,9*j+1]*decoded[w,j,3]+cframe[w,9*j+4]*decoded[w,j,4]+cframe[w,9*j+7]*decoded[w,j,5]
            t2 = cframe[w,9*j+2]*decoded[w,j,3]+cframe[w,9*j+5]*decoded[w,j,4]+cframe[w,9*j+8]*decoded[w,j,5]
            for side in range(2):
                geom_index = cgeom1[w,j]+side*(cgeom2[w,j]-cgeom1[w,j])
                body = geom_body[geom_index]
                if body > 0:
                    dx = com[w,root[body],0]-cpos[w,3*j]
                    dy = com[w,root[body],1]-cpos[w,3*j+1]
                    dz = com[w,root[body],2]-cpos[w,3*j+2]
                    # mju_subFrom on body1 and mju_addTo on body2.
                    if side == 0:
                        ext[w,6*body] -= t0-(dy*f2-dz*f1)
                        ext[w,6*body+1] -= t1-(dz*f0-dx*f2)
                        ext[w,6*body+2] -= t2-(dx*f1-dy*f0)
                        ext[w,6*body+3] -= f0
                        ext[w,6*body+4] -= f1
                        ext[w,6*body+5] -= f2
                    else:
                        ext[w,6*body] += t0-(dy*f2-dz*f1)
                        ext[w,6*body+1] += t1-(dz*f0-dx*f2)
                        ext[w,6*body+2] += t2-(dx*f1-dy*f0)
                        ext[w,6*body+3] += f0
                        ext[w,6*body+4] += f1
                        ext[w,6*body+5] += f2
    # Forward recurrence in original body order.
    for j in range(1,38):
        adr, count = dofadr[j], dofnum[j]
        for k in range(6):
            first = _dofmul(cdof_dot,qvel,w,adr,count,k)
            cacc[w,6*j+k] = cacc[w,6*parent[j]+k]+first
        for k in range(6):
            second = _dofmul(cdof,qacc,w,adr,count,k)
            cacc[w,6*j+k] += second
        inertial = _inert_acc(cinert,cacc,w,j)
        corr = _inert(cinert,cvel,w,j)
        a = 6*j
        cv0,cv1,cv2 = cvel[w,a],cvel[w,a+1],cvel[w,a+2]
        cv3,cv4,cv5 = cvel[w,a+3],cvel[w,a+4],cvel[w,a+5]
        x0 = -cv2*corr[1]+cv1*corr[2]
        x1 = cv2*corr[0]-cv0*corr[2]
        x2 = -cv1*corr[0]+cv0*corr[1]
        x3 = -cv2*corr[4]+cv1*corr[5]
        x4 = cv2*corr[3]-cv0*corr[5]
        x5 = -cv1*corr[3]+cv0*corr[4]
        x0 += -cv5*corr[4]+cv4*corr[5]
        x1 += cv5*corr[3]-cv3*corr[5]
        x2 += -cv4*corr[3]+cv3*corr[4]
        internal[w,j,0] = (inertial[0]+x0)-ext[w,a]
        internal[w,j,1] = (inertial[1]+x1)-ext[w,a+1]
        internal[w,j,2] = (inertial[2]+x2)-ext[w,a+2]
        internal[w,j,3] = (inertial[3]+x3)-ext[w,a+3]
        internal[w,j,4] = (inertial[4]+x4)-ext[w,a+4]
        internal[w,j,5] = (inertial[5]+x5)-ext[w,a+5]
    # Backward recurrence includes parent 0, unlike isolated N05 bias.
    for idx in range(37):
        j = 37-idx
        p = parent[j]
        for k in range(6):
            internal[w,p,k] += internal[w,j,k]
    for j in range(38):
        for k in range(6):
            if not wp.isfinite(cacc[w,6*j+k]) or not wp.isfinite(ext[w,6*j+k]) or \
               not wp.isfinite(internal[w,j,k]):
                status[w] = 5
    if status[w] == 5:
        for j in range(38):
            for k in range(6):
                cacc[w,6*j+k] = scalar.wp_real(0.0)
                ext[w,6*j+k] = scalar.wp_real(0.0)
                internal[w,j,k] = scalar.wp_real(0.0)


class RNEPostWorkspace:
    """Prepared fixed-model buffers; `launch` performs only CUDA kernels."""

    def __init__(self, static, capacity: int, contact_capacity: int,
                 efc_capacity: int, device='cuda:0'):
        if capacity < 1 or contact_capacity < 1 or efc_capacity < 1:
            raise ValueError('capacity')
        if not wp.get_device(device).is_cuda:
            raise ValueError('CUDA device required')
        self.capacity, self.contact_capacity, self.efc_capacity = capacity,contact_capacity,efc_capacity
        self.device = device
        self.static = static
        if (int(static.nbody),int(static.nv),int(static.ngeom),int(static.nsite),
            int(static.nsensor),int(static.neq),int(static.cone),
            int(static.opt_enableflags),int(static.opt_disableflags)) != (38,43,270,31,2,0,1,0,0):
            raise ValueError('unsupported fixed elliptic model or equality domain')
        def get_int(name,shape):
            a=np.asarray(getattr(static,name))
            if a.shape!=shape or a.dtype!=np.int32:
                raise TypeError(name)
            return a
        for name,shape in (('body_parentid',(38,)),('body_dofadr',(38,)),
                           ('body_dofnum',(38,)),('body_rootid',(38,)),
                           ('geom_bodyid',(270,)),('site_bodyid',(31,)),
                           ('sensor_type',(2,)),('sensor_objid',(2,)),('sensor_adr',(2,))):
            get_int(name,shape)
        bp=get_int('body_parentid',(38,)); ba=get_int('body_dofadr',(38,)); bn=get_int('body_dofnum',(38,))
        br=get_int('body_rootid',(38,)); gb=get_int('geom_bodyid',(270,))
        if bp[0]!=0 or bn[0]!=0 or np.any(bp[1:]<0) or np.any(bp[1:]>=np.arange(1,38)) or \
           np.any(br<0) or np.any(br>=38) or np.any(gb<0) or np.any(gb>=38):
            raise ValueError('invalid body topology')
        dof_coverage=np.zeros(43,np.int32)
        for j in range(1,38):
            if bn[j] < 0 or bn[j] > 6 or (bn[j] and (ba[j]<0 or ba[j]+bn[j]>43)):
                raise ValueError('invalid dof range')
            if bn[j]:dof_coverage[ba[j]:ba[j]+bn[j]]+=1
        if np.any(dof_coverage!=1):
            raise ValueError('invalid dof coverage')
        g=np.asarray(static.opt_gravity)
        if g.shape!=(3,) or g.dtype!=np.float64 or not np.all(np.isfinite(g)):
            raise TypeError('gravity')
        self.parent=wp.array(bp,dtype=wp.int32,device=device)
        self.dofadr=wp.array(ba,dtype=wp.int32,device=device)
        self.dofnum=wp.array(bn,dtype=wp.int32,device=device)
        self.root=wp.array(br,dtype=wp.int32,device=device)
        self.geom_body=wp.array(gb,dtype=wp.int32,device=device)
        self.gravity=wp.array(g,dtype=scalar.wp_real,device=device)
        n=capacity;c=contact_capacity;e=efc_capacity
        shapes={'active':((n,),wp.int32),'qvel':((n,43),scalar.wp_real),
                'qacc':((n,43),scalar.wp_real),'cinert':((n,380),scalar.wp_real),
                'cvel':((n,228),scalar.wp_real),'cdof':((n,258),scalar.wp_real),
                'cdof_dot':((n,258),scalar.wp_real),'xipos':((n,38,3),scalar.wp_real),
                'subtree_com':((n,38,3),scalar.wp_real),'xfrc_applied':((n,228),scalar.wp_real),
                'efc_force':((n,e),scalar.wp_real),'ncon':((n,),wp.int32),
                'nefc':((n,),wp.int32),'ne':((n,),wp.int32),'contact_dim':((n,c),wp.int32),
                'contact_efc_address':((n,c),wp.int32),'contact_geom1':((n,c),wp.int32),
                'contact_geom2':((n,c),wp.int32),'contact_frame':((n,c*9),scalar.wp_real),
                'contact_pos':((n,c*3),scalar.wp_real),'site_xpos':((n,31,3),scalar.wp_real),
                'site_xmat':((n,31,9),scalar.wp_real)}
        for name,(shape,dtype) in shapes.items():
            setattr(self,name,wp.zeros(shape,dtype=dtype,device=device))
        self.cacc=wp.zeros((n,228),dtype=scalar.wp_real,device=device)
        self.cfrc_ext=wp.zeros((n,228),dtype=scalar.wp_real,device=device)
        self.contact_force=wp.zeros((n,c,6),dtype=scalar.wp_real,device=device)
        self.status=wp.zeros(n,dtype=wp.int32,device=device)
        # N10 output is the SAME device allocation written by the RNE producer.
        sensor_model=SimpleNamespace(**{name:getattr(static,name) for name in
            ('nbody','nsite','nsensor','site_bodyid','body_rootid','sensor_type',
             'sensor_objtype','sensor_objid','sensor_adr','sensor_dim',
             'sensor_needstage','sensor_cutoff','sensor_noise')},
            nsensordata=6,nplugin=0,opt_disableflags=static.opt_disableflags,
            opt_enableflags=static.opt_enableflags)
        self.sensor=ForceSensorWorkspace(sensor_model,n,device=device)
        self.cfrc_int=self.sensor.cfrc
        self.sensor.active=self.active
        self.sensor.com=self.subtree_com
        self.sensor.site_pos=self.site_xpos
        self.sensor.site_mat=self.site_xmat

    def launch(self,stream=None):
        wp.launch(_rne,dim=self.capacity,inputs=[self.active,self.qvel,self.qacc,
            self.cinert,self.cvel,self.cdof,self.cdof_dot,self.xipos,self.subtree_com,
            self.site_xpos,self.site_xmat,
            self.xfrc_applied,self.efc_force,self.ncon,self.nefc,self.ne,self.contact_dim,
            self.contact_efc_address,self.contact_geom1,self.contact_geom2,
            self.contact_frame,self.contact_pos,self.parent,self.dofadr,self.dofnum,
            self.root,self.geom_body,self.gravity,self.contact_capacity,self.efc_capacity,
            self.cacc,self.cfrc_ext,self.cfrc_int,self.contact_force,self.status],
            device=self.device,stream=stream)
        self.sensor.launch(stream=stream)
        wp.launch(_sensor_finish,dim=self.capacity,inputs=[self.active,self.status,
            self.sensor.status,self.sensor.result],device=self.device,stream=stream)


@wp.kernel(enable_backward=False)
def _sensor_finish(active:wp.array(dtype=wp.int32),status:wp.array(dtype=wp.int32),
                   sensor_status:wp.array(dtype=wp.int32),
                   result:wp.array2d(dtype=scalar.wp_real)):
    w=wp.tid()
    if active[w]!=1 or status[w]!=0:
        return
    if sensor_status[w]!=0:
        status[w]=4
        return
    for k in range(6):
        if not wp.isfinite(result[w,k]):
            status[w]=5
    if status[w]==5:
        for k in range(6):
            result[w,k]=scalar.wp_real(0.0)
