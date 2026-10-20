"""MuJoCo 2.3.7 force/torque site sensors from given cfrc_int.

Apache-2.0 source: engine_sensor.c:mj_sensorAcc force/torque branches and
engine_util_spatial.c:mju_transformSpatial. This boundary excludes
mj_rnePostConstraint and all contact/constraint force production.
"""
import libero.gpu.precision as scalar

import numpy as np
import warp as wp

wp.set_module_options({'enable_backward': False, 'fuse_fp': True})


@wp.kernel(enable_backward=False)
def _sensor(cfrc: wp.array(dtype=scalar.wp_real, ndim=3),
            com: wp.array(dtype=scalar.wp_real, ndim=3),
            site_pos: wp.array(dtype=scalar.wp_real, ndim=3),
            site_mat: wp.array(dtype=scalar.wp_real, ndim=3),
            active: wp.array(dtype=wp.int32),
            sensor_type: wp.array(dtype=wp.int32),
            sensor_site: wp.array(dtype=wp.int32),
            sensor_adr: wp.array(dtype=wp.int32),
            site_body: wp.array(dtype=wp.int32),
            body_root: wp.array(dtype=wp.int32),
            result: wp.array(dtype=scalar.wp_real, ndim=2),
            status: wp.array(dtype=wp.int32)):
    w = wp.tid()
    status[w] = 0
    for k in range(6):
        result[w, k] = scalar.wp_real(0.0)
    if active[w] == 0:
        return
    if active[w] != 1:
        status[w] = 1
        return
    # Reject invalid active payloads before writing either sensor. Inactive
    # payloads are never read, including arbitrary NaN bit patterns.
    for i in range(38):
        for k in range(6):
            if not wp.isfinite(cfrc[w, i, k]):
                status[w] = 2
                return
        for k in range(3):
            if not wp.isfinite(com[w, i, k]):
                status[w] = 2
                return
    for i in range(31):
        for k in range(3):
            if not wp.isfinite(site_pos[w, i, k]):
                status[w] = 2
                return
        for k in range(9):
            if not wp.isfinite(site_mat[w, i, k]):
                status[w] = 2
                return

    for s in range(2):
        site = sensor_site[s]
        body = site_body[site]
        root = body_root[body]
        # mju_sub3(dif,newpos,oldpos), then mju_cross(dif,force),
        # then mju_sub3(tran,torque,cross).
        dx = site_pos[w, site, 0] - com[w, root, 0]
        dy = site_pos[w, site, 1] - com[w, root, 1]
        dz = site_pos[w, site, 2] - com[w, root, 2]
        fx = cfrc[w, body, 3]
        fy = cfrc[w, body, 4]
        fz = cfrc[w, body, 5]
        cx = dy*fz - dz*fy
        cy = dz*fx - dx*fz
        cz = dx*fy - dy*fx
        tx = cfrc[w, body, 0] - cx
        ty = cfrc[w, body, 1] - cy
        tz = cfrc[w, body, 2] - cz
        # mju_rotVecMatT, in original left-associative source order.
        if sensor_type[s] == 4:
            v0, v1, v2 = fx, fy, fz
        else:
            v0, v1, v2 = tx, ty, tz
        adr = sensor_adr[s]
        result[w, adr] = site_mat[w, site, 0]*v0 + site_mat[w, site, 3]*v1 + site_mat[w, site, 6]*v2
        result[w, adr+1] = site_mat[w, site, 1]*v0 + site_mat[w, site, 4]*v1 + site_mat[w, site, 7]*v2
        result[w, adr+2] = site_mat[w, site, 2]*v0 + site_mat[w, site, 5]*v1 + site_mat[w, site, 8]*v2


class ForceSensorWorkspace:
    """Preallocated fixed-MJB sensor transform; launch contains device work only."""

    def __init__(self, model, capacity, device='cuda:0'):
        if (int(model.nbody), int(model.nsite), int(model.nsensor),
            int(model.nsensordata), int(model.nplugin)) != (38, 31, 2, 6, 0):
            raise ValueError('unsupported sensor model shape')
        if capacity <= 0:
            raise ValueError('capacity')
        for name, shape in (('sensor_type',(2,)),('sensor_objtype',(2,)),
                            ('sensor_objid',(2,)),('sensor_adr',(2,)),
                            ('sensor_dim',(2,)),('sensor_needstage',(2,)),
                            ('sensor_cutoff',(2,)),('sensor_noise',(2,)),
                            ('site_bodyid',(31,)),
                            ('body_rootid',(38,))):
            if getattr(model,name).shape != shape:
                raise ValueError((name,'shape'))
        if (tuple(int(x) for x in model.sensor_type) != (4,5) or
            tuple(int(x) for x in model.sensor_objtype) != (6,6) or
            tuple(int(x) for x in model.sensor_dim) != (3,3) or
            tuple(int(x) for x in model.sensor_needstage) != (3,3) or
            tuple(int(x) for x in model.sensor_adr) != (0,3)):
            raise ValueError('unsupported sensor type/address')
        if (int(model.opt_disableflags)!=0 or int(model.opt_enableflags)!=0 or
            np.any(model.sensor_cutoff) or np.any(model.sensor_noise)):
            raise ValueError('disabled, noisy or cutoff sensor branch')
        if np.any(model.sensor_objid < 0) or np.any(model.sensor_objid >= 31):
            raise ValueError('site id')
        if np.any(model.site_bodyid < 0) or np.any(model.site_bodyid >= 38):
            raise ValueError('body id')
        if np.any(model.body_rootid < 0) or np.any(model.body_rootid >= 38):
            raise ValueError('root id')
        self.capacity, self.device = capacity, device
        self.cfrc = wp.zeros((capacity,38,6),dtype=scalar.wp_real,device=device)
        self.com = wp.zeros((capacity,38,3),dtype=scalar.wp_real,device=device)
        self.site_pos = wp.zeros((capacity,31,3),dtype=scalar.wp_real,device=device)
        self.site_mat = wp.zeros((capacity,31,9),dtype=scalar.wp_real,device=device)
        self.active = wp.zeros(capacity,dtype=wp.int32,device=device)
        self.result = wp.zeros((capacity,6),dtype=scalar.wp_real,device=device)
        self.status = wp.zeros(capacity,dtype=wp.int32,device=device)
        for attr, src in (('sensor_type',model.sensor_type),
                          ('sensor_site',model.sensor_objid),
                          ('sensor_adr',model.sensor_adr),
                          ('site_body',model.site_bodyid),
                          ('body_root',model.body_rootid)):
            setattr(self,attr,wp.array(np.ascontiguousarray(src,dtype=np.int32),
                                       dtype=wp.int32,device=device))

    def launch(self, stream=None):
        wp.launch(_sensor,dim=self.capacity,inputs=[self.cfrc,self.com,
            self.site_pos,self.site_mat,self.active,self.sensor_type,
            self.sensor_site,self.sensor_adr,self.site_body,self.body_root,
            self.result,self.status],device=self.device,stream=stream)
