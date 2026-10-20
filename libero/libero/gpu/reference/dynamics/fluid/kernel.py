"""Current MJB inertia-box fluid, object velocity and applyFT on CUDA.

Apache-2.0 MuJoCo 2.3.7 source: engine_passive.c:120,188-255;
engine_support.c:208-260,1174-1207,1220-1267;
engine_util_spatial.c:449-477. Static box parameters are checked by an
independent original-expression C helper in scripts/gpu_fluid/box_params.c.
"""
import libero.gpu.precision as scalar
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT

import numpy as np
import warp as wp

if __package__:
    from .prepare import prepare_model
else:
    # Standalone evidence runners load kernel.py with spec_from_file_location.
    import importlib.util
    from pathlib import Path
    import sys

    _prepare_spec = importlib.util.spec_from_file_location(
        'n08_fluid_prepare', (GPU_ROOT / 'reference/dynamics/fluid/prepare.py'))
    _prepare_module = importlib.util.module_from_spec(_prepare_spec)
    sys.modules[_prepare_spec.name] = _prepare_module
    _prepare_spec.loader.exec_module(_prepare_module)
    prepare_model = _prepare_module.prepare_model

wp.set_module_options({'enable_backward': False, 'fuse_fp': True})

PI = scalar.wp_real(3.14159265358979323846)


@wp.func
def _cross(a: scalar.vec3_real, b: scalar.vec3_real):
    return scalar.vec3_real(a[1]*b[2]-a[2]*b[1],
                    a[2]*b[0]-a[0]*b[2],
                    a[0]*b[1]-a[1]*b[0])


@wp.func
def _rot(m: wp.array(dtype=scalar.wp_real, ndim=3), w: int, body: int, v: scalar.vec3_real):
    return scalar.vec3_real(m[w, body, 0]*v[0]+m[w, body, 1]*v[1]+m[w, body, 2]*v[2],
                    m[w, body, 3]*v[0]+m[w, body, 4]*v[1]+m[w, body, 5]*v[2],
                    m[w, body, 6]*v[0]+m[w, body, 7]*v[1]+m[w, body, 8]*v[2])


@wp.func
def _rot_t(m: wp.array(dtype=scalar.wp_real, ndim=3), w: int, body: int, v: scalar.vec3_real):
    return scalar.vec3_real(m[w, body, 0]*v[0]+m[w, body, 3]*v[1]+m[w, body, 6]*v[2],
                    m[w, body, 1]*v[0]+m[w, body, 4]*v[1]+m[w, body, 7]*v[2],
                    m[w, body, 2]*v[0]+m[w, body, 5]*v[1]+m[w, body, 8]*v[2])


@wp.kernel(enable_backward=False)
def _fluid(cvel: wp.array(dtype=scalar.wp_real, ndim=3),
           subtree_com: wp.array(dtype=scalar.wp_real, ndim=3),
           xipos: wp.array(dtype=scalar.wp_real, ndim=3),
           ximat: wp.array(dtype=scalar.wp_real, ndim=3),
           cdof: wp.array(dtype=scalar.wp_real, ndim=3),
           active: wp.array(dtype=wp.int32),
           rootid: wp.array(dtype=wp.int32),
           mass: wp.array(dtype=scalar.wp_real),
           box: wp.array(dtype=scalar.wp_real, ndim=2),
           ancestor: wp.array(dtype=wp.int32, ndim=2),
           wind: wp.array(dtype=scalar.wp_real),
           opt: wp.array(dtype=scalar.wp_real),
           passive_disabled: int,
           diagnostic: int,
           stages: wp.array(dtype=scalar.wp_real, ndim=3),
           qfrc: wp.array(dtype=scalar.wp_real, ndim=2),
           status: wp.array(dtype=wp.int32)):
    w = wp.tid()
    status[w] = 0
    for j in range(43):
        qfrc[w, j] = scalar.wp_real(0.0)
    if diagnostic != 0:
        for body in range(38):
            for k in range(334):
                stages[w, body, k] = scalar.wp_real(0.0)
    if active[w] == 0:
        return
    if active[w] != 1:
        status[w] = 1
        return
    # mj_passive clears qfrc before returning for mjDSBL_PASSIVE.
    if passive_disabled != 0:
        return
    if opt[0] <= scalar.wp_real(0.0) and opt[1] <= scalar.wp_real(0.0):
        return
    # The fixed-input contract requires finite dynamic data. Checking here
    # prevents nonfinite inactive payloads from entering the fluid branch.
    for i in range(38):
        for k in range(6):
            if not wp.isfinite(cvel[w, i, k]):
                status[w] = 2
                return
        for k in range(3):
            if not wp.isfinite(subtree_com[w, i, k]) or not wp.isfinite(xipos[w, i, k]):
                status[w] = 2
                return
        for k in range(9):
            if not wp.isfinite(ximat[w, i, k]):
                status[w] = 2
                return
    for j in range(43):
        for k in range(6):
            if not wp.isfinite(cdof[w, j, k]):
                status[w] = 2
                return

    for body in range(1, 38):
        if mass[body] < scalar.wp_real(1.0e-15):
            continue
        root = rootid[body]
        offset = scalar.vec3_real(xipos[w, body, 0]-subtree_com[w, root, 0],
                          xipos[w, body, 1]-subtree_com[w, root, 1],
                          xipos[w, body, 2]-subtree_com[w, root, 2])
        ang = scalar.vec3_real(cvel[w, body, 0], cvel[w, body, 1], cvel[w, body, 2])
        lin = scalar.vec3_real(cvel[w, body, 3], cvel[w, body, 4], cvel[w, body, 5])
        # mj_objectVelocity: transformSpatial velocity from root COM to body COM.
        local_ang = _rot_t(ximat, w, body, ang)
        object_lin = _rot_t(ximat, w, body, lin-_cross(offset, ang))
        # Wind angular part is zero, so its translated linear component is wind.
        local_wind = _rot_t(ximat, w, body,
                            scalar.vec3_real(wind[0], wind[1], wind[2]))
        local_lin = object_lin - local_wind
        if diagnostic != 0:
            for r in range(3):
                stages[w, body, r] = box[body, r]
                stages[w, body, 3+r] = local_ang[r]
                stages[w, body, 6+r] = object_lin[r]
                stages[w, body, 9+r] = scalar.wp_real(0.0)
                stages[w, body, 12+r] = local_wind[r]
                stages[w, body, 15+r] = local_ang[r]
                stages[w, body, 18+r] = local_lin[r]
        x, y, z = box[body, 0], box[body, 1], box[body, 2]
        visc, density = opt[0], opt[1]
        torque = scalar.vec3_real(scalar.wp_real(0.0), scalar.wp_real(0.0), scalar.wp_real(0.0))
        force = scalar.vec3_real(scalar.wp_real(0.0), scalar.wp_real(0.0), scalar.wp_real(0.0))
        if visc > scalar.wp_real(0.0):
            diam = (x+y+z)/scalar.wp_real(3.0)
            ang_scl = -PI*diam*diam*diam*visc
            lin_scl = -scalar.wp_real(3.0)*PI*diam*visc
            torque = local_ang*ang_scl
            force = local_lin*lin_scl
        if density > scalar.wp_real(0.0):
            force[0] = force[0] - scalar.wp_real(0.5)*density*y*z*wp.abs(local_lin[0])*local_lin[0]
            force[1] = force[1] - scalar.wp_real(0.5)*density*x*z*wp.abs(local_lin[1])*local_lin[1]
            force[2] = force[2] - scalar.wp_real(0.5)*density*x*y*wp.abs(local_lin[2])*local_lin[2]
            torque[0] = torque[0] - density*x*(y*y*y*y+z*z*z*z)*wp.abs(local_ang[0])*local_ang[0]/scalar.wp_real(64.0)
            torque[1] = torque[1] - density*y*(x*x*x*x+z*z*z*z)*wp.abs(local_ang[1])*local_ang[1]/scalar.wp_real(64.0)
            torque[2] = torque[2] - density*z*(x*x*x*x+y*y*y*y)*wp.abs(local_ang[2])*local_ang[2]/scalar.wp_real(64.0)
        world_torque = _rot(ximat, w, body, torque)
        world_force = _rot(ximat, w, body, force)
        if diagnostic != 0:
            for r in range(3):
                stages[w, body, 21+r] = torque[r]
                stages[w, body, 24+r] = force[r]
                stages[w, body, 27+r] = world_torque[r]
                stages[w, body, 30+r] = world_force[r]
        # mj_applyFT: mj_jac(point=xipos,body), Jp^T force then Jr^T torque.
        for j in range(43):
            jp = scalar.vec3_real(scalar.wp_real(0.0), scalar.wp_real(0.0), scalar.wp_real(0.0))
            jr = scalar.vec3_real(scalar.wp_real(0.0), scalar.wp_real(0.0), scalar.wp_real(0.0))
            if ancestor[body, j] != 0:
                jr = scalar.vec3_real(cdof[w, j, 0], cdof[w, j, 1], cdof[w, j, 2])
                jp = scalar.vec3_real(cdof[w, j, 3], cdof[w, j, 4], cdof[w, j, 5]) + _cross(jr, offset)
            if diagnostic != 0:
                for r in range(3):
                    stages[w, body, 33+43*r+j] = jp[r]
                    stages[w, body, 162+43*r+j] = jr[r]
            qforce = scalar.wp_real(0.0)
            for r in range(3):
                if world_force[r] != scalar.wp_real(0.0):
                    qforce = qforce + jp[r]*world_force[r]
            qfrc[w, j] = qfrc[w, j] + qforce
            qtorque = scalar.wp_real(0.0)
            for r in range(3):
                if world_torque[r] != scalar.wp_real(0.0):
                    qtorque = qtorque + jr[r]*world_torque[r]
            qfrc[w, j] = qfrc[w, j] + qtorque
            if diagnostic != 0:
                stages[w, body, 291+j] = qfrc[w, j]


class FluidWorkspace:
    def __init__(self, model, capacity, device='cuda:0'):
        prepared = prepare_model(model)
        # Joint damping and spring parameters belong to another passive
        # prefix; original-model values may remain enabled here.
        if capacity <= 0:
            raise ValueError('capacity')
        self.capacity, self.device = capacity, device
        self.prepare_version = prepared.prepare_version
        self.model_identity_sha256 = prepared.model_identity_sha256
        self.passive_disabled = prepared.passive_disabled
        self.cvel = wp.zeros((capacity, 38, 6), scalar.wp_real, device=device)
        self.subtree_com = wp.zeros((capacity, 38, 3), scalar.wp_real, device=device)
        self.xipos = wp.zeros((capacity, 38, 3), scalar.wp_real, device=device)
        self.ximat = wp.zeros((capacity, 38, 9), scalar.wp_real, device=device)
        self.cdof = wp.zeros((capacity, 43, 6), scalar.wp_real, device=device)
        self.active = wp.zeros(capacity, wp.int32, device=device)
        self.qfrc = wp.zeros((capacity, 43), scalar.wp_real, device=device)
        self.status = wp.zeros(capacity, wp.int32, device=device)
        self._empty_stages = wp.zeros((1, 1, 1), scalar.wp_real, device=device)
        for name, value, dtype in (
            ('rootid', prepared.rootid, wp.int32),
            ('mass', prepared.mass, scalar.wp_real),
            ('box', prepared.box, scalar.wp_real),
            ('ancestor', prepared.ancestor, wp.int32),
            ('wind', prepared.wind, scalar.wp_real),
            ('opt', prepared.opt, scalar.wp_real),
        ):
            arr = np.ascontiguousarray(value, np.int32 if dtype == wp.int32 else np.float64)
            setattr(self, name, wp.array(arr, dtype=dtype, device=device))

    def enable_diagnostics(self):
        """Allocate optional stage storage before a diagnostic launch/capture."""
        self.stages = wp.zeros((self.capacity, 38, 334), scalar.wp_real,
                               device=self.device)

    def launch(self, stream=None, diagnostic=False):
        if diagnostic and not hasattr(self, 'stages'):
            raise ValueError('call enable_diagnostics before diagnostic launch')
        wp.launch(_fluid, dim=self.capacity, inputs=[
            self.cvel, self.subtree_com, self.xipos, self.ximat, self.cdof,
            self.active, self.rootid, self.mass, self.box, self.ancestor,
            self.wind, self.opt, int(self.passive_disabled), int(diagnostic),
            self.stages if diagnostic else self._empty_stages,
            self.qfrc, self.status],
            device=self.device, stream=stream)
