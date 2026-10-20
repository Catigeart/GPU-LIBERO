"""Source-ordered full passive chain for the fixed MuJoCo 2.3.7 model.

The suffix adapts accepted N08 to retain the accepted N07 joint/DOF seed.
Apache-2.0 MuJoCo 2.3.7 source: engine_passive.c:37-155,188-255;
engine_support.c:208-260,1174-1207,1220-1267;
engine_util_spatial.c:449-477. Static box parameters are checked by an
independent original-expression C helper in scripts/gpu_fluid/box_params.c.
"""
import libero.gpu.precision as scalar
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT

import importlib.util
import hashlib
from pathlib import Path
import sys

import numpy as np
import warp as wp

_root = (REPO_ROOT / '.')
_prepare_spec = importlib.util.spec_from_file_location(
    'd06_n08_fluid_prepare', (GPU_ROOT / 'reference/dynamics/fluid/prepare.py'))
_prepare_module = importlib.util.module_from_spec(_prepare_spec)
sys.modules[_prepare_spec.name] = _prepare_module
_prepare_spec.loader.exec_module(_prepare_module)
prepare_model = _prepare_module.prepare_model
_prefix_spec = importlib.util.spec_from_file_location(
    'd06_n07_joint_kernel', (GPU_ROOT / 'reference/dynamics/joint_passive/kernel.py'))
_prefix_module = importlib.util.module_from_spec(_prefix_spec)
sys.modules[_prefix_spec.name] = _prefix_module
_prefix_spec.loader.exec_module(_prefix_module)

_ORIGINAL_DAMPING_SHA = 'f32f440b12ed7c94ae5efb779d07f2f68a88717b60d4c0959b17ad5e1999cb31'
_ORIGINAL_STIFFNESS_SHA = '81c611f35bff79491538b2f7cf201c7597a661a5c549633541c62bdc8af1613f'
_JOINT_SHA = {
    'jnt_type':'4c0e0d380d48f2a6a7f66baa70f16e9bef2974dcffe93545a1da175cd5c9f2ce',
    'jnt_qposadr':'38ae4abac73ad2b845e1908b24e91cd00b1d3df40b5c6f885075fe46adb9ab35',
    'jnt_dofadr':'13c9448fa632713f92cc44c1c525c473e12ce7105bc7b23c30d8fcdf8b90a02b',
    'qpos_spring':'e616c8b9c20a4077b81cb9da775b4048f6473c6378fce69919d9cf4d7badc9fc',
}

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
    # N07 has written the joint/DOF seed.  Never clear or independently
    # reduce the fluid suffix: each body force then torque adds to this array.
    if diagnostic != 0:
        for body in range(38):
            for k in range(334):
                stages[w, body, k] = scalar.wp_real(0.0)
    if active[w] == 0:
        return
    if status[w] != 0:
        return
    # The prefix can overflow on a finite but extreme qpos/qvel.  Keep the
    # first error on device and never treat a nonfinite seed as a result.
    for j in range(43):
        if not wp.isfinite(qfrc[w, j]):
            status[w] = 2
            return
    # The adapter contract checks all active dynamic fields even when the
    # fluid arithmetic is disabled. Inactive slots and earlier prefix errors
    # returned above without reading these payloads.
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

    # mj_passive clears qfrc before returning for mjDSBL_PASSIVE. The fixed
    # supported model has passive_disabled=0; both zero fluid options are a
    # named local model whose active input status still follows this schema.
    if passive_disabled != 0:
        return
    if opt[0] <= scalar.wp_real(0.0) and opt[1] <= scalar.wp_real(0.0):
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
        for r in range(3):
            if (not wp.isfinite(offset[r]) or not wp.isfinite(local_ang[r])
                    or not wp.isfinite(object_lin[r]) or not wp.isfinite(local_wind[r])
                    or not wp.isfinite(local_lin[r])):
                status[w] = 2
                return
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
        for r in range(3):
            if (not wp.isfinite(torque[r]) or not wp.isfinite(force[r])
                    or not wp.isfinite(world_torque[r]) or not wp.isfinite(world_force[r])):
                status[w] = 2
                return
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
            for r in range(3):
                if not wp.isfinite(jp[r]) or not wp.isfinite(jr[r]):
                    status[w] = 2
                    return
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
            if not wp.isfinite(qfrc[w, j]):
                status[w] = 2
                return
            if diagnostic != 0:
                stages[w, body, 291+j] = qfrc[w, j]


@wp.kernel(enable_backward=False)
def _poison_output(qfrc: wp.array(dtype=scalar.wp_real, ndim=2), value: scalar.wp_real):
    """Verification-only device corruption; never called by production."""
    qfrc[0, 0] = value


@wp.kernel(enable_backward=False)
def _device_inject_dynamic(cvel: wp.array(dtype=scalar.wp_real, ndim=3),
                           subtree_com: wp.array(dtype=scalar.wp_real, ndim=3),
                           xipos: wp.array(dtype=scalar.wp_real, ndim=3),
                           ximat: wp.array(dtype=scalar.wp_real, ndim=3),
                           cdof: wp.array(dtype=scalar.wp_real, ndim=3),
                           field: int, world: int, value: scalar.wp_real):
    """Verification-only producer; bypasses host finite-input preparation."""
    if wp.tid() != 0:
        return
    if field == 0:
        cvel[world, 0, 0] = value
    elif field == 1:
        subtree_com[world, 0, 0] = value
    elif field == 2:
        xipos[world, 0, 0] = value
    elif field == 3:
        ximat[world, 0, 0] = value
    else:
        cdof[world, 0, 0] = value


@wp.kernel(enable_backward=False)
def _device_set_active(active: wp.array(dtype=wp.int32), world: int, value: int):
    """Verification-only prefix-error producer."""
    if wp.tid() == 0:
        active[world] = value


@wp.kernel(enable_backward=False)
def _terminal_add_wrong(seed: wp.array(dtype=scalar.wp_real, ndim=2),
                        fluid: wp.array(dtype=scalar.wp_real, ndim=2),
                        output: wp.array(dtype=scalar.wp_real, ndim=2)):
    """Verification-only wrong implementation: fluid from zero then add."""
    w,j=wp.tid()
    output[w,j]=seed[w,j]+fluid[w,j]


class PassiveChainWorkspace:
    def __init__(self, model, capacity, device='cuda:0'):
        prepared = prepare_model(model)
        if capacity <= 0:
            raise ValueError('capacity')
        for name, shape, dtype in (
            ('jnt_type',(18,),np.int32), ('jnt_qposadr',(18,),np.int32),
            ('jnt_dofadr',(18,),np.int32), ('jnt_stiffness',(18,),np.float64),
            ('qpos_spring',(48,),np.float64), ('dof_damping',(43,),np.float64)):
            array = np.asarray(getattr(model,name))
            if array.shape != shape or array.dtype != dtype:
                raise TypeError((name, shape, dtype, array.shape, array.dtype))
            if dtype == np.float64 and not np.isfinite(array).all():
                raise ValueError(('nonfinite static',name))
        for name,digest in {**_JOINT_SHA,'dof_damping':_ORIGINAL_DAMPING_SHA}.items():
            if hashlib.sha256(np.asarray(getattr(model,name)).tobytes()).hexdigest()!=digest:
                raise ValueError(('fixed model field differs',name))
        stiff=np.asarray(model.jnt_stiffness)
        local_stiff=np.zeros(18,np.float64)
        local_stiff[0],local_stiff[7]=10.0,100.0
        if (hashlib.sha256(stiff.tobytes()).hexdigest()!=_ORIGINAL_STIFFNESS_SHA
                and not np.array_equal(stiff.view(np.uint64),local_stiff.view(np.uint64))):
            raise ValueError('unsupported scalar stiffness variant')
        if int(model.opt_disableflags)!=0:
            raise ValueError('fixed model passive flags must be zero')
        option=np.concatenate((np.asarray(model.opt_viscosity).reshape(1),
                               np.asarray(model.opt_density).reshape(1),
                               np.asarray(model.opt_wind).reshape(3)))
        allowed=((2e-5,1.2,0.0,0.0,0.0),
                 (2e-5,1.2,0.5,-0.2,0.1),
                 (0.0,1.2,0.0,0.0,0.0),
                 (2e-5,0.0,0.0,0.0,0.0),
                 (0.0,0.0,0.0,0.0,0.0))
        if not any(np.array_equal(option.view(np.uint64),
                                  np.asarray(row,np.float64).view(np.uint64)) for row in allowed):
            raise ValueError('unsupported fluid option variant')
        for i in range(18):
            if model.jnt_type[i] in (0, 1) and model.jnt_stiffness[i] != 0:
                raise ValueError('nonzero free/ball rotational spring unsupported')
            if model.jnt_type[i] not in (0, 2, 3):
                raise ValueError('unsupported joint type')
        self.capacity, self.device = capacity, device
        self.prepare_version = prepared.prepare_version
        self.model_identity_sha256 = prepared.model_identity_sha256
        self.passive_disabled = prepared.passive_disabled
        self.qpos = wp.zeros((capacity, 48), scalar.wp_real, device=device)
        self.qvel = wp.zeros((capacity, 43), scalar.wp_real, device=device)
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
            ('jtype', model.jnt_type, wp.int32),
            ('qadr', model.jnt_qposadr, wp.int32),
            ('dadr', model.jnt_dofadr, wp.int32),
            ('stiffness', model.jnt_stiffness, scalar.wp_real),
            ('spring', model.qpos_spring, scalar.wp_real),
            ('damping', model.dof_damping, scalar.wp_real),
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
        wp.launch(_prefix_module._passive, dim=self.capacity, inputs=[
            self.qpos, self.qvel, self.active, self.jtype, self.qadr,
            self.dadr, self.stiffness, self.spring, self.damping,
            self.qfrc, self.status], device=self.device, stream=stream)
        self.launch_suffix(stream=stream,diagnostic=diagnostic)

    def launch_suffix(self, stream=None, diagnostic=False):
        """Diagnostic entry with caller-provided device seed; no copy/clear."""
        if diagnostic and not hasattr(self, 'stages'):
            raise ValueError('call enable_diagnostics before diagnostic launch')
        wp.launch(_fluid, dim=self.capacity, inputs=[
            self.cvel, self.subtree_com, self.xipos, self.ximat, self.cdof,
            self.active, self.rootid, self.mass, self.box, self.ancestor,
            self.wind, self.opt, int(self.passive_disabled), int(diagnostic),
            self.stages if diagnostic else self._empty_stages,
            self.qfrc, self.status],
            device=self.device, stream=stream)
