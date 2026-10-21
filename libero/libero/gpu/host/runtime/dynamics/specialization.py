"""New count-parametric D candidates with exact native model readsets.

The fixed constructors and licences remain unchanged. These kernels retain
MuJoCo 2.3.7 ordered Jacobian, packed-M and four-lane dot arithmetic. Diagnostic
source readiness never grants numerical acceptance for a new model.
"""
from __future__ import annotations
import libero.gpu.precision as scalar
import hashlib
import json
from pathlib import Path
import numpy as np
import warp as wp
from ..core.model_bytes import checked_records, source_ref, source_bytes
from .runtime_adapter import _token, _array

wp.set_module_options({'enable_backward': False, 'fuse_fp': True, 'fast_math': False})


def native_fields(arrays, metadata, spec, fields, *, source_records, metadata_record,
                  registry, candidate_id, mode):
    """Cold native bytes -> readonly participating fields, independently pinned."""
    from libero.gpu.host.runtime.foundation.runtime_adapter import _native_field_digest, _pin_native_metadata
    records = checked_records(source_records)
    if not records:
        raise ValueError('native/source records required')
    _pin_native_metadata(metadata, spec, records, metadata_record, fields)
    digests = {digest for _, digest in records}
    if (metadata['model_sha256'] != spec.model_sha256 or
        metadata['static_arrays_sha256'] != spec.static_arrays_sha256 or
        metadata['decoder_pin']['source_sha256'] not in digests or
        spec.model_sha256 not in digests):
        raise ValueError('actual D native identity/decoder source mismatch')
    for name, value in spec.dimensions.as_dict().items():
        if metadata['dimensions'].get(name) != value:
            raise ValueError('actual native dimensions differ from model spec: ' + name)
    admission = registry.admit(candidate_id, spec, mode=mode,
        required_scope='component-runtime' if mode == 'production' else None)
    candidate = registry.candidates[candidate_id]
    covered_source = (set(candidate.source_files.items()).issubset(set(records))
                      if candidate.source_files else candidate.source_sha256 in digests)
    if not admission.execution_allowed or not covered_source:
        raise ValueError('D source/model candidate admission refused')
    result = {}
    for name in fields:
        a = np.array(arrays[name], copy=True, order='C')
        if metadata['array_sha256'].get(name) != _native_field_digest(name, a):
            raise ValueError('D native readset field changed: ' + name)
        a.flags.writeable = False
        result[name] = a
    return result, admission


def validate_d_topology(arrays, dims):
    """Native joint/body/packed mass topology without fixed numerical counts."""
    nq, nv, nb, nm, nj = (int(dims[k]) for k in ('nq', 'nv', 'nbody', 'nM', 'njnt'))
    if min(nq, nv, nb, nm, nj) <= 0:
        raise ValueError('nonempty ordered rigid-body model required')
    expected = {'body_parentid': nb, 'body_rootid': nb, 'dof_bodyid': nv,
        'dof_parentid': nv, 'dof_Madr': nv, 'dof_simplenum': nv,
        'jnt_type': nj, 'jnt_qposadr': nj, 'jnt_dofadr': nj}
    for name, count in expected.items():
        a = arrays[name]
        if a.dtype != np.int32 or a.shape != (count,):
            raise ValueError('native topology schema: ' + name)
    bp, roots = arrays['body_parentid'], arrays['body_rootid']
    if bp[0] != 0 or roots[0] != 0 or np.any(bp[1:] < 0) or np.any(bp[1:] >= np.arange(1, nb)):
        raise ValueError('body ordering')
    for i in range(1, nb):
        if roots[i] != (i if bp[i] == 0 else roots[bp[i]]):
            raise ValueError('body root lineage')
    dp, ma = arrays['dof_parentid'], arrays['dof_Madr']
    address = 0
    for i in range(nv):
        if dp[i] < -1 or dp[i] >= i or arrays['dof_bodyid'][i] <= 0 or arrays['dof_bodyid'][i] >= nb:
            raise ValueError('dof ordering/body domain')
        if ma[i] != address or arrays['dof_simplenum'][i] < 0 or arrays['dof_simplenum'][i] > nv-i:
            raise ValueError('packed mass/simple range')
        address += 1
        j = int(dp[i])
        while j >= 0:
            address += 1
            j = int(dp[j])
    if address != nm:
        raise ValueError('packed mass final size')
    pa, va = 0, 0
    for typ, qa, da in zip(arrays['jnt_type'], arrays['jnt_qposadr'], arrays['jnt_dofadr']):
        if typ not in (0, 1, 2, 3) or qa != pa or da != va:
            raise ValueError('ordered joint coverage')
        pa += 7 if typ == 0 else 4 if typ == 1 else 1
        va += 6 if typ == 0 else 3 if typ == 1 else 1
    if pa != nq or va != nv:
        raise ValueError('joint state coverage')
    # Exact ancestor incidence, a structural constant, no numerical physics.
    ancestors = np.zeros((nb, nv), dtype=np.int32)
    for body in range(1, nb):
        lineage = set()
        b = body
        while b != 0:
            lineage.add(b)
            b = int(bp[b])
        for k in range(nv):
            ancestors[body, k] = int(int(arrays['dof_bodyid'][k]) in lineage)
    return ancestors


@wp.kernel(enable_backward=False)
def full_site_jacobian(cdof: wp.array(dtype=scalar.wp_real, ndim=3),
                      subtree: wp.array(dtype=scalar.wp_real, ndim=3),
                      site: wp.array(dtype=scalar.wp_real, ndim=2),
                      active: wp.array(dtype=wp.int32),
                      ancestor: wp.array(dtype=wp.int32, ndim=2), root: int, body: int,
                      nv: int, jp: wp.array(dtype=scalar.wp_real, ndim=3),
                      jr: wp.array(dtype=scalar.wp_real, ndim=3), status: wp.array(dtype=wp.int32)):
    w = wp.tid()
    status[w] = 0
    for row in range(3):
        for k in range(nv):
            jp[w, row, k] = scalar.wp_real(0.0)
            jr[w, row, k] = scalar.wp_real(0.0)
    if active[w] == 0:
        return
    if active[w] != 1:
        status[w] = 4
        return
    ox = site[w, 0] - subtree[w, root, 0]
    oy = site[w, 1] - subtree[w, root, 1]
    oz = site[w, 2] - subtree[w, root, 2]
    if not wp.isfinite(ox) or not wp.isfinite(oy) or not wp.isfinite(oz):
        status[w] = 5
        return
    for k in range(nv):
        if ancestor[body, k] == 1:
            ax = cdof[w, k, 0]
            ay = cdof[w, k, 1]
            az = cdof[w, k, 2]
            lx = cdof[w, k, 3]
            ly = cdof[w, k, 4]
            lz = cdof[w, k, 5]
            vx = lx + (ay * oz - az * oy)
            vy = ly + (az * ox - ax * oz)
            vz = lz + (ax * oy - ay * ox)
            if (not wp.isfinite(ax) or not wp.isfinite(ay) or not wp.isfinite(az) or
                not wp.isfinite(lx) or not wp.isfinite(ly) or not wp.isfinite(lz) or
                not wp.isfinite(vx) or not wp.isfinite(vy) or not wp.isfinite(vz)):
                status[w] = 5
                break
            jr[w, 0, k] = ax
            jr[w, 1, k] = ay
            jr[w, 2, k] = az
            jp[w, 0, k] = vx
            jp[w, 1, k] = vy
            jp[w, 2, k] = vz
    if status[w] != 0:
        for row in range(3):
            for k in range(nv):
                jp[w, row, k] = scalar.wp_real(0.0)
                jr[w, row, k] = scalar.wp_real(0.0)


@wp.func
def dot_row4(a: wp.array(dtype=scalar.wp_real, ndim=3),
             b: wp.array(dtype=scalar.wp_real, ndim=2), w: int, row: int, nv: int):
    s0 = scalar.wp_real(0.0)
    s1 = scalar.wp_real(0.0)
    s2 = scalar.wp_real(0.0)
    s3 = scalar.wp_real(0.0)
    for k in range(nv // 4):
        j = 4 * k
        s0 = s0 + a[w, row, j] * b[w, j]
        s1 = s1 + a[w, row, j + 1] * b[w, j + 1]
        s2 = s2 + a[w, row, j + 2] * b[w, j + 2]
        s3 = s3 + a[w, row, j + 3] * b[w, j + 3]
    result = (s0 + s2) + (s1 + s3)
    start = 4 * (nv // 4)
    if start < nv:
        tail = a[w, row, start] * b[w, start]
        for j in range(start + 1, nv):
            tail = tail + a[w, row, j] * b[w, j]
        result = result + tail
    return result


@wp.kernel(enable_backward=False)
def scalar_transmission_position(qpos: wp.array(dtype=scalar.wp_real, ndim=2),
                                active: wp.array(dtype=wp.int32),
                                qadr: wp.array(dtype=wp.int32), dadr: wp.array(dtype=wp.int32),
                                gear: wp.array(dtype=scalar.wp_real), nq: int, nv: int, nu: int,
                                length: wp.array(dtype=scalar.wp_real, ndim=2),
                                moment: wp.array(dtype=scalar.wp_real, ndim=3),
                                status: wp.array(dtype=wp.int32)):
    w = wp.tid()
    status[w] = 0
    for a in range(nu):
        length[w, a] = scalar.wp_real(0.0)
        for k in range(nv):
            moment[w, a, k] = scalar.wp_real(0.0)
    if active[w] == 0:
        return
    if active[w] != 1:
        status[w] = 1
        return
    for k in range(nq):
        if not wp.isfinite(qpos[w, k]):
            status[w] = 2
            return
    for a in range(nu):
        value = qpos[w, qadr[a]] * gear[a]
        if not wp.isfinite(value):
            status[w] = 4
            return
        length[w, a] = value
        moment[w, a, dadr[a]] = gear[a]


@wp.kernel(enable_backward=False)
def scalar_transmission_velocity(moment: wp.array(dtype=scalar.wp_real, ndim=3),
                                qvel: wp.array(dtype=scalar.wp_real, ndim=2),
                                active: wp.array(dtype=wp.int32),
                                position_status: wp.array(dtype=wp.int32), nv: int, nu: int,
                                velocity: wp.array(dtype=scalar.wp_real, ndim=2),
                                status: wp.array(dtype=wp.int32)):
    w = wp.tid()
    status[w] = 0
    for a in range(nu):
        velocity[w, a] = scalar.wp_real(0.0)
    if active[w] == 0:
        return
    if active[w] != 1:
        status[w] = 1
        return
    if position_status[w] != 0:
        status[w] = 3
        return
    for k in range(nv):
        if not wp.isfinite(qvel[w, k]):
            status[w] = 2
            return
    for a in range(nu):
        value = dot_row4(moment, qvel, w, a, nv)
        if not wp.isfinite(value):
            status[w] = 4
            return
        velocity[w, a] = value


class GenericTransmissionWorkspace:
    """New actual-count scalar-joint transmission, cold source proof required."""
    FIELDS = ('jnt_type', 'jnt_qposadr', 'jnt_dofadr', 'actuator_trntype',
              'actuator_trnid', 'actuator_gear')

    def __init__(self, arrays, metadata, spec, capacity, *, registry, candidate_id,
                 source_records, metadata_record, mode='production', device='cuda:0', inputs=None):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError('positive actual capacity required')
        self.static, self.admission = native_fields(arrays, metadata, spec, self.FIELDS,
            source_records=source_records, metadata_record=metadata_record,
            registry=registry, candidate_id=candidate_id, mode=mode)
        self.spec, self.registry, self.candidate_id, self.mode = spec, registry, candidate_id, mode
        dims = metadata['dimensions']
        self.nq, self.nv, self.nu, nj = (int(dims[k]) for k in ('nq', 'nv', 'nu', 'njnt'))
        for name, shape, dtype in (('jnt_type', (nj,), np.int32), ('jnt_qposadr', (nj,), np.int32),
            ('jnt_dofadr', (nj,), np.int32), ('actuator_trntype', (self.nu,), np.int32),
            ('actuator_trnid', (self.nu, 2), np.int32), ('actuator_gear', (self.nu, 6), np.float64)):
            a = self.static[name]
            if a.shape != shape or a.dtype != dtype or (dtype == np.float64 and not np.isfinite(a).all()):
                raise ValueError('actual scalar transmission schema: ' + name)
        ids = self.static['actuator_trnid'][:, 0]
        if (np.any(self.static['actuator_trntype'] != 0) or np.any(ids < 0) or np.any(ids >= nj) or
            np.any(~np.isin(self.static['jnt_type'][ids], [2, 3])) or
            np.any(self.static['actuator_gear'][:, 1:] != 0)):
            raise ValueError('only scalar-joint transmission family is implemented')
        qa, da = self.static['jnt_qposadr'][ids], self.static['jnt_dofadr'][ids]
        if np.any(qa < 0) or np.any(qa >= self.nq) or np.any(da < 0) or np.any(da >= self.nv):
            raise ValueError('scalar transmission address domain')
        self.capacity, self.device = capacity, wp.get_device(device)
        self.qadr = wp.array(qa, dtype=wp.int32, device=self.device)
        self.dadr = wp.array(da, dtype=wp.int32, device=self.device)
        self.gear = wp.array(self.static['actuator_gear'][:, 0].copy(), dtype=scalar.wp_real, device=self.device)
        inputs={} if inputs is None else inputs
        for name, shape, dtype in (('qpos', (capacity, self.nq), scalar.wp_real),
            ('qvel', (capacity, self.nv), scalar.wp_real), ('length', (capacity, self.nu), scalar.wp_real),
            ('moment', (capacity, self.nu, self.nv), scalar.wp_real), ('velocity', (capacity, self.nu), scalar.wp_real),
            ('position_active', (capacity,), wp.int32), ('velocity_active', (capacity,), wp.int32),
            ('position_status', (capacity,), wp.int32), ('velocity_status', (capacity,), wp.int32)):
            value=inputs.get(name)
            if value is None: value=wp.zeros(shape,dtype=dtype,device=self.device)
            _array(value,shape,dtype,self.device)
            setattr(self,name,value)
        self._frozen = {k: _token(v) for k, v in vars(self).items() if isinstance(v, wp.array)}

    def validate_bindings(self):
        if not self.registry.admit(self.candidate_id, self.spec, mode=self.mode,
                required_scope='component-runtime' if self.mode == 'production' else None).execution_allowed:
            raise ValueError('generic transmission admission changed')
        for k, token in self._frozen.items():
            if _token(getattr(self, k)) != token:
                raise ValueError('generic transmission binding changed: ' + k)

    def launch_position(self, stream=None):
        self.validate_bindings()
        wp.launch(scalar_transmission_position, dim=self.capacity, inputs=[self.qpos,
            self.position_active, self.qadr, self.dadr, self.gear, self.nq, self.nv, self.nu,
            self.length, self.moment, self.position_status], device=self.device, stream=stream)

    def launch_velocity(self, stream=None):
        self.validate_bindings()
        wp.launch(scalar_transmission_velocity, dim=self.capacity, inputs=[self.moment,
            self.qvel, self.velocity_active, self.position_status, self.nv, self.nu,
            self.velocity, self.velocity_status], device=self.device, stream=stream)


@wp.kernel(enable_backward=False)
def selected_osc_inputs(qpos: wp.array(dtype=scalar.wp_real, ndim=2),
    qvel: wp.array(dtype=scalar.wp_real, ndim=2), pos: wp.array(dtype=scalar.wp_real, ndim=2),
    rot: wp.array(dtype=scalar.wp_real, ndim=2), jp: wp.array(dtype=scalar.wp_real, ndim=3),
    jr: wp.array(dtype=scalar.wp_real, ndim=3), qm: wp.array(dtype=scalar.wp_real, ndim=2),
    qbias: wp.array(dtype=scalar.wp_real, ndim=2), active: wp.array(dtype=wp.int32),
    qi: wp.array(dtype=wp.int32), vi: wp.array(dtype=wp.int32),
    sparse: wp.array(dtype=wp.int32, ndim=2), nq: int, nv: int, nm: int,
    ee_pos: wp.array(dtype=scalar.wp_real, ndim=2), ee_mat: wp.array(dtype=scalar.wp_real, ndim=2),
    vp: wp.array(dtype=scalar.wp_real, ndim=2), vo: wp.array(dtype=scalar.wp_real, ndim=2),
    q: wp.array(dtype=scalar.wp_real, ndim=2), qd: wp.array(dtype=scalar.wp_real, ndim=2),
    J: wp.array(dtype=scalar.wp_real, ndim=3), M: wp.array(dtype=scalar.wp_real, ndim=3),
    bias: wp.array(dtype=scalar.wp_real, ndim=2), status: wp.array(dtype=wp.int32)):
    w = wp.tid()
    status[w] = 0
    for k in range(3):
        ee_pos[w, k] = scalar.wp_real(0.0)
        vp[w, k] = scalar.wp_real(0.0)
        vo[w, k] = scalar.wp_real(0.0)
    for k in range(9):
        ee_mat[w, k] = scalar.wp_real(0.0)
    for k in range(7):
        q[w, k] = scalar.wp_real(0.0)
        qd[w, k] = scalar.wp_real(0.0)
        bias[w, k] = scalar.wp_real(0.0)
        for r in range(6):
            J[w, r, k] = scalar.wp_real(0.0)
        for j in range(7):
            M[w, k, j] = scalar.wp_real(0.0)
    if active[w] == 0:
        return
    if active[w] != 1:
        status[w] = 1
        return
    for k in range(nq):
        if not wp.isfinite(qpos[w, k]):
            status[w] = 2
            return
    for k in range(nv):
        if not wp.isfinite(qvel[w, k]) or not wp.isfinite(qbias[w, k]):
            status[w] = 2
            return
        for r in range(3):
            if not wp.isfinite(jp[w, r, k]) or not wp.isfinite(jr[w, r, k]):
                status[w] = 2
                return
    for k in range(nm):
        if not wp.isfinite(qm[w, k]):
            status[w] = 2
            return
    for k in range(3):
        if not wp.isfinite(pos[w, k]):
            status[w] = 2
            return
    for k in range(9):
        if not wp.isfinite(rot[w, k]):
            status[w] = 2
            return
    for k in range(3):
        ee_pos[w, k] = pos[w, k]
    for k in range(9):
        ee_mat[w, k] = rot[w, k]
    # Original N18 is ordered scalar accumulation, not transmission's dot4.
    for r in range(3):
        a = scalar.wp_real(0.0)
        b = scalar.wp_real(0.0)
        for k in range(nv):
            a = a + jp[w, r, k] * qvel[w, k]
            b = b + jr[w, r, k] * qvel[w, k]
        vp[w, r] = a
        vo[w, r] = b
    for i in range(7):
        q[w, i] = qpos[w, qi[i]]
        qd[w, i] = qvel[w, vi[i]]
        bias[w, i] = qbias[w, vi[i]]
        for r in range(3):
            J[w, r, i] = jp[w, r, vi[i]]
            J[w, r + 3, i] = jr[w, r, vi[i]]
        for j in range(7):
            address = sparse[vi[i], vi[j]]
            if address >= 0:
                M[w, i, j] = qm[w, address]


class GenericJacobianInputsWorkspace:
    """Full native-count J -> seven-arm OSC gather with explicit selectors."""
    FIELDS = ('body_parentid', 'body_rootid', 'dof_bodyid', 'dof_parentid',
        'dof_Madr', 'dof_simplenum', 'jnt_type', 'jnt_qposadr', 'jnt_dofadr', 'site_bodyid')

    def __init__(self, arrays, metadata, spec, capacity, *, robot, registry, candidate_id,
                 source_records, metadata_record, mode='production', device='cuda:0', inputs=None):
        self.static, self.admission = native_fields(arrays, metadata, spec, self.FIELDS,
            source_records=source_records, metadata_record=metadata_record,
            registry=registry, candidate_id=candidate_id, mode=mode)
        # Robot selectors must be in the original independently pinned decoder
        # report; matching nv does not license an arbitrary 0:7 arm selection.
        original = json.loads(source_bytes(source_ref(metadata_record)))
        if robot not in original.get('robot_metadata', ()):
            raise ValueError('robot selectors absent from original metadata source')
        d = metadata['dimensions']
        self.nq, self.nv, self.nM, self.nbody, self.nsite = (int(d[k]) for k in ('nq', 'nv', 'nM', 'nbody', 'nsite'))
        ancestor = validate_d_topology(self.static, d)
        site = int(robot['eef_site_id'])
        joints = np.asarray(robot['joint_ids'])
        if joints.shape != (7,) or joints.dtype.kind not in 'iu' or np.any(joints < 0) or np.any(joints >= d['njnt']):
            raise ValueError('native seven-arm joint selectors')
        if self.static['site_bodyid'].dtype != np.int32 or self.static['site_bodyid'].shape != (self.nsite,) or not 0 <= site < self.nsite:
            raise ValueError('native eef site selector')
        if np.any(~np.isin(self.static['jnt_type'][joints], [2, 3])):
            raise ValueError('seven scalar arm joints required')
        qi, vi = self.static['jnt_qposadr'][joints], self.static['jnt_dofadr'][joints]
        if list(qi) != robot['joint_qpos_indices'] or list(vi) != robot['joint_qvel_indices']:
            raise ValueError('robot address metadata differs from native joints')
        self.body = int(self.static['site_bodyid'][site])
        if self.body <= 0 or self.body >= self.nbody:
            raise ValueError('eef site body domain')
        self.root = int(self.static['body_rootid'][self.body])
        self.capacity, self.device = capacity, wp.get_device(device)
        if type(capacity) is not int or capacity <= 0:
            raise ValueError('positive actual capacity required')
        self.spec, self.registry, self.candidate_id, self.mode = spec, registry, candidate_id, mode
        self.ancestor = wp.array(ancestor, dtype=wp.int32, device=self.device)
        self.qi = wp.array(qi, dtype=wp.int32, device=self.device)
        self.vi = wp.array(vi, dtype=wp.int32, device=self.device)
        sparse = np.full((self.nv, self.nv), -1, dtype=np.int32)
        for i in range(self.nv):
            j, address = i, int(self.static['dof_Madr'][i])
            while j >= 0:
                sparse[i, j] = sparse[j, i] = address
                j, address = int(self.static['dof_parentid'][j]), address + 1
        self.sparse = wp.array(sparse, dtype=wp.int32, device=self.device)
        inputs = {} if inputs is None else inputs
        schema = {'qpos': ((capacity, self.nq), scalar.wp_real), 'qvel': ((capacity, self.nv), scalar.wp_real),
            'qM': ((capacity, self.nM), scalar.wp_real), 'qfrc_bias': ((capacity, self.nv), scalar.wp_real),
            'cdof': ((capacity, self.nv, 6), scalar.wp_real), 'subtree_com': ((capacity, self.nbody, 3), scalar.wp_real),
            'site_pos': ((capacity, 3), scalar.wp_real), 'site_mat': ((capacity, 9), scalar.wp_real),
            'active': ((capacity,), wp.int32)}
        if set(inputs) - set(schema):
            raise ValueError('undeclared generic Jacobian input')
        for name, (shape, dtype) in schema.items():
            a = inputs[name] if name in inputs else wp.zeros(shape, dtype=dtype, device=self.device)
            setattr(self, name, _array(a, shape, dtype, self.device))
        for name, tail in (('jacp', (3, self.nv)), ('jacr', (3, self.nv)), ('ee_pos', (3,)),
            ('ee_mat', (9,)), ('xvelp', (3,)), ('xvelr', (3,)), ('joint_pos', (7,)), ('joint_vel', (7,)),
            ('J', (6, 7)), ('M', (7, 7)), ('bias', (7,))):
            setattr(self, name, wp.zeros((capacity,) + tail, dtype=scalar.wp_real, device=self.device))
        self.jac_status = wp.zeros(capacity, dtype=wp.int32, device=self.device)
        self.adapter_status = wp.zeros(capacity, dtype=wp.int32, device=self.device)
        self.adapter_active = wp.zeros(capacity, dtype=wp.int32, device=self.device)
        self.status = wp.zeros(capacity, dtype=wp.int32, device=self.device)
        self._frozen = {k: _token(v) for k, v in vars(self).items() if isinstance(v, wp.array)}

    def validate_bindings(self):
        if not self.registry.admit(self.candidate_id, self.spec, mode=self.mode,
                required_scope='component-runtime' if self.mode == 'production' else None).execution_allowed:
            raise ValueError('generic Jacobian admission changed')
        for k, token in self._frozen.items():
            if _token(getattr(self, k)) != token:
                raise ValueError('generic Jacobian binding changed: ' + k)

    def launch(self, stream=None):
        self.validate_bindings()
        wp.launch(full_site_jacobian, dim=self.capacity, inputs=[self.cdof, self.subtree_com,
            self.site_pos, self.active, self.ancestor, self.root, self.body, self.nv,
            self.jacp, self.jacr, self.jac_status], device=self.device, stream=stream)
        wp.launch(_jacobian_route, dim=self.capacity, inputs=[self.active, self.jac_status,
            self.adapter_active], device=self.device, stream=stream)
        wp.launch(selected_osc_inputs, dim=self.capacity, inputs=[self.qpos, self.qvel,
            self.site_pos, self.site_mat, self.jacp, self.jacr, self.qM, self.qfrc_bias,
            self.adapter_active, self.qi, self.vi, self.sparse, self.nq, self.nv, self.nM, self.ee_pos,
            self.ee_mat, self.xvelp, self.xvelr, self.joint_pos, self.joint_vel, self.J,
            self.M, self.bias, self.adapter_status], device=self.device, stream=stream)
        wp.launch(_jacobian_status, dim=self.capacity, inputs=[self.jac_status,
            self.adapter_status, self.status], device=self.device, stream=stream)


@wp.kernel(enable_backward=False)
def _jacobian_route(active: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32),
                    route: wp.array(dtype=wp.int32)):
    w = wp.tid()
    route[w] = 0
    if active[w] == 1 and status[w] == 0:
        route[w] = 1


@wp.kernel(enable_backward=False)
def _jacobian_status(producer: wp.array(dtype=wp.int32), adapter: wp.array(dtype=wp.int32),
                     status: wp.array(dtype=wp.int32)):
    w = wp.tid()
    status[w] = adapter[w]
    if producer[w] != 0:
        status[w] = 10 + producer[w]


class GenericPositionWorkspace:
    """Actual native K/Frames -> compos -> CRB -> factor, one resident graph.

    The accepted fixed BodyMassChain constructor remains untouched. This new
    candidate composes the original count-parametric arithmetic with explicit
    new static/model admission. No CPU Cartesian or qM data enters launch.
    """
    MASS_FIELDS = ('body_parentid', 'body_rootid', 'body_mass', 'body_subtreemass',
        'body_inertia', 'jnt_type', 'jnt_bodyid', 'jnt_qposadr', 'jnt_dofadr',
        'dof_bodyid', 'dof_parentid', 'dof_Madr', 'dof_simplenum', 'dof_M0', 'dof_armature')

    def __init__(self, arrays, metadata, spec, capacity, *, registry, candidate_ids,
                 source_records, metadata_record, mode='production', device='cuda:0'):
        from types import SimpleNamespace
        from libero.gpu.host.runtime.foundation.runtime_adapter import PreparedKinematicModel, PreparedFrameWorkspace
        from libero.gpu.reference.position.body_kinematics.workspace import BodyKinematicsWorkspace
        from libero.gpu.reference.position.compos.kernel import ComposModel
        from libero.gpu.reference.position.crb.kernel import CRBModel
        from libero.gpu.reference.integration.euler_integrator.generic_workspace import GenericMassWorkspace
        # Factor/compos/CRB are true source dependencies of this new candidate.
        # Production acceptance must identify their complete source bundle,
        # not only this allocation wrapper's SHA.
        if not registry.candidates[candidate_ids['mass']].source_files:
            raise ValueError('generic position needs complete original math source bundle')
        if type(capacity) is not int or capacity <= 0:
            raise ValueError('positive actual capacity required')
        fields, self.admission = native_fields(arrays, metadata, spec, self.MASS_FIELDS,
            source_records=source_records, metadata_record=metadata_record, registry=registry,
            candidate_id=candidate_ids['mass'], mode=mode)
        d = metadata['dimensions']
        validate_d_topology(fields, d)
        self.capacity, self.device = capacity, wp.get_device(device)
        self.spec, self.registry, self.candidate_ids, self.mode = spec, registry, dict(candidate_ids), mode
        frozen = PreparedKinematicModel(arrays, metadata, spec, registry=registry,
            candidate_id=candidate_ids['kinematics'], source_records=source_records,
            metadata_record=metadata_record, mode=mode)
        self.body = BodyKinematicsWorkspace(frozen, capacity, device=self.device)
        nb, nj, nv, nm = (int(d[k]) for k in ('nbody', 'njnt', 'nv', 'nM'))
        self.frame_active = wp.zeros(capacity, dtype=wp.int32, device=self.device)
        self.frames = PreparedFrameWorkspace(arrays, metadata, spec, capacity,
            registry=registry, candidate_id=candidate_ids['frames'], source_records=source_records,
            metadata_record=metadata_record, mode=mode, device=self.device,
            body_inputs={'body_xpos': self.body.xpos.reshape((capacity*nb*3,)),
                'body_xquat': self.body.xquat.reshape((capacity*nb*4,)),
                'body_xmat': self.body.xmat.reshape((capacity*nb*9,)), 'active_world': self.frame_active})
        self.model = SimpleNamespace(**d, **fields)
        self.compos = ComposModel.prepare(self.model, device=self.device)
        self.crb_model = CRBModel.prepare(self.model, device=self.device)
        self.factor = GenericMassWorkspace(metadata, fields, capacity, self.device)
        self.xipos = self.frames.xipos.reshape((capacity, nb*3))
        self.ximat = self.frames.ximat.reshape((capacity, nb*9))
        self.crb_active = wp.zeros(capacity, dtype=wp.int32, device=self.device)
        self.mass_active = wp.zeros(capacity, dtype=wp.int32, device=self.device)
        for name, width in (('subtree_com', nb*3), ('cinert', nb*10), ('cdof', nv*6), ('crb', nb*10)):
            setattr(self, name, wp.zeros((capacity, width), dtype=scalar.wp_real, device=self.device))
        for name in ('compos_status', 'crb_status', 'first_stage', 'first_code', 'result_kind',
                     'mass_first_stage', 'mass_first_code', 'mass_result_kind'):
            setattr(self, name, wp.zeros(capacity, dtype=wp.int32, device=self.device))
        self.factor.count.fill_(1)
        self._objects = (self.body, self.frames, self.compos, self.crb_model, self.factor)
        self._fields = fields
        self._static_digests = {k: hashlib.sha256(a.tobytes()).hexdigest() for k, a in fields.items()}
        self._frozen = self._tokens()

    @property
    def qpos(self):
        return self.body.qpos

    @property
    def active(self):
        return self.body.active

    @property
    def qM(self):
        return self.factor.qm

    def _tokens(self):
        out = {}
        for path, obj in (('', self), ('body', self.body), ('body.data', self.body.data),
            ('body.static', self.body.static), ('frames', self.frames), ('compos', self.compos),
            ('crb_model', self.crb_model), ('factor', self.factor)):
            for name in dir(obj):
                if name.startswith('_'):
                    continue
                value = getattr(obj, name)
                if isinstance(value, wp.array):
                    out[path+'.'+name] = _token(value)
        for name, a in self.frames.static.items():
            out['frames.static.'+name] = _token(a)
        return out

    def validate_bindings(self):
        if (self.body, self.frames, self.compos, self.crb_model, self.factor) != self._objects:
            raise ValueError('generic position prepared owner changed')
        self.body.model.validate()
        self.frames.validate_bindings()
        if self._tokens() != self._frozen:
            raise ValueError('generic position actual source/consumer binding changed')
        for k, a in self._fields.items():
            if hashlib.sha256(a.tobytes()).hexdigest() != self._static_digests[k]:
                raise ValueError('generic mass native field changed: '+k)
        if not self.registry.admit(self.candidate_ids['mass'], self.spec, mode=self.mode,
                required_scope='component-runtime' if self.mode == 'production' else None).execution_allowed:
            raise ValueError('generic position mass admission changed')

    def launch(self, stream=None):
        from libero.gpu.reference.position.body_mass_chain.chain import _after_body, _check_frames, _after_frames, _finish as body_finish
        from libero.gpu.reference.position.mass_chain.kernel import _gate1, _gate2, _finish
        from libero.gpu.reference.position.compos.kernel import launch as compos_launch
        from libero.gpu.reference.position.crb.kernel import launch as crb_launch
        self.validate_bindings()
        n = self.capacity
        self.body.launch(stream=stream)
        wp.launch(_after_body, dim=n, inputs=[self.active, self.body.status, self.frame_active],
                  device=self.device, stream=stream)
        self.frames.launch(stream=stream)
        f = self.frames
        wp.launch(_check_frames, dim=n, inputs=[self.frame_active, f.status, f.xipos, f.ximat,
            f.geom_xpos, f.geom_xmat, f.site_xpos, f.site_xmat, f.body_tmpquat,
            f.geom_tmpquat, f.site_tmpquat, f.nbody, f.ngeom, f.nsite], device=self.device, stream=stream)
        wp.launch(_after_frames, dim=n, inputs=[self.frame_active, f.status, self.mass_active],
                  device=self.device, stream=stream)
        compos_launch(self.compos, self.mass_active, self.xipos, self.ximat,
            self.body.xanchor, self.body.xmat, self.body.xaxis, self.subtree_com,
            self.cinert, self.cdof, self.compos_status, stream=stream)
        wp.launch(_gate1, dim=n, inputs=[self.mass_active, self.compos_status,
            self.crb_active, self.mass_first_stage, self.mass_first_code], device=self.device, stream=stream)
        crb_launch(self.crb_model, self.crb_active, self.cinert, self.cdof,
            self.crb, self.factor.qm, self.crb_status, stream=stream)
        wp.launch(_gate2, dim=n, inputs=[self.crb_active, self.crb_status, self.factor.active,
            self.mass_first_stage, self.mass_first_code], device=self.device, stream=stream)
        self.factor.launch(stream=stream)
        wp.launch(_finish, dim=n, inputs=[self.mass_active, self.factor.active, self.factor.status,
            self.mass_first_stage, self.mass_first_code, self.mass_result_kind], device=self.device, stream=stream)
        wp.launch(body_finish, dim=n, inputs=[self.body.active, self.body.status, self.frames.status,
            self.mass_active, self.mass_first_stage, self.mass_first_code, self.mass_result_kind,
            self.first_stage, self.first_code, self.result_kind], device=self.device, stream=stream)
