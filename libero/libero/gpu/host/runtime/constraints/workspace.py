"""PositionReferenceWorkspace: composed position + reference production chain.

Reuses the accepted phase-05 producers as kernels on shared resident buffers:

* constraint_scalar.ScalarWorkspace  -- S02 friction/limit segments,
* constraint_contact.ContactWorkspace -- S01 contact segment,
* constraint_impedance.ImpedanceWorkspace -- S03 R/D/KBIP/postdiag/mu.

The assembler (this package) owns the global dense-43 commit, prediag,
reference and the sticky error/capacity protocol.  Segment buffers are
scratch; all source stages validate before ``_commit_position`` publishes
any public position field for a world.  ``launch_position`` and ``launch_reference`` are separate
launch entries: position never reads qvel or writes aref, reference never
rewrites rows.

Aliasing note: S02/S03 workspaces store flat 1-D buffers and S01 stores shaped
arrays; the integrated arrays are laid out so each shared buffer has exactly
one owner and later stages read producer scratch on device. The final commit
copies validated written prefixes into separate public buffers on device.
"""
from __future__ import annotations
import libero.gpu.precision as scalar

import numpy as np
import warp as wp

from . import kernels
from .identity import NA, NBODY, NCONMAX, NEQ, NGEOM, NJNT, NM, NPLUGIN, NMOCAP, NQ, NTENDON, NV, NU, MODEL_FACTS, STATIC_SCHEMA, TOPOLOGY_FIELDS, validate_fixed_static, validate_model_facts, FIXED_STATIC_SHA
from .schema import DENSE_COLS, ERR_BAD_ACTIVE, ERR_BAD_SCHEMA, SEG_FRICTION_ROWS, SEG_LIMIT_ROWS, STATUS_OK, check_host_array, newton_input_descriptor

def _flat(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array).ravel()


class PositionReferenceWorkspace:
    """Preallocated N-world chain. upload_* are preparation; launch_* are production."""

    def __init__(self, static: dict, nworld: int, row_capacity: int,
                 contact_capacity: int, *, model_facts: dict, device='cuda:0'):
        if nworld < 1 or row_capacity < 1 or contact_capacity < 1:
            raise ValueError('positive N/R/C required')
        checked = validate_fixed_static(static)
        self.model_facts = validate_model_facts(model_facts)
        self.nworld = int(nworld)
        self.rows = int(row_capacity)
        self.contacts = int(contact_capacity)
        self.device = device
        self.static_checked = checked

        # ---- host-widened capacity arithmetic (documented int32 safety) ----
        R, C, N = self.rows, self.contacts, self.nworld
        self.capacity_proof = {
            'rows': R, 'contacts': C, 'nworld': N, 'dense_cols': NV,
            'nnzJ_per_world': R * NV,
            'efc_J_elements': N * R * NV,
            'int32_max': 2 ** 31 - 1,
            'safe': bool(N * R * NV < 2 ** 31 - 1 and R * NV < 2 ** 31 - 1
                         and N * R < 2 ** 31 - 1 and N * C * NV * 3 < 2 ** 31 - 1
                          and N * SEG_LIMIT_ROWS * NV < 2 ** 31 - 1
                          and N * NBODY * 3 < 2 ** 31 - 1),
        }
        if not self.capacity_proof['safe']:
            raise ValueError('device int32 index overflow risk')

        # ---- shared flat integrated buffers (single owner) ----------------
        nf_cap, nl_cap = SEG_FRICTION_ROWS, SEG_LIMIT_ROWS
        self.qpos = wp.zeros(N * NQ, dtype=scalar.wp_real, device=device)
        self.active = wp.zeros(N, dtype=wp.int32, device=device)
        self.ncon = wp.zeros(N, dtype=wp.int32, device=device)
        self.qvel = wp.zeros(N * NV, dtype=scalar.wp_real, device=device)
        self.qvel_epoch = wp.zeros(N, dtype=wp.int64, device=device)
        self.zero_incoming = wp.zeros(N, dtype=wp.int32, device=device)
        self.row_cap_flat = wp.zeros(N, dtype=wp.int32, device=device)
        self.incoming_contact = wp.zeros(N, dtype=wp.int32, device=device)
        self.active_s01 = wp.zeros(N, dtype=wp.int32, device=device)
        self.active_position = wp.zeros(N, dtype=wp.int32, device=device)
        self.reference_ready = wp.zeros(N, dtype=wp.int32, device=device)
        self.expected_qvel_epoch = wp.zeros(N, dtype=wp.int64, device=device)
        self._held_buffers = {}

        self.efc_J = wp.zeros(N * R * NV, dtype=scalar.wp_real, device=device)
        self.efc_type = wp.zeros(N * R, dtype=wp.int32, device=device)
        self.efc_id = wp.zeros(N * R, dtype=wp.int32, device=device)
        self.efc_pos = wp.zeros(N * R, dtype=scalar.wp_real, device=device)
        self.efc_margin = wp.zeros(N * R, dtype=scalar.wp_real, device=device)
        self.efc_frictionloss = wp.zeros(N * R, dtype=scalar.wp_real, device=device)
        self.prediag = wp.zeros(N * R, dtype=scalar.wp_real, device=device)
        self.efc_vel = wp.zeros(N * R, dtype=scalar.wp_real, device=device)
        self.efc_aref = wp.zeros(N * R, dtype=scalar.wp_real, device=device)
        self.ref_epoch = wp.zeros(N, dtype=wp.int64, device=device)
        self.counts = wp.zeros(N * 5, dtype=wp.int32, device=device)
        self.status = wp.zeros(N, dtype=wp.int32, device=device)
        self.committed = wp.zeros(N, dtype=wp.int32, device=device)
        # All position producers write scratch until every stage validates.
        self.stage_J = wp.zeros(N * R * NV, dtype=scalar.wp_real, device=device)
        self.stage_type = wp.zeros(N * R, dtype=wp.int32, device=device)
        self.stage_id = wp.zeros(N * R, dtype=wp.int32, device=device)
        self.stage_pos = wp.zeros(N * R, dtype=scalar.wp_real, device=device)
        self.stage_margin = wp.zeros(N * R, dtype=scalar.wp_real, device=device)
        self.stage_frictionloss = wp.zeros(N * R, dtype=scalar.wp_real, device=device)
        self.stage_prediag = wp.zeros(N * R, dtype=scalar.wp_real, device=device)
        self.stage_counts = wp.zeros(N * 5, dtype=wp.int32, device=device)
        self.position_ready = wp.zeros(N, dtype=wp.int32, device=device)
        self.efc_R = wp.zeros(N * R, dtype=scalar.wp_real, device=device)
        self.efc_D = wp.zeros(N * R, dtype=scalar.wp_real, device=device)
        self.efc_KBIP = wp.zeros(N * R * 4, dtype=scalar.wp_real, device=device)
        self.postdiag = wp.zeros(N * R, dtype=scalar.wp_real, device=device)
        self.contact_mu = wp.zeros(N * C, dtype=scalar.wp_real, device=device)
        self.contact_address = wp.zeros((N, C), dtype=wp.int32, device=device)
        self.contact_exclude = wp.zeros((N, C), dtype=wp.int32, device=device)
        self.route = wp.zeros(N * R, dtype=wp.int32, device=device)
        self.stage_vel = wp.zeros(N * R, dtype=scalar.wp_real, device=device)
        self.stage_aref = wp.zeros(N * R, dtype=scalar.wp_real, device=device)
        # Private group boundaries for the handwritten block impedance pass.
        # Allocated before the runtime seals its owner/layout inventory.
        self._cuda_impedance_heads = wp.zeros(N * R, dtype=wp.int32, device=device)

        # static copies used by the integrated prediag kernel
        self.geom_body = wp.array(
            np.ascontiguousarray(checked['geom_bodyid']), dtype=wp.int32, device=device)
        self.body_inv = wp.array(
            _flat(checked['body_invweight0']), dtype=scalar.wp_real, device=device)
        self.dof_inv = wp.array(
            np.ascontiguousarray(checked['dof_invweight0']), dtype=scalar.wp_real, device=device)
        self.jnt_dofadr = wp.array(
            np.ascontiguousarray(checked['jnt_dofadr']), dtype=wp.int32, device=device)

        # ---- reused phase-05 segment producers ----------------------------
        from libero.gpu.reference.constraints.constraint_scalar.kernel import ScalarWorkspace
        from libero.gpu.reference.constraints.constraint_contact.operator import ContactStatic, ContactWorkspace
        from libero.gpu.reference.constraints.constraint_impedance.kernel import ImpedanceWorkspace

        s02_static = {key: checked[key] for key in (
            'dof_frictionloss', 'jnt_type', 'jnt_limited', 'jnt_qposadr',
            'jnt_dofadr', 'jnt_range', 'jnt_margin')}
        s02_sha = {key: FIXED_STATIC_SHA[key] for key in s02_static}
        self.s02 = ScalarWorkspace(N, s02_static, s02_sha, device=device)
        topo = {key: checked[key] for key in TOPOLOGY_FIELDS}
        topo['geom_bodyid'] = checked['geom_bodyid']
        from .identity import MJB_SHA256
        self.contact_static = ContactStatic(topo, MJB_SHA256,
                                           jacobian_mode='dense', cone_mode='elliptic')
        self.s01 = ContactWorkspace(self.contact_static, N, C, R, device=device)
        s03_static = {
            'dof_solref': checked['dof_solref'], 'dof_solimp': checked['dof_solimp'],
            'jnt_solref': checked['jnt_solref'], 'jnt_solimp': checked['jnt_solimp'],
            'timestep': np.asarray(self.model_facts['timestep'], np.float64),
            'impratio': np.asarray(self.model_facts['impratio'], np.float64),
            'disableflags': np.asarray(self.model_facts['disableflags'], np.int32),
        }
        self.s03 = ImpedanceWorkspace(N, R, C, s03_static, device=device,
                                      option_domain='fixed_model')

        # ---- alias shared buffers (single owner per buffer) ---------------
        self.s02.qpos = self.qpos
        self.s02.active = self.active_position
        self.s02.incoming = self.zero_incoming
        self.s02.capacity = self.row_cap_flat

        self.s01.ncon = self.ncon
        self.s01.incoming = self.incoming_contact
        self.s01.capacity = self.row_cap_flat
        self.s01.active = self.active_s01

        self.s03.ncon = self.ncon
        self.s03.active = self.position_ready
        self.s03.typ = self.stage_type
        self.s03.eid = self.stage_id
        self.s03.pos = self.stage_pos
        self.s03.margin = self.stage_margin
        self.s03.prediag = self.stage_prediag

        # contact input buffers live in the S01 workspace (shaped) and are
        # mirrored into S03's flat parameter buffers at upload.
        self._mu = self.s03.mu
        self._route = self.s03.route

    # ------------------------------------------------------------------
    # preparation
    # ------------------------------------------------------------------
    def upload_inputs(self, data: dict):
        """Validate schema and assign every resident input buffer."""
        N, R, C = self.nworld, self.rows, self.contacts
        shapes = {
            'qpos': (N, NQ), 'active': (N,), 'ncon': (N,),
            'geom1': (N, C), 'geom2': (N, C), 'dim': (N, C), 'exclude': (N, C),
            'pos': (N, C, 3), 'frame': (N, C, 9), 'dist': (N, C), 'margin': (N, C),
            'solref': (N, C, 2), 'solreffriction': (N, C, 2),
            'solimp': (N, C, 5), 'friction': (N, C, 5),
            'cdof': (N, NV, 6), 'subtree_com': (N, NBODY, 3),
            'qvel': (N, NV), 'qvel_epoch': (N,),
        }
        dtypes = {
            'qpos': np.float64, 'active': np.int32, 'ncon': np.int32,
            'geom1': np.int32, 'geom2': np.int32, 'dim': np.int32, 'exclude': np.int32,
            'pos': np.float64, 'frame': np.float64, 'dist': np.float64, 'margin': np.float64,
            'solref': np.float64, 'solreffriction': np.float64,
            'solimp': np.float64, 'friction': np.float64,
            'cdof': np.float64, 'subtree_com': np.float64,
            'qvel': np.float64, 'qvel_epoch': np.int64,
        }
        if set(data) != set(shapes):
            raise ValueError(f'input key set {sorted(data)} != {sorted(shapes)}')
        prepared = {}
        for key, shape in shapes.items():
            prepared[key] = check_host_array(data[key], shape, dtypes[key], key)
        # host domain gate (P06S-R02-F03): contact solimp power must land in
        # the supported {1, 2} domain after the source clamp (max(1, power)).
        # Unsupported exponents are rejected at the input boundary instead of
        # reaching the device rejection path with committed rows.
        for w in range(N):
            nc = int(prepared['ncon'][w])
            if int(prepared['active'][w]) != 1 or nc < 0 or nc > C:
                continue  # the device first-error ordering owns these failures
            used = prepared['exclude'][w, :nc] == 0
            powers = prepared['solimp'][w, :nc, 4][used]
            clamped = np.maximum(1.0, powers)
            bad = (~np.isfinite(clamped)) | ((clamped != 1.0) & (clamped != 2.0))
            if bool(np.any(bad)):
                raise ValueError(f'unsupported solimp power in active included world {w}')

        self.qpos.assign(prepared['qpos'].ravel())
        self.active.assign(prepared['active'].ravel())
        self.ncon.assign(prepared['ncon'].ravel())
        self.qvel.assign(prepared['qvel'].ravel())
        self.qvel_epoch.assign(prepared['qvel_epoch'].ravel())
        self.expected_qvel_epoch.assign(prepared['qvel_epoch'].ravel())
        self.row_cap_flat.assign(np.full(N, R, np.int32))

        s01 = self.s01
        s01.geom1.assign(prepared['geom1'].ravel())
        s01.geom2.assign(prepared['geom2'].ravel())
        s01.dim.assign(prepared['dim'].ravel())
        s01.exclude.assign(prepared['exclude'].ravel())
        s01.pos.assign(prepared['pos'].ravel())
        s01.frame.assign(prepared['frame'].ravel())
        s01.dist.assign(prepared['dist'].ravel())
        s01.margin.assign(prepared['margin'].ravel())
        s01.cdof.assign(prepared['cdof'].ravel())
        s01.subtree.assign(prepared['subtree_com'].ravel())

        s03 = self.s03
        s03.cref.assign(prepared['solref'].ravel())
        s03.creff.assign(prepared['solreffriction'].ravel())
        s03.cimp.assign(prepared['solimp'].ravel())
        s03.cfric.assign(prepared['friction'].ravel())
        s03.cdim.assign(prepared['dim'].ravel().copy())
        self.contact_mu.assign(np.zeros(N * C, np.float64))

        # Contact write-set boundary values: at the collision-stage boundary
        # efc_address is -1 for every contact and exclude holds the collision
        # flags.  S writes efc_address only for included contacts (and -1/3 on
        # the no-dof path); exclude is never rewritten for the fixed nv=43
        # model, so seeding the output boundary reproduces the source hold.
        s01 = self.s01
        self.contact_address.assign(np.full((N, C), -1, np.int32))
        self.contact_exclude.assign(prepared['exclude'].copy())

    def seed_outputs(self, values: dict):
        """Seed public output buffers (dirty prefill) outside production."""
        N, R, C = self.nworld, self.rows, self.contacts
        specs = {
            'efc_J': (N * R * NV, np.float64), 'efc_type': (N * R, np.int32),
            'efc_id': (N * R, np.int32), 'efc_pos': (N * R, np.float64),
            'efc_margin': (N * R, np.float64), 'efc_frictionloss': (N * R, np.float64),
            'prediag': (N * R, np.float64), 'efc_vel': (N * R, np.float64),
            'efc_aref': (N * R, np.float64), 'ref_epoch': (N, np.int64),
            'counts': (N * 5, np.int32), 'status': (N, np.int32),
            'committed': (N, np.int32), 'route': (N * R, np.int32),
            'efc_R': (N * R, np.float64), 'efc_D': (N * R, np.float64),
            'efc_KBIP': (N * R * 4, np.float64), 'postdiag': (N * R, np.float64),
            'mu': (N * C, np.float64),
            'out_address': (N * C, np.int32), 'out_exclude': (N * C, np.int32),
        }
        for key, value in values.items():
            if key not in specs:
                raise ValueError(f'unknown seeded output {key}')
            size, dtype = specs[key]
            arr = np.asarray(value)
            if arr.dtype != dtype or arr.size != size:
                raise ValueError(f'seed schema {key}: {arr.dtype}/{arr.size} != {dtype}/{size}')
            self._output_buffer(key).assign(np.ascontiguousarray(arr).ravel())

    def _output_buffer(self, key: str):
        return {
            'efc_J': self.efc_J, 'efc_type': self.efc_type, 'efc_id': self.efc_id,
            'efc_pos': self.efc_pos, 'efc_margin': self.efc_margin,
            'efc_frictionloss': self.efc_frictionloss, 'prediag': self.prediag,
            'efc_vel': self.efc_vel, 'efc_aref': self.efc_aref,
            'ref_epoch': self.ref_epoch, 'counts': self.counts,
            'status': self.status, 'committed': self.committed,
            'route': self.route, 'mu': self.contact_mu,
            'efc_R': self.efc_R, 'efc_D': self.efc_D,
            'efc_KBIP': self.efc_KBIP, 'postdiag': self.postdiag,
            'out_address': self.contact_address, 'out_exclude': self.contact_exclude,
        }[key]

    def reset_status(self):
        """Explicit boundary reset of the sticky public status."""
        self.status.assign(np.zeros(self.nworld, np.int32))

    def update_qvel(self, qvel, qvel_epoch):
        """Replace the resident velocity state and its epoch tag."""
        N = self.nworld
        v = check_host_array(qvel, (N, NV), np.float64, 'qvel')
        e = check_host_array(qvel_epoch, (N,), np.int64, 'qvel_epoch')
        self.qvel.assign(v.ravel())
        self.qvel_epoch.assign(e.ravel())
        self.expected_qvel_epoch.assign(e.ravel())

    # ------------------------------------------------------------------
    # production
    # ------------------------------------------------------------------
    def launch_position(self, stream=None):
        """Full source-order position construction + prediag + impedance.

        Device-only sequence; no host array access, no allocation.
        """
        N, R, C = self.nworld, self.rows, self.contacts
        wp.launch(kernels._position_gate, dim=N,
                  inputs=[self.active, self.status, self.active_position, self.committed],
                  device=self.device, stream=stream)
        wp.launch(kernels._reset_stage_status, dim=N,
                  inputs=[self.s02.status, self.s01.status, self.s03.status],
                  device=self.device, stream=stream)
        self.s02.launch(stream=stream)
        wp.launch(kernels._contact_incoming, dim=N,
                  inputs=[self.s02.nf_delta, self.s02.nlimit_delta, self.incoming_contact],
                  device=self.device, stream=stream)
        wp.launch(kernels._gate_active_status, dim=N,
                  inputs=[self.active_position, self.s02.status, self.active_s01],
                  device=self.device, stream=stream)
        self.s01.launch(stream=stream)
        wp.launch(kernels._assemble, dim=N, inputs=[
            self.active_position, self.s02.status, self.s01.status,
            self.s02.nf_delta, self.s02.nlimit_delta, self.ncon,
            self.s02.friction_J, self.s02.friction_pos, self.s02.friction_margin,
            self.s02.friction_loss, self.s02.friction_type, self.s02.friction_id,
            self.s02.limit_J, self.s02.limit_pos, self.s02.limit_margin,
            self.s02.limit_loss, self.s02.limit_type, self.s02.limit_id,
            self.s01.rowstart, self.s01.local_count, self.s01.dim,
            self.s01.exclude, self.s01.J,
            self.s01.out_pos, self.s01.out_margin, self.s01.out_frictionloss,
            self.s01.out_type, self.s01.out_id,
            self.stage_J, self.stage_type, self.stage_id, self.stage_pos,
            self.stage_margin, self.stage_frictionloss,
            self.stage_counts, self.s03.ne, self.s03.nf, self.s03.count,
            self.status, self.position_ready,
            R, SEG_FRICTION_ROWS, SEG_LIMIT_ROWS, C, NV],
            device=self.device, stream=stream)
        wp.launch(kernels._prediag, dim=N, inputs=[
            self.position_ready, self.stage_counts, self.stage_type, self.stage_id,
            self.ncon, self.s01.geom1, self.s01.geom2, self.s01.dim,
            self.geom_body, self.body_inv, self.dof_inv, self.jnt_dofadr,
            self.stage_prediag, self.status, R, NV, NJNT],
            device=self.device, stream=stream)
        self.s03.launch(stream=stream)
        # fold any impedance-stage rejection into the sticky public status so
        # a failed world can never look like a partial success (R02-F03)
        wp.launch(kernels._merge_stage_status, dim=N,
                  inputs=[self.s03.status, self.status, self.position_ready],
                  device=self.device, stream=stream)

        wp.launch(kernels._commit_position, dim=N, inputs=[
            self.position_ready, self.status, self.committed, self.stage_counts,
            self.stage_J, self.stage_type, self.stage_id, self.stage_pos,
            self.stage_margin, self.stage_frictionloss, self.stage_prediag,
            self.s03.R, self.s03.D, self.s03.KBIP, self.s03.post,
            self.s03.route, self.s03.mu, self.ncon, self.s01.exclude,
            self.s01.dim, self.s01.out_address,
            self.counts, self.efc_J, self.efc_type, self.efc_id, self.efc_pos,
            self.efc_margin, self.efc_frictionloss, self.prediag,
            self.efc_R, self.efc_D, self.efc_KBIP, self.postdiag, self.route,
            self.contact_mu, self.contact_address, R, C, NV],
            device=self.device, stream=stream)

    def require_velocity_epoch(self, epoch):
        """Prepare an independently expected velocity-stage epoch on device."""
        e = check_host_array(epoch, (self.nworld,), np.int64, 'expected_qvel_epoch')
        self.expected_qvel_epoch.assign(e.ravel())

    def bind_holds(self, *, contact_H, efc_AR):
        """Retain caller-owned hold buffers; S never passes them to a kernel."""
        for name, buffer in [('contact_H', contact_H), ('efc_AR', efc_AR)]:
            if not isinstance(buffer, wp.array) or buffer.device != wp.get_device(self.device):
                raise ValueError(f'{name}: resident caller-owned array required')
            if buffer.dtype != scalar.wp_real:
                raise ValueError(f'{name}: float64 required')
        if contact_H.size != self.nworld * self.contacts * 36:
            raise ValueError('contact_H: complete contact Hessian shape required')
        self._held_buffers = {'contact_H': contact_H, 'efc_AR': efc_AR}

    def download_holds(self):
        return {k: v.numpy().copy() for k, v in self._held_buffers.items()}

    def launch_reference(self, stream=None):
        """Velocity reference stage: consumes position rows + current qvel."""
        N, R = self.nworld, self.rows
        wp.launch(kernels._reference_gate, dim=N, inputs=[
            self.active, self.status, self.committed, self.counts,
            self.qvel, self.qvel_epoch, self.expected_qvel_epoch,
            self.reference_ready, R, NV], device=self.device, stream=stream)
        wp.launch(kernels._reference, dim=(N, R), inputs=[
            self.reference_ready, self.counts, self.efc_J, self.efc_KBIP,
            self.efc_pos, self.efc_margin, self.qvel, self.qvel_epoch,
            self.stage_vel, self.stage_aref, self.expected_qvel_epoch, R, NV],
            device=self.device, stream=stream)
        wp.launch(kernels._commit_reference, dim=N, inputs=[
            self.reference_ready, self.status, self.committed, self.counts,
            self.stage_vel, self.stage_aref, self.qvel_epoch,
            self.efc_vel, self.efc_aref, self.ref_epoch, R],
            device=self.device, stream=stream)

    # ------------------------------------------------------------------
    # witnesses / downloads (outside the production interval)
    # ------------------------------------------------------------------
    def download(self) -> dict:
        N, R, C = self.nworld, self.rows, self.contacts
        return {
            'efc_J': self.efc_J.numpy().reshape(N, R, NV).copy(),
            'efc_type': self.efc_type.numpy().reshape(N, R).copy(),
            'efc_id': self.efc_id.numpy().reshape(N, R).copy(),
            'efc_pos': self.efc_pos.numpy().reshape(N, R).copy(),
            'efc_margin': self.efc_margin.numpy().reshape(N, R).copy(),
            'efc_frictionloss': self.efc_frictionloss.numpy().reshape(N, R).copy(),
            'prediag': self.prediag.numpy().reshape(N, R).copy(),
            'efc_R': self.efc_R.numpy().reshape(N, R).copy(),
            'efc_D': self.efc_D.numpy().reshape(N, R).copy(),
            'efc_KBIP': self.efc_KBIP.numpy().reshape(N, R, 4).copy(),
            'postdiag': self.postdiag.numpy().reshape(N, R).copy(),
            'efc_vel': self.efc_vel.numpy().reshape(N, R).copy(),
            'efc_aref': self.efc_aref.numpy().reshape(N, R).copy(),
            'ref_epoch': self.ref_epoch.numpy().copy(),
            'counts': self.counts.numpy().reshape(N, 5).copy(),
            'status': self.status.numpy().copy(),
            'committed': self.committed.numpy().copy(),
            'contact_efc_address': self.contact_address.numpy().reshape(N, C).copy(),
            'contact_exclude': self.contact_exclude.numpy().reshape(N, C).copy(),
            'contact_mu': self.contact_mu.numpy().reshape(N, C).copy(),
            'route': self.route.numpy().reshape(N, R).copy(),
            's02_nf': self.s02.nf_delta.numpy().copy(),
            's02_nlimit': self.s02.nlimit_delta.numpy().copy(),
            's01_local_count': self.s01.local_count.numpy().copy(),
            's01_outgoing': self.s01.outgoing.numpy().copy(),
        }

    def download_inputs(self) -> dict:
        """Consumer-side witness of the resident inputs (production inputs)."""
        N, R, C = self.nworld, self.rows, self.contacts
        s01, s03 = self.s01, self.s03
        return {
            'qpos': self.qpos.numpy().reshape(N, NQ).copy(),
            'active': self.active.numpy().copy(),
            'ncon': self.ncon.numpy().copy(),
            'geom1': s01.geom1.numpy().reshape(N, C).copy(),
            'geom2': s01.geom2.numpy().reshape(N, C).copy(),
            'dim': s01.dim.numpy().reshape(N, C).copy(),
            'exclude': s01.exclude.numpy().reshape(N, C).copy(),
            'pos': s01.pos.numpy().reshape(N, C, 3).copy(),
            'frame': s01.frame.numpy().reshape(N, C, 9).copy(),
            'dist': s01.dist.numpy().reshape(N, C).copy(),
            'margin': s01.margin.numpy().reshape(N, C).copy(),
            'solref': s03.cref.numpy().reshape(N, C, 2).copy(),
            'solreffriction': s03.creff.numpy().reshape(N, C, 2).copy(),
            'solimp': s03.cimp.numpy().reshape(N, C, 5).copy(),
            'friction': s03.cfric.numpy().reshape(N, C, 5).copy(),
            'cdof': s01.cdof.numpy().reshape(N, NV, 6).copy(),
            'subtree_com': s01.subtree.numpy().reshape(N, NBODY, 3).copy(),
            'qvel': self.qvel.numpy().reshape(N, NV).copy(),
            'qvel_epoch': self.qvel_epoch.numpy().copy(),
        }

    def newton_descriptor(self) -> dict:
        descriptor = newton_input_descriptor(self.rows, self.contacts, self.nworld)
        buffers = {k:self._output_buffer(k) for k in (
            'efc_J','efc_type','efc_id','efc_pos','efc_margin','efc_frictionloss',
            'efc_R','efc_D','efc_KBIP','efc_vel','efc_aref','counts','status','ref_epoch')}
        descriptor['actual_resident_bindings'] = {k:{'pointer':int(v.ptr),'shape':list(v.shape),
            'dtype':str(v.dtype),'device':str(v.device),'owner':'PositionReferenceWorkspace public'}
            for k,v in buffers.items()}
        descriptor['capacity_proof'] = dict(self.capacity_proof)
        descriptor['actual_N_consumer_receipt'] = 'pending; descriptor is an S producer witness'
        return descriptor
