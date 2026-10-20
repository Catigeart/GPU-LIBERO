"""Six source-order dynamics entries with device forward epochs and read witnesses.

The old candidate is archived separately. This implementation is an author
candidate until the full source bounds and independent review are complete.
"""
from __future__ import annotations
import libero.gpu.precision as scalar
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT

from types import SimpleNamespace
import importlib.util
import sys
from pathlib import Path

import numpy as np
import warp as wp

from .acceleration import SmoothXfrcSolveWorkspace, StaticTopology
from .actuation_split import ActuationForceWorkspace
from .control_clock import ControlClock
from .rules import FIELD_GROUPS
from .transmission import TransmissionWorkspace
from .witness import STAGE_ORDER, STAGE_READS, StageWitness
from . import epoch
from .control_witness import ControlConsumption, PHYS_FIELDS

wp.set_module_options({'enable_backward': False, 'fuse_fp': True, 'fast_math': False})

EEF_SITE_ID = 13


def _load(rel, name):
    if name in sys.modules:
        return sys.modules[name]
    path = (GPU_ROOT / '.') / rel
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@wp.kernel(enable_backward=False)
def _map_position_status(result_kind: wp.array(dtype=wp.int32),
                         first_code: wp.array(dtype=wp.int32),
                         transmission_status: wp.array(dtype=wp.int32),
                         out: wp.array(dtype=wp.int32)):
    """Map the accepted D04 result_kind (0 inactive, 1 ok, 2 repair, 3 error)."""
    w = wp.tid()
    if out[w] != 0:
        return
    kind = result_kind[w]
    if kind == 0 or kind == 1:
        out[w] = transmission_status[w]
    elif kind == 2:
        out[w] = 25
    else:
        out[w] = 30 + first_code[w]


@wp.kernel(enable_backward=False)
def _gather_eef_site(site_xpos: wp.array(dtype=scalar.wp_real, ndim=2),
                     site_xmat: wp.array(dtype=scalar.wp_real, ndim=2),
                     stage_active: wp.array(dtype=wp.int32),
                     site_pos: wp.array(dtype=scalar.wp_real, ndim=2),
                     site_mat: wp.array(dtype=scalar.wp_real, ndim=2),
                     nsite: int, eef: int):
    """Fixed device gather of the EEF site pose from the position frames
    output (the only D07 site source; no host values)."""
    w = wp.tid()
    if stage_active[w] != 1:
        for k in range(3):
            site_pos[w, k] = scalar.wp_real(0.0)
        for k in range(9):
            site_mat[w, k] = scalar.wp_real(0.0)
        return
    base_p = w * nsite * 3 + eef * 3
    base_m = w * nsite * 9 + eef * 9
    for k in range(3):
        site_pos[w, k] = site_xpos[w, eef * 3 + k]
    for k in range(9):
        site_mat[w, k] = site_xmat[w, eef * 9 + k]


@wp.kernel(enable_backward=False)
def _merge_status3(a: wp.array(dtype=wp.int32), b: wp.array(dtype=wp.int32),
                   c: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
    w = wp.tid()
    if a[w] != 0:
        out[w] = 100 + a[w]
    elif b[w] != 0:
        out[w] = 200 + b[w]
    elif c[w] != 0:
        out[w] = 300 + c[w]
    else:
        out[w] = 0


@wp.kernel(enable_backward=False)
def _copy_status(src: wp.array(dtype=wp.int32), dst: wp.array(dtype=wp.int32)):
    w = wp.tid()
    dst[w] = src[w]


@wp.kernel(enable_backward=False)
def _before_control(adapter_status: wp.array(dtype=wp.int32),
                    gate_control: wp.array(dtype=wp.int32),
                    force_refresh: wp.array(dtype=wp.int32),
                    velp: wp.array(dtype=scalar.wp_real, ndim=2),
                    velr: wp.array(dtype=scalar.wp_real, ndim=2),
                    velocity: wp.array(dtype=scalar.wp_real, ndim=2),
                    control_active: wp.array(dtype=wp.int32),
                    chain_status: wp.array(dtype=wp.int32),
                    needs_update: wp.array(dtype=wp.int32)):
    w = wp.tid()
    control_active[w] = 0
    if gate_control[w] != 1:
        chain_status[w] = 0 if gate_control[w] == 0 else 90 + gate_control[w]
        return
    if adapter_status[w] == 0:
        for i in range(3):
            velocity[w, i] = velp[w, i]
            velocity[w, i + 3] = velr[w, i]
        # Two-entry semantics (source base_controller.py update branch):
        # refresh consumes new_update (needs_update=1 -> cache refresh);
        # skip runs with new_update=False (needs_update=0 -> cache preserved).
        needs_update[w] = force_refresh[w]
        control_active[w] = 1
        chain_status[w] = 0
    else:
        chain_status[w] = 10 + adapter_status[w]


@wp.kernel(enable_backward=False)
def _after_control(chain_status: wp.array(dtype=wp.int32),
                   control_error: wp.array(dtype=wp.int32),
                   control_active: wp.array(dtype=wp.int32)):
    """F10: an OSC failure can never coexist with status_control == 0."""
    w = wp.tid()
    if chain_status[w] != 0:
        return
    if control_active[w] == 1 and control_error[w] != 0:
        chain_status[w] = 200 + control_error[w]


@wp.kernel(enable_backward=False)
def _before_actuation(status_control: wp.array(dtype=wp.int32),
                      control_active: wp.array(dtype=wp.int32),
                      control_error: wp.array(dtype=wp.int32),
                      gate_actuation: wp.array(dtype=wp.int32),
                      actuation_active: wp.array(dtype=wp.int32),
                      status_actuation: wp.array(dtype=wp.int32)):
    w = wp.tid()
    actuation_active[w] = 0
    if gate_actuation[w] != 1:
        status_actuation[w] = 0
        return
    if status_control[w] != 0:
        status_actuation[w] = status_control[w]
        return
    if control_error[w] != 0:
        status_actuation[w] = 200 + control_error[w]
        return
    status_actuation[w] = 0
    if control_active[w] == 1:
        actuation_active[w] = 1


@wp.kernel(enable_backward=False)
def _after_actuation(actuation_status: wp.array(dtype=wp.int32),
                     chain_status: wp.array(dtype=wp.int32)):
    w = wp.tid()
    if chain_status[w] == 0 and actuation_status[w] != 0:
        chain_status[w] = 300 + actuation_status[w]


class DynamicsStages:
    """One fixed model and capacity; six staged launch entries.

    Constructor wiring (zero-copy device edges, frozen afterwards):

    * state qpos/qvel/active: one owner each, viewed by every stage
    * position outputs xipos/ximat/subtree_com/cinert/cdof/qM/site frames:
      consumed by passive, comVel, RNE bias, D07 and the combinator
    * velocity outputs cvel/cdof_dot: consumed by passive and RNE bias
    * passive/bias/actuator force outputs: the three-RN inputs
    * controller ctrl: the actuation force input
    * each consumer stage's active mask is its gate (R02-F02/F03)
    """

    CAPACITIES = (1, 4, 20, 128)

    def __init__(self, model_pack, capacity: int, device='cuda:0'):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('positive integer world capacity required')
        self.model_pack = model_pack
        self.capacity = capacity
        self.device = device
        self.stream = wp.get_stream(device)
        self.clock = ControlClock()

        from libero.gpu.reference.position.body_mass_chain.chain import BodyMassChainWorkspace

        comvel_mod = _load('reference/dynamics/comvel/kernel.py', 'p06d_comvel')
        rne_mod = _load('reference/dynamics/rne_bias/kernel.py', 'p06d_rne_bias')
        passive_mod = _load('reference/dynamics/passive_chain/kernel.py', 'p06d_passive')
        d07_mod = _load('reference/control/physics_osc_inputs_chain/chain.py', 'p06d_d07')
        osc_ctrl_mod = _load('reference/control/osc/controller.py', 'p06d_osc_ctrl')

        self.position = BodyMassChainWorkspace(
            model_pack.frozen(), model_pack.frame_model(),
            model_pack.mass_model(), capacity, device)
        flat = model_pack.flat_model()
        self.transmission = TransmissionWorkspace(flat, capacity, device=device)
        vb = SimpleNamespace(**model_pack.velocity_bias_fields)
        self.comvel_model = comvel_mod.ComVelModel.prepare(vb, device)
        self.rne_model = rne_mod.RNEBiasModel.prepare(vb, device)
        self.passive = passive_mod.PassiveChainWorkspace(
            flat, capacity, device=device)
        self.d07 = d07_mod.ChainWorkspace(
            model_pack.osc_fields(), capacity,
            model_mjb_sha256=model_pack.mjb_sha256, device=device)
        self.control = osc_ctrl_mod.PandaOSCWorkspace(
            capacity, device=device, uncouple=True, diagnostics=True)
        self.actuation = ActuationForceWorkspace(flat, capacity, device=device)

        F, I = scalar.wp_real, wp.int32
        nb, nv, nq, ns = 38, 43, 48, 31

        # --- single-owner state arrays ---
        self.qpos = self.position.body.qpos
        self.gate_position = self.position.body.active
        self.active = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.qvel = wp.zeros((capacity, nv), dtype=F, device=device)
        self.ctrl = self.control.ctrl
        self.qfrc_applied = wp.zeros((capacity, nv), dtype=F, device=device)
        self.xfrc_applied = wp.zeros((capacity, nb, 6), dtype=F, device=device)

        # --- stage gates: the consumer active masks (R02-F02) ---
        self.gate_velocity = wp.zeros(capacity, dtype=I, device=device)
        self.gate_bias = wp.zeros(capacity, dtype=I, device=device)
        self.gate_control = wp.zeros(capacity, dtype=I, device=device)
        self.gate_actuation = wp.zeros(capacity, dtype=I, device=device)
        self.gate_acceleration = wp.zeros(capacity, dtype=I, device=device)

        # --- alias state into every consumer workspace ---
        self.transmission.qpos = self.qpos
        self.transmission.qvel = self.qvel
        self.transmission.position_active = self.gate_position
        self.transmission.velocity_active = self.gate_velocity
        self.passive.qpos = self.qpos
        self.passive.qvel = self.qvel
        self.passive.active = self.gate_velocity
        self.d07.osc.qpos = self.qpos
        self.d07.osc.qvel = self.qvel
        self.d07.osc.active = self.gate_control

        # --- velocity/bias outputs ---
        self.cvel = wp.zeros((capacity, nb * 6), dtype=F, device=device)
        self.cdof_dot = wp.zeros((capacity, nv * 6), dtype=F, device=device)
        self.cacc = wp.zeros((capacity, nb * 6), dtype=F, device=device)
        self.frc = wp.zeros((capacity, nb * 6), dtype=F, device=device)
        self.qfrc_passive = self.passive.qfrc
        self.qfrc_bias = wp.zeros((capacity, nv), dtype=F, device=device)
        self.status_comvel = wp.zeros(capacity, dtype=I, device=device)
        self.status_passive = self.passive.status
        self.status_rne = wp.zeros(capacity, dtype=I, device=device)

        # --- position outputs are the single geometry/mass truth ---
        pos = self.position
        pos_cdof_flat = pos.mass.cdof
        pos_com_flat = pos.mass.subtree_com
        pos_xipos_flat = pos.mass.xipos
        pos_ximat_flat = pos.mass.ximat
        self.passive.cvel = self.cvel.reshape((capacity, nb, 6))
        self.passive.subtree_com = pos_com_flat.reshape((capacity, nb, 3))
        self.passive.xipos = pos_xipos_flat.reshape((capacity, nb, 3))
        self.passive.ximat = pos_ximat_flat.reshape((capacity, nb, 9))
        self.passive.cdof = pos_cdof_flat.reshape((capacity, nv, 6))
        self.d07.cdof = pos_cdof_flat.reshape((capacity, nv, 6))
        self.d07.subtree_com = pos_com_flat.reshape((capacity, nb, 3))

        # --- D07 dynamic inputs (R02-F01): qM, qfrc_bias, site pose ---
        self.d07.osc.qM = pos.mass.mass.qm
        self.d07.osc.qfrc_bias = self.qfrc_bias
        self.site_xpos_flat = pos.frames.site_xpos
        self.site_xmat_flat = pos.frames.site_xmat
        if self.site_xpos_flat.shape != (capacity * ns * 3,):
            self.site_xpos_flat = self.site_xpos_flat.reshape((capacity, ns * 3))
        else:
            self.site_xpos_flat = self.site_xpos_flat.reshape((capacity, ns * 3))
        if self.site_xmat_flat.shape != (capacity * ns * 9,):
            self.site_xmat_flat = self.site_xmat_flat.reshape((capacity, ns * 9))
        else:
            self.site_xmat_flat = self.site_xmat_flat.reshape((capacity, ns * 9))

        # --- actuation consumes staged transmission + controller ctrl ---
        self.actuation.ctrl = self.ctrl
        self.control_active = wp.zeros(capacity, dtype=I, device=device)
        self.actuation_active = wp.zeros(capacity, dtype=I, device=device)
        self.actuation.active = self.actuation_active
        self.control.active = self.control_active

        # --- acceleration combinator consumes actual force outputs ---
        topology = StaticTopology({
            name: np.asarray(getattr(model_pack.model, name), dtype=np.int32)
            for name in ('body_parentid', 'body_rootid', 'body_dofadr',
                         'body_dofnum', 'dof_parentid')})
        self.acceleration = SmoothXfrcSolveWorkspace(
            topology, model_pack.solve_model(), capacity, device=device)
        acc = self.acceleration
        acc.passive = self.qfrc_passive
        acc.bias = self.qfrc_bias
        acc.applied = self.qfrc_applied
        acc.actuator = self.actuation.qfrc_actuator
        acc.active = self.gate_acceleration
        acc.d02.active_world = self.gate_acceleration
        acc.solver.active = self.gate_acceleration
        acc.xfrc = self.xfrc_applied
        acc.d02.xfrc_applied = self.xfrc_applied
        acc.xipos = pos_xipos_flat.reshape((capacity, nb, 3))
        acc.d02.xipos = acc.xipos
        acc.subtree_com = pos_com_flat.reshape((capacity, nb, 3))
        acc.d02.subtree_com = acc.subtree_com
        acc.cdof = pos_cdof_flat.reshape((capacity, nv, 6))
        acc.d02.cdof = acc.cdof
        acc.qm = pos.mass.mass.qm
        acc.solver.qm = pos.mass.mass.qm
        # R02-F09: D02 static topology is fixed; load it at prepare so the
        # staged path passes the body_count/body_ids checks in _project.
        ids = np.tile(np.arange(1, nb, dtype=np.int32), (capacity, 1))
        acc.d02.body_ids.assign(ids)
        acc.d02.body_count.assign(np.full(capacity, nb - 1, dtype=np.int32))
        # R02-F07: give the D02 output a deterministic zero start state.
        acc.d02.qfrc.fill_(0.0)
        acc.d02.scratch.fill_(0.0)

        # --- controller wires D07 adapter outputs (accepted D08 pattern) ---
        p, c = self.d07.osc, self.control
        c.phys_mass = p.M.reshape((capacity, 49))
        c.phys_jacobian = p.J.reshape((capacity, 42))
        c.phys_q = p.joint_pos
        c.phys_qvel = p.joint_vel
        c.phys_bias = p.bias
        c.phys_ee_pos = p.ee_pos
        c.phys_ee_rot = p.ee_mat
        # Fixed Phase-1 controller configuration (kp=150, kd=2*sqrt(150),
        # actuator ctrlrange from the frozen MJB).  Without these the OSC clip
        # stage would force ctrl to 0 -- they are static config, not results.
        ctrl_range = np.asarray(model_pack.model.actuator_ctrlrange,
                                dtype=np.float64).reshape(18)
        c.ctrl_range.assign(np.tile(ctrl_range, (capacity, 1)))
        gains_row = np.concatenate([np.full(6, 150.0),
                                    np.full(6, 2.0 * np.sqrt(150.0))])
        c.gains.assign(np.tile(gains_row, (capacity, 1)))

        # --- stage statuses and control inputs ---
        self.status_position = wp.zeros(capacity, dtype=I, device=device)
        self.status_velocity = wp.zeros(capacity, dtype=I, device=device)
        self.status_bias = wp.zeros(capacity, dtype=I, device=device)
        self.status_control = wp.zeros(capacity, dtype=I, device=device)
        self.status_actuation = wp.zeros(capacity, dtype=I, device=device)
        self.force_refresh = wp.zeros(capacity, dtype=I, device=device)
        self.policy_step = wp.zeros(capacity, dtype=I, device=device)
        # R03-F11: controller and staging share ONE device policy_step buffer;
        # entries only device-fill it (no D2H/H2D round-trip inside launch).
        c.policy_step = self.policy_step
        self.witness = StageWitness(capacity, device=device)
        self.reset_mask = wp.ones(capacity, dtype=I, device=device)
        self.input_versions = wp.zeros((capacity, 3), dtype=wp.int64, device=device)
        self.device_clock = wp.zeros((capacity, 7), dtype=wp.int64, device=device)
        self.cache_forward = wp.zeros(capacity, dtype=wp.int64, device=device)
        self.bias_read_version = wp.zeros(capacity, dtype=wp.int64, device=device)
        # Preserve every persistent output by bytes until control succeeds.
        input_names = {"active", "policy_step", "error", "q0", "action", "gains", "ctrl_range"}
        self.control_previous = {}
        self.control_bytes = {}
        self.control_previous_bytes = {}
        for name, arr in vars(c).items():
            if isinstance(arr, wp.array) and not name.startswith("phys_") and name not in input_names:
                saved = wp.empty_like(arr)
                self.control_previous[name] = saved
                self.control_bytes[name] = self._byte_view(arr)
                self.control_previous_bytes[name] = self._byte_view(saved)
        # The snapshots are actual device consumer inputs. No weighted sums.
        self.consumer_inputs = {
            "position": {"qpos": self.qpos},
            "velocity_prefix": {"qvel": self.qvel, "cdof": pos_cdof_flat,
                "xipos": pos_xipos_flat, "ximat": pos_ximat_flat,
                "subtree_com": pos_com_flat, "moment": self.transmission.moment},
            "bias": {"cdof_dot": self.cdof_dot, "qvel": self.qvel,
                "cinert": pos.mass.cinert, "cvel": self.cvel, "cdof": pos_cdof_flat},
            "control": {"qpos": self.qpos, "qvel": self.qvel, "qM": acc.qm,
                "qfrc_bias": self.qfrc_bias, "cdof": pos_cdof_flat,
                "subtree_com": pos_com_flat, "site_xpos": self.site_xpos_flat,
                "site_xmat": self.site_xmat_flat, "action": c.action, "q0": c.q0,
                "gripper": c.gripper, "input_versions": self.input_versions, "cached_mass": c.mass,
                "cached_jacobian": c.jacobian, "cached_pose": c.ee_pos,
                "cached_rotation": c.ee_rot, "cached_q": c.q, "cached_qvel": c.qvel,
                "cached_velocity": c.ee_velocity, "goal_pos": c.goal_pos, "goal_rot": c.goal_rot},
            "actuation": {"ctrl": self.ctrl, "length": self.transmission.length,
                "moment": self.transmission.moment, "velocity": self.transmission.velocity},
            "acceleration": {"passive": acc.passive, "bias": acc.bias, "applied": acc.applied,
                "actuator": acc.actuator, "xfrc": acc.xfrc, "xipos": acc.xipos,
                "subtree_com": acc.subtree_com, "cdof": acc.cdof, "qM": acc.qm},
        }
        self.consumer_witness = {stage: {name: wp.empty_like(arr) for name, arr in fields.items()}
                                 for stage, fields in self.consumer_inputs.items()}
        # Physical entries are captured late, after D07, rather than by _stage_gate.
        self.consumer_inputs["control"].update({name:getattr(c,name) for name in PHYS_FIELDS})
        self.consumer_witness["control"].update({name:wp.empty_like(getattr(c,name)) for name in PHYS_FIELDS})
        self.control_consumption = ControlConsumption(c, capacity, device)
        self._frozen_control_config = (c.nworld, str(c.device), c.diagnostics, c.uncouple)
        self._freeze_bindings()

    # ----- binding freeze/validate -------------------------------------
    @staticmethod
    def _byte_view(arr):
        rowbytes = int(arr.size * wp.types.type_size_in_bytes(arr.dtype)) // int(arr.shape[0])
        view = wp.array(ptr=arr.ptr, shape=(int(arr.shape[0]), rowbytes),
                        dtype=wp.uint8, device=arr.device, copy=False)
        view._ref = arr
        return view

    @staticmethod
    def _array_binding(arr):
        chain = []
        owner = arr
        while isinstance(getattr(owner, "_ref", None), wp.array):
            chain.append(id(owner))
            owner = owner._ref
            if id(owner) in chain:
                raise ValueError("cyclic array owner")
        item = int(wp.types.type_size_in_bytes(arr.dtype))
        extent = 0 if any(x == 0 for x in arr.shape) else item + sum((int(n)-1)*int(t) for n,t in zip(arr.shape,arr.strides))
        offset = int(arr.ptr)-int(owner.ptr)
        capacity = int(owner.capacity)
        if any(int(t) < 0 for t in arr.strides) or offset < 0 or offset+extent > capacity:
            raise ValueError("device view exceeds actual owner allocation")
        return (id(arr), int(arr.ptr), tuple(arr.shape), tuple(arr.strides),
                str(arr.dtype), str(arr.device), bool(arr.is_contiguous),
                tuple(chain), id(owner), int(owner.ptr), capacity, offset, extent,
                id(getattr(owner, "_allocator", None)), id(getattr(owner, "deleter", None)))

    def _binding_state(self):
        arrays, owners, refs, scalars = {}, {}, [], {}
        def walk(path, obj, ancestors):
            if isinstance(obj, (str, int, float, bool, np.generic, Path)):
                value = obj.tobytes().hex() if isinstance(obj, np.generic) else obj.hex() if isinstance(obj, float) else repr(obj)
                scalars[path] = (str(type(obj)), value)
                return
            if isinstance(obj, wp.array):
                arrays[path] = self._array_binding(obj)
                refs.append(obj)
                return
            if id(obj) in ancestors:
                return
            ancestors = ancestors + (id(obj),)
            if isinstance(obj, dict):
                for name, value in sorted(obj.items(), key=lambda t:str(t[0])):
                    walk(path+"."+str(name), value, ancestors)
            elif hasattr(obj, "__dict__") and (obj is self or
                    type(obj).__module__.startswith(("libero.", "p06d_", "warp._src.codegen"))):
                owners[path] = id(obj)
                refs.append(obj)
                for name,value in vars(obj).items():
                    if (not name.startswith("_") or name in ("_capacity", "_device")) and name not in ("clock", "model_pack", "stream"):
                        walk(path+"."+name, value, ancestors)
        walk("dynamics", self, ())
        return arrays, owners, refs, scalars

    def _freeze_bindings(self):
        self._frozen_binding_arrays, self._frozen_binding_owners, self._binding_references, self._frozen_binding_scalars = self._binding_state()
        self._frozen_stream = (id(self.stream), int(self.stream.cuda_stream), str(self.stream.device))

    def validate_bindings(self):
        c = self.control
        if (c.nworld, str(c.device), c.diagnostics, c.uncouple) != self._frozen_control_config:
            raise ValueError("controller launch scalar configuration differs from prepared values")
        current, owners, _, scalars = self._binding_state()
        stale = sorted(k for k in set(current) | set(self._frozen_binding_arrays)
                       if current.get(k) != self._frozen_binding_arrays.get(k))
        stale += sorted(k for k in set(owners) | set(self._frozen_binding_owners)
                        if owners.get(k) != self._frozen_binding_owners.get(k))
        stale += sorted(k for k in set(scalars) | set(self._frozen_binding_scalars)
                        if scalars.get(k) != self._frozen_binding_scalars.get(k))
        if stale:
            raise ValueError("stale device binding(s): " + ", ".join(stale))
        return True

    def _checked_stream(self, stream):
        selected = self.stream if stream is None else stream
        if (id(selected), int(selected.cuda_stream), str(selected.device)) != self._frozen_stream:
            raise ValueError("stream differs from prepared stream")
        return selected

    def binding_catalog(self):
        self.validate_bindings()
        fields = ("array_id", "ptr", "shape", "strides", "dtype", "device", "contiguous",
                  "view_chain", "owner_id", "owner_ptr", "owner_capacity_bytes", "offset_bytes",
                  "access_extent_bytes", "allocator_id", "deleter_id")
        return {name: dict(zip(fields, values)) for name,values in self._frozen_binding_arrays.items()}

    def scalar_catalog(self):
        self.validate_bindings()
        return {name: {"type":value[0], "exact_value_or_bits":value[1]}
                for name,value in sorted(self._frozen_binding_scalars.items())}

    # ----- upload (host boundary only) ---------------------------------
    def upload_state(self, *, active, qpos, qvel, mocap_pos, mocap_quat,
                     rhs, count, qfrc_applied, xfrc_applied):
        self.validate_bindings()
        n, nv, nb, nq = self.capacity, 43, 38, 48
        checks = {
            'active': (active, (n,), np.int32),
            'qpos': (qpos, (n, nq), np.float64),
            'qvel': (qvel, (n, nv), np.float64),
            'mocap_pos': (mocap_pos, (n, 0, 3), np.float64),
            'mocap_quat': (mocap_quat, (n, 0, 4), np.float64),
            'rhs': (rhs, (n, 4, nv), np.float64),
            'count': (count, (n,), np.int32),
            'qfrc_applied': (qfrc_applied, (n, nv), np.float64),
            'xfrc_applied': (xfrc_applied, (n, nb, 6), np.float64),
        }
        for name, (value, shape, dtype) in checks.items():
            arr = np.asarray(value)
            if arr.shape != shape or arr.dtype != dtype:
                raise ValueError(f'{name}: expected {shape} {dtype}, '
                                 f'got {arr.shape} {arr.dtype}')
        with wp.ScopedStream(self._checked_stream(None)):
            self.active.assign(np.ascontiguousarray(active))
            self.position.upload(active, qpos, mocap_pos, mocap_quat, rhs, count)
            self.qvel.assign(np.ascontiguousarray(qvel))
            self.qfrc_applied.assign(np.ascontiguousarray(qfrc_applied))
            self.xfrc_applied.assign(np.ascontiguousarray(xfrc_applied))
        self.bump_state_epoch()

    # ----- pipeline boundaries -----------------------------------------
    def begin_forward(self, stream=None):
        """Begin a distinct forward call and invalidate every stage stamp."""
        self.validate_bindings()
        stream = self._checked_stream(stream)
        wp.launch(epoch.begin_forward, dim=self.capacity,
                  inputs=[self.witness.state_epoch, self.witness.scratch_state_epoch,
                          self.witness.scratch_epoch, self.witness.first_error_stage,
                          self.witness.first_error_code], device=self.device, stream=stream)
        for stage in STAGE_ORDER:
            wp.launch(epoch.clear_valid, dim=self.capacity, inputs=[self.witness.valid[stage]],
                      device=self.device, stream=stream)
        self.witness.record('pipeline', 'begin_forward')

    def bump_state_epoch(self, stream=None):
        """State-write boundary; does not clear an earlier error cause."""
        self.validate_bindings()
        stream = self._checked_stream(stream)
        wp.launch(epoch.bump_state, dim=self.capacity,
                  inputs=[self.witness.state_epoch, self.witness.first_error_stage,
                          self.witness.first_error_code], device=self.device, stream=stream)
        for stage in STAGE_ORDER:
            wp.launch(epoch.clear_valid, dim=self.capacity, inputs=[self.witness.valid[stage]],
                      device=self.device, stream=stream)
        self.witness.record('pipeline', 'bump_state_epoch')

    def reset_selected_errors(self, mask, stream=None):
        """Explicit host boundary clears selected sticky causes, never epochs."""
        self.validate_bindings()
        arr=np.asarray(mask)
        if arr.shape != (self.capacity,) or arr.dtype != np.int32 or not np.all((arr==0)|(arr==1)):
            raise ValueError('reset mask must be binary int32 (N,)')
        stream = self._checked_stream(stream)
        with wp.ScopedStream(stream):
            self.reset_mask.assign(np.ascontiguousarray(arr))
        wp.launch(epoch.reset_errors, dim=self.capacity,
                  inputs=[self.reset_mask, self.witness.first_error_stage,
                          self.witness.first_error_code, self.control.error],
                  device=self.device, stream=stream)
        for stage in STAGE_ORDER:
            wp.launch(epoch.clear_selected_valid, dim=self.capacity,
                      inputs=[self.reset_mask, self.witness.valid[stage]],
                      device=self.device, stream=stream)
        self.witness.record('pipeline', 'explicit_error_reset')

    def _stage_gate(self, stage, gate, status, stream):
        sid = STAGE_ORDER.index(stage)+1
        wp.launch(epoch.start_stage, dim=self.capacity,
                  inputs=[self.active, self.witness.state_epoch, self.witness.scratch_state_epoch,
                          self.witness.scratch_epoch, self.witness.first_error_stage,
                          self.witness.first_error_code, gate, status, self.witness.versions[stage], sid],
                  device=self.device, stream=stream)
        for producer in STAGE_READS[stage]:
            wp.launch(epoch.require_read, dim=self.capacity,
                      inputs=[gate, self.witness.valid[producer], self.witness.stamps[producer],
                              self.witness.scratch_epoch, self.witness.first_error_stage,
                              self.witness.first_error_code, status, sid],
                      device=self.device, stream=stream)
            wp.copy(self.witness.read_stamps[stage][producer], self.witness.stamps[producer], stream=stream)
            wp.copy(self.witness.read_versions[stage][producer], self.witness.versions[producer], stream=stream)
        wp.copy(self.witness.read_mask[stage], gate, stream=stream)
        for name,source in self.consumer_inputs[stage].items():
            if stage == "control" and name in PHYS_FIELDS:
                continue
            wp.copy(self.consumer_witness[stage][name], source, stream=stream)
        # Re-executing a producer invalidates every transitive dependent.
        descendants={stage}
        for candidate in STAGE_ORDER:
            if any(p in descendants for p in STAGE_READS[candidate]):
                descendants.add(candidate)
        for candidate in descendants:
            wp.launch(epoch.clear_valid, dim=self.capacity, inputs=[self.witness.valid[candidate]],
                      device=self.device, stream=stream)

    def _stamp(self, stage, status_arr, active_arr, accel=False, stream=None):
        wp.launch(epoch.finish_stage, dim=self.capacity,
                  inputs=[self.active, active_arr, status_arr, self.witness.scratch_epoch,
                          self.witness.stamps[stage], self.witness.versions[stage],
                          self.witness.valid[stage], self.witness.first_error_stage,
                          self.witness.first_error_code, STAGE_ORDER.index(stage)+1],
                  device=self.device, stream=stream)

    def capture_input_witness(self):
        """Boundary-only exact byte snapshots of every declared dynamic edge."""
        self.validate_bindings()
        return {stage: {name: np.array(arr.numpy(), copy=True) for name,arr in fields.items()}
                for stage,fields in self.consumer_witness.items()}

    def gate_arrays(self):
        return {name: np.array(getattr(self, name).numpy(), copy=True)
                for name in ('gate_position', 'gate_velocity', 'gate_bias', 'gate_control',
                             'gate_actuation', 'gate_acceleration')}

    def repair_flags(self):
        """R02-N01: per-world pivot-repair visibility for both factor/solves."""
        return {
            'position_warning_count':
                np.array(self.position.mass.mass.warning_count.numpy(), copy=True),
            'position_warning_last':
                np.array(self.position.mass.mass.warning_last.numpy(), copy=True),
            'acceleration_warning_count':
                np.array(self.acceleration.solver.warning_count.numpy(), copy=True),
            'acceleration_warning_last':
                np.array(self.acceleration.solver.warning_last.numpy(), copy=True),
            'acceleration_pivot':
                np.array(self.acceleration.solver.pivot.numpy(), copy=True),
        }

    # ----- the six entries ---------------------------------------------
    def launch_position(self, stream=None):
        """D04 chain + transmission length/moment; writes the position group."""
        self.validate_bindings()
        stream = self._checked_stream(stream)
        self._stage_gate('position', self.gate_position, self.status_position, stream)
        self.position.launch(stream=stream)
        self.transmission.launch_position(stream=stream)
        wp.launch(_map_position_status, dim=self.capacity,
                  inputs=[self.position.result_kind, self.position.first_code,
                          self.transmission.position_status,
                          self.status_position],
                  device=self.device, stream=stream)
        self._stamp('position', self.status_position, self.gate_position, stream=stream)
        self.witness.record('position', 'launch',
                            writes=list(FIELD_GROUPS['position']))

    def launch_velocity_prefix(self, stream=None):
        """transmission velocity + comVel + passive, before S.reference.

        The consumer mask is gate_velocity: without a same-epoch position
        stamp nothing is computed and nothing is stamped (R02-F02)."""
        self.validate_bindings()
        stream = self._checked_stream(stream)
        self._stage_gate('velocity_prefix', self.gate_velocity, self.status_velocity, stream)
        self.transmission.launch_velocity(stream=stream)
        self._comvel_launch(stream=stream)
        self.passive.launch(stream=stream)
        wp.launch(_merge_status3, dim=self.capacity,
                  inputs=[self.transmission.velocity_status,
                          self.status_comvel, self.status_passive,
                          self.status_velocity],
                  device=self.device, stream=stream)
        self._stamp('velocity_prefix', self.status_velocity, self.gate_velocity, stream=stream)
        self.witness.record('velocity_prefix', 'launch',
                            reads=['position'],
                            writes=list(FIELD_GROUPS['velocity_prefix']))

    def launch_bias(self, stream=None):
        """RNE bias; S.reference must already have run at its source position."""
        self.validate_bindings()
        stream = self._checked_stream(stream)
        self._stage_gate('bias', self.gate_bias, self.status_bias, stream)
        self._rne_launch(stream=stream)
        wp.launch(_copy_status, dim=self.capacity,
                  inputs=[self.status_rne, self.status_bias],
                  device=self.device, stream=stream)
        self._stamp('bias', self.status_bias, self.gate_bias, stream=stream)
        self.witness.record('bias', 'launch', reads=list(STAGE_READS['bias']),
                            writes=list(FIELD_GROUPS['bias']))

    def set_initial_joints(self, q0):
        """Reset-pose input for the nullspace PD term (source: initial_joint)."""
        self.validate_bindings()
        arr = np.asarray(q0)
        if arr.shape != (self.capacity, 7) or arr.dtype != np.float64:
            raise ValueError('initial joints must be float64 (N, 7)')
        with wp.ScopedStream(self._checked_stream(None)):
            self.control.q0.assign(np.ascontiguousarray(arr))
        self._input_written(1)

    def stage_action(self, action):
        """Explicit host-input boundary (R03-F11): policy action (N, 7).

        Must run outside any CUDA Graph capture region; launch_control itself
        performs no H2D/D2H and stays capturable unconditionally."""
        self.validate_bindings()
        arr = np.asarray(action)
        if arr.shape != (self.capacity, 7) or arr.dtype != np.float64:
            raise ValueError('action must be float64 (N, 7)')
        with wp.ScopedStream(self._checked_stream(None)):
            self.control.action.assign(np.ascontiguousarray(arr))
        self._input_written(0)

    def stage_gripper(self, gripper):
        """Explicit host-input boundary: normalized gripper state (N, 2).

        This sets the normalized persistent current_action input. The source
        controller accumulates action[:, 6] on every successful control call;
        its device clock records that event, including Graph replay."""
        self.validate_bindings()
        arr = np.asarray(gripper)
        if arr.shape != (self.capacity, 2) or arr.dtype != np.float64:
            raise ValueError('gripper must be float64 (N, 2)')
        with wp.ScopedStream(self._checked_stream(None)):
            self.control.gripper.assign(np.ascontiguousarray(arr))
        self._input_written(2)

    def _input_written(self, slot):
        wp.launch(epoch.bump_input, dim=self.capacity,
                  inputs=[self.input_versions, self.witness.first_error_stage,
                          self.witness.first_error_code, slot], device=self.device,
                  stream=self._checked_stream(None))

    def launch_control(self, *, force_refresh: bool, policy_step: bool = False,
                       stream=None):
        """D07 Jacobian/OSC inputs + controller; two entries: refresh/skip.

        force_refresh=True runs the full cache refresh (source
        base_controller.py:125-161 update branch); False preserves the
        previous pose/velocity/Jacobian/mass caches exactly.  policy_step
        drives goal updates from the staged action (source: set_goal only at
        policy_step); the live qfrc_bias edge (phys_bias) is clocked on every
        control call (R02-F04).  Host inputs are staged beforehand via
        stage_action/stage_gripper/set_initial_joints; this launch path is
        device-only (kernels + memsets) and CUDA-Graph capturable (R03-F11)."""
        self.validate_bindings()
        stream = self._checked_stream(stream)
        self._stage_gate('control', self.gate_control, self.status_control, stream)
        wp.launch(epoch.clock_gate, dim=self.capacity,
                  inputs=[self.gate_control, self.status_control, self.device_clock,
                          self.witness.first_error_stage, self.witness.first_error_code,
                          int(bool(force_refresh)), int(bool(policy_step)),
                          self.control_consumption.adapter_version],
                  device=self.device, stream=stream)
        wp.copy(self.witness.read_mask['control'], self.gate_control, stream=stream)
        for name,previous in self.control_previous.items():
            wp.copy(previous, getattr(self.control,name), stream=stream)
        wp.launch(_gather_eef_site, dim=self.capacity,
                  inputs=[self.site_xpos_flat, self.site_xmat_flat,
                          self.gate_control, self.d07.osc.site_pos,
                          self.d07.osc.site_mat, 31, EEF_SITE_ID],
                  device=self.device, stream=stream)
        self.d07.launch(stream=stream)
        mode = self.clock.on_update(force_refresh=force_refresh)
        with wp.ScopedStream(stream):
            self.force_refresh.fill_(1 if force_refresh else 0)
        wp.launch(_before_control, dim=self.capacity,
                  inputs=[self.d07.osc.status, self.gate_control,
                          self.force_refresh,
                          self.d07.osc.xvelp, self.d07.osc.xvelr,
                          self.control.phys_ee_velocity, self.control_active,
                          self.status_control, self.control.needs_update],
                  device=self.device, stream=stream)
        with wp.ScopedStream(stream):
            self.policy_step.fill_(1 if policy_step else 0)
        if policy_step:
            self.clock.on_set_goal()
        self.control_consumption.launch(self, sys.modules["p06d_osc_ctrl"], stream)
        # These domains are written on every successful active controller call.
        # Conditional quaternion/cache hold domains are not included here.
        for name in ('ctrl', 'torques_raw', 'torques_clipped', 'mass_inv', 'A_full', 'A_pos',
                     'A_ori', 'lambda_full', 'lambda_pos', 'lambda_ori', 'nullspace_matrix',
                     'position_error', 'orientation_error', 'nullspace_torque', 'gripper'):
            wp.launch(epoch.written_finite, dim=self.capacity,
                      inputs=[self.control_active, self.control.error, getattr(self.control,name)],
                      device=self.device, stream=stream)
        wp.launch(_after_control, dim=self.capacity,
                  inputs=[self.status_control, self.control.error,
                          self.control_active],
                  device=self.device, stream=stream)
        self.clock.on_bias(live=True)
        self._stamp('control', self.status_control, self.gate_control, stream=stream)
        for name,current in self.control_bytes.items():
            wp.launch(epoch.restore_failed, dim=current.shape,
                      inputs=[self.gate_control, self.status_control,
                              self.control_previous_bytes[name], current],
                      device=self.device, stream=stream)
        wp.launch(epoch.record_clock, dim=self.capacity,
                  inputs=[self.gate_control, self.status_control, self.force_refresh,
                          self.policy_step, self.witness.scratch_epoch,
                          self.witness.versions['bias'], self.device_clock,
                          self.cache_forward, self.bias_read_version],
                  device=self.device, stream=stream)
        self.control_consumption.finish(self, stream)
        self.witness.record('control', 'launch',
                            refresh=mode,
                            cache_version=self.clock.cache_version,
                            policy_step=bool(policy_step))

    def capture_control_witness(self):
        """Explicit boundary read (R03-F11): the bias generation consumed by
        the last control call, per world.  Never called inside a
        launch/capture region."""
        self.validate_bindings()
        gens = np.array(self.bias_read_version.numpy(), copy=True)
        return {'actual_bias_writer_versions': gens.tolist(),
                'actual_device_clock': np.array(self.device_clock.numpy(), copy=True).tolist(),
                'actual_cache_forward': np.array(self.cache_forward.numpy(), copy=True).tolist()}

    def launch_actuation(self, stream=None):
        """actuator_force/qfrc_actuator from staged transmission + live ctrl.

        actuation_active (the force kernel's active mask) is 1 only for
        gated worlds whose control stage succeeded (R02-F03)."""
        self.validate_bindings()
        stream = self._checked_stream(stream)
        self._stage_gate('actuation', self.gate_actuation, self.status_actuation, stream)
        wp.launch(_before_actuation, dim=self.capacity,
                  inputs=[self.status_control, self.control_active,
                          self.control.error, self.gate_actuation,
                          self.actuation_active, self.status_actuation],
                  device=self.device, stream=stream)
        self.actuation.launch(self.transmission.length, self.transmission.moment,
                              self.transmission.velocity,
                              self.transmission.velocity_status, stream=stream)
        wp.launch(_after_actuation, dim=self.capacity,
                  inputs=[self.actuation.status, self.status_actuation],
                  device=self.device, stream=stream)
        self._stamp('actuation', self.status_actuation, self.actuation_active, stream=stream)
        self.clock.on_ctrl_write()
        self.witness.record('actuation', 'launch',
                            writes=list(FIELD_GROUPS['actuation']),
                            ctrl_version=self.clock.ctrl_version)

    def launch_acceleration(self, stream=None):
        """three-RN -> seeded D02 accumulation -> N06 factor/solve.

        The gate requires actuation validity (and transitively the whole
        upstream chain); a failed upstream never computes and never stamps
        (R02-F02/F03 new-risk check)."""
        self.validate_bindings()
        stream = self._checked_stream(stream)
        self._stage_gate('acceleration', self.gate_acceleration, self.acceleration.status, stream)
        self.acceleration.launch(stream=stream)
        self._stamp('acceleration', self.acceleration.status,
                    self.gate_acceleration, accel=True, stream=stream)
        self.witness.record('acceleration', 'launch',
                            writes=list(FIELD_GROUPS['acceleration']))

    # ----- accepted submodule launch shims ------------------------------
    def _comvel_launch(self, stream=None):
        comvel = self.comvel_model
        mod = sys.modules['p06d_comvel']
        mod.launch(comvel, self.gate_velocity, self.position.mass.cdof,
                   self.qvel, self.cvel, self.cdof_dot, self.status_comvel,
                   stream)

    def _rne_launch(self, stream=None):
        mod = sys.modules['p06d_rne_bias']
        mod.launch(self.rne_model, self.gate_bias, self.cdof_dot, self.qvel,
                   self.position.mass.cinert, self.cvel, self.position.mass.cdof,
                   self.cacc, self.frc, self.qfrc_bias, self.status_rne, stream)
