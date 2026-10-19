"""Named ctypes mirror of cuda/step/substep_types.cuh.

This file is maintained source. No schema discovery or CUDA source generation.
"""
import ctypes as ct

class FoundationGateData(ct.Structure):
    _fields_ = [
        ("admission", ct.c_void_p),
        ("count", ct.c_void_p),
        ("epoch", ct.c_void_p),
        ("expected", ct.c_void_p),
        ("producer_valid", ct.c_void_p),
        ("event", ct.c_void_p),
        ("errors", ct.c_void_p),
        ("stage_out", ct.c_void_p),
        ("local_out", ct.c_void_p),
        ("item_out", ct.c_void_p),
        ("event_out", ct.c_void_p),
        ("active", ct.c_void_p),
        ("valid", ct.c_void_p),
        ("capacity", ct.c_int32),
        ("stage", ct.c_int32),
        ("admitted", ct.c_void_p),
        ("public_error", ct.c_void_p),
        ("public_stage", ct.c_void_p),
        ("public_local", ct.c_void_p),
        ("public_item", ct.c_void_p),
        ("public_event", ct.c_void_p),
        ("world", ct.c_int32),
    ]


class BeginData(ct.Structure):
    _fields_ = [
        ("mask", ct.c_void_p),
        ("error", ct.c_void_p),
        ("state", ct.c_void_p),
        ("source", ct.c_void_p),
        ("epoch", ct.c_void_p),
        ("active", ct.c_void_p),
        ("position", ct.c_void_p),
        ("velocity", ct.c_void_p),
        ("bias", ct.c_void_p),
        ("smooth", ct.c_void_p),
        ("expected", ct.c_void_p),
    ]


class VelocityGateData(ct.Structure):
    _fields_ = [
        ("active", ct.c_void_p),
        ("error", ct.c_void_p),
        ("state", ct.c_void_p),
        ("source", ct.c_void_p),
        ("epoch", ct.c_void_p),
        ("valid", ct.c_void_p),
        ("written", ct.c_void_p),
        ("gate", ct.c_void_p),
        ("require", ct.c_int32),
    ]


class TransmissionData(ct.Structure):
    _fields_ = [
        ("moment", ct.c_void_p),
        ("qvel", ct.c_void_p),
        ("active", ct.c_void_p),
        ("position_status", ct.c_void_p),
        ("nv", ct.c_int32),
        ("nu", ct.c_int32),
        ("velocity", ct.c_void_p),
        ("status", ct.c_void_p),
    ]


class BodyVelocityData(ct.Structure):
    _fields_ = [
        ("active", ct.c_void_p),
        ("cdof", ct.c_void_p),
        ("qvel", ct.c_void_p),
        ("bp", ct.c_void_p),
        ("ba", ct.c_void_p),
        ("bn", ct.c_void_p),
        ("dj", ct.c_void_p),
        ("jt", ct.c_void_p),
        ("nb", ct.c_int32),
        ("nv", ct.c_int32),
        ("cvel", ct.c_void_p),
        ("cdot", ct.c_void_p),
        ("status", ct.c_void_p),
        ("cinert", ct.c_void_p),
        ("gravity", ct.c_void_p),
        ("disabled", ct.c_int32),
        ("cache_acc", ct.c_void_p),
        ("cache_force", ct.c_void_p),
        ("cache_status", ct.c_void_p),
        ("cache_valid", ct.c_void_p),
        ("cache_state", ct.c_void_p),
        ("cache_forward", ct.c_void_p),
        ("state_epoch", ct.c_void_p),
        ("forward_epoch", ct.c_void_p),
        ("level_offsets", ct.c_void_p),
        ("level_bodies", ct.c_void_p),
        ("nlevels", ct.c_int32),
    ]


class JointPassiveData(ct.Structure):
    _fields_ = [
        ("qpos", ct.c_void_p),
        ("qvel", ct.c_void_p),
        ("active", ct.c_void_p),
        ("jtype", ct.c_void_p),
        ("qadr", ct.c_void_p),
        ("dadr", ct.c_void_p),
        ("stiffness", ct.c_void_p),
        ("spring", ct.c_void_p),
        ("damping", ct.c_void_p),
        ("out", ct.c_void_p),
        ("status", ct.c_void_p),
        ("nv", ct.c_int32),
        ("nq", ct.c_int32),
        ("njnt", ct.c_int32),
        ("joint_disjoint", ct.c_int32),
        ("joint_type_offsets", ct.c_void_p),
        ("joint_type_indices", ct.c_void_p),
    ]


class FluidPassiveData(ct.Structure):
    _fields_ = [
        ("cvel", ct.c_void_p),
        ("subtree_com", ct.c_void_p),
        ("xipos", ct.c_void_p),
        ("ximat", ct.c_void_p),
        ("cdof", ct.c_void_p),
        ("active", ct.c_void_p),
        ("rootid", ct.c_void_p),
        ("mass", ct.c_void_p),
        ("box", ct.c_void_p),
        ("ancestor", ct.c_void_p),
        ("wind", ct.c_void_p),
        ("opt", ct.c_void_p),
        ("passive_disabled", ct.c_int32),
        ("diagnostic", ct.c_int32),
        ("stages", ct.c_void_p),
        ("qfrc", ct.c_void_p),
        ("status", ct.c_void_p),
        ("nbody", ct.c_int32),
        ("nv", ct.c_int32),
        ("body_dof_offsets", ct.c_void_p),
        ("body_dof_indices", ct.c_void_p),
    ]


class VelocityStatusData(ct.Structure):
    _fields_ = [
        ("a", ct.c_void_p),
        ("b", ct.c_void_p),
        ("c", ct.c_void_p),
        ("out", ct.c_void_p),
    ]


class VelocityFinishData(ct.Structure):
    _fields_ = [
        ("gate", ct.c_void_p),
        ("status", ct.c_void_p),
        ("epoch", ct.c_void_p),
        ("valid", ct.c_void_p),
        ("written", ct.c_void_p),
        ("error", ct.c_void_p),
        ("repair", ct.c_int32),
        ("code", ct.c_int32),
    ]


class ReferenceAdmissionData(ct.Structure):
    _fields_ = [
        ("valid", ct.c_void_p),
        ("written", ct.c_void_p),
        ("expected", ct.c_void_p),
        ("first", ct.c_void_p),
        ("active", ct.c_void_p),
    ]


class ReferenceGateData(ct.Structure):
    _fields_ = [
        ("active", ct.c_void_p),
        ("status", ct.c_void_p),
        ("committed", ct.c_void_p),
        ("counts", ct.c_void_p),
        ("qvel", ct.c_void_p),
        ("epoch", ct.c_void_p),
        ("expected", ct.c_void_p),
        ("ready", ct.c_void_p),
        ("rows", ct.c_int32),
        ("nv", ct.c_int32),
    ]


class ConstraintReferenceData(ct.Structure):
    _fields_ = [
        ("committed", ct.c_void_p),
        ("counts", ct.c_void_p),
        ("efc_J", ct.c_void_p),
        ("kbip", ct.c_void_p),
        ("efc_pos", ct.c_void_p),
        ("efc_margin", ct.c_void_p),
        ("qvel", ct.c_void_p),
        ("qvel_epoch", ct.c_void_p),
        ("vel", ct.c_void_p),
        ("aref", ct.c_void_p),
        ("ref_epoch", ct.c_void_p),
        ("rows", ct.c_int32),
        ("nv", ct.c_int32),
    ]


class ReferenceCommitData(ct.Structure):
    _fields_ = [
        ("ready", ct.c_void_p),
        ("status", ct.c_void_p),
        ("committed", ct.c_void_p),
        ("counts", ct.c_void_p),
        ("sv", ct.c_void_p),
        ("sa", ct.c_void_p),
        ("epoch", ct.c_void_p),
        ("pv", ct.c_void_p),
        ("pa", ct.c_void_p),
        ("ref_epoch", ct.c_void_p),
        ("rows", ct.c_int32),
    ]


class BiasGateData(ct.Structure):
    _fields_ = [
        ("active", ct.c_void_p),
        ("error", ct.c_void_p),
        ("state", ct.c_void_p),
        ("source", ct.c_void_p),
        ("epoch", ct.c_void_p),
        ("valid", ct.c_void_p),
        ("written", ct.c_void_p),
        ("gate", ct.c_void_p),
        ("require", ct.c_int32),
    ]


class BiasReferenceData(ct.Structure):
    _fields_ = [
        ("done", ct.c_void_p),
        ("err", ct.c_void_p),
        ("written", ct.c_void_p),
        ("expected", ct.c_void_p),
        ("epoch", ct.c_void_p),
        ("gate", ct.c_void_p),
        ("first", ct.c_void_p),
    ]


class BiasForcesData(ct.Structure):
    _fields_ = [
        ("active", ct.c_void_p),
        ("cdot", ct.c_void_p),
        ("qvel", ct.c_void_p),
        ("cinert", ct.c_void_p),
        ("cvel", ct.c_void_p),
        ("cdof", ct.c_void_p),
        ("bp", ct.c_void_p),
        ("ba", ct.c_void_p),
        ("bn", ct.c_void_p),
        ("db", ct.c_void_p),
        ("gravity", ct.c_void_p),
        ("disabled", ct.c_int32),
        ("nb", ct.c_int32),
        ("nv", ct.c_int32),
        ("cacc", ct.c_void_p),
        ("frc", ct.c_void_p),
        ("result", ct.c_void_p),
        ("status", ct.c_void_p),
        ("cache_acc", ct.c_void_p),
        ("cache_force", ct.c_void_p),
        ("cache_status", ct.c_void_p),
        ("cache_valid", ct.c_void_p),
        ("cache_state", ct.c_void_p),
        ("cache_forward", ct.c_void_p),
        ("state_epoch", ct.c_void_p),
        ("forward_epoch", ct.c_void_p),
        ("level_offsets", ct.c_void_p),
        ("level_bodies", ct.c_void_p),
        ("nlevels", ct.c_int32),
        ("children_offsets", ct.c_void_p),
        ("children_indices", ct.c_void_p),
    ]


class BiasCommitData(ct.Structure):
    _fields_ = [
        ("active", ct.c_void_p),
        ("gate", ct.c_void_p),
        ("status", ct.c_void_p),
        ("epoch", ct.c_void_p),
        ("error", ct.c_void_p),
        ("work", ct.c_void_p),
        ("pub", ct.c_void_p),
        ("valid", ct.c_void_p),
        ("written", ct.c_void_p),
        ("width", ct.c_int32),
    ]


class ActuationGateData(ct.Structure):
    _fields_ = [
        ("active", ct.c_void_p),
        ("error", ct.c_void_p),
        ("state", ct.c_void_p),
        ("source", ct.c_void_p),
        ("epoch", ct.c_void_p),
        ("valid", ct.c_void_p),
        ("written", ct.c_void_p),
        ("gate", ct.c_void_p),
        ("require", ct.c_int32),
    ]


class ControlGateData(ct.Structure):
    _fields_ = [
        ("valid", ct.c_void_p),
        ("gate", ct.c_void_p),
    ]


class ControlCopyData(ct.Structure):
    _fields_ = [
        ("gate", ct.c_void_p),
        ("source", ct.c_void_p),
        ("target", ct.c_void_p),
        ("width", ct.c_int32),
    ]


class ActuationData(ct.Structure):
    _fields_ = [
        ("length", ct.c_void_p),
        ("moment", ct.c_void_p),
        ("velocity", ct.c_void_p),
        ("ctrl", ct.c_void_p),
        ("active", ct.c_void_p),
        ("vel_status", ct.c_void_p),
        ("gain", ct.c_void_p),
        ("bias0", ct.c_void_p),
        ("bias1", ct.c_void_p),
        ("bias2", ct.c_void_p),
        ("bias_type", ct.c_void_p),
        ("ctrl_lo", ct.c_void_p),
        ("ctrl_hi", ct.c_void_p),
        ("force_lo", ct.c_void_p),
        ("force_hi", ct.c_void_p),
        ("force_limited", ct.c_void_p),
        ("disable_flags", ct.c_int32),
        ("force", ct.c_void_p),
        ("force_pre", ct.c_void_p),
        ("ctrl_used", ct.c_void_p),
        ("ctrl_route", ct.c_void_p),
        ("force_route", ct.c_void_p),
        ("qfrc_actuator", ct.c_void_p),
        ("status", ct.c_void_p),
        ("warn_badctrl_count", ct.c_void_p),
        ("warn_badctrl_lastinfo", ct.c_void_p),
        ("warning_number", ct.c_void_p),
        ("warning_lastinfo", ct.c_void_p),
        ("nv", ct.c_int32),
        ("nu", ct.c_int32),
    ]


class ActuationFinishData(ct.Structure):
    _fields_ = [
        ("gate", ct.c_void_p),
        ("status", ct.c_void_p),
        ("first", ct.c_void_p),
    ]


class SmoothGateData(ct.Structure):
    _fields_ = [
        ("active", ct.c_void_p),
        ("error", ct.c_void_p),
        ("state", ct.c_void_p),
        ("source", ct.c_void_p),
        ("epoch", ct.c_void_p),
        ("valid", ct.c_void_p),
        ("written", ct.c_void_p),
        ("gate", ct.c_void_p),
        ("require", ct.c_int32),
    ]


class SmoothForceGateData(ct.Structure):
    _fields_ = [
        ("valid", ct.c_void_p),
        ("gate", ct.c_void_p),
    ]


class SmoothAccelerationData(ct.Structure):
    _fields_ = [
        ("passive", ct.c_void_p),
        ("bias", ct.c_void_p),
        ("applied", ct.c_void_p),
        ("actuator", ct.c_void_p),
        ("active", ct.c_void_p),
        ("entry_status", ct.c_void_p),
        ("smooth", ct.c_void_p),
        ("seed", ct.c_void_p),
        ("nv", ct.c_int32),
        ("xfrc", ct.c_void_p),
        ("xipos", ct.c_void_p),
        ("subtree_com", ct.c_void_p),
        ("cdof", ct.c_void_p),
        ("body_ids", ct.c_void_p),
        ("body_count", ct.c_void_p),
        ("body_rootid", ct.c_void_p),
        ("ancestor", ct.c_void_p),
        ("scratch", ct.c_void_p),
        ("stages", ct.c_void_p),
        ("qfrc", ct.c_void_p),
        ("d02_status", ct.c_void_p),
        ("nbody", ct.c_int32),
        ("qm", ct.c_void_p),
        ("count", ct.c_void_p),
        ("madr", ct.c_void_p),
        ("parent", ct.c_void_p),
        ("ld", ct.c_void_p),
        ("inv", ct.c_void_p),
        ("sqrt_inv", ct.c_void_p),
        ("solution", ct.c_void_p),
        ("pivot", ct.c_void_p),
        ("solve_status", ct.c_void_p),
        ("warning_count", ct.c_void_p),
        ("warning_last", ct.c_void_p),
        ("nm", ct.c_int32),
        ("rhs", ct.c_void_p),
        ("qacc_smooth", ct.c_void_p),
        ("status", ct.c_void_p),
        ("source_ld", ct.c_void_p),
        ("source_inv", ct.c_void_p),
        ("source_sqrt_inv", ct.c_void_p),
        ("source_pivot", ct.c_void_p),
        ("source_status", ct.c_void_p),
        ("source_warning_count", ct.c_void_p),
        ("source_warning_last", ct.c_void_p),
        ("source_active", ct.c_void_p),
        ("position_valid", ct.c_void_p),
        ("position_written", ct.c_void_p),
        ("forward_epoch", ct.c_void_p),
        ("factor_reuse_count", ct.c_void_p),
        ("body_dof_offsets", ct.c_void_p),
        ("body_dof_indices", ct.c_void_p),
    ]


class SmoothCommitData(ct.Structure):
    _fields_ = [
        ("active", ct.c_void_p),
        ("gate", ct.c_void_p),
        ("status", ct.c_void_p),
        ("epoch", ct.c_void_p),
        ("error", ct.c_void_p),
        ("wf", ct.c_void_p),
        ("wa", ct.c_void_p),
        ("pf", ct.c_void_p),
        ("pa", ct.c_void_p),
        ("valid", ct.c_void_p),
        ("written", ct.c_void_p),
        ("repaired", ct.c_void_p),
        ("width", ct.c_int32),
    ]


class NewtonAdmissionData(ct.Structure):
    _fields_ = [
        ("active", ct.c_void_p),
        ("first", ct.c_void_p),
        ("smooth", ct.c_void_p),
        ("reference", ct.c_void_p),
        ("sstatus", ct.c_void_p),
        ("admitted", ct.c_void_p),
        ("error", ct.c_void_p),
    ]


class ConstraintCountsData(ct.Structure):
    _fields_ = [
        ("admitted", ct.c_void_p),
        ("source_counts", ct.c_void_p),
        ("source_ncon", ct.c_void_p),
        ("nefc", ct.c_void_p),
        ("ne", ct.c_void_p),
        ("nf", ct.c_void_p),
        ("ncon", ct.c_void_p),
        ("count", ct.c_void_p),
        ("empty", ct.c_void_p),
        ("newton", ct.c_void_p),
        ("active", ct.c_void_p),
        ("status", ct.c_void_p),
        ("item", ct.c_void_p),
        ("rows", ct.c_int32),
        ("contacts", ct.c_int32),
        ("nv", ct.c_int32),
    ]


class ConeParametersData(ct.Structure):
    _fields_ = [
        ("admitted", ct.c_void_p),
        ("nefc", ct.c_void_p),
        ("ne", ct.c_void_p),
        ("nf", ct.c_void_p),
        ("ncon", ct.c_void_p),
        ("typ", ct.c_void_p),
        ("id", ct.c_void_p),
        ("dim", ct.c_void_p),
        ("mu", ct.c_void_p),
        ("friction", ct.c_void_p),
        ("out_dim", ct.c_void_p),
        ("out_mu", ct.c_void_p),
        ("out_friction", ct.c_void_p),
        ("status", ct.c_void_p),
        ("item", ct.c_void_p),
        ("rows", ct.c_int32),
        ("contacts", ct.c_int32),
        ("compact_heads", ct.c_void_p),
        ("compact_count", ct.c_void_p),
        ("compact_valid", ct.c_void_p),
        ("head_dim", ct.c_void_p),
        ("head_mu", ct.c_void_p),
        ("head_friction", ct.c_void_p),
        ("head_cone_valid", ct.c_void_p),
    ]


class NewtonEpochData(ct.Structure):
    _fields_ = [
        ("mask", ct.c_void_p),
        ("first", ct.c_void_p),
        ("epoch", ct.c_void_p),
    ]


class ForwardPublishData(ct.Structure):
    _fields_ = [
        ("active", ct.c_void_p),
        ("nvalid", ct.c_void_p),
        ("nerror", ct.c_void_p),
        ("first", ct.c_void_p),
        ("projection", ct.c_void_p),
        ("source", ct.c_void_p),
        ("state", ct.c_void_p),
        ("valid", ct.c_void_p),
        ("written", ct.c_void_p),
        ("error", ct.c_void_p),
    ]


class ControllerPlanData(ct.Structure):
    _fields_ = [
        ("active", ct.c_void_p),
        ("first", ct.c_void_p),
        ("needs", ct.c_void_p),
        ("refresh", ct.c_void_p),
        ("call", ct.c_void_p),
        ("force", ct.c_int32),
    ]


class GatherGateData(ct.Structure):
    _fields_ = [
        ("mask", ct.c_void_p),
        ("error", ct.c_void_p),
        ("state", ct.c_void_p),
        ("source", ct.c_void_p),
        ("epoch", ct.c_void_p),
        ("valid", ct.c_void_p),
        ("written", ct.c_void_p),
        ("active", ct.c_void_p),
    ]


class EndEffectorPoseData(ct.Structure):
    _fields_ = [
        ("pos", ct.c_void_p),
        ("mat", ct.c_void_p),
        ("active", ct.c_void_p),
        ("outp", ct.c_void_p),
        ("outm", ct.c_void_p),
        ("ns", ct.c_int32),
        ("id", ct.c_int32),
    ]


class ControllerInputsData(ct.Structure):
    _fields_ = [
        ("cdof", ct.c_void_p),
        ("subtree", ct.c_void_p),
        ("pos", ct.c_void_p),
        ("active", ct.c_void_p),
        ("ancestor", ct.c_void_p),
        ("root", ct.c_int32),
        ("body", ct.c_int32),
        ("nv", ct.c_int32),
        ("nb", ct.c_int32),
        ("jp", ct.c_void_p),
        ("jr", ct.c_void_p),
        ("jac_status", ct.c_void_p),
        ("adapter_active", ct.c_void_p),
        ("qpos", ct.c_void_p),
        ("qvel", ct.c_void_p),
        ("rot", ct.c_void_p),
        ("qm", ct.c_void_p),
        ("qbias", ct.c_void_p),
        ("qi", ct.c_void_p),
        ("vi", ct.c_void_p),
        ("sparse", ct.c_void_p),
        ("nq", ct.c_int32),
        ("nm", ct.c_int32),
        ("ep", ct.c_void_p),
        ("er", ct.c_void_p),
        ("vp", ct.c_void_p),
        ("vo", ct.c_void_p),
        ("q", ct.c_void_p),
        ("qd", ct.c_void_p),
        ("J", ct.c_void_p),
        ("M", ct.c_void_p),
        ("bias", ct.c_void_p),
        ("adapter_status", ct.c_void_p),
        ("out_status", ct.c_void_p),
        ("body_dof_offsets", ct.c_void_p),
        ("body_dof_indices", ct.c_void_p),
    ]


class EndEffectorVelocityData(ct.Structure):
    _fields_ = [
        ("active", ct.c_void_p),
        ("status", ct.c_void_p),
        ("vp", ct.c_void_p),
        ("vo", ct.c_void_p),
        ("target", ct.c_void_p),
        ("first", ct.c_void_p),
    ]


class ControllerBiasData(ct.Structure):
    _fields_ = [
        ("mask", ct.c_void_p),
        ("first", ct.c_void_p),
        ("state", ct.c_void_p),
        ("source", ct.c_void_p),
        ("epoch", ct.c_void_p),
        ("valid", ct.c_void_p),
        ("written", ct.c_void_p),
        ("bias", ct.c_void_p),
        ("selected", ct.c_void_p),
        ("indices", ct.c_void_p),
        ("parent_valid", ct.c_void_p),
        ("parent_written", ct.c_void_p),
        ("nv", ct.c_int32),
    ]


class ControllerGateData(ct.Structure):
    _fields_ = [
        ("active", ct.c_void_p),
        ("pv", ct.c_void_p),
        ("written", ct.c_void_p),
        ("expected", ct.c_void_p),
        ("first", ct.c_void_p),
        ("error", ct.c_void_p),
        ("policy", ct.c_void_p),
        ("step", ct.c_int32),
    ]


class ControllerSolveData(ct.Structure):
    _fields_ = [
        ("mass", ct.c_void_p),
        ("jacobian", ct.c_void_p),
        ("q", ct.c_void_p),
        ("qvel", ct.c_void_p),
        ("bias", ct.c_void_p),
        ("q0", ct.c_void_p),
        ("ee_pos", ct.c_void_p),
        ("ee_rot", ct.c_void_p),
        ("ee_velocity", ct.c_void_p),
        ("action", ct.c_void_p),
        ("goal_pos", ct.c_void_p),
        ("goal_rot", ct.c_void_p),
        ("gripper", ct.c_void_p),
        ("active", ct.c_void_p),
        ("gains", ct.c_void_p),
        ("range", ct.c_void_p),
        ("ctrl", ct.c_void_p),
        ("raw_tau", ct.c_void_p),
        ("error", ct.c_void_p),
        ("ranks", ct.c_void_p),
        ("policy", ct.c_void_p),
        ("needs", ct.c_void_p),
        ("mass_inv", ct.c_void_p),
        ("A_full", ct.c_void_p),
        ("A_pos", ct.c_void_p),
        ("A_ori", ct.c_void_p),
        ("lambda_full", ct.c_void_p),
        ("lambda_pos", ct.c_void_p),
        ("lambda_ori", ct.c_void_p),
        ("nullspace_matrix", ct.c_void_p),
        ("position_error", ct.c_void_p),
        ("orientation_error", ct.c_void_p),
        ("quat_input", ct.c_void_p),
        ("quat_cast", ct.c_void_p),
        ("quat_result", ct.c_void_p),
        ("quat_valid", ct.c_void_p),
        ("nullspace_torque", ct.c_void_p),
        ("clipped", ct.c_void_p),
        ("diagnostics", ct.c_int32),
        ("uncouple", ct.c_int32),
        ("derived_valid", ct.c_void_p),
        ("derived_key", ct.c_void_p),
        ("derived_data", ct.c_void_p),
        ("derived_ranks", ct.c_void_p),
        ("cache_hits", ct.c_void_p),
        ("pm", ct.c_void_p),
        ("pj", ct.c_void_p),
        ("pq", ct.c_void_p),
        ("pv", ct.c_void_p),
        ("pp", ct.c_void_p),
        ("pr", ct.c_void_p),
        ("pe", ct.c_void_p),
        ("cache_valid", ct.c_void_p),
    ]


class ControllerCommitData(ct.Structure):
    _fields_ = [
        ("active", ct.c_void_p),
        ("error", ct.c_void_p),
        ("first", ct.c_void_p),
        ("commit", ct.c_void_p),
        ("event", ct.c_void_p),
        ("valid", ct.c_void_p),
    ]


class ControllerPublishData(ct.Structure):
    _fields_ = [
        ("mask", ct.c_void_p),
        ("event", ct.c_void_p),
        ("parent", ct.c_void_p),
        ("written", ct.c_void_p),
        ("consumed", ct.c_void_p),
        ("valid", ct.c_void_p),
    ]


class EulerData(ct.Structure):
    _fields_ = [
        ("active", ct.c_void_p),
        ("qp", ct.c_void_p),
        ("qv", ct.c_void_p),
        ("time", ct.c_void_p),
        ("qm", ct.c_void_p),
        ("smooth", ct.c_void_p),
        ("constraint", ct.c_void_p),
        ("acc", ct.c_void_p),
        ("damping", ct.c_void_p),
        ("madr", ct.c_void_p),
        ("mq", ct.c_void_p),
        ("rhs", ct.c_void_p),
        ("count", ct.c_void_p),
        ("ma", ct.c_void_p),
        ("damp", ct.c_void_p),
        ("status", ct.c_void_p),
        ("dt", ct.c_double),
        ("nq", ct.c_int32),
        ("nv", ct.c_int32),
        ("nm", ct.c_int32),
        ("parent", ct.c_void_p),
        ("ld", ct.c_void_p),
        ("inv", ct.c_void_p),
        ("si", ct.c_void_p),
        ("solution", ct.c_void_p),
        ("pivot", ct.c_void_p),
        ("ms", ct.c_void_p),
        ("wc", ct.c_void_p),
        ("wl", ct.c_void_p),
        ("simple", ct.c_void_p),
        ("work", ct.c_void_p),
        ("skip", ct.c_void_p),
        ("jt", ct.c_void_p),
        ("padr", ct.c_void_p),
        ("vadr", ct.c_void_p),
        ("sp", ct.c_void_p),
        ("sv", ct.c_void_p),
        ("used", ct.c_void_p),
        ("st", ct.c_void_p),
        ("axis", ct.c_void_p),
        ("angle", ct.c_void_p),
        ("increment", ct.c_void_p),
        ("pre", ct.c_void_p),
        ("af", ct.c_void_p),
        ("za", ct.c_void_p),
        ("qf", ct.c_void_p),
        ("nj", ct.c_int32),
        ("nq4", ct.c_int32),
        ("qh", ct.c_void_p),
        ("qhi", ct.c_void_p),
        ("qacc_stride", ct.c_int32),
        ("joint_quat", ct.c_void_p),
        ("joint_type_offsets", ct.c_void_p),
        ("joint_type_indices", ct.c_void_p),
    ]


class DoubleChannel(ct.Structure):
    _fields_ = [
        ("published", ct.c_void_p),
        ("working", ct.c_void_p),
        ("width", ct.c_int32),
        ("parallel_copy", ct.c_int32),
    ]


class IntChannel(ct.Structure):
    _fields_ = [
        ("published", ct.c_void_p),
        ("working", ct.c_void_p),
    ]


class EpochCopy(ct.Structure):
    _fields_ = [
        ("source", ct.c_void_p),
        ("destination", ct.c_void_p),
        ("width", ct.c_int32),
    ]


class ResetField(ct.Structure):
    _fields_ = [
        ("values", ct.c_void_p),
        ("defaults", ct.c_void_p),
        ("width", ct.c_int32),
    ]


class StateData(ct.Structure):
    _fields_ = [
        ("intent", ct.c_void_p),
        ("forward_mask", ct.c_void_p),
        ("requested_mask", ct.c_void_p),
        ("first_error", ct.c_void_p),
        ("epoch", ct.c_void_p),
        ("expected_epoch", ct.c_void_p),
        ("forward_valid", ct.c_void_p),
        ("forward_written", ct.c_void_p),
        ("forward_error", ct.c_void_p),
        ("step_valid", ct.c_void_p),
        ("step_written", ct.c_void_p),
        ("integration_count", ct.c_void_p),
        ("qpos", ct.c_void_p),
        ("qvel", ct.c_void_p),
        ("qacc", ct.c_void_p),
        ("nq", ct.c_int32),
        ("nv", ct.c_int32),
    ]


class PositionCacheData(ct.Structure):
    _fields_ = [
        ("saved", ct.c_void_p),
        ("cached", ct.c_void_p),
        ("constraint_status", ct.c_void_p),
        ("constraint_committed", ct.c_void_p),
        ("collision_error", ct.c_void_p),
        ("mass_warning", ct.c_void_p),
        ("collision_warning", ct.c_void_p),
        ("work", ct.c_void_p),
        ("hit", ct.c_void_p),
        ("written", ct.c_void_p),
        ("consumed", ct.c_void_p),
        ("transport_active", ct.c_void_p),
        ("collision_mask", ct.c_void_p),
    ]


class RecoveryData(ct.Structure):
    _fields_ = [
        ("active", ct.c_void_p),
        ("reset_mask", ct.c_void_p),
        ("reason", ct.c_void_p),
        ("bad_index", ct.c_void_p),
        ("source_error", ct.c_void_p),
        ("warning_number", ct.c_void_p),
        ("warning_lastinfo", ct.c_void_p),
        ("reset_events", ct.c_void_p),
        ("fields", ResetField * 14),
        ("field_count", ct.c_int32),
    ]


class ContactAddressData(ct.Structure):
    _fields_ = [
        ("active", ct.c_void_p),
        ("status", ct.c_void_p),
        ("committed", ct.c_void_p),
        ("count", ct.c_void_p),
        ("excluded", ct.c_void_p),
        ("address", ct.c_void_p),
        ("current", ct.c_void_p),
        ("capacity", ct.c_int32),
    ]


class NewtonData(ct.Structure):
    _fields_ = [
        ("parameters", ct.c_void_p),
        ("upstream", ct.c_void_p),
        ("enabled", ct.c_void_p),
        ("publication", ct.c_void_p),
        ("expected_epoch", ct.c_void_p),
        ("valid", ct.c_void_p),
        ("written", ct.c_void_p),
        ("stop_policy", ct.c_int32),
        ("audit", ct.c_void_p),
        ("first_disagreement", ct.c_void_p),
    ]


class ControllerCopyPacket(ct.Structure):
    _fields_ = [
        ("published", ct.c_void_p), ("working", ct.c_void_p),
        ("width", ct.c_int32), ("words", ct.c_int32),
    ]


class ControllerChannels(ct.Structure):
    _fields_ = [
        ("floating", DoubleChannel * 26),
        ("integer", IntChannel * 2),
        ("floating_count", ct.c_int32),
        ("integer_count", ct.c_int32),
        ("copy_packets", ct.c_void_p),
        ("seed_packets", ct.c_int32),
        ("commit_packets", ct.c_int32),
    ]


FORWARD_CACHE_WORDS = 6


class ForwardWorkspaceData(ct.Structure):
    _fields_ = [
        ("metadata", ct.c_void_p),
        ("position_generation", ct.c_void_p),
        ("fluid", ct.c_void_p),
        ("mass_plan", ct.c_void_p),
    ]


class SubstepArguments(ct.Structure):
    _fields_ = [
        ("foundation_gate", FoundationGateData),
        ("begin", BeginData),
        ("velocity_gate", VelocityGateData),
        ("transmission", TransmissionData),
        ("body_velocity", BodyVelocityData),
        ("joint_passive", JointPassiveData),
        ("fluid_passive", FluidPassiveData),
        ("velocity_status", VelocityStatusData),
        ("velocity_finish", VelocityFinishData),
        ("reference_admission", ReferenceAdmissionData),
        ("reference_gate", ReferenceGateData),
        ("constraint_reference", ConstraintReferenceData),
        ("reference_commit", ReferenceCommitData),
        ("bias_gate", BiasGateData),
        ("bias_reference", BiasReferenceData),
        ("bias_forces", BiasForcesData),
        ("bias_commit", BiasCommitData),
        ("actuation_gate", ActuationGateData),
        ("control_gate", ControlGateData),
        ("control_copy", ControlCopyData),
        ("actuation", ActuationData),
        ("actuation_finish", ActuationFinishData),
        ("smooth_gate", SmoothGateData),
        ("smooth_force_gate", SmoothForceGateData),
        ("smooth_acceleration", SmoothAccelerationData),
        ("smooth_commit", SmoothCommitData),
        ("newton_admission", NewtonAdmissionData),
        ("constraint_counts", ConstraintCountsData),
        ("cone_parameters", ConeParametersData),
        ("newton_epoch", NewtonEpochData),
        ("forward_publish", ForwardPublishData),
        ("controller_plan", ControllerPlanData),
        ("gather_gate", GatherGateData),
        ("end_effector_pose", EndEffectorPoseData),
        ("controller_inputs", ControllerInputsData),
        ("end_effector_velocity", EndEffectorVelocityData),
        ("controller_bias", ControllerBiasData),
        ("controller_gate", ControllerGateData),
        ("controller_solve", ControllerSolveData),
        ("controller_commit", ControllerCommitData),
        ("controller_publish", ControllerPublishData),
        ("euler", EulerData),
        ("state", StateData),
        ("position", PositionCacheData),
        ("recovery", RecoveryData),
        ("contact_addresses", ContactAddressData),
        ("newton", NewtonData),
        ("controller_channels", ControllerChannels),
        ("post_epochs", EpochCopy * 3),
        ("continuation", ct.c_void_p),
        ("forward_workspace", ForwardWorkspaceData),
    ]


