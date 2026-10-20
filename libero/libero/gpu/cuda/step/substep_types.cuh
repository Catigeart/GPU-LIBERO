#include "precision.cuh"
// Named inputs for one FP64 physical substep. No control-step loop.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include "substep_shared_layout.cuh"
#include "forward_workspace.cuh"
namespace physics_newton {
struct NewtonArguments;
}
namespace flash_libero {
constexpr int substep_threads = SubstepShape::threads;
// Entry/exit points are physical boundaries requiring a separate parallel prefix.
enum class SubstepEntry : int {
  Start = 0,
  ControllerPosition = 1,
  PhysicsPosition = 2,
  AccelerationReset = 3,
  Complete = 4,
  PositionChecks = 5
};

struct FoundationGateData {
  const int *admission;
  const int *count;
  const int64_t *epoch;
  const int64_t *expected;
  const int *producer_valid;
  const int64_t *event;
  int *errors;
  int *stage_out;
  int *local_out;
  int *item_out;
  int64_t *event_out;
  int *active;
  int *valid;
  int capacity;
  int stage;
  int *admitted;
  int *public_error;
  int *public_stage;
  int *public_local;
  int *public_item;
  int64_t *public_event;
  int world;
};

struct BeginData {
  const int *mask;
  int *error;
  const int64_t *state;
  int64_t *source;
  int64_t *epoch;
  int *active;
  int *position;
  int *velocity;
  int *bias;
  int *smooth;
  int64_t *expected;
};

struct VelocityGateData {
  int *active;
  int *error;
  int64_t *state;
  int64_t *source;
  int64_t *epoch;
  int *valid;
  int64_t *written;
  int *gate;
  int require;
};

struct TransmissionData {
  real_t *moment;
  real_t *qvel;
  int *active;
  int *position_status;
  int nv;
  int nu;
  real_t *velocity;
  int *status;
};

struct BodyVelocityData {
  int *active;
  real_t *cdof;
  real_t *qvel;
  int *bp;
  int *ba;
  int *bn;
  int *dj;
  int *jt;
  int nb;
  int nv;
  real_t *cvel;
  real_t *cdot;
  int *status;
  real_t *cinert;
  real_t *gravity;
  int disabled;
  real_t *cache_acc;
  real_t *cache_force;
  int *cache_status;
  int *cache_valid;
  int64_t *cache_state;
  int64_t *cache_forward;
  const int64_t *state_epoch;
  const int64_t *forward_epoch;
  const int *level_offsets;
  const int *level_bodies;
  int nlevels;
};

struct JointPassiveData {
  real_t *qpos;
  real_t *qvel;
  int *active;
  int *jtype;
  int *qadr;
  int *dadr;
  real_t *stiffness;
  real_t *spring;
  real_t *damping;
  real_t *out;
  int *status;
  int nv;
  int nq;
  int njnt;
  int joint_disjoint;
  const int *joint_type_offsets;
  const int *joint_type_indices;
};

struct FluidPassiveData {
  real_t *cvel;
  real_t *subtree_com;
  real_t *xipos;
  real_t *ximat;
  real_t *cdof;
  int *active;
  int *rootid;
  real_t *mass;
  real_t *box;
  int *ancestor;
  real_t *wind;
  real_t *opt;
  int passive_disabled;
  int diagnostic;
  real_t *stages;
  real_t *qfrc;
  int *status;
  int nbody;
  int nv;
  const int *body_dof_offsets;
  const int *body_dof_indices;
};

struct VelocityStatusData {
  int *a;
  int *b;
  int *c;
  int *out;
};

struct VelocityFinishData {
  int *gate;
  int *status;
  int64_t *epoch;
  int *valid;
  int64_t *written;
  int *error;
  int repair;
  int code;
};

struct ReferenceAdmissionData {
  const int *valid;
  const int64_t *written;
  const int64_t *expected;
  const int *first;
  int *active;
};

struct ReferenceGateData {
  int32_t *active;
  int32_t *status;
  int32_t *committed;
  int32_t *counts;
  real_t *qvel;
  int64_t *epoch;
  int64_t *expected;
  int32_t *ready;
  int32_t rows;
  int32_t nv;
};

struct ConstraintReferenceData {
  int32_t *committed;
  int32_t *counts;
  real_t *efc_J;
  real_t *kbip;
  real_t *efc_pos;
  real_t *efc_margin;
  real_t *qvel;
  int64_t *qvel_epoch;
  real_t *vel;
  real_t *aref;
  int64_t *ref_epoch;
  int32_t rows;
  int32_t nv;
};

struct ReferenceCommitData {
  int32_t *ready;
  int32_t *status;
  int32_t *committed;
  int32_t *counts;
  real_t *sv;
  real_t *sa;
  int64_t *epoch;
  real_t *pv;
  real_t *pa;
  int64_t *ref_epoch;
  int32_t rows;
};

struct BiasGateData {
  int *active;
  int *error;
  int64_t *state;
  int64_t *source;
  int64_t *epoch;
  int *valid;
  int64_t *written;
  int *gate;
  int require;
};

struct BiasReferenceData {
  int *done;
  int *err;
  int64_t *written;
  int64_t *expected;
  int64_t *epoch;
  int *gate;
  int *first;
};

struct BiasForcesData {
  int *active;
  real_t *cdot;
  real_t *qvel;
  real_t *cinert;
  real_t *cvel;
  real_t *cdof;
  int *bp;
  int *ba;
  int *bn;
  int *db;
  real_t *gravity;
  int disabled;
  int nb;
  int nv;
  real_t *cacc;
  real_t *frc;
  real_t *result;
  int *status;
  const real_t *cache_acc;
  const real_t *cache_force;
  const int *cache_status;
  const int *cache_valid;
  const int64_t *cache_state;
  const int64_t *cache_forward;
  const int64_t *state_epoch;
  const int64_t *forward_epoch;
  const int *level_offsets;
  const int *level_bodies;
  int nlevels;
  const int *children_offsets;
  const int *children_indices;
};

struct BiasCommitData {
  int *active;
  int *gate;
  int *status;
  int64_t *epoch;
  int *error;
  real_t *work;
  real_t *pub;
  int *valid;
  int64_t *written;
  int width;
};

struct ActuationGateData {
  int *active;
  int *error;
  int64_t *state;
  int64_t *source;
  int64_t *epoch;
  int *valid;
  int64_t *written;
  int *gate;
  int require;
};

struct ControlGateData {
  int *valid;
  int *gate;
};

struct ControlCopyData {
  int *gate;
  real_t *source;
  real_t *target;
  int width;
};

struct ActuationData {
  real_t *length;
  real_t *moment;
  real_t *velocity;
  real_t *ctrl;
  int *active;
  int *vel_status;
  real_t *gain;
  real_t *bias0;
  real_t *bias1;
  real_t *bias2;
  int *bias_type;
  real_t *ctrl_lo;
  real_t *ctrl_hi;
  real_t *force_lo;
  real_t *force_hi;
  int *force_limited;
  int disable_flags;
  real_t *force;
  real_t *force_pre;
  real_t *ctrl_used;
  int *ctrl_route;
  int *force_route;
  real_t *qfrc_actuator;
  int *status;
  int *warn_badctrl_count;
  int *warn_badctrl_lastinfo;
  int *warning_number;
  int *warning_lastinfo;
  int nv;
  int nu;
};

struct ActuationFinishData {
  int *gate;
  int *status;
  int *first;
};

struct SmoothGateData {
  int *active;
  int *error;
  int64_t *state;
  int64_t *source;
  int64_t *epoch;
  int *valid;
  int64_t *written;
  int *gate;
  int require;
};

struct SmoothForceGateData {
  int *valid;
  int *gate;
};

struct SmoothAccelerationData {
  const real_t *passive;
  const real_t *bias;
  const real_t *applied;
  const real_t *actuator;
  const int *active;
  int *entry_status;
  real_t *smooth;
  real_t *seed;
  int nv;
  const real_t *xfrc;
  const real_t *xipos;
  const real_t *subtree_com;
  const real_t *cdof;
  const int *body_ids;
  const int *body_count;
  const int *body_rootid;
  const int *ancestor;
  real_t *scratch;
  real_t *stages;
  real_t *qfrc;
  int *d02_status;
  int nbody;
  const real_t *qm;
  const int *count;
  const int *madr;
  const int *parent;
  real_t *ld;
  real_t *inv;
  real_t *sqrt_inv;
  real_t *solution;
  int *pivot;
  int *solve_status;
  int *warning_count;
  int *warning_last;
  int nm;
  real_t *rhs;
  real_t *qacc_smooth;
  int *status;
  const real_t *source_ld;
  const real_t *source_inv;
  const real_t *source_sqrt_inv;
  const int *source_pivot;
  const int *source_status;
  const int *source_warning_count;
  const int *source_warning_last;
  const int *source_active;
  const int *position_valid;
  const int64_t *position_written;
  const int64_t *forward_epoch;
  int64_t *factor_reuse_count;
  const int *body_dof_offsets;
  const int *body_dof_indices;
};

struct SmoothCommitData {
  int *active;
  int *gate;
  int *status;
  int64_t *epoch;
  int *error;
  real_t *wf;
  real_t *wa;
  real_t *pf;
  real_t *pa;
  int *valid;
  int64_t *written;
  int *repaired;
  int width;
};

struct NewtonAdmissionData {
  const int *active;
  const int *first;
  const int *smooth;
  const int *reference;
  const int *sstatus;
  int *admitted;
  int *error;
};

struct ConstraintCountsData {
  const int *admitted;
  const int *source_counts;
  const int *source_ncon;
  int *nefc;
  int *ne;
  int *nf;
  int *ncon;
  int *count;
  int *empty;
  int *newton;
  int *active;
  int *status;
  int *item;
  int rows;
  int contacts;
  int nv;
};

struct ConeParametersData {
  const int *admitted;
  const int *nefc;
  const int *ne;
  const int *nf;
  const int *ncon;
  const int *typ;
  const int *id;
  const int *dim;
  const real_t *mu;
  const real_t *friction;
  int *out_dim;
  real_t *out_mu;
  real_t *out_friction;
  int *status;
  int *item;
  int rows;
  int contacts;
  const int *compact_heads;
  const int *compact_count;
  const int *compact_valid;
  const int *head_dim;
  real_t *head_mu;
  real_t *head_friction;
  int *head_cone_valid;
};

struct NewtonEpochData {
  const int *mask;
  const int *first;
  int64_t *epoch;
};

struct ForwardPublishData {
  const int *active;
  const int *nvalid;
  const int *nerror;
  const int *first;
  const int *projection;
  const int64_t *source;
  const int64_t *state;
  int *valid;
  int64_t *written;
  int *error;
};

struct ControllerPlanData {
  int *active;
  int *first;
  int *needs;
  int *refresh;
  int *call;
  int force;
};

struct GatherGateData {
  int32_t *mask;
  int32_t *error;
  const int64_t *state;
  const int64_t *source;
  const int64_t *epoch;
  const int32_t *valid;
  const int64_t *written;
  int32_t *active;
};

struct EndEffectorPoseData {
  const real_t *pos;
  const real_t *mat;
  const int32_t *active;
  real_t *outp;
  real_t *outm;
  int ns;
  int id;
};

struct ControllerInputsData {
  const real_t *cdof;
  const real_t *subtree;
  const real_t *pos;
  const int32_t *active;
  const int32_t *ancestor;
  int root;
  int body;
  int nv;
  int nb;
  real_t *jp;
  real_t *jr;
  int32_t *jac_status;
  int32_t *adapter_active;
  const real_t *qpos;
  const real_t *qvel;
  const real_t *rot;
  const real_t *qm;
  const real_t *qbias;
  const int32_t *qi;
  const int32_t *vi;
  const int32_t *sparse;
  int nq;
  int nm;
  real_t *ep;
  real_t *er;
  real_t *vp;
  real_t *vo;
  real_t *q;
  real_t *qd;
  real_t *J;
  real_t *M;
  real_t *bias;
  int32_t *adapter_status;
  int32_t *out_status;
  const int32_t *body_dof_offsets;
  const int32_t *body_dof_indices;
};

struct EndEffectorVelocityData {
  const int32_t *active;
  const int32_t *status;
  const real_t *vp;
  const real_t *vo;
  real_t *target;
  int32_t *first;
};

struct ControllerBiasData {
  int32_t *mask;
  const int32_t *first;
  const int64_t *state;
  const int64_t *source;
  const int64_t *epoch;
  const int32_t *valid;
  const int64_t *written;
  const real_t *bias;
  real_t *selected;
  const int32_t *indices;
  int32_t *parent_valid;
  int64_t *parent_written;
  int nv;
};

struct ControllerGateData {
  int *active;
  int *pv;
  int64_t *written;
  int64_t *expected;
  int *first;
  int *error;
  int *policy;
  int step;
};

struct ControllerSolveData {
  real_t *mass;
  real_t *jacobian;
  real_t *q;
  real_t *qvel;
  real_t *bias;
  real_t *q0;
  real_t *ee_pos;
  real_t *ee_rot;
  real_t *ee_velocity;
  real_t *action;
  real_t *goal_pos;
  real_t *goal_rot;
  real_t *gripper;
  int *active;
  real_t *gains;
  real_t *range;
  real_t *ctrl;
  real_t *raw_tau;
  int *error;
  int *ranks;
  int *policy;
  int *needs;
  real_t *mass_inv;
  real_t *A_full;
  real_t *A_pos;
  real_t *A_ori;
  real_t *lambda_full;
  real_t *lambda_pos;
  real_t *lambda_ori;
  real_t *nullspace_matrix;
  real_t *position_error;
  real_t *orientation_error;
  real_t *quat_input;
  float *quat_cast;
  real_t *quat_result;
  int *quat_valid;
  real_t *nullspace_torque;
  real_t *clipped;
  int diagnostics;
  int uncouple;
  int *derived_valid;
  real_t *derived_key;
  real_t *derived_data;
  int *derived_ranks;
  int *cache_hits;
  real_t *pm;
  real_t *pj;
  real_t *pq;
  real_t *pv;
  real_t *pp;
  real_t *pr;
  real_t *pe;
  int *cache_valid;
};

struct ControllerCommitData {
  int *active;
  int *error;
  int *first;
  int *commit;
  int64_t *event;
  int *valid;
};

struct ControllerPublishData {
  int *mask;
  int64_t *event;
  int64_t *parent;
  int64_t *written;
  int64_t *consumed;
  int *valid;
};

struct EulerData {
  int *active;
  real_t *qp;
  real_t *qv;
  real_t *time;
  const real_t *qm;
  const real_t *smooth;
  const real_t *constraint;
  const real_t *acc;
  const real_t *damping;
  const int *madr;
  real_t *mq;
  real_t *rhs;
  int *count;
  int *ma;
  int *damp;
  int *status;
  real_t dt;
  int nq;
  int nv;
  int nm;
  const int *parent;
  real_t *ld;
  real_t *inv;
  real_t *si;
  real_t *solution;
  int *pivot;
  int *ms;
  int *wc;
  int *wl;
  const int *simple;
  real_t *work;
  int *skip;
  const int *jt;
  const int *padr;
  const int *vadr;
  real_t *sp;
  real_t *sv;
  real_t *used;
  real_t *st;
  real_t *axis;
  real_t *angle;
  real_t *increment;
  real_t *pre;
  int *af;
  int *za;
  int *qf;
  int nj;
  int nq4;
  real_t *qh;
  real_t *qhi;
  int qacc_stride;
  const int *joint_quat;
  const int *joint_type_offsets;
  const int *joint_type_indices;
};

struct DoubleChannel {
  real_t *published;
  real_t *working;
  int width;
  // Cold binding proves both spans independent of every other channel.
  int parallel_copy;  // bit 0: disjoint copy, bit 1: seed old value
};
static_assert(sizeof(DoubleChannel) == 24, "Keep the channel descriptor ABI stable");

struct IntChannel {
  int *published;
  int *working;
};

struct EpochCopy {
  const int64_t *source;
  int64_t *destination;
  int width;
};

struct ResetField {
  real_t *values;
  const real_t *defaults;
  int width;
};

struct StateData {
  const int *intent;
  int *forward_mask;
  int *requested_mask;
  int *first_error;
  int64_t *epoch;
  int64_t *expected_epoch;
  int *forward_valid;
  int64_t *forward_written;
  int *forward_error;
  int *step_valid;
  int64_t *step_written;
  int64_t *integration_count;
  real_t *qpos;
  real_t *qvel;
  real_t *qacc;
  int nq;
  int nv;
};

struct PositionCacheData {
  const int64_t *saved;
  const int *cached;
  const int *constraint_status;
  const int *constraint_committed;
  const int *collision_error;
  const int *mass_warning;
  const int *collision_warning;
  int *work;
  int *hit;
  int64_t *written;
  uint64_t *consumed;
  int *transport_active;
  int *collision_mask;
};

struct RecoveryData {
  int *active;
  int *reset_mask;
  int *reason;
  int *bad_index;
  int *source_error;
  int *warning_number;
  int *warning_lastinfo;
  int64_t *reset_events;
  ResetField fields[14];
  int field_count;
};

struct ContactAddressData {
  const int *active;
  const int *status;
  const int *committed;
  const int *count;
  const int *excluded;
  const int *address;
  int *current;
  int capacity;
};

struct NewtonData {
  const physics_newton::NewtonArguments *parameters;
  const int *upstream;
  const int *enabled;
  const int *publication;
  const int64_t *expected_epoch;
  const int *valid;
  const int64_t *written;
  int stop_policy;                 // physics_newton::StopPolicy
  int64_t *audit;                   // optional [worlds, 8]
  real_t *first_disagreement;       // optional [worlds, 8]
};

// A packet addresses one/two consecutive doubles of one original field.
// Bases include the packet offset; width remains the per-world field stride.
struct ControllerCopyPacket {
  real_t *published;
  real_t *working;
  int width;
  int words;
};
static_assert(sizeof(ControllerCopyPacket) == 24, "Keep the copy-plan ABI stable");

struct ControllerChannels {
  DoubleChannel floating[26];
  IntChannel integer[2];
  int floating_count;
  int integer_count;
  const ControllerCopyPacket *copy_packets;
  int seed_packets;
  int commit_packets;
};

struct SubstepArguments {
  FoundationGateData foundation_gate;
  BeginData begin;
  VelocityGateData velocity_gate;
  TransmissionData transmission;
  BodyVelocityData body_velocity;
  JointPassiveData joint_passive;
  FluidPassiveData fluid_passive;
  VelocityStatusData velocity_status;
  VelocityFinishData velocity_finish;
  ReferenceAdmissionData reference_admission;
  ReferenceGateData reference_gate;
  ConstraintReferenceData constraint_reference;
  ReferenceCommitData reference_commit;
  BiasGateData bias_gate;
  BiasReferenceData bias_reference;
  BiasForcesData bias_forces;
  BiasCommitData bias_commit;
  ActuationGateData actuation_gate;
  ControlGateData control_gate;
  ControlCopyData control_copy;
  ActuationData actuation;
  ActuationFinishData actuation_finish;
  SmoothGateData smooth_gate;
  SmoothForceGateData smooth_force_gate;
  SmoothAccelerationData smooth_acceleration;
  SmoothCommitData smooth_commit;
  NewtonAdmissionData newton_admission;
  ConstraintCountsData constraint_counts;
  ConeParametersData cone_parameters;
  NewtonEpochData newton_epoch;
  ForwardPublishData forward_publish;
  ControllerPlanData controller_plan;
  GatherGateData gather_gate;
  EndEffectorPoseData end_effector_pose;
  ControllerInputsData controller_inputs;
  EndEffectorVelocityData end_effector_velocity;
  ControllerBiasData controller_bias;
  ControllerGateData controller_gate;
  ControllerSolveData controller_solve;
  ControllerCommitData controller_commit;
  ControllerPublishData controller_publish;
  EulerData euler;
  StateData state;
  PositionCacheData position;
  RecoveryData recovery;
  ContactAddressData contact_addresses;
  NewtonData newton;
  ControllerChannels controller_channels;
  EpochCopy post_epochs[3];
  int *continuation;
  ForwardWorkspaceData forward_workspace;
};

} // namespace flash_libero
