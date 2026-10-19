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
  Complete = 4
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
  double *moment;
  double *qvel;
  int *active;
  int *position_status;
  int nv;
  int nu;
  double *velocity;
  int *status;
};

struct BodyVelocityData {
  int *active;
  double *cdof;
  double *qvel;
  int *bp;
  int *ba;
  int *bn;
  int *dj;
  int *jt;
  int nb;
  int nv;
  double *cvel;
  double *cdot;
  int *status;
  double *cinert;
  double *gravity;
  int disabled;
  double *cache_acc;
  double *cache_force;
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
  double *qpos;
  double *qvel;
  int *active;
  int *jtype;
  int *qadr;
  int *dadr;
  double *stiffness;
  double *spring;
  double *damping;
  double *out;
  int *status;
  int nv;
  int nq;
  int njnt;
  int joint_disjoint;
  const int *joint_type_offsets;
  const int *joint_type_indices;
};

struct FluidPassiveData {
  double *cvel;
  double *subtree_com;
  double *xipos;
  double *ximat;
  double *cdof;
  int *active;
  int *rootid;
  double *mass;
  double *box;
  int *ancestor;
  double *wind;
  double *opt;
  int passive_disabled;
  int diagnostic;
  double *stages;
  double *qfrc;
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
  double *qvel;
  int64_t *epoch;
  int64_t *expected;
  int32_t *ready;
  int32_t rows;
  int32_t nv;
};

struct ConstraintReferenceData {
  int32_t *committed;
  int32_t *counts;
  double *efc_J;
  double *kbip;
  double *efc_pos;
  double *efc_margin;
  double *qvel;
  int64_t *qvel_epoch;
  double *vel;
  double *aref;
  int64_t *ref_epoch;
  int32_t rows;
  int32_t nv;
};

struct ReferenceCommitData {
  int32_t *ready;
  int32_t *status;
  int32_t *committed;
  int32_t *counts;
  double *sv;
  double *sa;
  int64_t *epoch;
  double *pv;
  double *pa;
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
  double *cdot;
  double *qvel;
  double *cinert;
  double *cvel;
  double *cdof;
  int *bp;
  int *ba;
  int *bn;
  int *db;
  double *gravity;
  int disabled;
  int nb;
  int nv;
  double *cacc;
  double *frc;
  double *result;
  int *status;
  const double *cache_acc;
  const double *cache_force;
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
  double *work;
  double *pub;
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
  double *source;
  double *target;
  int width;
};

struct ActuationData {
  double *length;
  double *moment;
  double *velocity;
  double *ctrl;
  int *active;
  int *vel_status;
  double *gain;
  double *bias0;
  double *bias1;
  double *bias2;
  int *bias_type;
  double *ctrl_lo;
  double *ctrl_hi;
  double *force_lo;
  double *force_hi;
  int *force_limited;
  int disable_flags;
  double *force;
  double *force_pre;
  double *ctrl_used;
  int *ctrl_route;
  int *force_route;
  double *qfrc_actuator;
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
  const double *passive;
  const double *bias;
  const double *applied;
  const double *actuator;
  const int *active;
  int *entry_status;
  double *smooth;
  double *seed;
  int nv;
  const double *xfrc;
  const double *xipos;
  const double *subtree_com;
  const double *cdof;
  const int *body_ids;
  const int *body_count;
  const int *body_rootid;
  const int *ancestor;
  double *scratch;
  double *stages;
  double *qfrc;
  int *d02_status;
  int nbody;
  const double *qm;
  const int *count;
  const int *madr;
  const int *parent;
  double *ld;
  double *inv;
  double *sqrt_inv;
  double *solution;
  int *pivot;
  int *solve_status;
  int *warning_count;
  int *warning_last;
  int nm;
  double *rhs;
  double *qacc_smooth;
  int *status;
  const double *source_ld;
  const double *source_inv;
  const double *source_sqrt_inv;
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
  double *wf;
  double *wa;
  double *pf;
  double *pa;
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
  const double *mu;
  const double *friction;
  int *out_dim;
  double *out_mu;
  double *out_friction;
  int *status;
  int *item;
  int rows;
  int contacts;
  const int *compact_heads;
  const int *compact_count;
  const int *compact_valid;
  const int *head_dim;
  double *head_mu;
  double *head_friction;
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
  const double *pos;
  const double *mat;
  const int32_t *active;
  double *outp;
  double *outm;
  int ns;
  int id;
};

struct ControllerInputsData {
  const double *cdof;
  const double *subtree;
  const double *pos;
  const int32_t *active;
  const int32_t *ancestor;
  int root;
  int body;
  int nv;
  int nb;
  double *jp;
  double *jr;
  int32_t *jac_status;
  int32_t *adapter_active;
  const double *qpos;
  const double *qvel;
  const double *rot;
  const double *qm;
  const double *qbias;
  const int32_t *qi;
  const int32_t *vi;
  const int32_t *sparse;
  int nq;
  int nm;
  double *ep;
  double *er;
  double *vp;
  double *vo;
  double *q;
  double *qd;
  double *J;
  double *M;
  double *bias;
  int32_t *adapter_status;
  int32_t *out_status;
  const int32_t *body_dof_offsets;
  const int32_t *body_dof_indices;
};

struct EndEffectorVelocityData {
  const int32_t *active;
  const int32_t *status;
  const double *vp;
  const double *vo;
  double *target;
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
  const double *bias;
  double *selected;
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
  double *mass;
  double *jacobian;
  double *q;
  double *qvel;
  double *bias;
  double *q0;
  double *ee_pos;
  double *ee_rot;
  double *ee_velocity;
  double *action;
  double *goal_pos;
  double *goal_rot;
  double *gripper;
  int *active;
  double *gains;
  double *range;
  double *ctrl;
  double *raw_tau;
  int *error;
  int *ranks;
  int *policy;
  int *needs;
  double *mass_inv;
  double *A_full;
  double *A_pos;
  double *A_ori;
  double *lambda_full;
  double *lambda_pos;
  double *lambda_ori;
  double *nullspace_matrix;
  double *position_error;
  double *orientation_error;
  double *quat_input;
  float *quat_cast;
  double *quat_result;
  int *quat_valid;
  double *nullspace_torque;
  double *clipped;
  int diagnostics;
  int uncouple;
  int *derived_valid;
  double *derived_key;
  double *derived_data;
  int *derived_ranks;
  int *cache_hits;
  double *pm;
  double *pj;
  double *pq;
  double *pv;
  double *pp;
  double *pr;
  double *pe;
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
  double *qp;
  double *qv;
  double *time;
  const double *qm;
  const double *smooth;
  const double *constraint;
  const double *acc;
  const double *damping;
  const int *madr;
  double *mq;
  double *rhs;
  int *count;
  int *ma;
  int *damp;
  int *status;
  double dt;
  int nq;
  int nv;
  int nm;
  const int *parent;
  double *ld;
  double *inv;
  double *si;
  double *solution;
  int *pivot;
  int *ms;
  int *wc;
  int *wl;
  const int *simple;
  double *work;
  int *skip;
  const int *jt;
  const int *padr;
  const int *vadr;
  double *sp;
  double *sv;
  double *used;
  double *st;
  double *axis;
  double *angle;
  double *increment;
  double *pre;
  int *af;
  int *za;
  int *qf;
  int nj;
  int nq4;
  double *qh;
  double *qhi;
  int qacc_stride;
  const int *joint_quat;
  const int *joint_type_offsets;
  const int *joint_type_indices;
};

struct DoubleChannel {
  double *published;
  double *working;
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
  double *values;
  const double *defaults;
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
  double *qpos;
  double *qvel;
  double *qacc;
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
  double *first_disagreement;       // optional [worlds, 8]
};

// A packet addresses one/two consecutive doubles of one original field.
// Bases include the packet offset; width remains the per-world field stride.
struct ControllerCopyPacket {
  double *published;
  double *working;
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
