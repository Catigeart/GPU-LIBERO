#pragma once
// Numerical headers consume the configuration and scalar helpers in this order.
// clang-format off
#include "native_config.cuh"
#include "native_parameters.cuh"
#include "linesearch_scalar.cuh"
#include "../common/vector.cuh"
#include "../common/warp.cuh"
#include "../common/packed_mass_gather.cuh"
#include "math.cuh"
#include "constraint.cuh"
#include "hessian.cuh"
#include "state.cuh"
#include "glue.cuh"
#include "../common/math.cuh"
#include "../postprocess/post.cuh"
#include "../postprocess/post_dataflow_original.cuh"
#include "../postprocess/tail.cuh"
#include "hessian_blas_dx.cuh"
#include "direction_blas_dx.cuh"
#include "line_search_warp.cuh"
#include "s07_blas_dx.cuh"
#include "potrf_dx.cuh"
// clang-format on

namespace libero_native {
template <int Offset>
__device__ __forceinline__ real_t *resident(real_t *global, real_t *shared, int world, int pitch) {
  if constexpr (Offset >= 0)
    return shared + Offset;
  else
    return global + world * pitch;
}
template <int Offset>
__device__ __forceinline__ void resident_load(const real_t *global, real_t *local, int size) {
  if constexpr (Offset >= 0)
    libero_newton_io::copy_block(global, local, size);
}
template <int Offset>
__device__ __forceinline__ void resident_store(real_t *global, const real_t *local, int size) {
  if constexpr (Offset >= 0)
    libero_newton_io::copy_block(local, global, size);
}
// Retains the previous forward entry, warmstart, every solve iteration and
// Post publication. Only dimensions and private storage are template operands.
template <class S>
__global__ void native_newton(const Parameters *p, const int *entry_upstream,
                              const int *entry_enabled, const int *entry_publication,
                              const int64_t *entry_plan, int entry_supplied) {
  static_assert(S::nv == NV && S::packed_mass == NM && S::rows == ROWS,
                "The explicit instance must match its numerical device objects");
  using L = NativeNewtonLayout<S>;
  using P = NativeWorldPitch<S>;
  int w = blockIdx.y;
  extern __shared__ __align__(32) real_t shared[];
  __shared__ int invariant_valid, changed_count;
  if (threadIdx.x == 0) {
    invariant_valid = 0;
    changed_count = 0;
  }
  int32_t *hessian_algorithm_diagnostic =
      p->hessian_algorithm_diagnostic + w * P::hessian_algorithm_diagnostic;
  real_t *hessian_upper_backup = p->hessian_upper_backup + w * P::hessian_upper_backup;
  real_t *cone_pending_rows = p->cone_pending_rows + w * P::cone_pending_rows;
  int32_t *mass_plan_offsets = p->mass_plan_offsets;
  int32_t *mass_plan_coefficient = p->mass_plan_coefficient;
  int32_t *mass_plan_source = p->mass_plan_source;
  int32_t *mass_plan_assign = p->mass_plan_assign;
  int32_t *madr = p->madr;
  int32_t *mass_plan_madr_snapshot = p->mass_plan_madr_snapshot;
  int32_t *parentid = p->parentid;
  int32_t *mass_plan_parent_snapshot = p->mass_plan_parent_snapshot;
  int32_t *simplenum = p->simplenum;
  int32_t *mass_plan_simple_snapshot = p->mass_plan_simple_snapshot;
  int32_t *nefc = p->nefc + w * P::nefc;
  int32_t *ne = p->ne + w * P::ne;
  int32_t *nf = p->nf + w * P::nf;
  int32_t *ncon = p->ncon + w * P::ncon;
  int32_t *empty = p->empty + w * P::empty;
  int32_t *newton_world = p->newton_world + w * P::newton_world;
  int32_t *active = p->active + w * P::active;
  int32_t *err_code = p->err_code + w * P::err_code;
  int32_t *error_item = p->error_item + w * P::error_item;
  int32_t *efc_type = p->efc_type + w * P::efc_type;
  int32_t *efc_id = p->efc_id + w * P::efc_id;
  int32_t *cone_dim = p->cone_dim + w * P::cone_dim;
  real_t *cone_mu = p->cone_mu + w * P::cone_mu;
  real_t *cone_friction = p->cone_friction + w * P::cone_friction;
  int32_t *contact_dim = p->contact_dim + w * P::contact_dim;
  real_t *contact_mu = p->contact_mu + w * P::contact_mu;
  real_t *contact_friction = p->contact_friction + w * P::contact_friction;
  real_t *qacc_smooth = p->qacc_smooth + w * P::qacc_smooth;
  real_t *qacc = resident<L::qacc>(p->qacc, shared, w, P::qacc);
  real_t *qacc_warmstart = p->qacc_warmstart + w * P::qacc_warmstart;
  real_t *qfrc_constraint = p->qfrc_constraint + w * P::qfrc_constraint;
  int32_t *solver_iter = p->solver_iter + w * P::solver_iter;
  real_t *J = p->J + w * P::J;
  real_t *aref = p->aref + w * P::aref;
  real_t *efc_b = p->efc_b + w * P::efc_b;
  real_t *jar_warm = p->jar_warm + w * P::jar_warm;
  int32_t *count = p->count + w * P::count;
  int32_t *solver_flag_off = p->solver_flag_off + w * P::solver_flag_off;
  real_t *D = p->D + w * P::D;
  real_t *R = p->R + w * P::R;
  real_t *loss = p->loss + w * P::loss;
  real_t *force = p->force + w * P::force;
  int32_t *state = p->state + w * P::state;
  real_t *contact_H = p->contact_H + w * P::contact_H;
  real_t *cost = p->cost + w * P::cost;
  int32_t *status = p->status + w * P::status;
  int32_t *constraint_group_heads = p->constraint_group_heads + w * P::constraint_group_heads;
  int32_t *constraint_hessian_owner = p->constraint_hessian_owner + w * P::constraint_hessian_owner;
  int32_t *constraint_head_rows = p->constraint_head_rows + w * P::constraint_head_rows;
  int32_t *constraint_head_count = p->constraint_head_count + w * P::constraint_head_count;
  int32_t *constraint_tile_head_offset =
      p->constraint_tile_head_offset + w * P::constraint_tile_head_offset;
  int32_t *constraint_typed_head_rows =
      p->constraint_typed_head_rows + w * P::constraint_typed_head_rows;
  int32_t *constraint_typed_head_offsets =
      p->constraint_typed_head_offsets + w * P::constraint_typed_head_offsets;
  int32_t *solver_err_base_s04 = p->solver_err_base_s04 + w * P::solver_err_base_s04;
  real_t *qM = p->qM + w * P::qM;
  real_t *Ma_warm = p->Ma_warm + w * P::Ma_warm;
  real_t *qfrc_smooth = p->qfrc_smooth + w * P::qfrc_smooth;
  real_t *quad0 = p->quad0 + w * P::quad0;
  int32_t *solver_add_cost_on = p->solver_add_cost_on + w * P::solver_add_cost_on;
  real_t *warm_cost = p->warm_cost + w * P::warm_cost;
  real_t *smooth_cost = p->smooth_cost + w * P::smooth_cost;
  int32_t *choice = p->choice + w * P::choice;
  int32_t *s08a_status = p->s08a_status + w * P::s08a_status;
  int32_t *solver_err_base_s08b = p->solver_err_base_s08b + w * P::solver_err_base_s08b;
  real_t *Ma = resident<L::Ma>(p->Ma, shared, w, P::Ma);
  real_t *jar = p->jar + w * P::jar;
  int32_t *solver_flag_on = p->solver_flag_on + w * P::solver_flag_on;
  int32_t *nactive = p->nactive + w * P::nactive;
  int32_t *ncone = p->ncone + w * P::ncone;
  int32_t *solver_kind_direct = p->solver_kind_direct + w * P::solver_kind_direct;
  int32_t *oldstate = p->oldstate + w * P::oldstate;
  real_t *H = resident<L::H>(p->H, shared, w, P::H);
  int32_t *nnz = p->nnz + w * P::nnz;
  int32_t *route = p->route + w * P::route;
  real_t *Hcone = resident<L::Hcone>(p->Hcone, shared, w, P::Hcone);
  int32_t *nupdate = p->nupdate + w * P::nupdate;
  real_t *H_local = resident<L::H_local>(p->H_local, shared, w, P::H_local);
  real_t *H_ltj = resident<L::H_ltj>(p->H_ltj, shared, w, P::H_ltj);
  real_t *H_vec = resident<L::H_vec>(p->H_vec, shared, w, P::H_vec);
  int32_t *solver_err_base_s05 = p->solver_err_base_s05 + w * P::solver_err_base_s05;
  real_t *grad = resident<L::grad>(p->grad, shared, w, P::grad);
  real_t *Mgrad = resident<L::Mgrad>(p->Mgrad, shared, w, P::Mgrad);
  real_t *candidate = p->candidate + w * P::candidate;
  real_t *scratch_grad = resident<L::scratch_grad>(p->scratch_grad, shared, w, P::scratch_grad);
  real_t *scratch_x = resident<L::scratch_x>(p->scratch_x, shared, w, P::scratch_x);
  real_t *scratch_fwd = resident<L::scratch_fwd>(p->scratch_fwd, shared, w, P::scratch_fwd);
  int32_t *solver_err_base_s06 = p->solver_err_base_s06 + w * P::solver_err_base_s06;
  real_t *search = resident<L::search>(p->search, shared, w, P::search);
  real_t *eval_trace = p->eval_trace + w * P::eval_trace;
  int32_t *eval_tag = p->eval_tag + w * P::eval_tag;
  real_t *alpha = p->alpha + w * P::alpha;
  int32_t *s07_status = p->s07_status + w * P::s07_status;
  int32_t *stop_reason = p->stop_reason + w * P::stop_reason;
  real_t *Mv = resident<L::Mv>(p->Mv, shared, w, P::Mv);
  real_t *Jv = p->Jv + w * P::Jv;
  int32_t *committed = p->committed + w * P::committed;
  int32_t *s08b_status = p->s08b_status + w * P::s08b_status;
  real_t *oldcost = p->oldcost + w * P::oldcost;
  int32_t *solver_kind_incr = p->solver_kind_incr + w * P::solver_kind_incr;
  int32_t *stage = p->stage + w * P::stage;
  real_t *meaninertia = p->meaninertia + w * P::meaninertia;
  real_t *tolerance = p->tolerance + w * P::tolerance;
  real_t *slope = p->slope + w * P::slope;
  int32_t *LSiter = p->LSiter + w * P::LSiter;
  int32_t *iteration = p->iteration + w * P::iteration;
  int32_t *solver_iter_base = p->solver_iter_base + w * P::solver_iter_base;
  int32_t *maxiter_arr = p->maxiter_arr + w * P::maxiter_arr;
  real_t *solver_stats_f = p->solver_stats_f + w * P::solver_stats_f;
  int32_t *solver_stats_i = p->solver_stats_i + w * P::solver_stats_i;
  int32_t *out_nchange = p->out_nchange + w * P::out_nchange;
  real_t *out_improvement = p->out_improvement + w * P::out_improvement;
  real_t *out_gradient = p->out_gradient + w * P::out_gradient;
  int32_t *out_stop = p->out_stop + w * P::out_stop;
  int32_t *out_reason = p->out_reason + w * P::out_reason;
  int32_t *s08c_status = p->s08c_status + w * P::s08c_status;
  int32_t *solver_exit = p->solver_exit + w * P::solver_exit;
  int32_t *solver_finalize_active = p->solver_finalize_active + w * P::solver_finalize_active;
  int32_t *solver_nnz = p->solver_nnz + w * P::solver_nnz;
  int32_t *s08d_status = p->s08d_status + w * P::s08d_status;
  int32_t *solver_err_base_s08d = p->solver_err_base_s08d + w * P::solver_err_base_s08d;
  int64_t *attempted = p->attempted + w * P::attempted;
  real_t *solver_candidate = p->solver_candidate + w * P::solver_candidate;
  int64_t *solver_candidate_attempt = p->solver_candidate_attempt + w * P::solver_candidate_attempt;
  int32_t *solver_candidate_completed =
      p->solver_candidate_completed + w * P::solver_candidate_completed;
  int32_t *solver_candidate_origin = p->solver_candidate_origin + w * P::solver_candidate_origin;
  int32_t *solver_source_stage = p->solver_source_stage + w * P::solver_source_stage;
  int32_t *solver_rejected_final_entry =
      p->solver_rejected_final_entry + w * P::solver_rejected_final_entry;
  int32_t *upstream = p->upstream + w * P::upstream;
  int32_t *enabled = p->enabled + w * P::enabled;
  int32_t *valid = p->valid + w * P::valid;
  int32_t *written = p->written + w * P::written;
  int32_t *diagnostic = p->diagnostic + w * P::diagnostic;
  int64_t *expected = p->expected + w * P::expected;
  int64_t *n_written = p->n_written + w * P::n_written;
  real_t *model_scalars = p->model_scalars;
  real_t *quadGauss = p->quadGauss + w * P::quadGauss;
  real_t *quad = p->quad + w * P::quad;
  int32_t *LSresult = p->LSresult + w * P::LSresult;
  real_t *external_r_ef = p->external_r_ef + w * P::external_r_ef;
  real_t *canonical_qacc = p->canonical_qacc + w * P::canonical_qacc;
  real_t *canonical_qacc_warmstart = p->canonical_qacc_warmstart + w * P::canonical_qacc_warmstart;
  real_t *canonical_qfrc_constraint =
      p->canonical_qfrc_constraint + w * P::canonical_qfrc_constraint;
  int32_t *canonical_state = p->canonical_state + w * P::canonical_state;
  int32_t *canonical_solver_iter = p->canonical_solver_iter + w * P::canonical_solver_iter;
  int32_t *canonical_solver_nnz = p->canonical_solver_nnz + w * P::canonical_solver_nnz;
  real_t *canonical_solver_stats_f = p->canonical_solver_stats_f + w * P::canonical_solver_stats_f;
  int32_t *canonical_solver_stats_i = p->canonical_solver_stats_i + w * P::canonical_solver_stats_i;
  int32_t *potrf_diagnostic = p->potrf_diagnostic + w * P::potrf_diagnostic;
  int32_t *blas_diagnostic = p->blas_diagnostic + w * P::blas_diagnostic;
  resident_load<L::Ma>(p->Ma + w * P::Ma, Ma, P::Ma);
  resident_load<L::H>(p->H + w * P::H, H, P::H);
  resident_load<L::Hcone>(p->Hcone + w * P::Hcone, Hcone, P::Hcone);
  resident_load<L::H_local>(p->H_local + w * P::H_local, H_local, P::H_local);
  resident_load<L::H_ltj>(p->H_ltj + w * P::H_ltj, H_ltj, P::H_ltj);
  resident_load<L::H_vec>(p->H_vec + w * P::H_vec, H_vec, P::H_vec);
  resident_load<L::grad>(p->grad + w * P::grad, grad, P::grad);
  resident_load<L::Mgrad>(p->Mgrad + w * P::Mgrad, Mgrad, P::Mgrad);
  resident_load<L::scratch_grad>(p->scratch_grad + w * P::scratch_grad, scratch_grad,
                                 P::scratch_grad);
  resident_load<L::scratch_x>(p->scratch_x + w * P::scratch_x, scratch_x, P::scratch_x);
  resident_load<L::scratch_fwd>(p->scratch_fwd + w * P::scratch_fwd, scratch_fwd, P::scratch_fwd);
  resident_load<L::search>(p->search + w * P::search, search, P::search);
  resident_load<L::Mv>(p->Mv + w * P::Mv, Mv, P::Mv);
  if (threadIdx.x == 0) {
    diagnostic[0] = entry_publication[w];
    if (entry_enabled[w]) {
      ++attempted[0];
      expected[0] = entry_supplied ? entry_plan[w] : attempted[0];
    }
  }
  __syncthreads();
  libero_newton_io::copy_block(canonical_qacc, qacc, S::nv);
  libero_newton_io::copy_block(canonical_qacc_warmstart, qacc_warmstart, S::nv);
  libero_newton_io::copy_block(canonical_qfrc_constraint, qfrc_constraint, S::nv);
  libero_newton_io::copy_block(external_r_ef, force, max(0, min(nefc[0], S::rows)));
  libero_newton_io::copy_block(canonical_state, state, max(0, min(nefc[0], S::rows)));
  libero_newton_io::copy_block(canonical_solver_iter, solver_iter, 1);
  libero_newton_io::copy_block(canonical_solver_nnz, solver_nnz, 1);
  __syncthreads();
  if (threadIdx.x == 0) {
    iteration[0] = 0;
    committed[0] = 0;
    choice[0] = 0;
    stop_reason[0] = 0;
    err_code[0] = 0;
    s07_status[0] = 0;
    s08a_status[0] = 0;
    s08b_status[0] = 0;
    s08c_status[0] = 0;
    s08d_status[0] = 0;
    LSiter[0] = 0;
    LSresult[0] = 0;
    out_stop[0] = 0;
    out_reason[0] = 0;
    out_nchange[0] = 0;
    solver_exit[0] = 0;
    nactive[0] = 0;
    ncone[0] = 0;
    nupdate[0] = 0;
    nnz[0] = 0;
    route[0] = 0;
    alpha[0] = 0;
    slope[0] = 0;
    cost[0] = 0;
    oldcost[0] = 0;
    quad0[0] = 0;
    warm_cost[0] = 0;
    smooth_cost[0] = 0;
    out_improvement[0] = 0;
    out_gradient[0] = 0;
    error_item[0] = -1;
    stage[0] = 1;
    maxiter_arr[0] = 100;
    solver_iter_base[0] = 0;
    tolerance[0] = model_scalars[0];
    meaninertia[0] = model_scalars[1];
    empty[0] = 0;
    newton_world[0] = 0;
    active[0] = 0;
    count[0] = nefc[0];
    err_code[0] = entry_upstream[w];
    valid[0] = 0;
    written[0] = 0;
    if (entry_enabled[w] && !entry_upstream[w]) {
      if (nefc[0] == 0)
        empty[0] = 1;
      else if (nefc[0] > 0) {
        newton_world[0] = 1;
        active[0] = 1;
      }
    }
  }
  __syncthreads();
  __syncthreads();
  S07Args ls = {};
  ls.search = search;
  ls.Ma = Ma;
  ls.Jaref = jar;
  ls.quad0 = quad0;
  ls.smooth = qfrc_smooth;
  ls.qM = qM;
  ls.J = J;
  ls.D = D;
  ls.R = R;
  ls.loss = loss;
  ls.cone_mu = cone_mu;
  ls.cone_friction = cone_friction;
  ls.efc_type = efc_type;
  ls.efc_id = efc_id;
  ls.cone_dim = cone_dim;
  ls.madr = madr;
  ls.parent = parentid;
  ls.simplenum = simplenum;
  ls.nefc = nefc;
  ls.ne = ne;
  ls.nf = nf;
  ls.active = active;
  ls.model_scalars = model_scalars;
  ls.Mv = Mv;
  ls.Jv = Jv;
  ls.quadGauss = quadGauss;
  ls.quad = quad;
  ls.alpha = alpha;
  ls.slope = slope;
  ls.eval_trace = eval_trace;
  ls.LSiter = LSiter;
  ls.LSresult = LSresult;
  ls.eval_tag = eval_tag;
  ls.status = s07_status;
  real_t *dx_matrix = shared + L::dx_word_offset;
  __shared__ int mass_plan_valid;
  if (threadIdx.x == 0)
    mass_plan_valid = 1;
  __syncthreads();
  for (int i = threadIdx.x; i < NV; i += blockDim.x)
    if (madr[i] != mass_plan_madr_snapshot[i] || parentid[i] != mass_plan_parent_snapshot[i] ||
        simplenum[i] != mass_plan_simple_snapshot[i])
      atomicExch(&mass_plan_valid, 0);
  __syncthreads();
  if (threadIdx.x == 0)
    validate_entry_counts(0, nefc, ne, nf, ncon, empty, newton_world, active, err_code, error_item,
                          p->scalar_22, p->scalar_23);
  __syncthreads();
  validate_contact_rows_block(0, nefc, ne, nf, ncon, efc_type, efc_id, cone_dim, cone_mu,
                              cone_friction, contact_dim, contact_mu, contact_friction,
                              newton_world, active, err_code, error_item, p->scalar_32,
                              p->scalar_33);
  __syncthreads();
  if (threadIdx.x == 0)
    empty_fast_path(0, empty, qacc_smooth, qacc, qacc_warmstart, qfrc_constraint, solver_iter);
  __syncthreads();
  jacvec_sub_blas_dx(0, newton_world, nefc, J, qacc_smooth, aref, efc_b, p->scalar_42, dx_matrix,
                     blas_diagnostic);
  __syncthreads();
  if (threadIdx.x == 0)
    copy_qacc(0, newton_world, qacc_warmstart, qacc);
  __syncthreads();
  jacvec_sub_blas_dx(0, newton_world, nefc, J, qacc_warmstart, aref, jar_warm, p->scalar_44,
                     dx_matrix, blas_diagnostic);
  __syncthreads();
  constraint_update_block(
      0, active, count, ne, nf, ncon, solver_flag_off, jar_warm, D, R, loss, efc_type, efc_id, J,
      contact_mu, contact_friction, contact_dim, force, state, contact_H, cost, status,
      p->scalar_55, p->scalar_56, p->scalar_57, &invariant_valid, nullptr, nullptr, nullptr,
      nullptr, constraint_group_heads, constraint_hessian_owner, constraint_head_rows,
      constraint_head_count, constraint_tile_head_offset, constraint_typed_head_rows,
      constraint_typed_head_offsets, nullptr, nullptr, nullptr);
  __syncthreads();
  constraint_transpose_blas_dx(0, active, count, J, force, qfrc_constraint, status, p->scalar_65,
                               p->scalar_66, dx_matrix, blas_diagnostic);
  __syncthreads();
  if (threadIdx.x == 0)
    gate_status(0, active, status, solver_err_base_s04, err_code);
  __syncthreads();
  mulM_plan_block(0, newton_world, qacc_warmstart, qM, madr, parentid, simplenum, Ma_warm,
                  (mass_plan_valid ? mass_plan_offsets : nullptr),
                  (mass_plan_valid ? mass_plan_coefficient : nullptr),
                  (mass_plan_valid ? mass_plan_source : nullptr),
                  (mass_plan_valid ? mass_plan_assign : nullptr));
  __syncthreads();
  gauss_compute_block(0, newton_world, Ma_warm, qfrc_smooth, qacc_warmstart, qacc_smooth, cost,
                      quad0, solver_add_cost_on);
  __syncthreads();
  if (threadIdx.x == 0)
    copy_cost(0, newton_world, cost, warm_cost);
  __syncthreads();
  constraint_update_block(
      0, active, count, ne, nf, ncon, solver_flag_off, efc_b, D, R, loss, efc_type, efc_id, J,
      contact_mu, contact_friction, contact_dim, force, state, contact_H, cost, status,
      p->scalar_74, p->scalar_75, p->scalar_76, &invariant_valid, nullptr, nullptr, nullptr,
      nullptr, constraint_group_heads, constraint_hessian_owner, constraint_head_rows,
      constraint_head_count, constraint_tile_head_offset, constraint_typed_head_rows,
      constraint_typed_head_offsets, nullptr, nullptr, nullptr);
  __syncthreads();
  constraint_transpose_blas_dx(0, active, count, J, force, qfrc_constraint, status, p->scalar_77,
                               p->scalar_78, dx_matrix, blas_diagnostic);
  __syncthreads();
  if (threadIdx.x == 0)
    gate_status(0, active, status, solver_err_base_s04, err_code);
  __syncthreads();
  if (threadIdx.x == 0)
    copy_cost(0, newton_world, cost, smooth_cost);
  __syncthreads();
  warm_select_block(0, newton_world, nefc, warm_cost, smooth_cost, qacc_warmstart, qacc_smooth,
                    force, state, qfrc_constraint, qacc, force, state, qfrc_constraint, choice,
                    s08a_status, p->scalar_82, p->scalar_83);
  __syncthreads();
  if (threadIdx.x == 0)
    gate_status(0, active, s08a_status, solver_err_base_s08b, err_code);
  __syncthreads();
  if (threadIdx.x == 0)
    solver_iter_reset(0, newton_world, solver_iter);
  __syncthreads();
  warm_init_mass(active, choice, Ma_warm, qacc, qM, madr, parentid, simplenum, Ma,
                 (mass_plan_valid ? mass_plan_offsets : nullptr),
                 (mass_plan_valid ? mass_plan_coefficient : nullptr),
                 (mass_plan_valid ? mass_plan_source : nullptr),
                 (mass_plan_valid ? mass_plan_assign : nullptr));
  __syncthreads();
  warm_init_jacvec(active, nefc, choice, jar_warm, efc_b, jar);
  __syncthreads();
  constraint_update_block(
      0, active, count, ne, nf, ncon, solver_flag_on, jar, D, R, loss, efc_type, efc_id, J,
      contact_mu, contact_friction, contact_dim, force, state, contact_H, cost, status,
      p->scalar_89, p->scalar_90, p->scalar_91, &invariant_valid, nactive, ncone, nullptr, nullptr,
      constraint_group_heads, constraint_hessian_owner, constraint_head_rows, constraint_head_count,
      constraint_tile_head_offset, constraint_typed_head_rows, constraint_typed_head_offsets,
      nullptr, nullptr, nullptr);
  __syncthreads();
  constraint_transpose_blas_dx(0, active, count, J, force, qfrc_constraint, status, p->scalar_94,
                               p->scalar_95, dx_matrix, blas_diagnostic);
  __syncthreads();
  if (threadIdx.x == 0)
    gate_status(0, active, status, solver_err_base_s04, err_code);
  __syncthreads();
  gauss_compute_block(0, active, Ma, qfrc_smooth, qacc, qacc_smooth, cost, quad0,
                      solver_add_cost_on);
  __syncthreads();
  hessian_validate_block(0, count, ncon, active, solver_kind_direct, state, oldstate, D, J, qM, H,
                         efc_id, contact_dim, contact_H, ncone, nnz, status, route, p->scalar_101,
                         p->scalar_102, &invariant_valid);
  __syncthreads();
  hessian_assemble_blas_dx(0, count, active, solver_kind_direct, J, D, state, qM, madr, parentid,
                           simplenum, H, status, p->scalar_103, dx_matrix, blas_diagnostic);
  __syncthreads();
  hessian_factor_update_dx(0, count, ncon, active, solver_kind_direct, J, D, state, oldstate, qM,
                           madr, parentid, simplenum, efc_id, contact_dim, contact_H, ncone, H,
                           Hcone, nnz, nupdate, H_local, H_ltj, H_vec, status, route, p->scalar_109,
                           p->scalar_110, dx_matrix, potrf_diagnostic, blas_diagnostic,
                           cone_pending_rows, hessian_algorithm_diagnostic, nullptr,
                           hessian_upper_backup);
  __syncthreads();
  if (threadIdx.x == 0)
    gate_status(0, active, status, solver_err_base_s05, err_code);
  __syncthreads();
  direction_blas_dx(0, Ma, qfrc_smooth, qfrc_constraint, H, Hcone, ncone, nefc, active, status,
                    grad, Mgrad, candidate, status, scratch_grad, scratch_x, scratch_fwd, dx_matrix,
                    blas_diagnostic, DIRECTION_BASE_SHARED, DIRECTION_CONE_SHARED, nullptr);
  __syncthreads();
  if (threadIdx.x == 0)
    gate_status(0, active, status, solver_err_base_s06, err_code);
  __syncthreads();
  if (threadIdx.x == 0)
    direction_commit(0, active, candidate, search);
  __syncthreads();
  int iteration_slot = 0;
  for (; iteration_slot < 100; ++iteration_slot) {
    if (!active[0])
      break;
    if (threadIdx.x == 0)
      seed_eval(0, eval_trace, eval_tag);
    __syncthreads();
    search_kernel_blas_dx(0, ls, 1, NV, NM, ROWS, 52, dx_matrix, blas_diagnostic,
                          constraint_group_heads, constraint_head_rows, constraint_head_count,
                          constraint_tile_head_offset, nullptr, nullptr, nullptr);
    __syncthreads();
    if (threadIdx.x == 0)
      gate_ls(0, active, alpha, s07_status, stop_reason, err_code);
    __syncthreads();
    commit_step_block(0, active, nefc, alpha, search, Mv, Jv, qacc, Ma, jar, committed, s08b_status,
                      p->scalar_129, p->scalar_130);
    __syncthreads();
    if (threadIdx.x == 0)
      gate_status(0, active, s08b_status, solver_err_base_s08b, err_code);
    __syncthreads();
    if (threadIdx.x == 0)
      save_old(0, active, nefc, state, cost, oldstate, oldcost, p->scalar_132);
    __syncthreads();
    constraint_update_block(
        0, active, count, ne, nf, ncon, solver_flag_on, jar, D, R, loss, efc_type, efc_id, J,
        contact_mu, contact_friction, contact_dim, force, state, contact_H, cost, status,
        p->scalar_133, p->scalar_134, p->scalar_135, &invariant_valid, nactive, ncone, oldstate,
        &changed_count, constraint_group_heads, constraint_hessian_owner, constraint_head_rows,
        constraint_head_count, constraint_tile_head_offset, constraint_typed_head_rows,
        constraint_typed_head_offsets, nullptr, nullptr, nullptr);
    __syncthreads();
    constraint_transpose_blas_dx(0, active, count, J, force, qfrc_constraint, status, p->scalar_136,
                                 p->scalar_137, dx_matrix, blas_diagnostic);
    __syncthreads();
    if (threadIdx.x == 0)
      gate_status(0, active, status, solver_err_base_s04, err_code);
    __syncthreads();
    gauss_compute_block(0, active, Ma, qfrc_smooth, qacc, qacc_smooth, cost, quad0,
                        solver_add_cost_on);
    __syncthreads();
    hessian_validate_block(0, count, ncon, active, solver_kind_incr, state, oldstate, D, J, qM, H,
                           efc_id, contact_dim, contact_H, ncone, nnz, status, route, p->scalar_139,
                           p->scalar_140, &invariant_valid);
    __syncthreads();
    hessian_assemble_blas_dx(0, count, active, solver_kind_incr, J, D, state, qM, madr, parentid,
                             simplenum, H, status, p->scalar_141, dx_matrix, blas_diagnostic);
    __syncthreads();
    hessian_factor_update_dx(0, count, ncon, active, solver_kind_incr, J, D, state, oldstate, qM,
                             madr, parentid, simplenum, efc_id, contact_dim, contact_H, ncone, H,
                             Hcone, nnz, nupdate, H_local, H_ltj, H_vec, status, route,
                             p->scalar_142, p->scalar_143, dx_matrix, potrf_diagnostic,
                             blas_diagnostic, cone_pending_rows, hessian_algorithm_diagnostic,
                             nullptr, hessian_upper_backup);
    __syncthreads();
    if (threadIdx.x == 0)
      gate_status(0, active, status, solver_err_base_s05, err_code);
    __syncthreads();
    direction_blas_dx(0, Ma, qfrc_smooth, qfrc_constraint, H, Hcone, ncone, nefc, active, status,
                      grad, Mgrad, candidate, status, scratch_grad, scratch_x, scratch_fwd,
                      dx_matrix, blas_diagnostic, DIRECTION_BASE_SHARED, DIRECTION_CONE_SHARED,
                      nullptr);
    __syncthreads();
    if (threadIdx.x == 0)
      gate_status(0, active, status, solver_err_base_s06, err_code);
    __syncthreads();
    stats_stop_block(0, active, nefc, stage, committed, oldstate, state, grad, oldcost, cost,
                     meaninertia, tolerance, slope, nactive, nupdate, LSiter, iteration,
                     solver_iter_base, maxiter_arr, solver_stats_f, solver_stats_i, out_nchange,
                     out_improvement, out_gradient, out_stop, out_reason, iteration, s08c_status,
                     p->scalar_160, p->scalar_161, &changed_count, Ma, qfrc_smooth, J, force);
    __syncthreads();
    if (threadIdx.x == 0)
      gate_stats(0, active, out_stop, out_reason, s08c_status, stop_reason, err_code);
    __syncthreads();
    if (threadIdx.x == 0)
      direction_commit(0, active, candidate, search);
    __syncthreads();
    if (threadIdx.x == 0)
      gate_maxiter(0, active, out_stop, out_reason, stop_reason);
    __syncthreads();
  }
  if (iteration_slot < 100) {
    if (threadIdx.x == 0) {
      s07_status[0] = 0;
      status[0] = 0;
      route[0] = 0;
      seed_eval(0, eval_trace, eval_tag);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
    finalize_exit(0, newton_world, err_code, solver_exit, solver_finalize_active);
  __syncthreads();
  finalize_block(0, solver_finalize_active, solver_exit, iteration, qacc, qacc_warmstart,
                 solver_iter, solver_nnz, s08d_status, p->scalar_166);
  __syncthreads();
  if (threadIdx.x == 0)
    gate_status(0, solver_finalize_active, s08d_status, solver_err_base_s08d, err_code);
  __syncthreads();
  if (threadIdx.x == 0)
    final_event(0, qacc, empty, newton_world, solver_exit, err_code, s08d_status, attempted,
                solver_candidate, solver_candidate_attempt, solver_candidate_completed,
                solver_candidate_origin, solver_source_stage, solver_rejected_final_entry,
                p->scalar_175);
  __syncthreads();
  if (threadIdx.x == 0)
    publish(0, entry_upstream + w, entry_enabled + w, err_code, valid, written);
  __syncthreads();
  if (threadIdx.x == 0)
    publication_gate(0, valid, diagnostic);
  __syncthreads();
  if (threadIdx.x == 0)
    finish_publication(0, valid, written, expected, n_written);
  __syncthreads();
  if (valid[0]) {
    libero_newton_io::copy_block(qacc, canonical_qacc, S::nv);
    libero_newton_io::copy_block(qacc_warmstart, canonical_qacc_warmstart, S::nv);
    libero_newton_io::copy_block(qfrc_constraint, canonical_qfrc_constraint, S::nv);
    libero_newton_io::copy_block(force, external_r_ef, nefc[0]);
    libero_newton_io::copy_block(state, canonical_state, nefc[0]);
    libero_newton_io::copy_block(solver_iter, canonical_solver_iter, 1);
    libero_newton_io::copy_block(solver_nnz, canonical_solver_nnz, 1);
    libero_newton_io::copy_block(solver_stats_f, canonical_solver_stats_f,
                                 3 * min(iteration[0], 1000));
    libero_newton_io::copy_block(solver_stats_i, canonical_solver_stats_i,
                                 4 * min(iteration[0], 1000));
  }
  __syncthreads();
  libero_tail::post_private(
      w, dx_matrix, p->external_g_active, p->external_g_first, p->external_g_dv, p->valid,
      p->err_code, p->n_written, p->attempted, p->external_g_expected, p->external_g_state,
      p->external_g_consumed, p->external_g_forward, p->external_g_cf, p->external_g_g,
      p->external_g_g, p->external_r_qv, qacc, p->external_r_ci, p->external_r_cv, p->external_r_cd,
      p->external_r_cdd, p->external_r_xi, p->external_r_com, p->external_r_sp, p->external_r_sm,
      p->external_r_xf, p->external_r_ef, p->external_r_nc, p->nefc, p->ne, p->contact_dim,
      p->external_r_adr, p->external_r_g1, p->external_r_g2, p->external_r_frame, p->external_r_pos,
      p->external_r_parent, p->external_r_da, p->external_r_dn, p->external_r_root,
      p->external_r_gb, p->external_r_grav, p->scalar_219, p->scalar_220, p->external_r_acc,
      p->external_r_ext, p->external_r_intr, p->external_r_decoded, p->external_r_status,
      p->scalar_226, p->scalar_227, p->scalar_228, p->scalar_229, p->external_r_intr,
      p->external_r_com, p->external_r_sp, p->external_r_sm, p->external_g_g, p->external_s_type,
      p->external_s_site, p->external_s_adr, p->external_s_sb, p->external_s_br,
      p->external_s_result, p->external_s_status, p->scalar_237, p->scalar_238, p->external_g_g,
      p->external_r_status, p->external_s_status, p->external_g_first, p->external_g_expected,
      p->external_r_ext, p->external_r_intr, p->external_s_result, p->external_c_pe,
      p->external_c_pi, p->external_c_ps, p->external_c_valid, p->external_c_written, p->scalar_244,
      p->scalar_245, p->scalar_246, p->external_cache_acc, p->external_cache_force,
      p->external_cache_status, p->external_cache_valid, p->external_cache_state,
      p->external_cache_forward, p->scalar_253, p->external_body_index, p->external_body_offsets,
      p->external_body_counts, p->external_level_offsets, p->external_level_bodies, p->scalar_259,
      p->external_children_offsets, p->external_children_indices, p->external_event_ping,
      p->external_event_pong, p->scalar_264, p->scalar_265, p->external_cached_event_body,
      p->external_cached_contact_meta, p->scalar_268);
  __syncthreads();
  resident_store<L::qacc>(p->qacc + w * P::qacc, qacc, P::qacc);
  resident_store<L::Ma>(p->Ma + w * P::Ma, Ma, P::Ma);
  resident_store<L::H>(p->H + w * P::H, H, P::H);
  resident_store<L::Hcone>(p->Hcone + w * P::Hcone, Hcone, P::Hcone);
  resident_store<L::H_local>(p->H_local + w * P::H_local, H_local, P::H_local);
  resident_store<L::H_ltj>(p->H_ltj + w * P::H_ltj, H_ltj, P::H_ltj);
  resident_store<L::H_vec>(p->H_vec + w * P::H_vec, H_vec, P::H_vec);
  resident_store<L::grad>(p->grad + w * P::grad, grad, P::grad);
  resident_store<L::Mgrad>(p->Mgrad + w * P::Mgrad, Mgrad, P::Mgrad);
  resident_store<L::scratch_grad>(p->scratch_grad + w * P::scratch_grad, scratch_grad,
                                  P::scratch_grad);
  resident_store<L::scratch_x>(p->scratch_x + w * P::scratch_x, scratch_x, P::scratch_x);
  resident_store<L::scratch_fwd>(p->scratch_fwd + w * P::scratch_fwd, scratch_fwd, P::scratch_fwd);
  resident_store<L::search>(p->search + w * P::search, search, P::search);
  resident_store<L::Mv>(p->Mv + w * P::Mv, Mv, P::Mv);
}

} // namespace libero_native
