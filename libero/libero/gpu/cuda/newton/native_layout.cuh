#pragma once
#include "../step/instantiations/libero_models.cuh"
#include <array>
#include <cstddef>
namespace libero_native {
constexpr int align_bytes(int size, int alignment) {
  return (size + alignment - 1) / alignment * alignment;
}
constexpr int max_int(int a, int b) { return a > b ? a : b; }
template <class S> constexpr int vendor_bytes() {
  const int n = S::nv, k = S::newton_hessian_tile_rows;
  int cells = max_int(n * n, n * n + 2 * n * k);
  cells = max_int(cells, 17 * n + 16);
  cells = max_int(cells, 12 * n + 36);
  cells = max_int(cells, (n * n + 1) / 2 * 2 + n);
  return align_bytes(8 * cells, 16);
}
template <class S> struct NativeWorldPitch {
  static constexpr int D = S::rows;
  static constexpr int H = S::nv * S::nv;
  static constexpr int H_local = 36;
  static constexpr int H_ltj = 6 * S::nv;
  static constexpr int H_vec = S::nv;
  static constexpr int Hcone = S::nv * S::nv;
  static constexpr int J = S::rows * S::nv;
  static constexpr int Jv = S::rows;
  static constexpr int LSiter = 1;
  static constexpr int LSresult = 1;
  static constexpr int Ma = S::nv;
  static constexpr int Ma_warm = S::nv;
  static constexpr int Mgrad = S::nv;
  static constexpr int Mv = S::nv;
  static constexpr int R = S::rows;
  static constexpr int active = 1;
  static constexpr int alpha = 1;
  static constexpr int aref = S::rows;
  static constexpr int attempted = 1;
  static constexpr int blas_diagnostic = 7;
  static constexpr int candidate = S::nv;
  static constexpr int canonical_qacc = S::nv;
  static constexpr int canonical_qacc_warmstart = S::nv;
  static constexpr int canonical_qfrc_constraint = S::nv;
  static constexpr int canonical_solver_iter = 1;
  static constexpr int canonical_solver_nnz = 1;
  static constexpr int canonical_solver_stats_f = 3000;
  static constexpr int canonical_solver_stats_i = 4000;
  static constexpr int canonical_state = S::rows;
  static constexpr int choice = 1;
  static constexpr int committed = 1;
  static constexpr int cone_dim = S::rows;
  static constexpr int cone_friction = 5 * S::rows;
  static constexpr int cone_mu = S::rows;
  static constexpr int cone_pending_rows = S::nv * S::newton_hessian_tile_rows;
  static constexpr int constraint_group_heads = S::rows;
  static constexpr int constraint_head_count = 1;
  static constexpr int constraint_head_rows = S::rows;
  static constexpr int constraint_hessian_owner = 5000;
  static constexpr int constraint_tile_head_offset = (S::rows + 15) / 16 + 1;
  static constexpr int constraint_typed_head_offsets = 8;
  static constexpr int constraint_typed_head_rows = S::rows;
  static constexpr int contact_H = 180000;
  static constexpr int contact_dim = 5000;
  static constexpr int contact_friction = 25000;
  static constexpr int contact_mu = 5000;
  static constexpr int cost = 1;
  static constexpr int count = 1;
  static constexpr int diagnostic = 1;
  static constexpr int efc_b = S::rows;
  static constexpr int efc_id = S::rows;
  static constexpr int efc_type = S::rows;
  static constexpr int empty = 1;
  static constexpr int enabled = 1;
  static constexpr int err_code = 1;
  static constexpr int error_item = 1;
  static constexpr int eval_tag = 52;
  static constexpr int eval_trace = 208;
  static constexpr int expected = 1;
  static constexpr int external_r_ef = S::rows;
  static constexpr int force = S::rows;
  static constexpr int grad = S::nv;
  static constexpr int hessian_algorithm_diagnostic = 8;
  static constexpr int hessian_upper_backup = S::nv * S::nv;
  static constexpr int iteration = 1;
  static constexpr int jar = S::rows;
  static constexpr int jar_warm = S::rows;
  static constexpr int loss = S::rows;
  static constexpr int madr = 0;
  static constexpr int mass_plan_assign = 0;
  static constexpr int mass_plan_coefficient = 0;
  static constexpr int mass_plan_madr_snapshot = 0;
  static constexpr int mass_plan_offsets = 0;
  static constexpr int mass_plan_parent_snapshot = 0;
  static constexpr int mass_plan_simple_snapshot = 0;
  static constexpr int mass_plan_source = 0;
  static constexpr int maxiter_arr = 1;
  static constexpr int meaninertia = 1;
  static constexpr int model_scalars = 0;
  static constexpr int n_written = 1;
  static constexpr int nactive = 1;
  static constexpr int ncon = 1;
  static constexpr int ncone = 1;
  static constexpr int ne = 1;
  static constexpr int nefc = 1;
  static constexpr int newton_world = 1;
  static constexpr int nf = 1;
  static constexpr int nnz = 1;
  static constexpr int nupdate = 1;
  static constexpr int oldcost = 1;
  static constexpr int oldstate = S::rows;
  static constexpr int out_gradient = 1;
  static constexpr int out_improvement = 1;
  static constexpr int out_nchange = 1;
  static constexpr int out_reason = 1;
  static constexpr int out_stop = 1;
  static constexpr int parentid = 0;
  static constexpr int potrf_diagnostic = 4;
  static constexpr int qM = S::packed_mass;
  static constexpr int qacc = S::nv;
  static constexpr int qacc_smooth = S::nv;
  static constexpr int qacc_warmstart = S::nv;
  static constexpr int qfrc_constraint = S::nv;
  static constexpr int qfrc_smooth = S::nv;
  static constexpr int quad = 3 * S::rows;
  static constexpr int quad0 = 1;
  static constexpr int quadGauss = 3;
  static constexpr int route = 1;
  static constexpr int s07_status = 1;
  static constexpr int s08a_status = 1;
  static constexpr int s08b_status = 1;
  static constexpr int s08c_status = 1;
  static constexpr int s08d_status = 1;
  static constexpr int scratch_fwd = S::nv;
  static constexpr int scratch_grad = S::nv;
  static constexpr int scratch_x = S::nv;
  static constexpr int search = S::nv;
  static constexpr int simplenum = 0;
  static constexpr int slope = 1;
  static constexpr int smooth_cost = 1;
  static constexpr int solver_add_cost_on = 1;
  static constexpr int solver_candidate = S::nv;
  static constexpr int solver_candidate_attempt = 1;
  static constexpr int solver_candidate_completed = 1;
  static constexpr int solver_candidate_origin = 1;
  static constexpr int solver_err_base_s04 = 1;
  static constexpr int solver_err_base_s05 = 1;
  static constexpr int solver_err_base_s06 = 1;
  static constexpr int solver_err_base_s08b = 1;
  static constexpr int solver_err_base_s08d = 1;
  static constexpr int solver_exit = 1;
  static constexpr int solver_finalize_active = 1;
  static constexpr int solver_flag_off = 1;
  static constexpr int solver_flag_on = 1;
  static constexpr int solver_iter = 1;
  static constexpr int solver_iter_base = 1;
  static constexpr int solver_kind_direct = 1;
  static constexpr int solver_kind_incr = 1;
  static constexpr int solver_nnz = 1;
  static constexpr int solver_rejected_final_entry = 1;
  static constexpr int solver_source_stage = 1;
  static constexpr int solver_stats_f = 3000;
  static constexpr int solver_stats_i = 4000;
  static constexpr int stage = 1;
  static constexpr int state = S::rows;
  static constexpr int status = 1;
  static constexpr int stop_reason = 1;
  static constexpr int tolerance = 1;
  static constexpr int upstream = 1;
  static constexpr int valid = 1;
  static constexpr int warm_cost = 1;
  static constexpr int written = 1;
};
struct SharedAllocation {
  std::array<int, 14> offsets{};
  int end = 0;
};
// Exact existing placement order. A rejected resident matrix keeps its global
// pointer; subsequent smaller arrays may still use the remaining shared space.
template <class S> constexpr SharedAllocation allocate_shared() {
  using P = NativeWorldPitch<S>;
  constexpr std::array<int, 14> extent = {
      P::qacc,         P::search,      P::grad,  P::Mgrad,   P::Ma,    P::Mv, P::scratch_x,
      P::scratch_grad, P::scratch_fwd, P::H_vec, P::H_local, P::H_ltj, P::H,  P::Hcone};
  constexpr int reserve =
      2560 + 260 + 12 * S::newton_hessian_tile_rows + 24 + 32 * 4 + 9 * 32 * 8 + 10 * 4;
  SharedAllocation result{};
  for (int i = 0; i < 14; ++i) {
    const int offset = align_bytes(result.end, 32);
    const int end = offset + 8 * extent[i];
    result.offsets[i] = -1;
    if (align_bytes(end, 16) + vendor_bytes<S>() + reserve <= 48 * 1024) {
      result.offsets[i] = offset / 8;
      result.end = end;
    }
  }
  return result;
}
template <class S> struct NativeNewtonLayout {
  inline static constexpr auto allocation = allocate_shared<S>();
  static constexpr int qacc = allocation.offsets[0];
  static constexpr int search = allocation.offsets[1];
  static constexpr int grad = allocation.offsets[2];
  static constexpr int Mgrad = allocation.offsets[3];
  static constexpr int Ma = allocation.offsets[4];
  static constexpr int Mv = allocation.offsets[5];
  static constexpr int scratch_x = allocation.offsets[6];
  static constexpr int scratch_grad = allocation.offsets[7];
  static constexpr int scratch_fwd = allocation.offsets[8];
  static constexpr int H_vec = allocation.offsets[9];
  static constexpr int H_local = allocation.offsets[10];
  static constexpr int H_ltj = allocation.offsets[11];
  static constexpr int H = allocation.offsets[12];
  static constexpr int Hcone = allocation.offsets[13];
  static constexpr int dx_bytes = vendor_bytes<S>();
  static constexpr int dx_word_offset = align_bytes(allocation.end, 16) / 8;
  static constexpr int shared_bytes = dx_word_offset * 8 + dx_bytes;
};
} // namespace libero_native
