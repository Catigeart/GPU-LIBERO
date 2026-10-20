#include "precision.cuh"
#pragma once
#include "newton_types.cuh"
#include "newton_shared_layout.cuh"

namespace physics_newton {
// Uniform control values live in block shared storage. Arrays make the
// zero initialization cooperative; named accessors keep stage code readable.
struct NewtonControl {
  enum IntegerSlot {
    slot_empty,
    slot_newton_world,
    slot_active,
    slot_err_code,
    slot_error_item,
    slot_solver_iter,
    slot_count,
    slot_ne,
    slot_nf,
    slot_ncon,
    slot_status,
    slot_choice,
    slot_s08a_status,
    slot_nactive,
    slot_ncone,
    slot_nnz,
    slot_route,
    slot_nupdate,
    slot_s07_status,
    slot_stop_reason,
    slot_committed,
    slot_reserved_commit,
    slot_stage,
    slot_LSiter,
    slot_iteration,
    slot_solver_iter_base,
    slot_maxiter_arr,
    slot_out_nchange,
    slot_out_stop,
    slot_out_reason,
    slot_s08c_status,
    slot_solver_exit,
    slot_solver_finalize_active,
    slot_solver_nnz,
    slot_s08d_status,
    slot_valid,
    slot_written,
    slot_diagnostic,
    slot_invariant_valid,
    slot_changed_count,
    integer_count
  };
  enum RealSlot {
    slot_cost,
    slot_quad0,
    slot_warm_cost,
    slot_smooth_cost,
    slot_meaninertia,
    slot_tolerance,
    slot_slope,
    slot_alpha,
    slot_oldcost,
    slot_out_improvement,
    slot_out_gradient,
    real_count
  };
  int integers[integer_count];
  real_t reals[real_count];
  int64_t expected;
#if defined(__CUDACC__)
#define CF_NEWTON_HD __host__ __device__
#else
#define CF_NEWTON_HD
#endif
  CF_NEWTON_HD int &empty() { return integers[slot_empty]; }
  CF_NEWTON_HD int &newton_world() { return integers[slot_newton_world]; }
  CF_NEWTON_HD int &active() { return integers[slot_active]; }
  CF_NEWTON_HD int &err_code() { return integers[slot_err_code]; }
  CF_NEWTON_HD int &error_item() { return integers[slot_error_item]; }
  CF_NEWTON_HD int &solver_iter() { return integers[slot_solver_iter]; }
  CF_NEWTON_HD int &count() { return integers[slot_count]; }
  CF_NEWTON_HD int &ne() { return integers[slot_ne]; }
  CF_NEWTON_HD int &nf() { return integers[slot_nf]; }
  CF_NEWTON_HD int &ncon() { return integers[slot_ncon]; }
  CF_NEWTON_HD int &status() { return integers[slot_status]; }
  CF_NEWTON_HD int &choice() { return integers[slot_choice]; }
  CF_NEWTON_HD int &s08a_status() { return integers[slot_s08a_status]; }
  CF_NEWTON_HD int &nactive() { return integers[slot_nactive]; }
  CF_NEWTON_HD int &ncone() { return integers[slot_ncone]; }
  CF_NEWTON_HD int &nnz() { return integers[slot_nnz]; }
  CF_NEWTON_HD int &route() { return integers[slot_route]; }
  CF_NEWTON_HD int &nupdate() { return integers[slot_nupdate]; }
  CF_NEWTON_HD int &s07_status() { return integers[slot_s07_status]; }
  CF_NEWTON_HD int &stop_reason() { return integers[slot_stop_reason]; }
  CF_NEWTON_HD int &committed() { return integers[slot_committed]; }
  CF_NEWTON_HD int &stage() { return integers[slot_stage]; }
  CF_NEWTON_HD int &LSiter() { return integers[slot_LSiter]; }
  CF_NEWTON_HD int &iteration() { return integers[slot_iteration]; }
  CF_NEWTON_HD int &solver_iter_base() { return integers[slot_solver_iter_base]; }
  CF_NEWTON_HD int &maxiter_arr() { return integers[slot_maxiter_arr]; }
  CF_NEWTON_HD int &out_nchange() { return integers[slot_out_nchange]; }
  CF_NEWTON_HD int &out_stop() { return integers[slot_out_stop]; }
  CF_NEWTON_HD int &out_reason() { return integers[slot_out_reason]; }
  CF_NEWTON_HD int &s08c_status() { return integers[slot_s08c_status]; }
  CF_NEWTON_HD int &solver_exit() { return integers[slot_solver_exit]; }
  CF_NEWTON_HD int &solver_finalize_active() { return integers[slot_solver_finalize_active]; }
  CF_NEWTON_HD int &solver_nnz() { return integers[slot_solver_nnz]; }
  CF_NEWTON_HD int &s08d_status() { return integers[slot_s08d_status]; }
  CF_NEWTON_HD int &valid() { return integers[slot_valid]; }
  CF_NEWTON_HD int &written() { return integers[slot_written]; }
  CF_NEWTON_HD int &diagnostic() { return integers[slot_diagnostic]; }
  CF_NEWTON_HD int &invariant_valid() { return integers[slot_invariant_valid]; }
  CF_NEWTON_HD int &changed_count() { return integers[slot_changed_count]; }
  CF_NEWTON_HD real_t &cost() { return reals[slot_cost]; }
  CF_NEWTON_HD real_t &quad0() { return reals[slot_quad0]; }
  CF_NEWTON_HD real_t &warm_cost() { return reals[slot_warm_cost]; }
  CF_NEWTON_HD real_t &smooth_cost() { return reals[slot_smooth_cost]; }
  CF_NEWTON_HD real_t &meaninertia() { return reals[slot_meaninertia]; }
  CF_NEWTON_HD real_t &tolerance() { return reals[slot_tolerance]; }
  CF_NEWTON_HD real_t &slope() { return reals[slot_slope]; }
  CF_NEWTON_HD real_t &alpha() { return reals[slot_alpha]; }
  CF_NEWTON_HD real_t &oldcost() { return reals[slot_oldcost]; }
  CF_NEWTON_HD real_t &out_improvement() { return reals[slot_out_improvement]; }
  CF_NEWTON_HD real_t &out_gradient() { return reals[slot_out_gradient]; }
#undef CF_NEWTON_HD
};
// This POD describes the uniform scalar region only. It owns no allocation,
// pointers, or numerical vectors; accessors are aliases of the indexed cells.
static_assert(sizeof(NewtonControl) == flash_libero::NewtonLayout::control_bytes, "Shared control layout");
static_assert(offsetof(NewtonControl, reals) == 40 * sizeof(int), "Shared real control offset");
} // namespace physics_newton
