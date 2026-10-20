#include "precision.cuh"
#include "newton_jacobian_cache.cuh"
#include "newton_transpose.cuh"
#include "newton_reduction_policy.cuh"
#include "newton_gram_layout.cuh"
#include "../common/shared_access.cuh"
#include <cmath>

#ifndef FLASH_NEWTON_JACOBIAN_CACHE
#define FLASH_NEWTON_JACOBIAN_CACHE 1
#endif
#ifndef FLASH_NEWTON_TRANSPOSE_METHOD
#define FLASH_NEWTON_TRANSPOSE_METHOD 0
#endif
namespace physics_newton {
__device__ __constant__ NewtonReductionPolicy reduction_policy = {
    FLASH_NEWTON_JACOBIAN_CACHE, static_cast<TransposeMethod>(FLASH_NEWTON_TRANSPOSE_METHOD)};
}
extern "C" int flash_libero_newton_configure_reductions(int cache, int method) {
  if ((cache != 0 && cache != 1) || method < 0 || method > 8) return int(cudaErrorInvalidValue);
  const physics_newton::NewtonReductionPolicy value{cache, static_cast<physics_newton::TransposeMethod>(method)};
  return int(cudaMemcpyToSymbol(physics_newton::reduction_policy, &value, sizeof(value)));
}

namespace physics_newton::jacobian_cache {
using flash_libero::load_shared_real;
using flash_libero::store_shared_real;
// The cache and source rows are global allocations, including when this
// noinline function is reached from a kernel borrowing shared Newton storage.
// Explicit global access avoids the generic LD/ST address-space dispatch.
static __device__ __forceinline__ real_t global_real(const real_t *p) {
  real_t x; asm volatile("ld.global." REAL_PTX_TYPE " %0, [%1];" : REAL_PTX_OUT(x) : "l"(p) : "memory"); return x;
}
static __device__ __forceinline__ int global_int(const int *p) {
  int x; asm volatile("ld.global.s32 %0, [%1];" : "=r"(x) : "l"(p) : "memory"); return x;
}
static __device__ __forceinline__ int2 global_record(const int2 *p) {
  int2 x; asm volatile("ld.global.v2.s32 {%0, %1}, [%2];" : "=r"(x.x), "=r"(x.y) : "l"(p) : "memory"); return x;
}
static __device__ __forceinline__ void global_put(real_t *p, real_t x) {
  asm volatile("st.global." REAL_PTX_TYPE " [%0], %1;" :: "l"(p), REAL_PTX_IN(x) : "memory");
}
static __device__ __forceinline__ void global_put(int *p, int x) {
  asm volatile("st.global.s32 [%0], %1;" :: "l"(p), "r"(x) : "memory");
}

static __device__ __forceinline__ int shared_int(unsigned base, int i) {
  int x; asm volatile("ld.shared.s32 %0, [%1];" : "=r"(x) : "r"(base + unsigned(i) * 4) : "memory"); return x;
}
static __device__ __forceinline__ void shared_int_put(unsigned base, int i, int x) {
  asm volatile("st.shared.s32 [%0], %1;" :: "r"(base + unsigned(i) * 4), "r"(x) : "memory");
}
static __device__ __forceinline__ int reserve_product(unsigned base, int i) {
  int slot; asm volatile("atom.shared.add.u32 %0, [%1], 1;" : "=r"(slot) : "r"(base + unsigned(i) * 4) : "memory"); return slot;
}
__device__ __noinline__ void prepare(View v, const real_t *J, const real_t *D,
    int rows, uint64_t generation, real_t *shared) {
  const int tid = int(threadIdx.x), lane = tid & 31, warp = tid / 32;
  int *counts0 = v.integers(NewtonJacobianLayout::column_counts);
  int *offsets = v.integers(NewtonJacobianLayout::column_offsets(v.nv));
  // Cursors are private register counters. This global plane retains warp 1's
  // counts, separate from warp 0's counts and the final column offsets.
  int *counts1 = v.integers(NewtonJacobianLayout::column_cursors(v.nv));
  int *used = v.integers(NewtonJacobianLayout::used);
  int2 *records = v.records();
  real_t *values = v.values(), *weighted = v.weighted();
  int *columns = v.columns(), *csc_rows = v.csc_rows(), *csc_entries = v.csc_entries();
  if (tid == 0) {
    *reinterpret_cast<uint64_t *>(v.base) = generation;
    *v.integers(NewtonJacobianLayout::rows) = rows;
    *v.integers(NewtonJacobianLayout::products_valid) = 0;
    *v.integers(NewtonJacobianLayout::csc_valid) = 0;
  }
  const int cells = v.nv * (v.nv + 1) / 2;
  int *product_counts = v.integers(NewtonJacobianLayout::product_counts(v.row_capacity, v.nv));
  auto *product_values = reinterpret_cast<real_t *>(v.base + NewtonJacobianLayout::product_values(v.row_capacity, v.nv));
  int *product_rows = v.integers(NewtonJacobianLayout::product_rows(v.row_capacity, v.nv));
  // Loan the existing MathDx arena. Integer shared atomic reservation is native
  // on SM80; FP64 products/sums never use shared atomic/CAS instructions.
  const unsigned counters = unsigned(__cvta_generic_to_shared(shared));
  const unsigned row_values = counters + unsigned(GramRowLayout::after_counters(cells));
  const unsigned before = (1u << lane) - 1;
  bool products_failed = false;
  for (int z = tid; z < cells; z += int(blockDim.x)) shared_int_put(counters, z, 0);
  __syncthreads();
  if (warp < 2) {
    int cursor = 0, count_low = 0, count_high = 0;
    const int pool = v.capacity() / 2, begin = warp * pool;
    const bool supported = v.nv <= 64 && int(blockDim.x) >= 64 && rows <= v.row_capacity;
    if (supported) for (int r = warp; r < rows; r += 2) {
      const real_t d = global_real(D + r);
      const real_t low = lane < v.nv ? global_real(J + size_t(r) * v.nv + lane) : REAL_C(0.0);
      const real_t high = lane + 32 < v.nv ? global_real(J + size_t(r) * v.nv + lane + 32) : REAL_C(0.0);
      const bool finite = __all_sync(0xffffffffu, isfinite(low) && isfinite(high) && isfinite(d));
      const unsigned lm = __ballot_sync(0xffffffffu, lane < v.nv && (low != REAL_C(0.0) || !finite));
      const unsigned hm = __ballot_sync(0xffffffffu, lane + 32 < v.nv && (high != REAL_C(0.0) || !finite));
      const int nlow = __popc(lm), nnz = nlow + __popc(hm);
      if (cursor + nnz > pool) { cursor = -1; break; }
      if (lane == 0) records[r] = make_int2(begin + cursor, nnz);
      if (lm & (1u << lane)) {
        const int z = begin + cursor + __popc(lm & before);
        global_put(values + z, low); global_put(weighted + z, low * d); global_put(columns + z, lane);
        ++count_low;
      }
      if (hm & (1u << lane)) {
        const int z = begin + cursor + nlow + __popc(hm & before);
        global_put(values + z, high); global_put(weighted + z, high * d); global_put(columns + z, lane + 32);
        ++count_high;
      }
      // Dense/exceptional rows use the general row Gram. Admission only
      // chooses the optional product-cache algorithm; no row is discarded.
      const bool product_row = finite && nnz * 3 < v.nv;
      products_failed |= !product_row;
      if (product_row && d != REAL_C(0.0)) {
        const unsigned row_bytes = unsigned(GramRowLayout::bytes(v.nv));
        const unsigned rv = row_values + unsigned(warp) * row_bytes;
        const unsigned rw = rv + unsigned(GramRowLayout::weighted(v.nv));
        const unsigned rc = rv + unsigned(GramRowLayout::columns(v.nv));
        if (lm & (1u << lane)) {
          const int k = __popc(lm & before);
          store_shared_real(rv, k, low); store_shared_real(rw, k, low * d); shared_int_put(rc, k, lane);
        }
        if (hm & (1u << lane)) {
          const int k = nlow + __popc(hm & before);
          store_shared_real(rv, k, high); store_shared_real(rw, k, high * d); shared_int_put(rc, k, lane + 32);
        }
        __syncwarp(0xffffffffu);
        for (int ib = 0; ib < nnz; ib += 8)
          for (int jb = 0; jb <= ib; jb += 8) {
#pragma unroll
            for (int k = 0; k < 2; ++k) {
              const int i = ib + (lane >> 3) + 4 * k, j = jb + (lane & 7);
              if (i < nnz && j <= i) {
                const int ci = shared_int(rc, i), cj = shared_int(rc, j), z = ci * (ci + 1) / 2 + cj;
                const int slot = reserve_product(counters, z);
                products_failed |= slot >= NewtonJacobianLayout::product_slots;
                if (slot < NewtonJacobianLayout::product_slots) {
                  const int p = z * NewtonJacobianLayout::product_slots + slot;
                  global_put(product_values + p, load_shared_real(rw, i) * load_shared_real(rv, j));
                  global_put(product_rows + p, r);
                }
              }
            }
          }
        __syncwarp(0xffffffffu); // close row readers before the next producer
      }
      cursor += nnz;
    }
    int *counts = warp ? counts1 : counts0;
    if (lane < v.nv) counts[lane] = count_low;
    if (lane + 32 < v.nv) counts[lane + 32] = count_high;
    if (lane == 0) used[warp] = supported ? cursor : -1;
  }
  const bool product_failure = __syncthreads_or(products_failed);
  if (!v.valid()) return;
  for (int z = tid; z < cells; z += int(blockDim.x)) global_put(product_counts + z, shared_int(counters, z));
  if (tid == 0) *v.integers(NewtonJacobianLayout::products_valid) = !product_failure;
  // The three-forward workload cannot amortize a second dense J scan just to
  // save a little time in sparse J^T f. Auto uses direct column ownership;
  // explicit CSC candidates still construct and benchmark the column lists.
  if (reduction_policy.transpose < TransposeMethod::Csc1) return;
  for (int j = tid; j <= v.nv; j += int(blockDim.x)) {
    int begin = 0;
    for (int k = 0; k < j; ++k) begin += counts0[k] + counts1[k];
    offsets[j] = begin;
  }
  __syncthreads(); // disjoint column segments before CSC producers
  if (warp < 2) {
    int cursor_low = lane < v.nv ? offsets[lane] + (warp ? counts0[lane] : 0) : 0;
    int cursor_high = lane + 32 < v.nv ? offsets[lane + 32] + (warp ? counts0[lane + 32] : 0) : 0;
    // Each lane owns one/two columns in its warp's disjoint CSC segment.
    // One extra coalesced J pass removes both global atomic counting loops.
    for (int r = warp; r < rows; r += 2) {
      const real_t d = global_real(D + r);
      const real_t low = lane < v.nv ? global_real(J + size_t(r) * v.nv + lane) : REAL_C(0.0);
      const real_t high = lane + 32 < v.nv ? global_real(J + size_t(r) * v.nv + lane + 32) : REAL_C(0.0);
      const bool finite = __all_sync(0xffffffffu, isfinite(low) && isfinite(high) && isfinite(d));
      const unsigned lm = __ballot_sync(0xffffffffu, lane < v.nv && (low != REAL_C(0.0) || !finite));
      const unsigned hm = __ballot_sync(0xffffffffu, lane + 32 < v.nv && (high != REAL_C(0.0) || !finite));
      const int begin = global_record(records + r).x;
      if (lm & (1u << lane)) {
        global_put(csc_rows + cursor_low, r); global_put(csc_entries + cursor_low++, begin + __popc(lm & before));
      }
      if (hm & (1u << lane)) {
        global_put(csc_rows + cursor_high, r); global_put(csc_entries + cursor_high++, begin + __popc(lm) + __popc(hm & before));
      }
    }
  }
  if (tid == 0) *v.integers(NewtonJacobianLayout::csc_valid) = 1;
  // Caller publishes the completed CSC, cached products and header.
}

__device__ __noinline__ void gram(View v, const real_t *D, const int *state,
    int rows, real_t *scratch, real_t *shared) {
  const int tid = int(threadIdx.x), lane = tid & 31, warp = tid / 32;
  const int cells = v.nv * (v.nv + 1) / 2;
  if (*v.integers(NewtonJacobianLayout::products_valid)) {
    const int *counts = v.integers(NewtonJacobianLayout::product_counts(v.row_capacity, v.nv));
    const auto *values = reinterpret_cast<const real_t *>(v.base + NewtonJacobianLayout::product_values(v.row_capacity, v.nv));
    const int *product_rows = v.integers(NewtonJacobianLayout::product_rows(v.row_capacity, v.nv));
    // One owner per packed output. There is no clear, private shared matrix,
    // row warp barrier or FP64 atomic merge. Every call uses fresh row state.
    for (int z = tid; z < cells; z += int(blockDim.x)) {
      real_t value = REAL_C(0.0);
      const int begin = z * NewtonJacobianLayout::product_slots;
      for (int k = 0, n = global_int(counts + z); k < n; ++k) {
        const int p = begin + k;
        if (global_int(state + global_int(product_rows + p)) == 1)
          value += global_real(values + p);
      }
      scratch[z] = value;
    }
    __syncthreads();
    return;
  }
  const unsigned arena = unsigned(__cvta_generic_to_shared(shared));
  for (int z = tid; z < 2 * cells; z += int(blockDim.x)) store_shared_real(arena, z, REAL_C(0.0));
  for (int z = tid; z < cells; z += int(blockDim.x)) scratch[z] = REAL_C(0.0);
  __syncthreads();
  if (warp < 2) {
    const unsigned partial = arena + unsigned(warp * cells) * unsigned(sizeof(real_t));
    for (int r = warp; r < rows; r += 2) {
      if (global_int(state + r) != 1 || global_real(D + r) == REAL_C(0.0)) continue;
      const int2 row = global_record(v.records() + r);
      // Persistent global row data is already published. No per-row shared
      // staging, ballot, D multiply, or producer warp barrier remains here.
      for (int ib = 0; ib < row.y; ib += 8)
        for (int jb = 0; jb <= ib; jb += 8) {
#pragma unroll
          for (int k = 0; k < 2; ++k) {
            const int i = ib + (lane >> 3) + 4 * k, j = jb + (lane & 7);
            if (i < row.y && j <= i) {
              const int ei = row.x + i, ej = row.x + j;
              const int ci = global_int(v.columns() + ei), cj = global_int(v.columns() + ej);
              const int z = ci * (ci + 1) / 2 + cj;
              store_shared_real(partial, z, fma(global_real(v.weighted() + ei), global_real(v.values() + ej), load_shared_real(partial, z)));
            }
          }
        }
      __syncwarp(0xffffffffu); // same H cell can change producer lane next row
    }
    for (int z = lane; z < cells; z += 32) {
      const real_t value = load_shared_real(partial, z);
      if (value != REAL_C(0.0)) atomicAdd(scratch + z, value);
    }
  }
  __syncthreads(); // packed Gram publication; caller adds M
}

template<int Lanes> static __device__ void gather(View v, const real_t *force, real_t *out) {
  const int lane = int(threadIdx.x) % Lanes, group = int(threadIdx.x) / Lanes;
  const int groups = int(blockDim.x) / Lanes;
  const int *offsets = v.integers(NewtonJacobianLayout::column_offsets(v.nv));
  const unsigned mask = (0xffffffffu >> (32 - Lanes)) << ((int(threadIdx.x) & 31) / Lanes * Lanes);
  for (int j = group; j < v.nv; j += groups) {
    real_t value = REAL_C(0.0);
    for (int z = global_int(offsets + j) + lane, end = global_int(offsets + j + 1); z < end; z += Lanes) {
      const real_t f = global_real(force + global_int(v.csc_rows() + z));
      if (f != REAL_C(0.0)) value += global_real(v.values() + global_int(v.csc_entries() + z)) * f;
    }
    if constexpr (Lanes > 1)
      for (int delta = Lanes / 2; delta; delta /= 2) value += __shfl_down_sync(mask, value, delta, Lanes);
    if (lane == 0) out[j] = value;
  }
}

__device__ __noinline__ void transpose(View v, const real_t *J, const real_t *force,
    int rows, real_t *out, int lanes) {
  if (!*v.integers(NewtonJacobianLayout::csc_valid)) {
    transpose_reduction::dense(J, force, rows, v.nv, out, nullptr, transpose_reduction::Method::Direct);
    return;
  }
  bool nonfinite = false;
  for (int r = int(threadIdx.x); r < rows; r += int(blockDim.x)) nonfinite |= !isfinite(global_real(force + r));
  if (__syncthreads_or(nonfinite)) {
    transpose_reduction::dense(J, force, rows, v.nv, out, nullptr, transpose_reduction::Method::Direct);
    return;
  }
  switch (lanes) {
    case 1: gather<1>(v, force, out); return;
    case 4: gather<4>(v, force, out); return;
    default: gather<8>(v, force, out); return;
  }
}
} // namespace physics_newton::jacobian_cache
