// FP64 atomic reductions, compiled once and shared by every catalogue shape.
// Numerical entries remain noinline; no MathDx headers, generated bodies or
// shared allocations. gram_block borrows the caller's existing MathDx arena.
// Only the two-instruction register primitive is inlined.
#include "newton_atomic.cuh"
#include <cstddef>
#include <cmath>

namespace physics_newton::atomic_reduction {

// Exact FP64 register exchange, not arithmetic. In the developer assembler
// policy CUDA's double shuffle wrapper emitted device calls in this hot loop.
// Spell the two b32 operations directly; all 32 physical lanes participate.
static __device__ __forceinline__ double warp_value(double value, int owner) {
  int low = __double2loint(value), high = __double2hiint(value);
  asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;"
               : "=r"(low) : "r"(low), "r"(owner));
  asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;"
               : "=r"(high) : "r"(high), "r"(owner));
  return __hiloint2double(high, low);
}

__device__ __noinline__ void transpose(const double *J, const double *force,
    int rows, int nv, double *out, double *scratch) {
  const int tid = int(threadIdx.x), lane = tid & 31, warp = tid / 32;
  const int warps = int(blockDim.x) / 32;
  for (int j = tid; j < nv; j += int(blockDim.x)) scratch[j] = 0.0;
  __syncthreads(); // every accumulator initialized before any atomic update
  if (nv <= 64) {
    double low = 0.0, high = 0.0;
    for (int r = warp; r < rows; r += warps) {
      const double f = force[r];
      if (f != 0.0) {
        if (lane < nv) low = fma(J[size_t(r) * nv + lane], f, low);
        if (lane + 32 < nv) high = fma(J[size_t(r) * nv + lane + 32], f, high);
      }
    }
    if (lane < nv) atomicAdd(scratch + lane, low);
    if (lane + 32 < nv) atomicAdd(scratch + lane + 32, high);
  } else {
    for (int j = lane; j < nv; j += 32) {
      double partial = 0.0;
      for (int r = warp; r < rows; r += warps) {
        const double f = force[r];
        // A zero-force row may legally contain NaN J: never multiply 0*NaN.
        if (f != 0.0) partial = fma(J[size_t(r) * nv + j], f, partial);
      }
      atomicAdd(scratch + j, partial);
    }
  }
  __syncthreads(); // publish all atomic contributions before ordinary reads
  for (int j = tid; j < nv; j += int(blockDim.x)) out[j] = scratch[j];
  // The solver caller owns the output publication barrier.
}

template <int NV>
__device__ void gram_shape(const double *J, const double *D, const int *state,
    int rows, int runtime_nv, double *scratch) {
  const int nv = NV ? NV : runtime_nv;
  const int tid = int(threadIdx.x), lane = tid & 31, warp = tid / 32;
  const int warps = int(blockDim.x) / 32;
  const int cells = nv * (nv + 1) / 2;
  for (int z = tid; z < cells; z += int(blockDim.x)) scratch[z] = 0.0;
  __syncthreads();

  // A fixed four output cells per lane bounds register lifetimes independently
  // of NV/row count. Each warp handles a disjoint subset of live rows and
  // merges once per triangle cell, rather than once per row contribution.
  constexpr int items = 4, tile = 32 * items;
  for (int begin = 0; begin < cells; begin += tile) {
    int column[items], row[items], packed[items];
    bool live[items];
    double partial[items] = {};
#pragma unroll
    for (int k = 0; k < items; ++k) {
      const int z = begin + lane + 32 * k;
      // Decode a packed lower-triangle cell once per tile, outside the row
      // loop. Padded lanes use the last valid cell and still join shuffles.
      const int bounded = min(z, cells - 1);
      int lo = 0, hi = nv;
      while (lo + 1 < hi) {
        const int middle = (lo + hi) / 2;
        if (middle * (middle + 1) / 2 <= bounded) lo = middle;
        else hi = middle;
      }
      row[k] = lo;
      column[k] = bounded - lo * (lo + 1) / 2;
      live[k] = z < cells;
      packed[k] = z;
    }
    for (int r = warp; r < rows; r += warps) {
      // Retain CPU HessianDirect's quadratic-state mask. No J reads for D=0.
      if (state[r] != 1) continue;
      const double d = D[r];
      if (d == 0.0) continue;
      double low = 0.0, high = 0.0;
      if constexpr (NV && NV <= 64) {
        // Load each J component once for this warp/row, rather than twice
        // for every output cell. Catalogue rows fit two register planes.
        if (lane < nv) low = J[size_t(r) * nv + lane];
        if (lane + 32 < nv) high = J[size_t(r) * nv + lane + 32];
        // Row admission is warp-uniform. Close the lane-masked loads once,
        // before the sequence of full-warp register exchanges below.
        __syncwarp(0xffffffffu);
      }
#pragma unroll
      for (int k = 0; k < items; ++k) {
        double a, b;
        if constexpr (NV && NV <= 64) {
          // Every physical lane participates, including output padding.
          const double al = warp_value(low, row[k] & 31);
          const double bl = warp_value(low, column[k] & 31);
          if constexpr (NV > 32) {
            const double ah = warp_value(high, row[k] & 31);
            const double bh = warp_value(high, column[k] & 31);
            a = row[k] < 32 ? al : ah;
            b = column[k] < 32 ? bl : bh;
          } else { a = al; b = bl; }
        } else {
          a = J[size_t(r) * nv + row[k]];
          b = J[size_t(r) * nv + column[k]];
        }
        partial[k] = fma(a * d, b, partial[k]);
      }
    }
#pragma unroll
    for (int k = 0; k < items; ++k)
      if (live[k]) atomicAdd(scratch + packed[k], partial[k]);
  }
  __syncthreads(); // completed packed Gram; caller adds M and publishes H
}

__device__ __noinline__ void gram(const double *J, const double *D, const int *state,
    int rows, int nv, double *scratch) {
  // Maintained catalogue specializations; runtime fallback remains general.
  // This is ordinary CUDA source, compiled into one independent object.
#define FLASH_GRAM_CASE(n) case n: gram_shape<n>(J,D,state,rows,nv,scratch); return
  switch (nv) {
    FLASH_GRAM_CASE(21); FLASH_GRAM_CASE(22); FLASH_GRAM_CASE(24);
    FLASH_GRAM_CASE(27); FLASH_GRAM_CASE(30); FLASH_GRAM_CASE(33);
    FLASH_GRAM_CASE(36); FLASH_GRAM_CASE(37); FLASH_GRAM_CASE(39);
    FLASH_GRAM_CASE(43); FLASH_GRAM_CASE(45); FLASH_GRAM_CASE(51);
    FLASH_GRAM_CASE(57);
    default: gram_shape<0>(J,D,state,rows,nv,scratch); return;
  }
#undef FLASH_GRAM_CASE
}

// A noinline device entry receives generic pointers. Convert its loaned arena
// once, keeping shared loads/stores as LDS/STS rather than generic LD/ST.
// These four primitives only express address space; all arithmetic is below.
static __device__ __forceinline__ double load_shared_double(unsigned base, int index) {
  double value;
  asm volatile("ld.shared.f64 %0, [%1];" : "=d"(value) : "r"(base + unsigned(index) * 8) : "memory");
  return value;
}
static __device__ __forceinline__ int load_shared_int(unsigned base, int index) {
  int value;
  asm volatile("ld.shared.s32 %0, [%1];" : "=r"(value) : "r"(base + unsigned(index) * 4) : "memory");
  return value;
}
static __device__ __forceinline__ void store_shared_double(unsigned base, int index, double value) {
  asm volatile("st.shared.f64 [%0], %1;" :: "r"(base + unsigned(index) * 8), "d"(value) : "memory");
}
static __device__ __forceinline__ void store_shared_int(unsigned base, int index, int value) {
  asm volatile("st.shared.s32 [%0], %1;" :: "r"(base + unsigned(index) * 4), "r"(value) : "memory");
}

__device__ __noinline__ void gram_block(const double *J, const double *D, const int *state,
    int rows, int nv, double *scratch, double *block_workspace, int block_cells) {
  if (nv > 64 || !block_workspace || block_cells < gram_block_cells(nv)) {
    gram(J, D, state, rows, nv, scratch);
    return;
  }
  const int tid = int(threadIdx.x), lane = tid & 31, warp = tid / 32;
  // At most two private matrices: storage never grows with block/world count.
  // Additional warps cooperate in clearing; they join both block barriers.
  const int planes = min(2, int(blockDim.x) / 32);
  const int cells = nv * (nv + 1) / 2;
  const int row_cells = 2 * nv + (nv + 1) / 2;
  const unsigned arena = unsigned(__cvta_generic_to_shared(block_workspace));
  for (int z = tid; z < planes * cells; z += int(blockDim.x)) store_shared_double(arena, z, 0.0);
  for (int z = tid; z < cells; z += int(blockDim.x)) scratch[z] = 0.0;
  __syncthreads();

  if (warp < planes) {
    const unsigned partial = arena + unsigned(warp * cells) * 8;
    const unsigned values = arena + unsigned(planes * cells + warp * row_cells) * 8;
    const unsigned weighted = values + unsigned(nv) * 8;
    const unsigned columns = weighted + unsigned(nv) * 8;
    for (int r = warp; r < rows; r += planes) {
      // Matches CPU HessianDirect's state/D admission. Masked J stays unread.
      if (state[r] != 1 || D[r] == 0.0) continue;
      const double d = D[r];
      const double low = lane < nv ? J[size_t(r) * nv + lane] : 0.0;
      const double high = lane + 32 < nv ? J[size_t(r) * nv + lane + 32] : 0.0;
      // Exceptional live rows retain dense propagation (0*NaN, 0*Inf).
      // Ordinary finite rows compact exact nonzeros, without an epsilon.
      const bool finite = __all_sync(0xffffffffu, isfinite(low) && isfinite(high) && isfinite(d));
      const unsigned low_mask = __ballot_sync(0xffffffffu, lane < nv && (low != 0.0 || !finite));
      const unsigned high_mask = __ballot_sync(0xffffffffu, lane + 32 < nv && (high != 0.0 || !finite));
      const unsigned before = (1u << lane) - 1;
      const int low_count = __popc(low_mask), nnz = low_count + __popc(high_mask);
      if (low_mask & (1u << lane)) {
        const int p = __popc(low_mask & before);
        store_shared_double(values, p, low);
        store_shared_double(weighted, p, low * d);
        store_shared_int(columns, p, lane);
      }
      if (high_mask & (1u << lane)) {
        const int p = low_count + __popc(high_mask & before);
        store_shared_double(values, p, high);
        store_shared_double(weighted, p, high * d);
        store_shared_int(columns, p, lane + 32);
      }
      __syncwarp(0xffffffffu); // publish compact row, D*J and column indices

      // 8x8 tiles cover only the compact lower triangle. A sparse contact row
      // commonly fits one tile; dense inputs use the same bounded row storage.
      for (int ib = 0; ib < nnz; ib += 8)
        for (int jb = 0; jb <= ib; jb += 8) {
#pragma unroll
          for (int k = 0; k < 2; ++k) {
            const int i = ib + (lane >> 3) + 4 * k, j = jb + (lane & 7);
            if (i < nnz && j <= i) {
              const int ci = load_shared_int(columns, i), cj = load_shared_int(columns, j);
              const int z = ci * (ci + 1) / 2 + cj;
              const double value = fma(load_shared_double(weighted, i), load_shared_double(values, j),
                                       load_shared_double(partial, z));
              store_shared_double(partial, z, value);
            }
          }
        }
      // A cell can have a different producer lane in the next compact row.
      // Also close row-buffer readers before the next row overwrites it.
      __syncwarp(0xffffffffu);
    }
    // Every output has at most two contributors. All updates to each private
    // plane are ordered by its owning warp; no shared atomic/CAS is needed.
    for (int z = lane; z < cells; z += 32) {
      const double value = load_shared_double(partial, z);
      if (value != 0.0) atomicAdd(scratch + z, value);
    }
  }
  __syncthreads(); // publish global packed Gram before caller adds M
}
} // namespace physics_newton::atomic_reduction
