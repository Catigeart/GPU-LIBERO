#include "precision.cuh"
#include "newton_transpose.cuh"
#include <cstddef>

namespace physics_newton::transpose_reduction {
static __device__ __forceinline__ real_t global_double(const real_t *p) {
  real_t x; asm volatile("ld.global." REAL_PTX_TYPE " %0, [%1];" : REAL_PTX_OUT(x) : "l"(p) : "memory"); return x;
}
// These primitives express the loaned arena's address space across the
// separately compiled function boundary. No shared allocation or atomics.
static __device__ __forceinline__ void put(unsigned base, int i, real_t value) {
  asm volatile("st.shared." REAL_PTX_TYPE " [%0], %1;" :: "r"(base + unsigned(i) * unsigned(sizeof(real_t))), REAL_PTX_IN(value) : "memory");
}
static __device__ __forceinline__ real_t get(unsigned base, int i) {
  real_t value;
  asm volatile("ld.shared." REAL_PTX_TYPE " %0, [%1];" : REAL_PTX_OUT(value) : "r"(base + unsigned(i) * unsigned(sizeof(real_t))) : "memory");
  return value;
}

template<int Lanes> static __device__ void partitioned(const real_t *J, const real_t *force,
    int rows, int nv, real_t *out, real_t *shared) {
  const int tid = int(threadIdx.x), lane = tid % Lanes, group = tid / Lanes;
  const int groups = int(blockDim.x) / Lanes;
  const unsigned arena = unsigned(__cvta_generic_to_shared(shared));
  // Catalogue NV<=64. Retain all column planes in registers, traversing force
  // once per row partition, including the partly occupied last plane.
  real_t partial[64 / Lanes] = {};
  for (int r = group; r < rows; r += groups) {
    const real_t f = global_double(force + r);
    if (f != REAL_C(0.0)) {
#pragma unroll
      for (int k = 0; k < 64 / Lanes; ++k) {
        const int j = lane + k * Lanes;
        if (j < nv) partial[k] += global_double(J + size_t(r) * nv + j) * f;
      }
    }
  }
#pragma unroll
  for (int k = 0; k < 64 / Lanes; ++k) {
    const int j = lane + k * Lanes;
    if (j < nv) put(arena, group * nv + j, partial[k]);
  }
  // Every producer writes its own partial; no clear or initialization barrier.
  __syncthreads();
  for (int j = tid; j < nv; j += int(blockDim.x)) {
    real_t value = REAL_C(0.0);
    for (int g = 0; g < groups; ++g) value += get(arena, g * nv + j);
    out[j] = value;
  }
}

// Four lanes own one DOF. Each traverses a disjoint row subsequence and keeps
// its sum in a register. The exchange names only that four-lane subgroup;
// unequal column tails therefore need neither a block barrier nor shared data.
// Match direct ownership's zero-force masking, including unread NaN J rows.
__device__ __noinline__ void subwarp4(const real_t *J, const real_t *force,
    int rows, int nv, real_t *out) {
  const int tid = int(threadIdx.x), lane = tid & 3, group = tid / 4;
  const int groups = int(blockDim.x) / 4;
  const unsigned mask = 15u << ((tid & 31) / 4 * 4);
  for (int j = group; j < nv; j += groups) {
    real_t value = REAL_C(0.0);
    for (int r = lane; r < rows; r += 4) {
      const real_t f = force[r];
      if (f != REAL_C(0.0))
        value += J[size_t(r) * nv + j] * f;
    }
    value += __shfl_down_sync(mask, value, 2, 4);
    value += __shfl_down_sync(mask, value, 1, 4);
    if (!lane)
      out[j] = value;
  }
}

__device__ __noinline__ void dense(const real_t *J, const real_t *force,
    int rows, int nv, real_t *out, real_t *shared, Method method) {
  if (nv <= 64 && method != Method::Direct) {
    switch (method) {
      case Method::Shared8: partitioned<8>(J, force, rows, nv, out, shared); return;
      case Method::Shared16: partitioned<16>(J, force, rows, nv, out, shared); return;
      default: partitioned<32>(J, force, rows, nv, out, shared); return;
    }
  }
  for (int j = int(threadIdx.x); j < nv; j += int(blockDim.x)) {
    real_t value = REAL_C(0.0);
    for (int r = 0; r < rows; ++r) {
      const real_t f = global_double(force + r);
      if (f != REAL_C(0.0)) value += global_double(J + size_t(r) * nv + j) * f;
    }
    out[j] = value;
  }
}
} // namespace physics_newton::transpose_reduction
