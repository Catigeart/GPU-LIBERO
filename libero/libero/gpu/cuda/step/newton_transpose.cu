#include "newton_transpose.cuh"
#include <cstddef>

namespace physics_newton::transpose_reduction {
static __device__ __forceinline__ double global_double(const double *p) {
  double x; asm volatile("ld.global.f64 %0, [%1];" : "=d"(x) : "l"(p) : "memory"); return x;
}
// These primitives express the loaned arena's address space across the
// separately compiled function boundary. No shared allocation or atomics.
static __device__ __forceinline__ void put(unsigned base, int i, double value) {
  asm volatile("st.shared.f64 [%0], %1;" :: "r"(base + unsigned(i) * 8), "d"(value) : "memory");
}
static __device__ __forceinline__ double get(unsigned base, int i) {
  double value;
  asm volatile("ld.shared.f64 %0, [%1];" : "=d"(value) : "r"(base + unsigned(i) * 8) : "memory");
  return value;
}

template<int Lanes> static __device__ void partitioned(const double *J, const double *force,
    int rows, int nv, double *out, double *shared) {
  const int tid = int(threadIdx.x), lane = tid % Lanes, group = tid / Lanes;
  const int groups = int(blockDim.x) / Lanes;
  const unsigned arena = unsigned(__cvta_generic_to_shared(shared));
  // Catalogue NV<=64. Retain all column planes in registers, traversing force
  // once per row partition, including the partly occupied last plane.
  double partial[64 / Lanes] = {};
  for (int r = group; r < rows; r += groups) {
    const double f = global_double(force + r);
    if (f != 0.0) {
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
    double value = 0.0;
    for (int g = 0; g < groups; ++g) value += get(arena, g * nv + j);
    out[j] = value;
  }
}

__device__ __noinline__ void dense(const double *J, const double *force,
    int rows, int nv, double *out, double *shared, Method method) {
  if (nv <= 64 && method != Method::Direct) {
    switch (method) {
      case Method::Shared8: partitioned<8>(J, force, rows, nv, out, shared); return;
      case Method::Shared16: partitioned<16>(J, force, rows, nv, out, shared); return;
      default: partitioned<32>(J, force, rows, nv, out, shared); return;
    }
  }
  for (int j = int(threadIdx.x); j < nv; j += int(blockDim.x)) {
    double value = 0.0;
    for (int r = 0; r < rows; ++r) {
      const double f = global_double(force + r);
      if (f != 0.0) value += global_double(J + size_t(r) * nv + j) * f;
    }
    out[j] = value;
  }
}
} // namespace physics_newton::transpose_reduction
