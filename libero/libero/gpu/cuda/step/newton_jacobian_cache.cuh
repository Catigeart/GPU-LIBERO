#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "newton_jacobian_layout.cuh"

namespace physics_newton::jacobian_cache {
struct View {
  unsigned char *base;
  int row_capacity, nv;
  __host__ __device__ int capacity() const { return NewtonJacobianLayout::entries(row_capacity, nv); }
  __device__ int *integers(std::size_t offset) const { return reinterpret_cast<int *>(base + offset); }
  __device__ bool prepared(int rows, uint64_t generation) const {
    return *reinterpret_cast<const uint64_t *>(base) == generation &&
        *integers(NewtonJacobianLayout::rows) == rows;
  }
  __device__ bool valid() const {
    const int *used = integers(NewtonJacobianLayout::used);
    return used[0] >= 0 && used[1] >= 0;
  }
  __device__ int2 *records() const { return reinterpret_cast<int2 *>(base + NewtonJacobianLayout::row_records(nv)); }
  __device__ double *values() const { return reinterpret_cast<double *>(base + NewtonJacobianLayout::values(row_capacity, nv)); }
  __device__ double *weighted() const { return reinterpret_cast<double *>(base + NewtonJacobianLayout::weighted(row_capacity, nv)); }
  __device__ int *columns() const { return integers(NewtonJacobianLayout::columns(row_capacity, nv)); }
  __device__ int *csc_rows() const { return integers(NewtonJacobianLayout::csc_rows(row_capacity, nv)); }
  __device__ int *csc_entries() const { return integers(NewtonJacobianLayout::csc_entries(row_capacity, nv)); }
};

// Collective build, invoked once per successful position/layout generation.
// Counts and entries are published by the caller's existing entry barrier.
// No force/state is cached. Exceptional rows retain every dense component.
__device__ __noinline__ void prepare(View view, const double *J, const double *D,
    int rows, uint64_t generation, double *shared);

// No entry clear needed. Consumers retain the dense functions as fallbacks.
// Gram borrows two existing private shared planes; publishes packed scratch.
__device__ __noinline__ void gram(View view, const double *D, const int *state,
    int rows, double *scratch, double *shared);

// CSC gather, group size 1/4/8. Caller publishes out. Nonfinite force falls
// back to the dense column owner to retain implicit-zero NaN/Inf semantics.
__device__ __noinline__ void transpose(View view, const double *J, const double *force,
    int rows, double *out, int lanes);
} // namespace physics_newton::jacobian_cache
