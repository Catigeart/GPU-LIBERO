#pragma once
#include <cuda_runtime.h>

namespace physics_newton::atomic_reduction {
// Full-world-block collective calls, with world-local global scratch. Neither
// function allocates shared memory. The caller publishes the final output.
// All physical warp lanes must participate; block size is a multiple of 32.
__device__ __noinline__ void transpose(const double *J, const double *force,
    int rows, int nv, double *out, double *scratch);

// Produces only packed lower-triangle J^T D J. Quadratic states (1) and
// nonzero D admit a row; masked rows never read J. The caller adds sparse M,
// mirrors the lower triangle, then retains the existing cone/factor stages.
__device__ __noinline__ void gram(const double *J, const double *D, const int *state,
    int rows, int nv, double *scratch);

// Two private Gram planes and two compact (J, D*J, column) row records. The
// caller loans its existing MathDx arena; this changes no launch allocation.
__host__ __device__ constexpr int gram_block_cells(int nv) {
  return 2 * (nv * (nv + 1) / 2 + 2 * nv + (nv + 1) / 2);
}

// Sparse row outer products in warp-private shared planes, followed by a small
// global atomic merge into scratch. Exact zeros are skipped; no thresholding.
// Insufficient loaned storage or NV>64 selects the original global path.
__device__ __noinline__ void gram_block(const double *J, const double *D, const int *state,
    int rows, int nv, double *scratch, double *block_workspace, int block_cells);
} // namespace physics_newton::atomic_reduction
