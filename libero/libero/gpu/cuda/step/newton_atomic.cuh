#include "precision.cuh"
#pragma once
#include <cuda_runtime.h>
#include "newton_gram_layout.cuh"

namespace physics_newton::atomic_reduction {
// Full-world-block collective calls, with world-local global scratch. Neither
// function allocates shared memory. The caller publishes the final output.
// All physical warp lanes must participate; block size is a multiple of 32.
__device__ __noinline__ void transpose(const real_t *J, const real_t *force,
    int rows, int nv, real_t *out, real_t *scratch);

// Produces only packed lower-triangle J^T D J. Quadratic states (1) and
// nonzero D admit a row; masked rows never read J. The caller adds sparse M,
// mirrors the lower triangle, then retains the existing cone/factor stages.
__device__ __noinline__ void gram(const real_t *J, const real_t *D, const int *state,
    int rows, int nv, real_t *scratch);

// Two private Gram planes and two compact (J, D*J, column) row records. The
// caller loans its existing MathDx arena; this changes no launch allocation.
__host__ __device__ constexpr int gram_block_cells(int nv) {
  return 2 * (nv * (nv + 1) / 2 + int(GramRowLayout::bytes(nv) / sizeof(real_t)));
}

// Sparse row outer products in warp-private shared planes, followed by a small
// global atomic merge into scratch. Exact zeros are skipped; no thresholding.
// Insufficient loaned storage or NV>64 selects the original global path.
__device__ __noinline__ void gram_block(const real_t *J, const real_t *D, const int *state,
    int rows, int nv, real_t *scratch, real_t *block_workspace, int block_cells);
} // namespace physics_newton::atomic_reduction
