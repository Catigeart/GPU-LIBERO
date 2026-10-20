#include "precision.cuh"
#pragma once
#include <cuda_runtime.h>

namespace physics_newton::transpose_reduction {
// Independent FP64 device functions. Input force must already be published.
// The caller publishes out and closes the borrowed shared arena's lifetime.
// Groups partition rows; columns within each group remain contiguous.
enum class Method : int { Direct, Shared32, Shared16, Shared8 };
__device__ __noinline__ void dense(const real_t *J, const real_t *force,
    int rows, int nv, real_t *out, real_t *shared, Method method);
} // namespace physics_newton::transpose_reduction
