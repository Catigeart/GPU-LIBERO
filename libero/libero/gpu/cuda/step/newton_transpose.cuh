#pragma once
#include <cuda_runtime.h>

namespace physics_newton::transpose_reduction {
// Independent FP64 device functions. Input force must already be published.
// The caller publishes out and closes the borrowed shared arena's lifetime.
// Groups partition rows; columns within each group remain contiguous.
enum class Method : int { Direct, Shared32, Shared16, Shared8 };
__device__ __noinline__ void dense(const double *J, const double *force,
    int rows, int nv, double *out, double *shared, Method method);
} // namespace physics_newton::transpose_reduction
