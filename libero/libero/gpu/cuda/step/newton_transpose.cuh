#include "precision.cuh"
#pragma once
#include <cuda_runtime.h>

namespace physics_newton::transpose_reduction {
// Selected-precision device functions. Input force must already be published.
// The caller publishes out and closes the borrowed shared arena's lifetime.
// Groups partition rows; columns within each group remain contiguous.
enum class Method : int { Direct, Shared32, Shared16, Shared8, Subwarp4 };

// Selection depends on matrix shape and the actual calling CTA, never task ID.
// Larger CTAs retain the previously qualified shared-partial organization.
__host__ __device__ constexpr Method select_auto(int nv, int rows, int threads) {
  if (threads >= 128 && rows >= 128 && nv <= 64)
    return Method::Shared16;
#if FLASH_REAL_BITS == 64
  if (threads == 64 && nv <= 48 && rows >= 128 && rows <= 512)
    return Method::Subwarp4;
#endif
  return Method::Direct;
}
// Explicit entry avoids adding selector/branch cost to existing dense paths.
// A Subwarp4 selection uses this entry; dense accepts the Direct/Shared modes.
__device__ __noinline__ void subwarp4(const real_t *J, const real_t *force,
    int rows, int nv, real_t *out);
__device__ __noinline__ void dense(const real_t *J, const real_t *force,
    int rows, int nv, real_t *out, real_t *shared, Method method);
} // namespace physics_newton::transpose_reduction
