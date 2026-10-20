#pragma once
#include "precision.cuh"
#include <cstddef>

namespace physics_newton {
// Byte layout of one warp's compact (J, D*J, column-index) row. Both the
// direct Gram and optional Jacobian product cache use these same offsets.
struct GramRowLayout {
  static constexpr std::size_t values = 0;
  __host__ __device__ static constexpr std::size_t weighted(int nv) {
    return std::size_t(nv) * sizeof(real_t);
  }
  __host__ __device__ static constexpr std::size_t columns(int nv) {
    return 2 * weighted(nv);
  }
  __host__ __device__ static constexpr std::size_t bytes(int nv) {
    return columns(nv) + (std::size_t(nv) * sizeof(int) + sizeof(real_t) - 1) /
                            sizeof(real_t) * sizeof(real_t);
  }
  // Preserve the existing eight-byte boundary after the int counter plane.
  __host__ __device__ static constexpr std::size_t after_counters(int cells) {
    return (std::size_t(cells) * sizeof(int) + 7) / 8 * 8;
  }
};
} // namespace physics_newton
