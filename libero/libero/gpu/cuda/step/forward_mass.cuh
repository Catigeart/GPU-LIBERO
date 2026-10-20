#include "precision.cuh"
#pragma once
#include "../common/packed_mass.cuh"

namespace flash_libero::forward_mass {

// Immutable model descriptor, built once by forward_plan.py. Neither topology
// inspection nor factor staging is needed on an admitted position-factor hit.
struct Plan {
  const int *bounds;
  const int *depth;
  const unsigned *ancestors;
  int roots, width;
};

__device__ __forceinline__ Plan plan(const int *data) {
  const int nv = data[0], roots = data[1];
  const int *bounds = data + 3, *depth = bounds + roots + 1;
  return {bounds, depth, reinterpret_cast<const unsigned *>(depth + nv), roots, data[2]};
}

// One value per lane stays in registers across both triangular solves.
// Different components cooperate in independent subwarps. Within each output,
// descending descendant updates and ascending-row/parent-order substitution
// match mj_solveLD's single-RHS order, including its zero-descendant skip.
template <int Width>
__device__ __forceinline__ bool solve_subwarps(const real_t *rhs, real_t *solution,
    const real_t *ld, const real_t *inv, const int *madr, const int *parent, Plan topology) {
  const int lane = threadIdx.x % Width;
  const int group = threadIdx.x / Width;
  const int groups = blockDim.x / Width;
  const unsigned mask = (0xffffffffu >> (32 - Width)) << ((threadIdx.x & 31) - lane);
  bool bad = false;
  for (int root = group; root < topology.roots; root += groups) {
    const int begin = topology.bounds[root], end = topology.bounds[root + 1];
    const int length = end - begin, row = begin + lane;
    const bool live = lane < length;
    real_t value = live ? rhs[row] : REAL_C(0.0);
    const int depth = live ? topology.depth[row] : 0;
    const unsigned ancestors = live ? topology.ancestors[row] : 0;
    const int address = live ? madr[row] : 0;
    for (int i = length - 1; i >= 0; --i) {
      const real_t descendant = __shfl_sync(mask, value, i, Width);
      const unsigned affected = __shfl_sync(mask, ancestors, i, Width);
      const int distance = __shfl_sync(mask, depth, i, Width) - depth;
      const int source = __shfl_sync(mask, address, i, Width);
      if (descendant != REAL_C(0.0) && (affected & (1u << lane))) {
        value -= ld[source + distance] * descendant;
        bad |= !isfinite(value);
      }
    }
    if (live) {
      value *= inv[row];
      bad |= !isfinite(value);
    }
    for (int i = 0; i < length; ++i) {
      int coefficient = madr[begin + i] + 1;
      for (int p = parent[begin + i]; p >= begin; p = parent[p], ++coefficient) {
        const real_t ancestor = __shfl_sync(mask, value, p - begin, Width);
        if (lane == i) {
          value -= ld[coefficient] * ancestor;
          bad |= !isfinite(value);
        }
      }
    }
    if (live)
      solution[row] = value;
  }
  return __syncthreads_or(bad) == 0;
}

__device__ __forceinline__ bool solve(const real_t *rhs, real_t *solution,
    const real_t *ld, const real_t *inv, const int *madr, const int *parent,
    const int *descriptor) {
  const Plan topology = plan(descriptor);
  if (topology.width == 8)
    return solve_subwarps<8>(rhs, solution, ld, inv, madr, parent, topology);
  if (topology.width == 16)
    return solve_subwarps<16>(rhs, solution, ld, inv, madr, parent, topology);
  if (topology.width == 32)
    return solve_subwarps<32>(rhs, solution, ld, inv, madr, parent, topology);
  // Large or interleaved components retain the general implementation.
  const auto forest = libero_packed_mass::inspect_forest_block(descriptor[0], parent);
  return libero_packed_mass::solve_rhs_block(rhs, solution, ld, inv, 1,
                                             descriptor[0], madr, parent, forest);
}

} // namespace flash_libero::forward_mass
