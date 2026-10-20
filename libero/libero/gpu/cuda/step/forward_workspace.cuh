#include "precision.cuh"
#pragma once
#include <cstdint>

namespace flash_libero {

// Persistent device storage, partitioned by world. Numerical velocity/bias/
// reference outputs already have retained global owners; only their validity
// and fluid temporaries need a new allocation. OSC/Newton may overwrite the
// shared arena, so no cached pointer refers to that arena.
enum ForwardCacheWord : int {
  ForwardCacheValid,
  ForwardCacheState,
  ForwardCachePosition,
  ForwardCacheComputed,
  ForwardCacheReused,
  ForwardCacheInvalidated,
  ForwardCacheWords
};

struct ForwardWorkspaceData {
  uint64_t *metadata;                  // [world][ForwardCacheWords]
  const uint64_t *position_generation; // advances on every successful prefix rebuild
  real_t *fluid;                      // [world][9*nbody + nv]: body payloads, candidate force
  const int *mass_plan;                // immutable, shared by worlds; forward_mass.cuh
};

} // namespace flash_libero
