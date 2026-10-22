#include "precision.cuh"
#pragma once
#include <cuda_runtime.h>
#include "../step/newton_types.cuh"

namespace physics_post {
// Collective, one block per world. qacc remains live throughout Post; workspace
// overlays only the completed Newton math scratch, never qacc itself.
// The caller supplies both views, so this object is independent of NV/layout.
// Workspace: 18*nbody + 6 + max(288, 6*nbody) real cells. Contact decoding
// and raw subtree forces reuse the final region after their lifetime barrier.
// outputs_required=false is admitted only inside the fixed native substep,
// before another complete forward and without an intervening force consumer.
// A deferred call publishes working qacc, invalidates Post, and returns false;
// it never stamps stale force/sensor arrays as current. Diagnostics always run.
__device__ __noinline__ bool run(const physics_newton::NewtonPostData &post, int world,
                                real_t *qacc, real_t *workspace, bool outputs_required,
                                const int *subtree_sizes = nullptr);
} // namespace physics_post
