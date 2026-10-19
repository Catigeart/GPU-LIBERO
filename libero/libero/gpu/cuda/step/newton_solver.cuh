#pragma once
#include <cuda_runtime.h>
#include "newton_types.cuh"

namespace physics_newton {

// CPU MuJoCo 2.3.7 is the production stopping policy. The previous roundoff
// rule is retained for explicit comparisons, never selected implicitly.
enum class StopPolicy : int { Cpu = 0, RoundoffResidual = 1 };
struct NewtonOptions {
  int stop_policy = int(StopPolicy::Cpu);
  // Optional per-world audit: eight int64 counters and eight first-event
  // scalars. Null pointers incur no residual audit or global audit writes.
  int64_t *audit = nullptr;
  double *first_disagreement = nullptr;
  // A completed smooth phase may hand its shared inputs directly to Newton.
  // Null keeps the canonical global path for skipped/recovery forwards.
  double *smooth_acceleration = nullptr;
  double *smooth_force = nullptr;
};

// All block threads participate; shared is the parent kernel's workspace.
__device__ __noinline__ void solve_newton(unsigned char *shared, const NewtonModelData &model,
    const NewtonWorldData &world, const int *upstream, const int *enabled,
    const int *publication, const int64_t *plan, const NewtonOptions &options);
} // namespace physics_newton
