#include "precision.cuh"
#pragma once
#include <cuda_runtime.h>
#include "newton_types.cuh"

namespace physics_newton {

// Post-iteration stopping follows CPU MuJoCo 2.3.7. FP32 also permits the
// same gradient criterion at entry. The previous roundoff rule is retained
// for explicit comparisons, never selected implicitly.
enum class StopPolicy : int { Cpu = 0, RoundoffResidual = 1 };
struct NewtonOptions {
  int stop_policy = int(StopPolicy::Cpu);
  // FP32 may already satisfy the ordinary normalized-gradient criterion at
  // the selected initial acceleration. Preserve every solve and publication,
  // but avoid factoring a Hessian solely to apply an unresolved correction.
  bool initial_gradient_stop = FLASH_REAL_BITS == 32;
  // Optional per-world audit: eight int64 counters and eight first-event
  // scalars. Null pointers incur no residual audit or global audit writes.
  int64_t *audit = nullptr;
  real_t *first_disagreement = nullptr;
  // A completed smooth phase may hand its shared inputs directly to Newton.
  // Null keeps the canonical global path for skipped/recovery forwards.
  real_t *smooth_acceleration = nullptr;
  real_t *smooth_force = nullptr;
};

// Full preserves the ordinary single-call entry. The optional two-kernel
// schedule carries only initialization's live state in an owned checkpoint.
enum class NewtonPhase { Full, Initialize, Resume };

// All block threads participate; shared is the parent kernel's workspace.
template<NewtonPhase Phase = NewtonPhase::Full>
__device__ __noinline__ void solve_newton(unsigned char *shared, const NewtonModelData &model,
    const NewtonWorldData &world, const int *upstream, const int *enabled,
    const int *publication, const int64_t *plan, const NewtonOptions &options,
    unsigned char *checkpoint = nullptr);
} // namespace physics_newton
