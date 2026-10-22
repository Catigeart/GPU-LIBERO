#include "precision.cuh"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <type_traits>
#include <cublasdx.hpp>
#include "staged_substep.cuh"
#include "substep_shared.cuh"
#include "substep_profile.cuh"
#include "../common/math.cuh"
#include "../common/vector.cuh"
#include "../common/warp.cuh"
#include "../common/packed_mass.cuh"
#include "forward_mass.cuh"
#include "../runtime/forward.cuh"
using flash_libero::SubstepShape;
#include "substep_forward_math.cuh"

namespace flash_libero {
namespace {
#include "staged_phase.cuh"
#include "substep_forward_prepare.cuh"
}
} // namespace flash_libero
#include "staged_motion_math.cuh"
namespace flash_libero {

// M: one eight-lane subgroup walks each branch in registers. Passive
// producers keep their existing FP64 algorithms and status semantics.
__global__ void staged_motion_kernel(const StagedSubstepArguments *request,
                                    SubstepEntry entry, int phase) {
  const auto &s = *request;
  const auto &args = *s.arguments;
  const int w = blockIdx.y;
  if (!staged_enter(s, args, w, entry, phase)) return;
  const bool reuse = forward_reuse(args, w);
  if (threadIdx.x == 0) s.motion_reuse[w] = int(reuse);
  prepare_velocity(args, w, reuse, BranchMotion{s.motion_plan});
}
} // namespace flash_libero
