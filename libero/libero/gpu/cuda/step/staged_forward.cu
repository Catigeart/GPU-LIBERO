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

// A: existing velocity/reference/bias/actuation/smooth mathematics and bridge.
// No Newton state, factor or matrix is spilled across this boundary.
__global__ void staged_prepare_kernel(const StagedSubstepArguments *request,
                                     SubstepEntry entry, int phase, bool motion_prepared) {
  const auto &s = *request;
  const auto &args = *s.arguments;
  const int w = blockIdx.y;
  if (!motion_prepared) {
    if (!staged_enter(s, args, w, entry, phase)) return;
  } else if (s.phase[w] != phase) return;
  const bool reuse = prepare_dynamics(args, w, motion_prepared ? s.motion_reuse : nullptr);
  CF_STEP_PROFILE_STAGES(CF_STEP_PROFILE_PHASE(Actuation));
  prepare_solver_inputs(args, w, substep_shared_data(), reuse,
      phase == 1 ? s.smooth_reuse_status[w] : -1
#if defined(FLASH_SUBSTEP_PROFILE)
      , step_profile_stages
#endif
  );
}
} // namespace flash_libero
