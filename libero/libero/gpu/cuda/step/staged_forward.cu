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
#include "substep_forward_prepare.cuh"
}

// A: existing velocity/reference/bias/actuation/smooth mathematics and bridge.
// No Newton state, factor or matrix is spilled across this boundary.
__global__ void staged_prepare_kernel(const StagedSubstepArguments *request,
                                     SubstepEntry entry, int phase) {
  const auto &s = *request;
  const auto &args = *s.arguments;
  const int w = blockIdx.y;
  const int first = entry == SubstepEntry::Start ? 0 :
                    entry == SubstepEntry::ControllerPosition ? 1 : 2;
  if (phase == first) {
    if (threadIdx.x == 0) {
      s.phase[w] = entry == SubstepEntry::Start || args.continuation[w] == int(entry) ? phase : -1;
      s.smooth_reuse_status[w] = -1;
      if (entry == SubstepEntry::Start) {
        args.continuation[w] = int(SubstepEntry::Complete);
        args.forward_workspace.metadata[size_t(w) * ForwardCacheWords + ForwardCacheValid] = 0;
      }
    }
    __syncthreads();
  }
  if (s.phase[w] != phase) return;
  const bool reuse = prepare_dynamics(args, w);
  CF_STEP_PROFILE_STAGES(CF_STEP_PROFILE_PHASE(Actuation));
  prepare_solver_inputs(args, w, substep_shared_data(), reuse,
      phase == 1 ? s.smooth_reuse_status[w] : -1
#if defined(FLASH_SUBSTEP_PROFILE)
      , step_profile_stages
#endif
  );
}
} // namespace flash_libero
