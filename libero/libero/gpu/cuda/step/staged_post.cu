#include "precision.cuh"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <type_traits>
#include "staged_substep.cuh"
#include "substep_shared.cuh"
#include "substep_profile.cuh"
#include "newton_types.cuh"
#include "newton_shared_layout.cuh"
#include "staged_workspace.cuh"
#include "../postprocess/post_solver.cuh"
#include "../common/math.cuh"
#include "../common/vector.cuh"
#include "../common/warp.cuh"
#include "../common/packed_mass.cuh"
#include "../control/osc_runtime.cuh"
#include "../control/controller_pipeline.cuh"
#include "../runtime/forward.cuh"
#include "../runtime/position_reuse.cuh"
#include "../runtime/recovery.cuh"
using flash_libero::SubstepShape;
#include "substep_euler_math.cuh"

namespace flash_libero {
namespace {
#include "substep_forward_post.cuh"
#include "substep_tail.cuh"
}

// C: actual Post/publication, then the same adjacent control/check/Euler tail.
// A position miss suspends only that world; the existing Graph callback owns
// its multi-CTA rebuild and mixed-world preservation before the next entry.
__global__ void staged_post_kernel(const StagedSubstepArguments *request,
    SubstepEntry entry, int phase, bool policy_step, bool publish_post, bool run_tail) {
  const auto &s = *request;
  const auto &args = *s.arguments;
  const int w = blockIdx.y;
  if (s.phase[w] != phase) return;
  auto *shared = substep_shared_data();
  real_t *qacc = shared_at<real_t>(shared, StagedPostLayout::qacc);
  for (int j = threadIdx.x; j < SubstepShape::nv; j += blockDim.x)
    qacc[j] = s.qacc[size_t(w) * SubstepShape::nv + j];
  __syncthreads();
  CF_STEP_PROFILE_STAGES(CF_STEP_PROFILE_PHASE(Post));
  publish_forward<StagedPostLayout>(args, w, shared, phase < 2 ? policy_step : publish_post
#if defined(FLASH_SUBSTEP_PROFILE)
      , step_profile_stages
#endif
  );
  if (!run_tail) return; // explicit regional forward diagnostic only
  if (phase == 0) {
    if (threadIdx.x == 0) {
      const auto &smooth = args.smooth_commit;
      if (args.state.forward_valid[w] == 1 && !args.state.first_error[w] &&
          smooth.valid[w] == 1 && !smooth.error[w] && smooth.written[w] == smooth.epoch[w])
        s.smooth_reuse_status[w] = smooth.repaired[w] ? 25 : 0;
      const auto &in = args.controller_plan;
      osc_cuda::plan(w, in.active, in.first, in.needs, in.refresh, in.call, in.force);
    }
    begin_forward(args, w, args.controller_plan.refresh);
    if (threadIdx.x == 0) {
      if (args.position.work[w]) {
        args.continuation[w] = int(SubstepEntry::ControllerPosition);
        s.phase[w] = -1;
      } else s.phase[w] = 1;
    }
  } else if (phase == 1) {
    run_controller(args, w, policy_step);
    if (threadIdx.x == 0)
      cf_recovery::admit(w, args.state.intent, args.state.first_error, args.recovery.active);
    __syncthreads();
    check_and_reset(args, w, args.state.qpos, args.state.nq, 4);
    check_and_reset(args, w, args.state.qvel, args.state.nv, 5);
    begin_forward(args, w, args.state.intent);
    if (threadIdx.x == 0) {
      if (args.position.work[w]) {
        args.continuation[w] = int(SubstepEntry::PhysicsPosition);
        s.phase[w] = -1;
      } else s.phase[w] = 2;
    }
  } else {
    finish_substep(args, w, entry);
    if (threadIdx.x == 0) s.phase[w] = -1;
  }
}

__global__ void staged_reset_tail_kernel(const StagedSubstepArguments *request) {
  const auto &args = *request->arguments;
  const int w = blockIdx.y;
  if (args.continuation[w] != int(SubstepEntry::AccelerationReset)) return;
  finish_substep(args, w, SubstepEntry::AccelerationReset);
}
} // namespace flash_libero
