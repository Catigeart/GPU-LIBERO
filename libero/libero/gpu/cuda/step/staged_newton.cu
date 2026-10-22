#include "precision.cuh"
#include "staged_substep.cuh"
#include "substep_shared.cuh"
#include "newton_solver.cuh"
#include "newton_shared_layout.cuh"

namespace flash_libero {
namespace {
#include "substep_forward_newton.cuh"
}

// B: the exact production solver and its complete shared workspace lifetime.
__global__ void staged_newton_kernel(const StagedSubstepArguments *request, int phase) {
  const auto &s = *request;
  const int w = blockIdx.y;
  if (s.phase[w] != phase) return;
  const auto &args = *s.arguments;
  auto *shared = substep_shared_data();
  const auto &smooth = args.smooth_commit;
  const bool resident = args.state.requested_mask[w] != 0 && smooth.valid[w] == 1 &&
      !smooth.error[w] && smooth.written[w] == smooth.epoch[w];
  if (resident) {
    real_t *force = shared_at<real_t>(shared, SmoothLayout::force);
    real_t *acc = shared_at<real_t>(shared, SmoothLayout::acceleration);
    for (int j = threadIdx.x; j < SubstepShape::nv; j += blockDim.x) {
      force[j] = smooth.pf[size_t(w) * SubstepShape::nv + j];
      acc[j] = smooth.pa[size_t(w) * SubstepShape::nv + j];
    }
  }
  // solve_newton's existing entry barrier precedes its first smooth reader.
  run_forward_newton(args, w, shared);
  const real_t *qacc = shared_at<real_t>(shared, NewtonLayout::qacc);
  for (int j = threadIdx.x; j < SubstepShape::nv; j += blockDim.x)
    s.qacc[size_t(w) * SubstepShape::nv + j] = qacc[j];
}
} // namespace flash_libero
