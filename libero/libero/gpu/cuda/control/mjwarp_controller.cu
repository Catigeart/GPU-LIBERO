// MJWarp supplies resident physical inputs to the maintained CUDA OSC.
// This bridge owns no physics equations or alternative controller mathematics.
#include "../step/substep_shared_layout.cuh"
#include "../step/substep_types.cuh"
#include "osc_solver.cuh"

namespace {
template <bool Publish = false>
__global__ void control(const flash_libero::ControllerSolveData *inputs,
                        const int *ids, real_t *target, int stride) {
  extern __shared__ __align__(16) unsigned char shared[];
  physics_controller::solve(*inputs, blockIdx.x, shared);
  // Normal torque publication already ends in the maintained solver's block
  // barrier. Inactive/error paths retain ctrl without a current write. Thus
  // every source below is ready without adding another barrier here.
  if constexpr (Publish)
    if (threadIdx.x < 9) {
      const int w = blockIdx.x, x = threadIdx.x;
      target[w * stride + ids[x]] = inputs->ctrl[9 * w + x];
    }
}
} // namespace
extern "C" int flash_mjwarp_controller_arch() { return FLASH_CUDA_ARCH; }
extern "C" int flash_mjwarp_controller_scalar_bytes() { return sizeof(real_t); }
extern "C" size_t flash_mjwarp_controller_arguments_bytes() {
  return sizeof(flash_libero::ControllerSolveData);
}
extern "C" int
flash_mjwarp_controller_launch(const flash_libero::ControllerSolveData *inputs,
                               int worlds, cudaStream_t stream) {
  control<false>
      <<<worlds, flash_libero::SubstepShape::threads,
         flash_libero::OscLayout::bytes, stream>>>(inputs, nullptr, nullptr, 0);
  return int(cudaGetLastError());
}

extern "C" int
flash_mjwarp_controller_publish(const flash_libero::ControllerSolveData *inputs,
                                int worlds, const int *ids, real_t *target,
                                int stride, cudaStream_t stream) {
  if (!inputs || !ids || !target || worlds < 1 || stride < 9)
    return cudaErrorInvalidValue;
  control<true>
      <<<worlds, flash_libero::SubstepShape::threads,
         flash_libero::OscLayout::bytes, stream>>>(inputs, ids, target, stride);
  return cudaGetLastError();
}
