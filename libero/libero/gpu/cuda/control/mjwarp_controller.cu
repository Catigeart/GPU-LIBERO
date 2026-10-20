// MJWarp supplies resident physical inputs to the maintained CUDA OSC.
// This bridge owns no physics equations or alternative controller mathematics.
#include "osc_solver.cuh"
#include "../step/substep_types.cuh"
#include "../step/substep_shared_layout.cuh"

namespace {
__global__ void control(const flash_libero::ControllerSolveData *inputs) {
  extern __shared__ __align__(16) unsigned char shared[];
  physics_controller::solve(*inputs, blockIdx.x, shared);
}
}
extern "C" int flash_mjwarp_controller_scalar_bytes() { return sizeof(real_t); }
extern "C" size_t flash_mjwarp_controller_arguments_bytes() {
  return sizeof(flash_libero::ControllerSolveData);
}
extern "C" int flash_mjwarp_controller_launch(const flash_libero::ControllerSolveData *inputs,
                                              int worlds, cudaStream_t stream) {
  control<<<worlds,flash_libero::SubstepShape::threads,flash_libero::OscLayout::bytes,stream>>>(inputs);
  return int(cudaGetLastError());
}
