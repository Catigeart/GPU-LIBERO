#pragma once
#include <cuda_runtime.h>
namespace flash_libero { struct SubstepArguments; }
namespace physics_controller {
// All world-block lanes enter. Uses the existing shared OSC arena. On return,
// the caller must rendezvous before another lane reads final publication flags
// or reuses the arena. The fused caller combines this with recovery admission.
__device__ __noinline__ void run_controller(const flash_libero::SubstepArguments &inputs,
                                          int world, bool policy_step, unsigned char *shared);
}
