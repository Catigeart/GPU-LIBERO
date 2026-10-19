#pragma once
#include <cuda_runtime.h>
namespace flash_libero {
struct ControllerSolveData;
}
namespace physics_controller {
// Fixed native OSC. All world-block lanes participate; host binding proves
// disjoint workspace owners. shared points to the existing substep arena.
// Returns the same success predicate on all lanes. Every return follows a
// block rendezvous; no output is written after that rendezvous.
__device__ __noinline__ bool solve(const flash_libero::ControllerSolveData &inputs, int world,
                                   unsigned char *shared);
} // namespace physics_controller
