#pragma once
#include "substep_types.cuh"
namespace flash_libero {
__global__ void fused_substep_kernel(const SubstepArguments *, bool, SubstepEntry, bool publish_post);
__global__ void select_substep_continuation(const SubstepArguments *, SubstepEntry, int, int *);
} // namespace flash_libero
