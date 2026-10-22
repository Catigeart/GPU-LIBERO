#pragma once
#include "substep_kernel.cuh"

namespace flash_libero {
constexpr int staged_euler_ready = 3; // checks completed; Euler still owns its publication gate
// Private handoff, allocated once before capture. The public substep/Newton
// argument layouts and numerical workspaces retain their existing ABI.
struct StagedSubstepArguments {
  const SubstepArguments *arguments;
  int *phase;                       // [world], -1 means suspended/completed
  int *smooth_reuse_status;         // [world], dies at position suspension
  real_t *qacc;                     // [world][NV], Newton-private Post input
  int *motion_reuse;               // [world], produced before reference admission
  const int *motion_plan;           // immutable branch paths, shared by worlds
};
static_assert(sizeof(StagedSubstepArguments) == 6 * sizeof(void *), "staged ABI");
__global__ void staged_prepare_kernel(const StagedSubstepArguments *, SubstepEntry, int, bool);
__global__ void staged_motion_kernel(const StagedSubstepArguments *, SubstepEntry, int);
__global__ void staged_newton_kernel(const StagedSubstepArguments *, int);
__global__ void staged_post_kernel(const StagedSubstepArguments *, SubstepEntry, int, bool, bool, bool);
__global__ void staged_post_euler_kernel(const StagedSubstepArguments *, SubstepEntry, int, bool, bool, bool);
__global__ void aligned_euler_kernel(const StagedSubstepArguments *);
__global__ void staged_reset_tail_kernel(const StagedSubstepArguments *);
} // namespace flash_libero
