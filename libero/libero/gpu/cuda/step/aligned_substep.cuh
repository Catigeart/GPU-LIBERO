// Copyright 2026 Flash-LIBERO contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "staged_substep.cuh"
namespace flash_libero {
// The first member is the existing A/B/C handoff, consumed unchanged by Post.
struct AlignedSubstepArguments {
  StagedSubstepArguments stages;
  int *force_ready;
  unsigned char *newton_initial;
};
static_assert(sizeof(AlignedSubstepArguments) == 8 * sizeof(void *), "aligned ABI");
__global__ void aligned_prepare_kernel(const AlignedSubstepArguments *, SubstepEntry, int, bool, bool);
__global__ void aligned_newton_initial_kernel(const AlignedSubstepArguments *, int);
__global__ void aligned_newton_resume_kernel(const AlignedSubstepArguments *, int);
} // namespace flash_libero
