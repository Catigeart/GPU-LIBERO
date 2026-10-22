// Copyright 2026 Flash-LIBERO contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "substep_types.cuh"
namespace flash_libero {
// The descriptor is shared by worlds: the existing branch plan followed by
// one contiguous-subtree size per body. Ready is private to this launch pair;
// it is not the complete forward cache or an independently reusable epoch.
__global__ void fp64_motion_passive_kernel(const SubstepArguments *, const int *plan,
                                          int *ready, SubstepEntry entry);
__global__ void motion_residual_kernel(const SubstepArguments *, bool policy_step,
                                      SubstepEntry entry, bool publish_post, const int *ready);
} // namespace flash_libero
