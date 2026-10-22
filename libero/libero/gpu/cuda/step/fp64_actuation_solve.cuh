// Copyright 2026 Flash-LIBERO contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "fp64_motion_passive.cuh"
namespace flash_libero {
// This private producer/consumer handoff covers the first forward of an entry.
// The controller's later ctrl update is deliberately outside its lifetime.
__global__ void fp64_actuation_solve_kernel(const SubstepArguments *,
    const int *motion_ready, int *ready, SubstepEntry);
__global__ void force_residual_kernel(const SubstepArguments *, bool,
    SubstepEntry, bool, const int *motion_ready, const int *force_ready);
} // namespace flash_libero
