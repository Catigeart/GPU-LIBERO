// Copyright 2026 Flash-LIBERO contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "fp64_motion_passive.cuh"
namespace flash_libero {
// All block threads enter. The caller supplies flat shared scalar/DOF views;
// no matrix/factor or parameter pack is staged in that arena.
__device__ bool prepare_fp64_force(const SubstepArguments &, int world,
    const int *motion_ready, double *scalar_force, double *force, double *acceleration);
// Private handoff: 0 native fallback; 1 complete global entry outputs;
// 2 current-controller outputs also resident in the parent's shared arena.
__global__ void fp64_actuation_solve_kernel(const SubstepArguments *,
    const int *motion_ready, int *ready, SubstepEntry);
__global__ void force_residual_kernel(const SubstepArguments *, bool,
    SubstepEntry, bool, const int *motion_ready, int *force_ready);
} // namespace flash_libero
