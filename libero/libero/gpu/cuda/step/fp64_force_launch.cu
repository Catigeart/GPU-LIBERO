// Copyright 2026 Flash-LIBERO contributors
// SPDX-License-Identifier: Apache-2.0
#include "fp64_actuation_solve.cuh"
#include "newton_shared_layout.cuh"
using namespace flash_libero;
extern "C" int flash_libero_force_handoff_version() { return 2; }
extern "C" size_t flash_libero_force_arguments_size() { return sizeof(SubstepArguments); }
extern "C" int flash_libero_force_nv() { return SubstepShape::nv; }
extern "C" int flash_libero_force_configure(int *regs, int *shared, int *local) {
  auto error = cudaFuncSetAttribute(force_residual_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, substep_shared_bytes);
  if (error != cudaSuccess) return int(error);
  const void *kernels[] = {reinterpret_cast<const void*>(fp64_actuation_solve_kernel),
                          reinterpret_cast<const void*>(force_residual_kernel)};
  for (int i = 0; i < 2; ++i) {
    cudaFuncAttributes a;
    error = cudaFuncGetAttributes(&a, kernels[i]);
    if (error != cudaSuccess) return int(error);
    regs[i] = a.numRegs; shared[i] = int(a.sharedSizeBytes); local[i] = int(a.localSizeBytes);
  }
  return 0;
}
extern "C" int flash_libero_launch_force(const SubstepArguments *arguments,
    const int *motion_ready, int *ready, int worlds, int entry, int threads, void *stream) {
  if (worlds < 1 || entry < 0 || entry > 2 ||
      (threads != 64 && threads != 128 && threads != 256)) return int(cudaErrorInvalidValue);
  fp64_actuation_solve_kernel<<<dim3(1,worlds),threads,0,static_cast<cudaStream_t>(stream)>>>(
      arguments,motion_ready,ready,static_cast<SubstepEntry>(entry));
  return int(cudaGetLastError());
}
extern "C" int flash_libero_launch_force_substep(const SubstepArguments *arguments,
    const int *plan, int *motion_ready, int *force_ready, int worlds, int policy,
    int entry_value, int post, int motion_threads, int force_threads, void *stream) {
  if (worlds < 1 || entry_value < 0 || entry_value > 3 || (post != 0 && post != 1) ||
      (motion_threads != 0 && motion_threads != 128 && motion_threads != 256 && motion_threads != 512) ||
      (force_threads != 0 && force_threads != 64 && force_threads != 128 && force_threads != 256))
    return int(cudaErrorInvalidValue);
  const auto entry = static_cast<SubstepEntry>(entry_value);
  const auto s = static_cast<cudaStream_t>(stream);
  const bool produce = motion_threads && entry != SubstepEntry::AccelerationReset;
  if (produce) {
    fp64_motion_passive_kernel<<<dim3(1,worlds),motion_threads,0,s>>>(arguments,plan,motion_ready,entry);
    auto error = cudaGetLastError(); if (error != cudaSuccess) return int(error);
    if (force_threads) {
      const int status = flash_libero_launch_force(arguments,motion_ready,force_ready,
          worlds,entry_value,force_threads,stream);
      if (status) return status;
    }
  }
  force_residual_kernel<<<dim3(1,worlds),substep_threads,substep_shared_bytes,s>>>(arguments,
      policy != 0,entry,post != 0,produce ? motion_ready : nullptr,
      produce && force_threads ? force_ready : nullptr);
  return int(cudaGetLastError());
}
