// Copyright 2026 Flash-LIBERO contributors
// SPDX-License-Identifier: Apache-2.0
#include "aligned_substep.cuh"
#include "newton_workspace.cuh"
#include "staged_workspace.cuh"
using namespace flash_libero;
extern "C" size_t flash_libero_aligned_euler_shared_bytes();
extern "C" int flash_libero_euler_kernel_configure(int *);
extern "C" int flash_libero_aligned_euler_configure(int *resources, int bodies) {
  auto status = flash_libero_euler_kernel_configure(resources);
  if (status) return status;
  const auto bytes = staged_post_bytes(bodies);
  auto error = cudaFuncSetAttribute(staged_post_euler_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  if (error != cudaSuccess) return int(error);
  cudaFuncAttributes a;
  error = cudaFuncGetAttributes(&a, staged_post_euler_kernel);
  if (error != cudaSuccess) return int(error);
  resources[4] = a.numRegs; resources[5] = a.sharedSizeBytes;
  resources[6] = a.localSizeBytes; resources[7] = bytes;
  return 0;
}
extern "C" size_t flash_libero_aligned_arguments_size() { return sizeof(AlignedSubstepArguments); }
extern "C" size_t flash_libero_aligned_checkpoint_bytes() { return physics_newton::NewtonInitialLayout::bytes; }
extern "C" int flash_libero_aligned_initial_thread_limit() {
  // The optional MathDx GEMV fallback is a fixed 64-thread collective.
  return FLASH_NEWTON_NATIVE_GEMV ? 256 : 64;
}
extern "C" int flash_libero_aligned_configure(int *regs, int *shared, int *local, int *dynamic, int bodies) {
  const void *kernels[] = {reinterpret_cast<const void *>(aligned_prepare_kernel),
      reinterpret_cast<const void *>(aligned_newton_initial_kernel),
      reinterpret_cast<const void *>(aligned_newton_resume_kernel),
      reinterpret_cast<const void *>(staged_post_kernel),
      reinterpret_cast<const void *>(staged_reset_tail_kernel)};
  const size_t bytes[] = {SmoothLayout::bytes, physics_newton::NewtonInitialSharedLayout::bytes,
      NewtonLayout::bytes, staged_post_bytes(bodies), EulerLayout::bytes};
  for (int i = 0; i < 5; ++i) {
    auto error = cudaFuncSetAttribute(kernels[i], cudaFuncAttributeMaxDynamicSharedMemorySize, bytes[i]);
    if (error != cudaSuccess) return int(error);
    cudaFuncAttributes a;
    error = cudaFuncGetAttributes(&a, kernels[i]);
    if (error != cudaSuccess) return int(error);
    regs[i] = a.numRegs; shared[i] = a.sharedSizeBytes; local[i] = a.localSizeBytes; dynamic[i] = bytes[i];
  }
  return 0;
}
// Motion/entry-force are enqueued by the existing accepted host owner before
// this chain. Same-stream Graph ordering is the only inter-kernel publication.
template<bool SeparateEuler, bool SeparateController = false>
static int launch_aligned(const AlignedSubstepArguments *args,
    int worlds, int policy, int entry_value, int post, int motion_enabled, int force_enabled,
    int initial_threads, int post_bytes, void *stream) {
  if (worlds <= 0 || entry_value < 0 || entry_value > 3 || post_bytes <= 0 ||
      initial_threads > flash_libero_aligned_initial_thread_limit() ||
      (initial_threads != 64 && initial_threads != 128 && initial_threads != 256))
    return int(cudaErrorInvalidValue);
  const auto entry = static_cast<SubstepEntry>(entry_value);
  const auto s = static_cast<cudaStream_t>(stream);
  const dim3 grid(1, worlds);
  const auto *stages = reinterpret_cast<const StagedSubstepArguments *>(args);
  if (entry == SubstepEntry::AccelerationReset) {
    staged_reset_tail_kernel<<<grid,64,EulerLayout::bytes,s>>>(stages);
    return int(cudaGetLastError());
  }
  const int first = entry == SubstepEntry::Start ? 0 : entry == SubstepEntry::ControllerPosition ? 1 : 2;
  for (int phase = first; phase < 3; ++phase) {
    aligned_prepare_kernel<<<grid,64,SmoothLayout::bytes,s>>>(args, entry, phase, motion_enabled, force_enabled);
    auto error = cudaGetLastError(); if (error != cudaSuccess) return int(error);
    aligned_newton_initial_kernel<<<grid,initial_threads,physics_newton::NewtonInitialSharedLayout::bytes,s>>>(args,phase);
    error = cudaGetLastError(); if (error != cudaSuccess) return int(error);
    aligned_newton_resume_kernel<<<grid,64,NewtonLayout::bytes,s>>>(args,phase);
    error = cudaGetLastError(); if (error != cudaSuccess) return int(error);
    if constexpr (SeparateEuler) {
      if constexpr (SeparateController) {
        staged_post_controller_kernel<<<grid,64,post_bytes,s>>>(stages,entry,phase,policy != 0,post != 0,true);
        error = cudaGetLastError(); if (error != cudaSuccess) return int(error);
        if (phase == 1) {
          aligned_controller_kernel<<<grid,64,OscLayout::bytes,s>>>(stages,policy != 0);
        }
      } else {
        staged_post_euler_kernel<<<grid,64,post_bytes,s>>>(stages,entry,phase,policy != 0,post != 0,true);
      }
      error = cudaGetLastError(); if (error != cudaSuccess) return int(error);
      if (phase == 2) {
        aligned_euler_kernel<<<grid,64,flash_libero_aligned_euler_shared_bytes(),s>>>(stages);
        error = cudaGetLastError(); if (error != cudaSuccess) return int(error);
      }
    } else {
      staged_post_kernel<<<grid,64,post_bytes,s>>>(stages,entry,phase,policy != 0,post != 0,true);
      error = cudaGetLastError(); if (error != cudaSuccess) return int(error);
    }
  }
  return 0;
}

extern "C" int flash_libero_launch_aligned_substep(const AlignedSubstepArguments *args,
    int worlds, int policy, int entry, int post, int motion, int force,
    int initial_threads, int post_bytes, void *stream) {
  return launch_aligned<false>(args, worlds, policy, entry, post, motion, force,
      initial_threads, post_bytes, stream);
}
extern "C" int flash_libero_launch_aligned_euler_substep(const AlignedSubstepArguments *args,
    int worlds, int policy, int entry, int post, int motion, int force,
    int initial_threads, int post_bytes, void *stream) {
  if (SubstepShape::nv > 57 || SubstepShape::nq > 65 ||
      SubstepShape::packed_mass > 212 || SubstepShape::njnt > 18)
    return int(cudaErrorInvalidValue);
  return launch_aligned<true>(args, worlds, policy, entry, post, motion, force,
      initial_threads, post_bytes, stream);
}

extern "C" int flash_libero_controller_kernel_configure(int *);
extern "C" int flash_libero_aligned_controller_configure(int *resources, int bodies) {
  auto status = flash_libero_controller_kernel_configure(resources);
  if (status) return status;
  const auto bytes = staged_post_bytes(bodies);
  auto error = cudaFuncSetAttribute(staged_post_controller_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  if (error != cudaSuccess) return int(error);
  cudaFuncAttributes a;
  error = cudaFuncGetAttributes(&a, staged_post_controller_kernel);
  if (error != cudaSuccess) return int(error);
  resources[4] = a.numRegs; resources[5] = a.sharedSizeBytes;
  resources[6] = a.localSizeBytes; resources[7] = bytes;
  return 0;
}
extern "C" int flash_libero_launch_aligned_controller_substep(const AlignedSubstepArguments *args,
    int worlds, int policy, int entry, int post, int motion, int force,
    int initial_threads, int post_bytes, void *stream) {
  if (SubstepShape::nv > 57 || SubstepShape::nq > 65 ||
      SubstepShape::packed_mass > 212 || SubstepShape::njnt > 18)
    return int(cudaErrorInvalidValue);
  return launch_aligned<true, true>(args, worlds, policy, entry, post, motion, force,
      initial_threads, post_bytes, stream);
}
