// Copyright 2026 Flash-LIBERO contributors
// SPDX-License-Identifier: Apache-2.0
// One dimension-general Euler entry, with its own register/shared allocation.
#include "precision.cuh"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include "staged_substep.cuh"
#include "substep_shared.cuh"
#include "substep_profile.cuh"
#include "../common/vector.cuh"
#include "substep_euler_math.cuh"

namespace flash_libero {
namespace {
#include "substep_integrate.cuh"
}
__global__ void aligned_euler_kernel(const StagedSubstepArguments *request) {
  const int world = blockIdx.y;
  if (request->phase[world] != staged_euler_ready) return;
  const auto &args = *request->arguments;
  integrate(args, world);
  if (threadIdx.x == 0) {
    args.continuation[world] = int(SubstepEntry::Complete);
    request->phase[world] = -1;
  }
}
} // namespace flash_libero

extern "C" std::size_t flash_libero_aligned_euler_shared_bytes() {
  return flash_libero::EulerLayout::bytes;
}
extern "C" int flash_libero_euler_kernel_configure(int *resources) {
  constexpr auto bytes = flash_libero::EulerLayout::bytes;
  auto error = cudaFuncSetAttribute(flash_libero::aligned_euler_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  if (error != cudaSuccess) return int(error);
  cudaFuncAttributes a;
  error = cudaFuncGetAttributes(&a, flash_libero::aligned_euler_kernel);
  if (error != cudaSuccess) return int(error);
  resources[0] = a.numRegs; resources[1] = a.sharedSizeBytes;
  resources[2] = a.localSizeBytes; resources[3] = bytes;
  return 0;
}
