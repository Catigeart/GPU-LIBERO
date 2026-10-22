// Copyright 2026 Flash-LIBERO contributors
// SPDX-License-Identifier: Apache-2.0
#include "precision.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <string.h>
#include "staged_substep.cuh"
#include "substep_shared.cuh"
#include "substep_profile.cuh"
#include "../common/vector.cuh"
#include "../common/warp.cuh"
#include "../control/osc_runtime.cuh"
#include "../control/controller_pipeline.cuh"
#include "../runtime/forward.cuh"
// These legacy device bodies have external linkage in their original owner.
// The independent entry keeps its copies private without changing that ABI.
namespace {
#include "../runtime/position_reuse.cuh"
#include "../runtime/recovery.cuh"
}
#include "substep_euler_math.cuh"
namespace flash_libero {
namespace {
#include "substep_tail.cuh"
#include "substep_controller_tail.cuh"
}
__global__ void aligned_controller_kernel(const StagedSubstepArguments *request, bool policy) {
  const int world = blockIdx.y;
  if (request->phase[world] != staged_controller_ready) return;
  controller_tail(*request, world, policy);
}
}
extern "C" int flash_libero_controller_kernel_configure(int *resources) {
  using namespace flash_libero;
  auto error = cudaFuncSetAttribute(aligned_controller_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, OscLayout::bytes);
  if (error != cudaSuccess) return int(error);
  cudaFuncAttributes a;
  error = cudaFuncGetAttributes(&a, aligned_controller_kernel);
  if (error != cudaSuccess) return int(error);
  resources[0] = a.numRegs; resources[1] = a.sharedSizeBytes;
  resources[2] = a.localSizeBytes; resources[3] = OscLayout::bytes;
  return 0;
}
