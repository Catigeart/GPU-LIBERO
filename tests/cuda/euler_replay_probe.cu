// Fixed-input replay of the actual Euler region, independent of forward/control.
#include "precision.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include "substep_types.cuh"
#include "substep_shared.cuh"
#include "../common/vector.cuh"
#include "substep_euler_math.cuh"
namespace {
__global__ void replay(const flash_libero::EulerData *request) {
  const auto &in = *request;
  const int w = blockIdx.y;
  physics_euler::cf_euler::evaluate_core(w, in.active, in.qp, in.qv, in.time,
      in.qm, in.smooth, in.constraint, in.acc, in.damping, in.madr, in.parent,
      in.jt, in.padr, in.vadr, in.qh, in.qhi, in.count, in.ma, in.damp, in.status,
      in.ms, in.wc, in.wl, in.dt, in.nq, in.nv, in.nm, in.nj, in.qacc_stride,
      in.joint_quat, in.joint_type_offsets, in.joint_type_indices);
}
}
extern "C" std::size_t flash_euler_replay_arguments_bytes() {
  return sizeof(flash_libero::EulerData);
}
extern "C" int flash_euler_replay_configure(int nq, int nv, int nm, int *resources) {
  using flash_libero::SubstepShape;
  if (nq <= 0 || nv <= 0 || nm <= 0 || nq > SubstepShape::nq ||
      nv > SubstepShape::nv || nm > SubstepShape::packed_mass)
    return int(cudaErrorInvalidValue);
  constexpr auto bytes = flash_libero::EulerLayout::bytes;
  auto error = cudaFuncSetAttribute(replay, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  if (error != cudaSuccess) return int(error);
  cudaFuncAttributes a;
  error = cudaFuncGetAttributes(&a, replay);
  if (error != cudaSuccess) return int(error);
  resources[0] = a.numRegs; resources[1] = a.sharedSizeBytes;
  resources[2] = a.localSizeBytes; resources[3] = bytes;
  return 0;
}
extern "C" int flash_euler_replay_launch(const flash_libero::EulerData *request,
    int worlds, int threads, cudaStream_t stream) {
  if (!request || worlds <= 0 || (threads != 64 && threads != 128 && threads != 256))
    return int(cudaErrorInvalidValue);
  replay<<<dim3(1, worlds), threads, flash_libero::EulerLayout::bytes, stream>>>(request);
  return int(cudaGetLastError());
}
