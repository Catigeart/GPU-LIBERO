// Fixed-input replay of the actual independently compiled Post stage.
#include "post_solver.cuh"
#include <cuda_runtime.h>
#include <cstddef>
namespace {
__global__ void replay(const physics_newton::NewtonPostData *request,
    const real_t *private_qacc, const int *subtrees, int *evaluated, bool outputs_required) {
  extern __shared__ real_t workspace[];
  const auto &post = *request;
  const int world = blockIdx.y;
  for (int j = threadIdx.x; j < post.r_v; j += blockDim.x)
    workspace[j] = private_qacc[std::size_t(world) * post.r_v + j];
  __syncthreads();
  const bool full = physics_post::run(post, world, workspace,
      workspace + post.r_v, outputs_required, subtrees);
  if (threadIdx.x == 0) evaluated[world] = int(full);
}
}
extern "C" std::size_t flash_post_replay_arguments_bytes() {
  return sizeof(physics_newton::NewtonPostData);
}
extern "C" int flash_post_replay_configure(int bodies, int nv, int *resources) {
  if (bodies <= 0 || nv <= 0) return int(cudaErrorInvalidValue);
  const std::size_t raw = 6 * std::size_t(bodies);
  const std::size_t temporary = raw > 288 ? raw : 288;
  const std::size_t bytes = (std::size_t(nv) + 18 * std::size_t(bodies) + 6 + temporary) * sizeof(real_t);
  if (bytes > 2147483647) return int(cudaErrorInvalidValue);
  auto error = cudaFuncSetAttribute(replay, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes));
  if (error != cudaSuccess) return int(error);
  cudaFuncAttributes a;
  error = cudaFuncGetAttributes(&a, replay);
  if (error != cudaSuccess) return int(error);
  resources[0] = a.numRegs; resources[1] = int(a.sharedSizeBytes);
  resources[2] = int(a.localSizeBytes); resources[3] = int(bytes);
  return 0;
}
extern "C" int flash_post_replay_launch(const physics_newton::NewtonPostData *request,
    const real_t *private_qacc, const int *subtrees, int *evaluated, int worlds, int threads,
    int shared_bytes, int outputs_required, cudaStream_t stream) {
  if (!request || !private_qacc || !evaluated || worlds <= 0 ||
      (threads != 64 && threads != 128 && threads != 256) || shared_bytes <= 0)
    return int(cudaErrorInvalidValue);
  replay<<<dim3(1, worlds), threads, shared_bytes, stream>>>(request,
      private_qacc, subtrees, evaluated, outputs_required != 0);
  return int(cudaGetLastError());
}
