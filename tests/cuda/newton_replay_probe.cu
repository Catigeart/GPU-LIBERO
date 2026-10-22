// Optional diagnostic entry: call the actual separately compiled Newton once.
// No position, collision, control, integration, or alternative numerical math.
#include "newton_solver.cuh"
#include "newton_workspace.cuh"
#include "substep_shared.cuh"
#include <cuda_runtime.h>

namespace {
using namespace physics_newton;
using flash_libero::NewtonLayout;
using flash_libero::SubstepShape;
constexpr int info_count = 8, real_count = 3;
__global__ void replay(const NewtonArguments *request, const int *upstream, const int *enabled,
                       const int *publication, const int64_t *plan, int *info, real_t *values,
                       int initial_gradient_stop) {
  unsigned char *shared = flash_libero::substep_shared_data();
  const int w = blockIdx.y;
  NewtonOptions options{};
  options.initial_gradient_stop = initial_gradient_stop != 0;
  solve_newton(shared, request->model, request->worlds[w], upstream, enabled, publication, plan, options);
  __syncthreads();
  auto &c = *flash_libero::shared_at<NewtonControl>(shared, NewtonLayout::control);
  if (threadIdx.x == 0) {
    const int record[info_count] = {c.iteration(), c.LSiter(), c.nactive(), c.ncone(),
                                   c.choice(), c.stop_reason(), c.err_code(), c.nupdate()};
    for (int i=0; i<info_count; ++i) info[w*info_count+i]=record[i];
    values[w*(real_count+4*SubstepShape::nv)] = c.out_improvement();
    values[w*(real_count+4*SubstepShape::nv)+1] = c.out_gradient();
    values[w*(real_count+4*SubstepShape::nv)+2] = c.cost();
  }
  const std::size_t offsets[4] = {NewtonLayout::mass_acceleration, NewtonLayout::gradient,
                                 NewtonLayout::warm_mass, NewtonLayout::search};
  for (int vector=0; vector<4; ++vector) {
    const real_t *source=flash_libero::shared_at<real_t>(shared, offsets[vector]);
    for (int j=threadIdx.x; j<SubstepShape::nv; j+=blockDim.x)
      values[w*(real_count+4*SubstepShape::nv)+real_count+vector*SubstepShape::nv+j]=
          vector==3 && c.stop_reason()==11 ? real_t(0) : source[j];
  }
}
}
extern "C" int flash_newton_replay_scalar_bytes() { return sizeof(real_t); }
extern "C" int flash_newton_replay_info_count() { return info_count; }
extern "C" int flash_newton_replay_real_count() { return real_count+4*flash_libero::SubstepShape::nv; }
extern "C" int flash_newton_replay_launch(const physics_newton::NewtonArguments *request,
    const int *upstream, const int *enabled, const int *publication, const int64_t *plan,
    int *info, real_t *values, int worlds, cudaStream_t stream) {
  const auto bytes=flash_libero::substep_shared_bytes;
  cudaError_t error=cudaFuncSetAttribute(replay,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
  if(error!=cudaSuccess)return int(error);
  replay<<<dim3(1,worlds),flash_libero::SubstepShape::threads,bytes,stream>>>(
      request,upstream,enabled,publication,plan,info,values,FLASH_REAL_BITS==32);
  return int(cudaGetLastError());
}
extern "C" int flash_newton_replay_launch_entry_policy(const physics_newton::NewtonArguments *request,
    const int *upstream, const int *enabled, const int *publication, const int64_t *plan,
    int *info, real_t *values, int worlds, int initial_gradient_stop, cudaStream_t stream) {
  const auto bytes=flash_libero::substep_shared_bytes;
  cudaError_t error=cudaFuncSetAttribute(replay,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
  if(error!=cudaSuccess)return int(error);
  replay<<<dim3(1,worlds),flash_libero::SubstepShape::threads,bytes,stream>>>(
      request,upstream,enabled,publication,plan,info,values,initial_gradient_stop);
  return int(cudaGetLastError());
}

// A diagnostic first-forward boundary using the actual production function.
// The normal physical substep kernel and its call schedule are unchanged.
#ifndef FLASH_NEWTON_ONLY_PROBE
#include "fused_substep.cu"
namespace {
__global__ void forward_once(const flash_libero::SubstepArguments *arguments, real_t *smooth) {
  const auto &args=*arguments;
  const int w=blockIdx.y;
  auto *shared=flash_libero::substep_shared_data();
  if(threadIdx.x==0)
    args.forward_workspace.metadata[size_t(w)*flash_libero::ForwardCacheWords+
        flash_libero::ForwardCacheValid]=0;
  __syncthreads();
  flash_libero::forward_dynamics(args,w,shared,true);
  const real_t *force=flash_libero::shared_at<real_t>(shared,flash_libero::SmoothLayout::force);
  const real_t *acc=flash_libero::shared_at<real_t>(shared,flash_libero::SmoothLayout::acceleration);
  for(int j=threadIdx.x;j<flash_libero::SubstepShape::nv;j+=blockDim.x) {
    smooth[w*2*flash_libero::SubstepShape::nv+j]=force[j];
    smooth[(w*2+1)*flash_libero::SubstepShape::nv+j]=acc[j];
  }
}
}
extern "C" int flash_newton_forward_probe_launch(const flash_libero::SubstepArguments *arguments,
                                                  real_t *smooth,int worlds,cudaStream_t stream) {
  const auto bytes=flash_libero::substep_shared_bytes;
  cudaError_t error=cudaFuncSetAttribute(forward_once,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
  if(error!=cudaSuccess)return int(error);
  forward_once<<<dim3(1,worlds),flash_libero::SubstepShape::threads,bytes,stream>>>(arguments,smooth);
  return int(cudaGetLastError());
}
#endif
