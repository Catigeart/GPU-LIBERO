// Actual retained and cooperative production transmission bodies.
#include "precision.cuh"
#include <cuda_runtime.h>
#include <cstddef>
#include "../common/vector.cuh"
#include "../common/warp.cuh"
#include "../position/position.cuh"
struct TransmissionReplay {
  real_t *q; int *active, *qa, *da; real_t *gear, *length, *moment;
  int *status; int nq, nv, nu;
};
namespace {
template<bool Parallel> __global__ void replay(TransmissionReplay a) {
  const int w=blockIdx.y;
  if constexpr (Parallel)
    libero_position::transmission_block(w,a.q,a.active,a.qa,a.da,a.gear,
      a.length,a.moment,a.status,a.nq,a.nv,a.nu);
  else if(threadIdx.x==0)
    libero_position::transmission(w,a.q,a.active,a.qa,a.da,a.gear,
      a.length,a.moment,a.status,a.nq,a.nv,a.nu);
}
}
extern "C" int flash_transmission_replay_launch(const TransmissionReplay *a,
    int worlds,int candidate,int threads,cudaStream_t stream) {
  if(!a||worlds<1||(threads!=64&&threads!=128&&threads!=256))
    return int(cudaErrorInvalidValue);
  if(candidate)replay<true><<<dim3(1,worlds),threads,0,stream>>>(*a);
  else replay<false><<<dim3(1,worlds),threads,0,stream>>>(*a);
  return int(cudaGetLastError());
}
