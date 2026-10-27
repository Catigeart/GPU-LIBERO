// Optional offline replay product; exercise the maintained numerical source.
#include "../../libero/libero/gpu/cuda/control/osc_solver.cu"

namespace {
using physics_controller::osc_cuda::OscPseudoinverseWork;
using physics_controller::osc_cuda::PseudoinverseTrace;
template<int N> __global__ void replay(const double *input, double *output,
                                      int *ranks, int *ok, PseudoinverseTrace *trace) {
  __shared__ OscPseudoinverseWork work;
  constexpr int width = N == 3 ? 16 : 32;
  constexpr unsigned mask = N == 3 ? 0xffffu : 0xffffffffu;
  if (threadIdx.x < width)
    physics_controller::osc_cuda::pinv_group<N, width, true>(
        input + blockIdx.x * 36, output + blockIdx.x * 36, work,
        ranks + blockIdx.x, ok + blockIdx.x, mask, trace + blockIdx.x);
}
}

extern "C" size_t flash_osc_replay_trace_bytes() { return sizeof(PseudoinverseTrace); }
extern "C" size_t flash_osc_replay_shared_bytes() { return flash_libero::OscLayout::bytes; }
extern "C" int flash_osc_replay(const double *input, double *output, int *ranks,
                               int *ok, void *trace, int n, int count, cudaStream_t stream) {
  if (!input || !output || !ranks || !ok || !trace || count < 1 || (n != 3 && n != 6))
    return cudaErrorInvalidValue;
  if (n == 3)
    replay<3><<<count, 32, 0, stream>>>(input, output, ranks, ok, static_cast<PseudoinverseTrace *>(trace));
  else
    replay<6><<<count, 32, 0, stream>>>(input, output, ranks, ok, static_cast<PseudoinverseTrace *>(trace));
  return int(cudaGetLastError());
}
