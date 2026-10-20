// Optional diagnostic ABI for frozen BoxBox geometry. The numerical function
// and scalar/subgroup implementations are the maintained production bodies.
#include <cuda_runtime.h>
#include "../../libero/libero/gpu/cuda/common/vector.cuh"
#include "../../libero/libero/gpu/cuda/collision/native/native_source.cuh"

namespace {
constexpr int input_width = 31, contact_width = 7, capacity = mjMAXCONPAIR;
__global__ void replay(real_t *inputs, int *counts, int *axes, real_t *outputs,
                       int queries, int subgroup) {
  const int query = blockIdx.x;
  if (query >= queries || threadIdx.x >= (subgroup ? subgroup : 1)) return;
  real_t *g = inputs + query * input_width;
  int diag[16]{}, frame = 0;
  mjModel model{};
  mjData data{};
  mjContact contacts[capacity]{};
  model.geom_size = g;
  model.adapter_diag = diag;
  model.adapter_frame0 = &frame;
  model.adapter_subgroup_width = subgroup;
  data.geom_xpos = g + 6;
  data.geom_xmat = g + 12;
  const int count = mjc_BoxBox(&model, &data, contacts, 0, 1, g[30]);
  if (threadIdx.x == 0) {
    counts[query] = count;
    axes[query] = diag[6];
    for (int j = 0; j < count; ++j) {
      real_t *out = outputs + (query * capacity + j) * contact_width;
      out[0] = contacts[j].dist;
      for (int k = 0; k < 3; ++k) {
        out[1+k] = contacts[j].pos[k];
        out[4+k] = contacts[j].frame[k];
      }
    }
  }
}
}

extern "C" int flash_boxbox_replay_scalar_bytes() { return sizeof(real_t); }
extern "C" int flash_boxbox_replay_capacity() { return capacity; }
extern "C" int flash_boxbox_replay_launch(real_t *inputs, int *counts, int *axes,
    real_t *outputs, int queries, int subgroup, cudaStream_t stream) {
  if (queries < 1 || (subgroup != 0 && subgroup != 8 && subgroup != 16 && subgroup != 32))
    return int(cudaErrorInvalidValue);
  replay<<<queries,32,0,stream>>>(inputs,counts,axes,outputs,queries,subgroup);
  return int(cudaGetLastError());
}
