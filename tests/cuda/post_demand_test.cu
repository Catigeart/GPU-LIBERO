// Post demand is independent of dynamics completion: a deferred call must
// publish qacc without claiming fresh force/sensor data. Inactive full calls
// retain the same publication contract. Always exercise two world blocks.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include "../../libero/libero/gpu/cuda/postprocess/post_solver.cuh"

static void check(cudaError_t status) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
    std::exit(1);
  }
}

__global__ void exercise(physics_newton::NewtonPostData post, int *evaluated) {
  extern __shared__ double shared[];
  const int world = blockIdx.y;
  for (int i = threadIdx.x; i < post.r_v; i += blockDim.x)
    shared[i] = 1000.0 * world + i + 0.25;
  __syncthreads();
  const bool full = physics_post::run(post, world, shared, shared + post.r_v, world == 1);
  if (threadIdx.x == 0) evaluated[world] = int(full);
}

int main() {
  for (int threads : {32, 64, 128}) for (int nv : {21, 39, 57, 97})
      for (bool stale : {false, true}) {
    double *qacc;
    int *flags;
    int64_t *written;
    check(cudaMallocManaged(&qacc, 2 * nv * sizeof(double)));
    check(cudaMallocManaged(&flags, 16 * sizeof(int)));
    check(cudaMallocManaged(&written, 8 * sizeof(int64_t)));
    for (int i = 0; i < 2 * nv; ++i) qacc[i] = -1.0;
    for (int i = 0; i < 16; ++i) flags[i] = 0;
    flags[0] = flags[1] = 1; // old Post validity must be invalidated
    written[0] = 41; written[1] = 73;
    for (int i = 2; i < 8; ++i) written[i] = 7;
    if (stale) written[2] = 6;
    physics_newton::NewtonPostData post{};
    post.r_b = post.s_b = 2;
    post.r_v = nv;
    post.external_c_valid = flags;
    post.external_c_written = written;
    post.published_work_acceleration = qacc;
    post.external_g_active = flags + 2;  // inactive worlds
    flags[2] = 1; // world zero has a successful current Newton input
    post.external_g_first = flags + 4;
    post.valid = flags + 14;
    flags[14] = flags[15] = 1; // successful Newton publication
    post.err_code = flags + 4;
    post.external_g_dv = flags + 14;
    post.n_written = written + 2;
    post.attempted = written + 4;
    post.external_g_expected = written + 6;
    post.external_g_state = post.external_g_consumed = written + 6;
    post.external_g_forward = post.external_g_cf = written + 6;
    post.external_g_g = flags + 6;
    post.external_r_status = flags + 8;
    post.external_s_status = flags + 10;
    exercise<<<dim3(1, 2), threads, (nv + 18 * 2 + 6) * sizeof(double)>>>(post, flags + 12);
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
    bool passed = flags[0] == 0 && flags[1] == 0 && written[0] == 41 && written[1] == 73 &&
                  flags[12] == int(stale) && flags[13] == 1 && flags[4] == 0 && flags[5] == 0;
    for (int w = 0; w < 2; ++w) for (int i = 0; i < nv; ++i)
      passed &= qacc[w * nv + i] == 1000.0 * w + i + 0.25;
    check(cudaFree(qacc)); check(cudaFree(flags)); check(cudaFree(written));
    if (!passed) {
      std::fprintf(stderr, "Post demand failed: threads=%d nv=%d\n", threads, nv);
      return 1;
    }
  }
  std::puts("Post demand: 24 two-world configurations passed");
}
