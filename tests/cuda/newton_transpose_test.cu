// Actual production transpose: long-double reference, subgroup tails, output
// holds, masked NaN rows, and immediate gradient consumption in two worlds.
#include "newton_transpose.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
using namespace physics_newton::transpose_reduction;
static void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
constexpr real_t sentinel = real_t(123.25);
__global__ void evaluate(const real_t *J, const real_t *force, real_t *out,
                         int nv, int capacity, int method, bool inactive) {
  extern __shared__ real_t shared[];
  const int w = blockIdx.x;
  if (inactive && w == 1) return;
  const int rows = w && capacity ? capacity / 2 + 1 : capacity;
  J += size_t(w) * capacity * nv; force += w * capacity;
  Method selected = method < 0 ? select_auto(nv, rows, int(blockDim.x)) : static_cast<Method>(method);
  if (selected == Method::Subwarp4) subwarp4(J, force, rows, nv, shared);
  else dense(J, force, rows, nv, shared, shared + nv, selected);
  __syncthreads(); // native caller's output publication and scratch release
  for (int j = threadIdx.x; j < nv; j += blockDim.x) {
    out[w * (2 * nv + 2) + j + 1] = shared[j];
    // Exercise the same immediate per-DOF consumer as Newton's gradient.
    out[w * (2 * nv + 2) + nv + j + 1] =
        (real_t(j) * real_t(.125) - real_t(.75)) - shared[j];
  }
}
static double worst = 0;
static void run(int nv, int capacity, int threads, int method, bool inactive) {
  const size_t elements = std::max(size_t(1), size_t(2) * capacity * nv);
  std::vector<real_t> J(elements), force(std::max(1, 2 * capacity));
  for (int w = 0; w < 2; ++w)
    for (int r = 0; r < capacity; ++r) {
      force[w * capacity + r] = r % 3 ? real_t(std::sin(.07 * r + .2 * w)) : real_t(0);
      for (int j = 0; j < nv; ++j)
        J[(size_t(w) * capacity + r) * nv + j] = r % 3
            ? real_t(std::sin(.03 * r + .19 * j + .2 * w)) : real_t(NAN);
    }
  std::vector<real_t> output(2 * (2 * nv + 2), sentinel);
  real_t *dj, *df, *result;
  check(cudaMalloc(&dj, J.size() * sizeof(real_t)));
  check(cudaMalloc(&df, force.size() * sizeof(real_t)));
  check(cudaMalloc(&result, output.size() * sizeof(real_t)));
  check(cudaMemcpy(dj, J.data(), J.size() * sizeof(real_t), cudaMemcpyHostToDevice));
  check(cudaMemcpy(df, force.data(), force.size() * sizeof(real_t), cudaMemcpyHostToDevice));
  check(cudaMemcpy(result, output.data(), output.size() * sizeof(real_t), cudaMemcpyHostToDevice));
  evaluate<<<2, threads, (nv + threads / 8 * nv) * sizeof(real_t)>>>(
      dj, df, result, nv, capacity, method, inactive);
  check(cudaGetLastError()); check(cudaDeviceSynchronize());
  check(cudaMemcpy(output.data(), result, output.size() * sizeof(real_t), cudaMemcpyDeviceToHost));
  for (int w = 0; w < 2; ++w) {
    const int base = w * (2 * nv + 2), rows = w && capacity ? capacity / 2 + 1 : capacity;
    if (output[base] != sentinel || output[base + 2 * nv + 1] != sentinel)
      throw std::runtime_error("output redzone changed");
    for (int j = 0; j < nv; ++j) {
      if (inactive && w == 1) {
        if (output[base + j + 1] != sentinel || output[base + nv + j + 1] != sentinel)
          throw std::runtime_error("inactive world hold changed");
        continue;
      }
      long double value = 0, scale = 0;
      for (int r = 0; r < rows; ++r) if (force[w * capacity + r] != 0) {
        const long double term = static_cast<long double>(J[(size_t(w) * capacity + r) * nv + j])
            * force[w * capacity + r];
        value += term; scale += std::abs(term);
      }
      for (int field = 0; field < 2; ++field) {
        const long double reference = field ? (real_t(j) * real_t(.125) - real_t(.75)) - value : value;
        const real_t actual = output[base + field * nv + j + 1];
        const double error = double(std::abs(actual - reference) / (1 + scale + std::abs(reference)));
        worst = std::max(worst, error);
        if (!std::isfinite(actual) || error > (rows + 16) * REAL_EPSILON)
          throw std::runtime_error("transpose/gradient independent residual failed");
      }
    }
  }
  check(cudaFree(dj)); check(cudaFree(df)); check(cudaFree(result));
}
int main() {
  try {
    int cases = 0;
    for (int nv : {21,22,24,27,30,33,36,37,39,43,45,51,57})
      for (int rows : {0,1,17,127,128,129,165,256,257,512,513})
        for (int threads : {64,128})
          for (int method : {-1,int(Method::Subwarp4)}) {
            run(nv, rows, threads, method, false); ++cases;
          }
    run(39, 165, 64, -1, true); ++cases;
    std::printf("two-world transpose/gradient cases=%d worst_scaled_error=%.8g\n", cases, worst);
  } catch (const std::exception &e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
