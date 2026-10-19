// Correctness only: the maintained packed solver versus an independent dense
// long-double solve of M=L^T D L. No timers or production fallback wrapper.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "common/vector.cuh"
#include "step/forward_mass.cuh"

static void check(cudaError_t code) {
  if (code != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(code));
}

template <class T> struct Device {
  T *ptr = nullptr;
  explicit Device(const std::vector<T> &values) {
    check(cudaMalloc(reinterpret_cast<void **>(&ptr), values.size() * sizeof(T)));
    check(cudaMemcpy(ptr, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  ~Device() { cudaFree(ptr); }
  Device(const Device &) = delete;
  std::vector<T> read(size_t count) const {
    std::vector<T> result(count);
    check(cudaMemcpy(result.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
    return result;
  }
};

__global__ void solve_test(const double *rhs, double *out, const double *ld,
    const double *inv, const int *madr, const int *parent, const int *plan,
    int *status, int nv, int nm) {
  const int world = blockIdx.x;
  const bool ok = flash_libero::forward_mass::solve(rhs + world * nv, out + world * nv,
      ld + world * nm, inv + world * nv, madr, parent, plan);
  if (threadIdx.x == 0)
    status[world] = ok;
}

static std::vector<long double> dense_solve(std::vector<long double> matrix,
                                           std::vector<long double> rhs, int n) {
  for (int k = 0; k < n; ++k) {
    int pivot = k;
    for (int i = k + 1; i < n; ++i)
      if (std::abs(matrix[i*n+k]) > std::abs(matrix[pivot*n+k])) pivot = i;
    for (int j = 0; j < n; ++j) std::swap(matrix[k*n+j], matrix[pivot*n+j]);
    std::swap(rhs[k], rhs[pivot]);
    for (int i = k + 1; i < n; ++i) {
      const long double scale = matrix[i*n+k] / matrix[k*n+k];
      for (int j = k; j < n; ++j) matrix[i*n+j] -= scale * matrix[k*n+j];
      rhs[i] -= scale * rhs[k];
    }
  }
  for (int i = n - 1; i >= 0; --i) {
    for (int j = i + 1; j < n; ++j) rhs[i] -= matrix[i*n+j] * rhs[j];
    rhs[i] /= matrix[i*n+i];
  }
  return rhs;
}

static void run(const std::vector<int> &parent, int expected_width) {
  const int nv = int(parent.size()), worlds = 2;
  std::vector<int> madr(nv), depth(nv), bounds, descriptor;
  std::vector<unsigned> ancestors(nv);
  int nm = 0, first = 0, longest = 0;
  bool contiguous = true;
  for (int i = 0; i < nv; ++i) {
    if (parent[i] == -1) { bounds.push_back(i); first = i; }
    else { depth[i] = depth[parent[i]] + 1; contiguous &= parent[i] >= first; }
    madr[i] = nm;
    nm += depth[i] + 1;
    if (i - first < 32 && parent[i] >= first)
      ancestors[i] = ancestors[parent[i]] | (1u << (parent[i] - first));
  }
  bounds.push_back(nv);
  for (size_t i = 1; i < bounds.size(); ++i) longest = std::max(longest, bounds[i] - bounds[i-1]);
  int width = 8;
  while (width < longest) width *= 2;
  if (!contiguous || width > 32) width = 0;
  if (width != expected_width) throw std::runtime_error("test topology dispatch");
  descriptor = {nv, int(bounds.size()) - 1, width};
  descriptor.insert(descriptor.end(), bounds.begin(), bounds.end());
  descriptor.insert(descriptor.end(), depth.begin(), depth.end());
  for (unsigned value : ancestors) descriptor.push_back(int(value));
  std::vector<double> ld(worlds*nm), inv(worlds*nv), rhs(worlds*nv), out(worlds*nv);
  std::vector<int> status(worlds);
  std::vector<std::vector<long double>> expected;
  for (int world = 0; world < worlds; ++world) {
    std::vector<long double> lower(nv*nv), diagonal(nv), matrix(nv*nv), vector(nv);
    for (int i = 0; i < nv; ++i) {
      const double d = 1.0 + (i % 7) * 0.2 + world * 0.13;
      diagonal[i] = ld[world*nm + madr[i]] = d;
      lower[i*nv+i] = 1;
      inv[world*nv+i] = 1.0 / d;
      int coefficient = madr[i] + 1;
      for (int p = parent[i]; p >= 0; p = parent[p], ++coefficient) {
        const double value = 0.025 * std::sin(double((i+1)*(p+2) + world));
        lower[i*nv+p] = ld[world*nm+coefficient] = value;
      }
      // World 1 includes exact zero descendants and a distinct factor.
      vector[i] = rhs[world*nv+i] = world && i % 3 == 0 ? 0.0 : std::cos(0.17*i + world);
    }
    for (int i = 0; i < nv; ++i)
      for (int j = 0; j < nv; ++j)
        for (int k = 0; k < nv; ++k)
          matrix[i*nv+j] += lower[k*nv+i] * diagonal[k] * lower[k*nv+j];
    expected.push_back(dense_solve(matrix, vector, nv));
  }
  Device<double> dld(ld), dinv(inv), drhs(rhs), dout(out);
  Device<int> dmadr(madr), dparent(parent), dplan(descriptor), dstatus(status);
  solve_test<<<worlds,64>>>(drhs.ptr, dout.ptr, dld.ptr, dinv.ptr, dmadr.ptr,
      dparent.ptr, dplan.ptr, dstatus.ptr, nv, nm);
  check(cudaGetLastError());
  check(cudaDeviceSynchronize());
  const auto actual = dout.read(out.size());
  const auto ok = dstatus.read(worlds);
  for (int world = 0; world < worlds; ++world) {
    if (!ok[world]) throw std::runtime_error("cooperative solver rejected finite input");
    for (int i = 0; i < nv; ++i)
      if (!std::isfinite(actual[world*nv+i]) ||
          std::abs(actual[world*nv+i] - expected[world][i]) >
          1e-11L * (1 + std::abs(expected[world][i])))
        throw std::runtime_error("packed solve disagrees with independent dense reference");
  }
}

int main() {
  try {
    // The robot branch plus independent free/scalar components covers every
    // shipped NV family; the kernel itself has no task/dimension specialization.
    for (int nv : {21,22,24,27,30,33,36,37,39,43,45,51,57}) {
      std::vector<int> parent(nv);
      for (int i = 0; i < nv; ++i) parent[i] = i == 0 ? -1 : i - 1;
      parent[8] = 6;
      for (int i = 9; i < nv; i += 6) parent[i] = -1;
      run(parent,16);
    }
    for (int length : {6,31,32,70}) {
      std::vector<int> parent(length);
      for (int i = 0; i < length; ++i) parent[i] = i - 1;
      run(parent,length <= 8 ? 8 : (length <= 32 ? 32 : 0));
    }
    run({-1,-1,0,1,2,3},0);
    std::cout << "PASS: 18 topologies, two worlds, subgroups 8/16/32, large/interleaved fallback\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
