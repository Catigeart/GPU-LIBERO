#pragma once
#include <cuda_runtime.h>

namespace physics_newton::gemv {

// A subwarp owns one row. Every named lane participates, including tails of
// odd NV. The mask excludes unrelated subwarps with a different row count.
template <int Lanes> __device__ double row_sum(double value) {
  const int first = (int(threadIdx.x) & 31) / Lanes * Lanes;
  const unsigned mask = (0xffffffffu >> (32 - Lanes)) << first;
  for (int delta = Lanes / 2; delta; delta /= 2)
    value += __shfl_down_sync(mask, value, delta, Lanes);
  return value;
}

template <int NV, int Lanes>
__device__ void candidates(const double *J, const double *smooth, const double *warm,
                           const double *aref, int rows, double *out_smooth, double *out_warm) {
  const int lane = int(threadIdx.x) % Lanes;
  for (int row = int(threadIdx.x) / Lanes; row < rows; row += int(blockDim.x) / Lanes) {
    double a = 0, b = 0;
    for (int j = lane; j < NV; j += Lanes) {
      const double value = J[size_t(row) * NV + j];
      a += value * smooth[j];
      b += value * warm[j];
    }
    a = row_sum<Lanes>(a); b = row_sum<Lanes>(b);
    if (lane == 0) {
      const double reference = aref[row];
      out_smooth[row] = a - reference;
      out_warm[row] = b - reference;
    }
  }
}

// Produce exactly one live 16-row tile for its immediate line-search consumer.
// The caller publishes it, mirrors Jv for commit, and owns staging lifetime.
template <int NV, int Lanes>
__device__ void tile(const double *J, const double *x, int begin, int rows, double *out) {
  const int lane = int(threadIdx.x) % Lanes;
  for (int k = int(threadIdx.x) / Lanes; k < 16; k += int(blockDim.x) / Lanes) {
    double value = 0;
    if (begin + k < rows)
      for (int j = lane; j < NV; j += Lanes)
        value += J[size_t(begin + k) * NV + j] * x[j];
    value = row_sum<Lanes>(value);
    if (lane == 0) out[k] = value;
  }
}

// Columns remain contiguous in global J. Long row ranges, and NV fitting one
// warp, use all warps as row partitions; partials need only warps*NV doubles.
template <int NV>
__device__ void transpose(const double *J, const double *force, int rows,
                          double *out, double *partials) {
  const int tid = int(threadIdx.x), warps = int(blockDim.x) / 32;
  const bool split = NV <= 32 || rows >= 256;
  if (split) {
    const int warp = tid / 32, lane = tid & 31;
    for (int j = lane; j < NV; j += 32) {
      double value = 0;
      for (int r = warp; r < rows; r += warps) {
        const double f = force[r];
        // A masked Jacobian row may contain NaN. Do not read it for f==0.
        if (f != 0) value += J[size_t(r) * NV + j] * f;
      }
      partials[warp * NV + j] = value;
    }
    __syncthreads();
    for (int j = tid; j < NV; j += int(blockDim.x)) {
      double value = 0;
      for (int warp = 0; warp < warps; ++warp) value += partials[warp * NV + j];
      out[j] = value;
    }
  } else {
    for (int j = tid; j < NV; j += int(blockDim.x)) {
      double value = 0;
      for (int r = 0; r < rows; ++r) {
        const double f = force[r];
        if (f != 0) value += J[size_t(r) * NV + j] * f;
      }
      out[j] = value;
    }
  }
}
} // namespace physics_newton::gemv
