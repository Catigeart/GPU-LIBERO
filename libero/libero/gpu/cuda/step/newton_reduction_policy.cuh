#pragma once
#include <cuda_runtime.h>

namespace physics_newton {
enum class TransposeMethod : int {
  Auto, Atomic, Direct, Shared32, Shared16, Shared8, Csc1, Csc4, Csc8
};
struct NewtonReductionPolicy { int cache_enabled; TransposeMethod transpose; };
// Defaults are selected by CMake in the small cache object. The optional host
// configurator is for matched comparisons at a quiescent graph boundary; it
// changes neither the fused kernel ABI nor the separately compiled solver.
extern __device__ __constant__ NewtonReductionPolicy reduction_policy;
} // namespace physics_newton
