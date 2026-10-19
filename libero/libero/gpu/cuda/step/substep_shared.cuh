#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include "substep_shared_layout.cuh"

namespace flash_libero {
// One allocation per world block. This helper only forms an address; it has
// no local workspace or argument package. Offsets are compile-time byte offsets.
static __device__ __forceinline__ unsigned char *substep_shared_data() {
  extern __shared__ __align__(32) unsigned char shared[];
  return shared;
}
template <class T> static __device__ __forceinline__ T *shared_at(unsigned char *base, std::size_t offset) {
  return reinterpret_cast<T *>(base + offset);
}
} // namespace flash_libero
