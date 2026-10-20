#pragma once
#include "precision.cuh"

namespace flash_libero {
// Explicit shared address-space access for noinline device functions that
// receive a caller-owned arena. Offsets count reals, not bytes or FP64 words.
__device__ __forceinline__ real_t load_shared_real(unsigned base, int index) {
  real_t value;
  asm volatile("ld.shared." REAL_PTX_TYPE " %0, [%1];"
      : REAL_PTX_OUT(value) : "r"(base + unsigned(index) * unsigned(sizeof(real_t))) : "memory");
  return value;
}
__device__ __forceinline__ void store_shared_real(unsigned base, int index, real_t value) {
  asm volatile("st.shared." REAL_PTX_TYPE " [%0], %1;"
      :: "r"(base + unsigned(index) * unsigned(sizeof(real_t))), REAL_PTX_IN(value) : "memory");
}
} // namespace flash_libero
