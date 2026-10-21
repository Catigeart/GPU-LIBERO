// Copyright 2026 Flash-LIBERO contributors
// World-owned fusion of the five accepted low-dimensional field copies.
#include "fused_observations.cuh"
#ifndef FLASH_FP32_OBSERVATIONS_SOURCE_ID
#error "Build through flash_fp32_fused_observations"
#endif
namespace flash::fp32 {
template<int Threads>
__global__ void pack_raw(RawPackInputs a) {
  const int world=blockIdx.x,lane=threadIdx.x;
  // Every field writes a disjoint output interval. One CTA owns all five
  // copies for its world, with no cross-world data or synchronization.
  for (int field=0;field<5;++field) {
    const float* source=a.source[field]+world*a.width[field];
    float* target=a.target+world*a.stride+a.offset[field];
    for (int element=lane;element<a.width[field];element+=Threads)
      target[element]=source[element];
  }
}
}
extern "C" {
const char* flash_fp32_observations_source_id() { return FLASH_FP32_OBSERVATIONS_SOURCE_ID; }
int flash_fp32_observations_arch() { return FLASH_CUDA_ARCH; }
std::size_t flash_fp32_observations_pack_bytes() { return sizeof(flash::fp32::RawPackInputs); }
int flash_fp32_observations_pack(const flash::fp32::RawPackInputs* a,cudaStream_t stream) {
  if (!a || a->worlds<1 || a->stride<1) return cudaErrorInvalidValue;
  flash::fp32::pack_raw<128><<<a->worlds,128,0,stream>>>(*a);
  return cudaGetLastError();
}
}
