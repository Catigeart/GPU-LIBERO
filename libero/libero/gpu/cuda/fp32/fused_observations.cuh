// Copyright 2026 Flash-LIBERO contributors
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
namespace flash::fp32 {
struct RawPackInputs {
  int worlds,stride;
  int width[5],offset[5];
  const float* source[5];
  float* target;
};
// Same pointer ABI as the accepted observable sampler in runtime/bridges.cu.
struct SampleInputs {
  int worlds,ns,width,stride;
  const int *op,*offset,*size,*index_offset,*indices,*dep0,*dep1,*dep2,*presence,*enabled;
  const float *periods,*source;
  float *cache,*observed,*sidequat,*elapsed;
  int *sampled,*present;
  const int* mask;
  int* events;
};
}
