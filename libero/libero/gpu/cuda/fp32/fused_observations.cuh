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
}
