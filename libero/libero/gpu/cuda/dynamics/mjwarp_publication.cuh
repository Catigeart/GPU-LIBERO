// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
namespace flash::mjwarp {
struct PublicationInputs {
  int worlds,timestep_rows,sparse,contacts,rows,nonzeros,warnings;
  const float* timestep;
  const int *count,*rownnz,*rowadr,*contact_count,*candidate_count;
  float* time;
  int* overflow;
};
}
