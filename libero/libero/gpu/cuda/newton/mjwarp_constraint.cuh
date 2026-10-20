// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstddef>

namespace flash::mjwarp {
// Host launch arguments; all arrays are borrowed resident device storage.
struct ConstraintInputs {
  int worlds, nv, pad, rows, capacity, impratio_count;
  const bool* done;
  const int *ne, *nf, *nefc, *contact_count, *contact_dimension, *contact_address;
  const float *impratio, *contact_friction;
  const int *row_type, *row_contact;
  const float *diagonal, *frictionloss, *jaref, *jacobian;
  float* force;
  int* state;
  float* qfrc_constraint;
};
}
extern "C" {
const char* flash_mjwarp_constraint_source_id();
std::size_t flash_mjwarp_constraint_arguments_bytes();
int flash_mjwarp_constraint_arch();
// 0: both kernels, 1: force/state only, 2: J^T force only.
int flash_mjwarp_constraint_launch(const flash::mjwarp::ConstraintInputs*, int,
                                  cudaStream_t);
}
