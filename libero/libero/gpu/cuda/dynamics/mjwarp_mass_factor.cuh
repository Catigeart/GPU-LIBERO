// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
namespace flash::mjwarp {
struct MassFactorInputs {
  int worlds, nv, blocks, matrix_width, factor_width, rhs_width, solution_width;
  const int *matrix_row, *factor_address, *block_start, *gather;
  const float *matrix, *rhs;
  float *factor, *diagonal_inverse, *solution;
};
}
extern "C" {
const char* flash_mjwarp_mass_factor_source_id();
std::size_t flash_mjwarp_mass_factor_arguments_bytes();
int flash_mjwarp_mass_factor_arch();
// Scalar/compact kind=0, sizes 1/6; cooperative kind=1, size 9.
int flash_mjwarp_mass_factor_launch(const flash::mjwarp::MassFactorInputs*, int kind,
                                   int size, int solve, cudaStream_t);
}
