// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstddef>

namespace flash::mjwarp {
// Host launch descriptor borrowing the current resident factorization and RHS.
// Strides are in float elements; factors are NEVER cached between substeps.
struct MassSolveInputs {
  int worlds, nv, blocks, threads;
  int factor_stride, diagonal_stride, rhs_stride, solution_stride;
  const int *factor_address, *block_start;
  const float *factor, *diagonal_inverse, *rhs;
  float* solution;
};
}
extern "C" {
const char* flash_mjwarp_mass_solve_source_id();
std::size_t flash_mjwarp_mass_solve_arguments_bytes();
int flash_mjwarp_mass_solve_arch();
// kind=0: scalar/compact (size=1 or 6); kind=1: cooperative POTRS (size=9).
int flash_mjwarp_mass_solve_launch(const flash::mjwarp::MassSolveInputs*, int kind,
                                  int size, cudaStream_t);
}
