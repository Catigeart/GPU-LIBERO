// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstddef>

namespace flash::mjwarp {
// Host ABI only. Each kernel consumes resident arrays directly; no shared
// parameter pack, persistent physical-value cache or device allocation.
struct InitInputs {
  int worlds, nv, pad, rows, mass_entries, warmstart;
  const int *nefc, *mass_row, *mass_col, *mass_address;
  const float *mass, *jacobian, *aref, *warm_acceleration, *smooth_acceleration;
  float *qacc, *ma, *jaref, *search_dot;
  int* niter;
  bool* done;
};
struct StopInputs {
  int worlds, nv, iterations, tolerance_count, inertia_count, warn_overflow;
  const float *tolerance, *inertia, *grad_dot, *decrement, *improvement, *alpha;
  int *niter, *overflow, *nsolving;
  bool* done;
};
}
extern "C" {
const char* flash_mjwarp_solver_state_source_id();
std::size_t flash_mjwarp_solver_state_init_bytes();
std::size_t flash_mjwarp_solver_state_stop_bytes();
int flash_mjwarp_solver_state_arch();
int flash_mjwarp_solver_state_initialize(const flash::mjwarp::InitInputs*, cudaStream_t);
int flash_mjwarp_solver_state_fused(const flash::mjwarp::InitInputs*, float*, bool*, int*, cudaStream_t);
int flash_mjwarp_solver_state_stop(const flash::mjwarp::StopInputs*, cudaStream_t);
}
