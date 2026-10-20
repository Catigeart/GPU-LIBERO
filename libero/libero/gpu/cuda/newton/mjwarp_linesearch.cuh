// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstddef>

namespace flash::mjwarp {
// Host launch descriptor, borrowing MJWarp device arrays. No shared-memory
// packing, device allocation, generated Warp code, or hidden upstream calls.
struct LinesearchInputs {
  int worlds, nv, pad, rows, mass_entries, capacity;
  int tolerance_count, ls_tolerance_count, impratio_count, inertia_count;
  int iterations, warn_overflow;
  const bool* done;
  const int *ne, *nf, *nefc, *contact_count;
  const int *mass_row, *mass_col, *mass_address;
  const float *mass, *jacobian, *search, *search_dot, *smooth;
  const float *tolerance, *ls_tolerance, *impratio, *inertia;
  const int *row_type, *row_contact, *contact_dimension, *contact_address;
  const float *contact_friction, *diagonal, *frictionloss;
  float *mv, *jv, *quad, *qacc, *ma, *jaref, *improvement, *alpha;
  int* overflow;
};
}
extern "C" {
const char* flash_mjwarp_linesearch_source_id();
std::size_t flash_mjwarp_linesearch_arguments_bytes();
int flash_mjwarp_linesearch_arch();
// 0: products + iterative search; 1: products only (Jv separate only if NV>50);
// 2: iterative search only (fused Jv if NV<=50); 3: standalone Jv for any NV.
int flash_mjwarp_linesearch_launch(const flash::mjwarp::LinesearchInputs*, int,
                                  cudaStream_t);
}
