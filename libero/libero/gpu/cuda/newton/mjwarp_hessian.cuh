// SPDX-License-Identifier: Apache-2.0
// Host ABI for the independently compiled, handwritten FP32 assembly stage.
#pragma once
#include <cuda_runtime.h>
#include <cstddef>

namespace flash::mjwarp {

// Arrays use MJWarp's contiguous dense-J / lower-CSR-M layout. These are host
// launch arguments, not a device-side wrapper or a shared-memory allocation.
// The caller owns every array and its lifetime, including graph replays.
struct HessianInputs {
  int worlds, nv, padded_nv, rows, mass_entries, contact_capacity;
  int impratio_count, cone_launch_groups, cone_contacts_per_group;
  const int* mass_upper_address;  // [padded_nv,padded_nv], -1 for absent entries
  const int* nefc;               // [worlds]
  const float* mass;             // [worlds,mass_entries]
  const float* jacobian;         // [worlds,rows,padded_nv]
  const float* diagonal;         // [worlds,rows]
  const int* state;              // [worlds,rows], MuJoCo constraint-state enum
  const float* jaref;            // [worlds,rows]
  const bool* done;              // [worlds], skipped worlds retain ALL outputs
  const float* impratio_invsqrt; // [1] or [worlds]
  const int* triangle_row;       // [nv*(nv+1)/2]
  const int* triangle_column;    // [nv*(nv+1)/2]
  const int* contact_count;      // [1], pooled count bounded by capacity
  const int* contact_world;      // [contact_capacity]
  const int* contact_dimension;  // [contact_capacity], 1/3/4/6
  const float* contact_friction;// [contact_capacity,5]
  const float* contact_distance;// [contact_capacity]
  const float* contact_margin;  // [contact_capacity]
  const int* contact_address;   // [contact_capacity,6], -1 for absent rows
  float* hessian;               // [worlds,padded_nv,padded_nv]
};

}  // namespace flash::mjwarp

extern "C" {
std::size_t flash_mjwarp_hessian_arguments_bytes();
const char* flash_mjwarp_hessian_source_id();
int flash_mjwarp_hessian_arch();
// stage=0: full assembly; 1: mass/quadratic/padding; 2: add cone curvature.
// Stage 2 requires the mass/quadratic stage to have completed on this stream.
int flash_mjwarp_hessian_launch(const flash::mjwarp::HessianInputs*, int stage,
                               cudaStream_t stream);
}
