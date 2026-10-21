// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstddef>

namespace flash::mjwarp {
struct ExternalWrenchInputs {
  int worlds, bodies, geoms, contact_capacity, rows, address_columns, cone;
  const int *root, *geom_body, *contact_dim, *contact_geom;
  const int *contact_address, *contact_world, *contact_count;
  const float *applied, *xipos, *subtree_com;
  const float *contact_pos, *contact_frame, *contact_friction, *contact_adhesion;
  const float *constraint_force;
  float *external;
};
}
