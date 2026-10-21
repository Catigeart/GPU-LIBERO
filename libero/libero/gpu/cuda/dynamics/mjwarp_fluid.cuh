// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
namespace flash::mjwarp {
struct FluidInputs {
  int worlds,bodies,nv,mass_rows,inertia_rows,wind_rows,density_rows,viscosity_rows;
  const int *root,*dof_body,*subtree_size;
  const float *mass,*inertia,*wind,*density,*viscosity;
  const float *xipos,*ximat,*subtree_com,*cvel,*cdof;
  float *body_force,*qfrc;
};
}
