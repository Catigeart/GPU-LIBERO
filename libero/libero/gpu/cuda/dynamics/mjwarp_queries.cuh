// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Pinned support.jac_dof, force/torque sensors and rigid gravity projection.
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
namespace flash::mjwarp {
struct QueryGeometry {
  int worlds,bodies,dofs,ancestor_stride;
  const int *root,*ancestor;
  const float *com,*cdof;
};
struct JacobianInputs {
  QueryGeometry geometry;
  const float* point;
  const int* body;
  float *position,*rotation;
};
struct ForceSensorInputs {
  int worlds,bodies,sites,sensors,channels,disabled;
  const int *root,*site_body,*type,*object,*address,*datatype;
  const float *cutoff,*site_position,*site_rotation,*com,*wrench;
  float* output;
};
struct GravityInputs {
  QueryGeometry geometry;
  int gravity_rows,mass_rows,coefficient_rows;
  const float *gravity,*mass,*coefficient,*position;
  float* output;
};
}
