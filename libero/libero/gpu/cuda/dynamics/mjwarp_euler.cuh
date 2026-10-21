// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
namespace flash::mjwarp {
// Borrow current Model/Data arrays. The integration matrix/acceleration are
// private workspace; public Newton qacc and mass factors are never overwritten.
struct EulerInputs {
  int worlds, nv, nq, joints, matrix_width;
  int timestep_rows, damping_rows, dampingpoly_rows;
  const int *joint_type, *qposadr, *dofadr, *rowadr, *rownnz;
  const float *timestep, *damping, *dampingpoly, *acceleration;
  float *derivative, *matrix, *qvel, *qpos;
};
}
