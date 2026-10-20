// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstddef>

namespace flash::mjwarp {
// Resident arrays use the pinned MJWarp layout. Only dimensions/addresses are
// passed by value; physical inputs are read again on every Graph replay.
struct MassMatrixInputs {
  int worlds, nv, bodies, matrix_width, armature_rows;
  const int *body_parent, *dof_body, *dof_parent, *matrix_row, *matrix_nnz;
  const float *armature, *cdof;
  float *crb, *matrix;
};
}
