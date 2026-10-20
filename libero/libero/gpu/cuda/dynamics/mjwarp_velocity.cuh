// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
namespace flash::mjwarp {
struct VelocityInputs {
  int worlds, nv, bodies, branches;
  const int *body_jntnum, *body_jntadr, *body_dofadr, *jnt_type;
  const int *branch_bodies, *branch_start, *body_owner;
  const float *qvel, *cdof;
  float *cvel, *cdof_dot;
};
}
