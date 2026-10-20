// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
namespace flash::mjwarp {
// Postconstraint RNE consumes solved qacc and assembled external body wrenches.
// It never writes generalized forces or Newton's qacc/warmstart.
struct PostconstraintInputs {
  int worlds, nv, bodies, branches, gravity_rows;
  const int *body_dofnum, *body_dofadr, *branch_bodies, *branch_start;
  const int *body_owner, *body_parent;
  const float *gravity, *qvel, *qacc, *cdof, *cinert, *cvel, *cdof_dot, *cfrc_ext;
  float *cacc, *cfrc_int;
};
}
