// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
namespace flash::mjwarp {
// Bias RNE deliberately has no qacc or external-force argument (flg_acc=false).
struct BiasInputs {
  int worlds, nv, bodies, branches, gravity_rows;
  const int *body_dofnum, *body_dofadr, *branch_bodies, *branch_start;
  const int *body_owner, *body_parent, *dof_body;
  const float *gravity, *qvel, *cdof, *cinert, *cvel, *cdof_dot;
  float *cacc, *cfrc_int, *qfrc_bias;
};
}
