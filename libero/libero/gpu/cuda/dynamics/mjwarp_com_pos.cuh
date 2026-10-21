// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstddef>

namespace flash::mjwarp {
// Borrowed contiguous MJWarp arrays; vec10 and spatial vectors are angular-first.
struct ComPosInputs {
  int worlds, bodies, joints, nv, mass_rows, subtree_rows, inertia_rows;
  const int *parent, *root, *kind, *dofadr, *jbody;
  const float *mass, *subtreemass, *inertia;
  const float *xipos, *ximat, *xmat, *anchor, *axis;
  float *subtree, *cinert, *cdof;
};
}
