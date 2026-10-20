// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
namespace flash::mjwarp {
struct PassiveInputs {
  int worlds, nv, nq, joints;
  int spring_rows, stiffness_rows, stiffnesspoly_rows, damping_rows, dampingpoly_rows;
  int disable_spring, disable_damper, gravity_enabled, has_fluid, has_adhesion;
  const int *joint_type, *qposadr, *dofadr, *dof_joint, *actgravcomp;
  const float *qpos_spring, *stiffness, *stiffnesspoly, *damping, *dampingpoly;
  const float *qpos, *qvel, *gravcomp, *fluid, *adhesion;
  float *spring, *damper, *passive;
};
}
