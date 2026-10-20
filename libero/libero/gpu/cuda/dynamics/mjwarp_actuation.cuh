// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
namespace flash::mjwarp {
// Stateless scalar-joint actuator domain; all arrays borrow current Model/Data.
struct ActuationInputs {
  int worlds, nv, controls, actuators, joints, moments;
  int gain_rows, bias_rows, ctrlrange_rows, forcerange_rows, jointrange_rows;
  int disable_clampctrl, gravity_enabled;
  const int *gain_type, *bias_type, *ctrladr, *ctrlnum, *dof_joint, *actgravcomp;
  const bool *ctrllimited, *forcelimited, *jointlimited;
  const float *gainprm, *biasprm, *ctrlrange, *forcerange, *jointrange;
  const int *moment_rownnz, *moment_rowadr, *moment_colind;
  const float *moment, *ctrl, *length, *velocity, *gravcomp;
  float *force, *qfrc;
};
}
