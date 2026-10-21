// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
namespace flash::mjwarp {
struct TransmissionInputs {
  int worlds, nv, nq, actuators, moments, gear_rows;
  const int *joint_qposadr, *joint_dofadr, *actuator_joint;
  const float *gear, *qpos, *qvel;
  int *total, *rownnz, *rowadr, *column;
  float *moment, *length, *velocity;
};
}
