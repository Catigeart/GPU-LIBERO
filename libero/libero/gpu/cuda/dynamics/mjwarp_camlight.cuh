// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstddef>

namespace flash::mjwarp {
// Borrowed MJWarp arrays; quaternion components follow its wxyz convention.
struct CamLightInputs {
  int worlds, bodies, cameras, lights;
  int cam_pos_rows, cam_quat_rows, light_pos_rows, light_dir_rows;
  const int *cam_body, *light_body;
  const float *cam_pos, *cam_quat, *light_pos, *light_dir, *xpos, *xquat;
  float *cam_xpos, *cam_xmat, *light_xpos, *light_xdir;
};
}
