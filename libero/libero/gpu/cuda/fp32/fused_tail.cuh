// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "../dynamics/mjwarp_euler.cuh"
#include "../dynamics/mjwarp_publication.cuh"
namespace flash::fp32 {
struct TailInputs {
  mjwarp::EulerInputs integration;
  mjwarp::PublicationInputs publication;
  const float* solved_acceleration;
  float* warmstart;
};
}
