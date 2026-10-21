// Copyright 2026 Flash-LIBERO contributors
#pragma once
#include "../runtime/bridge_types.cuh"
#include "../dynamics/mjwarp_queries.cuh"
namespace flash::fp32 {
struct ControlPrepareInputs {
  flash::bridge::Gather gather;
  flash::mjwarp::JacobianInputs jacobian;
  float* points;
  int* policy;
};
}
