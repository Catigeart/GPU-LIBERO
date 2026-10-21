// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
namespace flash::mjwarp {
struct BroadphaseInputs {
  int worlds,geoms,pairs,explicit_pairs,capacity,filter;
  int dataid_rows,aabb_rows,rbound_rows,margin_rows,gap_rows,pair_margin_rows,pair_gap_rows;
  const int *geom_type,*geom_dataid,*pairs_in,*pair_ids_in;
  const float *aabb,*rbound,*margin,*gap,*pair_margin,*pair_gap,*xpos,*xmat;
  int *count,*pairs_out,*pair_ids_out,*world_out;
};
}
