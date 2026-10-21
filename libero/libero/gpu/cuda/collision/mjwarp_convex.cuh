// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "mjwarp_convex_geometry.cuh"
#include "mjwarp_convex_simplex.cuh"
#include "mjwarp_convex_gjk.cuh"
#include "mjwarp_convex_epa.cuh"
#include "mjwarp_box_manifold.cuh"
#include "mjwarp_convex_collision.cuh"
#include <cstddef>
namespace flash::mjwarp {
// Function probe ABI, not yet a production convex collision selector.
struct ConvexGJKInputs {
  int worlds,geoms,requests,size_rows,dataid_rows;
  const int *geom_type,*geom_dataid,*mesh_vertadr,*mesh_vertnum,*mesh_graphadr,*mesh_graph;
  const float *size,*vertices,*xmat;
  const int *world,*pair,*iterations,*discrete,*cache1,*cache2;
  const float *position1,*position2,*initial1,*initial2,*margin1,*margin2,*tolerance,*cutoff;
  int *separated_out,*dim_out,*index1_out,*index2_out,*vertex1_out,*vertex2_out,*valid_out,*stop_out,*iterations_out,*queries_out;
  float *dist_out,*x1_out,*x2_out,*simplex_out,*simplex1_out,*simplex2_out,*direction_out,*lower_out;
};
struct ConvexFrozenGJK {
  const float *simplex,*simplex1,*simplex2,*distance,*x1,*x2;
  const int *count,*vertex1,*vertex2;
};
struct ConvexCCDInputs {
  ConvexGJKInputs query;
  int epa_iterations,vertex_capacity,face_capacity,horizon_capacity,warn,use_frozen_gjk;
  ConvexFrozenGJK frozen;
  float *vertices,*projection,*norm2;
  int *vertex_index,*face,*horizon,*overflow,*needs_epa_out,*ncontact_out,*face_index_out,*nvert_out,*nface_out,*status_out;
};
struct ConvexManifoldInputs {
  ConvexGJKInputs query;
  int vertex_capacity,face_capacity;
  const float* vertices;
  const int *vertex_index,*face,*face_index;
  const float *x1,*x2;
  int* count;
  float *witness1,*witness2,*distance;
};
struct ConvexSimplexInputs {
  int requests;
  const float* simplex;
  const int* count;
  float *weights,*point,*distance;
  int* mask;
};
struct ConvexSupportInputs {
  int worlds,geoms,requests,size_rows,dataid_rows;
  const int *geom_type,*geom_dataid,*mesh_vertadr,*mesh_vertnum,*mesh_graphadr,*mesh_graph;
  const float *size,*vertices,*xpos,*xmat;
  const int *world,*geom,*cached;
  const float *direction,*margin;
  float *point;
  int *cached_out,*vertex_out;
};
}
