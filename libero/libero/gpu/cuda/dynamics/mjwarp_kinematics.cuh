// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include "mjwarp_camlight.cuh"
namespace flash::mjwarp {
struct KinematicsInputs {
  int worlds,bodies,joints,nq,branches,geoms,sites,mocaps;
  int qpos0_rows,body_pos_rows,body_quat_rows,jnt_pos_rows,jnt_axis_rows;
  int ipos_rows,iquat_rows,gpos_rows,gquat_rows,spos_rows,squat_rows;
  const int *parent,*mocap,*jnum,*jadr,*kind,*qadr,*branch_body,*branch_start,*owner;
  const int *root,*weld,*geom_body,*site_body;
  const float *qpos0,*body_pos,*body_quat,*jnt_pos,*jnt_axis;
  const float *ipos,*iquat,*gpos,*gquat,*spos,*squat,*qpos,*mocap_pos,*mocap_quat;
  float *xpos,*xquat,*anchor,*axis,*xmat,*xipos,*ximat,*geom_xpos,*geom_xmat,*site_xpos,*site_xmat;
};
// Immutable canonical-branch index grouping; no cached physical values.
struct AttachedTopology {
  const int *geom_start,*geom_id,*site_start,*site_id;
};
struct FrameAttachments {
  const int *start,*job;
};

}
