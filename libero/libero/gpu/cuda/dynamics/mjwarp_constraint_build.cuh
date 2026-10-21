// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstddef>

namespace flash::mjwarp {
struct ConstraintBroadcast { const float* data; int rows; };
struct ConstraintBuildInputs {
  int worlds,nv,nq,bodies,joints,geoms,contacts,rows,padded_rows,padded_nv,address_columns,limited;
  int cone,constraints_enabled,friction_enabled,limits_enabled,contacts_enabled,refsafe,adhesion_enabled;
  ConstraintBroadcast timestep,impratio_invsqrt,body_weight,dof_weight,dof_friction,dof_ref,dof_imp;
  ConstraintBroadcast joint_ref,joint_imp,joint_range,joint_margin;
  const int *parent,*root,*geom_body,*ancestor,*joint_qpos,*joint_dof,*limited_joint;
  const int *contact_count,*contact_dim,*contact_world,*contact_geom,*contact_type;
  const float *qpos,*qvel,*cdof,*subtree_com;
  const float *contact_dist,*contact_margin,*contact_pos,*contact_frame,*contact_friction;
  const float *contact_ref,*contact_ref_friction,*contact_imp,*contact_adhesion;
  int *ne,*nf,*nl,*nefc,*jtdaj_count,*contact_address,*row_type,*row_id;
  float *J,*Jqvel,*row_pos,*row_margin,*row_D,*row_vel,*row_aref,*row_friction;
};
}
