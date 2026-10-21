// Copyright 2026 Flash-LIBERO contributors
#pragma once
namespace flash::bridge {
struct Gather {
  int worlds,site,nv,nq,nmass,nsites;
  const int *qi,*vi,*mi;
  const float *qpos,*qvel,*bias,*mass,*jp,*jr,*position,*rotation;
  float *out_mass,*out_jac,*out_q,*out_v,*out_bias,*out_pos,*out_rot,*out_vel;
};
}
