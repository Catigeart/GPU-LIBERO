// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cmath>
namespace flash::mjwarp::passive_math {
__device__ inline float polynomial(float linear, const float* poly, float x) {
  float value=linear;
  value+=poly[0]*x;
  value+=poly[1]*x*x;
  return value;
}

__device__ inline void rotation_difference(const float* q, const float* ref, float* out) {
  float norm=sqrtf(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
  float v[4];
  // Warp1.18 quat::normalize has kEps=0 and this exact fallback layout.
  if (norm>0.0f) { float inv=1.0f/norm; for (int k=0;k<4;++k) v[k]=q[k]*inv; }
  else { v[0]=0.0f; v[1]=0.0f; v[2]=0.0f; v[3]=1.0f; }
  float rw=ref[0], rx=-ref[1], ry=-ref[2], rz=-ref[3];
  float w=rw*v[0]-rx*v[1]-ry*v[2]-rz*v[3];
  float x=rw*v[1]+rx*v[0]+ry*v[3]-rz*v[2];
  float y=rw*v[2]-rx*v[3]+ry*v[0]+rz*v[1];
  float z=rw*v[3]+rx*v[2]-ry*v[1]+rz*v[0];
  float sine=sqrtf(x*x+y*y+z*z);
  if (sine==0.0f) { out[0]=out[1]=out[2]=0.0f; return; }
  constexpr float pi=3.14159265358979323846f;
  float speed=2.0f*atan2f(sine,w);
  if (speed>pi) speed-=2.0f*pi;
  // Keep axis*speed/sine order instead of moving the division before multiply.
  out[0]=x*speed/sine;out[1]=y*speed/sine;out[2]=z*speed/sine;
}

}
