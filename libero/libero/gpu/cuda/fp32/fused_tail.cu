// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Readable fusion of the accepted MJWarp-derived FP32 Euler advancement.
#include "../dynamics/mjwarp_euler.cuh"
#include <cmath>

#ifndef FLASH_FP32_TAIL_SOURCE_ID
#error "Build through flash_fp32_fused_tail"
#endif

namespace flash::fp32 {
// One CTA owns one world. The two original numerical stages are adjacent;
// the barrier replaces the previous velocity-kernel completion dependency.
template<int Threads>
__global__ void advance(mjwarp::EulerInputs a) {
  const int world = blockIdx.x, lane = threadIdx.x;
  const float dt = a.timestep[world % a.timestep_rows];
  float* velocity = a.qvel + world * a.nv;
  for (int dof = lane; dof < a.nv; dof += Threads)
    velocity[dof] = velocity[dof] + a.acceleration[world*a.nv+dof]*dt;
  __syncthreads();

  for (int joint = lane; joint < a.joints; joint += Threads) {
    const int kind = a.joint_type[joint];
    float* q = a.qpos + world*a.nq + a.qposadr[joint];
    const float* v = velocity + a.dofadr[joint];
    if (kind == 0) {
      q[0] += dt*v[0]; q[1] += dt*v[1]; q[2] += dt*v[2];
      q += 3; v += 3;
    }
    if (kind < 2) {
      const float speed = sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
      float x=0.f, y=0.f, z=0.f;
      if (speed > 0.f) { x=v[0]/speed; y=v[1]/speed; z=v[2]/speed; }
      const float angle = dt*speed;
      const float s = sinf(angle*.5f), c = cosf(angle*.5f);
      const float bw=c, bx=x*s, by=y*s, bz=z*s;
      float qw=q[0], qx=q[1], qy=q[2], qz=q[3];
      float length = sqrtf(qw*qw + qx*qx + qy*qy + qz*qz);
      if (length > 0.f) { const float r=1.f/length; qw*=r; qx*=r; qy*=r; qz*=r; }
      else { qw=qx=qy=0.f; qz=1.f; } // pinned degenerate quaternion rule
      const float rw=qw*bw-qx*bx-qy*by-qz*bz;
      const float rx=qw*bx+qx*bw+qy*bz-qz*by;
      const float ry=qw*by-qx*bz+qy*bw+qz*bx;
      const float rz=qw*bz+qx*by-qy*bx+qz*bw;
      length = sqrtf(rw*rw + rx*rx + ry*ry + rz*rz);
      if (length > 0.f) {
        const float r=1.f/length; q[0]=rw*r; q[1]=rx*r; q[2]=ry*r; q[3]=rz*r;
      } else { q[0]=q[1]=q[2]=0.f; q[3]=1.f; }
    } else q[0] += dt*v[0];
  }
}
}

extern "C" {
const char* flash_fp32_tail_source_id() { return FLASH_FP32_TAIL_SOURCE_ID; }
int flash_fp32_tail_arch() { return FLASH_CUDA_ARCH; }
std::size_t flash_fp32_tail_arguments_bytes() { return sizeof(flash::mjwarp::EulerInputs); }
int flash_fp32_tail_advance(const flash::mjwarp::EulerInputs* a, cudaStream_t stream) {
  if (!a || a->worlds<1 || a->nv<1 || a->joints<1) return cudaErrorInvalidValue;
  flash::fp32::advance<64><<<a->worlds,64,0,stream>>>(*a);
  return cudaGetLastError();
}
}
