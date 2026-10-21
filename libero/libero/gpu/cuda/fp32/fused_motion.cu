// Copyright 2025 The Newton Developers
// Copyright 2026 Flash-LIBERO contributors
// SPDX-License-Identifier: Apache-2.0
// Current-input world-local spatial velocity and ordinary bias RNE.
#include "../dynamics/mjwarp_bias.cuh"
#include "../dynamics/mjwarp_velocity.cuh"
#ifndef FLASH_FP32_MOTION_SOURCE_ID
#error "Build through flash_fp32_fused_motion"
#endif
namespace flash::fp32 {
using flash::mjwarp::BiasInputs;
using flash::mjwarp::VelocityInputs;
constexpr int BodyLimit = 64, Threads = 512, Slots = 6 * BodyLimit;
__global__ void motion_bias(VelocityInputs v, BiasInputs b,
                            const int *interval_size, int gravity_disabled) {
  const int w = blockIdx.x, lane = threadIdx.x, warp = lane / 32,
            component = lane & 31;
  const float *__restrict__ qvel = v.qvel + w * v.nv;
  const float *__restrict__ cdof = v.cdof + w * v.nv * 6;
  const float *__restrict__ cinert = b.cinert;
  const float *__restrict__ gravity = b.gravity;
  float *__restrict__ cvel = v.cvel;
  float *__restrict__ dot_out = v.cdof_dot;
  float *__restrict__ cacc = b.cacc;
  float *__restrict__ cfrc = b.cfrc_int;
  float *__restrict__ qfrc = b.qfrc_bias;
  // The first arena carries acceleration then raw body force. The second
  // carries velocity then composite force. Both lifetimes are within one call.
  __shared__ float scratch[2 * Slots];
  float *raw_force = scratch;
  float *body_velocity = scratch + Slots;
  if (lane < 6) {
    cvel[6 * w * v.bodies + lane] = 0.f;
    cacc[6 * w * v.bodies + lane] =
        lane < 3 || gravity_disabled
            ? 0.f
            : -gravity[3 * (w % b.gravity_rows) + lane - 3];
  }
  for (int branch = warp; branch < v.branches; branch += Threads / 32) {
    if (component < 8) {
      float velocity = 0.f;
      float acceleration =
          component >= 3 && component < 6 && !gravity_disabled
              ? -gravity[3 * (w % b.gravity_rows) + component - 3]
              : 0.f;
      for (int at = v.branch_start[branch]; at < v.branch_start[branch + 1];
           ++at) {
        const int body = v.branch_bodies[at], first = v.body_jntadr[body],
                  count = v.body_jntnum[body];
        const bool publish = v.body_owner[body] == branch;
        int dof = v.body_dofadr[body];
        for (int joint = first; joint < first + count; ++joint) {
          const int type = v.jnt_type[joint];
          if (type == 0) {
            for (int j = 0; j < 3; ++j)
              if (component < 6) {
                const float speed = qvel[dof + j];
                velocity += cdof[6 * (dof + j) + component] * speed;
                acceleration += 0.f * speed; // same free-translation derivative
                if (publish)
                  dot_out[6 * (w * v.nv + dof + j) + component] = 0.f;
              }
            dof += 3;
          }
          const int width = type == 0 || type == 1 ? 3 : 1;
          float speed[3], motion_component[3];
#pragma unroll
          for (int j = 0; j < 3; ++j)
            if (j < width) {
              speed[j] = qvel[dof + j];
              motion_component[j] =
                  component < 6 ? cdof[6 * (dof + j) + component] : 0.f;
            }
          // Ball/free axes all see the pre-rotation velocity. Noncanonical
          // prefixes also compute derivatives for their private acceleration;
          // only the canonical branch publishes public derivative rows.
          const int axis = component % 3, i = (axis + 1) % 3,
                    k = (axis + 2) % 3;
          const float vi = __shfl_sync(0xffu, velocity, i, 8),
                      vk = __shfl_sync(0xffu, velocity, k, 8);
          const float li = __shfl_sync(0xffu, velocity, i + 3, 8),
                      lk = __shfl_sync(0xffu, velocity, k + 3, 8);
#pragma unroll
          for (int j = 0; j < 3; ++j)
            if (j < width) {
              const float *motion = cdof + 6 * (dof + j);
              float dot = vi * motion[k] - vk * motion[i];
              if (component >= 3)
                dot = (li * motion[k] - lk * motion[i]) +
                      (vi * motion[k + 3] - vk * motion[i + 3]);
              if (component < 6) {
                acceleration += dot * speed[j];
                if (publish)
                  dot_out[6 * (w * v.nv + dof + j) + component] = dot;
              }
            }
#pragma unroll
          for (int j = 0; j < 3; ++j)
            if (j < width && component < 6)
              velocity += motion_component[j] * speed[j];
          dof += width;
        }
        if (publish && component < 6) {
          cvel[6 * (w * v.bodies + body) + component] = velocity;
          cacc[6 * (w * v.bodies + body) + component] = acceleration;
          body_velocity[6 * body + component] = velocity;
          raw_force[6 * body + component] = acceleration;
        }
      }
    }
  }
  __syncthreads(); // independent body jobs can consume every current row
  for (int body = lane; body < b.bodies; body += Threads) {
    if (body == 0) {
      for (int k = 0; k < 6; ++k)
        raw_force[k] = 0.f;
      continue;
    }
    const int t = w * b.bodies + body;
    const float *inertia = cinert + 10 * t,
                *acceleration = raw_force + 6 * body,
                *velocity = body_velocity + 6 * body;
    float force[6], momentum[6], correction[6];
    force[0] = inertia[0] * acceleration[0] + inertia[3] * acceleration[1] +
               inertia[4] * acceleration[2] - inertia[8] * acceleration[4] +
               inertia[7] * acceleration[5];
    force[1] = inertia[3] * acceleration[0] + inertia[1] * acceleration[1] +
               inertia[5] * acceleration[2] + inertia[8] * acceleration[3] -
               inertia[6] * acceleration[5];
    force[2] = inertia[4] * acceleration[0] + inertia[5] * acceleration[1] +
               inertia[2] * acceleration[2] - inertia[7] * acceleration[3] +
               inertia[6] * acceleration[4];
    force[3] = inertia[8] * acceleration[1] - inertia[7] * acceleration[2] +
               inertia[9] * acceleration[3];
    force[4] = inertia[6] * acceleration[2] - inertia[8] * acceleration[0] +
               inertia[9] * acceleration[4];
    force[5] = inertia[7] * acceleration[0] - inertia[6] * acceleration[1] +
               inertia[9] * acceleration[5];
    momentum[0] = inertia[0] * velocity[0] + inertia[3] * velocity[1] +
                  inertia[4] * velocity[2] - inertia[8] * velocity[4] +
                  inertia[7] * velocity[5];
    momentum[1] = inertia[3] * velocity[0] + inertia[1] * velocity[1] +
                  inertia[5] * velocity[2] + inertia[8] * velocity[3] -
                  inertia[6] * velocity[5];
    momentum[2] = inertia[4] * velocity[0] + inertia[5] * velocity[1] +
                  inertia[2] * velocity[2] - inertia[7] * velocity[3] +
                  inertia[6] * velocity[4];
    momentum[3] = inertia[8] * velocity[1] - inertia[7] * velocity[2] +
                  inertia[9] * velocity[3];
    momentum[4] = inertia[6] * velocity[2] - inertia[8] * velocity[0] +
                  inertia[9] * velocity[4];
    momentum[5] = inertia[7] * velocity[0] - inertia[6] * velocity[1] +
                  inertia[9] * velocity[5];
    correction[0] = (velocity[1] * momentum[2] - velocity[2] * momentum[1]) +
                    (velocity[4] * momentum[5] - velocity[5] * momentum[4]);
    correction[1] = (velocity[2] * momentum[0] - velocity[0] * momentum[2]) +
                    (velocity[5] * momentum[3] - velocity[3] * momentum[5]);
    correction[2] = (velocity[0] * momentum[1] - velocity[1] * momentum[0]) +
                    (velocity[3] * momentum[4] - velocity[4] * momentum[3]);
    correction[3] = velocity[1] * momentum[5] - velocity[2] * momentum[4];
    correction[4] = velocity[2] * momentum[3] - velocity[0] * momentum[5];
    correction[5] = velocity[0] * momentum[4] - velocity[1] * momentum[3];
    for (int k = 0; k < 6; ++k)
      raw_force[6 * body + k] = force[k] + correction[k];
  }
  __syncthreads(); // velocity arena is dead; all raw force rows are current
  float *composite = body_velocity;
  for (int work = lane; work < 6 * b.bodies; work += Threads) {
    const int body = work / 6, k = work % 6;
    float value = raw_force[work];
    for (int node = body + 1; node < body + interval_size[body]; ++node)
      value += raw_force[6 * node + k];
    composite[work] = value;
    cfrc[6 * w * b.bodies + work] = value;
  }
  __syncthreads(); // projection uses the reused arena, not a global reload
  for (int dof = lane; dof < b.nv; dof += Threads) {
    const float *force = composite + 6 * b.dof_body[dof],
                *motion = cdof + 6 * dof;
    float value = 0.f;
    for (int k = 0; k < 6; ++k)
      value += motion[k] * force[k];
    qfrc[w * b.nv + dof] = value;
  }
}
} // namespace flash::fp32
extern "C" {
const char *flash_fp32_motion_source_id() {
  return FLASH_FP32_MOTION_SOURCE_ID;
}
int flash_fp32_motion_arch() { return FLASH_CUDA_ARCH; }
std::size_t flash_fp32_motion_velocity_bytes() {
  return sizeof(flash::mjwarp::VelocityInputs);
}
std::size_t flash_fp32_motion_bias_bytes() {
  return sizeof(flash::mjwarp::BiasInputs);
}
int flash_fp32_motion_launch(const flash::mjwarp::VelocityInputs *v,
                             const flash::mjwarp::BiasInputs *b,
                             const int *intervals, int gravity_disabled,
                             cudaStream_t stream) {
  if (!v || !b || !intervals || v->worlds < 1 || v->bodies < 2 ||
      v->bodies > 64 || v->branches < 1 || v->branches > 32 || v->nv < 1 ||
      v->worlds != b->worlds || v->bodies != b->bodies || v->nv != b->nv ||
      v->branches != b->branches || b->gravity_rows < 1 ||
      (gravity_disabled != 0 && gravity_disabled != 1))
    return cudaErrorInvalidValue;
  flash::fp32::motion_bias<<<v->worlds, flash::fp32::Threads, 0, stream>>>(
      *v, *b, intervals, gravity_disabled);
  return cudaGetLastError();
}
}
