// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Pinned MJWarp smooth.py::rne_postconstraint, after external-wrench assembly.
#include "mjwarp_postconstraint.cuh"
#include "mjwarp_queries.cuh"
#include "mjwarp_spatial.cuh"
#ifndef FLASH_MJW_POSTCONSTRAINT_SOURCE_ID
#error "Build through flash_mjwarp_postconstraint_fp32"
#endif
namespace flash::mjwarp {
namespace {
__global__ void acceleration_root(PostconstraintInputs a) {
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= a.worlds)
    return;
  float *acc = a.cacc + w * a.bodies * 6;
  for (int k = 0; k < 3; ++k) {
    acc[k] = 0.0f;
    acc[k + 3] = -a.gravity[(w % a.gravity_rows) * 3 + k];
  }
}

__global__ void acceleration_branch(PostconstraintInputs a) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.worlds * a.branches)
    return;
  int w = t / a.branches, branch = t % a.branches;
  float acc[6];
  for (int k = 0; k < 6; ++k)
    acc[k] = a.cacc[w * a.bodies * 6 + k];
  // Complete leaf paths allow a private prefix; only the body's owner
  // publishes.
  for (int at = a.branch_start[branch]; at < a.branch_start[branch + 1]; ++at) {
    int b = a.branch_bodies[at], first = a.body_dofadr[b];
    for (int j = 0; j < a.body_dofnum[b]; ++j) {
      int d = w * a.nv + first + j;
      // Keep the pinned per-DOF order: velocity term, then acceleration term.
      for (int k = 0; k < 6; ++k)
        acc[k] += a.cdof_dot[d * 6 + k] * a.qvel[d];
      for (int k = 0; k < 6; ++k)
        acc[k] += a.cdof[d * 6 + k] * a.qacc[d];
    }
    if (a.body_owner[b] == branch)
      for (int k = 0; k < 6; ++k)
        a.cacc[(w * a.bodies + b) * 6 + k] = acc[k];
  }
}

__global__ void body_force(PostconstraintInputs a) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.worlds * a.bodies)
    return;
  float *out = a.cfrc_int + t * 6;
  // World external wrench is ignored, matching upstream's public convention.
  if (t % a.bodies == 0) {
    for (int k = 0; k < 6; ++k)
      out[k] = 0.0f;
    return;
  }
  const float *inertia = a.cinert + t * 10, *velocity = a.cvel + t * 6;
  float force[6], momentum[6], cross[6];
  inert_vec(inertia, a.cacc + t * 6, force);
  inert_vec(inertia, velocity, momentum);
  force_cross(velocity, momentum, cross);
  for (int k = 0; k < 6; ++k)
    out[k] = (force[k] + cross[k]) - a.cfrc_ext[t * 6 + k];
}

__global__ void force_backward(PostconstraintInputs a, const int *layer,
                               int count) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.worlds * count)
    return;
  int w = t / count, b = layer[t % count];
  if (b == 0)
    return;
  const float *child = a.cfrc_int + (w * a.bodies + b) * 6;
  float *parent = a.cfrc_int + (w * a.bodies + a.body_parent[b]) * 6;
  for (int k = 0; k < 6; ++k)
    atomicAdd(parent + k, child[k]);
}
// One CTA owns the complete bounded world suffix. Every component builds an
// independent private acceleration prefix. Shared acceleration rows become raw
// force rows only after each body's sole force consumer has read all six.
constexpr int PostThreads = 256;
template <bool Sensors>
__global__ void world_postconstraint(PostconstraintInputs a,
                                     const int *intervals, int gravity_disabled,
                                     ForceSensorInputs sensor) {
  const int w = blockIdx.x, lane = threadIdx.x;
  __shared__ float scratch[6 * 64];
  if (lane < 6)
    a.cacc[6 * w * a.bodies + lane] =
        lane < 3 || gravity_disabled
            ? 0.f
            : -a.gravity[3 * (w % a.gravity_rows) + lane - 3];
  const int component = lane & 7;
  if (component < 6) {
    for (int branch = lane / 8; branch < a.branches;
         branch += PostThreads / 8) {
      float acc = component < 3 || gravity_disabled
                      ? 0.f
                      : -a.gravity[3 * (w % a.gravity_rows) + component - 3];
      for (int at = a.branch_start[branch]; at < a.branch_start[branch + 1];
           ++at) {
        const int body = a.branch_bodies[at], first = a.body_dofadr[body];
        for (int j = 0; j < a.body_dofnum[body]; ++j) {
          const int d = w * a.nv + first + j;
          acc += a.cdof_dot[6 * d + component] * a.qvel[d];
          acc += a.cdof[6 * d + component] * a.qacc[d];
        }
        if (a.body_owner[body] == branch) {
          a.cacc[6 * (w * a.bodies + body) + component] = acc;
          scratch[6 * body + component] = acc;
        }
      }
    }
  }
  __syncthreads(); // all current acceleration producers have finished
  for (int body = lane; body < a.bodies; body += PostThreads) {
    if (!body) {
      for (int k = 0; k < 6; ++k)
        scratch[k] = 0.f;
      continue;
    }
    const int t = w * a.bodies + body;
    const float *inertia = a.cinert + 10 * t,
                *acceleration = scratch + 6 * body, *velocity = a.cvel + 6 * t;
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
      scratch[6 * body + k] =
          (force[k] + correction[k]) - a.cfrc_ext[6 * t + k];
  }
  __syncthreads(); // each raw force row is complete; no hierarchy atomics
  for (int slot = lane; slot < 6 * a.bodies; slot += PostThreads) {
    const int body = slot / 6, k = slot % 6;
    float force = 0.f;
    for (int node = body; node < body + intervals[body]; ++node)
      force += scratch[6 * node + k];
    a.cfrc_int[6 * w * a.bodies + slot] = force;
  }
  // Sensor owners reduce their own bounded raw subtree directly. The shared
  // raw forces remain live, so sensors need neither a global wrench reload nor
  // a new block barrier after composite-wrench publication.
  if constexpr (Sensors) {
    if (!sensor.disabled) {
      const int component = lane & 7;
      const unsigned group_mask = 0xffu << (lane & 24);
      for (int s = lane / 8; s < sensor.sensors; s += PostThreads / 8) {
        const int site = sensor.object[s], body = sensor.site_body[site];
        float sum = 0.f;
        if (component < 6)
          for (int node = body; node < body + intervals[body]; ++node)
            sum += scratch[6 * node + component];
        float wrench[6];
        for (int k = 0; k < 6; ++k)
          wrench[k] = __shfl_sync(group_mask, sum, k, 8);
        if (component)
          continue; // the scalar frame/cutoff consumer uses all six
        float value[3];
        if (sensor.type[s] == 4) {
          for (int k = 0; k < 3; ++k)
            value[k] = wrench[k + 3];
        } else {
          const float *com =
              sensor.com + 3 * (w * sensor.bodies + sensor.root[body]);
          const float *position =
              sensor.site_position + 3 * (w * sensor.sites + site);
          const float dx = position[0] - com[0], dy = position[1] - com[1],
                      dz = position[2] - com[2];
          value[0] = wrench[0] - (dy * wrench[5] - dz * wrench[4]);
          value[1] = wrench[1] - (dz * wrench[3] - dx * wrench[5]);
          value[2] = wrench[2] - (dx * wrench[4] - dy * wrench[3]);
        }
        const float *rotation =
            sensor.site_rotation + 9 * (w * sensor.sites + site);
        const float cutoff = sensor.cutoff[s];
        for (int k = 0; k < 3; ++k) {
          float value_out = rotation[k] * value[0] +
                            rotation[3 + k] * value[1] +
                            rotation[6 + k] * value[2];
          if (cutoff > 0) {
            if (sensor.datatype[s] == 0)
              value_out = fminf(cutoff, fmaxf(-cutoff, value_out));
            else if (sensor.datatype[s] == 1)
              value_out = fminf(cutoff, value_out);
          }
          sensor.output[w * sensor.channels + sensor.address[s] + k] =
              value_out;
        }
      }
    }
  }
}

} // namespace
} // namespace flash::mjwarp
extern "C" {
const char *flash_mjwarp_postconstraint_source_id() {
  return FLASH_MJW_POSTCONSTRAINT_SOURCE_ID;
}
std::size_t flash_mjwarp_postconstraint_arguments_bytes() {
  return sizeof(flash::mjwarp::PostconstraintInputs);
}
int flash_mjwarp_postconstraint_arch() { return FLASH_MJW_POSTCONSTRAINT_ARCH; }

int flash_mjwarp_postconstraint_fused(
    const flash::mjwarp::PostconstraintInputs *a, const int *intervals,
    int gravity_disabled, cudaStream_t stream) {
  if (!a || !intervals || a->worlds < 1 || a->bodies < 2 || a->bodies > 64 ||
      a->branches < 1 || a->branches > 32 || a->nv < 1 || a->gravity_rows < 1 ||
      (gravity_disabled != 0 && gravity_disabled != 1))
    return int(cudaErrorInvalidValue);
  flash::mjwarp::world_postconstraint<false>
      <<<a->worlds, flash::mjwarp::PostThreads, 0, stream>>>(
          *a, intervals, gravity_disabled, {});
  return int(cudaGetLastError());
}

std::size_t flash_mjwarp_postconstraint_sensor_bytes() {
  return sizeof(flash::mjwarp::ForceSensorInputs);
}
int flash_mjwarp_postconstraint_sensors(
    const flash::mjwarp::PostconstraintInputs *a, const int *intervals,
    int gravity_disabled, const flash::mjwarp::ForceSensorInputs *sensor,
    cudaStream_t stream) {
  if (!a || !intervals || !sensor || a->worlds < 1 || a->bodies < 2 ||
      a->bodies > 64 || a->branches < 1 || a->branches > 32 ||
      a->gravity_rows < 1 || a->nv < 1 ||
      (gravity_disabled != 0 && gravity_disabled != 1) ||
      sensor->worlds != a->worlds || sensor->bodies != a->bodies ||
      sensor->sensors < 1 || sensor->sensors > 32 ||
      sensor->sensors * a->bodies > 256 || sensor->wrench != a->cfrc_int)
    return int(cudaErrorInvalidValue);
  flash::mjwarp::world_postconstraint<true>
      <<<a->worlds, flash::mjwarp::PostThreads, 0, stream>>>(
          *a, intervals, gravity_disabled, *sensor);
  return int(cudaGetLastError());
}
int flash_mjwarp_postconstraint_launch(
    const flash::mjwarp::PostconstraintInputs *a, int phase, const int *layer,
    int count, cudaStream_t stream) {
  if (!a || a->worlds < 1 || a->nv < 1 || a->bodies < 2 || a->branches < 1 ||
      a->gravity_rows < 1)
    return int(cudaErrorInvalidValue);
  using namespace flash::mjwarp;
  if (phase == 0)
    acceleration_root<<<(a->worlds + 255) / 256, 256, 0, stream>>>(*a);
  else if (phase == 1)
    acceleration_branch<<<(a->worlds * a->branches + 255) / 256, 256, 0,
                          stream>>>(*a);
  else if (phase == 2)
    body_force<<<(a->worlds * a->bodies + 255) / 256, 256, 0, stream>>>(*a);
  else if (phase == 3) {
    if (!layer || count < 1)
      return int(cudaErrorInvalidValue);
    force_backward<<<(a->worlds * count + 255) / 256, 256, 0, stream>>>(
        *a, layer, count);
  } else
    return int(cudaErrorInvalidValue);
  return int(cudaGetLastError());
}
}
