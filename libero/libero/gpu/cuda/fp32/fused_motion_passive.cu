// Copyright 2025 The Newton Developers
// Copyright 2026 Flash-LIBERO contributors
// SPDX-License-Identifier: Apache-2.0
// Current-input world-local motion, bias, joint/fluid and passive forces.
#include "../dynamics/mjwarp_bias.cuh"
#include "../dynamics/mjwarp_fluid.cuh"
#include "../dynamics/mjwarp_passive.cuh"
#include "../dynamics/mjwarp_passive_math.cuh"
#include "../dynamics/mjwarp_velocity.cuh"
#ifndef FLASH_FP32_MOTION_PASSIVE_SOURCE_ID
#error "Build through flash_fp32_fused_motion_passive"
#endif
namespace flash::fp32 {
using flash::mjwarp::BiasInputs;
using flash::mjwarp::VelocityInputs;
using flash::mjwarp::passive_math::polynomial;
using flash::mjwarp::passive_math::rotation_difference;
struct V {
  float x, y, z;
};
__device__ inline V load3(const float *p, int i) {
  p += 3 * i;
  return {p[0], p[1], p[2]};
}
__device__ inline V operator-(V a, V b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
__device__ inline V cross(V a, V b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
__device__ inline V rotate(const float *r, V v) {
  return {r[0] * v.x + r[1] * v.y + r[2] * v.z,
          r[3] * v.x + r[4] * v.y + r[5] * v.z,
          r[6] * v.x + r[7] * v.y + r[8] * v.z};
}
__device__ inline V rotate_transpose(const float *r, V v) {
  return {r[0] * v.x + r[3] * v.y + r[6] * v.z,
          r[1] * v.x + r[4] * v.y + r[7] * v.z,
          r[2] * v.x + r[5] * v.y + r[8] * v.z};
}

constexpr int BodyLimit = 64, Threads = 512, Slots = 6 * BodyLimit;
__global__ void motion_passive(VelocityInputs v, BiasInputs b,
                               flash::mjwarp::FluidInputs f,
                               flash::mjwarp::PassiveInputs p, float *adhesion,
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
  const bool packed = 18 * b.bodies <= 2 * Slots;
  // Three current spatial arrays fit for NB<=42. Other admitted bodies keep
  // the two original fixed arenas and the published global fluid workspace.
  float *raw_force = scratch;
  float *body_velocity = scratch + (packed ? 6 * b.bodies : Slots);
  float *fluid_force =
      packed ? scratch + 12 * b.bodies : f.body_force + 6 * w * f.bodies;
  if (lane < 6) {
    cvel[6 * w * v.bodies + lane] = 0.f;
    cacc[6 * w * v.bodies + lane] =
        lane < 3 || gravity_disabled
            ? 0.f
            : -gravity[3 * (w % b.gravity_rows) + lane - 3];
  }
  // Joint and spatial-prefix producers are independent. The first existing
  // world barrier orders both before force consumption.
  for (int j = lane; j < p.joints; j += Threads) {
    const int d = p.dofadr[j], kind = p.joint_type[j];
    int pos = p.qposadr[j];
    const int width = kind == 0 ? 6 : (kind == 1 ? 3 : 1);
    const float *q = p.qpos + w * p.nq;
    const float *ref = p.qpos_spring + (w % p.spring_rows) * p.nq;
    float k = p.stiffness[(w % p.stiffness_rows) * p.joints + j];
    const float *sp =
        p.stiffnesspoly + ((w % p.stiffnesspoly_rows) * p.joints + j) * 2;
    bool has_stiffness =
        !p.disable_spring && (k != 0.0f || sp[0] != 0.0f || sp[1] != 0.0f);
    for (int i = 0; i < width; ++i) {
      float damping = p.damping[(w % p.damping_rows) * p.nv + d + i];
      const float *dp =
          p.dampingpoly + ((w % p.dampingpoly_rows) * p.nv + d + i) * 2;
      float v = p.qvel[w * p.nv + d + i], force = 0.0f;
      if (!p.disable_damper &&
          (damping != 0.0f || dp[0] != 0.0f || dp[1] != 0.0f))
        force = -v * polynomial(damping, dp, fabsf(v));
      p.damper[w * p.nv + d + i] = force;
      p.spring[w * p.nv + d + i] = 0.0f;
    }
    if (!has_stiffness)
      continue;
    float *spring = p.spring + w * p.nv + d;
    if (kind == 0) {
      float x = q[pos] - ref[pos], y = q[pos + 1] - ref[pos + 1],
            z = q[pos + 2] - ref[pos + 2];
      float gain = polynomial(k, sp, sqrtf(x * x + y * y + z * z));
      spring[0] = -gain * x;
      spring[1] = -gain * y;
      spring[2] = -gain * z;
      spring += 3;
      pos += 3;
    }
    if (kind == 0 || kind == 1) {
      float delta[3];
      rotation_difference(q + pos, ref + pos, delta);
      float r = sqrtf(delta[0] * delta[0] + delta[1] * delta[1] +
                      delta[2] * delta[2]);
      float gain = polynomial(k, sp, r);
      for (int i = 0; i < 3; ++i)
        spring[i] = -gain * delta[i];
    } else {
      float delta = q[pos] - ref[pos];
      spring[0] = -delta * polynomial(k, sp, delta);
    }
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
      for (int k = 0; k < 6; ++k) {
        raw_force[k] = 0.f;
        fluid_force[k] = 0.f;
      }
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
    // The shared current velocity row is still live here. Publish fluid
    // wrenches before this arena becomes composite bias force.
    {
      float *output = fluid_force + 6 * body;
      float mass = f.mass[(w % f.mass_rows) * f.bodies + body];
      constexpr float minval = 1.0e-15f, pi = 3.14159265358979323846f;
      if (body == 0 || mass < minval) {
        for (int k = 0; k < 6; ++k)
          output[k] = 0.0f;
        continue;
      }
      const float *rotation = f.ximat + 9 * t;
      const float *velocity = body_velocity + 6 * body;
      V angular = {velocity[0], velocity[1], velocity[2]},
        linear = {velocity[3], velocity[4], velocity[5]};
      V offset =
          load3(f.xipos, t) - load3(f.subtree_com, w * f.bodies + f.root[body]);
      // cvel uses the root-CoM origin; translate to this body's inertial
      // origin.
      V local_angular = rotate_transpose(rotation, angular);
      V local_linear =
          rotate_transpose(rotation, linear - cross(offset, angular));
      V wind = load3(f.wind, w % f.wind_rows);
      if (wind.x || wind.y || wind.z)
        local_linear = local_linear - rotate_transpose(rotation, wind);
      float density = f.density[w % f.density_rows],
            viscosity = f.viscosity[w % f.viscosity_rows];
      float box[3] = {0.0f, 0.0f, 0.0f};
      if (viscosity > 0.0f || density > 0.0f) {
        V inertia = load3(f.inertia, (w % f.inertia_rows) * f.bodies + body);
        float scale = 6.0f / mass;
        box[0] =
            sqrtf(fmaxf(minval, inertia.y + inertia.z - inertia.x) * scale);
        box[1] =
            sqrtf(fmaxf(minval, inertia.x + inertia.z - inertia.y) * scale);
        box[2] =
            sqrtf(fmaxf(minval, inertia.x + inertia.y - inertia.z) * scale);
      }
      float angular_local[3] = {local_angular.x, local_angular.y,
                                local_angular.z};
      float linear_local[3] = {local_linear.x, local_linear.y, local_linear.z};
      float force[3] = {0.0f, 0.0f, 0.0f}, torque[3] = {0.0f, 0.0f, 0.0f};
      if (viscosity > 0.0f) {
        float diameter = (box[0] + box[1] + box[2]) / 3.0f;
        float diameter3 = powf(diameter, 3.0f);
        for (int k = 0; k < 3; ++k) {
          torque[k] = -angular_local[k] * diameter3 * pi * viscosity;
          force[k] = -3.0f * linear_local[k] * diameter * pi * viscosity;
        }
      }
      if (density > 0.0f) {
        float power4[3] = {powf(box[0], 4.0f), powf(box[1], 4.0f),
                           powf(box[2], 4.0f)};
        float scale = density / 64.0f;
        for (int k = 0; k < 3; ++k) {
          int i = (k + 1) % 3, j = (k + 2) % 3;
          // Keep x/z factor order as in the pinned component expressions.
          if (k == 1) {
            i = 0;
            j = 2;
          }
          force[k] -= 0.5f * density * box[i] * box[j] *
                      fabsf(linear_local[k]) * linear_local[k];
          torque[k] -= box[k] * (power4[i] + power4[j]) *
                       fabsf(angular_local[k]) * angular_local[k] * scale;
        }
      }
      V world_force = rotate(rotation, {force[0], force[1], force[2]});
      V world_torque = rotate(rotation, {torque[0], torque[1], torque[2]});
      // Wrench storage is (force,torque); cvel/cdof storage is
      // (angular,linear).
      output[0] = world_force.x;
      output[1] = world_force.y;
      output[2] = world_force.z;
      output[3] = world_torque.x;
      output[4] = world_torque.y;
      output[5] = world_torque.z;
    }
  }
  __syncthreads(); // velocity arena is dead; all raw force rows are current
  float *composite = body_velocity;
  for (int work = lane; work < 6 * b.bodies; work += Threads) {
    const int body = work / 6, k = work % 6;
    if (packed)
      f.body_force[6 * w * f.bodies + work] = fluid_force[work];
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
    {
      const int t = w * f.nv + dof, fluid_body = f.dof_body[dof];
      // Reuse this DOF's current motion vector from the bias projection.
      float jac[6] = {motion[3], motion[4], motion[5],
                      motion[0], motion[1], motion[2]};
      V angular = {motion[0], motion[1], motion[2]};
      V root_com = load3(f.subtree_com, w * f.bodies + f.root[fluid_body]);
      float sum = 0.0f;
      for (int node = fluid_body;
           node < fluid_body + f.subtree_size[fluid_body]; ++node) {
        const float *force = fluid_force + 6 * node;
        if (force[0] == 0.0f && force[1] == 0.0f && force[2] == 0.0f &&
            force[3] == 0.0f && force[4] == 0.0f && force[5] == 0.0f)
          continue;
        V shift =
            cross(angular, load3(f.xipos, w * f.bodies + node) - root_com);
        float dot6 = 0.0f;
        for (int k = 0; k < 6; ++k)
          dot6 += jac[k] * force[k];
        float dot3 =
            shift.x * force[0] + shift.y * force[1] + shift.z * force[2];
        sum += dot6 + dot3;
      }
      f.qfrc[w * f.nv + dof] = sum;
      adhesion[t] = 0.f;
      float total = p.spring[t];
      total += p.damper[t];
      if (p.gravity_enabled && !p.actgravcomp[p.dof_joint[dof]])
        total += p.gravcomp[t];
      total += sum;
      if (p.has_adhesion)
        total += 0.f;
      p.passive[t] = total;
    }
  }
}
} // namespace flash::fp32
extern "C" {
const char *flash_fp32_motion_passive_source_id() {
  return FLASH_FP32_MOTION_PASSIVE_SOURCE_ID;
}
int flash_fp32_motion_passive_arch() { return FLASH_CUDA_ARCH; }
std::size_t flash_fp32_motion_passive_velocity_bytes() {
  return sizeof(flash::mjwarp::VelocityInputs);
}
std::size_t flash_fp32_motion_passive_bias_bytes() {
  return sizeof(flash::mjwarp::BiasInputs);
}
std::size_t flash_fp32_motion_passive_fluid_bytes() {
  return sizeof(flash::mjwarp::FluidInputs);
}
std::size_t flash_fp32_motion_passive_passive_bytes() {
  return sizeof(flash::mjwarp::PassiveInputs);
}
int flash_fp32_motion_passive_launch(const flash::mjwarp::VelocityInputs *v,
                                     const flash::mjwarp::BiasInputs *b,
                                     const flash::mjwarp::FluidInputs *f,
                                     const flash::mjwarp::PassiveInputs *p,
                                     float *adhesion, const int *intervals,
                                     int gravity_disabled,
                                     cudaStream_t stream) {
  if (!v || !b || !f || !p || !intervals || !adhesion || v->worlds < 1 ||
      v->bodies < 2 || v->bodies > 64 || v->branches < 1 || v->branches > 32 ||
      v->nv < 1 || p->joints < 1 || p->nq < 1 || v->worlds != b->worlds ||
      v->worlds != f->worlds || v->worlds != p->worlds || v->nv != b->nv ||
      v->nv != f->nv || v->nv != p->nv || v->bodies != b->bodies ||
      v->bodies != f->bodies || v->branches != b->branches ||
      b->gravity_rows < 1 || !p->has_fluid || p->fluid != f->qfrc ||
      p->adhesion != adhesion || f->mass_rows < 1 || f->inertia_rows < 1 ||
      f->wind_rows < 1 || f->density_rows < 1 || f->viscosity_rows < 1 ||
      p->spring_rows < 1 || p->stiffness_rows < 1 ||
      p->stiffnesspoly_rows < 1 || p->damping_rows < 1 ||
      p->dampingpoly_rows < 1 ||
      (gravity_disabled != 0 && gravity_disabled != 1))
    return cudaErrorInvalidValue;
  flash::fp32::motion_passive<<<v->worlds, flash::fp32::Threads, 0, stream>>>(
      *v, *b, *f, *p, adhesion, intervals, gravity_disabled);
  return cudaGetLastError();
}
}
