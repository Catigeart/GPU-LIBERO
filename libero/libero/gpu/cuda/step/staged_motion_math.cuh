// Copyright 2025 The Newton Developers
// Copyright 2026 Flash-LIBERO contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Included after substep_forward_math.cuh and substep_forward_prepare.cuh.
// FP32 fused_motion_passive.cu supplies the branch-prefix organization. All
// arithmetic and exceptional publication below retain the native FP64 owners.
namespace flash_libero {
struct BranchMotion {
  const int *plan;
  __device__ void operator()(const BodyVelocityData &in, int w) const {
    using namespace physics_velocity;
    const int nb = in.nb, nv = in.nv;
    const real_t *a = in.cdof + size_t(w) * nv * 6;
    const real_t *q = in.qvel + size_t(w) * nv;
    const real_t *inertia = in.cinert + size_t(w) * nb * 10;
    if (in.active[w] != 1 || !d1_finite_block(a, nv * 6, q, nv, inertia, nb * 10)) {
      NativeMotion{}(in, w);
      return;
    }
    const int branches = plan[0];
    const int *starts = plan + 1, *bodies = starts + branches + 1;
    const int *owner = bodies + starts[branches];
    const int component = threadIdx.x & 7;
    const unsigned mask = 0xffu << (threadIdx.x & 24);
    real_t *velocity = in.cvel + size_t(w) * nb * 6;
    real_t *dot = in.cdot + size_t(w) * nv * 6;
    real_t *acc = in.cache_acc + size_t(w) * nb * 6;
    real_t *force = in.cache_force + size_t(w) * nb * 6;
    if (threadIdx.x == 0) {
      in.status[w] = 0;
      in.cache_valid[w] = 0;
      in.cache_status[w] = 0;
    }
    // Every non-world body/DOF is written once by its canonical branch. Only
    // the world rows need initialization; there is no shared spatial arena.
    if (threadIdx.x < 6) {
      velocity[threadIdx.x] = 0;
      force[threadIdx.x * nb] = 0;
      acc[threadIdx.x * nb] = threadIdx.x >= 3 && !in.disabled
          ? -in.gravity[threadIdx.x - 3] : REAL_C(0.0);
    }
    for (int branch = threadIdx.x / 8; branch < branches; branch += blockDim.x / 8) {
      real_t v = 0, acceleration = component >= 3 && component < 6 && !in.disabled
          ? -in.gravity[component - 3] : REAL_C(0.0);
      for (int at = starts[branch]; at < starts[branch + 1]; ++at) {
        const int b = bodies[at];
        const bool publish = owner[b] == branch;
        real_t acc_delta = 0;
        int j = 0;
        while (j < in.bn[b]) {
          int dof = in.ba[b] + j;
          const int type = in.jt[in.dj[dof]];
          if (type == 0) {
            real_t translation = 0;
            for (int k = 0; k < 3; ++k) {
              if (component < 6) {
                if (q[dof + k] != 0) translation += a[6 * (dof + k) + component] * q[dof + k];
                if (publish) dot[6 * (dof + k) + component] = 0;
              }
            }
            v += translation;
            j += 3;
            dof += 3;
          }
          const int width = type <= 1 ? 3 : 1;
          const int axis = component % 3, i = (axis + 1) % 3, k = (axis + 2) % 3;
          const real_t vi = __shfl_sync(mask, v, i, 8), vk = __shfl_sync(mask, v, k, 8);
          const real_t li = __shfl_sync(mask, v, i + 3, 8), lk = __shfl_sync(mask, v, k + 3, 8);
          real_t delta = 0;
          for (int t = 0; t < width; ++t) {
            const real_t *motion = a + 6 * (dof + t);
            if (component < 6) {
              real_t derivative = vi * motion[k] - vk * motion[i];
              if (component >= 3)
                derivative = (li * motion[k] - lk * motion[i]) +
                             (vi * motion[k + 3] - vk * motion[i + 3]);
              if (publish) dot[6 * (dof + t) + component] = derivative;
              // CPU mj_comVel groups the three ball/free axis products before
              // adding them to velocity. RNE adds one body delta to its parent.
              if (width == 1 || q[dof + t] != 0)
                delta += motion[component] * q[dof + t];
              if (in.bn[b] == 1 || q[dof + t] != 0)
                acc_delta += derivative * q[dof + t];
            }
          }
          v += delta;
          j += width;
        }
        acceleration += acc_delta;
        if (publish && component < 6) {
          velocity[6 * b + component] = v;
          acc[component * nb + b] = acceleration;
        }
      }
    }
    __syncthreads(); // raw inertia force consumes all canonical velocity rows
    for (int b = 1 + threadIdx.x; b < nb; b += blockDim.x) {
      real_t local_acc[6], fi[6], momentum[6], correction[6];
      for (int k = 0; k < 6; ++k) local_acc[k] = acc[k * nb + b];
      d_inertia(inertia + 10 * b, local_acc, fi);
      d_inertia(inertia + 10 * b, velocity + 6 * b, momentum);
      d_cross_force(velocity + 6 * b, momentum, correction);
      for (int k = 0; k < 6; ++k) force[k * nb + b] = fi[k] + correction[k];
    }
    __syncthreads();
    if (!d1_finite_block(velocity, nb * 6, dot, nv * 6)) {
      if (threadIdx.x == 0) in.status[w] = 3;
      d1_zero_block(velocity, nb * 6); d1_zero_block(dot, nv * 6);
      d1_zero_block(acc, nb * 6); d1_zero_block(force, nb * 6);
      __syncthreads();
      return;
    }
    if (!d1_finite_block(acc, nb * 6, force, nb * 6)) {
      if (threadIdx.x == 0) in.cache_status[w] = 3;
      d1_zero_block(acc, nb * 6); d1_zero_block(force, nb * 6);
      __syncthreads();
      return;
    }
    if (threadIdx.x == 0) {
      in.cache_state[w] = in.state_epoch[w];
      in.cache_forward[w] = in.forward_epoch[w];
      in.cache_valid[w] = 1;
    }
    __syncthreads();
  }
};
} // namespace flash_libero
