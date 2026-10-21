// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Pinned MJWarp smooth.py::rne(flg_acc=False) and its five numerical phases.
#include "mjwarp_bias.cuh"
#include "mjwarp_spatial.cuh"
#ifndef FLASH_MJW_BIAS_SOURCE_ID
#error "Build through flash_mjwarp_bias_fp32"
#endif
namespace flash::mjwarp {
namespace {
__global__ void acceleration_root(BiasInputs a) {
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= a.worlds)
    return;
  float *acc = a.cacc + w * a.bodies * 6;
  for (int k = 0; k < 3; ++k) {
    acc[k] = 0.0f;
    acc[k + 3] = -a.gravity[(w % a.gravity_rows) * 3 + k];
  }
}

__global__ void acceleration_branch(BiasInputs a) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.worlds * a.branches)
    return;
  int w = t / a.branches, branch = t % a.branches;
  float acc[6];
  for (int k = 0; k < 6; ++k)
    acc[k] = a.cacc[w * a.bodies * 6 + k];
  for (int at = a.branch_start[branch]; at < a.branch_start[branch + 1]; ++at) {
    int b = a.branch_bodies[at], first = a.body_dofadr[b];
    for (int j = 0; j < a.body_dofnum[b]; ++j)
      for (int k = 0; k < 6; ++k)
        acc[k] += a.cdof_dot[(w * a.nv + first + j) * 6 + k] *
                  a.qvel[w * a.nv + first + j];
    if (a.body_owner[b] == branch)
      for (int k = 0; k < 6; ++k)
        a.cacc[(w * a.bodies + b) * 6 + k] = acc[k];
  }
}

__global__ void body_force(BiasInputs a) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.worlds * a.bodies)
    return;
  float *out = a.cfrc_int + t * 6;
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
    out[k] = force[k] + cross[k];
}

__global__ void force_backward(BiasInputs a, const int *layer, int count) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.worlds * count)
    return;
  int w = t / count, b = layer[t % count];
  if (b == 0)
    return;
  // Unlike CRB, pinned RNE also publishes root forces into world (parent=0).
  const float *child = a.cfrc_int + (w * a.bodies + b) * 6;
  float *parent = a.cfrc_int + (w * a.bodies + a.body_parent[b]) * 6;
  for (int k = 0; k < 6; ++k)
    atomicAdd(parent + k, child[k]);
}

__global__ void project_bias(BiasInputs a) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.worlds * a.nv)
    return;
  int w = t / a.nv, dof = t % a.nv;
  const float *force = a.cfrc_int + (w * a.bodies + a.dof_body[dof]) * 6;
  float value = 0.0f;
  for (int k = 0; k < 6; ++k)
    value += a.cdof[t * 6 + k] * force[k];
  a.qfrc_bias[t] = value;
}
// A complete ordinary bias RNE within one world. The root and every branch
// build independent acceleration prefixes before the body-force consumers.
template <int Threads, bool Shared, bool Direct, bool BranchForces>
__global__ void world_bias(BiasInputs a, const int *bodies, const int *starts,
                           int levels, const int *child_start,
                           const int *child_id, const int *interval_size,
                           int gravity_disabled) {
  const int w = blockIdx.x, lane = threadIdx.x;
  const float *__restrict__ qvel = a.qvel;
  const float *__restrict__ cdof = a.cdof;
  const float *__restrict__ cdof_dot = a.cdof_dot;
  const float *__restrict__ cinert = a.cinert;
  const float *__restrict__ cvel = a.cvel;
  const float *__restrict__ gravity = a.gravity;
  float *__restrict__ cacc = a.cacc;
  float *__restrict__ cfrc_int = a.cfrc_int;
  float *__restrict__ qfrc_bias = a.qfrc_bias;
  __shared__ float scratch[Shared ? 6 * 64 : 1];
  float *force_tree = Shared ? scratch : cfrc_int + w * a.bodies * 6;
  if (lane < 6)
    cacc[w * a.bodies * 6 + lane] =
        lane < 3 || gravity_disabled
            ? 0.f
            : -gravity[3 * (w % a.gravity_rows) + lane - 3];
  if constexpr (!BranchForces) {
    if (a.branches <= 32 && levels >= 8) {
      // Independent spatial coordinates cooperate on the same private prefix.
      // There are no cross-component reads and no subgroup synchronization.
      const int component = lane & 7;
      if (component < 6)
        for (int branch = lane / 8; branch < a.branches;
             branch += Threads / 8) {
          float acc = component < 3 || gravity_disabled
                          ? 0.f
                          : -gravity[3 * (w % a.gravity_rows) + component - 3];
          for (int at = a.branch_start[branch]; at < a.branch_start[branch + 1];
               ++at) {
            const int b = a.branch_bodies[at], first = a.body_dofadr[b];
            for (int j = 0; j < a.body_dofnum[b]; ++j)
              acc += cdof_dot[6 * (w * a.nv + first + j) + component] *
                     qvel[w * a.nv + first + j];
            if (a.body_owner[b] == branch)
              cacc[6 * (w * a.bodies + b) + component] = acc;
          }
        }
    } else {
      for (int branch = lane; branch < a.branches; branch += Threads) {
        float acc[6] = {};
        if (!gravity_disabled)
          for (int k = 0; k < 3; ++k)
            acc[k + 3] = -gravity[3 * (w % a.gravity_rows) + k];
        for (int at = a.branch_start[branch]; at < a.branch_start[branch + 1];
             ++at) {
          const int b = a.branch_bodies[at], first = a.body_dofadr[b];
          for (int j = 0; j < a.body_dofnum[b]; ++j)
            for (int k = 0; k < 6; ++k)
              acc[k] += cdof_dot[6 * (w * a.nv + first + j) + k] *
                        qvel[w * a.nv + first + j];
          if (a.body_owner[b] == branch)
            for (int k = 0; k < 6; ++k)
              cacc[6 * (w * a.bodies + b) + k] = acc[k];
        }
      }
    }
    __syncthreads(); // canonical current accelerations are ready for each body
  }
  // A canonical warp produces and consumes its own body's accelerations.
  // Deferred owned-body jobs preserve force parallelism across32lanes rather
  // than serializing force algebra at every prefix node.
  const int cohort = BranchForces ? lane / 32 : 0,
            worker = BranchForces ? lane % 32 : lane;
  for (int branch = cohort; branch < (BranchForces ? a.branches : 1);
       branch += (BranchForces ? Threads / 32 : 1)) {
    if constexpr (BranchForces) {
      if (worker < 6) {
        float acc = worker < 3 || gravity_disabled
                        ? 0.f
                        : -gravity[3 * (w % a.gravity_rows) + worker - 3];
        for (int at = a.branch_start[branch]; at < a.branch_start[branch + 1];
             ++at) {
          const int b = a.branch_bodies[at], first = a.body_dofadr[b];
          for (int j = 0; j < a.body_dofnum[b]; ++j)
            acc += cdof_dot[6 * (w * a.nv + first + j) + worker] *
                   qvel[w * a.nv + first + j];
          if (a.body_owner[b] == branch) {
            cacc[6 * (w * a.bodies + b) + worker] = acc;
            force_tree[6 * b + worker] = acc;
          }
        }
      }
      __syncwarp(); // all32lanes consume only their own warp's produced rows
    }
    const int first = BranchForces ? a.branch_start[branch] : 0;
    const int end = BranchForces ? a.branch_start[branch + 1] : a.bodies;
    for (int at = (BranchForces ? first - (branch == 0) : 0) + worker; at < end;
         at += (BranchForces ? 32 : Threads)) {
      const int b = BranchForces ? (at < first ? 0 : a.branch_bodies[at]) : at;
      if constexpr (BranchForces)
        if (b && a.body_owner[b] != branch)
          continue;
      if (b == 0) {
        for (int k = 0; k < 6; ++k)
          force_tree[k] = 0.f;
        continue;
      }
      const int t = w * a.bodies + b;
      const float *inertia = cinert + 10 * t,
                  *acceleration =
                      BranchForces ? force_tree + 6 * b : cacc + 6 * t,
                  *velocity = cvel + 6 * t;
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
        force_tree[6 * b + k] = force[k] + correction[k];
    }
  }
  __syncthreads(); // input body forces are complete before child gathering
  if constexpr (Direct) {
    // Immutable leaf forces remain in shared storage while independent body
    // components publish bounded subtree sums. World0 includes every root.
    for (int work = lane; work < 6 * a.bodies; work += Threads) {
      const int b = work / 6, component = work % 6;
      float value = force_tree[work];
      for (int node = b + 1; node < b + interval_size[b]; ++node)
        value += force_tree[6 * node + component];
      cfrc_int[6 * w * a.bodies + work] = value;
    }
    __syncthreads(); // all current composite outputs are ready for projection
    force_tree = cfrc_int + 6 * w * a.bodies;
  } else {
    // One writer per tensor component, including the public world-force sum.
    for (int level = levels - 2; level >= 0; --level) {
      const int count = starts[level + 1] - starts[level];
      const int only = bodies[starts[level]];
      if (count == 1 && child_start[only + 1] - child_start[only] >= 64) {
        // A broad parent needs parallel child reads, not six serial collectors.
        // Whole warps own components, using registers and masked reductions.
        const int warp = lane / 32, sub = lane & 31;
        for (int component = warp; component < 6; component += Threads / 32) {
          float value = 0.f;
          for (int edge = child_start[only] + sub; edge < child_start[only + 1];
               edge += 32)
            value += force_tree[6 * child_id[edge] + component];
          for (int offset = 16; offset; offset /= 2)
            value += __shfl_down_sync(0xffffffffu, value, offset);
          if (sub == 0)
            force_tree[6 * only + component] += value;
        }
      } else {
        for (int work = lane; work < 6 * count; work += Threads) {
          const int b = bodies[starts[level] + work / 6], component = work % 6;
          float value = force_tree[6 * b + component];
          for (int edge = child_start[b]; edge < child_start[b + 1]; ++edge)
            value += force_tree[6 * child_id[edge] + component];
          force_tree[6 * b + component] = value;
        }
      }
      __syncthreads();
    }
    if constexpr (Shared)
      for (int slot = lane; slot < 6 * a.bodies; slot += Threads)
        cfrc_int[6 * w * a.bodies + slot] = force_tree[slot];
  }
  for (int dof = lane; dof < a.nv; dof += Threads) {
    const float *force = force_tree + 6 * a.dof_body[dof],
                *motion = cdof + 6 * (w * a.nv + dof);
    float value = 0.f;
    for (int k = 0; k < 6; ++k)
      value += motion[k] * force[k];
    qfrc_bias[w * a.nv + dof] = value;
  }
}

} // namespace
} // namespace flash::mjwarp
extern "C" {
const char *flash_mjwarp_bias_source_id() { return FLASH_MJW_BIAS_SOURCE_ID; }
std::size_t flash_mjwarp_bias_arguments_bytes() {
  return sizeof(flash::mjwarp::BiasInputs);
}
int flash_mjwarp_bias_arch() { return FLASH_MJW_BIAS_ARCH; }
int flash_mjwarp_bias_launch(const flash::mjwarp::BiasInputs *a, int phase,
                             const int *layer, int count, cudaStream_t stream) {
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
  } else if (phase == 4)
    project_bias<<<(a->worlds * a->nv + 255) / 256, 256, 0, stream>>>(*a);
  else
    return int(cudaErrorInvalidValue);
  return int(cudaGetLastError());
}
int flash_mjwarp_bias_fused(const flash::mjwarp::BiasInputs *a,
                            const int *bodies, const int *starts, int levels,
                            const int *child_start, const int *child_id,
                            const int *interval_size, int gravity_disabled,
                            cudaStream_t stream) {
  if (!a || !bodies || !starts || !child_start || !child_id || a->worlds < 1 ||
      a->bodies < 2 || a->nv < 1 || a->branches < 1 || a->gravity_rows < 1 ||
      levels < 2 || levels > a->bodies ||
      (gravity_disabled != 0 && gravity_disabled != 1))
    return cudaErrorInvalidValue;
  if (a->bodies <= 64 && interval_size && a->branches <= 32 && levels >= 8)
    flash::mjwarp::world_bias<512, true, true, true>
        <<<a->worlds, 512, 0, stream>>>(*a, bodies, starts, levels, child_start,
                                        child_id, interval_size,
                                        gravity_disabled);
  else if (a->bodies <= 64 && interval_size)
    flash::mjwarp::world_bias<128, true, true, false>
        <<<a->worlds, 128, 0, stream>>>(*a, bodies, starts, levels, child_start,
                                        child_id, interval_size,
                                        gravity_disabled);
  else if (a->bodies <= 64)
    flash::mjwarp::world_bias<128, true, false, false>
        <<<a->worlds, 128, 0, stream>>>(*a, bodies, starts, levels, child_start,
                                        child_id, interval_size,
                                        gravity_disabled);
  else
    flash::mjwarp::world_bias<128, false, false, false>
        <<<a->worlds, 128, 0, stream>>>(*a, bodies, starts, levels, child_start,
                                        child_id, interval_size,
                                        gravity_disabled);
  return cudaGetLastError();
}
}
