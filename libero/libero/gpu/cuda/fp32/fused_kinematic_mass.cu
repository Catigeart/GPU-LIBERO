// Copyright 2025 The Newton Developers
// Copyright 2026 Flash-LIBERO contributors
// SPDX-License-Identifier: Apache-2.0
// World-local readable kinematics/frames -> centers/inertia -> CRB/CSR mass.
#include "../dynamics/mjwarp_com_pos.cuh"
#include "../dynamics/mjwarp_kinematics.cuh"
#include "../dynamics/mjwarp_mass_matrix.cuh"
#ifndef FLASH_FP32_KINEMATIC_MASS_SOURCE_ID
#error "Build through flash_fp32_fused_kinematic_mass"
#endif
namespace flash::fp32 {
using flash::mjwarp::CamLightInputs;
using flash::mjwarp::ComPosInputs;
using flash::mjwarp::FrameAttachments;
using flash::mjwarp::KinematicsInputs;
using flash::mjwarp::MassMatrixInputs;
struct V {
  float x, y, z;
};
struct Q {
  float w, x, y, z;
};
__device__ inline V operator+(V a, V b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
__device__ inline V operator-(V a, V b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
__device__ inline V operator*(float a, V b) {
  return {a * b.x, a * b.y, a * b.z};
}
__device__ inline float dot(V a, V b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ inline V cross(V a, V b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
__device__ inline V load3(const float *p, int i) {
  p += 3 * i;
  return {p[0], p[1], p[2]};
}
__device__ inline Q load4(const float *p, int i) {
  p += 4 * i;
  return {p[0], p[1], p[2], p[3]};
}
__device__ inline void store3(float *p, int i, V v) {
  p += 3 * i;
  p[0] = v.x;
  p[1] = v.y;
  p[2] = v.z;
}
__device__ inline void store4(float *p, int i, Q q) {
  p += 4 * i;
  p[0] = q.w;
  p[1] = q.x;
  p[2] = q.y;
  p[3] = q.z;
}
__device__ inline Q normalize(Q q) {
  float length = sqrtf(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
  if (length > 0.0f) {
    float r = 1.0f / length;
    return {q.w * r, q.x * r, q.y * r, q.z * r};
  }
  // Pinned Warp storage convention, also when components are treated as wxyz.
  return {0.0f, 0.0f, 0.0f, 1.0f};
}
__device__ inline Q mul(Q a, Q b) {
  return {a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
          a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
          a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
          a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w};
}
__device__ inline V rotate(V v, Q q) {
  V u = {q.x, q.y, q.z};
  V r = 2.0f * (dot(u, v) * u) + (q.w * q.w - dot(u, u)) * v;
  return r + (2.0f * q.w) * cross(u, v);
}
__device__ inline Q axis_angle(V axis, float angle) {
  float s, c;
  sincosf(angle * 0.5f, &s, &c);
  return {c, axis.x * s, axis.y * s, axis.z * s};
}
__device__ inline void matrix(float *p, int i, Q q) {
  p += 9 * i;
  float a = q.w * q.w, b = q.w * q.x, c = q.w * q.y, d = q.w * q.z;
  float e = q.x * q.x, f = q.x * q.y, h = q.x * q.z, j = q.y * q.y,
        k = q.y * q.z, l = q.z * q.z;
  p[0] = a + e - j - l;
  p[1] = 2.0f * (f - d);
  p[2] = 2.0f * (h + c);
  p[3] = 2.0f * (f + d);
  p[4] = a - e + j - l;
  p[5] = 2.0f * (k - b);
  p[6] = 2.0f * (h - c);
  p[7] = 2.0f * (k + b);
  p[8] = a - e - j + l;
}

__device__ void rotational_axis(float *out, float x, float y, float z, float dx,
                                float dy, float dz) {
  out[0] = x;
  out[1] = y;
  out[2] = z;
  out[3] = y * dz - z * dy;
  out[4] = z * dx - x * dz;
  out[5] = x * dy - y * dx;
}

constexpr int Threads = 512;
__global__ void kinematic_mass(KinematicsInputs a, CamLightInputs frames,
                               FrameAttachments frame_ids, ComPosInputs c,
                               MassMatrixInputs m, const int *interval_size) {
  const int w = blockIdx.x;
  __shared__ float scratch[10 * 64];
  // Joint-local deltas depend only on current qpos and local model inputs.
  // Compute them once instead of repeating sin/cos and quaternion normalization
  // along every leaf path. They share dead CRB storage, disjoint from weighted
  // centers while branch producers run; no values survive this invocation.
  float *joint_delta = scratch + 3 * a.bodies;
  for (int j = threadIdx.x; j < a.joints; j += Threads) {
    const int kind = a.kind[j], q = w * a.nq + a.qadr[j];
    if (kind == 0) {
      store4(joint_delta, j,
             normalize(
                 {a.qpos[q + 3], a.qpos[q + 4], a.qpos[q + 5], a.qpos[q + 6]}));
    } else if (kind == 1) {
      store4(
          joint_delta, j,
          normalize({a.qpos[q], a.qpos[q + 1], a.qpos[q + 2], a.qpos[q + 3]}));
    } else {
      const float delta =
          a.qpos[q] - a.qpos0[(w % a.qpos0_rows) * a.nq + a.qadr[j]];
      if (kind == 2)
        joint_delta[4 * j] = delta;
      else
        store4(
            joint_delta, j,
            axis_angle(load3(a.jnt_axis, (w % a.jnt_axis_rows) * a.joints + j),
                       delta));
    }
  }
  __syncthreads(); // all branch paths may now read their joint-local deltas
  // Private branch prefixes and canonical publication are still world-local.
  // Every warp completes its owned derived frames before the CoM boundary.
  // One resident warp per branch; wider trees use the accepted components.
  // Serializing extra leaf paths in a warp lost parallelism in wide fixtures.
  const int branch = threadIdx.x / 32;
  if (branch < a.branches) {
    const int lane = threadIdx.x & 31;
    if (lane == 0) {
      // Complete root-to-leaf paths were validated cold. Read only world pose;
      // carry every parent pose privately instead of reading another CTA's
      // output.
      V parent_pos = load3(a.xpos, w * a.bodies);
      Q parent_quat = load4(a.xquat, w * a.bodies);
      for (int at = a.branch_start[branch]; at < a.branch_start[branch + 1];
           ++at) {
        int body = a.branch_body[at], first = a.jadr[body],
            count = a.jnum[body];
        bool publish = a.owner[body] == branch;
        V pos;
        Q quat;
        if (count == 1 && a.kind[first] == 0) {
          int q = w * a.nq + a.qadr[first];
          pos = {a.qpos[q], a.qpos[q + 1], a.qpos[q + 2]};
          quat = load4(joint_delta, first);
          if (publish) {
            store3(a.anchor, w * a.joints + first, pos);
            store3(a.axis, w * a.joints + first,
                   load3(a.jnt_axis, (w % a.jnt_axis_rows) * a.joints + first));
          }
        } else {
          int mocap = a.mocap[body];
          if (mocap >= 0) {
            pos = load3(a.mocap_pos, w * a.mocaps + mocap);
            quat = load4(a.mocap_quat, w * a.mocaps + mocap);
          } else {
            pos = load3(a.body_pos, (w % a.body_pos_rows) * a.bodies + body);
            quat = load4(a.body_quat, (w % a.body_quat_rows) * a.bodies + body);
          }
          pos = rotate(pos, parent_quat) + parent_pos;
          quat = mul(parent_quat, quat);
          for (int j = first; j < first + count; ++j) {
            int kind = a.kind[j];
            V local_pos = load3(a.jnt_pos, (w % a.jnt_pos_rows) * a.joints + j);
            V local_axis =
                load3(a.jnt_axis, (w % a.jnt_axis_rows) * a.joints + j);
            V anchor = rotate(local_pos, quat) + pos,
              axis = rotate(local_axis, quat);
            if (kind == 1) {
              quat = mul(quat, load4(joint_delta, j));
              pos = anchor - rotate(local_pos, quat);
            } else if (kind == 2) {
              float displacement = joint_delta[4 * j];
              pos = pos + displacement * axis;
            } else if (kind == 3) {
              quat = mul(quat, load4(joint_delta, j));
              pos = anchor - rotate(local_pos, quat);
            }
            if (publish) {
              store3(a.anchor, w * a.joints + j, anchor);
              store3(a.axis, w * a.joints + j, axis);
            }
          }
          quat = normalize(quat);
        }
        parent_pos = pos;
        parent_quat = quat;
        if (publish) {
          store3(a.xpos, w * a.bodies + body, pos);
          store4(a.xquat, w * a.bodies + body, quat);
        }
      }
    }
    __syncwarp(); // every owned pose is ready, without waiting for other
                  // branches
    const int start = a.branch_start[branch];
    for (int at = start - (branch == 0) + lane; at < a.branch_start[branch + 1];
         at += 32) {
      const int b = at < start ? 0 : a.branch_body[at], t = w * a.bodies + b;
      if (b && a.owner[b] != branch)
        continue;
      Q q = load4(a.xquat, t);
      matrix(a.xmat, t, q);
      const V inertial_position =
          load3(a.xpos, t) +
          rotate(load3(a.ipos, (w % a.ipos_rows) * a.bodies + b), q);
      store3(a.xipos, t, inertial_position);
      // Unique canonical body producers feed the existing center arena;
      // no later frame reload or separate producer barrier is needed.
      const float mass = c.mass[(w % c.mass_rows) * c.bodies + b];
      scratch[3 * b] = inertial_position.x * mass;
      scratch[3 * b + 1] = inertial_position.y * mass;
      scratch[3 * b + 2] = inertial_position.z * mass;
      matrix(a.ximat, t,
             mul(q, load4(a.iquat, (w % a.iquat_rows) * a.bodies + b)));
    }

    // One canonical-branch queue for geom/site/camera/light jobs. All four
    // share the pose transform; only lights replace the orientation matrix
    // with their accepted normalized direction. No additional barrier.
    for (int at = frame_ids.start[branch] + lane;
         at < frame_ids.start[branch + 1]; at += 32) {
      const int job = frame_ids.job[at];
      int item, body, count, position_rows, quaternion_rows = 1;
      const float *local_position, *local_quaternion = nullptr;
      float *world_position, *world_matrix = nullptr;
      const bool light = job >= a.geoms + a.sites + frames.cameras;
      if (job < a.geoms) {
        item = job;
        body = a.geom_body[item];
        count = a.geoms;
        if (a.weld[body] == 0 && a.mocap[a.root[body]] == -1)
          continue;
        position_rows = a.gpos_rows;
        quaternion_rows = a.gquat_rows;
        local_position = a.gpos;
        local_quaternion = a.gquat;
        world_position = a.geom_xpos;
        world_matrix = a.geom_xmat;
      } else if (job < a.geoms + a.sites) {
        item = job - a.geoms;
        body = a.site_body[item];
        count = a.sites;
        position_rows = a.spos_rows;
        quaternion_rows = a.squat_rows;
        local_position = a.spos;
        local_quaternion = a.squat;
        world_position = a.site_xpos;
        world_matrix = a.site_xmat;
      } else if (!light) {
        item = job - a.geoms - a.sites;
        body = frames.cam_body[item];
        count = frames.cameras;
        position_rows = frames.cam_pos_rows;
        quaternion_rows = frames.cam_quat_rows;
        local_position = frames.cam_pos;
        local_quaternion = frames.cam_quat;
        world_position = frames.cam_xpos;
        world_matrix = frames.cam_xmat;
      } else {
        item = job - a.geoms - a.sites - frames.cameras;
        body = frames.light_body[item];
        count = frames.lights;
        position_rows = frames.light_pos_rows;
        local_position = frames.light_pos;
        world_position = frames.light_xpos;
      }
      const int t = w * count + item;
      const Q q = load4(a.xquat, w * a.bodies + body);
      store3(
          world_position, t,
          load3(a.xpos, w * a.bodies + body) +
              rotate(load3(local_position, (w % position_rows) * count + item),
                     q));
      if (light) {
        V dir = rotate(
            load3(frames.light_dir, (w % frames.light_dir_rows) * count + item),
            q);
        const float length =
            sqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
        if (length > 0.f)
          dir = {dir.x / length, dir.y / length, dir.z / length};
        else
          dir = {0.f, 0.f, 0.f};
        store3(frames.light_xdir, t, dir);
      } else {
        matrix(world_matrix, t,
               mul(q, load4(local_quaternion,
                            (w % quaternion_rows) * count + item)));
      }
    }
  }

  const int lane = threadIdx.x;
  // First 3*NB weighted centers, then 10*NB composite tensors: disjoint
  // lifetimes share the same fixed allocation, independent of world count.

  float *sums = scratch;

  // Current physical weights are staged once. Immutable intervals only
  // replace tree traversal; no mass or coordinate is cached across calls.
  __syncthreads();
  for (int b = lane; b < c.bodies; b += Threads) {
    float value[3] = {};
    for (int node = b; node < b + interval_size[b]; ++node)
      for (int k = 0; k < 3; ++k)
        value[k] += sums[3 * node + k];
    const float denominator =
        c.subtreemass[(w % c.subtree_rows) * c.bodies + b];
    for (int k = 0; k < 3; ++k)
      c.subtree[3 * (w * c.bodies + b) + k] =
          denominator != 0.f ? value[k] / denominator : value[k];
  }
  __syncthreads();
  sums = c.subtree + 3 * w * c.bodies;

  // Independent inertia and joint-axis jobs share the world work queue.
  for (int work = lane; work < c.bodies + c.joints; work += Threads) {
    if (work < c.bodies) {
      const int b = work;
      const int t = w * c.bodies + b;
      const float *r = c.ximat + 9 * t;
      const float *inertia =
          c.inertia + 3 * ((w % c.inertia_rows) * c.bodies + b);
      const float *com = sums + 3 * c.root[b];
      float mass = c.mass[(w % c.mass_rows) * c.bodies + b];
      float dx = c.xipos[3 * t] - com[0], dy = c.xipos[3 * t + 1] - com[1],
            dz = c.xipos[3 * t + 2] - com[2];
      // R * diag(I) * R^T, including world body0. Keep vec10's public layout.
      float scaled[9];
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          scaled[3 * i + j] = r[3 * i + j] * inertia[j];
      float *out = c.cinert + 10 * t;
      const int rows[6] = {0, 1, 2, 0, 0, 1}, cols[6] = {0, 1, 2, 1, 2, 2};
      for (int i = 0; i < 6; ++i) {
        int row = 3 * rows[i], col = 3 * cols[i];
        out[i] = scaled[row] * r[col] + scaled[row + 1] * r[col + 1] +
                 scaled[row + 2] * r[col + 2];
      }
      out[0] += mass * (dy * dy + dz * dz);
      out[1] += mass * (dx * dx + dz * dz);
      out[2] += mass * (dx * dx + dy * dy);
      out[3] -= mass * dx * dy;
      out[4] -= mass * dx * dz;
      out[5] -= mass * dy * dz;
      out[6] = mass * dx;
      out[7] = mass * dy;
      out[8] = mass * dz;
      out[9] = mass;
    } else {
      const int j = work - c.bodies;
      const int t = w * c.joints + j, b = c.jbody[j], kind = c.kind[j];
      float *out = c.cdof + 6 * (w * c.nv + c.dofadr[j]);
      const float *com = sums + 3 * c.root[b];
      float dx = com[0] - c.anchor[3 * t], dy = com[1] - c.anchor[3 * t + 1],
            dz = com[2] - c.anchor[3 * t + 2];
      if (kind == 0) { // FREE: world translations followed by body rotations.
        for (int axis = 0; axis < 3; ++axis)
          for (int k = 0; k < 6; ++k)
            out[6 * axis + k] = float(k == axis + 3);
        out += 18;
      }
      if (kind == 0 ||
          kind == 1) { // Columns of the local-to-global body matrix.
        const float *r = c.xmat + 9 * (w * c.bodies + b);
        for (int k = 0; k < 3; ++k)
          rotational_axis(out + 6 * k, r[k], r[3 + k], r[6 + k], dx, dy, dz);
      } else if (kind == 2) { // SLIDE: angular part is exactly zero.
        for (int k = 0; k < 3; ++k) {
          out[k] = 0.0f;
          out[k + 3] = c.axis[3 * t + k];
        }
      } else { // HINGE; cold admission proves the joint type and unique DOF
               // spans.
        rotational_axis(out, c.axis[3 * t], c.axis[3 * t + 1],
                        c.axis[3 * t + 2], dx, dy, dz);
      }
    }
  }
  __syncthreads(); // centered reads end; current cinert/cdof are ready for mass
  float *composite = scratch;

  // Each tensor component reads a bounded descendant interval. World0
  // retains its own tensor, matching the CRB definition.
  for (int work = lane; work < 10 * m.bodies; work += Threads) {
    const int b = work / 10, component = work % 10;
    float value = c.cinert[(w * m.bodies + b) * 10 + component];
    if (b)
      for (int node = b + 1; node < b + interval_size[b]; ++node)
        value += c.cinert[(w * m.bodies + node) * 10 + component];
    composite[work] = value;
  }
  __syncthreads();

  for (int slot = lane; slot < m.bodies * 10; slot += Threads)
    m.crb[10 * w * m.bodies + slot] = composite[slot];
  for (int dof = lane; dof < m.nv; dof += Threads) {
    const float *inertia = composite + m.dof_body[dof] * 10;
    const float *v = m.cdof + (w * m.nv + dof) * 6;
    // Spatial vectors are (angular, linear); vec10 stores symmetric rotational
    // inertia, first moment, and mass. Keep the pinned scalar expression order.
    float product[6] = {
        inertia[0] * v[0] + inertia[3] * v[1] + inertia[4] * v[2] -
            inertia[8] * v[4] + inertia[7] * v[5],
        inertia[3] * v[0] + inertia[1] * v[1] + inertia[5] * v[2] +
            inertia[8] * v[3] - inertia[6] * v[5],
        inertia[4] * v[0] + inertia[5] * v[1] + inertia[2] * v[2] -
            inertia[7] * v[3] + inertia[6] * v[4],
        inertia[8] * v[1] - inertia[7] * v[2] + inertia[9] * v[3],
        inertia[6] * v[2] - inertia[8] * v[0] + inertia[9] * v[4],
        inertia[7] * v[0] - inertia[6] * v[1] + inertia[9] * v[5]};
    int at = m.matrix_row[dof] + m.matrix_nnz[dof] - 1;
    float *matrix = m.matrix + w * m.matrix_width;
    const float armature = m.armature[(w % m.armature_rows) * m.nv + dof];
    for (int ancestor = dof; ancestor >= 0;
         ancestor = m.dof_parent[ancestor], --at) {
      const float *motion = m.cdof + (w * m.nv + ancestor) * 6;
      float value = 0.0f;
      for (int k = 0; k < 6; ++k)
        value += motion[k] * product[k];
      matrix[at] = ancestor == dof ? armature + value : value;
    }
  }
}
} // namespace flash::fp32
extern "C" {
const char *flash_fp32_kinematic_mass_source_id() {
  return FLASH_FP32_KINEMATIC_MASS_SOURCE_ID;
}
int flash_fp32_kinematic_mass_arch() { return FLASH_CUDA_ARCH; }
std::size_t flash_fp32_kinematic_mass_kinematics_bytes() {
  return sizeof(flash::mjwarp::KinematicsInputs);
}
std::size_t flash_fp32_kinematic_mass_frames_bytes() {
  return sizeof(flash::mjwarp::CamLightInputs);
}
std::size_t flash_fp32_kinematic_mass_frame_ids_bytes() {
  return sizeof(flash::mjwarp::FrameAttachments);
}
std::size_t flash_fp32_kinematic_mass_centered_bytes() {
  return sizeof(flash::mjwarp::ComPosInputs);
}
std::size_t flash_fp32_kinematic_mass_mass_bytes() {
  return sizeof(flash::mjwarp::MassMatrixInputs);
}
int flash_fp32_kinematic_mass_launch(
    const flash::mjwarp::KinematicsInputs *a,
    const flash::mjwarp::CamLightInputs *frames,
    const flash::mjwarp::FrameAttachments *frame_ids,
    const flash::mjwarp::ComPosInputs *c,
    const flash::mjwarp::MassMatrixInputs *m, const int *intervals,
    cudaStream_t stream) {
  if (!a || !frames || !frame_ids || !c || !m || !intervals || a->worlds < 1 ||
      a->bodies < 2 || a->bodies > 64 || a->branches < 1 ||
      a->branches > flash::fp32::Threads / 32 || a->joints < 1 ||
      a->joints > (640 - 3 * a->bodies) / 4 || c->nv < 1 ||
      a->worlds != c->worlds || a->worlds != m->worlds ||
      a->worlds != frames->worlds || a->bodies != c->bodies ||
      a->bodies != m->bodies || a->bodies != frames->bodies || c->nv != m->nv ||
      a->joints != c->joints)
    return int(cudaErrorInvalidValue);
  flash::fp32::kinematic_mass<<<a->worlds, flash::fp32::Threads, 0, stream>>>(
      *a, *frames, *frame_ids, *c, *m, intervals);
  return int(cudaGetLastError());
}
}
