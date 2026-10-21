// Copyright 2025 The Newton Developers
// Copyright 2026 Flash-LIBERO contributors
// SPDX-License-Identifier: Apache-2.0
// World-local centered frames and mass assembly, following accepted MJWarp
// math.
#include "../dynamics/mjwarp_com_pos.cuh"
#include "../dynamics/mjwarp_mass_matrix.cuh"
#ifndef FLASH_FP32_POSITION_SOURCE_ID
#error "Build through flash_fp32_fused_position"
#endif
namespace flash::fp32 {
using flash::mjwarp::ComPosInputs;
using flash::mjwarp::MassMatrixInputs;
__device__ void rotational_axis(float *out, float x, float y, float z, float dx,
                                float dy, float dz) {
  out[0] = x;
  out[1] = y;
  out[2] = z;
  out[3] = y * dz - z * dy;
  out[4] = z * dx - x * dz;
  out[5] = x * dy - y * dx;
}

template <int Threads, bool Shared, bool Direct>
__global__ void centered_mass(ComPosInputs c, MassMatrixInputs m,
                              const int *bodies, const int *starts, int levels,
                              const int *child_start, const int *child_id,
                              const int *interval_size) {
  const int w = blockIdx.x, lane = threadIdx.x;
  // First 3*NB weighted centers, then 10*NB composite tensors: disjoint
  // lifetimes share the same fixed allocation, independent of world count.
  __shared__ float scratch[Shared ? 10 * 64 : 1];
  float *sums = Shared ? scratch : c.subtree + 3 * w * c.bodies;
  if constexpr (Direct) {
    // Current physical weights are staged once. Immutable intervals only
    // replace tree traversal; no mass or coordinate is cached across calls.
    for (int slot = lane; slot < 3 * c.bodies; slot += Threads) {
      const int b = slot / 3;
      sums[slot] = c.xipos[3 * w * c.bodies + slot] *
                   c.mass[(w % c.mass_rows) * c.bodies + b];
    }
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
  } else {
    // Gather each body's children after their depth is complete. Each sum has
    // one writer; no atomic lock or initialization pass is required.
    for (int level = levels - 1; level >= 0; --level) {
      for (int at = starts[level] + lane; at < starts[level + 1];
           at += Threads) {
        const int b = bodies[at], t = w * c.bodies + b;
        const float mass = c.mass[(w % c.mass_rows) * c.bodies + b];
        float value[3];
        for (int k = 0; k < 3; ++k)
          value[k] = c.xipos[3 * t + k] * mass;
        for (int edge = child_start[b]; edge < child_start[b + 1]; ++edge) {
          const int child = child_id[edge];
          for (int k = 0; k < 3; ++k)
            value[k] += sums[3 * child + k];
        }
        for (int k = 0; k < 3; ++k)
          sums[3 * b + k] = value[k];
      }
      __syncthreads();
    }
    for (int b = lane; b < c.bodies; b += Threads) {
      const float mass = c.subtreemass[(w % c.subtree_rows) * c.bodies + b];
      if (mass != 0.f)
        for (int k = 0; k < 3; ++k)
          sums[3 * b + k] /= mass;
    }
    if constexpr (Shared)
      for (int b = lane; b < c.bodies; b += Threads)
        for (int k = 0; k < 3; ++k)
          c.subtree[3 * (w * c.bodies + b) + k] = sums[3 * b + k];
    __syncthreads(); // normalized roots are ready for inertia and joint
                     // consumers
  }
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
  float *composite = Shared ? scratch : m.crb + 10 * w * m.bodies;
  if constexpr (Direct) {
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
  } else {
    // World composite inertia is held. Every other body gathers its own
    // children at the previous completed depth, with one writer per tensor.
    for (int k = lane; k < 10; k += Threads)
      composite[k] = c.cinert[w * m.bodies * 10 + k];
    if (levels == 1)
      __syncthreads(); // root-only models have no depth barrier
    for (int level = levels - 1; level > 0; --level) {
      const int count = starts[level + 1] - starts[level];
      for (int work = lane; work < 10 * count; work += Threads) {
        const int b = bodies[starts[level] + work / 10], component = work % 10;
        float value = c.cinert[(w * m.bodies + b) * 10 + component];
        for (int edge = child_start[b]; edge < child_start[b + 1]; ++edge)
          value += composite[10 * child_id[edge] + component];
        composite[10 * b + component] = value;
      }
      __syncthreads();
    }
  }
  if constexpr (Shared)
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
const char *flash_fp32_position_source_id() {
  return FLASH_FP32_POSITION_SOURCE_ID;
}
int flash_fp32_position_arch() { return FLASH_CUDA_ARCH; }
std::size_t flash_fp32_position_centered_bytes() {
  return sizeof(flash::mjwarp::ComPosInputs);
}
std::size_t flash_fp32_position_mass_bytes() {
  return sizeof(flash::mjwarp::MassMatrixInputs);
}
int flash_fp32_position_launch(const flash::mjwarp::ComPosInputs *c,
                               const flash::mjwarp::MassMatrixInputs *m,
                               const int *bodies, const int *starts, int levels,
                               const int *child_start, const int *child_id,
                               const int *interval_size, cudaStream_t stream) {
  if (!c || !m || !bodies || !starts || !child_start || !child_id ||
      c->worlds < 1 || c->bodies < 2 || c->nv < 1 || c->worlds != m->worlds ||
      c->bodies != m->bodies || c->nv != m->nv || levels < 2 ||
      levels > c->bodies)
    return cudaErrorInvalidValue;
  using namespace flash::fp32;
  if (interval_size && c->bodies <= 64)
    centered_mass<128, true, true><<<c->worlds, 128, 0, stream>>>(
        *c, *m, bodies, starts, levels, child_start, child_id, interval_size);
  else if (c->bodies <= 64)
    centered_mass<128, true, false><<<c->worlds, 128, 0, stream>>>(
        *c, *m, bodies, starts, levels, child_start, child_id, interval_size);
  else
    centered_mass<128, false, false><<<c->worlds, 128, 0, stream>>>(
        *c, *m, bodies, starts, levels, child_start, child_id, interval_size);
  return cudaGetLastError();
}
}
