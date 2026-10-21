// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Pinned MJWarp smooth.py::_cfrc_ext/_cfrc_ext_contact and support.py force helpers.
// Rigid bodies only. Equality, tendon and flex callers retain the upstream prefix.
#include "mjwarp_external_wrench.cuh"

namespace flash::mjwarp {
namespace {
struct V { float x, y, z; };
__device__ inline V load3(const float* p, int i) {
  p += 3*i;
  return {p[0], p[1], p[2]};
}
__device__ inline V operator-(V a, V b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
__device__ inline V cross(V a, V b) {
  return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}
__device__ inline V rotate_transpose(const float* r, V v) {
  return {v.x*r[0]+v.y*r[3]+v.z*r[6],
          v.x*r[1]+v.y*r[4]+v.z*r[7],
          v.x*r[2]+v.y*r[5]+v.z*r[8]};
}

// One thread owns a world/body. Input applied wrench is (force,torque), while
// RNE output is (torque,force), about the body's root subtree CoM in world axes.
__global__ void initialize_applied(ExternalWrenchInputs a) {
  int t = blockIdx.x*blockDim.x + threadIdx.x;
  if (t >= a.worlds*a.bodies) return;
  int w = t/a.bodies, b = t%a.bodies;
  float* output = a.external + 6*t;
  if (b == 0) {
    for (int k=0; k<6; ++k) output[k] = 0.0f;
    return;
  }
  const float* applied = a.applied + 6*t;
  V force = {applied[0], applied[1], applied[2]};
  V torque = {applied[3], applied[4], applied[5]};
  V offset = load3(a.subtree_com, w*a.bodies+a.root[b]) - load3(a.xipos, t);
  torque = torque - cross(offset, force);
  output[0]=torque.x; output[1]=torque.y; output[2]=torque.z;
  output[3]=force.x; output[4]=force.y; output[5]=force.z;
}

// One thread owns a contact, with the original six component FP32 atomics.
// Contacts form a shared flat list; worldid is not inferred from list position.
__global__ void accumulate_contacts(ExternalWrenchInputs a) {
  int c = blockIdx.x*blockDim.x + threadIdx.x;
  if (c >= a.contact_capacity || c >= a.contact_count[0]) return;
  int b1 = a.geom_body[a.contact_geom[2*c]], b2 = a.geom_body[a.contact_geom[2*c+1]];
  if (b1 == 0 && b2 == 0) return;
  int w = a.contact_world[c], dim = a.contact_dim[c];
  const int* address = a.contact_address + c*a.address_columns;
  float local[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  if (address[0] >= 0) {
    const float* efc = a.constraint_force + w*a.rows;
    if (a.cone == 0) {  // ConeType.PYRAMIDAL
      if (dim == 1) local[0] = efc[address[0]];
      else {
        const float* friction = a.contact_friction + 5*c;
        for (int i=0; i<dim-1; ++i) {
          int row = address[0]+2*i;
          float positive = row < a.rows ? efc[row] : 0.0f;
          float negative = row+1 < a.rows ? efc[row+1] : 0.0f;
          local[0] += positive+negative;
          local[i+1] = (positive-negative)*friction[i];
        }
      }
    } else {  // ConeType.ELLIPTIC: rows can have independent, scattered addresses.
      for (int i=0; i<dim; ++i)
        if (address[i] < a.rows) local[i] = efc[address[i]];
    }
    // Pinned MJWarp reports the net interface force, including adhesive pull.
    // An excluded contact (address[0]<0) contributes no adhesion either.
    local[0] -= a.contact_adhesion[c];
  }
  const float* frame = a.contact_frame+9*c;
  V force = rotate_transpose(frame, {local[0], local[1], local[2]});
  V torque = rotate_transpose(frame, {local[3], local[4], local[5]});
  V position = load3(a.contact_pos, c);
  for (int side=0; side<2; ++side) {
    int b = side == 0 ? b1 : b2;
    if (b == 0) continue;
    V offset = load3(a.subtree_com, w*a.bodies+a.root[b]) - position;
    V shifted_torque = torque - cross(offset, force);
    float sign = side == 0 ? -1.0f : 1.0f;
    float* output = a.external+6*(w*a.bodies+b);
    atomicAdd(output+0, sign*shifted_torque.x);
    atomicAdd(output+1, sign*shifted_torque.y);
    atomicAdd(output+2, sign*shifted_torque.z);
    atomicAdd(output+3, sign*force.x);
    atomicAdd(output+4, sign*force.y);
    atomicAdd(output+5, sign*force.z);
  }
}
}
}

extern "C" {
const char* flash_mjwarp_external_wrench_source_id() { return FLASH_MJW_EXTERNAL_WRENCH_SOURCE_ID; }
std::size_t flash_mjwarp_external_wrench_arguments_bytes() { return sizeof(flash::mjwarp::ExternalWrenchInputs); }
int flash_mjwarp_external_wrench_arch() { return FLASH_MJW_EXTERNAL_WRENCH_ARCH; }
int flash_mjwarp_external_wrench_launch(const flash::mjwarp::ExternalWrenchInputs* a, int phase, cudaStream_t stream) {
  if (!a || a->worlds<1 || a->bodies<1 || a->geoms<0 || a->contact_capacity<0 ||
      a->rows<0 || a->address_columns<1 || (a->cone!=0 && a->cone!=1)) return int(cudaErrorInvalidValue);
  using namespace flash::mjwarp;
  if (phase==0) initialize_applied<<<(a->worlds*a->bodies+255)/256,256,0,stream>>>(*a);
  else if (phase==1) {
    if (a->contact_capacity) accumulate_contacts<<<(a->contact_capacity+255)/256,256,0,stream>>>(*a);
  } else return int(cudaErrorInvalidValue);
  return int(cudaGetLastError());
}
}
