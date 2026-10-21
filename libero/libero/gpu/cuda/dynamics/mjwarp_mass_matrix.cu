// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Readable adaptation of pinned MJWarp c384ac51df67eb92ab7657cc472ff7494dc036bf:
// smooth.py::_crb_accumulate, _M; math.py::inert_vec.
#include "mjwarp_mass_matrix.cuh"
#ifndef FLASH_MJW_MASS_MATRIX_SOURCE_ID
#error "Build through flash_mjwarp_mass_matrix_fp32"
#endif

namespace flash::mjwarp {
namespace {
// One thread per (world, body) at this depth. Separate launches order child
// reads before parent updates at the next depth; siblings use component atomics.
__global__ void accumulate(MassMatrixInputs a, const int* layer, int count) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.worlds * count) return;
  int w = t / count, body = layer[t % count], parent = a.body_parent[body];
  if (parent == 0) return;  // World inertia is held, as in upstream CRB.
  const float* child = a.crb + (w * a.bodies + body) * 10;
  float* target = a.crb + (w * a.bodies + parent) * 10;
  for (int k = 0; k < 10; ++k) atomicAdd(target + k, child[k]);
}

// Each DOF owns its entire CSR row. The diagonal is last; walking dof_parent
// visits columns in reverse order. Other entries start at zero, set by enqueue.
__global__ void assemble(MassMatrixInputs a) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.worlds * a.nv) return;
  int w = t / a.nv, dof = t % a.nv;
  const float* inertia = a.crb + (w * a.bodies + a.dof_body[dof]) * 10;
  const float* v = a.cdof + (w * a.nv + dof) * 6;
  // Spatial vectors are (angular, linear); vec10 stores symmetric rotational
  // inertia, first moment, and mass. Keep the pinned scalar expression order.
  float product[6] = {
      inertia[0]*v[0] + inertia[3]*v[1] + inertia[4]*v[2] - inertia[8]*v[4] + inertia[7]*v[5],
      inertia[3]*v[0] + inertia[1]*v[1] + inertia[5]*v[2] + inertia[8]*v[3] - inertia[6]*v[5],
      inertia[4]*v[0] + inertia[5]*v[1] + inertia[2]*v[2] - inertia[7]*v[3] + inertia[6]*v[4],
      inertia[8]*v[1] - inertia[7]*v[2] + inertia[9]*v[3],
      inertia[6]*v[2] - inertia[8]*v[0] + inertia[9]*v[4],
      inertia[7]*v[0] - inertia[6]*v[1] + inertia[9]*v[5]};
  int at = a.matrix_row[dof] + a.matrix_nnz[dof] - 1;
  float* matrix = a.matrix + w * a.matrix_width;
  matrix[at] = a.armature[(w % a.armature_rows) * a.nv + dof];
  for (int ancestor = dof; ancestor >= 0; ancestor = a.dof_parent[ancestor], --at) {
    const float* motion = a.cdof + (w * a.nv + ancestor) * 6;
    float value = 0.0f;
    for (int k = 0; k < 6; ++k) value += motion[k] * product[k];
    matrix[at] += value;
  }
}
// The copy, reverse CRB reduction and CSR assembly belong to one world.
// Global publication remains complete when this CTA returns.
template<int Threads,bool Shared>
__global__ void world_mass(MassMatrixInputs a,const float* cinert,const int* bodies,
    const int* starts,int levels,const int* child_start,const int* child_id) {
  const int w=blockIdx.x,lane=threadIdx.x;
  __shared__ float scratch[Shared ? 10*64 : 1];
  float* composite=Shared ? scratch : a.crb+10*w*a.bodies;
  // World composite inertia is held. Every other body gathers its own
  // children at the previous completed depth, with one writer per tensor.
  for (int k=lane;k<10;k+=Threads) composite[k]=cinert[w*a.bodies*10+k];
  if (levels==1) __syncthreads(); // root-only models have no depth barrier
  for (int level=levels-1;level>0;--level) {
    const int count=starts[level+1]-starts[level];
    for (int work=lane;work<10*count;work+=Threads) {
      const int b=bodies[starts[level]+work/10],component=work%10;
      float value=cinert[(w*a.bodies+b)*10+component];
      for (int edge=child_start[b];edge<child_start[b+1];++edge)
        value+=composite[10*child_id[edge]+component];
      composite[10*b+component]=value;
    }
    __syncthreads();
  }
  if constexpr (Shared) for (int slot=lane;slot<a.bodies*10;slot+=Threads)
    a.crb[10*w*a.bodies+slot]=composite[slot];
  for (int dof=lane;dof<a.nv;dof+=Threads) {
    const float* inertia = composite + a.dof_body[dof] * 10;
    const float* v = a.cdof + (w * a.nv + dof) * 6;
    // Spatial vectors are (angular, linear); vec10 stores symmetric rotational
    // inertia, first moment, and mass. Keep the pinned scalar expression order.
    float product[6] = {
        inertia[0]*v[0] + inertia[3]*v[1] + inertia[4]*v[2] - inertia[8]*v[4] + inertia[7]*v[5],
        inertia[3]*v[0] + inertia[1]*v[1] + inertia[5]*v[2] + inertia[8]*v[3] - inertia[6]*v[5],
        inertia[4]*v[0] + inertia[5]*v[1] + inertia[2]*v[2] - inertia[7]*v[3] + inertia[6]*v[4],
        inertia[8]*v[1] - inertia[7]*v[2] + inertia[9]*v[3],
        inertia[6]*v[2] - inertia[8]*v[0] + inertia[9]*v[4],
        inertia[7]*v[0] - inertia[6]*v[1] + inertia[9]*v[5]};
    int at = a.matrix_row[dof] + a.matrix_nnz[dof] - 1;
    float* matrix = a.matrix + w * a.matrix_width;
    const float armature=a.armature[(w % a.armature_rows)*a.nv+dof];
    for (int ancestor = dof; ancestor >= 0; ancestor = a.dof_parent[ancestor], --at) {
      const float* motion = a.cdof + (w * a.nv + ancestor) * 6;
      float value = 0.0f;
      for (int k = 0; k < 6; ++k) value += motion[k] * product[k];
      matrix[at] = ancestor==dof ? armature+value : value;
    }
  }
}

}
}

extern "C" {
const char* flash_mjwarp_mass_matrix_source_id() { return FLASH_MJW_MASS_MATRIX_SOURCE_ID; }
std::size_t flash_mjwarp_mass_matrix_arguments_bytes() { return sizeof(flash::mjwarp::MassMatrixInputs); }
int flash_mjwarp_mass_matrix_arch() { return FLASH_MJW_MASS_MATRIX_ARCH; }
int flash_mjwarp_mass_matrix_accumulate(const flash::mjwarp::MassMatrixInputs* a,
                                      const int* layer, int count, cudaStream_t stream) {
  if (!a || !layer || a->worlds < 1 || count < 1) return int(cudaErrorInvalidValue);
  flash::mjwarp::accumulate<<<(a->worlds * count + 255) / 256, 256, 0, stream>>>(*a, layer, count);
  return int(cudaGetLastError());
}
int flash_mjwarp_mass_matrix_assemble(const flash::mjwarp::MassMatrixInputs* a, cudaStream_t stream) {
  if (!a || a->worlds < 1 || a->nv < 1 || a->bodies < 1 || a->armature_rows < 1)
    return int(cudaErrorInvalidValue);
  flash::mjwarp::assemble<<<(a->worlds * a->nv + 255) / 256, 256, 0, stream>>>(*a);
  return int(cudaGetLastError());
}
int flash_mjwarp_mass_matrix_fused(const flash::mjwarp::MassMatrixInputs* a,const float* cinert,
    const int* bodies,const int* starts,int levels,const int* child_start,const int* child_id,cudaStream_t stream) {
  if (!a || !cinert || !bodies || !starts || !child_start || a->worlds<1 || a->bodies<1 || a->nv<1 ||
      a->armature_rows<1 || (a->bodies>1 && !child_id) || levels<1 || levels>a->bodies)
    return cudaErrorInvalidValue;
  using namespace flash::mjwarp;
  if (a->bodies<=64) world_mass<128,true><<<a->worlds,128,0,stream>>>(*a,cinert,bodies,starts,levels,child_start,child_id);
  else world_mass<128,false><<<a->worlds,128,0,stream>>>(*a,cinert,bodies,starts,levels,child_start,child_id);
  return cudaGetLastError();
}

}
