// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Pinned smooth.py::com_pos: mass-weighted subtree CoM, centered inertia, DOFs.
#include "mjwarp_com_pos.cuh"

namespace flash::mjwarp {
namespace {
__global__ void subtree_init(ComPosInputs a) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.worlds * a.bodies) return;
  int w = t / a.bodies, b = t % a.bodies;
  float mass = a.mass[(w % a.mass_rows) * a.bodies + b];
  for (int k = 0; k < 3; ++k) a.subtree[3*t+k] = a.xipos[3*t+k] * mass;
}

__global__ void subtree_accumulate(ComPosInputs a, const int* layer, int count) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.worlds * count) return;
  int w = t / count, b = layer[t % count];
  if (b == 0) return;
  int source = 3*(w*a.bodies+b), target = 3*(w*a.bodies+a.parent[b]);
  // This reduction includes world body0. Layers serialize parent dependencies;
  // siblings at one depth use FP32 atomics, exactly as in the pinned schedule.
  for (int k = 0; k < 3; ++k) atomicAdd(a.subtree+target+k, a.subtree[source+k]);
}

__global__ void subtree_divide(ComPosInputs a) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.worlds * a.bodies) return;
  int w = t / a.bodies, b = t % a.bodies;
  float mass = a.subtreemass[(w % a.subtree_rows)*a.bodies+b];
  // Zero mass holds the accumulated sum; do not import CPU's xipos fallback.
  if (mass != 0.0f)
    for (int k = 0; k < 3; ++k) a.subtree[3*t+k] /= mass;
}

__global__ void centered_inertia(ComPosInputs a) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.worlds * a.bodies) return;
  int w = t / a.bodies, b = t % a.bodies;
  const float* r = a.ximat+9*t;
  const float* inertia = a.inertia+3*((w % a.inertia_rows)*a.bodies+b);
  const float* com = a.subtree+3*(w*a.bodies+a.root[b]);
  float mass = a.mass[(w % a.mass_rows)*a.bodies+b];
  float dx = a.xipos[3*t]-com[0], dy = a.xipos[3*t+1]-com[1], dz = a.xipos[3*t+2]-com[2];
  // R * diag(I) * R^T, including world body0. Keep vec10's public layout.
  float scaled[9];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) scaled[3*i+j] = r[3*i+j]*inertia[j];
  float* out = a.cinert+10*t;
  const int rows[6] = {0,1,2,0,0,1}, cols[6] = {0,1,2,1,2,2};
  for (int i = 0; i < 6; ++i) {
    int row = 3*rows[i], col = 3*cols[i];
    out[i] = scaled[row]*r[col]+scaled[row+1]*r[col+1]+scaled[row+2]*r[col+2];
  }
  out[0] += mass*(dy*dy+dz*dz);
  out[1] += mass*(dx*dx+dz*dz);
  out[2] += mass*(dx*dx+dy*dy);
  out[3] -= mass*dx*dy;
  out[4] -= mass*dx*dz;
  out[5] -= mass*dy*dz;
  out[6] = mass*dx; out[7] = mass*dy; out[8] = mass*dz; out[9] = mass;
}

__device__ void rotational_axis(float* out, float x, float y, float z,
                                float dx, float dy, float dz) {
  out[0] = x; out[1] = y; out[2] = z;
  out[3] = y*dz-z*dy; out[4] = z*dx-x*dz; out[5] = x*dy-y*dx;
}

__global__ void motion_axes(ComPosInputs a) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.worlds * a.joints) return;
  int w = t / a.joints, j = t % a.joints, b = a.jbody[j], kind = a.kind[j];
  float* out = a.cdof+6*(w*a.nv+a.dofadr[j]);
  const float* com = a.subtree+3*(w*a.bodies+a.root[b]);
  float dx = com[0]-a.anchor[3*t], dy = com[1]-a.anchor[3*t+1], dz = com[2]-a.anchor[3*t+2];
  if (kind == 0) {  // FREE: world translations followed by body rotations.
    for (int axis = 0; axis < 3; ++axis)
      for (int k = 0; k < 6; ++k) out[6*axis+k] = float(k == axis+3);
    out += 18;
  }
  if (kind == 0 || kind == 1) {  // Columns of the local-to-global body matrix.
    const float* r = a.xmat+9*(w*a.bodies+b);
    for (int k = 0; k < 3; ++k) rotational_axis(out+6*k,r[k],r[3+k],r[6+k],dx,dy,dz);
  } else if (kind == 2) {  // SLIDE: angular part is exactly zero.
    for (int k = 0; k < 3; ++k) {out[k] = 0.0f; out[k+3] = a.axis[3*t+k];}
  } else {  // HINGE; cold admission proves the joint type and unique DOF spans.
    rotational_axis(out,a.axis[3*t],a.axis[3*t+1],a.axis[3*t+2],dx,dy,dz);
  }
}
// One CTA owns the complete subtree reduction for a world. Layer barriers
// order child reads; each parent gathers its children without atomic writes.
template<int Threads,bool Shared>
__global__ void centered_frames(ComPosInputs a,const int* bodies,const int* starts,int levels,const int* child_start,const int* child_id) {
  const int w=blockIdx.x,lane=threadIdx.x;
  __shared__ float scratch[Shared ? 3*64 : 1];
  float* sums=Shared ? scratch : a.subtree+3*w*a.bodies;
  // Gather each body's children after their depth is complete. Each sum has
  // one writer; no atomic lock or initialization pass is required.
  for (int level=levels-1;level>=0;--level) {
    for (int at=starts[level]+lane;at<starts[level+1];at+=Threads) {
      const int b=bodies[at],t=w*a.bodies+b;
      const float mass=a.mass[(w%a.mass_rows)*a.bodies+b];
      float value[3];for (int k=0;k<3;++k) value[k]=a.xipos[3*t+k]*mass;
      for (int edge=child_start[b];edge<child_start[b+1];++edge) {
        const int child=child_id[edge];
        for (int k=0;k<3;++k) value[k]+=sums[3*child+k];
      }
      for (int k=0;k<3;++k) sums[3*b+k]=value[k];
    }
    __syncthreads();
  }
  for (int b=lane;b<a.bodies;b+=Threads) {
    const float mass=a.subtreemass[(w%a.subtree_rows)*a.bodies+b];
    if (mass!=0.f) for (int k=0;k<3;++k) sums[3*b+k]/=mass;
  }
  if constexpr (Shared) for (int b=lane;b<a.bodies;b+=Threads)
    for (int k=0;k<3;++k) a.subtree[3*(w*a.bodies+b)+k]=sums[3*b+k];
  __syncthreads(); // normalized roots are ready for inertia and joint consumers
  for (int b=lane;b<a.bodies;b+=Threads) {
    const int t=w*a.bodies+b;
    const float* r = a.ximat+9*t;
    const float* inertia = a.inertia+3*((w % a.inertia_rows)*a.bodies+b);
    const float* com = sums+3*a.root[b];
    float mass = a.mass[(w % a.mass_rows)*a.bodies+b];
    float dx = a.xipos[3*t]-com[0], dy = a.xipos[3*t+1]-com[1], dz = a.xipos[3*t+2]-com[2];
    // R * diag(I) * R^T, including world body0. Keep vec10's public layout.
    float scaled[9];
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) scaled[3*i+j] = r[3*i+j]*inertia[j];
    float* out = a.cinert+10*t;
    const int rows[6] = {0,1,2,0,0,1}, cols[6] = {0,1,2,1,2,2};
    for (int i = 0; i < 6; ++i) {
      int row = 3*rows[i], col = 3*cols[i];
      out[i] = scaled[row]*r[col]+scaled[row+1]*r[col+1]+scaled[row+2]*r[col+2];
    }
    out[0] += mass*(dy*dy+dz*dz);
    out[1] += mass*(dx*dx+dz*dz);
    out[2] += mass*(dx*dx+dy*dy);
    out[3] -= mass*dx*dy;
    out[4] -= mass*dx*dz;
    out[5] -= mass*dy*dz;
    out[6] = mass*dx; out[7] = mass*dy; out[8] = mass*dz; out[9] = mass;
  }
  for (int j=lane;j<a.joints;j+=Threads) {
    const int t=w*a.joints+j,b=a.jbody[j],kind=a.kind[j];
    float* out = a.cdof+6*(w*a.nv+a.dofadr[j]);
    const float* com = sums+3*a.root[b];
    float dx = com[0]-a.anchor[3*t], dy = com[1]-a.anchor[3*t+1], dz = com[2]-a.anchor[3*t+2];
    if (kind == 0) {  // FREE: world translations followed by body rotations.
      for (int axis = 0; axis < 3; ++axis)
        for (int k = 0; k < 6; ++k) out[6*axis+k] = float(k == axis+3);
      out += 18;
    }
    if (kind == 0 || kind == 1) {  // Columns of the local-to-global body matrix.
      const float* r = a.xmat+9*(w*a.bodies+b);
      for (int k = 0; k < 3; ++k) rotational_axis(out+6*k,r[k],r[3+k],r[6+k],dx,dy,dz);
    } else if (kind == 2) {  // SLIDE: angular part is exactly zero.
      for (int k = 0; k < 3; ++k) {out[k] = 0.0f; out[k+3] = a.axis[3*t+k];}
    } else {  // HINGE; cold admission proves the joint type and unique DOF spans.
      rotational_axis(out,a.axis[3*t],a.axis[3*t+1],a.axis[3*t+2],dx,dy,dz);
    }
  }
}

}
}

extern "C" {
const char* flash_mjwarp_com_pos_source_id() {return FLASH_MJW_COM_POS_SOURCE_ID;}
std::size_t flash_mjwarp_com_pos_arguments_bytes() {return sizeof(flash::mjwarp::ComPosInputs);}
int flash_mjwarp_com_pos_arch() {return FLASH_MJW_COM_POS_ARCH;}
int flash_mjwarp_com_pos_launch(const flash::mjwarp::ComPosInputs* a, int phase,
                              const int* layer, int count, cudaStream_t stream) {
  if (!a || a->worlds<1 || a->bodies<2 || a->joints<1 || a->nv<1 ||
      a->mass_rows<1 || a->subtree_rows<1 || a->inertia_rows<1) return int(cudaErrorInvalidValue);
  using namespace flash::mjwarp;
  int bodies = (a->worlds*a->bodies+255)/256;
  if (phase == 0) subtree_init<<<bodies,256,0,stream>>>(*a);
  else if (phase == 1) {
    if (!layer || count<1 || count>a->bodies) return int(cudaErrorInvalidValue);
    subtree_accumulate<<<(a->worlds*count+255)/256,256,0,stream>>>(*a,layer,count);
  } else if (phase == 2) subtree_divide<<<bodies,256,0,stream>>>(*a);
  else if (phase == 3) centered_inertia<<<bodies,256,0,stream>>>(*a);
  else if (phase == 4) motion_axes<<<(a->worlds*a->joints+255)/256,256,0,stream>>>(*a);
  else return int(cudaErrorInvalidValue);
  return int(cudaGetLastError());
}
int flash_mjwarp_com_pos_fused(const flash::mjwarp::ComPosInputs* a,const int* bodies,
    const int* starts,int levels,const int* child_start,const int* child_id,cudaStream_t stream) {
  if (!a || !bodies || !starts || !child_start || !child_id || a->worlds<1 || a->bodies<2 ||
      a->joints<1 || a->nv<1 || levels<2 || levels>a->bodies) return cudaErrorInvalidValue;
  using namespace flash::mjwarp;
  // Common LIBERO models use 768 bytes; larger trees retain global storage.
  if (a->bodies<=64) centered_frames<64,true><<<a->worlds,64,0,stream>>>(*a,bodies,starts,levels,child_start,child_id);
  else centered_frames<64,false><<<a->worlds,64,0,stream>>>(*a,bodies,starts,levels,child_start,child_id);
  return cudaGetLastError();
}

}
