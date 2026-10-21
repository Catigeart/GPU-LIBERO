// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Handwritten CUDA adaptation of MJWarp c384ac51df67eb92ab7657cc472ff7494dc036bf:
// solver.py::_eval_constraint, _eval_elliptic_middle,
// _update_constraint_efc(False), _update_constraint_init_qfrc_constraint_dense(False).
// Same per-row force ownership and per-DOF sequential J^T force accumulation.
// Inputs are existing constraint rows, not contact/constraint generation.
#include "mjwarp_constraint.cuh"
#include "mjwarp_direction.cuh"
#include <cmath>

#ifndef FLASH_MJW_CONSTRAINT_SOURCE_ID
#error "Build through flash_mjwarp_constraint_fp32"
#endif

namespace flash::mjwarp {
namespace {
constexpr int kSatisfied = 0, kQuadratic = 1, kLinearNegative = 2;
constexpr int kLinearPositive = 3, kCone = 4;
constexpr int kContactElliptic = 7;

// Pinned math.safe_div: replace an exactly zero denominator with mjMINVAL.
__device__ float safe_div(float x, float denominator) {
  return x / (denominator == 0.f ? 1.e-15f : denominator);
}

__global__ void constraint_rows(ConstraintInputs a) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= a.worlds * a.rows) return;
  const int world = index / a.rows, row = index % a.rows;
  if (a.done[world] || row >= a.nefc[world]) return;
  const float d = a.diagonal[index], residual = a.jaref[index];
  const bool equality = row < a.ne[world];
  const bool friction = !equality && row < a.ne[world] + a.nf[world];
  const bool elliptic = a.row_type[index] == kContactElliptic;
  int head = -1;
  float normal = 0.f, d0 = 0.f, mu = 0.f, tangent_force_scale = 0.f, tt = 0.f;
  if (elliptic) {
    const int contact = a.row_contact[index];
    if (contact >= a.contact_count[0]) return;
    const int* addresses = a.contact_address + 6 * contact;
    head = addresses[0];
    if (head < 0) return;
    const int base = world * a.rows;
    const float* f = a.contact_friction + 5 * contact;
    mu = f[0] * a.impratio[world % a.impratio_count];
    normal = a.jaref[base + head];
    d0 = a.diagonal[base + head];
    for (int j = 1; j < a.contact_dimension[contact]; ++j) {
      const int secondary = addresses[j];
      if (secondary < 0) return;
      const float u = a.jaref[base + secondary] * f[j - 1];
      tt += u * u;
      if (row == secondary) tangent_force_scale = u * f[j - 1];
    }
  }

  float force;
  int state;
  if (equality) {
    force = -d * residual;
    state = kQuadratic;
  } else if (friction) {
    const float loss = a.frictionloss[index], radius = safe_div(loss, d);
    if (residual <= -radius) { force = loss; state = kLinearNegative; }
    else if (residual >= radius) { force = -loss; state = kLinearPositive; }
    else { force = -d * residual; state = kQuadratic; }
  } else if (elliptic) {
    const float n = normal * mu;
    const float t = tt <= 0.f ? 0.f : sqrtf(tt);
    if (n >= mu * t || (t <= 0.f && n >= 0.f)) {
      force = 0.f;
      state = kSatisfied;
    } else if (mu * n + t <= 0.f || (t <= 0.f && n < 0.f)) {
      force = -d * residual;
      state = kQuadratic;
    } else {
      const float dm = safe_div(d0, mu * mu * (1.f + mu * mu));
      const float force_normal = -dm * (n - mu * t) * mu;
      force = row == head ? force_normal : -safe_div(force_normal, t) * tangent_force_scale;
      state = kCone;
    }
  } else if (residual >= 0.f) {
    force = 0.f;
    state = kSatisfied;
  } else {
    force = -d * residual;
    state = kQuadratic;
  }
  a.force[index] = force;
  a.state[index] = state;
}

__global__ void constraint_transpose(ConstraintInputs a) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= a.worlds * a.nv) return;
  const int world = index / a.nv, dof = index % a.nv;
  if (a.done[world]) return;
  const int base = world * a.rows;
  float total = 0.f;
  for (int row = 0; row < min(a.rows, a.nefc[world]); ++row)
    total += a.jacobian[(std::size_t(base + row) * a.pad) + dof] * a.force[base + row];
  a.qfrc_constraint[index] = total;  // an active zero-row world publishes zero
}
// Current row forces -> J^T force -> gradient, with a single world barrier.
// Pooled contact metadata is read only; no CTA writes another world's state.
template<int Threads>
__global__ void constraint_gradient(ConstraintInputs a,DirectionInputs d) {
  __shared__ float scratch[1024];
  const int world=blockIdx.x,x=threadIdx.x;
  if(x==0) d.search_unchanged[world]=false;
  if(a.done[world]) return;
  if(x==0) {d.grad_dot[world]=0.f;d.decrement[world]=0.f;d.grad_scale[world]=1.f;}
  const int base=world*a.rows,active=min(a.rows,a.nefc[world]);
  const bool cached=active<=1024;
  for(int row=x;row<active;row+=Threads) {
    const int index=base+row;
    const float d = a.diagonal[index], residual = a.jaref[index];
    const bool equality = row < a.ne[world];
    const bool friction = !equality && row < a.ne[world] + a.nf[world];
    const bool elliptic = a.row_type[index] == kContactElliptic;
    int head = -1;
    float normal = 0.f, d0 = 0.f, mu = 0.f, tangent_force_scale = 0.f, tt = 0.f;
    if (elliptic) {
      const int contact = a.row_contact[index];
      if(contact >= a.contact_count[0]) {if(cached) scratch[row]=a.force[index];continue;}
      const int* addresses = a.contact_address + 6 * contact;
      head = addresses[0];
      if(head < 0) {if(cached) scratch[row]=a.force[index];continue;}
      const int base = world * a.rows;
      const float* f = a.contact_friction + 5 * contact;
      mu = f[0] * a.impratio[world % a.impratio_count];
      normal = a.jaref[base + head];
      d0 = a.diagonal[base + head];
      bool valid=true;
      for (int j = 1; j < a.contact_dimension[contact]; ++j) {
        const int secondary = addresses[j];
        if (secondary < 0) {valid=false;break;}
        const float u = a.jaref[base + secondary] * f[j - 1];
        tt += u * u;
        if (row == secondary) tangent_force_scale = u * f[j - 1];
      }
      if(!valid) {if(cached) scratch[row]=a.force[index];continue;}
    }
  
    float force;
    int state;
    if (equality) {
      force = -d * residual;
      state = kQuadratic;
    } else if (friction) {
      const float loss = a.frictionloss[index], radius = safe_div(loss, d);
      if (residual <= -radius) { force = loss; state = kLinearNegative; }
      else if (residual >= radius) { force = -loss; state = kLinearPositive; }
      else { force = -d * residual; state = kQuadratic; }
    } else if (elliptic) {
      const float n = normal * mu;
      const float t = tt <= 0.f ? 0.f : sqrtf(tt);
      if (n >= mu * t || (t <= 0.f && n >= 0.f)) {
        force = 0.f;
        state = kSatisfied;
      } else if (mu * n + t <= 0.f || (t <= 0.f && n < 0.f)) {
        force = -d * residual;
        state = kQuadratic;
      } else {
        const float dm = safe_div(d0, mu * mu * (1.f + mu * mu));
        const float force_normal = -dm * (n - mu * t) * mu;
        force = row == head ? force_normal : -safe_div(force_normal, t) * tangent_force_scale;
        state = kCone;
      }
    } else if (residual >= 0.f) {
      force = 0.f;
      state = kSatisfied;
    } else {
      force = -d * residual;
      state = kQuadratic;
    }
    a.force[index] = force;
    if(cached) scratch[row]=force;
    a.state[index] = state;
  }
  __syncthreads();
  // A warp owns four DOFs and eight interleaved row ranges. The producer
  // boundary is unchanged, while a single world exposes row parallelism too.
  const int lane=x%32,warp=x/32;
  for(int pack=warp;pack*4<a.nv;pack+=Threads/32) {
    const int dof=pack*4+lane%4;float total=0.f;
    if(dof<a.nv)
      for(int row=lane/4;row<active;row+=8)
        total+=a.jacobian[(std::size_t(base+row)*a.pad)+dof]*(cached ? scratch[row] : a.force[base+row]);
    total+=__shfl_down_sync(0xffffffffu,total,16);
    total+=__shfl_down_sync(0xffffffffu,total,8);
    total+=__shfl_down_sync(0xffffffffu,total,4);
    if(lane<4 && dof<a.nv) {
      a.qfrc_constraint[world*a.nv+dof]=total;
      const float g=d.ma[world*d.ma_stride+dof]-d.smooth[world*d.smooth_stride+dof]-total;
      d.grad[world*d.pad+dof]=g;
      atomicAdd(d.grad_dot+world,g*g);
    }
  }
}

}
}

extern "C" const char* flash_mjwarp_constraint_source_id() { return FLASH_MJW_CONSTRAINT_SOURCE_ID; }
extern "C" std::size_t flash_mjwarp_constraint_arguments_bytes() { return sizeof(flash::mjwarp::ConstraintInputs); }
extern "C" int flash_mjwarp_constraint_arch() { return FLASH_MJW_CONSTRAINT_ARCH; }
extern "C" int flash_mjwarp_constraint_launch(const flash::mjwarp::ConstraintInputs* input,
                                              int stage, cudaStream_t stream) {
  using namespace flash::mjwarp;
  if (!input || stage < 0 || stage > 2) return int(cudaErrorInvalidValue);
  const auto& a = *input;
  if (a.worlds < 1 || a.nv < 1 || a.pad < a.nv || a.rows < 1) return int(cudaErrorInvalidValue);
  if (stage != 2)
    constraint_rows<<<(a.worlds * a.rows + 255) / 256, 256, 0, stream>>>(a);
  if (stage != 1)
    constraint_transpose<<<(a.worlds * a.nv + 255) / 256, 256, 0, stream>>>(a);
  return int(cudaGetLastError());
}

extern "C" std::size_t flash_mjwarp_constraint_gradient_bytes() {return sizeof(flash::mjwarp::DirectionInputs);}
extern "C" int flash_mjwarp_constraint_gradient(const flash::mjwarp::ConstraintInputs* a,
                                                const flash::mjwarp::DirectionInputs* d,cudaStream_t stream) {
  if(!a || !d || a->worlds<1 || a->nv<1 || a->rows<1 || a->pad<a->nv || a->capacity<0 || a->impratio_count<1 ||
     a->worlds!=d->worlds || a->nv!=d->nv || a->pad!=d->pad || a->done!=d->done ||
     d->force_stride!=a->nv || a->qfrc_constraint!=d->force) return cudaErrorInvalidValue;
  flash::mjwarp::constraint_gradient<512><<<a->worlds,512,0,stream>>>(*a,*d);
  return cudaGetLastError();
}
