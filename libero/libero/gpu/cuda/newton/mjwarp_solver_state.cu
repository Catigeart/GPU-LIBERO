// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Readable CUDA port of pinned MJWarp c384ac51df67eb92ab7657cc472ff7494dc036bf:
// solver.py::_solve_init_dof(False sparse), _solve_init_efc,
// _solve_init_jaref_kernel(False sparse, False compact), _solve_done.
// The FP32 warmstart/decrement rules follow MJWarp, not MuJoCo 2.3.7's driver.
#include "mjwarp_solver_state.cuh"
#include "mjwarp_products.cuh"
#include <cstdio>

#ifndef FLASH_MJW_SOLVER_STATE_SOURCE_ID
#error "Build through flash_mjwarp_solver_state_fp32"
#endif

namespace flash::mjwarp {
namespace {
__global__ void select_acceleration(InitInputs a) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.worlds * a.nv) return;
  int w = t / a.nv;
  a.qacc[t] = a.warmstart && a.nefc[w] > 0 ? a.warm_acceleration[t] : a.smooth_acceleration[t];
  // Dense initialization does not touch qfrc_constraint; projection owns it.
}

__global__ void reset_worlds(InitInputs a) {
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= a.worlds) return;
  a.niter[w] = 0;
  a.done[w] = false;
  a.search_dot[w] = 0.f;
}

__global__ void initialize_jaref(InitInputs a, int parts, int chunk) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.worlds * a.rows * parts) return;
  int part = t % parts, row = (t / parts) % a.rows, w = t / (parts * a.rows);
  if (row >= a.nefc[w]) return;
  int at = w * a.rows + row;
  float value = jacobian_partial_dot(a.nv, part * chunk, chunk,
      a.jacobian + at * a.pad, a.qacc + w * a.nv);
  if (parts == 1) a.jaref[at] = value - a.aref[at];
  else {
    // Subtract aref in the first partial BEFORE the atomic accumulation,
    // matching the upstream grouping; not a separate post-subtraction kernel.
    if (part == 0) value -= a.aref[at];
    atomicAdd(a.jaref + at, value);
  }
}

__global__ void initialize_ma(InitInputs a) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.worlds * a.nv) return;
  int w = t / a.nv, i = t % a.nv;
  if (a.done[w]) return;
  a.ma[t] = mass_row_dot(i, a.mass_row, a.mass_col, a.mass_address,
                        a.mass + w * a.mass_entries, a.qacc + w * a.nv);
}

// One world owns initialization, so qacc selection and full zero/hold rules
// share one producer barrier before independent Ma/Jaref consumers.
template<int Threads,bool Shared>
__global__ void initialize_world(InitInputs a,float* grad,bool* exhausted,int* nsolving) {
  const int w=blockIdx.x,x=threadIdx.x;
  __shared__ float scratch[Shared ? 128 : 1];
  const float* selected=Shared ? scratch : a.qacc+w*a.nv;
  const float* __restrict__ jac=a.jacobian;
  const int chunk=a.nv>50 ? 20 : 50,parts=(a.nv+chunk-1)/chunk;
  for(int i=x;i<a.nv;i+=Threads) {
    float value=a.warmstart && a.nefc[w]>0 ? a.warm_acceleration[w*a.nv+i] : a.smooth_acceleration[w*a.nv+i];
    a.qacc[w*a.nv+i]=value;
    if constexpr(Shared) scratch[i]=value;
  }
  for(int i=x;i<a.pad;i+=Threads) grad[w*a.pad+i]=0.f;
  if(parts>1) for(int r=x;r<a.rows;r+=Threads) a.jaref[w*a.rows+r]=0.f;
  if(x==0) {
    a.niter[w]=0;a.done[w]=false;a.search_dot[w]=0.f;
    exhausted[w]=false;
    if(w==0) *nsolving=a.worlds;
  }
  __syncthreads();
  for(int i=x;i<a.nv;i+=Threads)
    a.ma[w*a.nv+i]=mass_row_dot(i,a.mass_row,a.mass_col,a.mass_address,a.mass+w*a.mass_entries,selected);
  // Four adjacent lanes cooperate on one row; the complete dot has a
  // single publisher, replacing split partial atomics within this world.
  const int lane=x%4;const unsigned mask=0xfu<<((x%32)/4*4);
  for(int row=x/4;row<min(a.rows,a.nefc[w]);row+=Threads/4) {
    const int at=w*a.rows+row;float value=0.f;
    for(int i=lane;i<a.nv;i+=4) value+=jac[at*a.pad+i]*selected[i];
    for(int shift=2;shift;shift/=2) value+=__shfl_down_sync(mask,value,shift,4);
    if(lane==0) a.jaref[at]=value-a.aref[at];
  }
}

// One thread per world, with an exact once-only transition to done. There is
// no reduction/barrier here; nsolving is the device conditional-Graph counter.
__global__ void stop_worlds(StopInputs a) {
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= a.worlds || a.done[w]) return;
  int iterations = ++a.niter[w];
  float tolerance = a.tolerance[w % a.tolerance_count];
  float scale = a.inertia[w % a.inertia_count] * float(a.nv);
  float improvement = a.improvement[w] / scale;
  float gradient = sqrtf(a.grad_dot[w]) / scale;
  float model_improvement = (0.5f * a.decrement[w]) / scale;
  bool done = a.alpha[w] == 0.f
      || (improvement > 0.f && improvement < tolerance)
      || gradient < tolerance || model_improvement < tolerance;
  if (done || iterations == a.iterations) {
    if (!done && iterations == a.iterations) {
      constexpr int IterationsOverflow = 1 << 9;
      if (a.warn_overflow & IterationsOverflow)
        printf("solver iterations limit reached - please increase iterations beyond %u\n"
               "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.ITERATIONS (or = 0 for all)\n", a.iterations);
      a.overflow[w] |= IterationsOverflow;
    }
    a.done[w] = true;
    atomicAdd(a.nsolving, -1);
  }
}
}
}

extern "C" {
const char* flash_mjwarp_solver_state_source_id() { return FLASH_MJW_SOLVER_STATE_SOURCE_ID; }
std::size_t flash_mjwarp_solver_state_init_bytes() { return sizeof(flash::mjwarp::InitInputs); }
std::size_t flash_mjwarp_solver_state_stop_bytes() { return sizeof(flash::mjwarp::StopInputs); }
int flash_mjwarp_solver_state_arch() { return FLASH_MJW_SOLVER_STATE_ARCH; }
int flash_mjwarp_solver_state_initialize(const flash::mjwarp::InitInputs* input, cudaStream_t stream) {
  using namespace flash::mjwarp;
  if (!input || input->worlds < 1 || input->nv < 1 || input->rows < 1) return int(cudaErrorInvalidValue);
  const auto& a = *input;
  select_acceleration<<<(a.worlds * a.nv + 255) / 256, 256, 0, stream>>>(a);
  reset_worlds<<<(a.worlds + 255) / 256, 256, 0, stream>>>(a);
  int chunk = a.nv > 50 ? 20 : 50, parts = (a.nv + chunk - 1) / chunk;
  if (parts > 1) {
    // Upstream clears the FULL Jaref allocation, including unused rows. Jv's
    // active-row-only clear must not be reused for this initialization.
    cudaError_t error = cudaMemsetAsync(a.jaref, 0, std::size_t(a.worlds) * a.rows * sizeof(float), stream);
    if (error != cudaSuccess) return int(error);
  }
  initialize_jaref<<<(a.worlds * a.rows * parts + 255) / 256, 256, 0, stream>>>(a, parts, chunk);
  initialize_ma<<<(a.worlds * a.nv + 255) / 256, 256, 0, stream>>>(a);
  return int(cudaGetLastError());
}
int flash_mjwarp_solver_state_fused(const flash::mjwarp::InitInputs* a,float* grad,
                                  bool* exhausted,int* nsolving,cudaStream_t stream) {
  if(!a || a->worlds<1 || a->nv<1 || a->pad<a->nv || a->rows<1 || !grad || !exhausted || !nsolving)
    return cudaErrorInvalidValue;
  if(a->nv<=128) flash::mjwarp::initialize_world<1024,true><<<a->worlds,1024,0,stream>>>(*a,grad,exhausted,nsolving);
  else flash::mjwarp::initialize_world<1024,false><<<a->worlds,1024,0,stream>>>(*a,grad,exhausted,nsolving);
  return cudaGetLastError();
}
int flash_mjwarp_solver_state_stop(const flash::mjwarp::StopInputs* input, cudaStream_t stream) {
  using namespace flash::mjwarp;
  if (!input || input->worlds < 1 || input->nv < 1 || !input->nsolving) return int(cudaErrorInvalidValue);
  stop_worlds<<<(input->worlds + 255) / 256, 256, 0, stream>>>(*input);
  return int(cudaGetLastError());
}
}
