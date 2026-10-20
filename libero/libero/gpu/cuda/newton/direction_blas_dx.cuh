#include "precision.cuh"
#pragma once

// Both calls are collective block operations; matrix and RHS are shared memory.
__device__ void dx_direction_forward(real_t* lower, real_t* rhs);
__device__ void dx_direction_backward(real_t* lower, real_t* rhs);
#if LIBERO_NEWTON_ISLANDS
namespace libero_islands {
__device__ bool direction_with_forward(const real_t*,const real_t*,real_t*,real_t*,real_t*,int*);
}
#endif

__device__ void direction_blas_dx(int w, real_t* Ma, real_t* smooth,
 real_t* constraint, real_t* H, real_t* HC, int* ncone, int* nefc,
 int* active, int* upstream, real_t* grad, real_t* Mgrad, real_t* candidate,
 int* status, real_t* scratch_grad, real_t* x, real_t* fwd,
  real_t* workspace, int* blas_diag,bool base_shared=false,bool cone_shared=false,
  int* islands_diag=nullptr) {
  const int lane = threadIdx.x;
  const int vb = w * NV;
  const real_t* L = (ncone[w] > 0 ? HC : H) + w * NV * NV;
  bool resident=ncone[w]>0?cone_shared:base_shared;
  real_t* lower = resident?const_cast<real_t*>(L):workspace;
  // Each Dx operand begins at a 16-byte aligned address. An odd NV gives
  // NV*NV FP64 cells an 8-byte remainder; pad only the staging subview.
  // The factor itself retains compact row-major lda=NV.
  constexpr int matrix_cells=(NV*NV+1)&~1;
  real_t* rhs = workspace + (resident?0:matrix_cells);
  __shared__ int first_bad;
  if(lane==0){
    status[w]=upstream[w];
    if(!status[w]&&active[w]){
      if(active[w]!=1)status[w]=1;
      else if(nefc[w]<0||ncone[w]<0||ncone[w]>nefc[w])status[w]=2;
    }
  }
  __syncthreads();
  if(status[w]||!active[w])return;
  if(lane<32){
    bool bad=false,pivot=false;
    for(int i=lane;i<NV;i+=32){
      bad|=!isfinite(Ma[vb+i])||!isfinite(smooth[vb+i])||!isfinite(constraint[vb+i]);
      pivot|=L[i*NV+i]==0;
    }
    for(int z=lane;z<NV*NV;z+=32)
      if(z%NV<=z/NV)bad|=!isfinite(L[z]);
    bad=libero_warp::any(bad);pivot=libero_warp::any(pivot);
    if(lane==0)status[w]=bad?3:(pivot?4:0);
  }
  __syncthreads();
  if(status[w])return;
  if(lane<32){
    int bad=libero_warp::no_index;
    for(int i=lane;i<NV;i+=32){
      real_t v=(Ma[vb+i]-smooth[vb+i])-constraint[vb+i];
      if(!isfinite(v)&&i<bad)bad=i;
    }
    bad=libero_warp::min(bad);
    if(lane==0){first_bad=bad;if(bad!=libero_warp::no_index)status[w]=5;}
  }
  __syncthreads();
  // Preserve the original diagnostic prefix, including on overflow failure.
  for(int i=lane;i<NV;i+=blockDim.x)if(i<first_bad){
    real_t v=(Ma[vb+i]-smooth[vb+i])-constraint[vb+i];
    scratch_grad[vb+i]=v;x[vb+i]=v;
  }
  __syncthreads();
  if(status[w])return;
#if LIBERO_NEWTON_ISLANDS
  // The full gradient/entry contract above is unchanged. Rebuild exact current
  // factor islands now: rank updates may have introduced new coupling since
  // the preceding factor call. Success publishes both forward/final scratch;
  // failure holds them and can safely use the source full-NV diagnostic route.
  const bool island_solved=libero_islands::direction_with_forward(
      L,x+vb,x+vb,fwd+vb,workspace,islands_diag);
  if(island_solved) {
    if(lane==0)++blas_diag[2];
    for(int i=lane;i<NV;i+=blockDim.x) {
      grad[vb+i]=scratch_grad[vb+i];Mgrad[vb+i]=x[vb+i];
      candidate[vb+i]=copysign(x[vb+i],-copysign(REAL_C(1.0),x[vb+i]));
    }
    __syncthreads();
    return;
  }
#endif
  if(!resident)for (int z = lane; z < NV*NV; z += blockDim.x)
    lower[z] = z % NV <= z / NV ? L[z] : REAL_C(0.0);
  for (int i = lane; i < NV; i += blockDim.x) rhs[i] = x[vb+i];
  if (lane == 0) ++blas_diag[2];
  __syncthreads();
  dx_direction_forward(lower, rhs);
  __syncthreads();
  if(lane<32){
    int bad=libero_warp::no_index;
    for(int i=lane;i<NV;i+=32)if(!isfinite(rhs[i])&&i<bad)bad=i;
    bad=libero_warp::min(bad);
    if(lane==0){first_bad=bad;if(bad!=libero_warp::no_index)status[w]=5;}
  }
  __syncthreads();
  for(int i=lane;i<NV;i+=blockDim.x)if(i<first_bad){x[vb+i]=rhs[i];fwd[vb+i]=rhs[i];}
  __syncthreads();
  if(status[w])return;
  dx_direction_backward(lower, rhs);
  __syncthreads();
  if(lane<32){
    int bad=libero_warp::no_index;
    // Reverse solve commits a suffix; reduce reverse ordinal rather than index.
    for(int i=lane;i<NV;i+=32)if(!isfinite(rhs[i])&&NV-1-i<bad)bad=NV-1-i;
    bad=libero_warp::min(bad);
    if(lane==0){first_bad=bad;if(bad!=libero_warp::no_index)status[w]=5;}
  }
  __syncthreads();
  for(int i=lane;i<NV;i+=blockDim.x)if(NV-1-i<first_bad)x[vb+i]=rhs[i];
  __syncthreads();
  if(status[w])return;
  for (int i = lane; i < NV; i += blockDim.x) {
    grad[vb+i] = scratch_grad[vb+i];
    Mgrad[vb+i] = x[vb+i];
    candidate[vb+i] = copysign(x[vb+i], -copysign(REAL_C(1.0), x[vb+i]));
  }
  __syncthreads();
}
