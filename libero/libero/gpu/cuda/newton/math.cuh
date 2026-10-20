#include "precision.cuh"
#include "io_policy.cuh"
// Source-order FP64 helpers. MuJoCo 2.3.7, Apache-2.0.
// NV and NM are cold capacity parameters, never task identifiers.
#pragma once
// Compile-time reference path for rank-update performance comparisons only.
// Both choices retain pivot order, zero skips and rank regularization.
#ifndef LIBERO_CHOL_UPDATE_WARP
#define LIBERO_CHOL_UPDATE_WARP 1
#endif
__device__ inline real_t dot4(const real_t* a, const real_t* b, int n) {
  real_t s0=0, s1=0, s2=0, s3=0; int k=0;
#if LIBERO_NEWTON_SCALAR_IO
  if(n>=4){s0=a[0]*b[0];s1=a[1]*b[1];s2=a[2]*b[2];s3=a[3]*b[3];k=4;}
  for(;k+4<=n;k+=4){s0+=a[k]*b[k];s1+=a[k+1]*b[k+1];s2+=a[k+2]*b[k+2];s3+=a[k+3]*b[k+3];}
#else
  // Pair only the loads: accumulator lanes, initialization, and tails retain
  // the source expression order. Each pair lies inside a full four-wide chunk.
  if(n>=4){
    const real2_t a01=libero_vec::load_pair(a),b01=libero_vec::load_pair(b);
    const real2_t a23=libero_vec::load_pair(a+2),b23=libero_vec::load_pair(b+2);
    s0=a01.x*b01.x;s1=a01.y*b01.y;s2=a23.x*b23.x;s3=a23.y*b23.y;k=4;
  }
  for(;k+4<=n;k+=4){
    const real2_t a01=libero_vec::load_pair(a+k),b01=libero_vec::load_pair(b+k);
    const real2_t a23=libero_vec::load_pair(a+k+2),b23=libero_vec::load_pair(b+k+2);
    s0+=a01.x*b01.x;s1+=a01.y*b01.y;s2+=a23.x*b23.x;s3+=a23.y*b23.y;
  }
#endif // LIBERO_NEWTON_SCALAR_IO
  real_t s=(s0+s2)+(s1+s3);
  if(n-k==3)s+=a[k]*b[k]+a[k+1]*b[k+1]+a[k+2]*b[k+2];
  else if(n-k==2)s+=a[k]*b[k]+a[k+1]*b[k+1];
  else if(n-k==1)s+=a[k]*b[k];
  return s;
}
__device__ inline int mass_index(int i,int j,const int* madr,const int* parent,const int* simple){
  int adr=madr[i];
  for(int k=i;k>=0;k=parent[k],++adr){if(k==j)return adr;if(simple[i])break;}
  return -1;
}
__device__ inline void mass_multiply(const real_t* v,const real_t* M,const int* madr,
                                    const int* parent,const int* simple,real_t* out){
  for(int i=0;i<NV;++i)out[i]=REAL_C(0.0);
  for(int i=0;i<NV;++i){int adr=madr[i];
    if(simple[i])out[i]=M[adr]*v[i];
    else {out[i]+=M[adr]*v[i];++adr;
      for(int j=parent[i];j>=0;j=parent[j],++adr){out[i]+=M[adr]*v[j];out[j]+=M[adr]*v[i];}}
  }
}
__device__ inline int chol_factor(real_t* H,int n,int stride){
  int rank=n;
  for(int j=0;j<n;++j){real_t d=H[j*stride+j];if(j)d-=dot4(H+j*stride,H+j*stride,j);
    if(d<REAL_C(1e-15)){d=REAL_C(1e-15);--rank;}H[j*stride+j]=sqrt(d);real_t inv=REAL_C(1.0)/H[j*stride+j];
    for(int i=j+1;i<n;++i)H[i*stride+j]=(H[i*stride+j]-dot4(H+i*stride,H+j*stride,j))*inv;
  }return rank;
}
__device__ inline int chol_update(real_t* H,real_t* v,int plus){
  int rank=NV;
  for(int k=0;k<NV;++k){real_t x=v[k];if(x==REAL_C(0.0))continue;
    real_t L=H[k*NV+k],tmp=L*L;tmp=plus?tmp+x*x:tmp-x*x;
    if(tmp<REAL_C(1e-15)){tmp=REAL_C(1e-15);--rank;}real_t r=sqrt(tmp),c=r/L,inv=REAL_C(1.0)/c,s=x/L;H[k*NV+k]=r;
    for(int i=k+1;i<NV;++i)H[i*NV+k]=(plus?H[i*NV+k]+s*v[i]:H[i*NV+k]-s*v[i])*inv;
    for(int i=k+1;i<NV;++i)v[i]=c*v[i]-s*H[i*NV+k];
  }return rank;
}
// All block threads enter. Only warp 0 touches the update; the final block
// barrier publishes its complete private factor/vector before another phase.
// Pivot order and each element's arithmetic match chol_update exactly.
__device__ inline int chol_update_block(real_t* H,real_t* v,int plus){
  __shared__ int result;
  if constexpr (NV <= 32 || !LIBERO_CHOL_UPDATE_WARP) {
    if(threadIdx.x==0) result=chol_update(H,v,plus);
  } else if(threadIdx.x<32) {
    const int lane=threadIdx.x;
    int rank=NV;
    for(int k=0;k<NV;++k){
      real_t x=lane==0?v[k]:REAL_C(0.0);
      x=__shfl_sync(0xffffffffu,x,0);
      if(x==REAL_C(0.0)) continue; // uniform warp branch, including signed zero
      real_t c=REAL_C(0.0),inv=REAL_C(0.0),s=REAL_C(0.0);
      if(lane==0){
        real_t L=H[k*NV+k],tmp=L*L;
        tmp=plus?tmp+x*x:tmp-x*x;
        if(tmp<REAL_C(1e-15)){tmp=REAL_C(1e-15);--rank;}
        real_t r=sqrt(tmp); c=r/L; inv=REAL_C(1.0)/c; s=x/L;
        H[k*NV+k]=r;
      }
      c=__shfl_sync(0xffffffffu,c,0);
      inv=__shfl_sync(0xffffffffu,inv,0);
      s=__shfl_sync(0xffffffffu,s,0);
      for(int i=k+1+lane;i<NV;i+=32){
        real_t updated=(plus?H[i*NV+k]+s*v[i]:H[i*NV+k]-s*v[i])*inv;
        H[i*NV+k]=updated;
        v[i]=c*v[i]-s*updated;
      }
      __syncwarp(0xffffffffu); // next pivot may read another lane's v[k]
    }
    if(lane==0) result=rank;
  }
  __syncthreads();
  return result;
}
