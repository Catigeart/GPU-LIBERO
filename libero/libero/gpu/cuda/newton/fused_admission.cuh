#include "precision.cuh"
#pragma once
// Ordered topology has already been admitted by constraint_update_block.
// Keep its cheap head-map construction ordered, and distribute the expensive
// per-cone numerical admission and last-middle-cone ownership over CTA lanes.
__device__ inline void fused_numeric_admission_block(int w,int n,int eq,int fc,
 int rb,int cb,bool cached,const int* typ,const int* cid,const int* dim,
 const real_t* jar,const real_t* mu,const real_t* fr,int* group_head,
 int* hessian_owner,int* invariant_valid,int* status,int* warp_bad){
  if(threadIdx.x==0&&!cached){
    for(int r=0;r<eq+fc;++r)group_head[rb+r]=r;
    for(int r=eq+fc;r<n;){
      const int d=typ[rb+r]==7?dim[cb+cid[rb+r]]:1;
      group_head[rb+r]=r;
      for(int j=1;j<d;++j)group_head[rb+r+j]=-1;
      r+=d;
    }
  }
  __syncthreads();
  int bad=libero_warp::no_index;
  for(int r=eq+fc+threadIdx.x;r<n;r+=blockDim.x){
    if(group_head[rb+r]!=r||typ[rb+r]!=7)continue;
    const int c=cb+cid[rb+r],d=dim[c];const real_t m=mu[c];
    if(!cached){
      if(m<REAL_C(.0001)||m>10000)bad=0;
      for(int j=0;j<d-1;++j)if(fr[c*5+j]<REAL_C(.0001)||fr[c*5+j]>10000)bad=0;
    }
    const real_t T=cone_norm(jar+rb+r,fr+c*5,d),N=jar[rb+r]*m;
    const bool top=N>=m*T||(T<=0&&N>=0),bottom=m*N+T<=0||(T<=0&&N<0);
    if(!top&&!bottom){
      if(T<REAL_C(1e-12))bad=0;
      // Repeated ids are legal: later top/bottom groups leave the earlier H
      // intact. Only the last middle group is a mathematical Hessian writer.
      atomicMax(hessian_owner+c,r);
    }
  }
  bad=libero_warp::min(bad);
  if(libero_warp::lane()==0)warp_bad[libero_warp::warp()]=bad;
  __syncthreads();
  if(threadIdx.x==0){
    int first=libero_warp::no_index;
    for(int k=0;k<blockDim.x/32;++k)first=min(first,warp_bad[k]);
    if(first!=libero_warp::no_index)status[w]=10;
    else if(invariant_valid)invariant_valid[0]=1;
  }
  __syncthreads();
}
