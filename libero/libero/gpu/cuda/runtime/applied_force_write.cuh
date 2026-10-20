#include "precision.cuh"
#pragma once
// Independent fresh private force roots. No force arithmetic is performed.
// Same original integer-preserving copy payload in both variants; only the
// submission count differs (two original payload calls versus one composed CTA).
namespace cf_force_write {
__device__ void single(int w,const real_t* source,real_t* target,const int* requested,
 const int* active,int* first,int* force_valid,int* full_valid,int width) {
 if(threadIdx.x==0 && requested[w]!=0 && requested[w]!=1 && first[w]==0)first[w]=201;
 if(threadIdx.x==0 && requested[w]==1 && active[w]!=0 && active[w]!=1 && first[w]==0)first[w]=201;
 __syncthreads();
 if(requested[w]!=1 || active[w]!=1 || first[w]!=0)return;
 if(threadIdx.x==0){force_valid[w]=0;full_valid[w]=0;}
 __syncthreads();
 cf_copy::payload(source+size_t(w)*width,target+size_t(w)*width,width);
}
__device__ void both(int w,const real_t* qsource,real_t* qtarget,const real_t* xsource,real_t* xtarget,
 const int* requested,const int* active,int* first,int* force_valid,int* full_valid,int nv,int nx) {
 single(w,qsource,qtarget,requested,active,first,force_valid,full_valid,nv);
 __syncthreads();
 single(w,xsource,xtarget,requested,active,first,force_valid,full_valid,nx);
}
}
