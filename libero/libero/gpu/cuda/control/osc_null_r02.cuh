#include "precision.cuh"
#pragma once
#ifndef OSC_NULL_DIRECT
#define OSC_NULL_DIRECT 0
#endif
namespace osc_null_r02 {
// Complete independent primitive: world-major raw J(6,7), X(7,6), L(6,6),
// v(7) -> (I-X*L*J)^T*v. Each compile-time variant prepares only its required
// layout. No JT/XT/LT is supplied by a caller and no common transpose cost is
// hidden outside this boundary. This is not the full controller entry point.
__device__ inline void action(int w,const real_t* J,const real_t* X,
                             const real_t* L,const real_t* V,real_t* O){
 __shared__ real_t v[7],out[7];
#if OSC_NULL_DIRECT
 __shared__ real_t jt[42],xt[42],lt[36],z6[6],y6[6],c[7];
 for(int z=threadIdx.x;z<42;z+=blockDim.x){
  jt[(z%7)*6+z/7]=J[w*42+z];xt[(z%6)*7+z/6]=X[w*42+z];}
 for(int z=threadIdx.x;z<36;z+=blockDim.x)lt[(z%6)*6+z/6]=L[w*36+z];
#else
 __shared__ real_t j[42],x[42],l[36],b[42],n[49],nt[49];
 for(int z=threadIdx.x;z<42;z+=blockDim.x){j[z]=J[w*42+z];x[z]=X[w*42+z];}
 for(int z=threadIdx.x;z<36;z+=blockDim.x)l[z]=L[w*36+z];
#endif
 for(int z=threadIdx.x;z<7;z+=blockDim.x)v[z]=V[w*7+z];__syncthreads();
#if OSC_NULL_DIRECT
 osc_mm_6_1_7(xt,v,z6);osc_mm_6_1_6(lt,z6,y6);osc_mm_7_1_6(jt,y6,c);
 for(int z=threadIdx.x;z<7;z+=blockDim.x)out[z]=v[z]-c[z];__syncthreads();
#else
 osc_mm_7_6_6(x,l,b);osc_mm_7_7_6(b,j,n);
 for(int z=threadIdx.x;z<49;z+=blockDim.x)
  nt[(z%7)*7+z/7]=(z/7==z%7?REAL_C(1.):REAL_C(0.))-n[z];__syncthreads();
 osc_mm_7_1_7(nt,v,out);
#endif
 for(int z=threadIdx.x;z<7;z+=blockDim.x)O[w*7+z]=out[z];__syncthreads();
}
}
