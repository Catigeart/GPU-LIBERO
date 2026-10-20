#include "precision.cuh"
#pragma once
#ifndef OSC_MASS_RECEIPT
#define OSC_MASS_RECEIPT 0
#endif
namespace osc_component {
__device__ inline void mass(int w,const real_t* M,const real_t* J,real_t* X,real_t* A,int* status,int direct,real_t* MI){
 __shared__ real_t lu[49],mi[49],j[42],jt[42],jw[42],x[42],a[36];
 __shared__ int piv[7],info;
 for(int z=threadIdx.x;z<49;z+=blockDim.x){lu[z]=M[w*49+z];mi[z]=z/7==z%7?1:0;}
 for(int z=threadIdx.x;z<42;z+=blockDim.x){j[z]=J[w*42+z];jt[(z%7)*6+z/7]=j[z];}
 if(threadIdx.x==0)info=0;__syncthreads();
 if(direct){for(int z=threadIdx.x;z<42;z+=blockDim.x)x[z]=jt[z];__syncthreads();osc_solve7x6(lu,x,piv,&info);}
 else osc_inverse7(lu,mi,piv,&info);
 if(threadIdx.x==0)status[w]=info;__syncthreads();if(info)return;
 if(direct)osc_mm_6_6_7(j,x,a);
 else{osc_mm_6_7_7(j,mi,jw);osc_mm_6_6_7(jw,jt,a);osc_mm_7_6_7(mi,jt,x);}
 for(int z=threadIdx.x;z<42;z+=blockDim.x)X[w*42+z]=x[z];
 for(int z=threadIdx.x;z<36;z+=blockDim.x)A[w*36+z]=a[z];__syncthreads();
 // Validation-only receipt; both performance variants compile this store out.
 if(OSC_MASS_RECEIPT&&!direct)for(int z=threadIdx.x;z<49;z+=blockDim.x)MI[w*49+z]=mi[z];__syncthreads();
}
__device__ inline void tasks(int w,const real_t* A,real_t* L,int* R,int* O,int* stats,int candidate){
 __shared__ real_t a[36],ap[36],ar[36],lf[36],lp[36],lr[36];__shared__ int r[3],ok[3];
 for(int z=threadIdx.x;z<36;z+=blockDim.x){a[z]=A[w*36+z];ap[z]=ar[z]=0;}__syncthreads();
 if(threadIdx.x==0)for(int i=0;i<3;i++)for(int k=0;k<3;k++){ap[i*6+k]=a[i*6+k];ar[i*6+k]=a[(i+3)*6+k+3];}__syncthreads();
 if(candidate)osc_cuda::operator_tasks(w,a,ap,ar,lf,lp,lr,r,ok,stats);
 else{if(threadIdx.x==0){osc_cuda::pinv(a,6,lf,r[0],ok[0]);osc_cuda::pinv(ap,3,lp,r[1],ok[1]);osc_cuda::pinv(ar,3,lr,r[2],ok[2]);}__syncthreads();}
 for(int z=threadIdx.x;z<36;z+=blockDim.x)L[w*54+z]=lf[z];
 for(int z=threadIdx.x;z<9;z+=blockDim.x){int i=z/3,k=z%3;L[w*54+36+z]=lp[i*6+k];L[w*54+45+z]=lr[i*6+k];}
 if(threadIdx.x==0)for(int z=0;z<3;z++){R[w*3+z]=r[z];O[w*3+z]=ok[z];}__syncthreads();
}
}
