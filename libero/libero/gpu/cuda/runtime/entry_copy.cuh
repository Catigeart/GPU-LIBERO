#include "precision.cuh"
#pragma once
namespace cf_entry_copy {
template<class T> __device__ void field(int w,const T*src,T*dst,int width){
 for(int i=threadIdx.x;i<width;i+=blockDim.x)dst[w*width+i]=src[w*width+i];
}
__device__ void f64(int w,const real_t*s,real_t*t,int n){field(w,s,t,n);}
// Separate positive vector route, selected by cold host binding only.
// Default f64 above stays byte-identical to the pre-vector function.
template<class T> __device__ void field_vector(int w,const T*src,T*dst,int width){
 libero_vec::copy_block(src+size_t(w)*width,dst+size_t(w)*width,size_t(width));
}
__device__ void f64_vector(int w,const real_t*s,real_t*t,int n){field_vector(w,s,t,n);}
// Preserve the unmeasured integer R07 route.
__device__ void i32(int w,const int32_t*s,int32_t*t,int n){field_vector(w,s,t,n);}
__device__ void i64(int w,const int64_t*s,int64_t*t,int n){field_vector(w,s,t,n);}
}
