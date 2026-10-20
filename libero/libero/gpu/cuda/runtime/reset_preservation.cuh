#include "precision.cuh"
#pragma once
#include <stdint.h>
namespace reset_preservation {
template<class T> __device__ __forceinline__ void save(int w,const int* reset,const T* source,T* saved,int width){
    if(reset[w]!=0)return;
    libero_vec::copy_block(source+size_t(w)*width,saved+size_t(w)*width,size_t(width));
}
template<class T> __device__ __forceinline__ void restore(int w,const int* reset,T* target,const T* saved,int width){
    if(reset[w]!=0)return;
    libero_vec::copy_block(saved+size_t(w)*width,target+size_t(w)*width,size_t(width));
}
__device__ __forceinline__ void save_f64(int w,const int* m,const real_t* s,real_t* b,int n){save(w,m,s,b,n);}
__device__ __forceinline__ void save_i32(int w,const int* m,const int* s,int* b,int n){save(w,m,s,b,n);}
__device__ __forceinline__ void save_i64(int w,const int* m,const int64_t* s,int64_t* b,int n){save(w,m,s,b,n);}
__device__ __forceinline__ void restore_f64(int w,const int* m,real_t* s,const real_t* b,int n){restore(w,m,s,b,n);}
__device__ __forceinline__ void restore_i32(int w,const int* m,int* s,const int* b,int n){restore(w,m,s,b,n);}
__device__ __forceinline__ void restore_i64(int w,const int* m,int64_t* s,const int64_t* b,int n){restore(w,m,s,b,n);}
}