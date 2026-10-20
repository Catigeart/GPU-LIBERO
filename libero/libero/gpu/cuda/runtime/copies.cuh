#include "precision.cuh"
#pragma once
#ifndef LIBERO_CF_COPY_PAYLOAD_CUH
#define LIBERO_CF_COPY_PAYLOAD_CUH
namespace cf_copy {
// Integer vector payload preserves raw bits. Test alignment after world/slot.
template<class T> __device__ void payload(const T* src,T* dst,int width) {
 libero_vec::copy_block(src,dst,size_t(width));
}
// Field access units share the tile, including actual-address scalar edges.
template<class T> __device__ int tile_field(int w,const T* src,T* dst,int width,int rows,const int* slot,int begin) {
 const int row=slot?slot[w]:0;if(row<0||row>=rows)return 0;
 src+=size_t(w)*width;dst+=(size_t(w)*rows+row)*width;
 const size_t a=reinterpret_cast<size_t>(src),b=reinterpret_cast<size_t>(dst);
 const int pack=16/int(sizeof(T));
 const bool congruent=((a^b)&15)==0;
 int head=congruent?int(((size_t(0)-a)&15)/sizeof(T)):width;
 if(head>width)head=width;
 const int vectors=(width-head)/pack,tail=width-head-vectors*pack;
 const int units=head+vectors+tail;
 const int first=(int(threadIdx.x)+int(blockDim.x)-begin%int(blockDim.x))%int(blockDim.x);
 for(int unit=first;unit<units;unit+=blockDim.x) {
  if(unit<head)dst[unit]=src[unit];
  else if(unit<head+vectors) {
   const int at=head+(unit-head)*pack;
   *reinterpret_cast<uint4*>(dst+at)=*reinterpret_cast<const uint4*>(src+at);
  } else {
   const int at=head+vectors*pack+unit-head-vectors;
   dst[at]=src[at];
  }
 }
 return units;
}

template<class T> __device__ void field(int w,const T* src,T* dst,int width,int rows,const int* slot) {
  int row=slot?slot[w]:0;
  if(row<0||row>=rows)return;
  src+=size_t(w)*width;
  dst+=(size_t(w)*rows+row)*width;
  payload(src,dst,width);
}
__device__ void f64(int w,const real_t*s,real_t*t,int n,int r,const int*i){field(w,s,t,n,r,i);}
__device__ void i32(int w,const int*s,int*t,int n,int r,const int*i){field(w,s,t,n,r,i);}
__device__ void i64(int w,const int64_t*s,int64_t*t,int n,int r,const int*i){field(w,s,t,n,r,i);}
template<class T> __device__ void hold(int w,const int* mask,const T*s,T*t,int n){
 if(mask[w])return;
 payload(s+size_t(w)*n,t+size_t(w)*n,n);
}
__device__ void hold_f64(int w,const int*m,const real_t*s,real_t*t,int n){hold(w,m,s,t,n);}
__device__ void hold_i32(int w,const int*m,const int*s,int*t,int n){hold(w,m,s,t,n);}
__device__ void hold_i64(int w,const int*m,const int64_t*s,int64_t*t,int n){hold(w,m,s,t,n);}
__device__ void advance(int w,int* index,int* overflow,int capacity){if(index[w]<capacity)index[w]++;else overflow[w]=1;}
}

#endif // LIBERO_CF_COPY_PAYLOAD_CUH
