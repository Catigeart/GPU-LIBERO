#include "precision.cuh"
#pragma once
namespace libero_scene {
// Fixed-mode scene transforms consume body matrices. Masked worlds retain prior outputs.
__device__ void camera(int w,int nb,int nc,const real_t* bp,const real_t* bm,const int* mask,
 const int* body,const real_t* lp,const real_t* lm,real_t* op,real_t* om){
 if(mask[w]==0)return;for(int c=threadIdx.x;c<nc;c+=blockDim.x){int b=w*nb+body[c],o=w*nc+c;
  for(int i=0;i<3;++i){real_t p=bp[3*b+i];for(int k=0;k<3;++k)p+=bm[9*b+3*i+k]*lp[3*c+k];op[3*o+i]=p;
   for(int j=0;j<3;++j){real_t v=0;for(int k=0;k<3;++k)v+=bm[9*b+3*i+k]*lm[9*c+3*k+j];om[9*o+3*i+j]=v;}
  }
 }
}
__device__ void light(int w,int nb,int nl,const real_t* bp,const real_t* bm,const int* mask,
 const int* body,const real_t* lp,const real_t* ld,real_t* op,real_t* od){
 if(mask[w]==0)return;for(int l=threadIdx.x;l<nl;l+=blockDim.x){int b=w*nb+body[l],o=w*nl+l;
  for(int i=0;i<3;++i){real_t p=bp[3*b+i],d=0;for(int k=0;k<3;++k){p+=bm[9*b+3*i+k]*lp[3*l+k];d+=bm[9*b+3*i+k]*ld[3*l+k];}op[3*o+i]=p;od[3*o+i]=d;}
 }
}
} // namespace libero_scene

namespace libero_scene {
__device__ __forceinline__ bool same_bits(real_t a,real_t b){return real_bits(a)==real_bits(b);}
// Only pairs wholly inside this record are accessed. The common helpers test
// the actual address, since 3/9-double record strides alternate alignment.
template<int N> __device__ __forceinline__ bool same_record(const real_t* a,const real_t* b){
 for(int i=0;i+1<N;i+=2){real2_t x=libero_vec::load_pair(a+i),y=libero_vec::load_pair(b+i);
  if(!same_bits(x.x,y.x)||!same_bits(x.y,y.y))return false;}
 return !(N&1)||same_bits(a[N-1],b[N-1]);
}
template<int N> __device__ __forceinline__ void load_record(const real_t* p,real_t (&v)[N]){
 for(int i=0;i+1<N;i+=2){real2_t q=libero_vec::load_pair(p+i);v[i]=q.x;v[i+1]=q.y;}
 if(N&1)v[N-1]=p[N-1];
}
__device__ void camera_cached(int w,int nb,int nc,const real_t* bp,const real_t* bm,const int* mask,
 const int* body,const real_t* lp,const real_t* lm,real_t* op,real_t* om,
 real_t* snapshot,int* oldbody,int* ready,int* reused,int* computed){
 if(!mask[w])return;
 for(int c=threadIdx.x;c<nc;c+=blockDim.x){
  int o=w*nc+c,b=w*nb+body[c];real_t* s=snapshot+o*36;
  bool hit=ready[o]&&oldbody[o]==body[c];
  hit=hit&&same_record<3>(s,bp+3*b);
  hit=hit&&same_record<9>(s+3,bm+9*b);
  hit=hit&&same_record<3>(s+12,lp+3*c);
  hit=hit&&same_record<9>(s+15,lm+9*c);
  // Also check retained output: unknown writes/continuation never fool the cache.
  hit=hit&&same_record<3>(s+24,op+3*o);
  hit=hit&&same_record<9>(s+27,om+9*o);
  if(hit){reused[o]+=1;continue;}
  real_t bpos[3],bmat[9],lpos[3],lmat[9],pos[3],mat[9];
  load_record(bp+3*b,bpos);load_record(bm+9*b,bmat);load_record(lp+3*c,lpos);load_record(lm+9*c,lmat);
  for(int i=0;i<3;++i){real_t p=bpos[i];
   for(int k=0;k<3;++k)p+=bmat[3*i+k]*lpos[k];pos[i]=p;
   for(int j=0;j<3;++j){real_t v=0;for(int k=0;k<3;++k)v+=bmat[3*i+k]*lmat[3*k+j];mat[3*i+j]=v;}
  }
  libero_vec::copy_serial(pos,op+3*o,3);libero_vec::copy_serial(mat,om+9*o,9);
  libero_vec::copy_serial(bp+3*b,s,3);libero_vec::copy_serial(lp+3*c,s+12,3);libero_vec::copy_serial(op+3*o,s+24,3);
  libero_vec::copy_serial(bm+9*b,s+3,9);libero_vec::copy_serial(lm+9*c,s+15,9);libero_vec::copy_serial(om+9*o,s+27,9);
  oldbody[o]=body[c];ready[o]=1;computed[o]+=1;
 }
}
__device__ void light_cached(int w,int nb,int nl,const real_t* bp,const real_t* bm,const int* mask,
 const int* body,const real_t* lp,const real_t* ld,real_t* op,real_t* od,
 real_t* snapshot,int* oldbody,int* ready,int* reused,int* computed){
 if(!mask[w])return;
 for(int c=threadIdx.x;c<nl;c+=blockDim.x){
  int o=w*nl+c,b=w*nb+body[c];real_t* s=snapshot+o*24;
  bool hit=ready[o]&&oldbody[o]==body[c];
  hit=hit&&same_record<3>(s,bp+3*b);
  hit=hit&&same_record<9>(s+3,bm+9*b);
  hit=hit&&same_record<3>(s+12,lp+3*c)&&same_record<3>(s+15,ld+3*c);
  hit=hit&&same_record<3>(s+18,op+3*o)&&same_record<3>(s+21,od+3*o);
  if(hit){reused[o]+=1;continue;}
  real_t bpos[3],bmat[9],lpos[3],ldir[3],pos[3],dir[3];
  load_record(bp+3*b,bpos);load_record(bm+9*b,bmat);load_record(lp+3*c,lpos);load_record(ld+3*c,ldir);
  for(int i=0;i<3;++i){real_t p=bpos[i],d=0;
   for(int k=0;k<3;++k){p+=bmat[3*i+k]*lpos[k];d+=bmat[3*i+k]*ldir[k];}
   pos[i]=p;dir[i]=d;
  }
  libero_vec::copy_serial(pos,op+3*o,3);libero_vec::copy_serial(dir,od+3*o,3);
  libero_vec::copy_serial(bp+3*b,s,3);libero_vec::copy_serial(lp+3*c,s+12,3);libero_vec::copy_serial(ld+3*c,s+15,3);
  libero_vec::copy_serial(op+3*o,s+18,3);libero_vec::copy_serial(od+3*o,s+21,3);libero_vec::copy_serial(bm+9*b,s+3,9);
  oldbody[o]=body[c];ready[o]=1;computed[o]+=1;
 }
}
}
