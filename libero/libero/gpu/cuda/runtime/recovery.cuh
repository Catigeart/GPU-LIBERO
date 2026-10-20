#include "precision.cuh"
#pragma once
#include <string.h>
namespace cf_recovery {
__device__ void check(int w,const real_t* values,const int* active,int* reset,int* reason,int* index,int* error,int width,int warning){
 // Only the first full warp participates; the caller supplies stage barriers.
 if(libero_warp::warp()!=0)return;
 const int lane=libero_warp::lane(), enabled=active[w];
 if(lane==0)reset[w]=0;
 if(enabled!=0&&enabled!=1){if(lane==0&&!error[w])error[w]=1;return;}
 if(!enabled)return;
 int bad=libero_warp::no_index;
 const real_t* source=values+size_t(w)*width;
 for(int i=2*lane;i<width;i+=64){
  const real2_t pair=i+1<width?libero_vec::load_pair(source+i):make_real2(source[i],REAL_C(0.0));
  if(isnan(pair.x)||pair.x>REAL_C(1e10)||pair.x<-REAL_C(1e10)){bad=i;break;}
  if(i+1<width&&(isnan(pair.y)||pair.y>REAL_C(1e10)||pair.y<-REAL_C(1e10))){bad=i+1;break;}
 }
 bad=libero_warp::min(bad);
 if(lane==0&&bad!=libero_warp::no_index){reset[w]=1;reason[w]=warning;index[w]=bad;}
}
template<class T> __device__ void reset(int w,T* dst,const T* defaults,const int* mask,int width){
 if(mask[w]!=1)return;
 dst+=size_t(w)*width;
 // Broadcast source has no world offset. Even when the two residues differ,
 // pack complete valid records and independently select each wide access.
 if(((reinterpret_cast<size_t>(defaults)^reinterpret_cast<size_t>(dst))&15)==0){
  libero_vec::copy_block(defaults,dst,size_t(width));return;
 }
 constexpr int pack=16/int(sizeof(T));
 const int packets=width/pack;
 for(int packet=int(threadIdx.x);packet<packets;packet+=int(blockDim.x)){
  const int at=packet*pack;uint4 bits;
  if(libero_vec::aligned16(defaults+at))bits=*reinterpret_cast<const uint4*>(defaults+at);
  else for(int k=0;k<pack;++k)memcpy(reinterpret_cast<unsigned char*>(&bits)+k*sizeof(T),defaults+at+k,sizeof(T));
  if(libero_vec::aligned16(dst+at))*reinterpret_cast<uint4*>(dst+at)=bits;
  else for(int k=0;k<pack;++k)memcpy(dst+at+k,reinterpret_cast<const unsigned char*>(&bits)+k*sizeof(T),sizeof(T));
 }
 for(int i=packets*pack+int(threadIdx.x);i<width;i+=int(blockDim.x))dst[i]=defaults[i];
}
__device__ void reset_f64(int w,real_t*d,const real_t*s,const int*m,int n){reset(w,d,s,m,n);}
__device__ void reset_i32(int w,int*d,const int*s,const int*m,int n){reset(w,d,s,m,n);}
__device__ void reset_i64(int w,int64_t*d,const int64_t*s,const int*m,int n){reset(w,d,s,m,n);}
__device__ void warnings(int w,int* number,int* last,const int* mask,const int* reason,const int* index){
 if(mask[w]!=1)return;for(int i=0;i<8;i++){number[w*8+i]=0;last[w*8+i]=0;}number[w*8+reason[w]]=1;last[w*8+reason[w]]=index[w];
}
__device__ void protocol(int w,const int* mask,int* first,int64_t* epoch,int64_t* expected,int* fv,int* valid,int64_t* events){
 if(mask[w]!=1||first[w])return;
 if(epoch[w]==INT64_MAX||expected[w]==INT64_MAX){first[w]=9;return;}
 epoch[w]++;expected[w]++;fv[w]=0;valid[w]=0;events[w]++;
}
__device__ void admit(int w,const int* intent,const int* first,int* active){active[w]=first[w]?0:intent[w];}
__device__ void acceleration(int w,const int* intent,const int* first,const int* fv,const int64_t* fw,const int64_t* epoch,const int* nv,const int64_t* nw,const int64_t* ne,int* active){
 active[w]=(intent[w]==1&&first[w]==0&&fv[w]==1&&fw[w]==epoch[w]&&nv[w]==1&&nw[w]==ne[w]);
}
__device__ void select(int w,const int* active,const real_t* qa,const real_t* candidate,const int64_t* ca,const int* completed,const int* origin,const int* stage,const int64_t* attempt,const int64_t* written,const int64_t* expected,const int64_t* state,const int64_t* forward,int* count,int* slot,int* overflow,int64_t* metadata,int capacity,int nv){
  if(libero_warp::warp()!=0)return;
  const int lane=libero_warp::lane();if(lane==0)slot[w]=-1;
  if(active[w]!=1)return;
  // This gate qualifies diagnostic event recording only. The subsequent qacc
  // recovery check still validates the live vector even for ineligible events.
  // No cached result may cross a reset / forward / integration write.
  if(completed[w]!=1||origin[w]!=1||ca[w]!=attempt[w])return;
  qa+=w*nv;candidate+=w*nv;
  int bad=libero_warp::no_index;bool different=false;
  for(int j=2*lane;j<nv;j+=64){
   const real2_t v=j+1<nv?libero_vec::load_pair(qa+j):make_real2(qa[j],REAL_C(0.0));
   const real2_t c=j+1<nv?libero_vec::load_pair(candidate+j):make_real2(candidate[j],REAL_C(0.0));
   different|=(v.x!=c.x);if(isnan(v.x)||v.x>REAL_C(1e10)||v.x<-REAL_C(1e10)){if(j<bad)bad=j;}
   if(j+1<nv){different|=(v.y!=c.y);if(isnan(v.y)||v.y>REAL_C(1e10)||v.y<-REAL_C(1e10)){if(j+1<bad)bad=j+1;}}
  }
  bad=libero_warp::min(bad);const int same=!libero_warp::any(different);
  if(lane!=0||bad==libero_warp::no_index)return;int row=count[w];if(row>=capacity){overflow[w]=1;return;}
  slot[w]=row;count[w]=row+1;int64_t* m=metadata+(w*capacity+row)*9;m[0]=bad;m[1]=attempt[w];m[2]=written[w];m[3]=expected[w];m[4]=state[w];m[5]=forward[w];m[6]=stage[w];m[7]=ca[w];m[8]=same;
}
}
