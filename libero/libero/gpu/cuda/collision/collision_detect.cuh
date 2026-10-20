#include "precision.cuh"
// Collective world detector: parallel convex queries, ordered public commit.
#pragma once
#include <stdint.h>
#include "../common/warp.cuh"
#include "collision_detect_primitive.cuh"
namespace libero_collision_cuda {
struct Params {
 const int *call_g1,*call_g2,*call_type1,*call_type2;
 const real_t* call_margin; const int *call_count,*active;
 const real_t *xpos,*xmat,*geom_size,*geom_rbound;
 const int* geom_dataid; const float* mesh_vert;
 const int *mesh_vertadr,*mesh_vertnum,*mesh_graphadr,*mesh_graph,*route_id;
 real_t* raw; int *raw_count,*det_status,*scratch_i;
 real_t* native_raw; int *native_count,*native_status,*native_diag;
 real_t* native_frame0; int *native_framewritten,*fe_stage,*fe_code,*fe_item;
 int64_t* cuda_detector_execution_count;
 real_t* primitive_raw; int *primitive_count,*primitive_status;
 int *query_order,*query_tmp,*bucket_offsets;
 int *task_bucket,*task_start,*task_count;
 const int* mesh_edge_end;
 const int* route_subgroup_widths; // Reserved cold ABI; this variant only admits0/32.
 int nworld,pairs,ngeom,nmesh,nvert,ngraph,mpr_iterations,multiccd;
 int diagnostic_profile;
 int query_subgroup_width;
 int box_private_capacity; // Reserved cold ABI; producer must admit exactly50.
 real_t mpr_tolerance;
};
// BoxBox's clipping/control body is scalar. A lane owns the whole query in
// production; compatibility retains the historical collective diagnostics.
__device__ __forceinline__ bool detect_collective(const Params& p,int bucket) {
 return p.query_subgroup_width==32 && (bucket==5||bucket==11||bucket==12||bucket==13)
     && !(p.diagnostic_profile!=0 && bucket==12);
}
__device__ void error(Params& p,int w,int j,int status,int code) {
 p.det_status[w*p.pairs+j]=status;p.fe_stage[w]=6;p.fe_code[w]=code;p.fe_item[w]=j;
}
template<int Capacity> __device__ __forceinline__ void contact_load_n(mjContact* con,const real_t* raw) {
 for(int k=0;k<Capacity;k++) {
  // Five complete pairs belong to this raw10 record; no capacity tail read.
  const real2_t a=libero_vec::load_pair(raw+10*k);
  const real2_t b=libero_vec::load_pair(raw+10*k+2);
  con[k].dist=a.x;con[k].pos[0]=a.y;con[k].pos[1]=b.x;con[k].pos[2]=b.y;
  for(int h=0;h<6;h+=2){const real2_t v=libero_vec::load_pair(raw+10*k+4+h);con[k].frame[h]=v.x;con[k].frame[h+1]=v.y;}
  for(int h=6;h<9;h++)con[k].frame[h]=0;
 }
}
__device__ __forceinline__ void contact_load(mjContact* con,const real_t* raw) {contact_load_n<50>(con,raw);}
__device__ void contact_store(real_t* raw,const mjContact* con,int n) {
 for(int k=0;k<n;k++) {
  libero_vec::store_pair(raw+10*k,con[k].dist,con[k].pos[0]);
  libero_vec::store_pair(raw+10*k+2,con[k].pos[1],con[k].pos[2]);
  for(int h=0;h<6;h+=2)libero_vec::store_pair(raw+10*k+4+h,con[k].frame[h],con[k].frame[h+1]);
 }
}
__device__ bool prepare_query(Params& p,int w,int slot,mjModel& m,mjData& d,
 real_t* sz,real_t* rb,int* types,int* ids,real_t* pos,real_t* mat,
 int* err,int* trace,int* fw) {
 int a=p.call_g1[slot],b=p.call_g2[slot];
 if(a<0||b<0||a>=p.ngeom||b>=p.ngeom||a==b){*err=4;return false;}
 types[0]=p.call_type1[slot];types[1]=p.call_type2[slot];
 ids[0]=p.geom_dataid[a];ids[1]=p.geom_dataid[b];
 rb[0]=p.geom_rbound[a];rb[1]=p.geom_rbound[b];
 for(int i=0;i<2;++i){
  const int g=i?b:a;
  const real2_t size01=libero_vec::load_pair(p.geom_size+3*g);
  sz[3*i]=size01.x;sz[3*i+1]=size01.y;sz[3*i+2]=p.geom_size[3*g+2];
  const real2_t pos01=libero_vec::load_pair(p.xpos+3*(w*p.ngeom+g));
  pos[3*i]=pos01.x;pos[3*i+1]=pos01.y;pos[3*i+2]=p.xpos[3*(w*p.ngeom+g)+2];
  for(int k=0;k<8;k+=2){const real2_t v=libero_vec::load_pair(p.xmat+9*(w*p.ngeom+g)+k);mat[9*i+k]=v.x;mat[9*i+k+1]=v.y;}
  mat[9*i+8]=p.xmat[9*(w*p.ngeom+g)+8];
 }
 for(int k=0;k<6;k++)if(!isfinite(sz[k])||!isfinite(pos[k])){*err=5;return false;}
 for(int k=0;k<18;k++)if(!isfinite(mat[k])){*err=5;return false;}
 if(!isfinite(p.call_margin[slot])||p.mpr_iterations<0||!isfinite(p.mpr_tolerance)||p.mpr_tolerance<0){*err=5;return false;}
 for(int k=0;k<2;k++)if(types[k]==7&&ids[k]>=0) {
  int id=ids[k];
  if(id>=p.nmesh||p.mesh_vertadr[id]<0||p.mesh_vertnum[id]<=0||
     3LL*(p.mesh_vertadr[id]+(long long)p.mesh_vertnum[id])>p.nvert){*err=7;return false;}
  int ga=p.mesh_graphadr[id];
  if(ga>=0&&(ga>=p.ngraph||p.mesh_graph[ga]<=0||ga+2+2LL*p.mesh_graph[ga]>p.ngraph)){*err=7;return false;}
 }
 m={sz,rb,types,ids,(float*)p.mesh_vert,(int*)p.mesh_vertadr,(int*)p.mesh_vertnum,
    (int*)p.mesh_graphadr,(int*)p.mesh_graph,{p.mpr_iterations,p.multiccd?1:0,p.mpr_tolerance},
    err,trace,fw,p.nvert,p.ngraph,p.nmesh,4096,0,p.mesh_edge_end};d={pos,mat};return true;
}
template<int Capacity,int FixedRoute=0> __device__ __noinline__ void native_query(Params& p,int w,int slot,int route) {
 if constexpr(FixedRoute!=0)route=FixedRoute;
 real_t* out=p.native_raw+(size_t)slot*500;
 const real_t* prior=p.raw+(size_t)slot*500;
 real_t sz[6],rb[2],pos[6],mat[18];int types[2],ids[2],err=0,fw=0,trace[16]={0};
 mjModel m;mjData d;
 if(!prepare_query(p,w,slot,m,d,sz,rb,types,ids,pos,mat,&err,trace,&fw)){p.native_status[slot]=err;return;}
 for(int k=0;k<2;k++)if(types[k]==7&&ids[k]<0){p.native_status[slot]=7;return;}
 mjContact con[Capacity];
 // BoxBox writes every published field, including the zero-contact frame.
 // Its unobserved private capacity tail does not require a history mirror.
 if(p.diagnostic_profile==0||route!=12)contact_load_n<Capacity>(con,prior);
 int n=route==12?mjc_BoxBox(&m,&d,con,0,1,p.call_margin[slot]):
       route==13?mjc_CapsuleBox(&m,&d,con,0,1,p.call_margin[slot]):
                  mjc_Convex(&m,&d,con,0,1,p.call_margin[slot]);
 if(route==13)fw=n>0;
 trace[4]=n;trace[5]=fw;
 for(int k=0;k<16;k++)p.native_diag[16*slot+k]=trace[k];
 if(err||n<0||n>Capacity){p.native_status[slot]=err?err:8;return;}
 for(int k=0;k<n;k++) {
  if(!isfinite(con[k].dist)){p.native_status[slot]=5;return;}
  for(int h=0;h<3;h++)if(!isfinite(con[k].pos[h])){p.native_status[slot]=5;return;}
  for(int h=0;h<6;h++)if(!isfinite(con[k].frame[h])){p.native_status[slot]=5;return;}
 }
 if(fw) {
  for(int h=0;h<6;h++)if(!isfinite(con[0].frame[h])){p.native_status[slot]=5;return;}
  for(int h=0;h<6;h++)p.native_frame0[6*slot+h]=con[0].frame[h];
  p.native_framewritten[slot]=1;
 }
 contact_store(out,con,n);p.native_count[slot]=n;
}
// Scalar alternate entry retains its own rawbits mirror. Production prepare
// performs that mirror collectively before initializing per-slot metadata.
__device__ bool detect_native_init(Params& p,int w,int j,bool copy_raw=true) {
 if(p.diagnostic_profile!=0&&(p.fe_code[w]||p.active[w]!=1||
    p.call_count[w]<0||p.call_count[w]>p.pairs||j>=p.call_count[w]))return false;
 int slot=w*p.pairs+j;real_t* out=p.native_raw+(size_t)slot*500;
 const real_t* prior=p.raw+(size_t)slot*500;
 // Compatibility exposes the complete native mirror, including unused slots.
 // Fast diagnostics expose only detected prefixes; public physics raw is unchanged.
 if(copy_raw&&p.diagnostic_profile==0)
  libero_vec::copy_serial(prior,out,500);
 p.native_count[slot]=0;p.native_status[slot]=0;p.native_framewritten[slot]=0;
 libero_vec::fill_serial(p.native_diag+16*slot,16,0);
 if(p.active[w]==0)return false;
 if(p.active[w]!=1){p.native_status[slot]=2;return false;}
 if(p.call_count[w]<0||p.call_count[w]>p.pairs){p.native_status[slot]=3;return false;}
 return j<p.call_count[w];
}
__device__ void detect_native_slot(Params& p,int w,int j) {
 if(!detect_native_init(p,w,j))return;
 const int slot=w*p.pairs+j;
 int t1=p.call_type1[slot],t2=p.call_type2[slot];
 if(t1<0||t1>7||t2<0||t2>7)return;
 int route=p.route_id[8*t1+t2];if(route!=11&&route!=12&&route!=13)return;
 if(route==13)native_query<8>(p,w,slot,route);
 else native_query<50>(p,w,slot,route);
}
__device__ __forceinline__ int detect_bucket(Params& p,int slot) {
 const int t1=p.call_type1[slot],t2=p.call_type2[slot];
 if(t1<0||t1>7||t2<0||t2>7)return 0;
 const int route=p.route_id[8*t1+t2];
 return route>=1&&route<=13?route:(route==0?0:14);
}
// Stable four-bit radix schedule, original local call index remains the key
// payload. Exactly 64 block threads enter, matching both production launchers.
__device__ void detect_prepare_routes(Params& p,int w) {
 __shared__ int hist[15],warp_zeros[2],processed_zeros,processed_ones;
 for(int r=threadIdx.x;r<15;r+=64)hist[r]=0;
 __syncthreads();
 // All lanes enter the rawbits mirror stage, including inactive/unused slots.
 // Fast diagnostics retain their original no-mirror behavior. This completes
 // before any query owner reads or overwrites its private native output.
 if(p.diagnostic_profile==0) {
  const size_t base=(size_t)w*p.pairs*500;
  const size_t words=(size_t)p.pairs*500;
  libero_vec::copy_block(p.raw+base,p.native_raw+base,words);
 }
 const int initialize_count=p.diagnostic_profile!=0?
     ((p.active[w]==1&&!p.fe_code[w]&&p.call_count[w]>=0&&p.call_count[w]<=p.pairs)?p.call_count[w]:0):p.pairs;
 for(int j=threadIdx.x;j<initialize_count;j+=64)detect_native_init(p,w,j,false);
 const int count=(p.active[w]==1&&p.call_count[w]>=0&&p.call_count[w]<=p.pairs&&
                  !(p.diagnostic_profile!=0&&p.fe_code[w]))?p.call_count[w]:0;
 int* const offsets=p.bucket_offsets+16*w;
 int* in=p.query_order+(size_t)w*p.pairs;
 int* out=p.query_tmp+(size_t)w*p.pairs;
 for(int j=threadIdx.x;j<count;j+=64){in[j]=j;atomicAdd(hist+detect_bucket(p,w*p.pairs+j),1);}
 __syncthreads();
 if(threadIdx.x==0){offsets[0]=0;for(int r=0;r<15;r++)offsets[r+1]=offsets[r]+hist[r];}
 __syncthreads();
 for(int bit=0;bit<4;bit++) {
  int total_zeros=0;for(int r=0;r<15;r++)if(!(r&(1<<bit)))total_zeros+=hist[r];
  if(threadIdx.x==0){processed_zeros=0;processed_ones=0;}
  __syncthreads();
  for(int tile=0;tile<count;tile+=64) {
   const int i=tile+threadIdx.x;
   const bool valid=i<count;
   const int j=valid?in[i]:0;
   const bool zero=valid&&!(detect_bucket(p,w*p.pairs+j)&(1<<bit));
   const unsigned vote=__ballot_sync(0xffffffffu,zero);
   const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
   if(lane==0)warp_zeros[warp]=__popc(vote);
   __syncthreads();
   const int zero_rank=__popc(vote&((1u<<lane)-1))+(warp?warp_zeros[0]:0);
   if(valid){
    const int dest=zero?processed_zeros+zero_rank:
       total_zeros+processed_ones+int(threadIdx.x)-zero_rank;
    out[dest]=j;
   }
   __syncthreads();
   if(threadIdx.x==0){const int nz=warp_zeros[0]+warp_zeros[1];
    processed_zeros+=nz;processed_ones+=(count-tile<64?count-tile:64)-nz;}
   __syncthreads();
  }
  int* swap=in;in=out;out=swap;
 }
 // Four passes finish in query_order; no copy or public permutation is needed.
}
// Compact device queue contains only nonempty route tiles. Each task has a
// unique original-slot range; producer completion precedes ordered commit.
__device__ void detect_prepare_compact(Params& p,int w) {
 if(threadIdx.x==0)atomicAdd((unsigned long long*)(p.cuda_detector_execution_count+w),1ULL);
 detect_prepare_routes(p,w);
 if(threadIdx.x==0) {
  const size_t base=(size_t)w*((p.pairs+1)/2+15);
  const int* offsets=p.bucket_offsets+16*w;
  int count=0;
  for(int bucket=0;bucket<15;bucket++) {
   const bool collective=detect_collective(p,bucket);
   const int width=collective?2:64;
   for(int start=offsets[bucket];start<offsets[bucket+1];start+=width) {
    p.task_bucket[base+count]=bucket;p.task_start[base+count]=start;count++;
   }
  }
  p.task_count[w]=count;
 }
 __syncthreads();
}
// Primitive routes emit at most eight contacts (PlaneConvex maxplanemesh=3).
// Split their small stack frame from the native convex work state.
template<int Capacity> __device__ __noinline__ void primitive_query(
 Params& p,int w,int slot,int route) {
 real_t sz[6],rb[2],pos[6],mat[18];int types[2],ids[2],err=0,fw=0,trace[16]={0};
 mjModel m;mjData d;
 if(!prepare_query(p,w,slot,m,d,sz,rb,types,ids,pos,mat,&err,trace,&fw)) {
  p.primitive_status[slot]=10+err;return;
 }
 // Every successful primitive contact defines dist, pos[3], frame[6].
 // Zero-contact partial frame writes are not published by the primitive ABI.
 mjContact con[Capacity];
 int n=0;real_t margin=p.call_margin[slot];
 switch(route) {
  case 1:n=mjc_PlaneSphere(&m,&d,con,0,1,margin);break;
  case 2:n=mjc_PlaneCapsule(&m,&d,con,0,1,margin);break;
  case 3:n=mjc_PlaneCylinder(&m,&d,con,0,1,margin);break;
  case 4:n=mjc_PlaneBox(&m,&d,con,0,1,margin);break;
  case 5:n=mjc_PlaneConvex(&m,&d,con,0,1,margin);break;
  case 6:n=mjc_SphereSphere(&m,&d,con,0,1,margin);break;
  case 7:n=mjc_SphereCapsule(&m,&d,con,0,1,margin);break;
  case 8:n=mjc_SphereCylinder(&m,&d,con,0,1,margin);break;
  case 9:n=mjc_SphereBox(&m,&d,con,0,1,margin);break;
  case 10:n=mjc_CapsuleCapsule(&m,&d,con,0,1,margin);break;
  default:p.primitive_status[slot]=4;return;
 }
 if(err||n<0||n>Capacity){p.primitive_status[slot]=10+(err?err:6);return;}
 // Deliberately no extra finite rejection: this preserves primitive baseline.
 contact_store(p.primitive_raw+(size_t)slot*80,con,n);
 p.primitive_count[slot]=n;
}
__device__ void detect_primitive_slot(Params& p,int w,int j) {
 if(p.fe_code[w]||p.active[w]!=1||p.call_count[w]<0||p.call_count[w]>p.pairs||j>=p.call_count[w])return;
 int slot=w*p.pairs+j,a=p.call_g1[slot],b=p.call_g2[slot];
 int t1=p.call_type1[slot],t2=p.call_type2[slot];
 if(a<0||b<0||a>=p.ngeom||b>=p.ngeom||t1<0||t1>7||t2<0||t2>7)return;
 int route=p.route_id[8*t1+t2];if(route<1||(route>=11&&route<=13))return;
 p.primitive_count[slot]=0;p.primitive_status[slot]=0;
 primitive_query<8>(p,w,slot,route);
}
// Separate collective entries: every lane of one FULL warp owns the same
// original slot. Private model/data are identical; only lane 0 publishes.
template<int Capacity,int FixedRoute=0> __device__ __noinline__ void native_query_warp(
 Params& p,int w,int slot,int route) {
 if constexpr(FixedRoute!=0)route=FixedRoute;
 const bool leader=(threadIdx.x&31)==0;
 real_t sz[6],rb[2],pos[6],mat[18];int types[2],ids[2],err=0,fw=0,trace[16]={0};
 mjModel m;mjData d;
 if(!prepare_query(p,w,slot,m,d,sz,rb,types,ids,pos,mat,&err,trace,&fw)){
  if(leader)p.native_status[slot]=err;return;
 }
 for(int k=0;k<2;k++)if(types[k]==7&&ids[k]<0){if(leader)p.native_status[slot]=7;return;}
 m.adapter_subgroup_width=route==13?0:32;
 mjContact con[Capacity];contact_load_n<Capacity>(con,p.raw+(size_t)slot*500);
 const int n=route==12?mjc_BoxBox(&m,&d,con,0,1,p.call_margin[slot]):
       route==13?mjc_CapsuleBox_warp(&m,&d,con,0,1,p.call_margin[slot]):
                 mjc_Convex(&m,&d,con,0,1,p.call_margin[slot]);
 if(!leader)return;
 if(route==13)fw=n>0;
 trace[4]=n;trace[5]=fw;for(int k=0;k<16;k++)p.native_diag[16*slot+k]=trace[k];
 if(err||n<0||n>Capacity){p.native_status[slot]=err?err:8;return;}
 for(int k=0;k<n;k++) {
  if(!isfinite(con[k].dist)){p.native_status[slot]=5;return;}
  for(int h=0;h<3;h++)if(!isfinite(con[k].pos[h])){p.native_status[slot]=5;return;}
  for(int h=0;h<6;h++)if(!isfinite(con[k].frame[h])){p.native_status[slot]=5;return;}
 }
 if(fw){for(int h=0;h<6;h++)if(!isfinite(con[0].frame[h])){p.native_status[slot]=5;return;}
  for(int h=0;h<6;h++)p.native_frame0[6*slot+h]=con[0].frame[h];
  p.native_framewritten[slot]=1;}
 contact_store(p.native_raw+(size_t)slot*500,con,n);p.native_count[slot]=n;
}
__device__ __noinline__ void primitive_query_warp(Params& p,int w,int slot) {
 if(p.fe_code[w])return;
 const bool leader=(threadIdx.x&31)==0;
 const int a=p.call_g1[slot],b=p.call_g2[slot];
 if(a<0||b<0||a>=p.ngeom||b>=p.ngeom)return;
 if(leader){p.primitive_count[slot]=0;p.primitive_status[slot]=0;}
 real_t sz[6],rb[2],pos[6],mat[18];int types[2],ids[2],err=0,fw=0,trace[16]={0};
 mjModel m;mjData d;
 if(!prepare_query(p,w,slot,m,d,sz,rb,types,ids,pos,mat,&err,trace,&fw)){
  if(leader)p.primitive_status[slot]=10+err;return;
 }
 mjContact con[8];const int n=mjc_PlaneConvex_warp(&m,&d,con,0,1,p.call_margin[slot]);
 if(!leader)return;
 if(err||n<0||n>8){p.primitive_status[slot]=10+(err?err:6);return;}
 contact_store(p.primitive_raw+(size_t)slot*80,con,n);p.primitive_count[slot]=n;
}
__device__ void detect_scheduled_query(Params& p,int w,int j,int bucket,bool collective) {
 const int slot=w*p.pairs+j;
 if(collective) {
  if(bucket==5)primitive_query_warp(p,w,slot);
  else if(bucket==13)native_query_warp<8>(p,w,slot,bucket);
  else native_query_warp<50>(p,w,slot,bucket);
 } else {
  if(bucket>=11&&bucket<=13) {
   if(bucket==13)native_query<8>(p,w,slot,bucket);else native_query<50>(p,w,slot,bucket);
  } else detect_primitive_slot(p,w,j);
 }
}
__device__ int detect_admit_world(Params& p,int w) {
 if(p.fe_code[w]||p.active[w]==0)return 0;
 if(p.active[w]!=1){error(p,w,0,1,1);p.fe_item[w]=w;return 0;}
 int count=p.call_count[w];if(count<0||count>p.pairs){error(p,w,0,2,2);return 0;}
 for(int j=0;j<count;j++) {
  int slot=w*p.pairs+j,a=p.call_g1[slot],b=p.call_g2[slot],t1=p.call_type1[slot],t2=p.call_type2[slot];
  if(a<0||b<0||a>=p.ngeom||b>=p.ngeom||t1<0||t1>7||t2<0||t2>7){error(p,w,j,8,8);return j;}
  int route=p.route_id[8*t1+t2];
  if(route==0){error(p,w,j,3,8);return j;}if(route<0){error(p,w,j,4,4);return j;}
  int n=0;
  if(route>=11&&route<=13) {
   int status=p.native_status[slot];if(status){error(p,w,j,100+status,status==6?9:8);return j;}
   n=p.native_count[slot];
   if(n<0||n>50){error(p,w,j,8,2);return j;}
   if(p.diagnostic_profile==0)for(int k=0;k<n*10;k++)if(!isfinite(p.native_raw[(size_t)slot*500+k])){error(p,w,j,8,8);return j;}


  } else {
   int status=p.primitive_status[slot];if(status){error(p,w,j,status,status==4?4:8);return j;}
   n=p.primitive_count[slot];

  }
  p.raw_count[slot]=n;
 }
 return count;
}
// Independent read-only call checks retain the original within-call priority.
// No primitive finite check is added: its producer ABI deliberately differs.
__device__ __forceinline__ bool detect_admit_call(
 Params& p,int w,int j,int& status,int& code) {
 const int slot=w*p.pairs+j,a=p.call_g1[slot],b=p.call_g2[slot];
 const int t1=p.call_type1[slot],t2=p.call_type2[slot];
 if(a<0||b<0||a>=p.ngeom||b>=p.ngeom||t1<0||t1>7||t2<0||t2>7) {
  status=8;code=8;return false;
 }
 const int route=p.route_id[8*t1+t2];
 if(route==0){status=3;code=8;return false;}
 if(route<0){status=4;code=4;return false;}
 if(route>=11&&route<=13) {
  const int native_status=p.native_status[slot];
  if(native_status){status=100+native_status;code=native_status==6?9:8;return false;}
  const int n=p.native_count[slot];
  if(n<0||n>50){status=8;code=2;return false;}
  if(p.diagnostic_profile==0)for(int k=0;k<n*10;k++)if(!isfinite(p.native_raw[(size_t)slot*500+k])) {
   status=8;code=8;return false;
  }
 } else {
  const int primitive_status=p.primitive_status[slot];
  if(primitive_status){status=primitive_status;code=primitive_status==4?4:8;return false;}
 }
 return true;
}
// All block threads enter. Full warps check independent calls, then publish
// exactly the serial successful prefix. No raw_count beyond that prefix changes.
__device__ int detect_admit_world_block(Params& p,int w) {
 __shared__ int accepted,check_calls,warp_first[32];
 // The production launcher uses 64 threads. Retain scalar compatibility for
 // alternate callers with a partial warp; full-mask collectives cannot run there.
 if(blockDim.x%32) {
  if(threadIdx.x==0)accepted=detect_admit_world(p,w);
  __syncthreads();return accepted;
 }
 if(threadIdx.x==0) {
  accepted=0;check_calls=0;
  if(!p.fe_code[w]&&p.active[w]!=0) {
   if(p.active[w]!=1){error(p,w,0,1,1);p.fe_item[w]=w;}
   else if(p.call_count[w]<0||p.call_count[w]>p.pairs){error(p,w,0,2,2);}
   else {accepted=p.call_count[w];check_calls=1;}
  }
 }
 __syncthreads();
 if(!check_calls)return accepted;
 int first=libero_warp::no_index,status=0,code=0;
 for(int j=threadIdx.x;j<accepted;j+=blockDim.x) {
  if(!detect_admit_call(p,w,j,status,code)){first=j;break;}
 }
 const int first_in_warp=libero_warp::min(first);
 if(libero_warp::lane()==0)warp_first[libero_warp::warp()]=first_in_warp;
 __syncthreads();
 if(threadIdx.x==0)
  for(int k=0;k<blockDim.x/32;k++)if(warp_first[k]<accepted)accepted=warp_first[k];
 __syncthreads();
 // Only the owner of the earliest failing call can publish the error tuple.
 if(first==accepted)error(p,w,first,status,code);
 for(int j=threadIdx.x;j<accepted;j+=blockDim.x) {
  const int slot=w*p.pairs+j;
  const int route=p.route_id[8*p.call_type1[slot]+p.call_type2[slot]];
  p.raw_count[slot]=(route>=11&&route<=13)?p.native_count[slot]:p.primitive_count[slot];
 }
 __syncthreads();return accepted;
}
// Admission identifies the ordered successful prefix before any producer writes
// become public. Blocks never synchronize across grid: tiled producer completes
// in an earlier kernel, or the collective path uses its own block barrier.
__device__ void detect_commit_world(Params& p,int w) {
 const int clear_count=p.diagnostic_profile!=0?
     ((p.call_count[w]>=0&&p.call_count[w]<=p.pairs)?p.call_count[w]:0):p.pairs;
 if(!p.fe_code[w]&&p.active[w]!=0)
  for(int j=threadIdx.x;j<clear_count;j+=blockDim.x) {
   p.raw_count[w*p.pairs+j]=0;p.det_status[w*p.pairs+j]=0;
  }
 __syncthreads();
 const int accepted=detect_admit_world_block(p,w);
 // Eight lanes share each original call record. Adjacent lanes copy adjacent
 // 64-bit words; no contact permutation or second full query cache is needed.
 // Retain a scalar mapping for alternate non-multiple-of-eight block callers.
 const bool tiled=blockDim.x>=8&&blockDim.x%8==0;
 const int lane=tiled?threadIdx.x%8:0;
 const int worker=tiled?threadIdx.x/8:threadIdx.x;
 const int workers=tiled?blockDim.x/8:blockDim.x;
 const int width=tiled?8:1;
 for(int j=worker;j<accepted;j+=workers) {
  const int slot=w*p.pairs+j,n=p.raw_count[slot];
  const int route=p.route_id[8*p.call_type1[slot]+p.call_type2[slot]];
  real_bits_t* raw=(real_bits_t*)(p.raw+(size_t)slot*500);
  if(route>=11&&route<=13) {
   const real_bits_t* source=(const real_bits_t*)(p.native_raw+(size_t)slot*500);
   for(int k=lane;k<n*10;k+=width)raw[k]=source[k];
   if(n==0&&p.native_framewritten[slot]) {
    const real_bits_t* frame=(const real_bits_t*)(p.native_frame0+6*slot);
    for(int k=lane;k<6;k+=width)raw[4+k]=frame[k];
   }
  } else {
   const real_bits_t* source=(const real_bits_t*)(p.primitive_raw+(size_t)slot*80);
   for(int k=lane;k<n*10;k+=width)raw[k]=source[k];
  }
 }
 __syncthreads();
}
// All block threads enter. The world barrier is the native/public dependency.
__device__ void detect_world(Params& p,int w) {
 if(threadIdx.x==0)atomicAdd((unsigned long long*)(p.cuda_detector_execution_count+w),1ULL);
 detect_prepare_routes(p,w);
 const int* offsets=p.bucket_offsets+16*w;
 for(int bucket=0;bucket<15;bucket++) {
  const bool collective=detect_collective(p,bucket);
  const int worker=collective?threadIdx.x/32:threadIdx.x;
  const int stride=collective?2:64;
  for(int q=offsets[bucket]+worker;q<offsets[bucket+1];q+=stride) {
   const int j=p.query_order[(size_t)w*p.pairs+q];
   detect_scheduled_query(p,w,j,bucket,collective);
  }
 }
 __syncthreads();detect_commit_world(p,w);
}
}


