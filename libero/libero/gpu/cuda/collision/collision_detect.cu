#include "collision_detect.cuh"
extern "C" size_t collision_detect_params_size() {return sizeof(libero_collision_cuda::Params);}
__global__ void collision_detect_producer(libero_collision_cuda::Params* p) {
 const int w=blockIdx.y,bucket=blockIdx.z;
 const bool collective=libero_collision_cuda::detect_collective(*p,bucket);
 const int worker=collective?threadIdx.x/32:threadIdx.x;
 const int workers=collective?2:64;
 for(int q=p->bucket_offsets[16*w+bucket]+blockIdx.x*workers+worker;
     q<p->bucket_offsets[16*w+bucket+1];q+=gridDim.x*workers) {
  const int j=p->query_order[(size_t)w*p->pairs+q];
  libero_collision_cuda::detect_scheduled_query(*p,w,j,bucket,collective);
 }
}
__global__ void collision_detect_prepare(libero_collision_cuda::Params* p) {
 const int w=blockIdx.y;
 if(threadIdx.x==0)atomicAdd((unsigned long long*)(p->cuda_detector_execution_count+w),1ULL);
 libero_collision_cuda::detect_prepare_routes(*p,w);
}
__global__ void collision_detect_commit(libero_collision_cuda::Params* p) {
 libero_collision_cuda::detect_commit_world(*p,blockIdx.y);
}
__global__ void collision_detect_collective(libero_collision_cuda::Params* p) {
 libero_collision_cuda::detect_world(*p,blockIdx.y);
}
extern "C" int launch_collision_detect(void* p,int worlds,void* stream) {
 collision_detect_collective<<<dim3(1,worlds),64,0,(cudaStream_t)stream>>>((libero_collision_cuda::Params*)p);
 return (int)cudaGetLastError();
}
extern "C" int launch_collision_detect_tiled(void* p,int worlds,int pairs,void* stream) {
 collision_detect_prepare<<<dim3(1,worlds),64,0,(cudaStream_t)stream>>>((libero_collision_cuda::Params*)p);
 int code=(int)cudaGetLastError();if(code)return code;
 // Route is uniform across each block; warp queries use grid-stride work.
 collision_detect_producer<<<dim3((pairs+63)/64,worlds,15),64,0,(cudaStream_t)stream>>>((libero_collision_cuda::Params*)p);
 code=(int)cudaGetLastError();if(code)return code;
 collision_detect_commit<<<dim3(1,worlds),64,0,(cudaStream_t)stream>>>((libero_collision_cuda::Params*)p);
 return (int)cudaGetLastError();
}

__global__ void collision_detect_compact_producer(libero_collision_cuda::Params* p,const int* guard) {
 const int w=blockIdx.y;
 if(guard && !guard[w])return;
 const size_t base=(size_t)w*((p->pairs+1)/2+15);
 for(int task=blockIdx.x;task<p->task_count[w];task+=gridDim.x) {
  const int bucket=p->task_bucket[base+task],start=p->task_start[base+task];
  const bool collective=libero_collision_cuda::detect_collective(*p,bucket);
  const int worker=collective?threadIdx.x/32:threadIdx.x;
  const int q=start+worker;
  if(q<p->bucket_offsets[16*w+bucket+1]) {
   const int j=p->query_order[(size_t)w*p->pairs+q];
   libero_collision_cuda::detect_scheduled_query(*p,w,j,bucket,collective);
  }
  // All warp collectives finish before this CTA changes its uniform route.
  __syncthreads();
 }
}
extern "C" int launch_collision_detect_compact_producer(void* p,int worlds,int blocks,void* stream,const int* guard) {
 if(worlds<=0||blocks<=0)return (int)cudaErrorInvalidValue;
 collision_detect_compact_producer<<<dim3(blocks,worlds),64,0,(cudaStream_t)stream>>>((libero_collision_cuda::Params*)p,guard);
 return (int)cudaGetLastError();
}

// Four route families use independent entry points: primitive, convex,
// box-box, capsule-box. Private contact capacities and arithmetic are unchanged.
// Prefixes compact only *live* tiles across all worlds, with stable identities.
__device__ int collision_route_tiles(const libero_collision_cuda::Params* p,int w,int bucket) {
 const int* o=p->bucket_offsets+16*w;
 const bool collective=libero_collision_cuda::detect_collective(*p,bucket);
 const int width=collective?2:64;
 return (o[bucket+1]-o[bucket]+width-1)/width;
}
__global__ void collision_queue_prefix(libero_collision_cuda::Params* p,int64_t* offsets,const int* guard) {
 const int family=threadIdx.x;
 if(family>=4)return;
 int64_t* out=offsets+(size_t)family*(p->nworld+1);
 out[0]=0;
 for(int w=0;w<p->nworld;w++) {
  int count=0;
  if((!guard||guard[w]) && p->active[w]==1 && (p->diagnostic_profile==0||!p->fe_code[w])) {
   if(family)count=collision_route_tiles(p,w,10+family);
   else for(int b=0;b<15;b++)if(b<11||b==14)count+=collision_route_tiles(p,w,b);
  }
  out[w+1]=out[w]+count;
 }
}
template<int Family> __global__ void collision_route_producer(
 libero_collision_cuda::Params* p,const int64_t* offsets) {
 const int64_t* o=offsets+(size_t)Family*(p->nworld+1);
 for(int64_t job=blockIdx.x;job<o[p->nworld];job+=gridDim.x) {
  int lo=0,hi=p->nworld;
  while(lo<hi){int mid=lo+(hi-lo)/2;if(o[mid+1]<=job)lo=mid+1;else hi=mid;}
  const int w=lo;
  int first=0;for(int b=0;b<11;b++)first+=collision_route_tiles(p,w,b);
  int task=int(job-o[w]);
  if constexpr(Family==0) {
   if(task>=first)for(int b=11;b<=13;b++)task+=collision_route_tiles(p,w,b);
  } else {
   task+=first;
   for(int b=11;b<10+Family;b++)task+=collision_route_tiles(p,w,b);
  }
  const size_t base=(size_t)w*((p->pairs+1)/2+15);
  const int bucket=p->task_bucket[base+task],start=p->task_start[base+task];
  const bool collective=libero_collision_cuda::detect_collective(*p,bucket);
  const int q=start+(collective?int(threadIdx.x)/32:int(threadIdx.x));
  if(q<p->bucket_offsets[16*w+bucket+1]) {
   const int j=p->query_order[(size_t)w*p->pairs+q],slot=w*p->pairs+j;
   if constexpr(Family==0) {
    if(collective)libero_collision_cuda::primitive_query_warp(*p,w,slot);
    else libero_collision_cuda::detect_primitive_slot(*p,w,j);
   } else {
    // MuJoCo 2.3.7 Convex: one initial contact plus four perturbation trials.
    // This bounds private scratch; the public raw stride remains 50 contacts.
    constexpr int route=10+Family,capacity=route==11?5:(route==13?8:50);
    if(collective)libero_collision_cuda::native_query_warp<capacity,route>(*p,w,slot,route);
    else libero_collision_cuda::native_query<capacity,route>(*p,w,slot,route);
   }
  }
  __syncthreads();
 }
}
extern "C" int launch_collision_detect_parallel(void* raw,int worlds,int blocks,
 int64_t* offsets,const int* guard,void* stream) {
 if(worlds<=0||blocks<=0||!offsets)return (int)cudaErrorInvalidValue;
 auto* p=(libero_collision_cuda::Params*)raw;
 auto s=(cudaStream_t)stream;
 collision_queue_prefix<<<1,32,0,s>>>(p,offsets,guard);
 int code=(int)cudaGetLastError();if(code)return code;
 collision_route_producer<0><<<blocks,64,0,s>>>(p,offsets);
 code=(int)cudaGetLastError();if(code)return code;
 collision_route_producer<1><<<blocks,64,0,s>>>(p,offsets);
 code=(int)cudaGetLastError();if(code)return code;
 collision_route_producer<2><<<blocks,64,0,s>>>(p,offsets);
 code=(int)cudaGetLastError();if(code)return code;
 collision_route_producer<3><<<blocks,64,0,s>>>(p,offsets);
 return (int)cudaGetLastError();
}
