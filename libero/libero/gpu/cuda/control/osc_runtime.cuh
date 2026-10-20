#include "precision.cuh"
#pragma once
#include "../common/vector.cuh"
#include <stdint.h>
namespace osc_cuda {
// Preserve ordered forward-copy semantics for overlapping public views.
__device__ inline void serial_copy_double(const real_t* s,real_t* d,int n){
 if(n<=0)return;
 const uintptr_t a=(uintptr_t)s,b=(uintptr_t)d;
 const bool overlap=a<=b?b-a<(uintptr_t)n*sizeof(real_t):a-b<(uintptr_t)n*sizeof(real_t);
 if(overlap){for(int z=0;z<n;z++)d[z]=s[z];}
 else libero_vec::copy_serial(s,d,n);
}
__device__ inline void plan(int w,int*active,int*first,int*needs,int*refresh,int*call,int force){
 call[w]=refresh[w]=0;if(active[w]==1&&!first[w]){call[w]=1;if(needs[w]==1||force==1)refresh[w]=1;}}
__device__ inline void gate(int w,int*active,int*pv,int64_t*written,int64_t*expected,int*first,int*error,int*policy,int step){
 error[w]=0;policy[w]=step;if(pv[w]!=1||written[w]!=expected[w]||first[w])active[w]=0;}
__device__ inline void commit_mask(int w,int*active,int*error,int*first,int*commit,int64_t*event,int*valid){
 commit[w]=valid[w]=0;if(active[w]!=1||first[w])return;if(error[w]){first[w]=202;return;}
 if(event[w]==INT64_MAX){first[w]=9;return;}event[w]++;commit[w]=1;}
__device__ inline void copy_double(int w,int*mask,real_t*source,real_t*target,int width){if(mask[w]==1)serial_copy_double(source+w*width,target+w*width,width);}
__device__ inline void copy_int(int w,int*mask,int*source,int*target){if(mask[w]==1)target[w]=source[w];}
__device__ inline void seed_double(int w,real_t*source,real_t*target,int width){serial_copy_double(source+w*width,target+w*width,width);}
__device__ inline void seed_int(int w,int*source,int*target){target[w]=source[w];}
__device__ inline void publish(int w,int*mask,int64_t*event,int64_t*parent,int64_t*written,int64_t*consumed,int*valid){if(mask[w]==1){consumed[w]=parent[w];written[w]=event[w];valid[w]=1;}}
__device__ inline void bootstrap_finish(int w,int*mask,int*error,int*valid,int*needs,int*first){if(mask[w]==1){if(error[w]||valid[w]!=1){mask[w]=0;if(!first[w])first[w]=202;}else needs[w]=0;}}
__device__ inline void reset_public(int w,int*mask,int*first,real_t*source,real_t*ctrl,int*valid,int64_t*written,int64_t*event){
 if(mask[w]!=1||first[w])return;if(libero_vec::any_nonfinite_lane(source+w*9,9,0,1)){first[w]=202;valid[w]=0;return;}
 serial_copy_double(source+w*9,ctrl+w*9,9);event[w]=written[w]=0;valid[w]=1;}
}
