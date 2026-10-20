#include <precision.cuh>
// LIBERO GPU query-local support reuse. This helper is handwritten here;
// MuJoCo/libccd provenance for its callers is in THIRD_PARTY_NOTICES.md.
#pragma once

#ifndef LIBERO_NATIVE_SUPPORT_REUSE
// Exact-direction repeats were absent in the first real-query sample. Retain
// the experiment without adding its private-state cost to the default path.
#define LIBERO_NATIVE_SUPPORT_REUSE 0
#endif
#ifndef LIBERO_NATIVE_SUPPORT_REUSE_DIAGNOSTIC
#define LIBERO_NATIVE_SUPPORT_REUSE_DIAGNOSTIC 0
#endif
#ifndef LIBERO_NATIVE_PORTAL_REUSE
#define LIBERO_NATIVE_PORTAL_REUSE 1
#endif

// The mesh and margin are immutable for the lifetime of mjtCCD. The key is
// the exact local-frame direction, not a rounded direction or a warm seed.
// Keeping the local result makes pose changes in MultiCCD safe: both direction
// and result are transformed using the current pose on every support call.
struct NativeSupportReuse {
  unsigned long long key[3];
  real_t local_result[3];
  int meshindex;
  unsigned valid;
  unsigned hits;
  unsigned misses;
};

static __device__ __forceinline__ bool native_support_reuse_get(
    NativeSupportReuse* cache, const real_t* dir, real_t* result, int* meshindex) {
#if LIBERO_NATIVE_SUPPORT_REUSE
  bool hit=cache->valid!=0;
  for(int i=0;i<3;++i)
    hit=hit && cache->key[i]==static_cast<unsigned long long>(real_bits(dir[i]));
  if(hit){
    for(int i=0;i<3;++i)result[i]=cache->local_result[i];
    *meshindex=cache->meshindex;
    ++cache->hits;
    return true;
  }
  ++cache->misses;
#endif
  return false;
}

static __device__ __forceinline__ void native_support_reuse_put(
    NativeSupportReuse* cache,const real_t* dir,const real_t* result,int meshindex) {
#if LIBERO_NATIVE_SUPPORT_REUSE
  // Failed searches, NaN directions and invalid vertices never create entries.
  if(meshindex<0)return;
  for(int i=0;i<3;++i)if(!isfinite(dir[i]) || !isfinite(result[i]))return;
  for(int i=0;i<3;++i){
    cache->key[i]=static_cast<unsigned long long>(real_bits(dir[i]));
    cache->local_result[i]=result[i];
  }
  cache->meshindex=meshindex;
  cache->valid=1;
#endif
}
