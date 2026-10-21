// Copyright 2026 Flash-LIBERO contributors
// World-owned raw-field packing and ordered low-dimensional observation sampling.
#include "fused_observations.cuh"
#include <cmath>
#ifndef FLASH_FP32_OBSERVATIONS_SOURCE_ID
#error "Build through flash_fp32_fused_observations"
#endif
namespace flash::fp32 {
// Preserve the accepted robosuite FP32 quaternion convention and cutoff.
__device__ void rotation32(const float* q,float* r) {
  float v[4]={q[3],q[0],q[1],q[2]},n=0;
  for (int i=0;i<4;++i) n+=v[i]*v[i];
  for (int i=0;i<9;++i) r[i]=i%4==0 ? 1.f : 0.f;
  if (n>=float(4.*2.220446049250313e-16)) {
    float scale=sqrtf(2.f/n);for (int i=0;i<4;++i) v[i]*=scale;
    r[0]=1-v[2]*v[2]-v[3]*v[3];r[1]=v[1]*v[2]-v[3]*v[0];r[2]=v[1]*v[3]+v[2]*v[0];
    r[3]=v[1]*v[2]+v[3]*v[0];r[4]=1-v[1]*v[1]-v[3]*v[3];r[5]=v[2]*v[3]-v[1]*v[0];
    r[6]=v[1]*v[3]-v[2]*v[0];r[7]=v[2]*v[3]+v[1]*v[0];r[8]=1-v[1]*v[1]-v[2]*v[2];
  }
}
__device__ void matrix_quaternion32(const float* m,float* out) {
  float k[16]={m[0]-m[4]-m[8],m[1]+m[3],m[2]+m[6],m[7]-m[5],
      m[1]+m[3],m[4]-m[0]-m[8],m[5]+m[7],m[2]-m[6],
      m[2]+m[6],m[5]+m[7],m[8]-m[0]-m[4],m[3]-m[1],
      m[7]-m[5],m[2]-m[6],m[3]-m[1],m[0]+m[4]+m[8]};
  for (int i=0;i<16;++i) k[i]/=3.f;
  int largest=0;for (int i=1;i<4;++i) if (k[5*i]>k[5*largest]) largest=i;
  float q[4]={};q[largest]=1;
  for (int it=0;it<24;++it) {
    float v[4]={},norm=0;
    for (int i=0;i<4;++i) for (int j=0;j<4;++j) v[i]+=k[4*i+j]*q[j];
    for (int i=0;i<4;++i) norm+=v[i]*v[i];norm=sqrtf(norm);
    for (int i=0;i<4;++i) q[i]=v[i]/norm;
  }
  for (int i=0;i<4;++i) out[i]=q[3]<0 ? -q[i] : q[i];
}

template<bool Sample,bool ParallelClock=false>
__global__ void pack_and_sample(RawPackInputs pack,SampleInputs a,float dt,int force) {
  constexpr int Threads=Sample && !ParallelClock ? 32 : 128;
  const int world=blockIdx.x,lane=threadIdx.x;
  // Every field writes a disjoint output interval. One CTA owns all five
  // copies for its world, with no cross-world data or synchronization.
  if constexpr (!Sample) for (int field=0;field<5;++field) {
    const float* source=pack.source[field]+world*pack.width[field];
    float* target=pack.target+world*pack.stride+pack.offset[field];
    for (int element=lane;element<pack.width[field];element+=Threads)
      target[element]=source[element];
  }
  if constexpr (Sample) {
    // All producers and consumers below belong to this CTA's world.
    if constexpr (!Sample) __syncthreads();
    const int w=world;
    if (!a.mask[w]) return; // block-uniform: masked worlds retain their observation state
    const int b=w*a.width,sb=w*a.ns;
    __shared__ unsigned active_words[4]; // 16 bytes, independent of worlds/NV
    if constexpr (ParallelClock) {
      bool active=false;
      if (lane<a.ns) {
        const int s=lane,at=sb+s;
        a.events[at]=0;
        if (a.enabled[s]) {
          const int sampled=a.sampled[at];
          const float elapsed=a.elapsed[at]+dt,period=a.periods[s];
          const bool due=elapsed>=period;
          active=(!sampled && period>=elapsed)||force||(due && !sampled);
          // Only clock decisions move ahead of the ordered math. Cache and
          // presence dependencies below still execute in increasing spec order.
          a.elapsed[at]=due ? fmodf(elapsed,period) : elapsed;
          a.sampled[at]=due ? 0 : active ? 1 : sampled;
        }
      }
      const unsigned bits=__ballot_sync(0xffffffff,active);
      if ((lane&31)==0) active_words[lane/32]=bits;
      __syncthreads();
    }
    if (lane!=0) return;
    for (int s=0;s<a.ns;++s) {
      float elapsed=0.f;bool take=false,fallback=false;
      if constexpr (ParallelClock) {
        if (!(active_words[s/32]&(1u<<(s&31)))) continue;
        take=true;
      } else {
        a.events[sb+s]=0;if (!a.enabled[s]) continue;
        elapsed=a.elapsed[sb+s]+dt;a.elapsed[sb+s]=elapsed;
        take=(!a.sampled[sb+s] && a.periods[s]>=elapsed)||force;
        fallback=elapsed>=a.periods[s] && !a.sampled[sb+s];
      }
      if (take || fallback) {
        int at=b+a.offset[s],d0=b+a.dep0[s],d1=b+a.dep1[s],d2=b+a.dep2[s];
        for (int j=0;j<a.size[s];++j) {
          float value=0;
          if (a.op[s]==0) {
            const int index=a.indices[a.index_offset[s]+j];
            if constexpr (Sample) {
              // Resolve the CURRENT gather index rather than caching a
              // derived table that could outlive mutable metadata.
              for (int field=0;field<5;++field) {
                const int local=index-pack.offset[field];
                if (local>=0 && local<pack.width[field]) {
                  value=pack.source[field][w*pack.width[field]+local];break;
                }
              }
            } else value=a.source[w*a.stride+index];
          }
          else if (a.op[s]==1 && a.present[sb+a.presence[3*s]]) value=cosf(a.cache[d0+j]);
          else if (a.op[s]==2 && a.present[sb+a.presence[3*s]]) value=sinf(a.cache[d0+j]);
          a.cache[at+j]=value;
        }
        if (a.op[s]==3) {
          if (!a.present[sb+a.presence[3*s+1]] || !a.present[sb+a.presence[3*s]]) {
            for (int row=0;row<4;++row) a.cache[at+5*row]=1;
          } else {
            float r[9];rotation32(a.cache+d1,r);
            for (int row=0;row<3;++row) {
              float t=0;
              for (int col=0;col<3;++col) {a.cache[at+4*row+col]=r[3*col+row];t-=r[3*col+row]*a.cache[d0+col];}
              a.cache[at+4*row+3]=t;
            }
            a.cache[at+15]=1;
          }
        } else if (a.op[s]==4) {
          if (a.present[sb+a.presence[3*s]] && a.present[sb+a.presence[3*s+1]] && a.present[sb+a.presence[3*s+2]]) {
            float ro[9],rr[9];rotation32(a.cache+d1,ro);
            for (int row=0;row<3;++row) {
              float t=a.cache[d2+4*row+3];
              for (int col=0;col<3;++col) {
                t+=a.cache[d2+4*row+col]*a.cache[d0+col];float value=0;
                for (int k=0;k<3;++k) value+=a.cache[d2+4*row+k]*ro[3*k+col];rr[3*row+col]=value;
              }
              a.cache[at+row]=t;
            }
            matrix_quaternion32(rr,a.sidequat+b+a.offset[s]);
          }
        } else if (a.op[s]==5 && a.present[sb+a.presence[3*s]]) {
          for (int j=0;j<4;++j) a.cache[at+j]=a.sidequat[d0+j];
        }
        for (int j=0;j<a.size[s];++j) a.observed[at+j]=a.cache[at+j];
        a.present[sb+s]=a.events[sb+s]=1;
        if constexpr (!ParallelClock) a.sampled[sb+s]=1;
      }
      if constexpr (!ParallelClock)
        if (elapsed>=a.periods[s]) {a.elapsed[sb+s]=fmodf(elapsed,a.periods[s]);a.sampled[sb+s]=0;}
    }
  }
}
}
extern "C" {
const char* flash_fp32_observations_source_id() { return FLASH_FP32_OBSERVATIONS_SOURCE_ID; }
int flash_fp32_observations_arch() { return FLASH_CUDA_ARCH; }
std::size_t flash_fp32_observations_pack_bytes() { return sizeof(flash::fp32::RawPackInputs); }
int flash_fp32_observations_pack(const flash::fp32::RawPackInputs* a,cudaStream_t stream) {
  if (!a || a->worlds<1 || a->stride<1) return cudaErrorInvalidValue;
  flash::fp32::pack_and_sample<false><<<a->worlds,128,0,stream>>>(*a,{},0.f,0);
  return cudaGetLastError();
}
std::size_t flash_fp32_observations_sample_bytes() { return sizeof(flash::fp32::SampleInputs); }
int flash_fp32_observations_sample(const flash::fp32::RawPackInputs* pack,
    const flash::fp32::SampleInputs* sample,float dt,int force,cudaStream_t stream) {
  if (!pack || !sample || pack->worlds!=sample->worlds || pack->stride!=sample->stride)
    return cudaErrorInvalidValue;
  // Bounded per-world clock bitmap. Large custom plans keep the scalar path;
  // shared storage never grows with the number of worlds or observations.
  if (sample->ns<=128) flash::fp32::pack_and_sample<true,true><<<pack->worlds,128,0,stream>>>(*pack,*sample,dt,force);
  else flash::fp32::pack_and_sample<true><<<pack->worlds,32,0,stream>>>(*pack,*sample,dt,force);
  return cudaGetLastError();
}
}
