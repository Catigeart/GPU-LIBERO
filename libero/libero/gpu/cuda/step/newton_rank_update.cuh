#pragma once
#include "precision.cuh"
#include "newton_profile.cuh"

namespace physics_newton::rank_update {

// MuJoCo 2.3.7 mju_cholUpdate recurrence. Factor and vector are disjoint
// internal owners. Only the lower triangle is an operand or destination.
template<int N>
static __device__ inline int serial(real_t *factor, real_t *vector, int plus) {
  int rank=N;
  for (int k=0;k<N;++k) {
    const real_t x=vector[k];
    if (x==REAL_C(0.0)) continue;
    const real_t diagonal=factor[k*N+k];
    real_t square=diagonal*diagonal;
    square=plus ? square+x*x : square-x*x;
    if (square<REAL_C(1e-15)) { square=REAL_C(1e-15); --rank; }
    const real_t root=sqrt(square), c=root/diagonal, inverse=REAL_C(1.0)/c, s=x/diagonal;
    factor[k*N+k]=root;
    for (int i=k+1;i<N;++i)
      factor[i*N+k]=(plus ? factor[i*N+k]+s*vector[i] : factor[i*N+k]-s*vector[i])*inverse;
    for (int i=k+1;i<N;++i) vector[i]=c*vector[i]-s*factor[i*N+k];
  }
  return rank;
}

// Exactly one complete warp enters. Lower rows retain fixed lane ownership
// across a batch; lane zero owns every diagonal. Only register shuffles carry
// inter-lane dependencies, so the caller publishes factor/vector after the
// complete batch. The returned rank is authoritative in lane zero.
template<int N>
static __device__ __forceinline__ int warp_values(real_t *factor, real_t &first, real_t &second, int plus) {
  static_assert(N<=64,"Register vector owns at most two cells per lane");
  const int lane=int(threadIdx.x)&31;
  int rank=N;
  for (int k=0;k<N;++k) {
    const real_t x=__shfl_sync(0xffffffffu,k<32 ? first : second,k&31);
    if (x==REAL_C(0.0)) continue;
    real_t c=REAL_C(0.0), inverse=REAL_C(0.0), s=REAL_C(0.0);
    if (lane==0) {
      const real_t diagonal=factor[k*N+k];
      real_t square=diagonal*diagonal;
      square=plus ? square+x*x : square-x*x;
      if (square<REAL_C(1e-15)) { square=REAL_C(1e-15); --rank; }
      const real_t root=sqrt(square);
      c=root/diagonal; inverse=REAL_C(1.0)/c; s=x/diagonal;
      factor[k*N+k]=root;
    }
    c=__shfl_sync(0xffffffffu,c,0);
    inverse=__shfl_sync(0xffffffffu,inverse,0);
    s=__shfl_sync(0xffffffffu,s,0);
    if (lane>k && lane<N) {
      const real_t updated=(plus ? factor[lane*N+k]+s*first : factor[lane*N+k]-s*first)*inverse;
      factor[lane*N+k]=updated;
      first=c*first-s*updated;
    }
    if constexpr (N>32) {
      const int i=lane+32;
      if (i>k && i<N) {
        const real_t updated=(plus ? factor[i*N+k]+s*second : factor[i*N+k]-s*second)*inverse;
        factor[i*N+k]=updated;
        second=c*second-s*updated;
      }
    }
  }
  return rank;
}

// Ordered positive cone contributions. Input LTJ is already block-published.
// Only the final vector needs materializing; intermediate vectors are private.
template<int N>
static __device__ __noinline__ void ordered_plus(real_t *factor, const real_t *vectors,
                                                int count, real_t *last_vector) {
  const int lane=int(threadIdx.x)&31;
  real_t first=REAL_C(0.0), second=REAL_C(0.0);
  for (int r=0;r<count;++r) {
    CF_NEWTON_PROFILE_SCOPE(RankUpdate);
    first=lane<N ? vectors[r*N+lane] : REAL_C(0.0);
    if constexpr (N>32) second=lane+32<N ? vectors[r*N+lane+32] : REAL_C(0.0);
    warp_values<N>(factor,first,second,1);
  }
  if (count) {
    if (lane<N) last_vector[lane]=first;
    if constexpr (N>32) if (lane+32<N) last_vector[lane+32]=second;
  }
}

// Ordered quadratic-state changes. A negative return is the number completed
// through the first rank loss; later rows must not be accessed before rebuild.
template<int N>
static __device__ __noinline__ int ordered_rows(real_t *factor, const real_t *J, const real_t *sqrt_d,
                                               const int *rows, const int *state, int count,
                                               real_t *last_vector) {
  const int lane=int(threadIdx.x)&31;
  real_t first=REAL_C(0.0), second=REAL_C(0.0);
  int completed=0;
  bool lost=false;
  for (int k=0;k<count;++k) {
    CF_NEWTON_PROFILE_SCOPE(RankUpdate);
    const int r=rows[k], plus=state[r]==1;
    const real_t scale=sqrt_d[r];
    first=lane<N ? J[r*N+lane]*scale : REAL_C(0.0);
    if constexpr (N>32) second=lane+32<N ? J[r*N+lane+32]*scale : REAL_C(0.0);
    const int rank=warp_values<N>(factor,first,second,plus);
    ++completed;
    lost=__shfl_sync(0xffffffffu,rank,0)<N;
    if (lost) break;
  }
  if (completed) {
    if (lane<N) last_vector[lane]=first;
    if constexpr (N>32) if (lane+32<N) last_vector[lane+32]=second;
  }
  return lost ? -completed : completed;
}

// Every block lane enters. Warp 0 owns the recurrence; the last block barrier
// publishes both outputs. FP64 and dimensions above 64 retain the shared-vector
// implementation. Current LIBERO dimensions range from 21 through 57.
template<int N, bool RegisterVector=(FLASH_REAL_BITS==32 && N<=64)>
static __device__ inline int collective(real_t *factor, real_t *vector, int plus) {
  CF_NEWTON_PROFILE_SCOPE(RankUpdate);
  __shared__ int result;
  if constexpr (RegisterVector) {
    if (threadIdx.x<32) {
      const int lane=int(threadIdx.x);
      real_t first=lane<N ? vector[lane] : REAL_C(0.0), second=REAL_C(0.0);
      if constexpr (N>32) second=lane+32<N ? vector[lane+32] : REAL_C(0.0);
      const int rank=warp_values<N>(factor,first,second,plus);
      if (lane<N) vector[lane]=first;
      if constexpr (N>32) if (lane+32<N) vector[lane+32]=second;
      if (lane==0) result=rank;
    }
  } else if constexpr (N<=32) {
    if (threadIdx.x==0) result=serial<N>(factor,vector,plus);
  } else if (threadIdx.x<32) {
    const int lane=int(threadIdx.x);
    int rank=N;
    for (int k=0;k<N;++k) {
      real_t x=lane==0 ? vector[k] : REAL_C(0.0);
      x=__shfl_sync(0xffffffffu,x,0);
      if (x==REAL_C(0.0)) continue;
      real_t c=REAL_C(0.0), inverse=REAL_C(0.0), s=REAL_C(0.0);
      if (lane==0) {
        const real_t diagonal=factor[k*N+k];
        real_t square=diagonal*diagonal;
        square=plus ? square+x*x : square-x*x;
        if (square<REAL_C(1e-15)) { square=REAL_C(1e-15); --rank; }
        const real_t root=sqrt(square);
        c=root/diagonal; inverse=REAL_C(1.0)/c; s=x/diagonal;
        factor[k*N+k]=root;
      }
      c=__shfl_sync(0xffffffffu,c,0);
      inverse=__shfl_sync(0xffffffffu,inverse,0);
      s=__shfl_sync(0xffffffffu,s,0);
      for (int i=k+1+lane;i<N;i+=32) {
        const real_t updated=(plus ? factor[i*N+k]+s*vector[i] : factor[i*N+k]-s*vector[i])*inverse;
        factor[i*N+k]=updated;
        vector[i]=c*vector[i]-s*updated;
      }
      __syncwarp(0xffffffffu);
    }
    if (lane==0) result=rank;
  }
  __syncthreads();
  return result;
}
} // namespace physics_newton::rank_update
