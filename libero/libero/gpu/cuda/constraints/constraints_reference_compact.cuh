#include "precision.cuh"
// Explicit S08 prototype. No production default selects this header.
// The caller must prove complete disjoint row ownership, ascending support,
// current zero values outside support, and current qvel finite readset first.
// Capacity and public commit remain the original dense reference owner's job.
#pragma once
#include <cublasdx.hpp>
namespace cf_s {
template<int K>
using ReferenceGemm=decltype(cublasdx::Size<32,1,K>()+cublasdx::Precision<real_t>()+
    cublasdx::Type<cublasdx::type::real>()+
    cublasdx::Arrangement<cublasdx::row_major,cublasdx::col_major,cublasdx::col_major>()+
    cublasdx::Function<cublasdx::function::MM>()+cublasdx::SM<CF_S_DX_SM>()+
    cublasdx::Block()+cublasdx::BlockDim<64>());

template<int V>
__device__ __noinline__ char* reference_storage(){
  // All K specializations in this entry use one block-lifetime arena. Keeping
  // a shared array per K can accumulate their static resource reservations.
  __shared__ __align__(16) char memory[cublasdx::get_shared_storage_size<ReferenceGemm<V>>()];
  return memory;
}

template<int K,int V=K>
__device__ void reference_compact(int w,int rows,int nv,int group_rows,
  const int* row_ids,const int* columns,const real_t* J,const real_t* qvel,
  const real_t* kbip,const real_t* pos,const real_t* margin,real_t* vel,real_t* aref){
  static_assert(K>0&&K<=V,"positive compact K <= full V required; zero-support rows use dense fallback");
  using Gemm=ReferenceGemm<K>;
  static_assert(cublasdx::get_shared_storage_size<Gemm>()<=cublasdx::get_shared_storage_size<ReferenceGemm<V>>(),"compact Dx layout must fit full-size arena");
  char* memory=reference_storage<V>();
  auto [pa,pb,pc]=cublasdx::slice_shared_memory<Gemm>(memory);
  auto A=cublasdx::make_tensor(pa,Gemm::get_layout_smem_a());
  auto B=cublasdx::make_tensor(pb,Gemm::get_layout_smem_b());
  auto C=cublasdx::make_tensor(pc,Gemm::get_layout_smem_c());
  for(int k=threadIdx.x;k<K;k+=blockDim.x)B(k,0)=qvel[w*nv+columns[k]];
  __syncthreads();
  for(int base=0;base<group_rows;base+=32){
    for(int z=threadIdx.x;z<32*K;z+=blockDim.x){
      const int r=z/K,k=z%K;
      A(r,k)=base+r<group_rows?J[(w*rows+row_ids[base+r])*nv+columns[k]]:REAL_C(0.);
    }
    for(int r=threadIdx.x;r<32;r+=blockDim.x)C(r,0)=REAL_C(0.);
    __syncthreads();Gemm().execute(REAL_C(1.),A,B,REAL_C(0.),C);__syncthreads();
    for(int r=threadIdx.x;r<32&&base+r<group_rows;r+=blockDim.x){
      const int g=w*rows+row_ids[base+r];const real_t acc=C(r,0);
      vel[g]=acc;aref[g]=-kbip[g*4+1]*acc-kbip[g*4]*kbip[g*4+2]*(pos[g]-margin[g]);
    }
    __syncthreads();
  }
}

template<int K,int V>
__device__ void reference_group_guarded(int w,int rows,int nv,int group_rows,
  const int* row_ids,const int* columns,const int* support_mask,const int* full_columns,
  const real_t* J,const real_t* qvel,const real_t* kbip,const real_t* pos,
  const real_t* margin,real_t* vel,real_t* aref){
  // Arbitrary raw GPU changes are detected on every invocation. A topology
  // descriptor never licenses skipping a now nonzero or nonfinite J element.
  __shared__ int votes[32],valid;
  bool bad=false;
  for(int z=threadIdx.x;z<group_rows*nv;z+=blockDim.x){
    const int k=z%nv,r=row_ids[z/nv];
    if(!support_mask[k]){const real_t x=J[(w*rows+r)*nv+k];bad|=x!=REAL_C(0.);}
  }
  const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
  const bool vote=libero_warp::any(bad);if(lane==0)votes[warp]=vote;
  __syncthreads();
  if(warp==0){const bool any=libero_warp::any(lane<(blockDim.x>>5)&&votes[lane]);if(lane==0)valid=!any;}
  __syncthreads();
  if(valid)reference_compact<K,V>(w,rows,nv,group_rows,row_ids,columns,J,qvel,kbip,pos,margin,vel,aref);
  else reference_compact<V,V>(w,rows,nv,group_rows,row_ids,full_columns,J,qvel,kbip,pos,margin,vel,aref);
}
}
