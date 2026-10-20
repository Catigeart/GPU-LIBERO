#include "precision.cuh"
// Standard dense J*qvel: cuBLASDx device GEMM with N=1 (GEMV).
// Tiling bounds shared memory independently of row capacity. Tail A rows are
// zero padded; only the admitted valid prefix is written to private scratch.
#pragma once
#include <cublasdx.hpp>
namespace cf_s {
using ReferenceGemm=decltype(cublasdx::Size<32,1,NV>()+cublasdx::Precision<real_t>()+
  cublasdx::Type<cublasdx::type::real>()+
  cublasdx::Arrangement<cublasdx::row_major,cublasdx::col_major,cublasdx::col_major>()+
  cublasdx::Function<cublasdx::function::MM>()+cublasdx::SM<CF_S_DX_SM>()+
  cublasdx::Block()+cublasdx::BlockDim<64>());
__device__ void CF_S_reference {
  if(committed[w]!=1)return;
  constexpr int tile=32;
  __shared__ __align__(16) char storage[cublasdx::get_shared_storage_size<ReferenceGemm>()];
  auto [pa,pb,pc]=cublasdx::slice_shared_memory<ReferenceGemm>(storage);
  auto A=cublasdx::make_tensor(pa,ReferenceGemm::get_layout_smem_a());
  auto B=cublasdx::make_tensor(pb,ReferenceGemm::get_layout_smem_b());
  auto C=cublasdx::make_tensor(pc,ReferenceGemm::get_layout_smem_c());
  int total=counts[w*5+3];
  // B is invariant for all row tiles. Keep it resident in the same shared
  // allocation instead of reloading qvel once per tile.
  for(int k=threadIdx.x;k<NV;k+=blockDim.x)B(k,0)=qvel[w*NV+k];
  __syncthreads();
  for(int base=0;base<total;base+=tile){
    // Scatter wide global loads through tensor accessors; keep library layout.
    for(int z=2*threadIdx.x;z<tile*NV;z+=2*blockDim.x){
      const int r=z/NV,k=z%NV,next=z+1,r1=next/NV,k1=next%NV;
      if(next<tile*NV&&base+r1<total){
        const real2_t v=libero_vec::load_pair(efc_J+(w*rows+base)*NV+z);
        A(r,k)=v.x;A(r1,k1)=v.y;
      }else{
        A(r,k)=base+r<total?efc_J[(w*rows+base)*NV+z]:REAL_C(0.);
        if(next<tile*NV)A(r1,k1)=base+r1<total?efc_J[(w*rows+base)*NV+next]:REAL_C(0.);
      }
    }
    for(int r=threadIdx.x;r<tile;r+=blockDim.x)C(r,0)=REAL_C(0.);
    __syncthreads();
    ReferenceGemm().execute(REAL_C(1.),A,B,REAL_C(0.),C);
    __syncthreads();
    for(int r=threadIdx.x;r<tile&&base+r<total;r+=blockDim.x){
      int g=w*rows+base+r;real_t acc=C(r,0);
      vel[g]=acc;aref[g]=-kbip[g*4+1]*acc-kbip[g*4]*kbip[g*4+2]*(efc_pos[g]-efc_margin[g]);
    }
    __syncthreads();
  }
}
}
