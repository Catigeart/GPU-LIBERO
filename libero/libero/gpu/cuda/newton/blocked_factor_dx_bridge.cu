#include "precision.cuh"
// Handwritten fixed-size tiles. The full matrix remains in global storage.
#include <cusolverdx.hpp>
#include <cublasdx.hpp>

using BlockedPotrf=decltype(cusolverdx::Size<BF_TILE>()+
    cusolverdx::Precision<real_t>()+cusolverdx::Type<cusolverdx::type::real>()+
    cusolverdx::Function<cusolverdx::function::potrf>()+
    cusolverdx::FillMode<cusolverdx::lower>()+
    cusolverdx::Arrangement<cusolverdx::row_major>()+
    cusolverdx::SM<DX_SM>()+cusolverdx::BlockDim<DX_THREADS>()+cusolverdx::Block());

template<cusolverdx::side Side,cusolverdx::transpose Trans,int M,int N>
using BlockedTrsm=decltype(cusolverdx::Size<M,N>()+
    cusolverdx::Precision<real_t>()+cusolverdx::Type<cusolverdx::type::real>()+
    cusolverdx::Function<cusolverdx::function::trsm>()+
    cusolverdx::Side<Side>()+cusolverdx::FillMode<cusolverdx::lower>()+
    cusolverdx::TransposeMode<Trans>()+cusolverdx::Diag<cusolverdx::diag::non_unit>()+
    cusolverdx::Arrangement<cusolverdx::row_major,cusolverdx::row_major>()+
    cusolverdx::SM<DX_SM>()+cusolverdx::BlockDim<DX_THREADS>()+cusolverdx::Block());

__device__ void dx_blocked_potrf(real_t* d,int* info) {
    static_assert(BlockedPotrf::get_shared_memory_size()<=BF_SHARED_BYTES,
                  "blocked POTRF exceeds constant shared reservation");
    static_assert(sizeof(typename BlockedPotrf::status_type)==sizeof(int),"POTRF status ABI");
    BlockedPotrf().execute(d,BF_TILE,info);
    __syncthreads();
}
__device__ void dx_blocked_panel(real_t* d,real_t* a) {
    using Op=BlockedTrsm<cusolverdx::right,cusolverdx::trans,BF_TILE,BF_TILE>;
    static_assert(Op::get_shared_memory_size()<=BF_SHARED_BYTES,"blocked panel TRSM workspace");
    Op().execute(d,BF_TILE,a,BF_TILE);
    __syncthreads();
}
__device__ void dx_blocked_update(real_t* a,real_t* b,real_t* c) {
    using Op=decltype(cublasdx::Size<BF_TILE,BF_TILE,BF_TILE>()+
        cublasdx::Precision<real_t>()+cublasdx::Type<cublasdx::type::real>()+
        cublasdx::Arrangement<cublasdx::row_major,cublasdx::row_major,cublasdx::row_major>()+
        cublasdx::Alignment<sizeof(real_t),sizeof(real_t),sizeof(real_t)>()+cublasdx::LeadingDimension<BF_TILE,BF_TILE,BF_TILE>()+
        cublasdx::Function<cublasdx::function::MM>()+
        cublasdx::SM<DX_SM>()+cublasdx::BlockDim<DX_THREADS>()+cublasdx::Block());
    Op().execute(-REAL_C(1.0),a,BF_TILE,b,BF_TILE,REAL_C(1.0),c,BF_TILE);
    __syncthreads();
}
__device__ void dx_blocked_rhs_update(real_t* a,real_t* b,real_t* c) {
    using Op=decltype(cublasdx::Size<BF_TILE,1,BF_TILE>()+
        cublasdx::Precision<real_t>()+cublasdx::Type<cublasdx::type::real>()+
        cublasdx::Arrangement<cublasdx::row_major,cublasdx::row_major,cublasdx::row_major>()+
        cublasdx::Alignment<sizeof(real_t),sizeof(real_t),sizeof(real_t)>()+cublasdx::LeadingDimension<BF_TILE,1,1>()+
        cublasdx::Function<cublasdx::function::MM>()+
        cublasdx::SM<DX_SM>()+cublasdx::BlockDim<DX_THREADS>()+cublasdx::Block());
    Op().execute(-REAL_C(1.0),a,BF_TILE,b,1,REAL_C(1.0),c,1);
    __syncthreads();
}
template<cusolverdx::transpose Trans>
__device__ void blocked_vector_trsm(real_t* d,real_t* b) {
    using Op=BlockedTrsm<cusolverdx::left,Trans,BF_TILE,1>;
    static_assert(Op::get_shared_memory_size()<=BF_SHARED_BYTES,"blocked vector TRSM workspace");
    Op().execute(d,BF_TILE,b,1);
    __syncthreads();
}
__device__ void dx_blocked_forward(real_t* d,real_t* b) {
    blocked_vector_trsm<cusolverdx::non_trans>(d,b);
}
__device__ void dx_blocked_backward(real_t* d,real_t* b) {
    blocked_vector_trsm<cusolverdx::trans>(d,b);
}
