#include "precision.cuh"
// Handwritten cuSOLVERDx bridge; compile with the same NV/SM/block capacity.
#include <cusolverdx.hpp>

template<int N>
using IslandPotrf=decltype(cusolverdx::Size<N>()
    +cusolverdx::Precision<real_t>()+cusolverdx::Type<cusolverdx::type::real>()
    +cusolverdx::Function<cusolverdx::function::potrf>()
    +cusolverdx::FillMode<cusolverdx::fill_mode::lower>()
    +cusolverdx::Arrangement<cusolverdx::row_major>()
    +cusolverdx::SM<DX_SM>()+cusolverdx::BlockDim<DX_THREADS>()+cusolverdx::Block());
template<int N,cusolverdx::transpose T>
using IslandTrsm=decltype(cusolverdx::Size<N,1>()
    +cusolverdx::Precision<real_t>()+cusolverdx::Type<cusolverdx::type::real>()
    +cusolverdx::Function<cusolverdx::function::trsm>()
    +cusolverdx::Side<cusolverdx::side::left>()
    +cusolverdx::FillMode<cusolverdx::lower>()+cusolverdx::TransposeMode<T>()
    +cusolverdx::Diag<cusolverdx::diag::non_unit>()
    +cusolverdx::Arrangement<cusolverdx::row_major,cusolverdx::row_major>()
    +cusolverdx::SM<DX_SM>()+cusolverdx::BlockDim<DX_THREADS>()+cusolverdx::Block());

template<int N> __device__ inline void island_potrf(real_t* A,int* info){
    using S=IslandPotrf<N>;
    static_assert(S::get_shared_memory_size()<=DX_MATRIX_BYTES,"island POTRF workspace");
    static_assert(sizeof(typename S::status_type)==sizeof(int),"island status ABI");
    S().execute(A,N,info);
}
template<int N,cusolverdx::transpose T>
__device__ inline void island_trsm(real_t* A,real_t* b){
    using S=IslandTrsm<N,T>;
    static_assert(S::get_shared_memory_size()<=DX_MATRIX_BYTES,"island TRSM workspace");
    S().execute(A,N,b,1);
}

#define LIBERO_ISLAND_DISPATCH(OP) \
    if(n==1){OP(1);return;} \
    if constexpr(NV>=2)if(n==2){OP(2);return;} \
    if constexpr(NV>=4)if(n==4){OP(4);return;} \
    if constexpr(NV>=8)if(n==8){OP(8);return;} \
    if constexpr(NV>=16)if(n==16){OP(16);return;} \
    if constexpr(NV>=32)if(n==32){OP(32);return;} \
    OP(NV)

__device__ void dx_island_potrf(int n,real_t* A,int* info){
#define LIBERO_ISLAND_POTRF(N) island_potrf<N>(A,info)
    LIBERO_ISLAND_DISPATCH(LIBERO_ISLAND_POTRF);
#undef LIBERO_ISLAND_POTRF
}
__device__ void dx_island_forward(int n,real_t* A,real_t* b){
#define LIBERO_ISLAND_FORWARD(N) island_trsm<N,cusolverdx::non_trans>(A,b)
    LIBERO_ISLAND_DISPATCH(LIBERO_ISLAND_FORWARD);
#undef LIBERO_ISLAND_FORWARD
}
__device__ void dx_island_backward(int n,real_t* A,real_t* b){
#define LIBERO_ISLAND_BACKWARD(N) island_trsm<N,cusolverdx::trans>(A,b)
    LIBERO_ISLAND_DISPATCH(LIBERO_ISLAND_BACKWARD);
#undef LIBERO_ISLAND_BACKWARD
}
#undef LIBERO_ISLAND_DISPATCH
