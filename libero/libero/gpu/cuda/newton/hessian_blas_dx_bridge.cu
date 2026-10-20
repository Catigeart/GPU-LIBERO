#include "precision.cuh"
// Handwritten ABI bridge: standard dense tile multiplication is cuBLASDx.
#include <cublasdx.hpp>

__device__ void dx_hessian_gemm(real_t* a,real_t* b,real_t* c) {
    using Op=decltype(cublasdx::Size<NV,NV,DX_HESSIAN_K>()+
        cublasdx::Precision<real_t>()+cublasdx::Type<cublasdx::type::real>()+
        cublasdx::Arrangement<cublasdx::row_major,cublasdx::row_major,cublasdx::row_major>()+
        cublasdx::Alignment<sizeof(real_t),sizeof(real_t),sizeof(real_t)>()+
        cublasdx::LeadingDimension<DX_HESSIAN_K,NV,NV>()+
        cublasdx::Function<cublasdx::function::MM>()+
        cublasdx::SM<DX_SM>()+cublasdx::BlockDim<DX_THREADS>()+cublasdx::Block());
    // Explicit leading dimensions keep caller staging compact and parameterized.
    Op().execute(REAL_C(1.0),a,DX_HESSIAN_K,b,NV,REAL_C(1.0),c,NV);
    __syncthreads();
}

template<int M,int N,int K>
__device__ void dx_dense_mm(real_t* a,real_t* b,real_t* c,real_t beta) {
    using Op=decltype(cublasdx::Size<M,N,K>()+
        cublasdx::Precision<real_t>()+cublasdx::Type<cublasdx::type::real>()+
        cublasdx::Arrangement<cublasdx::row_major,cublasdx::row_major,cublasdx::row_major>()+
        cublasdx::Alignment<sizeof(real_t),sizeof(real_t),sizeof(real_t)>()+cublasdx::LeadingDimension<K,N,N>()+
        cublasdx::Function<cublasdx::function::MM>()+
        cublasdx::SM<DX_SM>()+cublasdx::BlockDim<DX_THREADS>()+cublasdx::Block());
    Op().execute(REAL_C(1.0),a,K,b,N,beta,c,N);
    __syncthreads();
}
__device__ void dx_newton_jv(real_t* a,real_t* b,real_t* c) {
    dx_dense_mm<16,1,NV>(a,b,c,REAL_C(0.0));
}
__device__ void dx_newton_transpose(real_t* a,real_t* b,real_t* c) {
    dx_dense_mm<NV,1,DX_HESSIAN_K>(a,b,c,REAL_C(1.0));
}
__device__ void dx_newton_cone_ltj(real_t* a,real_t* b,real_t* c) {
    dx_dense_mm<6,NV,6>(a,b,c,REAL_C(0.0));
}
