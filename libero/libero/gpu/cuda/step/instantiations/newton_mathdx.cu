#include "precision.cuh"
#include <cusolverdx.hpp>
#include <cublasdx.hpp>
#include "../numerics/detail/newton_mathdx.cuh"
#include "../numerics/detail/newton_config.cuh"
namespace physics_newton {




// Compile with -dlto --fmad=true to match the NVIDIA device library.
// Called collectively from the same Newton kernel, not a child kernel.


template<int N>
using BlockPotrf = decltype(cusolverdx::Size<N>()
    + cusolverdx::Precision<real_t>()
    + cusolverdx::Type<cusolverdx::type::real>()
    + cusolverdx::Function<cusolverdx::function::potrf>()
    + cusolverdx::FillMode<cusolverdx::fill_mode::lower>()
    + cusolverdx::Arrangement<cusolverdx::row_major>()
    + cusolverdx::SM<DX_SM>()
    + cusolverdx::BlockDim<DX_THREADS>()
    + cusolverdx::Block());

template<int N>
static __device__ inline void potrf(real_t* matrix, int* info) {
    using Solver = BlockPotrf<N>;
    static_assert(Solver::get_shared_memory_size() <= DX_MATRIX_BYTES,
                  "POTRF shared workspace exceeds reserved capacity");
    static_assert(sizeof(typename Solver::status_type) == sizeof(int), "status ABI");
    Solver().execute(matrix, N, info);
}

__device__ void dx_potrf_nv(real_t* A, int* info) { potrf<NV>(A, info); }
__device__ void dx_potrf_1(real_t* A, int* info) { potrf<1>(A, info); }
__device__ void dx_potrf_3(real_t* A, int* info) { potrf<3>(A, info); }
__device__ void dx_potrf_4(real_t* A, int* info) { potrf<4>(A, info); }
__device__ void dx_potrf_6(real_t* A, int* info) { potrf<6>(A, info); }


// Handwritten ABI bridge: standard dense tile multiplication is cuBLASDx.


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
static __device__ void dx_dense_mm(real_t* a,real_t* b,real_t* c,real_t beta) {
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
__device__ void dx_newton_candidates(real_t* a,real_t* b,real_t* c) {
    dx_dense_mm<16,2,NV>(a,b,c,REAL_C(0.0));
}
__device__ void dx_newton_transpose(real_t* a,real_t* b,real_t* c) {
    dx_dense_mm<NV,1,DX_HESSIAN_K>(a,b,c,REAL_C(1.0));
}
__device__ void dx_newton_cone_ltj(real_t* a,real_t* b,real_t* c) {
    dx_dense_mm<6,NV,6>(a,b,c,REAL_C(0.0));
}

// Handwritten ABI bridge. Compile with -dlto --fmad=true.


template<cusolverdx::transpose T>
using DirectionTrsm = decltype(cusolverdx::Size<NV, 1>()
    + cusolverdx::Precision<real_t>()
    + cusolverdx::Type<cusolverdx::type::real>()
    + cusolverdx::Function<cusolverdx::function::trsm>()
    + cusolverdx::Side<cusolverdx::side::left>()
    + cusolverdx::FillMode<cusolverdx::lower>()
    + cusolverdx::TransposeMode<T>()
    + cusolverdx::Diag<cusolverdx::diag::non_unit>()
    + cusolverdx::Arrangement<cusolverdx::row_major, cusolverdx::row_major>()
    + cusolverdx::SM<DX_SM>()
    + cusolverdx::BlockDim<DX_THREADS>()
    + cusolverdx::Block());

template<cusolverdx::transpose T>
static __device__ inline void direction_trsm(real_t* lower, real_t* rhs) {
    using Solver = DirectionTrsm<T>;
    static_assert(Solver::get_shared_memory_size() <= DX_MATRIX_BYTES,
                  "TRSM shared workspace exceeds reserved capacity");
    // Row-major NV by NV factor (lda=NV), NV by 1 RHS (ldb=1).
    Solver().execute(lower, NV, rhs, 1);
}

__device__ void dx_direction_forward(real_t* lower, real_t* rhs) {
    direction_trsm<cusolverdx::non_trans>(lower, rhs);
}
__device__ void dx_direction_backward(real_t* lower, real_t* rhs) {
    direction_trsm<cusolverdx::trans>(lower, rhs);
}

}


namespace physics_newton {
__device__ void dx_direction_potrs(real_t* lower, real_t* rhs) {
    using Solver = decltype(cusolverdx::Size<NV, NV, 1>()
        + cusolverdx::Precision<real_t>() + cusolverdx::Type<cusolverdx::type::real>()
        + cusolverdx::Function<cusolverdx::function::potrs>()
        + cusolverdx::FillMode<cusolverdx::lower>()
        + cusolverdx::Arrangement<cusolverdx::row_major, cusolverdx::row_major>()
        + cusolverdx::SM<DX_SM>() + cusolverdx::BlockDim<DX_THREADS>() + cusolverdx::Block());
    Solver().execute(lower, NV, rhs, 1);
}
}
