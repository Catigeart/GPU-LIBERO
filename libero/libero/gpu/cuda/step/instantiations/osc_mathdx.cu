#include "precision.cuh"
#include <cusolverdx.hpp>
#include <cublasdx.hpp>
#include "../substep_shape.cuh"
namespace physics_controller {



// All operands reside in shared memory. A/B/C are dense row-major;
// transposes are materialized once in the caller for reuse across GEMMs.
template<int M,int N,int K>
static __device__ void osc_gemm(real_t* a,real_t* b,real_t* c) {
    using Op=decltype(cublasdx::Size<M,N,K>()+
        cublasdx::Precision<real_t>()+cublasdx::Type<cublasdx::type::real>()+
        cublasdx::Arrangement<cublasdx::row_major,cublasdx::row_major,cublasdx::row_major>()+
        cublasdx::Alignment<sizeof(real_t),sizeof(real_t),sizeof(real_t)>()+cublasdx::Function<cublasdx::function::MM>()+
        cublasdx::SM<flash_libero::SubstepShape::sm>()+
        cublasdx::BlockDim<flash_libero::SubstepShape::threads>()+cublasdx::Block());
    // Explicit initialization avoids depending on a vendor beta==0 shortcut
    // when a shared output region has never held an initialized value.
    for(int z=threadIdx.x;z<M*N;z+=blockDim.x)c[z]=0;
    __syncthreads();
    Op().execute(REAL_C(1.0),a,b,REAL_C(0.0),c);
    __syncthreads();
}
#define OSC_MM(M,N,K) extern "C" __device__ void osc_mm_##M##_##N##_##K(real_t*a,real_t*b,real_t*c){osc_gemm<M,N,K>(a,b,c);}
OSC_MM(6,7,7) OSC_MM(6,6,7) OSC_MM(7,6,7) OSC_MM(7,6,6)
OSC_MM(7,7,6) OSC_MM(3,3,3) OSC_MM(6,1,6) OSC_MM(3,1,3)
OSC_MM(7,1,7) OSC_MM(7,1,6)

 extern "C" __device__ void osc_inverse7(real_t* a,real_t* inverse,int* piv,int* info) {
    using Op=decltype(cusolverdx::Size<7,7,7>()+cusolverdx::Precision<real_t>()+
        cusolverdx::Type<cusolverdx::type::real>()+
        cusolverdx::Function<cusolverdx::function::gesv_partial_pivot>()+
        cusolverdx::Arrangement<cusolverdx::row_major,cusolverdx::row_major>()+
        cusolverdx::SM<flash_libero::SubstepShape::sm>()+
        cusolverdx::BlockDim<flash_libero::SubstepShape::threads>()+cusolverdx::Block());
    Op().execute(a,7,piv,inverse,7,info);
    __syncthreads();
}

OSC_MM(6,1,7)
template<int N,int RHS>
static __device__ void osc_solve(real_t* a,real_t* b,int* piv,int* info){
    using Op=decltype(cusolverdx::Size<N,N,RHS>()+cusolverdx::Precision<real_t>()+
        cusolverdx::Type<cusolverdx::type::real>()+
        cusolverdx::Function<cusolverdx::function::gesv_partial_pivot>()+
        cusolverdx::Arrangement<cusolverdx::row_major,cusolverdx::row_major>()+
        cusolverdx::SM<flash_libero::SubstepShape::sm>()+cusolverdx::BlockDim<flash_libero::SubstepShape::threads>()+cusolverdx::Block());
    Op().execute(a,N,piv,b,RHS,info);__syncthreads();
}
 extern "C" __device__ void osc_solve7x6(real_t*a,real_t*b,int*p,int*i){osc_solve<7,6>(a,b,p,i);}
 extern "C" __device__ void osc_task_inverse6(real_t*a,real_t*b,int*p,int*i){osc_solve<6,6>(a,b,p,i);}
 extern "C" __device__ void osc_task_inverse3(real_t*a,real_t*b,int*p,int*i){osc_solve<3,3>(a,b,p,i);}

}

