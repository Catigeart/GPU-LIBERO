#include "precision.cuh"
#include "../../control/osc_precision.cuh"
#include <cusolverdx.hpp>
#include <cublasdx.hpp>
#include "../substep_shape.cuh"

namespace physics_controller {
// Only maintained OSC operations are instantiated here. The retained inline
// component path has its own real_t bridge in cuda/control/osc_bridge.cu.
template<int M, int N, int K>
static __device__ void osc_gemm(compute_t *a, compute_t *b, compute_t *c) {
    using Op = decltype(cublasdx::Size<M,N,K>() +
        cublasdx::Precision<compute_t>() + cublasdx::Type<cublasdx::type::real>() +
        cublasdx::Arrangement<cublasdx::row_major,cublasdx::row_major,cublasdx::row_major>() +
        cublasdx::Alignment<sizeof(compute_t),sizeof(compute_t),sizeof(compute_t)>() +
        cublasdx::Function<cublasdx::function::MM>() +
        cublasdx::SM<flash_libero::SubstepShape::sm>() +
        cublasdx::BlockDim<flash_libero::SubstepShape::threads>() + cublasdx::Block());
    // Do not rely on a vendor beta==0 shortcut for uninitialized shared memory.
    for (int z = threadIdx.x; z < M*N; z += blockDim.x) c[z] = 0;
    __syncthreads();
    Op().execute(compute_t(1), a, b, compute_t(0), c);
    __syncthreads();
}
#define OSC_MM(M,N,K) extern "C" __device__ void osc_fp64_mm_##M##_##N##_##K(compute_t*a,compute_t*b,compute_t*c){osc_gemm<M,N,K>(a,b,c);}
OSC_MM(6,6,7) OSC_MM(7,6,7) OSC_MM(7,6,6)
OSC_MM(7,7,6) OSC_MM(3,3,3)
#undef OSC_MM

extern "C" __device__ void osc_fp64_inverse7(compute_t *a, compute_t *inverse, int *piv, int *info) {
    using Op = decltype(cusolverdx::Size<7,7,7>() + cusolverdx::Precision<compute_t>() +
        cusolverdx::Type<cusolverdx::type::real>() +
        cusolverdx::Function<cusolverdx::function::gesv_partial_pivot>() +
        cusolverdx::Arrangement<cusolverdx::row_major,cusolverdx::row_major>() +
        cusolverdx::SM<flash_libero::SubstepShape::sm>() +
        cusolverdx::BlockDim<flash_libero::SubstepShape::threads>() + cusolverdx::Block());
    Op().execute(a, 7, piv, inverse, 7, info);
    __syncthreads();
}
}
