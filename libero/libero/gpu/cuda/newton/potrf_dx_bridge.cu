#include "precision.cuh"
// Compile with -dlto --fmad=true to match the NVIDIA device library.
// Called collectively from the same Newton kernel, not a child kernel.
#include <cusolverdx.hpp>

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
__device__ inline void potrf(real_t* matrix, int* info) {
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
