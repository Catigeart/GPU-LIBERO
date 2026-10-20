#include "precision.cuh"
// Handwritten ABI bridge. Compile with -dlto --fmad=true.
#include <cusolverdx.hpp>

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
__device__ inline void direction_trsm(real_t* lower, real_t* rhs) {
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
