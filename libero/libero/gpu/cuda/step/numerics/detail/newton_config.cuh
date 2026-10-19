#pragma once
#include "../../substep_shape.cuh"
// Short names local to the independently compiled Newton MathDx operators.
// The substep and operators obtain dimensions from the same canonical shape.
namespace physics_newton {
inline constexpr int NV = flash_libero::SubstepShape::nv;
inline constexpr int DX_SM = flash_libero::SubstepShape::sm;
inline constexpr int DX_THREADS = flash_libero::SubstepShape::threads;
inline constexpr int DX_MATRIX_BYTES = flash_libero::SubstepShape::newton_mathdx_cells * sizeof(double);
inline constexpr int DX_HESSIAN_K = flash_libero::SubstepShape::newton_hessian_tile_rows;
} // namespace physics_newton
