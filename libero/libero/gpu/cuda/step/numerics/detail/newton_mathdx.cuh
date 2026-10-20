#include "precision.cuh"
#pragma once
#include <cuda_runtime.h>
namespace physics_newton {
__device__ void dx_direction_potrs(real_t*, real_t*);
__device__ void dx_newton_candidates(real_t*,real_t*,real_t*);
__device__ void dx_potrf_nv(real_t*, int*);
__device__ void dx_potrf_1(real_t*, int*);
__device__ void dx_potrf_3(real_t*, int*);
__device__ void dx_potrf_4(real_t*, int*);
__device__ void dx_potrf_6(real_t*, int*);
__device__ void dx_hessian_gemm(real_t*,real_t*,real_t*);
__device__ void dx_newton_jv(real_t*,real_t*,real_t*);
__device__ void dx_newton_transpose(real_t*,real_t*,real_t*);
__device__ void dx_newton_cone_ltj(real_t*,real_t*,real_t*);
__device__ void dx_direction_forward(real_t* lower, real_t* rhs);
__device__ void dx_direction_backward(real_t* lower, real_t* rhs);
}
