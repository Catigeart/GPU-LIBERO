#pragma once
#include <cuda_runtime.h>
namespace physics_newton {
__device__ void dx_direction_potrs(double*, double*);
__device__ void dx_newton_candidates(double*,double*,double*);
__device__ void dx_potrf_nv(double*, int*);
__device__ void dx_potrf_1(double*, int*);
__device__ void dx_potrf_3(double*, int*);
__device__ void dx_potrf_4(double*, int*);
__device__ void dx_potrf_6(double*, int*);
__device__ void dx_hessian_gemm(double*,double*,double*);
__device__ void dx_newton_jv(double*,double*,double*);
__device__ void dx_newton_transpose(double*,double*,double*);
__device__ void dx_newton_cone_ltj(double*,double*,double*);
__device__ void dx_direction_forward(double* lower, double* rhs);
__device__ void dx_direction_backward(double* lower, double* rhs);
}
