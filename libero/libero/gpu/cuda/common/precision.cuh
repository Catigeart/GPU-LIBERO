// One scalar policy for the maintained CUDA implementation and its ABI.
#ifndef FLASH_LIBERO_PRECISION_CUH
#define FLASH_LIBERO_PRECISION_CUH
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>
#include <cstring>

#ifndef FLASH_REAL_BITS
#define FLASH_REAL_BITS 64
#endif
#if FLASH_REAL_BITS == 32
using real_t = float;
using real2_t = float2;
using real_bits_t = uint32_t;
#define REAL_EPSILON FLT_EPSILON
#define REAL_MAX FLT_MAX
#define REAL_SAFE_MAGNITUDE real_t(1e10)
#define REAL_NORM_LIMIT FLT_MAX
#define REAL_PTX_TYPE "f32"
#define REAL_PTX_OUT "=f"
#define REAL_PTX_IN "f"
#else
static_assert(FLASH_REAL_BITS == 64, "FLASH_REAL_BITS must be 32 or 64");
using real_t = double;
using real2_t = double2;
using real_bits_t = uint64_t;
#define REAL_EPSILON DBL_EPSILON
#define REAL_MAX DBL_MAX
#define REAL_SAFE_MAGNITUDE real_t(1e100)
#define REAL_NORM_LIMIT real_t(1e300)
#define REAL_PTX_TYPE "f64"
#define REAL_PTX_OUT "=d"
#define REAL_PTX_IN "d"
#endif
#define REAL_C(value) real_t(value)

__device__ __forceinline__ real_t real_linesearch_tolerance(real_t value) {
#if FLASH_REAL_BITS == 32
  // MuJoCo Warp's FP32 derivative stopping floor; FP64 keeps MuJoCo 2.3.7.
  return fmaxf(value, 1e-6f);
#else
  return value;
#endif
}

__host__ __device__ inline real2_t make_real2(real_t x, real_t y) { return {x, y}; }
__device__ __forceinline__ real_bits_t real_bits(real_t value) {
#if FLASH_REAL_BITS == 32
  return __float_as_uint(value);
#else
  return static_cast<uint64_t>(__double_as_longlong(value));
#endif
}
__device__ __forceinline__ real_t real_from_bits(real_bits_t value) {
#if FLASH_REAL_BITS == 32
  return __uint_as_float(value);
#else
  return __longlong_as_double(static_cast<long long>(value));
#endif
}

#if FLASH_REAL_BITS == 32
#define FLASH_REAL_INTRINSIC(name) __f##name
#else
#define FLASH_REAL_INTRINSIC(name) __d##name
#endif
#define FLASH_REAL_BINARY(name) \
  __device__ __forceinline__ real_t real_##name(real_t a, real_t b) { \
    return FLASH_REAL_INTRINSIC(name)(a, b); \
  }
FLASH_REAL_BINARY(add_rd)
FLASH_REAL_BINARY(add_rn)
FLASH_REAL_BINARY(add_ru)
FLASH_REAL_BINARY(sub_rn)
FLASH_REAL_BINARY(mul_rd)
FLASH_REAL_BINARY(mul_rn)
FLASH_REAL_BINARY(mul_ru)
FLASH_REAL_BINARY(div_rn)
#undef FLASH_REAL_BINARY
__device__ __forceinline__ real_t real_sqrt_rn(real_t x) { return FLASH_REAL_INTRINSIC(sqrt_rn)(x); }
__device__ __forceinline__ real_t real_sqrt_ru(real_t x) { return FLASH_REAL_INTRINSIC(sqrt_ru)(x); }
#undef FLASH_REAL_INTRINSIC
#endif // FLASH_LIBERO_PRECISION_CUH
