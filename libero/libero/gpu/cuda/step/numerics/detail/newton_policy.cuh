#pragma once
// Private solver policies, independent of fixed-size MathDx instantiations.
#define LIBERO_CONE_HESSIAN_STRATEGY 0
#define LIBERO_CONE_BATCH_THRESHOLD 39
#define LIBERO_RANK_REBUILD_THRESHOLD 0
#define LIBERO_NEWTON_ISLANDS 0
#define LIBERO_HESSIAN_BLAS 1
#define LIBERO_NEWTON_SCALAR_IO 0
#define LIBERO_NEWTON_TRACE 0
#define DATAFLOW_FAST_PROFILE 0

// Offline A/B switch; it changes GEMV organization only.
#ifndef FLASH_NEWTON_NATIVE_GEMV
#define FLASH_NEWTON_NATIVE_GEMV 1
#endif
