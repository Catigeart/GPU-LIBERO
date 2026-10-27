#pragma once
#include "native_layout.cuh"
#include "precision.cuh"
#ifndef FLASH_SUBSTEP_SM
#define FLASH_SUBSTEP_SM 800
#endif
using NativeShape = flash_libero::LiberoModel<FLASH_SUBSTEP_NV, FLASH_SUBSTEP_NQ, FLASH_SUBSTEP_NM, FLASH_SUBSTEP_NJ, FLASH_SUBSTEP_NU, FLASH_SUBSTEP_SM, 128>;
using NativeLayout = libero_native::NativeNewtonLayout<NativeShape>;
static_assert(sizeof(real_t) == 8, "The native catalogue entry is FP64");
#define NV FLASH_SUBSTEP_NV
#define NM FLASH_SUBSTEP_NM
#define ROWS (30000 + FLASH_SUBSTEP_NV + FLASH_SUBSTEP_NJ)
#define DX_SM FLASH_SUBSTEP_SM
#define DX_THREADS 128
#define NEWTON_BLOCK_THREADS 128
#define DX_MATRIX_BYTES NativeLayout::dx_bytes
#define DX_HESSIAN_K (NV <= 43 ? 16 : 8)
#define LIBERO_CONE_HESSIAN_STRATEGY 2
#define LIBERO_CONE_BATCH_THRESHOLD NV
#define LIBERO_RANK_REBUILD_THRESHOLD (NV / 2)
#define LIBERO_NEWTON_ISLANDS 0
#define LIBERO_HESSIAN_BLAS 1
#define LIBERO_NEWTON_SCALAR_IO (NV == 21 || NV == 43)
#define DATAFLOW_FAST_PROFILE 1
#define DIRECTION_BASE_SHARED (NativeLayout::H >= 0)
#define DIRECTION_CONE_SHARED (NativeLayout::Hcone >= 0)
#define CF_S_KBIP_PLANES 1
