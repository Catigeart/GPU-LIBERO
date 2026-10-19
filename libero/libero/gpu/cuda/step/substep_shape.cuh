#pragma once
#include "substep_parameters.cuh"

// CMake passes these dimensions to each independent catalogue instance. The
// defaults retain the original NV39 development target and host layout tools.
#ifndef FLASH_SUBSTEP_NV
#define FLASH_SUBSTEP_NV 39
#define FLASH_SUBSTEP_NQ 44
#define FLASH_SUBSTEP_NM 149
#define FLASH_SUBSTEP_NJ 14
#define FLASH_SUBSTEP_NU 9
#define FLASH_SUBSTEP_CONTACTS 5000
#endif
#ifndef FLASH_SUBSTEP_SM
#define FLASH_SUBSTEP_SM 800
#endif
#ifndef FLASH_SUBSTEP_THREADS
#define FLASH_SUBSTEP_THREADS 64
#endif

namespace flash_libero {
// Shared by the maintained substep and the independent MathDx compilation
// units. Keep storage offsets out of this header: layout-only edits must not
// invalidate the cached vendor instantiations.
using SubstepShape = SubstepDimensions<FLASH_SUBSTEP_NV, FLASH_SUBSTEP_NQ,
    FLASH_SUBSTEP_NM, FLASH_SUBSTEP_NJ, FLASH_SUBSTEP_NU, FLASH_SUBSTEP_CONTACTS,
    FLASH_SUBSTEP_SM, FLASH_SUBSTEP_THREADS>;
} // namespace flash_libero
