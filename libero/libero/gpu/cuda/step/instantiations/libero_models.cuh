#pragma once
#include "../substep_parameters.cuh"
namespace flash_libero {
template <int Nv, int Sm = 800, int Threads = 128> struct LiberoModel;
#define LIBERO_MODEL(nv, nq, nm, nj, nu)                                                           \
  template <int Sm, int Threads>                                                                   \
  struct LiberoModel<nv, Sm, Threads> : SubstepDimensions<nv, nq, nm, nj, nu, 5000, Sm, Threads> { \
  };
#include "libero_models.def"
#undef LIBERO_MODEL
} // namespace flash_libero
