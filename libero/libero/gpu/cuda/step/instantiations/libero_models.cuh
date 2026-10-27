#pragma once
#include "../substep_parameters.cuh"
namespace flash_libero {
template <int Nv, int Nq, int Nm, int Nj, int Nu, int Sm, int Threads>
using LiberoModel = SubstepDimensions<Nv, Nq, Nm, Nj, Nu, 5000, Sm, Threads>;
} // namespace flash_libero
