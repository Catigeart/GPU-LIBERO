#include "precision.cuh"
#pragma once
#include <cstddef>
#include <cstdint>
#include "substep_shape.cuh"
#include "newton_jacobian_layout.cuh"

namespace physics_newton {
// One persistent, world-local global allocation. The header records the
// successful position-prefix generation; the three planes cover every row.
// This is separate from the short-lived shared Newton/MathDx workspace.
struct NewtonCoefficientLayout {
  static constexpr std::size_t alignment = 128;
  static constexpr std::size_t generation = 0;
  // Layout is published independently: coefficients alone do not prove that
  // contact projection and all head/tile metadata passed their entry guards.
  static constexpr std::size_t layout_generation = 8;
  static constexpr std::size_t layout_valid = 16;
  static constexpr std::size_t layout_counts = 20; // nefc, ne, nf, ncon
  static constexpr std::size_t row_bytes =
      (flash_libero::SubstepShape::rows * sizeof(real_t) + alignment - 1) / alignment * alignment;
  static constexpr std::size_t sqrt_d = alignment;
  static constexpr std::size_t friction_limit = sqrt_d + row_bytes;
  static constexpr std::size_t cone_dm = friction_limit + row_bytes;
  static constexpr std::size_t bytes = cone_dm + row_bytes;
};
// Ephemeral reduction scratch follows persistent coefficient planes. Transpose
// and Gram lifetimes do not overlap, so both alias one packed-triangle region.
// The host allocates this world pitch even for the ordered comparison product.
struct NewtonDeviceWorkspaceLayout {
  static constexpr std::size_t atomic = NewtonCoefficientLayout::bytes;
  static constexpr std::size_t atomic_cells =
      flash_libero::SubstepShape::nv * (flash_libero::SubstepShape::nv + 1) / 2;
  static constexpr std::size_t jacobian =
      (atomic + atomic_cells * sizeof(real_t) + NewtonCoefficientLayout::alignment - 1) /
      NewtonCoefficientLayout::alignment * NewtonCoefficientLayout::alignment;
  static constexpr std::size_t bytes = jacobian + NewtonJacobianLayout::bytes(
      flash_libero::SubstepShape::rows, flash_libero::SubstepShape::nv);
};
static_assert(NewtonCoefficientLayout::bytes % NewtonCoefficientLayout::alignment == 0,
              "Every world's coefficient planes must remain aligned");
static_assert(sizeof(uint64_t) <= NewtonCoefficientLayout::sqrt_d, "Coefficient header overlap");
} // namespace physics_newton
