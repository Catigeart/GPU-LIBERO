#include "precision.cuh"
#pragma once
#include "substep_shared_layout.cuh"

// Private Newton storage and the host launch capacity. The outer device kernel
// also uses qacc/mathdx offsets when handing the completed solve to Post.
namespace flash_libero {
// Vector pitches and compact matrices are derived from the selected model.
// These regions coexist throughout a solve. Post retains qacc while reusing
// mathdx; its final acceleration publication ends the Newton lifetime.
struct NewtonLayout {
  static constexpr int hessian_tile_rows = SubstepShape::newton_hessian_tile_rows;
  static constexpr int mathdx_cells = SubstepShape::newton_mathdx_cells;
  static constexpr int mathdx_bytes = mathdx_cells * sizeof(real_t);
  static constexpr std::size_t base = SmoothLayout::temporary;
  static constexpr std::size_t qacc = base, search = qacc + shared_vector_bytes;
  static constexpr std::size_t gradient = search + shared_vector_bytes;
  static constexpr std::size_t preconditioned_gradient = gradient + shared_vector_bytes;
  static constexpr std::size_t mass_acceleration = preconditioned_gradient + shared_vector_bytes;
  static constexpr std::size_t mass_search = mass_acceleration + shared_vector_bytes;
  static constexpr std::size_t solution = mass_search + shared_vector_bytes;
  static constexpr std::size_t gradient_scratch = solution + shared_vector_bytes;
  static constexpr std::size_t forward_solution = gradient_scratch + shared_vector_bytes;
  static constexpr std::size_t update_vector = forward_solution + shared_vector_bytes;
  static constexpr std::size_t local_factor = update_vector + shared_vector_bytes;
  static constexpr std::size_t local_product = local_factor + 36 * sizeof(real_t);
  static constexpr std::size_t hessian = shared_align(local_product + 6 * SubstepShape::nv * sizeof(real_t));
  static constexpr std::size_t mathdx =
      (hessian + SubstepShape::nv * SubstepShape::nv * sizeof(real_t) + 15) & ~std::size_t(15);
  static constexpr std::size_t control = mathdx + mathdx_bytes;
  static constexpr std::size_t workspace_bytes = control - base;
  static constexpr std::size_t control_bytes = ((40 * sizeof(int) + 11 * sizeof(real_t) + 7) & ~std::size_t(7)) + sizeof(int64_t);
  static constexpr std::size_t warmstart = shared_align(control + control_bytes);
  static constexpr std::size_t constraint_force = warmstart + shared_vector_bytes;
  static constexpr std::size_t warm_mass = constraint_force + shared_vector_bytes;
  // Row coefficients have a persistent global owner, outside this arena.
  static constexpr std::size_t bytes = warm_mass + shared_vector_bytes;
};

constexpr std::size_t substep_shared_bytes = NewtonLayout::bytes;
static_assert(SmoothLayout::bytes <= substep_shared_bytes && EulerLayout::bytes <= substep_shared_bytes &&
                  OscLayout::bytes <= substep_shared_bytes,
              "Phase workspace exceeds the shared allocation");
static_assert(NewtonLayout::local_product + 6 * SubstepShape::nv * sizeof(real_t) <= NewtonLayout::hessian,
              "Newton tile overlap");
static_assert(NewtonLayout::hessian + SubstepShape::nv * SubstepShape::nv * sizeof(real_t) <=
                  NewtonLayout::mathdx,
              "Hessian overlaps MathDx staging");
static_assert(NewtonLayout::mathdx + NewtonLayout::mathdx_bytes <= NewtonLayout::control,
              "MathDx exceeds staging");
static_assert(NewtonLayout::mathdx % 16 == 0 && NewtonLayout::hessian % 16 == 0, "MathDx operand alignment");
} // namespace flash_libero

