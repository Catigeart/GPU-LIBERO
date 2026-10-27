#include "precision.cuh"
#pragma once
#include <cstddef>
#include <cstdint>
#include "substep_shape.cuh"
#include "../control/osc_precision.cuh"

namespace flash_libero {

constexpr std::size_t shared_align(std::size_t bytes) { return (bytes + 31) & ~std::size_t(31); }
constexpr std::size_t shared_vector_bytes = shared_align(SubstepShape::nv * sizeof(real_t));

// Smooth production -> Newton: only force and acceleration survive this
// boundary. Each DOF lane owns the same element in both vectors. The remaining
// smooth operands die before Newton writes its region starting at temporary.
struct SmoothLayout {
  static constexpr std::size_t force = 0, acceleration = shared_vector_bytes;
  static constexpr std::size_t temporary = 2 * shared_vector_bytes;
  static constexpr std::size_t scalar_force = temporary;
  static constexpr std::size_t actuator = shared_align(scalar_force + SubstepShape::nu * sizeof(real_t));
  static constexpr std::size_t smooth = actuator + shared_vector_bytes;
  static constexpr std::size_t seed = smooth + shared_vector_bytes;
  static constexpr std::size_t scratch = seed + shared_vector_bytes;
  static constexpr std::size_t rhs = scratch + shared_vector_bytes;
  static constexpr std::size_t ld = rhs + shared_vector_bytes;
  static constexpr std::size_t inv = shared_align(ld + SubstepShape::packed_mass * sizeof(real_t));
  static constexpr std::size_t sqrt_inv = inv + shared_vector_bytes;
  static constexpr std::size_t pivot = sqrt_inv + shared_vector_bytes;
  static constexpr std::size_t bytes = shared_align(pivot + SubstepShape::nv * sizeof(int));
};

// Euler starts only after forward/Post and the controller have published all
// outputs. Its complete private working set overlays the beginning of shared.
struct EulerLayout {
  static constexpr std::size_t mass = 0;
  static constexpr std::size_t factor = shared_align(SubstepShape::packed_mass * sizeof(real_t));
  static constexpr std::size_t rhs = shared_align(factor + SubstepShape::packed_mass * sizeof(real_t));
  static constexpr std::size_t solution = rhs + shared_vector_bytes;
  static constexpr std::size_t inv = solution + shared_vector_bytes;
  static constexpr std::size_t pos = inv + shared_vector_bytes;
  static constexpr std::size_t vel = shared_align(pos + SubstepShape::nq * sizeof(real_t));
  static constexpr std::size_t pivot = vel + shared_vector_bytes;
  static constexpr std::size_t warnings = pivot + SubstepShape::nv * sizeof(int);
  static constexpr std::size_t last_warning = warnings + sizeof(int);
  static constexpr std::size_t bytes = shared_align(last_warning + sizeof(int));
};

// OSC operands are live together, including the three simultaneous pinv
// problems. Only the phase as a whole overlays Newton/Euler, never its operands.
struct OscLayout {
  static constexpr std::size_t m = 0;
  static constexpr std::size_t lu = shared_align(m + 49 * sizeof(physics_controller::compute_t));
  static constexpr std::size_t mi = shared_align(lu + 49 * sizeof(physics_controller::compute_t));
  static constexpr std::size_t j = shared_align(mi + 49 * sizeof(physics_controller::compute_t));
  static constexpr std::size_t jt = shared_align(j + 42 * sizeof(physics_controller::compute_t));
  static constexpr std::size_t jw = shared_align(jt + 42 * sizeof(physics_controller::compute_t));
  static constexpr std::size_t a = shared_align(jw + 42 * sizeof(physics_controller::compute_t));
  static constexpr std::size_t ap = a + 36 * sizeof(physics_controller::compute_t), ar = ap + 36 * sizeof(physics_controller::compute_t);
  static constexpr std::size_t lf = ar + 36 * sizeof(physics_controller::compute_t), lp = lf + 36 * sizeof(physics_controller::compute_t);
  static constexpr std::size_t lr = lp + 36 * sizeof(physics_controller::compute_t);
  static constexpr std::size_t mjt = lr + 36 * sizeof(physics_controller::compute_t);
  static constexpr std::size_t jbar = shared_align(mjt + 42 * sizeof(physics_controller::compute_t));
  static constexpr std::size_t project = shared_align(jbar + 42 * sizeof(physics_controller::compute_t));
  static constexpr std::size_t delta = project + 49 * sizeof(physics_controller::compute_t), current = delta + 9 * sizeof(physics_controller::compute_t);
  static constexpr std::size_t gr = current + 9 * sizeof(physics_controller::compute_t), gp = gr + 9 * sizeof(physics_controller::compute_t);
  static constexpr std::size_t desired = gp + 3 * sizeof(physics_controller::compute_t), wrench = desired + 6 * sizeof(physics_controller::compute_t);
  static constexpr std::size_t lp3 = wrench + 6 * sizeof(physics_controller::compute_t), lr3 = lp3 + 9 * sizeof(physics_controller::compute_t);
  static constexpr std::size_t pd = lr3 + 9 * sizeof(physics_controller::compute_t), pose = pd + 7 * sizeof(physics_controller::compute_t);
  static constexpr std::size_t nt = pose + 7 * sizeof(physics_controller::compute_t), tau = nt + 7 * sizeof(physics_controller::compute_t);
  static constexpr std::size_t project_t = shared_align(tau + 7 * sizeof(physics_controller::compute_t));
  static constexpr std::size_t info = project_t + 49 * sizeof(physics_controller::compute_t), piv = info + sizeof(int);
  static constexpr std::size_t admitted = piv + 7 * sizeof(int), reuse = admitted + sizeof(int);
  static constexpr std::size_t serial_load = reuse + sizeof(int), pinv_ranks = serial_load + sizeof(int);
  static constexpr std::size_t pinv_ok = pinv_ranks + 3 * sizeof(int);
  static constexpr std::size_t pinv = shared_align(pinv_ok + 3 * sizeof(int));
  static constexpr std::size_t local_state = pinv + 3 * 72 * sizeof(physics_controller::compute_t);
  static constexpr std::size_t live = local_state + 32 * sizeof(real_t), copy_mode = live + sizeof(int);
  static constexpr std::size_t bytes = shared_align(copy_mode + sizeof(int));
};

} // namespace flash_libero
