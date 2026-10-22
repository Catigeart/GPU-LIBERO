#pragma once
#include "substep_shared_layout.cuh"
namespace flash_libero {
struct StagedPostLayout {
  static constexpr std::size_t qacc = 0, mathdx = shared_vector_bytes;
};
// Post retains qacc beside three spatial body arrays and the six force/torque
// sensor outputs, plus optional raw subtree forces. The later controller and
// Euler arenas reuse the same bytes.
constexpr std::size_t staged_post_bytes(int bodies) {
  const std::size_t raw = 6 * std::size_t(bodies);
  const std::size_t temporary = raw > 288 ? raw : 288;
  const auto post = shared_align(StagedPostLayout::mathdx +
      (18 * std::size_t(bodies) + 6 + temporary) * sizeof(real_t));
  const auto tail = OscLayout::bytes > EulerLayout::bytes ? OscLayout::bytes : EulerLayout::bytes;
  return post > tail ? post : tail;
}
} // namespace flash_libero
