#pragma once

namespace flash_libero {
// One numerical source, explicitly instantiated offline for the task catalogue.
// Row capacity is a pitch; every numerical loop uses the current live count.
template <int NV, int NQ, int NM, int NJ, int NU = 9, int CONTACTS = 5000,
          int SM = 800, int THREADS = 64>
struct SubstepDimensions {
  static constexpr int nv = NV, nq = NQ, packed_mass = NM, njnt = NJ, nu = NU;
  static constexpr int contacts = CONTACTS, rows = 6 * CONTACTS + NV + NJ;
  static constexpr int quaternion_joints = NQ - NV;
  static constexpr int threads = THREADS, sm = SM;
  static constexpr int newton_hessian_tile_rows = NV <= 43 ? 16 : 8;
  static constexpr int newton_mathdx_cells =
      (NV * NV + 2 * NV * newton_hessian_tile_rows + 1) / 2 * 2;
  static_assert(NV > 0 && NQ >= NV && NM >= NV && NJ > 0, "Invalid model dimensions");
};
} // namespace flash_libero
