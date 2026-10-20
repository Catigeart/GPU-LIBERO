#include "precision.cuh"
#pragma once
#include <cstddef>
#include <cstdint>

namespace physics_newton {
// Bounded per-world global storage, independent of block size. A full cache is
// optional: oversized inputs retain the dense source and existing reductions.
// Row entries are split into two warp-owned pools, avoiding a shared/global
// allocation counter and a block scan of every live row.
struct NewtonJacobianLayout {
  static constexpr int max_entries = 65536;
  static constexpr std::size_t align(std::size_t n) { return (n + 127) / 128 * 128; }
  static constexpr int entries(int rows, int nv) {
    return (rows + 1) / 2 < max_entries / (2 * nv) ? (rows + 1) / 2 * nv * 2 : max_entries;
  }
  static constexpr std::size_t generation = 0; // uint64_t
  static constexpr std::size_t rows = 8;       // int
  static constexpr std::size_t used = 12;      // two int, negative => fallback
  static constexpr std::size_t products_valid = 20;
  static constexpr std::size_t csc_valid = 24;
  static constexpr int product_slots = FLASH_REAL_BITS == 32 ? 256 : 128; // bounded global slots per packed H cell
  static constexpr std::size_t column_counts = 128;
  static constexpr std::size_t column_offsets(int nv) { return column_counts + nv * sizeof(int); }
  static constexpr std::size_t column_cursors(int nv) { return column_offsets(nv) + (nv + 1) * sizeof(int); }
  static constexpr std::size_t row_records(int nv) { return align(column_cursors(nv) + nv * sizeof(int)); }
  static constexpr std::size_t values(int rows, int nv) { return align(row_records(nv) + rows * 2 * sizeof(int)); }
  static constexpr std::size_t weighted(int rows, int nv) { return values(rows, nv) + entries(rows, nv) * sizeof(real_t); }
  static constexpr std::size_t columns(int rows, int nv) { return weighted(rows, nv) + entries(rows, nv) * sizeof(real_t); }
  static constexpr std::size_t csc_rows(int rows, int nv) { return columns(rows, nv) + entries(rows, nv) * sizeof(int); }
  static constexpr std::size_t csc_entries(int rows, int nv) { return csc_rows(rows, nv) + entries(rows, nv) * sizeof(int); }
  static constexpr std::size_t product_counts(int rows, int nv) { return align(csc_entries(rows, nv) + entries(rows, nv) * sizeof(int)); }
  static constexpr std::size_t product_values(int rows, int nv) { return align(product_counts(rows, nv) + nv * (nv + 1) / 2 * sizeof(int)); }
  static constexpr std::size_t product_rows(int rows, int nv) { return product_values(rows, nv) + std::size_t(nv) * (nv + 1) / 2 * product_slots * sizeof(real_t); }
  static constexpr std::size_t bytes(int rows, int nv) { return align(product_rows(rows, nv) + std::size_t(nv) * (nv + 1) / 2 * product_slots * sizeof(int)); }
};
} // namespace physics_newton
