#pragma once

// Compiled only into flash_substep_newton_profile. Normal numerical objects
// contain neither timers, profiling branches, nor a profiling device symbol.
#if defined(FLASH_NEWTON_PROFILE)
#include <cuda_runtime.h>
#include <cstdint>

namespace physics_newton::profile_detail {
enum class Phase : int {
  Entry, WarmJacobian, WarmCosts, InitialResiduals, InitialConstraint,
  InitialTranspose, InitialAssembly, InitialFactor, InitialDirection,
  LineSearch, CommitSave, IterConstraint, IterTranspose, IterPrepare,
  IterFactor, IterDirection, Statistics, Finalize, Total,
  EntrySchema, Coefficients, SearchPrepare, SearchJacobian, SearchContinue,
  SearchPoint, ChangedRows, RankUpdates, Rebuild, ConeFactor, RankUpdate, Potrf, SearchQuadratic, JacobianCache,
  Count
};
enum class Info : int {
  Rows, Contacts, FrictionRows, Iterations, SearchEvaluations, CoefficientRefresh,
  StopReason, Error, Sm, Enabled, LayoutRefresh, InitialConeRows, WarmChoice,
  InitialConeMaxForceBits, InitialGradientNormBits, InitialDecrementBits, Count
};
constexpr int header_cells = 16, info_cells = 16;
static_assert(int(Info::Count)<=info_cells,"Profile info exceeds its reserved header");
constexpr int record_cells = info_cells + 2 * int(Phase::Count);
struct Configuration { uint64_t *buffer; int capacity; };
extern __device__ Configuration configuration;

__device__ __forceinline__ uint64_t timer() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value) :: "memory");
  return value;
}
__device__ __forceinline__ uint64_t *current_record() {
  if (threadIdx.x != 0 || !configuration.buffer)
    return nullptr;
  uint64_t *base = configuration.buffer + size_t(blockIdx.y) *
      (header_cells + size_t(configuration.capacity) * record_cells);
  const uint64_t count = base[0];
  return count && count <= uint64_t(configuration.capacity)
      ? base + header_cells + (count - 1) * record_cells : nullptr;
}
__device__ __forceinline__ void add_phase(uint64_t *record, Phase phase, uint64_t elapsed) {
  record[info_cells + 2 * int(phase)] += elapsed;
  ++record[info_cells + 2 * int(phase) + 1];
}
struct Recorder {
  uint64_t *record = nullptr;
  uint64_t start = 0, last = 0;
  Phase phase = Phase::Entry;
  __device__ __forceinline__ explicit Recorder(bool enabled) {
    if (threadIdx.x == 0 && configuration.buffer) {
      uint64_t *base = configuration.buffer + size_t(blockIdx.y) *
          (header_cells + size_t(configuration.capacity) * record_cells);
      // No numerical helper runs before Entry has opened its record. Disabled
      // solves are still recorded, making nested scope ownership unambiguous.
      const uint64_t slot = base[0]++;
      if (slot < uint64_t(configuration.capacity)) {
        record = base + header_cells + slot * record_cells;
        record[int(Info::Enabled)] = enabled;
        unsigned sm;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
        record[int(Info::Sm)] = sm;
        start = last = timer();
      } else {
        ++base[1];
      }
    }
  }
  __device__ __forceinline__ void stage(Phase next) {
    if (record) {
      const uint64_t now = timer();
      add_phase(record, phase, now - last);
      phase = next;
      last = timer(); // exclude this marker's stores from the next stage
    }
  }
  __device__ __forceinline__ void value(Info field, uint64_t value, bool add = false) {
    if (record) {
      if (add) record[int(field)] += value;
      else record[int(field)] = value;
    }
  }
  __device__ __forceinline__ void finish() {
    if (record) {
      const uint64_t now = timer();
      add_phase(record, phase, now - last);
      add_phase(record, Phase::Total, now - start);
    }
  }
};
struct Scope {
  uint64_t *record;
  uint64_t start = 0;
  Phase phase;
  __device__ __forceinline__ explicit Scope(Phase value) : record(current_record()), phase(value) {
    if (record) start = timer();
  }
  __device__ __forceinline__ ~Scope() {
    if (record) add_phase(record, phase, timer() - start);
  }
};
} // namespace physics_newton::profile_detail

#define CF_NEWTON_PROFILE_BEGIN(enabled) physics_newton::profile_detail::Recorder profile_record(enabled)
#define CF_NEWTON_PROFILE_STAGE(phase) profile_record.stage(physics_newton::profile_detail::Phase::phase)
#define CF_NEWTON_PROFILE_VALUE(field, sample) profile_record.value(physics_newton::profile_detail::Info::field, sample)
#define CF_NEWTON_PROFILE_ADD(field, sample) profile_record.value(physics_newton::profile_detail::Info::field, sample, true)
#define CF_NEWTON_PROFILE_END() profile_record.finish()
#define CF_NEWTON_PROFILE_SCOPE(phase) physics_newton::profile_detail::Scope profile_scope(physics_newton::profile_detail::Phase::phase)
#else
#define CF_NEWTON_PROFILE_BEGIN(...) ((void)0)
#define CF_NEWTON_PROFILE_STAGE(...) ((void)0)
#define CF_NEWTON_PROFILE_VALUE(...) ((void)0)
#define CF_NEWTON_PROFILE_ADD(...) ((void)0)
#define CF_NEWTON_PROFILE_END() ((void)0)
#define CF_NEWTON_PROFILE_SCOPE(...) ((void)0)
#endif
