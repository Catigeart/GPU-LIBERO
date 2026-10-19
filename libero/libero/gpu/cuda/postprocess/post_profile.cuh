#pragma once

// Diagnostic product only: lane-zero timestamps at existing synchronization
// boundaries. Adds neither barriers nor per-contact records to the normal path.
#if defined(FLASH_POST_PROFILE)
#include <cuda_runtime.h>
#include <cstdint>
namespace physics_post::profile {
enum class Phase : int {
  Gate, Validate, Index, Contacts, Acceleration, Inertia, Reverse,
  ValidateOutput, Sensor, Publish, Total, Count
};
constexpr int header_cells = 8, info_cells = 8;
constexpr int record_cells = info_cells + 2 * int(Phase::Count);
struct Configuration { uint64_t *buffer; int capacity; };
extern __device__ Configuration configuration;
__device__ __forceinline__ uint64_t timer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
  return t;
}
__device__ __forceinline__ uint64_t *current() {
  if (threadIdx.x || !configuration.buffer) return nullptr;
  auto *base = configuration.buffer + size_t(blockIdx.y) *
      (header_cells + size_t(configuration.capacity) * record_cells);
  const auto n = base[0];
  return n && n <= uint64_t(configuration.capacity)
      ? base + header_cells + (n - 1) * record_cells : nullptr;
}
__device__ __forceinline__ void add(uint64_t *record, Phase phase, uint64_t t) {
  record[info_cells + 2 * int(phase)] += t;
  ++record[info_cells + 2 * int(phase) + 1];
}
struct Recorder {
  uint64_t *record = nullptr;
  uint64_t start = 0;
  __device__ Recorder(int bodies, int contacts, int levels) {
    if (!threadIdx.x && configuration.buffer) {
      auto *base = configuration.buffer + size_t(blockIdx.y) *
          (header_cells + size_t(configuration.capacity) * record_cells);
      const auto slot = base[0]++;
      if (slot < uint64_t(configuration.capacity)) {
        record = base + header_cells + slot * record_cells;
        record[0] = bodies; record[1] = contacts; record[2] = levels;
        unsigned sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
        record[3] = sm;
        start = timer();
      } else ++base[1];
    }
  }
  __device__ ~Recorder() { if (record) add(record, Phase::Total, timer() - start); }
};
struct Stages {
  uint64_t *record;
  uint64_t start = 0;
  Phase phase;
  __device__ explicit Stages(Phase p) : record(current()), phase(p) {
    if (record) start = timer();
  }
  __device__ void next(Phase p) {
    if (record) { add(record, phase, timer() - start); phase = p; start = timer(); }
  }
  __device__ ~Stages() { if (record) add(record, phase, timer() - start); }
};
} // namespace physics_post::profile
#define CF_POST_PROFILE_BEGIN(b, c, l) physics_post::profile::Recorder post_record(b, c, l)
#define CF_POST_PROFILE_STAGES(p) physics_post::profile::Stages post_stages(physics_post::profile::Phase::p)
#define CF_POST_PROFILE_NEXT(p) post_stages.next(physics_post::profile::Phase::p)
#else
#define CF_POST_PROFILE_BEGIN(...) ((void)0)
#define CF_POST_PROFILE_STAGES(...) ((void)0)
#define CF_POST_PROFILE_NEXT(...) ((void)0)
#endif
