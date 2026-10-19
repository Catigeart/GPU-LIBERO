#pragma once

// Optional product only. Timestamps use the existing synchronization points;
// the ordinary controller contains no recorder or profiling branch.
#if defined(FLASH_OSC_PROFILE)
#include <cuda_runtime.h>
#include <cstdint>
namespace physics_controller::profile {
enum class Phase : int {
  Gather,
  Seed,
  SolveEnvelope,
  Solve,
  Commit,
  State,
  Load,
  CacheLookup,
  CacheRestore,
  MassInverse,
  OperationalMatrix,
  Pseudoinverse,
  Projector,
  CacheStore,
  DiagnosticPublish,
  Goal,
  Torque,
  Total,
  Count
};
constexpr int header_cells = 8, info_cells = 12;
constexpr int record_cells = info_cells + 2 * int(Phase::Count);
struct Configuration {
  uint64_t *buffer;
  int capacity;
};
extern __device__ Configuration configuration;
__device__ inline uint64_t timer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)::"memory");
  return t;
}
__device__ inline uint64_t *current_any_lane() {
  if (!configuration.buffer)
    return nullptr;
  auto *base = configuration.buffer +
               size_t(blockIdx.y) * (header_cells + size_t(configuration.capacity) * record_cells);
  const auto n = base[0];
  return n && n <= uint64_t(configuration.capacity) ? base + header_cells + (n - 1) * record_cells : nullptr;
}
__device__ inline uint64_t *current() { return threadIdx.x ? nullptr : current_any_lane(); }
__device__ inline void add(uint64_t *r, Phase p, uint64_t t) {
  r[info_cells + 2 * int(p)] += t;
  ++r[info_cells + 2 * int(p) + 1];
}
struct Recorder {
  uint64_t *record = nullptr;
  uint64_t start = 0;
  __device__ Recorder() {
    if (!threadIdx.x && configuration.buffer) {
      auto *base = configuration.buffer +
                   size_t(blockIdx.y) * (header_cells + size_t(configuration.capacity) * record_cells);
      const auto slot = base[0]++;
      if (slot < uint64_t(configuration.capacity)) {
        record = base + header_cells + slot * record_cells;
        unsigned sm;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
        record[7] = sm;
        start = timer();
      } else
        ++base[1];
    }
  }
  __device__ ~Recorder() {
    if (record)
      add(record, Phase::Total, timer() - start);
  }
};
struct Stages {
  uint64_t *record;
  uint64_t start = 0;
  Phase phase;
  __device__ explicit Stages(Phase p) : record(current()), phase(p) {
    if (record)
      start = timer();
  }
  __device__ void next(Phase p) {
    if (record) {
      add(record, phase, timer() - start);
      phase = p;
      start = timer();
    }
  }
  __device__ ~Stages() {
    if (record)
      add(record, phase, timer() - start);
  }
};
__device__ inline void pinv_result(int sweeps, int rotations) {
  // pinv_group leaders are threads 0, 32 and 48, with disjoint metadata cells.
  if (auto *r = current_any_lane()) {
    const int problem = threadIdx.x < 32 ? 0 : 1 + (threadIdx.x - 32) / 16;
    r[1 + problem] = sweeps;
    r[4 + problem] = rotations;
  }
}
} // namespace physics_controller::profile
#define CF_OSC_JOIN_(a, b) a##b
#define CF_OSC_JOIN(a, b) CF_OSC_JOIN_(a, b)
#define CF_OSC_PROFILE_BEGIN() physics_controller::profile::Recorder osc_record
#define CF_OSC_PROFILE_STAGES(p)                                                                             \
  physics_controller::profile::Stages osc_stages(physics_controller::profile::Phase::p)
#define CF_OSC_PROFILE_NEXT(p) osc_stages.next(physics_controller::profile::Phase::p)
#define CF_OSC_PROFILE_SCOPE(p)                                                                              \
  physics_controller::profile::Stages CF_OSC_JOIN(osc_scope_, __LINE__)(physics_controller::profile::Phase::p)
#define CF_OSC_PROFILE_META(i, v)                                                                            \
  do {                                                                                                       \
    if (auto *r = physics_controller::profile::current())                                                    \
      r[i] = (v);                                                                                            \
  } while (0)
#define CF_OSC_PROFILE_PINV_COUNTERS() int profile_sweeps = 0, profile_rotations = 0
#define CF_OSC_PROFILE_SWEEP() (++profile_sweeps)
#define CF_OSC_PROFILE_ROTATION() (++profile_rotations)
#define CF_OSC_PROFILE_PINV_RESULT()                                                                         \
  physics_controller::profile::pinv_result(profile_sweeps, profile_rotations)
#else
#define CF_OSC_PROFILE_BEGIN() ((void)0)
#define CF_OSC_PROFILE_STAGES(...) ((void)0)
#define CF_OSC_PROFILE_NEXT(...) ((void)0)
#define CF_OSC_PROFILE_SCOPE(...) ((void)0)
#define CF_OSC_PROFILE_META(...) ((void)0)
#define CF_OSC_PROFILE_PINV_COUNTERS() ((void)0)
#define CF_OSC_PROFILE_SWEEP() ((void)0)
#define CF_OSC_PROFILE_ROTATION() ((void)0)
#define CF_OSC_PROFILE_PINV_RESULT() ((void)0)
#endif
