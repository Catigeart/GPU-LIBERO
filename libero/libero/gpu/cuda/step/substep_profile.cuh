#pragma once

// Present only in the separate profiling product. Markers add no barriers,
// do not change numerical storage, and measure lane 0 at existing boundaries.
#if defined(FLASH_SUBSTEP_PROFILE)
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_libero::substep_profile {
enum class Phase : int {
  Forward, VelocityCold, VelocityReuse, ReferenceCold, ReferenceReuse,
  BiasCold, BiasReuse, Actuation, Smooth, NewtonBridge, Newton, Post,
  ForwardPublish, Controller, BeginForward, Checks, Integrate, Total, Count
};
constexpr int header_cells = 8, info_cells = 8;
constexpr int record_cells = info_cells + 2 * int(Phase::Count);
struct Configuration { uint64_t *buffer; int capacity; };
extern __device__ Configuration configuration;

__device__ __forceinline__ uint64_t timer() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value) :: "memory");
  return value;
}
__device__ __forceinline__ uint64_t *current_record() {
  if (threadIdx.x != 0 || !configuration.buffer) return nullptr;
  uint64_t *base = configuration.buffer + size_t(blockIdx.y) *
      (header_cells + size_t(configuration.capacity) * record_cells);
  const uint64_t count = base[0];
  return count && count <= uint64_t(configuration.capacity)
      ? base + header_cells + (count - 1) * record_cells : nullptr;
}
__device__ __forceinline__ void add(uint64_t *record, Phase phase, uint64_t elapsed) {
  record[info_cells + 2 * int(phase)] += elapsed;
  ++record[info_cells + 2 * int(phase) + 1];
}
struct Recorder {
  uint64_t *record = nullptr;
  uint64_t started = 0;
  __device__ __forceinline__ Recorder(int entry, bool policy) {
    if (threadIdx.x == 0 && configuration.buffer) {
      uint64_t *base = configuration.buffer + size_t(blockIdx.y) *
          (header_cells + size_t(configuration.capacity) * record_cells);
      const uint64_t slot = base[0]++;
      if (slot < uint64_t(configuration.capacity)) {
        record = base + header_cells + slot * record_cells;
        record[0] = entry;
        record[1] = policy;
        unsigned sm;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
        record[2] = sm;
        started = timer();
      } else ++base[1];
    }
  }
  __device__ __forceinline__ ~Recorder() {
    if (record) add(record, Phase::Total, timer() - started);
  }
};
struct Scope {
  uint64_t *record;
  uint64_t started = 0;
  Phase phase;
  __device__ __forceinline__ explicit Scope(Phase value) : record(current_record()), phase(value) {
    if (record) started = timer();
  }
  __device__ __forceinline__ ~Scope() {
    if (record) add(record, phase, timer() - started);
  }
};
struct Stages {
  uint64_t *record;
  uint64_t started = 0;
  Phase phase;
  __device__ __forceinline__ explicit Stages(Phase value) : record(current_record()), phase(value) {
    if (record) started = timer();
  }
  __device__ __forceinline__ void next(Phase value) {
    if (record) {
      add(record, phase, timer() - started);
      phase = value;
      started = timer();
    }
  }
  __device__ __forceinline__ ~Stages() {
    if (record) add(record, phase, timer() - started);
  }
};
} // namespace flash_libero::substep_profile
#define CF_STEP_PROFILE_SCOPE(phase) flash_libero::substep_profile::Scope step_profile_scope(flash_libero::substep_profile::Phase::phase)
#define CF_STEP_PROFILE_STAGES(phase) flash_libero::substep_profile::Stages step_profile_stages(phase)
#define CF_STEP_PROFILE_NEXT(phase) step_profile_stages.next(phase)
#define CF_STEP_PROFILE_PHASE(phase) flash_libero::substep_profile::Phase::phase
#define CF_STEP_PROFILE_BEGIN(entry, policy) flash_libero::substep_profile::Recorder step_profile_record(int(entry), policy)
#else
#define CF_STEP_PROFILE_SCOPE(...) ((void)0)
#define CF_STEP_PROFILE_STAGES(...) ((void)0)
#define CF_STEP_PROFILE_NEXT(...) ((void)0)
#define CF_STEP_PROFILE_PHASE(...) 0
#define CF_STEP_PROFILE_BEGIN(...) ((void)0)
#endif
