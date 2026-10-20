#include "precision.cuh"
// Project-written register collectives. Every lane in a full warp participates.
// Callers pad invalid elements with the operation's identity, never skip lanes.
// These functions do not synchronize shared or global memory.
#pragma once
#ifndef LIBERO_WARP_CUH
#define LIBERO_WARP_CUH
namespace libero_warp {
constexpr unsigned mask = 0xffffffffu;
constexpr int no_index = 2147483647;
__device__ __forceinline__ int lane() { return threadIdx.x & 31; }
__device__ __forceinline__ int warp() { return threadIdx.x >> 5; }
// Only lane 0 may consume this result. The FP64 addition tree is unchanged.
__device__ __forceinline__ real_t sum_to_leader(real_t value) {
    for (int offset=16; offset>0; offset>>=1)
        value += __shfl_down_sync(mask, value, offset);
    return value;
}
__device__ __forceinline__ real_t sum(real_t value) {
    return __shfl_sync(mask, sum_to_leader(value), 0);
}
__device__ __forceinline__ int sum(int value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __reduce_add_sync(mask, value);
#else
    for (int offset=16; offset>0; offset>>=1)
        value += __shfl_down_sync(mask, value, offset);
    return __shfl_sync(mask, value, 0);
#endif
}
__device__ __forceinline__ bool any(bool value) {
    return __any_sync(mask, value);
}
__device__ __forceinline__ int min(int value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __reduce_min_sync(mask, value);
#else
    for (int offset=16; offset>0; offset>>=1) {
        int other = __shfl_down_sync(mask, value, offset);
        value = value < other ? value : other;
    }
    return __shfl_sync(mask, value, 0);
#endif
}
__device__ __forceinline__ int exclusive_sum(int value) {
    const int original = value;
    const int index = lane();
    for (int offset=1; offset<32; offset<<=1) {
        int other = __shfl_up_sync(mask, value, offset);
        if (index >= offset) value += other;
    }
    return value - original;
}
// Contiguous aligned subgroups; all 32 lanes still enter each collective.
template<int Width>
__device__ __forceinline__ int subgroup_lane() {
    static_assert(Width == 4 || Width == 8 || Width == 16 || Width == 32,
                  "warp subgroup width must be 4, 8, 16 or 32");
    return lane() & (Width-1);
}
template<int Width>
__device__ __forceinline__ real_t subgroup_sum_to_leader(real_t value) {
    (void)subgroup_lane<Width>();
    for (int offset=Width/2; offset>0; offset>>=1)
        value += __shfl_down_sync(mask, value, offset, Width);
    return value;
}
template<int Width>
__device__ __forceinline__ real_t subgroup_sum(real_t value) {
    return __shfl_sync(mask, subgroup_sum_to_leader<Width>(value), 0, Width);
}
template<int Width>
__device__ __forceinline__ int subgroup_sum(int value) {
    (void)subgroup_lane<Width>();
    for (int offset=Width/2; offset>0; offset>>=1)
        value += __shfl_down_sync(mask, value, offset, Width);
    return __shfl_sync(mask, value, 0, Width);
}
template<int Width>
__device__ __forceinline__ int subgroup_min(int value) {
    (void)subgroup_lane<Width>();
    for (int offset=Width/2; offset>0; offset>>=1) {
        const int other = __shfl_down_sync(mask, value, offset, Width);
        value = value < other ? value : other;
    }
    return __shfl_sync(mask, value, 0, Width);
}
template<int Width>
__device__ __forceinline__ bool subgroup_any(bool value) {
    const int base = lane() - subgroup_lane<Width>();
    const unsigned group_mask = mask >> (32-Width);
    return ((__ballot_sync(mask, value) >> base) & group_mask) != 0;
}
template<int Width>
__device__ __forceinline__ int subgroup_exclusive_sum(int value) {
    const int original = value;
    const int index = subgroup_lane<Width>();
    for (int offset=1; offset<Width; offset<<=1) {
        const int other = __shfl_up_sync(mask, value, offset, Width);
        if (index >= offset) value += other;
    }
    return value - original;
}
// Stable predicate compaction rank, valid for selected and unselected lanes.
__device__ __forceinline__ int compact_rank(bool selected) {
    const unsigned selected_lanes = __ballot_sync(mask, selected);
    const unsigned preceding = (1u << lane()) - 1u;
    return __popc(selected_lanes & preceding);
}
__device__ __forceinline__ int compact_count(bool selected) {
    return __popc(__ballot_sync(mask, selected));
}
// Supply semantic indices: error codes need not have index ordering.
__device__ __forceinline__ int first_index(bool selected, int index) {
    return min(selected ? index : no_index);
}
} // namespace libero_warp
#endif // LIBERO_WARP_CUH
