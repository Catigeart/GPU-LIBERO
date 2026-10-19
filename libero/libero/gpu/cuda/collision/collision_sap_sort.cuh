#pragma once
#include <limits.h>
#include <math.h>

// All 32 lanes of the SAP warp enter each helper. No shared storage is needed.
namespace collision_cuda {
__device__ __forceinline__ bool sap_before(float a, int ordinal_a, float b, int ordinal_b) {
    return a < b || (a == b && ordinal_a < ordinal_b);
}

// Stable sorting network in registers. NaNs retain the original insertion
// sort path, including its comparison barriers. Padding sorts after real ties.
__device__ __forceinline__ bool sap_events64(float* keys, int* tags, int count) {
    if (count > 64) return false;
    const int lane = threadIdx.x & 31;
    float value[2];
    int ordinal[2], tag[2];
    bool nan = false;
    #pragma unroll
    for (int item = 0; item < 2; ++item) {
        const int index = 32 * item + lane;
        value[item] = index < count ? keys[index] : INFINITY;
        ordinal[item] = index < count ? index : INT_MAX;
        tag[item] = index < count ? tags[index] : 0;
        nan |= isnan(value[item]);
    }
    if (__any_sync(0xffffffffu, nan)) return false;
    #pragma unroll
    for (int span = 2; span <= 64; span *= 2) {
        #pragma unroll
        for (int stride = span / 2; stride > 0; stride /= 2) {
            float next_value[2];
            int next_ordinal[2], next_tag[2];
            #pragma unroll
            for (int item = 0; item < 2; ++item) {
                const int index = 32 * item + lane;
                const float other_value = stride == 32 ? value[item ^ 1] :
                    __shfl_xor_sync(0xffffffffu, value[item], stride);
                const int other_ordinal = stride == 32 ? ordinal[item ^ 1] :
                    __shfl_xor_sync(0xffffffffu, ordinal[item], stride);
                const int other_tag = stride == 32 ? tag[item ^ 1] :
                    __shfl_xor_sync(0xffffffffu, tag[item], stride);
                const bool want_min = ((index & stride) == 0) == ((index & span) == 0);
                const bool take_other = want_min ?
                    sap_before(other_value, other_ordinal, value[item], ordinal[item]) :
                    sap_before(value[item], ordinal[item], other_value, other_ordinal);
                next_value[item] = take_other ? other_value : value[item];
                next_ordinal[item] = take_other ? other_ordinal : ordinal[item];
                next_tag[item] = take_other ? other_tag : tag[item];
            }
            #pragma unroll
            for (int item = 0; item < 2; ++item) {
                value[item] = next_value[item];
                ordinal[item] = next_ordinal[item];
                tag[item] = next_tag[item];
            }
        }
    }
    #pragma unroll
    for (int item = 0; item < 2; ++item) {
        const int index = 32 * item + lane;
        if (index < count) { keys[index] = value[item]; tags[index] = tag[item]; }
    }
    return true;
}

// Integer pair order has no NaNs or distinct equal payloads. Larger batches
// keep the old ordered implementation; the network needs no shared workspace.
__device__ __forceinline__ bool sap_pairs128(int* pairs, int count) {
    if (count > 128) return false;
    const int lane = threadIdx.x & 31;
    int value[4];
    #pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int index = 32 * item + lane;
        value[item] = index < count ? pairs[index] : INT_MAX;
    }
    #pragma unroll
    for (int span = 2; span <= 128; span *= 2) {
        #pragma unroll
        for (int stride = span / 2; stride > 0; stride /= 2) {
            int next[4];
            #pragma unroll
            for (int item = 0; item < 4; ++item) {
                const int index = 32 * item + lane;
                const int other = stride >= 32 ? value[item ^ (stride / 32)] :
                    __shfl_xor_sync(0xffffffffu, value[item], stride);
                const bool want_min = ((index & stride) == 0) == ((index & span) == 0);
                next[item] = want_min ? min(value[item], other) : max(value[item], other);
            }
            #pragma unroll
            for (int item = 0; item < 4; ++item) value[item] = next[item];
        }
    }
    #pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int index = 32 * item + lane;
        if (index < count) pairs[index] = value[item];
    }
    return true;
}
} // namespace collision_cuda
