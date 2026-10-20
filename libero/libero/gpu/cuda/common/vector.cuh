#include "precision.cuh"
#pragma once
#ifndef LIBERO_VECTOR_CUH
#define LIBERO_VECTOR_CUH
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <cmath>

// No barriers or votes: safe in lane0 and subset calls. first >= 0 and
// stride > 0 describe participating lanes; together they must cover the range
// of work indices. The indices address head elements, 16-byte body packets,
// and tail elements separately, not a fixed scalar-element ownership mapping.
// Copy spans must be nonoverlapping or identical. Ordered/overlapping copies
// must retain caller logic. Counts must fit size_t byte spans. T must be a
// trivially copyable object, with valid naturally aligned source/destination.
namespace libero_vec {
__device__ __forceinline__ bool aligned16(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & uintptr_t(15)) == 0;
}
__device__ __forceinline__ real2_t load_pair(const real_t* p) {
    // Exactly two valid naturally aligned doubles required. The address
    // predicate stays inside PTX; no C++ wide-pointer alignment fact escapes
    // this helper into a subsequent access at the same address.
#if defined(__CUDA_ARCH__) && FLASH_REAL_BITS == 32
    real2_t value;
    asm volatile("{ .reg .pred aligned; .reg .b64 residue; "
                 "and.b64 residue, %2, 7; "
                 "setp.eq.u64 aligned, residue, 0; "
                 "@aligned ld.v2.f32 {%0, %1}, [%2]; "
                 "@!aligned ld.f32 %0, [%2]; "
                 "@!aligned ld.f32 %1, [%2+4]; }"
                 : "=&f"(value.x), "=&f"(value.y) : "l"(p) : "memory");
    return value;
#elif defined(__CUDA_ARCH__)
    real2_t value;
    asm volatile("{ .reg .pred aligned; .reg .b64 residue; "
                 "and.b64 residue, %2, 15; "
                 "setp.eq.u64 aligned, residue, 0; "
                 "@aligned ld.v2.b64 {%0, %1}, [%2]; "
                 "@!aligned ld.b64 %0, [%2]; "
                 "@!aligned ld.b64 %1, [%2+8]; }"
                 : "=&d"(value.x), "=&d"(value.y) : "l"(p) : "memory");
    return value;
#else
    // Compiler/CPU emulation branch; never used by a CUDA device build.
    if (aligned16(p)) return *reinterpret_cast<const real2_t*>(p);
    return make_real2(p[0], p[1]);
#endif
}
__device__ __forceinline__ void store_pair(real_t* p, real_t x, real_t y) {
    // Exactly two writable naturally aligned doubles required. Evaluate the
    // actual address in PTX so earlier vector accesses cannot delete this
    // scalar fallback. Both paths transfer raw FP64 register bits.
#if defined(__CUDA_ARCH__) && FLASH_REAL_BITS == 32
    asm volatile("{ .reg .pred aligned; .reg .b64 residue; "
                 "and.b64 residue, %0, 7; "
                 "setp.eq.u64 aligned, residue, 0; "
                 "@aligned st.v2.f32 [%0], {%1, %2}; "
                 "@!aligned st.f32 [%0], %1; "
                 "@!aligned st.f32 [%0+4], %2; }"
                 :: "l"(p), "f"(x), "f"(y) : "memory");
#elif defined(__CUDA_ARCH__)
    asm volatile("{ .reg .pred aligned; .reg .b64 residue; "
                 "and.b64 residue, %0, 15; "
                 "setp.eq.u64 aligned, residue, 0; "
                 "@aligned st.v2.b64 [%0], {%1, %2}; "
                 "@!aligned st.b64 [%0], %1; "
                 "@!aligned st.b64 [%0+8], %2; }"
                 :: "l"(p), "d"(x), "d"(y) : "memory");
#else
    if (aligned16(p)) *reinterpret_cast<real2_t*>(p) = make_real2(x,y);
    else { p[0]=x; p[1]=y; }
#endif
}
__device__ __forceinline__ void copy_bytes_lane(const void* src, void* dst,
                                               size_t bytes, int first, int stride) {
    if (src == dst || bytes == 0 || first < 0 || stride <= 0) return;
    const unsigned char* s = static_cast<const unsigned char*>(src);
    unsigned char* d = static_cast<unsigned char*>(dst);
    const size_t lane = size_t(first), step = size_t(stride);
    // Different residues cannot be aligned by a common head. Byte fallback
    // remains bit-preserving and never issues an unaligned vector access.
    if ((reinterpret_cast<uintptr_t>(s) & 15) != (reinterpret_cast<uintptr_t>(d) & 15)) {
        for (size_t i=lane; i<bytes; i+=step) d[i]=s[i];
        return;
    }
    size_t head = (16-(reinterpret_cast<uintptr_t>(s)&15))&15;
    if (head>bytes) head=bytes;
    for (size_t i=lane; i<head; i+=step) d[i]=s[i];
    const size_t packets=(bytes-head)/16;
    const uint4* sv=reinterpret_cast<const uint4*>(s+head);
    uint4* dv=reinterpret_cast<uint4*>(d+head);
    for (size_t i=lane; i<packets; i+=step) dv[i]=sv[i];
    const size_t tail=head+packets*16;
    for (size_t i=lane; i<bytes-tail; i+=step) d[tail+i]=s[tail+i];
}
// Primitive scalar transfers compile to their native load/store width, without
// arithmetic conversions. General trivial objects use representation copies.
template<class T>
__device__ __forceinline__ void copy_object(const T* src,T* dst) {
    if constexpr (std::is_arithmetic<T>::value || std::is_enum<T>::value) *dst=*src;
    else {
        const unsigned char* s=reinterpret_cast<const unsigned char*>(src);
        unsigned char* d=reinterpret_cast<unsigned char*>(dst);
        for(size_t b=0;b<sizeof(T);++b) d[b]=s[b];
    }
}
template<class T>
__device__ __forceinline__ void copy_lane(const T* src,T* dst,size_t count,int first,int stride) {
    static_assert(std::is_trivially_copyable<T>::value,"copy requires trivial object bits");
    if(src==dst || count==0 || first<0 || stride<=0) return;
    const size_t lane=size_t(first),step=size_t(stride);
    // Exact measured double/64-lane cases only: short17/28 regress;
    // 342 keeps its positive16B-aligned route. No range threshold is inferred.
    // Unmeasured types, lengths and participation strides retain their route.
    if constexpr (std::is_same<T,double>::value) {
        if (stride==64 && (count==17 || count==28 ||
            (count==342 && (!aligned16(src) || !aligned16(dst))))) {
            for(size_t i=lane;i<count;i+=step) dst[i]=src[i];
            return;
        }
    }
    if constexpr (sizeof(T)<=16 && 16%sizeof(T)==0) {
        const size_t residue=reinterpret_cast<uintptr_t>(src)&15;
        const size_t head_bytes=(16-residue)&15;
        if(residue==(reinterpret_cast<uintptr_t>(dst)&15) && head_bytes%sizeof(T)==0) {
            size_t head=head_bytes/sizeof(T);
            if(head>count) head=count;
            for(size_t i=lane;i<head;i+=step) copy_object(src+i,dst+i);
            constexpr size_t width=16/sizeof(T);
            const size_t packets=(count-head)/width;
            const uint4* s=reinterpret_cast<const uint4*>(src+head);
            uint4* d=reinterpret_cast<uint4*>(dst+head);
            for(size_t i=lane;i<packets;i+=step) d[i]=s[i];
            const size_t tail=head+packets*width;
            for(size_t i=lane;i<count-tail;i+=step) copy_object(src+tail+i,dst+tail+i);
            return;
        }
    }
    // Different 16B residues and unusual object sizes retain typed scalar
    // width, avoiding eight byte transactions for an unpaired double.
    for(size_t i=lane;i<count;i+=step) copy_object(src+i,dst+i);
}
template<class T>
__device__ __forceinline__ void copy_block(const T* src,T* dst,size_t count) {
    copy_lane(src,dst,count,int(threadIdx.x),int(blockDim.x));
}
template<class T>
__device__ __forceinline__ void copy_serial(const T* src,T* dst,size_t count) {
    copy_lane(src,dst,count,0,1);
}
template<class T>
__device__ __forceinline__ void fill_lane(T* dst,size_t count,T value,int first,int stride) {
    static_assert(std::is_trivially_copyable<T>::value,"fill requires trivial object bits");
    if (count == 0 || first < 0 || stride <= 0) return;
    const size_t lane=size_t(first),step=size_t(stride);
    const unsigned char* bits=reinterpret_cast<const unsigned char*>(&value);
    unsigned char* d=reinterpret_cast<unsigned char*>(dst);
    // A vector packet repeats whole objects. For unusual object sizes or
    // addresses that cannot reach 16B alignment, use whole-object bit copies.
    if constexpr (sizeof(T)<=16 && 16%sizeof(T)==0) {
        const size_t head_bytes=(16-(reinterpret_cast<uintptr_t>(d)&15))&15;
        if (head_bytes%sizeof(T)==0) {
            size_t head=head_bytes/sizeof(T);
            if (head>count) head=count;
            for (size_t i=lane;i<head;i+=step)
                copy_object(&value,dst+i);
            uint4 pattern;
            unsigned char* pattern_bits=reinterpret_cast<unsigned char*>(&pattern);
            for (size_t b=0;b<16;++b) pattern_bits[b]=bits[b%sizeof(T)];
            constexpr size_t width=16/sizeof(T);
            const size_t packets=(count-head)/width;
            uint4* body=reinterpret_cast<uint4*>(d+head*sizeof(T));
            for(size_t i=lane;i<packets;i+=step) body[i]=pattern;
            const size_t tail=head+packets*width;
            for(size_t i=lane;i<count-tail;i+=step)
                copy_object(&value,dst+tail+i);
            return;
        }
    }
    for(size_t i=lane;i<count;i+=step)
        copy_object(&value,dst+i);
}
template<class T>
__device__ __forceinline__ void fill_block(T* dst,size_t count,T value) {
    fill_lane(dst,count,value,int(threadIdx.x),int(blockDim.x));
}
template<class T>
__device__ __forceinline__ void fill_serial(T* dst,size_t count,T value) {
    fill_lane(dst,count,value,0,1);
}
__device__ __forceinline__ bool any_nonfinite_lane(const real_t* src,size_t count,int first,int stride) {
    if (first<0 || stride<=0 || count==0) return false;
    const size_t lane=size_t(first),step=size_t(stride);
    // Match only the64-lane performance samples. Scalar17/28 and misaligned
    // 342 avoid measured regressions/mixed directions; aligned342 stays paired.
    if (stride==64 && (count==17 || count==28 ||
        (count==342 && !aligned16(src)))) {
        bool bad=false;
        for(size_t i=lane;i<count;i+=step) bad|=!isfinite(src[i]);
        return bad; // Lane-local; callers retain their original vote.
    }
    const size_t head=aligned16(src)?0:1;
    bool bad=false;
    if(head && lane==0) bad=!isfinite(src[0]);
    const size_t packets=(count-head)/2;
    for(size_t i=lane;i<packets;i+=step) {
        const real2_t v=load_pair(src+head+2*i);
        bad=bad || !isfinite(v.x) || !isfinite(v.y);
    }
    const size_t tail=head+2*packets;
    if(tail<count && lane==0) bad=bad || !isfinite(src[tail]);
    return bad; // Lane-local only; caller supplies its existing vote.
}
__device__ __forceinline__ bool any_nonfinite_block_lane(const real_t* src,size_t count) {
    return any_nonfinite_lane(src,count,int(threadIdx.x),int(blockDim.x));
}
} // namespace libero_vec
#endif // LIBERO_VECTOR_CUH
