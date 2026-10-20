#include "precision.cuh"
#pragma once
#ifndef LIBERO_PACKED_MASS_GATHER_CUH
#define LIBERO_PACKED_MASS_GATHER_CUH

// Nonstandard tree-packed symmetric product, not a dense BLAS operand.
// The caller admits a valid tree and storage, and owns publication/barriers.
// x/qM/topology must not alias result. Each block owns one world.
namespace libero_packed_mass {

// All block threads call. Each output has one owner, so no atomics or shared
// scratch are needed. Preserve the scalar scatter's contribution order,
// including simple-row assignment rather than addition.
__device__ __forceinline__ void multiply_gather(
    const real_t* x, const real_t* qM, const int* madr, const int* parent,
    const int* simple, real_t* result, int nv) {
    for (int row = int(threadIdx.x); row < nv; row += int(blockDim.x)) {
        real_t value = REAL_C(0.0);
        for (int i = 0; i < nv; ++i) {
            int adr = madr[i];
            if (simple && simple[i]) {
                if (row == i) value = qM[adr] * x[i];
                continue;
            }
            if (row == i) value += qM[adr] * x[i];
            ++adr;
            for (int j = parent[i]; j >= 0; j = parent[j], ++adr) {
                if (row == i) value += qM[adr] * x[j];
                if (row == j) value += qM[adr] * x[i];
            }
        }
        result[row] = value;
    }
}

// Experimental broadcast variant: one lane loads each packed contribution
// and all output owners in its warp consume it. Requires blockDim.x % 32 == 0
// and uniform entry. Inactive output lanes still participate in every shuffle.
// No floating-point reduction/reassociation and no implicit memory barrier.
// Keep opt-in until whole-kernel timing justifies the repeated warp traversal.
__device__ __forceinline__ void multiply_gather_warp(
    const real_t* x, const real_t* qM, const int* madr, const int* parent,
    const int* simple, real_t* result, int nv) {
    const int lane = int(threadIdx.x) & 31;
    const int warp_base = int(threadIdx.x) & ~31;
    for (int base = warp_base; base < nv; base += int(blockDim.x)) {
        const int row = base + lane;
        real_t value = REAL_C(0.0);
        for (int i = 0; i < nv; ++i) {
            int adr = 0, row_simple = 0;
            real_t diagonal = REAL_C(0.0), xi = REAL_C(0.0);
            if (lane == 0) {
                adr = madr[i];
                row_simple = simple && simple[i];
                diagonal = qM[adr];
                xi = x[i];
            }
            adr = __shfl_sync(0xffffffffu, adr, 0);
            row_simple = __shfl_sync(0xffffffffu, row_simple, 0);
            diagonal = __shfl_sync(0xffffffffu, diagonal, 0);
            xi = __shfl_sync(0xffffffffu, xi, 0);
            if (row_simple) {
                if (row == i) value = diagonal * xi;
                continue;
            }
            if (row == i) value += diagonal * xi;
            ++adr;
            for (int j = parent[i]; j >= 0; j = parent[j], ++adr) {
                real_t coefficient = REAL_C(0.0), xj = REAL_C(0.0);
                if (lane == 0) { coefficient = qM[adr]; xj = x[j]; }
                coefficient = __shfl_sync(0xffffffffu, coefficient, 0);
                xj = __shfl_sync(0xffffffffu, xj, 0);
                if (row == i) value += coefficient * xj;
                if (row == j) value += coefficient * xi;
            }
        }
        if (row < nv) result[row] = value;
    }
}

// Cold ordered CSR: offsets have nv+1 cells; each event carries a packed
// coefficient address, x address, and assignment flag (all int32). The builder
// emits events in the original scalar i/ancestor traversal order. A simple
// diagonal event assigns, including overwriting earlier descendant additions.
// No floating point reduction, no scratch, and no topology traversal at runtime.
__device__ __forceinline__ void multiply_plan(
    const real_t* x,const real_t* qM,const int* offsets,const int* coefficient,
    const int* source,const int* assign,real_t* result,int nv) {
    for(int row=int(threadIdx.x);row<nv;row+=int(blockDim.x)) {
        real_t value=REAL_C(0.0);
        for(int e=offsets[row];e<offsets[row+1];++e) {
            if(assign[e]) value=qM[coefficient[e]]*x[source[e]];
            else value+=qM[coefficient[e]]*x[source[e]];
        }
        result[row]=value;
    }
}

// Generic non-simple packed storage. Newton passes its dof_simple array to
// the overload above; dropping that array there would change semantics.
__device__ __forceinline__ void multiply_gather(
    const real_t* x, const real_t* qM, const int* madr, const int* parent,
    real_t* result, int nv) {
    multiply_gather(x, qM, madr, parent, nullptr, result, nv);
}
__device__ __forceinline__ void multiply_gather_warp(
    const real_t* x, const real_t* qM, const int* madr, const int* parent,
    real_t* result, int nv) {
    multiply_gather_warp(x, qM, madr, parent, nullptr, result, nv);
}
}
#endif
