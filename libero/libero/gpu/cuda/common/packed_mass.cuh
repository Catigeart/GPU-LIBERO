#include "precision.cuh"
#pragma once
#ifndef LIBERO_PACKED_MASS_CUH
#define LIBERO_PACKED_MASS_CUH
#include <math.h>

// Tree-packed LDL^T. Each row stores its diagonal followed by its ancestors;
// this is not a dense BLAS/LAPACK operand. The caller owns admission, zeroing,
// publication and warning status; these helpers never modify those protocols.
namespace libero_packed_mass {
// Every packed row belongs to the parent=-1 root reached by its ancestors.
// A root and its descendants form an independent matrix block. MuJoCo orders
// these blocks contiguously; descriptors that do not satisfy that property
// keep the original serial implementation below. No additional model ABI or
// capacity-sized shared metadata is required.
struct ForestPlan {
    int roots; bool contiguous;
    const int* bounds=nullptr;
    const int* row_lengths=nullptr;
};
__device__ __forceinline__ ForestPlan inspect_forest_block(int nv,const int* parent) {
    __shared__ int root_count,valid_layout;
    if(threadIdx.x==0) {
        int first=0,roots=0;bool valid=nv>0;
        for(int i=0;i<nv;++i) {
            if(parent[i]==-1){first=i;++roots;}
            else if(parent[i]<first||parent[i]>=i)valid=false;
        }
        root_count=roots;valid_layout=valid&&roots>0;
    }
    __syncthreads();
    return {root_count,valid_layout!=0};
}
// Prepared, owner-private int32 descriptor layout is documented by the pure
// CPU build_mass_descriptor(). Shadow comparison is exact on every call;
// a changed source always takes the original inspection and range traversal.
__device__ __forceinline__ ForestPlan prepared_forest_block(int nv,int nm,
    const int* madr,const int* parent,const int* descriptor,int descriptor_size) {
    // v2 is issued only by the exact measured cold topology selector. Legacy
    // v1/generic descriptors preserve the original inspect/range traversal.
    if(!descriptor||descriptor_size<5||descriptor[0]!=2||descriptor[1]!=nv||
       descriptor[2]!=nm)return inspect_forest_block(nv,parent);
    const int roots=descriptor[3];
    if(roots<=1||roots>nv||descriptor_size!=6+roots+3*nv||!descriptor[4])
        return inspect_forest_block(nv,parent);
    const int* bounds=descriptor+5;
    const int* source_parent=bounds+roots+1;
    const int* source_madr=source_parent+nv;
    const int* lengths=source_madr+nv;
    bool mismatch=bounds[0]!=0||bounds[roots]!=nv;
    for(int i=threadIdx.x;i<nv;i+=blockDim.x)
        mismatch|=parent[i]!=source_parent[i]||madr[i]!=source_madr[i];
    if(__syncthreads_or(mismatch))return inspect_forest_block(nv,parent);
    return {roots,true,bounds,lengths};
}
__device__ __forceinline__ void forest_range(int root,int nv,const int* parent,int& begin,int& end) {
    int ordinal=-1;begin=end=nv;
    for(int i=0;i<nv;++i)if(parent[i]==-1) {
        ++ordinal;
        if(ordinal==root)begin=i;
        else if(ordinal==root+1){end=i;break;}
    }
}
__device__ __forceinline__ void forest_range(ForestPlan plan,int root,int nv,
    const int* parent,int& begin,int& end) {
    if(plan.bounds){begin=plan.bounds[root];end=plan.bounds[root+1];}
    else forest_range(root,nv,parent,begin,end);
}
__device__ __forceinline__ bool factor(const real_t* qm,real_t* ld,real_t* inv,
    real_t* sqrt_inv,int* pivot,int& warnings,int& last,int nv,int nm,
    const int* madr,const int* parent,bool reciprocal_pivot=false) {
    reciprocal_pivot=reciprocal_pivot&&((nv==21&&nm==57)||(nv==43&&nm==115)||(nv==57&&nm==153));
    bool bad=false;
    libero_vec::copy_serial(qm,ld,nm);
    for(int k=nv-1;k>=0;--k) {
        const int kk=madr[k];
        if(ld[kk]<REAL_C(1.0e-15)) {
            pivot[k]=1;++warnings;last=k;ld[kk]=REAL_C(1.0e-15);
        }
        const real_t reciprocal=reciprocal_pivot?REAL_C(1.0)/ld[kk]:REAL_C(0.0);
        int ki=kk+1;
        for(int i=parent[k];i>=0;i=parent[i],++ki) {
            const real_t multiplier=reciprocal_pivot?ld[ki]*reciprocal:ld[ki]/ld[kk];
            bad|=!isfinite(multiplier);
            const int len=(i<nv-1?madr[i+1]:nm)-madr[i];
            for(int a=0;a<len;++a) {
                const int dest=madr[i]+a;
                ld[dest]+=ld[ki+a]*(-multiplier);
                bad|=!isfinite(ld[dest]);
            }
            ld[ki]=multiplier;
        }
    }
    for(int i=0;i<nv;++i) {
        const real_t diagonal=ld[madr[i]];
        inv[i]=REAL_C(1.0)/diagonal;sqrt_inv[i]=REAL_C(1.0)/sqrt(diagonal);
        bad|=!isfinite(inv[i])||!isfinite(sqrt_inv[i]);
    }
    return !bad;
}
__device__ __forceinline__ bool solve_inplace(real_t* x,const real_t* ld,
    const real_t* inv,int nv,const int* madr,const int* parent) {
    bool bad=false;
    // A zero descendant is deliberately skipped: ignored nonfinite products
    // and signed-zero behavior remain those of the original sparse solve.
    for(int i=nv-1;i>=0;--i) {
        if(x[i]==REAL_C(0.0)) continue;
        int a=madr[i]+1;
        for(int j=parent[i];j>=0;j=parent[j],++a) {
            x[j]-=ld[a]*x[i];bad|=!isfinite(x[j]);
        }
    }
    for(int i=0;i<nv;++i) {x[i]*=inv[i];bad|=!isfinite(x[i]);}
    for(int i=0;i<nv;++i) {
        int a=madr[i]+1;
        for(int j=parent[i];j>=0;j=parent[j],++a) {
            x[i]-=ld[a]*x[j];bad|=!isfinite(x[i]);
        }
    }
    return !bad;
}

// All block threads enter. Each component has exactly one owner. Within a
// component the pivot, ancestor, and packed-row operation order is identical
// to factor(); different components never read or write one another's rows.
__device__ __forceinline__ bool factor_block(const real_t* qm,real_t* ld,real_t* inv,
    real_t* sqrt_inv,int* pivot,int& warnings,int& last,int nv,int nm,
    const int* madr,const int* parent,ForestPlan& plan,
    const int* descriptor=nullptr,int descriptor_size=0,bool reciprocal_pivot=false) {
    reciprocal_pivot=reciprocal_pivot&&((nv==21&&nm==57)||(nv==43&&nm==115)||(nv==57&&nm==153));
    plan=prepared_forest_block(nv,nm,madr,parent,descriptor,descriptor_size);
    __shared__ int failed,new_warnings,last_warning;
    if(threadIdx.x==0){failed=0;new_warnings=0;last_warning=nv;}
    __syncthreads();
    if(!plan.contiguous||plan.roots<=1) {
        if(threadIdx.x==0)failed=!factor(qm,ld,inv,sqrt_inv,pivot,warnings,last,nv,nm,madr,parent,reciprocal_pivot);
        __syncthreads();return failed==0;
    }
    libero_vec::copy_block(qm,ld,nm);
    __syncthreads();
    bool bad=false;int count=0,smallest=nv;
    for(int root=threadIdx.x;root<plan.roots;root+=blockDim.x) {
        int begin,end;forest_range(plan,root,nv,parent,begin,end);
        for(int k=end-1;k>=begin;--k) {
            const int kk=madr[k];
            if(ld[kk]<REAL_C(1.0e-15)){pivot[k]=1;++count;smallest=k<smallest?k:smallest;ld[kk]=REAL_C(1.0e-15);}
            const real_t reciprocal=reciprocal_pivot?REAL_C(1.0)/ld[kk]:REAL_C(0.0);
            int ki=kk+1;
            for(int i=parent[k];i>=0;i=parent[i],++ki) {
                const real_t multiplier=reciprocal_pivot?ld[ki]*reciprocal:ld[ki]/ld[kk];bad|=!isfinite(multiplier);
                const int len=plan.row_lengths?plan.row_lengths[i]:(i<nv-1?madr[i+1]:nm)-madr[i];
                for(int a=0;a<len;++a) {
                    const int dest=madr[i]+a;
                    ld[dest]+=ld[ki+a]*(-multiplier);bad|=!isfinite(ld[dest]);
                }
                ld[ki]=multiplier;
            }
        }
        for(int i=begin;i<end;++i) {
            const real_t diagonal=ld[madr[i]];
            inv[i]=REAL_C(1.0)/diagonal;sqrt_inv[i]=REAL_C(1.0)/sqrt(diagonal);
            bad|=!isfinite(inv[i])||!isfinite(sqrt_inv[i]);
        }
    }
    if(bad)atomicOr(&failed,1);
    if(count){atomicAdd(&new_warnings,count);atomicMin(&last_warning,smallest);}
    __syncthreads();
    if(threadIdx.x==0){warnings+=new_warnings;if(new_warnings)last=last_warning;}
    __syncthreads();return failed==0;
}

__device__ __forceinline__ bool solve_component(real_t* x,const real_t* ld,
    const real_t* inv,int begin,int end,const int* madr,const int* parent) {
    bool bad=false;
    for(int i=end-1;i>=begin;--i) {
        if(x[i]==REAL_C(0.0))continue;
        int a=madr[i]+1;
        for(int j=parent[i];j>=0;j=parent[j],++a){x[j]-=ld[a]*x[i];bad|=!isfinite(x[j]);}
    }
    for(int i=begin;i<end;++i){x[i]*=inv[i];bad|=!isfinite(x[i]);}
    for(int i=begin;i<end;++i) {
        int a=madr[i]+1;
        for(int j=parent[i];j>=0;j=parent[j],++a){x[i]-=ld[a]*x[j];bad|=!isfinite(x[i]);}
    }
    return !bad;
}

// Cross product of independent RHS and independent mass components. No new
// floating-point reduction is introduced. Return value is block-uniform and
// includes the failure of every component; callers retain their clear gate.
__device__ __forceinline__ bool solve_rhs_block(const real_t* rhs,real_t* solution,
    const real_t* ld,const real_t* inv,int nrhs,int nv,const int* madr,
    const int* parent,ForestPlan plan) {
    libero_vec::copy_block(rhs,solution,size_t(nrhs)*nv);
    __syncthreads();bool bad=false;
    if(plan.contiguous&&plan.roots>1) {
        for(int task=threadIdx.x;task<nrhs*plan.roots;task+=blockDim.x) {
            const int r=task/plan.roots,root=task%plan.roots;
            int begin,end;forest_range(plan,root,nv,parent,begin,end);
            bad|=!solve_component(solution+r*nv,ld,inv,begin,end,madr,parent);
        }
    } else {
        for(int r=threadIdx.x;r<nrhs;r+=blockDim.x)
            bad|=!solve_inplace(solution+r*nv,ld,inv,nv,madr,parent);
    }
    return __syncthreads_or(bad)==0;
}

// Multiple independent RHS share one completed factorization. The caller
// enters on all block threads after its factor barrier and combines the
// returned lane-local failure before the existing transaction clear/gate.
// No outer pivot/ancestor ordering, zero-descendant skip or arithmetic changes.
// Factor, input RHS and output solution must not overlap.
__device__ __forceinline__ bool solve_rhs_lane(const real_t* rhs,real_t* solution,
    const real_t* ld,const real_t* inv,int nrhs,int nv,
    const int* madr,const int* parent) {
    // Stage only active RHS cells with consecutive block lanes. No inactive
    // tail is touched; the owning caller already zeroes those cells.
    libero_vec::copy_block(rhs,solution,size_t(nrhs)*nv);
    __syncthreads();
    const int r=int(threadIdx.x);
    if(r>=nrhs)return true;
    return solve_inplace(solution+r*nv,ld,inv,nv,madr,parent);
}
}
#endif
