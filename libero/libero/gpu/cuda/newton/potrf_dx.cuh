#include "precision.cuh"
#include "io_policy.cuh"
// Collective FP64 Cholesky, embedded in the persistent Newton block.
// All block threads must enter every function in this file together.
#pragma once
// The device bridge and caller both use FMA-enabled arithmetic.
extern __device__ void dx_potrf_nv(real_t*, int*);
extern __device__ void dx_potrf_1(real_t*, int*);
extern __device__ void dx_potrf_3(real_t*, int*);
extern __device__ void dx_potrf_4(real_t*, int*);
extern __device__ void dx_potrf_6(real_t*, int*);
#ifndef LIBERO_CONE_HESSIAN_STRATEGY
#define LIBERO_CONE_HESSIAN_STRATEGY 0
#endif
#ifndef LIBERO_CONE_BATCH_THRESHOLD
#define LIBERO_CONE_BATCH_THRESHOLD NV
#endif
#ifndef LIBERO_RANK_REBUILD_THRESHOLD
#define LIBERO_RANK_REBUILD_THRESHOLD 0
#endif
#ifndef LIBERO_NEWTON_ISLANDS
#define LIBERO_NEWTON_ISLANDS 0
#endif
#if LIBERO_NEWTON_ISLANDS
namespace libero_islands {
__device__ bool factor(real_t*,real_t*,int*,const real_t*,const int*,int,
    const int*,const int*,const real_t*,const int*,const int*,const int*,int*);
}
#endif

// All block lanes participate. Warp votes cover strided matrix/input scans;
// a block vote publishes the uniform decision across multiple warps.
__device__ inline bool factor_any_bad(bool bad) {
    return __syncthreads_or(libero_warp::any(bad));
}

template<int N>
__device__ void block_potrf(real_t* H, real_t* temporary, int* diagnostic) {
    __shared__ int info;
    __shared__ int use_source;

    // Keep original operands intact until POTRF is admitted. This also keeps
    // the unused upper triangle intact, including caller-owned dirty bits.
#if LIBERO_NEWTON_SCALAR_IO
    for (int i = threadIdx.x; i < N*N; i += blockDim.x)
        temporary[i] = H[i];
#else
    libero_newton_io::copy_block(H, temporary, size_t(N)*N);
#endif // LIBERO_NEWTON_SCALAR_IO
    __syncthreads();
    if constexpr (N == NV) dx_potrf_nv(temporary, &info);
    else if constexpr (N == 1) dx_potrf_1(temporary, &info);
    else if constexpr (N == 3) dx_potrf_3(temporary, &info);
    else if constexpr (N == 4) dx_potrf_4(temporary, &info);
    else if constexpr (N == 6) dx_potrf_6(temporary, &info);
    __syncthreads();

    if (threadIdx.x == 0) {
        ++diagnostic[N == NV ? 0 : 1];
        use_source = info != 0;
        if (info != 0) ++diagnostic[3];
        // MuJoCo clamps each elimination pivot below MINVAL. POTRF has no
        // such regularization, even for positive but tiny pivots.
        if constexpr (N != NV || N <= 6) {
            for (int j = 0; j < N && !use_source; ++j) {
                real_t d = temporary[j*N+j];
                if (!isfinite(d) || d*d < REAL_C(1e-15)) use_source = 1;
            }
        }
    }
    __syncthreads();
    if constexpr (N == NV && N > 6) {
        bool bad = false;
        if (!use_source) {
            for (int j = threadIdx.x; j < N; j += blockDim.x) {
                const real_t d = temporary[j*N+j];
                bad |= !isfinite(d) || d*d < REAL_C(1e-15);
            }
        }
        const bool bad_pivot = factor_any_bad(bad);
        if (threadIdx.x == 0 && bad_pivot) use_source = 1;
        __syncthreads();
    }
    if (use_source) {
        if (threadIdx.x == 0) {
            ++diagnostic[2];
            chol_factor(H, N, N);
        }
    } else {
#if LIBERO_NEWTON_SCALAR_IO
        for (int i = threadIdx.x; i < N*N; i += blockDim.x)
            if (i/N >= i%N) H[i] = temporary[i];
#else
        // Each lane owns complete lower-row spans. No packet reaches the
        // caller-owned upper triangle, including the odd-row scalar tail.
        for (int row = threadIdx.x; row < N; row += blockDim.x)
            libero_newton_io::copy_serial(temporary + row*N, H + row*N, row+1);
#endif // LIBERO_NEWTON_SCALAR_IO
    }
    __syncthreads();
}

__device__ void local_potrf(int n, real_t* H, real_t* temporary, int* diagnostic) {
    // Contact dimensionality is a runtime value; descriptors are compile-time.
    switch (n) {
        case 1: block_potrf<1>(H, temporary, diagnostic); break;
        case 3: block_potrf<3>(H, temporary, diagnostic); break;
        case 4: block_potrf<4>(H, temporary, diagnostic); break;
        case 6: block_potrf<6>(H, temporary, diagnostic); break;
    }
}

__device__ bool hessian_potrf_dispatch_dx(real_t* H,real_t* temporary,int* diagnostic,
    real_t* J,int* state,int count,int* id,int* dim,real_t* qm,int* madr,
    int* parent,int* simple,int* islands_diagnostic) {
#if LIBERO_NEWTON_ISLANDS
    // factor owns both exact-block admission and its complete dense fallback.
    // The complete authoritative lower matrix is sufficient and is rebuilt
    // for every factor call. Do not introduce reads of otherwise-unused qM
    // (kind 2) or zero-weight quadratic Jacobian rows.
    return libero_islands::factor(H,temporary,diagnostic,nullptr,nullptr,0,
                                 nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
                                 islands_diagnostic);
#else
    block_potrf<NV>(H,temporary,diagnostic);
    return true;
#endif
}

#if LIBERO_HESSIAN_BLAS
__device__ void hessian_cone_batch_dx(int* count,int* state,int* id,int* dim,
    real_t* contact_H,real_t* J,real_t* H,real_t* HC,real_t* local,real_t* ltj,
    int* nupdate,real_t* temporary,int* diagnostic,int* blas_diagnostic,
    real_t* pending,int* algorithm_diagnostic,real_t* qm,int* madr,int* parent,
    int* simple,int* islands_diagnostic,int* factor_status) {
    // H remains the quadratic/base factor. The final HC lower factor is built
    // from exactly that effective base and the same regularized contact factors.
    // H/HC upper owner bits are never operands and retain the source values.
    hessian_from_factor_blas_dx(H,HC,temporary,algorithm_diagnostic);
    int used=0;
    for(int i=0;i<count[0];) {
        if(state[i]!=4){++i;continue;}
        const int c=id[i],d=dim[c];
        libero_newton_io::copy_block(contact_H+c*36,local,size_t(d)*d);
        __syncthreads();
        local_potrf(d,local,temporary,diagnostic);
        cone_ltj_blas_dx(i,d,J,local,ltj,temporary,blas_diagnostic);
        for(int r=0;r<d;) {
            const int take=min(d-r,DX_HESSIAN_K-used);
            for(int z=threadIdx.x;z<take*NV;z+=blockDim.x)
                pending[used*NV+z]=ltj[r*NV+z];
            __syncthreads();
            used+=take;r+=take;
            if(used==DX_HESSIAN_K) {
                cone_gram_append_blas_dx(pending,used,HC,temporary,algorithm_diagnostic);
                used=0;
            }
        }
        if(threadIdx.x==0)nupdate[0]+=d;
        __syncthreads();
        i+=d;
    }
    if(used)cone_gram_append_blas_dx(pending,used,HC,temporary,algorithm_diagnostic);
    const bool success=hessian_potrf_dispatch_dx(HC,temporary,diagnostic,J,state,count[0],id,dim,
                                               qm,madr,parent,simple,islands_diagnostic);
    if(threadIdx.x==0&&!success&&factor_status)factor_status[0]=5;
    if(threadIdx.x==0&&algorithm_diagnostic)++algorithm_diagnostic[0];
    __syncthreads();
}
#endif

__device__ void hessian_cone_dx(int rows, int contacts, int* count, int* state,
    int* id, int* dim, real_t* contact_H, real_t* J, real_t* H, real_t* HC,
    real_t* local, real_t* ltj, real_t* vec, int* nupdate,
    real_t* temporary, int* diagnostic,int* blas_diagnostic=nullptr,
    real_t* pending=nullptr,int* algorithm_diagnostic=nullptr,
    real_t* qm=nullptr,int* madr=nullptr,int* parent=nullptr,int* simple=nullptr,
    int* islands_diagnostic=nullptr,int* factor_status=nullptr) {
#if LIBERO_HESSIAN_BLAS && LIBERO_CONE_HESSIAN_STRATEGY != 0
    int ranks=0;
    // ncone counts rows, but this standalone helper deliberately has no ncone
    // parameter. Count validated state==4 rows without reading inactive J.
    if constexpr(LIBERO_CONE_HESSIAN_STRATEGY==2)
        for(int r=0;r<count[0];++r)ranks+=state[r]==4;
    const bool batch=pending&&(LIBERO_CONE_HESSIAN_STRATEGY==1||
                               ranks>=LIBERO_CONE_BATCH_THRESHOLD);
    if(batch) {
        hessian_cone_batch_dx(count,state,id,dim,contact_H,J,H,HC,local,ltj,
            nupdate,temporary,diagnostic,blas_diagnostic,pending,algorithm_diagnostic,
            qm,madr,parent,simple,islands_diagnostic,factor_status);
        return;
    }
#endif
    if(threadIdx.x==0&&algorithm_diagnostic)++algorithm_diagnostic[1];
#if LIBERO_NEWTON_SCALAR_IO
    for (int z = threadIdx.x; z < NV*NV; z += blockDim.x) HC[z] = H[z];
#else
    libero_newton_io::copy_block(H, HC, size_t(NV)*NV);
#endif // LIBERO_NEWTON_SCALAR_IO
    __syncthreads();
    for (int i = 0; i < count[0];) {
        if (state[i] != 4) { ++i; continue; }
        int c = id[i], d = dim[c];
#if LIBERO_NEWTON_SCALAR_IO
        for (int z = threadIdx.x; z < d*d; z += blockDim.x)
            local[z] = contact_H[c*36+z];
#else
        libero_newton_io::copy_block(contact_H+c*36, local, size_t(d)*d);
#endif // LIBERO_NEWTON_SCALAR_IO
        __syncthreads();
        local_potrf(d, local, temporary, diagnostic);
#if LIBERO_HESSIAN_BLAS
        cone_ltj_blas_dx(i,d,J,local,ltj,temporary,blas_diagnostic);
#endif
        if (threadIdx.x == 0) {
#if !LIBERO_HESSIAN_BLAS
            for (int z = 0; z < d*NV; ++z) ltj[z] = REAL_C(0.0);
            for (int r = 0; r < d; ++r)
                for (int c0 = 0; c0 <= r; ++c0) {
                    real_t s = local[r*d+c0];
                    for (int k = 0; k < NV; ++k)
                        ltj[c0*NV+k] += J[(i+r)*NV+k]*s;
                }
#endif
        }
        __syncthreads();
        if constexpr (NV <= 32 || !LIBERO_CHOL_UPDATE_WARP) {
            if (threadIdx.x == 0) {
                for (int r = 0; r < d; ++r) {
#if LIBERO_NEWTON_SCALAR_IO
                    for (int k = 0; k < NV; ++k) vec[k] = ltj[r*NV+k];
#else
                    libero_newton_io::copy_serial(ltj+r*NV, vec, NV);
#endif // LIBERO_NEWTON_SCALAR_IO
                    chol_update(HC, vec, 1);
                }
            }
        } else {
            for (int r = 0; r < d; ++r) {
#if LIBERO_NEWTON_SCALAR_IO
                for (int k = threadIdx.x; k < NV; k += blockDim.x)
                    vec[k] = ltj[r*NV+k];
#else
                libero_newton_io::copy_block(ltj+r*NV, vec, NV);
#endif // LIBERO_NEWTON_SCALAR_IO
                __syncthreads();
                chol_update_block(HC, vec, 1);
            }
        }
        if (threadIdx.x == 0) nupdate[0] += d;
        __syncthreads();
        i += d;
    }
}

__device__ void hessian_factor_update_dx(int w, int* count, int* ncon,
    int* active, int* kind, real_t* J, real_t* D, int* state, int* oldstate,
    real_t* qm, int* madr, int* parent, int* simple, int* id, int* dim,
    real_t* contact_H, int* ncone, real_t* H, real_t* HC, int* nnz,
    int* nupdate, real_t* local, real_t* ltj, real_t* vec, int* status,
    int* route, int rows, int contacts, real_t* temporary, int* diagnostic,
    int* blas_diagnostic=nullptr,real_t* pending=nullptr,
    int* algorithm_diagnostic=nullptr,int* islands_diagnostic=nullptr,
    real_t* upper_backup=nullptr) {
    // Python supplies per-world pointer views, so this collective uses w=0.
    if (!active[0] || status[0]) return;
    __shared__ int lost,forced,changed_rows;
    int k = kind[0];
    if(threadIdx.x==0&&algorithm_diagnostic)algorithm_diagnostic[6]+=ncone[0];
    if (k == 1) {
        const bool success=hessian_potrf_dispatch_dx(H,temporary,diagnostic,J,state,count[0],id,dim,
                                                   qm,madr,parent,simple,islands_diagnostic);
        if(threadIdx.x==0&&!success)status[0]=5;
        __syncthreads();
        if(status[0])return;
        if (threadIdx.x == 0) { nnz[0] = NV*NV; nupdate[0] = 0; }
        __syncthreads();
        if (ncone[0]) hessian_cone_dx(rows, contacts, count, state, id, dim,
            contact_H, J, H, HC, local, ltj, vec, nupdate, temporary, diagnostic,
            blas_diagnostic,pending,algorithm_diagnostic,qm,madr,parent,simple,islands_diagnostic,status);
        if (threadIdx.x == 0) { route[0] = ncone[0] ? 2 : 1; nupdate[0] = count[0]; }
    } else if (k == 2) {
        hessian_cone_dx(rows, contacts, count, state, id, dim,
            contact_H, J, H, HC, local, ltj, vec, nupdate, temporary, diagnostic,
            blas_diagnostic,pending,algorithm_diagnostic,qm,madr,parent,simple,islands_diagnostic,status);
        if (threadIdx.x == 0) route[0] = 6;
    } else {
        if (threadIdx.x == 0) {
            int changes=0;
            for(int r=0;r<count[0];++r)
                changes+=(oldstate[r]==1)!=(state[r]==1);
            // Threshold 0 disables this policy. It is a configurable algorithm
            // crossover experiment, not a measured platform tuning result.
            forced=upper_backup&&LIBERO_RANK_REBUILD_THRESHOLD>0&&changes>=LIBERO_RANK_REBUILD_THRESHOLD;
            lost=forced;changed_rows=changes;
            nupdate[0]=0;
            if(algorithm_diagnostic){
                algorithm_diagnostic[5]+=changes;
                algorithm_diagnostic[4]+=lost;
            }
        }
        __syncthreads();
        if (!lost) {
        if constexpr (NV <= 32 || !LIBERO_CHOL_UPDATE_WARP) {
            if (threadIdx.x == 0) {
                for (int r = 0; r < count[0]; ++r) {
                    const bool before = oldstate[r] == 1, after = state[r] == 1;
                    if (before == after) continue;
                    const real_t s = sqrt(D[r]);
                    for (int j = 0; j < NV; ++j) vec[j] = J[r*NV+j]*s;
                    const int rank = chol_update(H, vec, after ? 1 : 0);
                    ++nupdate[0];
                    if (rank < NV) { lost = 1; break; }
                }
            }
        } else {
            for (int r = 0; r < count[0]; ++r) {
                const bool before = oldstate[r] == 1, after = state[r] == 1;
                if (before == after) continue;
                const real_t s = sqrt(D[r]);
                for (int j = threadIdx.x; j < NV; j += blockDim.x)
                    vec[j] = J[r*NV+j]*s;
                __syncthreads();
                const int rank = chol_update_block(H, vec, after ? 1 : 0);
                if (threadIdx.x == 0) { ++nupdate[0]; if (rank < NV) lost = 1; }
                __syncthreads();
                if (lost) break;
            }
        }
        }
        __syncthreads();
        if (lost) {
            bool bad = false;
            for (int z = threadIdx.x; z < NM; z += blockDim.x)
                bad |= !isfinite(qm[z]);
            for (int r = 0; r < count[0]; ++r) {
                if (state[r] != 1) continue;
                if (threadIdx.x == 0) bad |= D[r] < 0 || !isfinite(D[r]);
                if (D[r] != 0)
                    for (int j = threadIdx.x; j < NV; j += blockDim.x)
                        bad |= !isfinite(J[r*NV+j]);
            }
            const bool bad_input = factor_any_bad(bad);
            if (threadIdx.x == 0 && bad_input) status[0] = 4;
            __syncthreads();
        }
        if (status[0]) return; // uniform block return after the barrier
        if (lost) {
            if(forced) {
                for(int z=threadIdx.x;z<NV*NV;z+=blockDim.x)
                    if(z/NV<z%NV)upper_backup[z]=H[z];
                __syncthreads();
            }
#if LIBERO_HESSIAN_BLAS
            hessian_rebuild_blas_dx(count,J,D,state,qm,madr,parent,simple,
                                   H,temporary,blas_diagnostic);
#else
            for (int z = threadIdx.x; z < NV*NV; z += blockDim.x) {
                int i = z/NV, j = z%NV;
                if (j > i) continue;
                real_t v = assembly_cell(0, i, j, rows, count, J, D, state,
                                          qm, madr, parent, simple);
                H[z] = v;
                if (i != j) H[j*NV+i] = v;
            }
            __syncthreads();
#endif
            const bool success=hessian_potrf_dispatch_dx(H,temporary,diagnostic,J,state,count[0],id,dim,
                                                       qm,madr,parent,simple,islands_diagnostic);
            if(threadIdx.x==0&&!success)status[0]=5;
            __syncthreads();
            if(status[0])return;
            if(forced) {
                for(int z=threadIdx.x;z<NV*NV;z+=blockDim.x)
                    if(z/NV<z%NV)H[z]=upper_backup[z];
                __syncthreads();
            }
            if (threadIdx.x == 0) {
                nnz[0] = NV*NV;
                if(forced)nupdate[0]=changed_rows;
                else if(ncone[0])nupdate[0]=0;
            }
            __syncthreads();
            if (ncone[0]) hessian_cone_dx(rows, contacts, count, state, id, dim,
                contact_H, J, H, HC, local, ltj, vec, nupdate, temporary, diagnostic,
                blas_diagnostic,pending,algorithm_diagnostic,qm,madr,parent,simple,islands_diagnostic,status);
            if (threadIdx.x == 0) {
                if(forced)route[0]=ncone[0]?4:3;
                else {nupdate[0]=count[0];route[0]=5;}
            }
        } else {
            if (ncone[0]) hessian_cone_dx(rows, contacts, count, state, id, dim,
                contact_H, J, H, HC, local, ltj, vec, nupdate, temporary, diagnostic,
                blas_diagnostic,pending,algorithm_diagnostic,qm,madr,parent,simple,islands_diagnostic,status);
            if (threadIdx.x == 0) route[0] = ncone[0] ? 4 : 3;
        }
    }
    __syncthreads();
    bool bad_output = false;
    for (int z = threadIdx.x; z < NV*NV; z += blockDim.x) {
        const bool check = k == 1 || route[0] == 5 ||
                          (k == 3 && route[0] != 5 && z/NV >= z%NV);
        bad_output |= (check && !isfinite(H[z])) ||
                      (ncone[0] && !isfinite(HC[z]));
    }
    const bool bad_factor = factor_any_bad(bad_output);
    if (threadIdx.x == 0 && bad_factor) status[0] = 5;
    __syncthreads();
}
