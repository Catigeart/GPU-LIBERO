#include "precision.cuh"
#include "io_policy.cuh"
#pragma once
// A block-local C = sum_r (J_r^T D_r) J_r. Only admitted quadratic rows
// enter the dense BLAS tile. Input packing and sparse M expansion are hand code.
extern __device__ void dx_hessian_gemm(real_t*,real_t*,real_t*);
extern __device__ void dx_newton_jv(real_t*,real_t*,real_t*);
extern __device__ void dx_newton_transpose(real_t*,real_t*,real_t*);
extern __device__ void dx_newton_cone_ltj(real_t*,real_t*,real_t*);

// Dense reconstruction from the admitted lower factor. Upper-triangle owner
// bits are ignored. This makes the batch cone path independent of the current
// source matrix/factor lifetime and keeps base H available for the next update.
__device__ void hessian_from_factor_blas_dx(const real_t* factor,real_t* H,
    real_t* workspace,int* algorithm_diagnostic) {
    real_t* C=workspace;
    real_t* A=C+NV*NV;
    real_t* B=A+NV*DX_HESSIAN_K;
    libero_newton_io::fill_block(C,NV*NV,REAL_C(0.0));
    __syncthreads();
    for(int begin=0;begin<NV;begin+=DX_HESSIAN_K) {
        for(int z=threadIdx.x;z<NV*DX_HESSIAN_K;z+=blockDim.x) {
            const int i=z/DX_HESSIAN_K,k=begin+z%DX_HESSIAN_K;
            const real_t v=k<NV&&k<=i?factor[i*NV+k]:REAL_C(0.0);
            A[z]=v;
            B[(z%DX_HESSIAN_K)*NV+i]=v;
        }
        __syncthreads();
        dx_hessian_gemm(A,B,C);
        if(threadIdx.x==0&&algorithm_diagnostic)++algorithm_diagnostic[3];
        __syncthreads();
    }
    for(int z=threadIdx.x;z<NV*NV;z+=blockDim.x)
        H[z]=z/NV>=z%NV?C[z]:factor[z];
    __syncthreads();
}

// A pending U tile is a separate owner: cone POTRF/L^T J scratch may overwrite
// workspace between appends. Only this call consumes/clears the pending tile.
__device__ void cone_gram_append_blas_dx(const real_t* pending,int used,
    real_t* H,real_t* workspace,int* algorithm_diagnostic) {
    real_t* C=workspace;
    real_t* A=C+NV*NV;
    real_t* B=A+NV*DX_HESSIAN_K;
    libero_newton_io::fill_block(C,NV*NV,REAL_C(0.0));
    for(int z=threadIdx.x;z<NV*DX_HESSIAN_K;z+=blockDim.x) {
        const int k=z/NV,j=z%NV;
        const real_t v=k<used?pending[z]:REAL_C(0.0);
        B[z]=v;
        A[j*DX_HESSIAN_K+k]=v;
    }
    __syncthreads();
    dx_hessian_gemm(A,B,C);
    for(int z=threadIdx.x;z<NV*NV;z+=blockDim.x)
        if(z/NV>=z%NV)H[z]+=C[z];
    if(threadIdx.x==0&&algorithm_diagnostic)++algorithm_diagnostic[2];
    __syncthreads();
}

__device__ void hessian_rebuild_blas_dx(int* count,real_t* J,real_t* D,int* state,
    real_t* qm,int* madr,int* parent,int* simple,real_t* H,real_t* workspace,
    int* diagnostic) {
    real_t* C=workspace;
    real_t* A=C+NV*NV;
    real_t* B=A+NV*DX_HESSIAN_K;
    // Row metadata is shared once per tile. These arrays add 12*K static
    // shared bytes (192 for K=16), accounted by the persistent host owner.
    __shared__ real_t row_D[DX_HESSIAN_K];
    __shared__ int row_admitted[DX_HESSIAN_K];
    for(int z=threadIdx.x;z<NV*NV;z+=blockDim.x) C[z]=0;
    __syncthreads();
    for(int begin=0;begin<count[0];begin+=DX_HESSIAN_K) {
        for(int k=threadIdx.x;k<DX_HESSIAN_K;k+=blockDim.x) {
            const int r=begin+k;
            // Preserve short-circuiting: rejected/tail rows do not read D.
            const bool quadratic=r<count[0]&&state[r]==1;
            const real_t d=quadratic ? D[r] : 0;
            row_D[k]=d;
            row_admitted[k]=quadratic&&d!=0;
        }
        __syncthreads();
#if LIBERO_NEWTON_SCALAR_IO
        for(int z=threadIdx.x;z<NV*DX_HESSIAN_K;z+=blockDim.x) {
            // Linear [k][i] traversal coalesces the single global J load.
#else
        constexpr int row_pairs=(NV+1)/2;
        for(int z=threadIdx.x;z<row_pairs*DX_HESSIAN_K;z+=blockDim.x) {
            // Pair within one admitted [k][i] row; an odd final cell is scalar.
#endif // LIBERO_NEWTON_SCALAR_IO
            // Scatter its transpose into compact MathDx A; B stays row-major.
#if LIBERO_NEWTON_SCALAR_IO
            const int k=z/NV, i=z%NV, r=begin+k;
#else
            const int k=z/row_pairs, i=2*(z%row_pairs), r=begin+k;
#endif // LIBERO_NEWTON_SCALAR_IO
            const bool admitted=row_admitted[k];
            // Masked rows may legally hold nonfinite J and must not be read.
#if LIBERO_NEWTON_SCALAR_IO
            const real_t v=admitted ? J[r*NV+i] : 0;
            A[i*DX_HESSIAN_K+k]=admitted ? v*row_D[k] : 0;
            B[z]=v;
#else
            if(i+1<NV) {
                const real2_t v=admitted ? libero_vec::load_pair(J+r*NV+i) : make_real2(0,0);
                A[i*DX_HESSIAN_K+k]=admitted ? v.x*row_D[k] : 0;
                A[(i+1)*DX_HESSIAN_K+k]=admitted ? v.y*row_D[k] : 0;
                libero_vec::store_pair(B+k*NV+i,v.x,v.y);
            } else {
                const real_t v=admitted ? J[r*NV+i] : 0;
                A[i*DX_HESSIAN_K+k]=admitted ? v*row_D[k] : 0;
                B[k*NV+i]=v;
            }
#endif // LIBERO_NEWTON_SCALAR_IO
        }
        __syncthreads();
        dx_hessian_gemm(A,B,C);
        if(threadIdx.x==0) ++diagnostic[1];
        __syncthreads();
    }
    for(int z=threadIdx.x;z<NV*NV;z+=blockDim.x) {
        int i=z/NV,j=z%NV;
        if(j>i) continue;
        int adr=mass_index(i,j,madr,parent,simple);
        // Add sparse M after Gram summation, following the original source.
        real_t v=C[z]+(adr>=0?qm[adr]:0);
        H[z]=v;
        if(i!=j) H[j*NV+i]=v;
    }
    if(threadIdx.x==0) ++diagnostic[0];
    __syncthreads();
}

__device__ void hessian_assemble_blas_dx(int w,int* count,int* active,int* kind,
    real_t* J,real_t* D,int* state,real_t* qm,int* madr,int* parent,int* simple,
    real_t* H,int* status,int rows,real_t* workspace,int* diagnostic) {
    // All inputs are already sliced to one world by the persistent wrapper.
    if(!active[0]||kind[0]!=1||status[0]) return;
    hessian_rebuild_blas_dx(count,J,D,state,qm,madr,parent,simple,H,workspace,diagnostic);
}

__device__ void jacvec_sub_blas_dx(int w,int* active,int* count,real_t* J,
    real_t* q,real_t* aref,real_t* out,int rows,real_t* workspace,int* diagnostic) {
    if(!active[0]) return;
    real_t* A=workspace;
    real_t* B=A+16*NV;
    real_t* C=B+NV;
#if LIBERO_NEWTON_SCALAR_IO
    for(int j=threadIdx.x;j<NV;j+=blockDim.x) B[j]=q[j];
#else
    // Copies guard the actual shared subsegment, which may be only 8B aligned.
    libero_newton_io::copy_block(q,B,NV);
#endif // LIBERO_NEWTON_SCALAR_IO
    __syncthreads();
    for(int begin=0;begin<count[0];begin+=16) {
#if LIBERO_NEWTON_SCALAR_IO
        for(int z=threadIdx.x;z<16*NV;z+=blockDim.x) {
            int r=begin+z/NV;
            A[z]=r<count[0]?J[r*NV+z%NV]:0;
#else
        constexpr int row_pairs=(NV+1)/2;
        for(int z=threadIdx.x;z<16*row_pairs;z+=blockDim.x) {
            const int k=z/row_pairs,i=2*(z%row_pairs),r=begin+k;
            if(i+1<NV) {
                const real2_t v=r<count[0]?libero_vec::load_pair(J+r*NV+i):make_real2(0,0);
                libero_vec::store_pair(A+k*NV+i,v.x,v.y);
            } else A[k*NV+i]=r<count[0]?J[r*NV+i]:0;
#endif // LIBERO_NEWTON_SCALAR_IO
        }
#if LIBERO_NEWTON_SCALAR_IO
        for(int r=threadIdx.x;r<16;r+=blockDim.x) C[r]=0;
#else
        libero_newton_io::fill_block(C,16,REAL_C(0.0));
#endif // LIBERO_NEWTON_SCALAR_IO
        __syncthreads();
        dx_newton_jv(A,B,C);
#if LIBERO_NEWTON_SCALAR_IO
        for(int r=threadIdx.x;r<16;r+=blockDim.x)
            if(begin+r<count[0]) out[begin+r]=C[r]-aref[begin+r];
#else
        for(int p=threadIdx.x;p<8;p+=blockDim.x) {
            const int r=2*p;
            if(begin+r+1<count[0]) {
                const real2_t c=libero_vec::load_pair(C+r);
                const real2_t a=libero_vec::load_pair(aref+begin+r);
                libero_vec::store_pair(out+begin+r,c.x-a.x,c.y-a.y);
            } else if(begin+r<count[0]) out[begin+r]=C[r]-aref[begin+r];
        }
#endif // LIBERO_NEWTON_SCALAR_IO
        __syncthreads();
    }
    if(threadIdx.x==0) ++diagnostic[3];
    __syncthreads();
}

__device__ void constraint_transpose_blas_dx(int w,int* active,int* count,
    real_t* J,real_t* force,real_t* qfrc,int* status,int rows,int nv,
    real_t* workspace,int* diagnostic) {
    if(!active[0]||status[0]) return;
    real_t* C=workspace;
    real_t* A=C+NV;
    real_t* B=A+NV*DX_HESSIAN_K;
    for(int j=threadIdx.x;j<NV;j+=blockDim.x) C[j]=0;
    __syncthreads();
    for(int begin=0;begin<count[0];begin+=DX_HESSIAN_K) {
        // Load the force tile once. The block-wide vote also publishes B
        // before any lane reads it. No Jacobian load precedes this decision.
        bool nonzero=false;
        for(int k=threadIdx.x;k<DX_HESSIAN_K;k+=blockDim.x) {
            const int r=begin+k;
            const real_t f=r<count[0]?force[r]:0;
            B[k]=f;
            nonzero |= f!=0;
        }
        // Preserve row/tile order. A zero tile contributes no arithmetic;
        // importantly, even NaN Jacobian rows remain completely unread.
        if(!__syncthreads_or(nonzero)) continue;
#if LIBERO_NEWTON_SCALAR_IO
        for(int z=threadIdx.x;z<NV*DX_HESSIAN_K;z+=blockDim.x) {
#else
        constexpr int row_pairs=(NV+1)/2;
        for(int z=threadIdx.x;z<row_pairs*DX_HESSIAN_K;z+=blockDim.x) {
#endif // LIBERO_NEWTON_SCALAR_IO
            // Read global J in its contiguous [k][j] order, transposing only
            // this block-local tile into the existing compact MathDx A.
#if LIBERO_NEWTON_SCALAR_IO
            const int k=z/NV,j=z%NV,r=begin+k;
#else
            const int k=z/row_pairs,j=2*(z%row_pairs),r=begin+k;
#endif // LIBERO_NEWTON_SCALAR_IO
            const real_t f=B[k];
            // Zero force intentionally ignores its corresponding Jacobian row.
#if LIBERO_NEWTON_SCALAR_IO
            A[j*DX_HESSIAN_K+k]=f!=0?J[r*NV+j]:0;
#else
            if(j+1<NV) {
                const real2_t v=f!=0?libero_vec::load_pair(J+r*NV+j):make_real2(0,0);
                A[j*DX_HESSIAN_K+k]=v.x;
                A[(j+1)*DX_HESSIAN_K+k]=v.y;
            } else A[j*DX_HESSIAN_K+k]=f!=0?J[r*NV+j]:0;
#endif // LIBERO_NEWTON_SCALAR_IO
        }
        __syncthreads();
        dx_newton_transpose(A,B,C);
        __syncthreads();
    }
    for(int j=threadIdx.x;j<NV;j+=blockDim.x) qfrc[j]=C[j];
    if(threadIdx.x==0) ++diagnostic[4];
    __syncthreads();
}

__device__ void cone_ltj_blas_dx(int begin,int dim,real_t* J,real_t* local,
    real_t* ltj,real_t* workspace,int* diagnostic) {
    real_t* C=workspace;
    real_t* A=C+6*NV;
    real_t* B=A+36;
    for(int z=threadIdx.x;z<36;z+=blockDim.x) {
        int r=z/6,k=z%6;
        A[z]=(r<dim&&k<dim&&k>=r)?local[k*dim+r]:0;
    }
    for(int z=threadIdx.x;z<6*NV;z+=blockDim.x) {
        int r=z/NV;
        B[z]=r<dim?J[(begin+r)*NV+z%NV]:0;
        C[z]=0;
    }
    __syncthreads();
    dx_newton_cone_ltj(A,B,C);
    for(int z=threadIdx.x;z<dim*NV;z+=blockDim.x) ltj[z]=C[z];
    if(threadIdx.x==0) ++diagnostic[5];
    __syncthreads();
}
