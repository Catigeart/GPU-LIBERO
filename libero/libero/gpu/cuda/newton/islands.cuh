#include "precision.cuh"
// Exact numerical Hessian islands. All block lanes enter each public function.
// The lower triangle is authoritative; upper cells remain caller-owned.
#pragma once
#include <math.h>

extern __device__ void dx_island_potrf(int, real_t*, int*);
extern __device__ void dx_island_forward(int, real_t*, real_t*);
extern __device__ void dx_island_backward(int, real_t*, real_t*);

namespace libero_islands {
enum Diagnostic {
    partition_calls=0, partitioned_calls=1, factor_calls=2, dense_fallbacks=3,
    padded_cells_total=4, max_components=5, max_component_size=6,
    direction_calls=7, diagnostic_count=8
};
struct Partition {
    int parent[NV], permutation[NV], begin[NV+1], capacity[NV], matrix[NV];
    int components, cells, bad;
};

__device__ inline int root(int* parent, int v) {
    int p=atomicAdd(parent+v,0);
    while(p!=v) {v=p;p=atomicAdd(parent+v,0);}
    return v;
}
__device__ inline void join(int* parent,int a,int b) {
    for (;;) {
        a=root(parent,a);b=root(parent,b);
        if(a==b)return;
        if(a<b){int t=a;a=b;b=t;}
        if(atomicCAS(parent+a,a,b)==a)return;
    }
}
__device__ inline int padded_capacity(int n) {
    if(n<=1)return 1;
    if constexpr(NV>=2)if(n<=2)return 2;
    if constexpr(NV>=4)if(n<=4)return 4;
    if constexpr(NV>=8)if(n<=8)return 8;
    if constexpr(NV>=16)if(n<=16)return 16;
    if constexpr(NV>=32)if(n<=32)return 32;
    return NV;
}

// Mandatory graph: every exact nonzero in the complete lower Hessian.
// Optional graph: actual packed mass edges and unions of effective J support.
// A cone merges the support of ALL its rows, including cross-row Hessian terms.
// These optional edges can only merge components; they never remove an H edge.
__device__ inline bool partition(const real_t* lower,Partition& p,
    const real_t* J,const int* state,int rows,const int* id,const int* dim,
    const real_t* M,const int* madr,const int* ancestor,const int* simple,
    const real_t* D=nullptr) {
    if(threadIdx.x==0){p.bad=0;p.components=0;p.cells=0;
        if(rows<0) p.bad=1;
        if((J!=nullptr)!=(state!=nullptr))p.bad=1;
        if(M && (!madr||!ancestor||!simple))p.bad=1;
    }
    for(int i=threadIdx.x;i<NV;i+=blockDim.x)p.parent[i]=i;
    __syncthreads();
    for(int z=threadIdx.x;z<NV*NV;z+=blockDim.x){
        const int r=z/NV,c=z%NV;
        if(c<=r){const real_t v=lower[z];
            if(!isfinite(v))atomicExch(&p.bad,1);
            else if(r!=c && v!=REAL_C(0.0))join(p.parent,r,c);
        }
    }
    if(M && madr && ancestor && simple) {
        for(int i=threadIdx.x;i<NV;i+=blockDim.x){
            int address=madr[i];
            if(address<0){atomicExch(&p.bad,1);continue;}
            // Model ancestors and packed bounds are admitted by the caller.
            int depth=0;
            for(int j=i;j>=0;j=ancestor[j],++address){
                if(j>=NV||++depth>NV){atomicExch(&p.bad,1);break;}
                const real_t v=M[address];
                if(!isfinite(v))atomicExch(&p.bad,1);
                else if(j!=i&&v!=REAL_C(0.0))join(p.parent,i,j);
                if(simple[i])break;
            }
        }
    }
    if(J && state && rows>=0) {
        for(int row=threadIdx.x;row<rows;row+=blockDim.x){
            int count=1;
            if(state[row]==4){
                if(!id||!dim){atomicExch(&p.bad,1);continue;}
                const int contact=id[row];
                if(contact<0){atomicExch(&p.bad,1);continue;}
                // Only a cone head owns the group; schema admission precedes us.
                if(row>0&&state[row-1]==4&&id[row-1]==contact)continue;
                count=dim[contact];
                if((count!=1&&count!=3&&count!=4&&count!=6)||row+count>rows){
                    atomicExch(&p.bad,1);continue;
                }
            } else if(state[row]!=1)continue;
            else {
                // D==0 quadratic rows do not consume J in the baseline. If D
                // was not supplied, the actual H graph remains the proof and
                // this optional conservative support pass skips scalar rows.
                if(!D||D[row]==REAL_C(0.0))continue;
                if(!isfinite(D[row])){atomicExch(&p.bad,1);continue;}
            }
            int first=-1;
            for(int r=row;r<row+count;++r) {
                if(count>1&&(state[r]!=4||id[r]!=id[row]))atomicExch(&p.bad,1);
                for(int c=0;c<NV;++c){const real_t v=J[r*NV+c];
                    if(!isfinite(v))atomicExch(&p.bad,1);
                    else if(v!=REAL_C(0.0)){if(first<0)first=c;else join(p.parent,first,c);}
                }
            }
        }
    }
    __syncthreads();
    if(p.bad)return false;
    // Canonical components and columns preserve original DOF order within a block.
    if(threadIdx.x==0){
        int cursor=0;
        for(int i=0;i<NV;++i)p.parent[i]=root(p.parent,i);
        for(int owner=0;owner<NV;++owner)if(p.parent[owner]==owner){
            const int component=p.components++;
            p.begin[component]=cursor;
            for(int i=0;i<NV;++i)if(p.parent[i]==owner)p.permutation[cursor++]=i;
            p.capacity[component]=padded_capacity(cursor-p.begin[component]);
            // Align each Dx operand, including odd-capacity NV operands, to 16 B.
            p.cells=(p.cells+1)&~1;p.matrix[component]=p.cells;
            p.cells+=p.capacity[component]*p.capacity[component];
        }
        p.begin[p.components]=cursor;
    }
    __syncthreads();
    // Defense in depth: no threshold, dropped tiny coupling, or body-only proof.
    for(int z=threadIdx.x;z<NV*NV;z+=blockDim.x){
        const int r=z/NV,c=z%NV;
        if(c<r&&p.parent[r]!=p.parent[c]&&lower[z]!=REAL_C(0.0))atomicExch(&p.bad,1);
    }
    __syncthreads();
    return !p.bad;
}

__device__ inline void note_partition(const Partition& p,int* diag) {
    if(threadIdx.x==0&&diag){
        ++diag[partition_calls];
        if(p.components>diag[max_components])diag[max_components]=p.components;
        for(int c=0;c<p.components;++c){const int n=p.begin[c+1]-p.begin[c];
            if(n>diag[max_component_size])diag[max_component_size]=n;
        }
    }
}

__device__ inline bool lower_valid(const real_t* matrix,int n,int stride,
                                 bool check_floor) {
    bool bad=false;
    for(int z=threadIdx.x;z<n*n;z+=blockDim.x){
        const int r=z/n,c=z%n;
        if(c<=r)bad|=!isfinite(matrix[r*stride+c]);
        if(c==r){const real_t v=matrix[r*stride+c];
            bad|=v<=REAL_C(0.0);
            if(check_floor)bad|=v*v<REAL_C(1e-15);
        }
    }
    return !__syncthreads_or(bad);
}

// All exceptional pivots are processed in private staging, preserving input
// until an entire usable factor exists. Regularization matches chol_factor.
__device__ inline bool whole_factor(real_t* H,real_t* temporary,int* diag,
                                   int* islands_diag) {
    __shared__ int info,source;
    for(int z=threadIdx.x;z<NV*NV;z+=blockDim.x)
        temporary[z]=(z%NV<=z/NV)?H[z]:REAL_C(0.0);
    __syncthreads();
    dx_island_potrf(NV,temporary,&info);
    __syncthreads();
    const bool valid=lower_valid(temporary,NV,NV,true);
    if(threadIdx.x==0){
        if(diag){++diag[0];if(info)++diag[3];}
        if(islands_diag){++islands_diag[factor_calls];++islands_diag[dense_fallbacks];}
        source=info!=0||!valid;
        if(source&&diag)++diag[2];
    }
    __syncthreads();
    if(source){
        for(int z=threadIdx.x;z<NV*NV;z+=blockDim.x)
            temporary[z]=(z%NV<=z/NV)?H[z]:REAL_C(0.0);
        __syncthreads();
        if(threadIdx.x==0)chol_factor(temporary,NV,NV);
        __syncthreads();
    }
    if(!lower_valid(temporary,NV,NV,false))return false;
    for(int z=threadIdx.x;z<NV*NV;z+=blockDim.x)
        if(z%NV<=z/NV)H[z]=temporary[z];
    __syncthreads();
    return true;
}

// temporary is the caller's existing, aligned NV*NV Dx workspace. No second
// large matrix is allocated. Padded-size admission falls back to full POTRF.
// Return true only after an entire factor is committed; false preserves H.
__device__ bool factor(real_t* H,real_t* temporary,int* diagnostic,
    const real_t* J,const int* state,int rows,const int* id,const int* dim,
    const real_t* M,const int* madr,const int* ancestor,const int* simple,
    int* islands_diag) {
    __shared__ Partition p;
    __shared__ int info;
    if(!partition(H,p,J,state,rows,id,dim,M,madr,ancestor,simple))return false;
    note_partition(p,islands_diag);
    if(p.components==1||p.cells>NV*NV)
        return whole_factor(H,temporary,diagnostic,islands_diag);
    // All staging blocks remain available until every POTRF is admitted.
    for(int component=0;component<p.components;++component){
        const int begin=p.begin[component],n=p.begin[component+1]-begin;
        const int cap=p.capacity[component];real_t* A=temporary+p.matrix[component];
        for(int z=threadIdx.x;z<cap*cap;z+=blockDim.x){
            const int r=z/cap,c=z%cap;
            if(r<n&&c<n){const int rr=p.permutation[begin+r],cc=p.permutation[begin+c];
                A[z]=r>=c?H[rr*NV+cc]:REAL_C(0.0);
            }else A[z]=r==c?REAL_C(1.0):REAL_C(0.0);
        }
        __syncthreads();
        dx_island_potrf(cap,A,&info);
        __syncthreads();
        const bool valid=lower_valid(A,n,cap,true);
        if(threadIdx.x==0){if(diagnostic){++diagnostic[0];if(info)++diagnostic[3];}
            if(islands_diag)++islands_diag[factor_calls];}
        __syncthreads();
        if(info!=0||!valid)return whole_factor(H,temporary,diagnostic,islands_diag);
    }
    // Components are independent: zero off-block factor cells exactly.
    for(int z=threadIdx.x;z<NV*NV;z+=blockDim.x)
        if(z%NV<=z/NV)H[z]=REAL_C(0.0);
    __syncthreads();
    for(int component=0;component<p.components;++component){
        const int begin=p.begin[component],n=p.begin[component+1]-begin;
        const int cap=p.capacity[component];const real_t* A=temporary+p.matrix[component];
        for(int z=threadIdx.x;z<n*n;z+=blockDim.x){
            const int r=z/n,c=z%n;
            if(c<=r)H[p.permutation[begin+r]*NV+p.permutation[begin+c]]=A[r*cap+c];
        }
    }
    if(threadIdx.x==0&&islands_diag){++islands_diag[partitioned_calls];
        islands_diag[padded_cells_total]+=p.cells;}
    __syncthreads();
    return true;
}

// Solve H x = rhs using the committed factor. Input/output may alias. The
// entire result is held privately until finite, and the global Newton caller
// keeps its existing gradient, common line search, and stopping reduction.
__device__ bool direction_with_forward(const real_t* L,const real_t* rhs,real_t* out,
                         real_t* forward_out,real_t* temporary,int* islands_diag) {
    __shared__ Partition p;
    __shared__ __align__(16) real_t b[NV],candidate[NV],forward[NV];
    if(!partition(L,p,nullptr,nullptr,0,nullptr,nullptr,
                  nullptr,nullptr,nullptr,nullptr))return false;
    bool bad=false;
    for(int i=threadIdx.x;i<NV;i+=blockDim.x)
        bad|=!isfinite(rhs[i])||!isfinite(L[i*NV+i])||L[i*NV+i]<=REAL_C(0.0);
    if(__syncthreads_or(bad))return false;
    for(int component=0;component<p.components;++component){
        const int begin=p.begin[component],n=p.begin[component+1]-begin;
        const int cap=p.capacity[component];
        for(int z=threadIdx.x;z<cap*cap;z+=blockDim.x){
            const int r=z/cap,c=z%cap;
            if(r<n&&c<n)temporary[z]=r>=c?L[p.permutation[begin+r]*NV+p.permutation[begin+c]]:REAL_C(0.0);
            else temporary[z]=r==c?REAL_C(1.0):REAL_C(0.0);
        }
        for(int i=threadIdx.x;i<cap;i+=blockDim.x)
            b[i]=i<n?rhs[p.permutation[begin+i]]:REAL_C(0.0);
        __syncthreads();
        dx_island_forward(cap,temporary,b);__syncthreads();
        for(int i=threadIdx.x;i<n;i+=blockDim.x)
            forward[p.permutation[begin+i]]=b[i];
        __syncthreads();
        dx_island_backward(cap,temporary,b);__syncthreads();
        bad=false;
        for(int i=threadIdx.x;i<n;i+=blockDim.x){
            bad|=!isfinite(b[i])||!isfinite(forward[p.permutation[begin+i]]);
            candidate[p.permutation[begin+i]]=b[i];
        }
        if(__syncthreads_or(bad))return false;
    }
    for(int i=threadIdx.x;i<NV;i+=blockDim.x){
        if(forward_out)forward_out[i]=forward[i];
        out[i]=candidate[i];
    }
    if(threadIdx.x==0&&islands_diag)++islands_diag[direction_calls];
    __syncthreads();
    return true;
}
__device__ bool direction(const real_t* L,const real_t* rhs,real_t* out,
                         real_t* temporary,int* islands_diag) {
    return direction_with_forward(L,rhs,out,nullptr,temporary,islands_diag);
}
} // namespace libero_islands
