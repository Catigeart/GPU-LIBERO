#include "precision.cuh"
#pragma once
// Handwritten connected dense factor/solve. One CTA owns one world. This
// independent experimental route does not alter persistent Newton stops/LS.
#ifndef BF_TILE
#define BF_TILE 16
#endif
#ifndef BF_MAX_NV
#define BF_MAX_NV 128
#endif
#ifndef BF_SHARED_BYTES
#define BF_SHARED_BYTES (4*BF_TILE*BF_TILE*8+64)
#endif
static_assert(BF_TILE==16||BF_TILE==32,"blocked tile must be 16 or 32");
static_assert(BF_MAX_NV>=1,"positive blocked capacity required");
static_assert(BF_SHARED_BYTES<=49152,"portable shared ceiling exceeded");
extern __device__ void dx_blocked_potrf(real_t*,int*);
extern __device__ void dx_blocked_panel(real_t*,real_t*);
extern __device__ void dx_blocked_update(real_t*,real_t*,real_t*);
extern __device__ void dx_blocked_rhs_update(real_t*,real_t*,real_t*);
extern __device__ void dx_blocked_forward(real_t*,real_t*);
extern __device__ void dx_blocked_backward(real_t*,real_t*);

namespace libero_blocked {
// This fallback retains the existing chol_factor dot4 association and floor.
// It starts from original H, never the partially updated Schur complement.
__device__ inline real_t source_dot4(const real_t* a,const real_t* b,int n) {
    real_t s0=0,s1=0,s2=0,s3=0;int k=0;
    if(n>=4){s0=a[0]*b[0];s1=a[1]*b[1];s2=a[2]*b[2];s3=a[3]*b[3];k=4;}
    for(;k+4<=n;k+=4){s0+=a[k]*b[k];s1+=a[k+1]*b[k+1];
        s2+=a[k+2]*b[k+2];s3+=a[k+3]*b[k+3];}
    real_t sum=(s0+s2)+(s1+s3);
    if(n-k==3)sum+=a[k]*b[k]+a[k+1]*b[k+1]+a[k+2]*b[k+2];
    else if(n-k==2)sum+=a[k]*b[k]+a[k+1]*b[k+1];
    else if(n-k==1)sum+=a[k]*b[k];
    return sum;
}
__device__ inline int source_factor(real_t* a,int n,int ld) {
    int rank=n;
    for(int j=0;j<n;++j){real_t pivot=a[j*ld+j];
        if(j)pivot-=source_dot4(a+j*ld,a+j*ld,j);
        if(pivot<REAL_C(1e-15)){pivot=REAL_C(1e-15);--rank;}
        a[j*ld+j]=sqrt(pivot);const real_t inv=REAL_C(1.0)/a[j*ld+j];
        for(int i=j+1;i<n;++i)
            a[i*ld+j]=(a[i*ld+j]-source_dot4(a+i*ld,a+j*ld,j))*inv;
    }
    return rank;
}
__device__ inline void diagonal_tile(const real_t* a,int begin,int n,int ld,real_t* d) {
    constexpr int T=BF_TILE;
    for(int z=threadIdx.x;z<T*T;z+=blockDim.x){int r=z/T,c=z%T;
        int i=begin+r,j=begin+c;
        d[z]=i<n&&j<n?(r>=c?a[i*ld+j]:REAL_C(0.0)):(r==c?REAL_C(1.0):REAL_C(0.0));
    }
    __syncthreads();
}
__device__ inline void check_lower(const real_t* a,int n,int ld,int* bad) {
    for(int z=threadIdx.x;z<n*n;z+=blockDim.x){int i=z/n,j=z%n;
        if(i>=j&&!isfinite(a[i*ld+j]))atomicExch(bad,1);}
    __syncthreads();
}

// args use world-major [world][BF_MAX_NV][ld], vector [world][BF_MAX_NV].
// H/factor and RHS/output alias are safe. Direction and forward must not alias.
// diagnostics[8]: route, panels, rank, regularized pivots, fallback reason,
// error, n, tile. route=1 tiled, 2 whole source-floor fallback. Reasons:
// 1 Dx POTRF info, 2 tiny pivot, 3 nonfinite tile arithmetic. No upper/tail
// output is touched. A blocked/inactive world changes nothing, including diag.
__device__ void factor_direction(int w,const real_t* H,const real_t* rhs,
    real_t* factor,real_t* direction,real_t* forward,const int* nworld,
    int ld,const int* active,int* status,real_t* working,real_t* working_x,
    real_t* working_y,int* diagnostic) {
    if(!active[w]||status[w])return;
    constexpr int T=BF_TILE,TT=T*T;
    __shared__ __align__(16) real_t tile[4*TT];
    __shared__ int control[8];
    real_t* d=tile;real_t* a=tile+TT;real_t* b=tile+2*TT;real_t* c=tile+3*TT;
    const int n=nworld[w];
    if(threadIdx.x==0){for(int z=0;z<8;++z)control[z]=0;
        control[2]=n;control[0]=(n<0||n>BF_MAX_NV||ld<BF_MAX_NV)?1:0;}
    __syncthreads();
    // Reject dimensions uniformly before even forming a matrix pointer offset.
    if(control[0]){
        if(threadIdx.x==0){status[w]=1;int* out=diagnostic+8*w;
            for(int z=0;z<8;++z)out[z]=0;
            out[5]=1;out[6]=n;out[7]=T;}
        return;
    }
    const size_t matrix_stride=size_t(BF_MAX_NV)*size_t(ld);
    const real_t* input=H+size_t(w)*matrix_stride;
    const real_t* r=rhs+size_t(w)*BF_MAX_NV;
    real_t* W=working+size_t(w)*matrix_stride;
    real_t* x=working_x+size_t(w)*BF_MAX_NV;
    real_t* y=working_y+size_t(w)*BF_MAX_NV;
    if(!control[0]){
        check_lower(input,n,ld,control);
        for(int i=threadIdx.x;i<n;i+=blockDim.x)
            if(!isfinite(r[i]))atomicExch(control,1);
        __syncthreads();
        if(control[0]&&threadIdx.x==0)control[0]=2;
        __syncthreads();
    }
    if(!control[0]){
        for(int z=threadIdx.x;z<n*n;z+=blockDim.x){int i=z/n,j=z%n;
            if(i>=j)W[i*ld+j]=input[i*ld+j];}
        __syncthreads();
        for(int k=0;k<n;k+=T){
            diagonal_tile(W,k,n,ld,d);
            if(threadIdx.x==0)control[3]=0;
            __syncthreads();dx_blocked_potrf(d,control+3);
            if(threadIdx.x==0){++control[4];if(control[3])control[1]=1;}
            __syncthreads(); // publish info priority before atomic tiny checks
            for(int i=threadIdx.x;i<T&&k+i<n;i+=blockDim.x){real_t v=d[i*T+i];
                if(!isfinite(v))atomicCAS(control+1,0,3);
                else if(v*v<REAL_C(1e-15))atomicCAS(control+1,0,2);}
            __syncthreads();if(control[1])break;
            for(int z=threadIdx.x;z<TT;z+=blockDim.x){int i=k+z/T,j=k+z%T;
                if(i<n&&j<=i)W[i*ld+j]=d[z];}
            __syncthreads();
            for(int i=k+T;i<n;i+=T){
                for(int z=threadIdx.x;z<TT;z+=blockDim.x){int ri=i+z/T,cj=k+z%T;
                    a[z]=ri<n&&cj<n?W[ri*ld+cj]:REAL_C(0.0);}
                __syncthreads();dx_blocked_panel(d,a);
                for(int z=threadIdx.x;z<TT;z+=blockDim.x){int ri=i+z/T,cj=k+z%T;
                    if(ri<n&&cj<n)W[ri*ld+cj]=a[z];}
                __syncthreads();
            }
            for(int i=k+T;i<n;i+=T)for(int j=k+T;j<=i;j+=T){
                for(int z=threadIdx.x;z<TT;z+=blockDim.x){int row=z/T,col=z%T;
                    a[z]=i+row<n&&k+col<n?W[(i+row)*ld+k+col]:REAL_C(0.0);
                    b[z]=j+col<n&&k+row<n?W[(j+col)*ld+k+row]:REAL_C(0.0);
                    c[z]=i+row<n&&j+col<n&&i+row>=j+col?W[(i+row)*ld+j+col]:REAL_C(0.0);}
                __syncthreads();dx_blocked_update(a,b,c);
                for(int z=threadIdx.x;z<TT;z+=blockDim.x){int ri=i+z/T,cj=j+z%T;
                    if(ri<n&&cj<=ri)W[ri*ld+cj]=c[z];}
                __syncthreads();
            }
            // Finite original input can overflow a Schur update. Preserve the
            // original for one whole fallback before refusing publication.
            if(threadIdx.x==0)control[3]=0;
            __syncthreads();check_lower(W,n,ld,control+3);
            if(control[3]&&threadIdx.x==0)control[1]=3;
            __syncthreads();if(control[1])break;
        }
        if(control[1]){
            for(int z=threadIdx.x;z<n*n;z+=blockDim.x){int i=z/n,j=z%n;
                if(i>=j)W[i*ld+j]=input[i*ld+j];}
            __syncthreads();
            if(threadIdx.x==0){control[2]=source_factor(W,n,ld);control[5]=n-control[2];}
            __syncthreads();
        }
        check_lower(W,n,ld,control);
        if(control[0]&&threadIdx.x==0)control[0]=3;
        __syncthreads();
    }
    if(!control[0]){
        for(int k=0;k<n;k+=T){
            diagonal_tile(W,k,n,ld,d);
            for(int i=threadIdx.x;i<T;i+=blockDim.x)c[i]=k+i<n?r[k+i]:REAL_C(0.0);
            __syncthreads();
            for(int q=0;q<k;q+=T){
                for(int z=threadIdx.x;z<TT;z+=blockDim.x){int row=z/T,col=z%T;
                    a[z]=k+row<n?W[(k+row)*ld+q+col]:REAL_C(0.0);}
                for(int j=threadIdx.x;j<T;j+=blockDim.x)b[j]=y[q+j];
                __syncthreads();dx_blocked_rhs_update(a,b,c);
            }
            for(int i=threadIdx.x;i<T;i+=blockDim.x)a[i]=c[i];
            __syncthreads();dx_blocked_forward(d,a);
            for(int i=threadIdx.x;i<T&&k+i<n;i+=blockDim.x)y[k+i]=a[i];
            __syncthreads();
        }
        for(int k=((n+T-1)/T-1)*T;k>=0;k-=T){
            diagonal_tile(W,k,n,ld,d);
            for(int i=threadIdx.x;i<T;i+=blockDim.x)c[i]=k+i<n?y[k+i]:REAL_C(0.0);
            __syncthreads();
            for(int q=k+T;q<n;q+=T){
                for(int z=threadIdx.x;z<TT;z+=blockDim.x){int row=z/T,col=z%T;
                    a[z]=k+row<n&&q+col<n?W[(q+col)*ld+k+row]:REAL_C(0.0);}
                for(int j=threadIdx.x;j<T;j+=blockDim.x)b[j]=q+j<n?x[q+j]:REAL_C(0.0);
                __syncthreads();dx_blocked_rhs_update(a,b,c);
            }
            for(int i=threadIdx.x;i<T;i+=blockDim.x)a[i]=c[i];
            __syncthreads();dx_blocked_backward(d,a);
            for(int i=threadIdx.x;i<T&&k+i<n;i+=blockDim.x)x[k+i]=a[i];
            __syncthreads();
        }
        for(int i=threadIdx.x;i<n;i+=blockDim.x)
            if(!isfinite(x[i])||!isfinite(y[i]))atomicExch(control,3);
        __syncthreads();
    }
    // Joint transaction: no factor published until its direction is finite.
    if(!control[0]){
        real_t* output=factor+size_t(w)*matrix_stride;
        for(int z=threadIdx.x;z<n*n;z+=blockDim.x){int i=z/n,j=z%n;
            if(i>=j)output[i*ld+j]=W[i*ld+j];}
        for(int i=threadIdx.x;i<n;i+=blockDim.x){
            direction[size_t(w)*BF_MAX_NV+i]=x[i];forward[size_t(w)*BF_MAX_NV+i]=y[i];}
    }
    __syncthreads();
    if(threadIdx.x==0){
        if(control[0])status[w]=control[0];
        int* out=diagnostic+8*w;
        out[0]=control[1]?2:1;out[1]=control[4];out[2]=control[2];out[3]=control[5];
        out[4]=control[1];out[5]=control[0];out[6]=n;out[7]=T;
    }
}
} // namespace libero_blocked
