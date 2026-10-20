#include "precision.cuh"
// Diagnostic helper candidate ONLY; not integrated with any Newton runtime.
// One uniform 64-thread block/world. Every original k column completes before
// the next. Each i owns its original data cell and vector element. NV<=57 in
// the actual corpus; no row/contact/rank-update order is changed by this helper.
#include <cuda_runtime.h>

extern "C" __global__ void n_cooperative_rank_update_r01(
    real_t* data, real_t* vec, const int* plus, const int* active,
    const int* status, int* rank_output, int nv, int nworld) {
  int w=blockIdx.x;
  int lane=threadIdx.x;
  if (w>=nworld || active[w]!=1 || status[w]!=0) return;
  // Cold launch contract requires nv in [1,57], blockDim.x=64. No fallback
  // or truncation is allowed; dimension/launch validation lives in the future
  // isolated diagnostic loader, never a device silent success.
  __shared__ real_t c;
  __shared__ real_t cinv;
  __shared__ real_t s;
  __shared__ int nonzero;
  __shared__ int rank;
  if (lane==0) rank=nv;
  __syncthreads();
  int hb=w*nv*nv;
  int vb=w*nv;
  for (int k=0;k<nv;++k) {
    if (lane==0) {
      real_t x=vec[vb+k];
      nonzero=(x!=REAL_C(0.0));
      if (nonzero) {
        real_t Lkk=data[hb+k*nv+k];
        real_t tmp=real_mul_rn(Lkk,Lkk);
        real_t xx=real_mul_rn(x,x);
        tmp=plus[w]!=0 ? real_add_rn(tmp,xx) : real_sub_rn(tmp,xx);
        if (tmp<REAL_C(1.0e-15)) {tmp=REAL_C(1.0e-15);--rank;}
        real_t r=real_sqrt_rn(tmp);
        c=real_div_rn(r,Lkk);
        cinv=real_div_rn(REAL_C(1.0),c);
        s=real_div_rn(x,Lkk);
        data[hb+k*nv+k]=r;
      }
    }
    __syncthreads();
    if (nonzero && lane>k && lane<nv) {
      int i=lane;
      real_t current=data[hb+i*nv+k];
      real_t sv=real_mul_rn(s,vec[vb+i]);
      real_t updated=plus[w]!=0 ? real_add_rn(current,sv) : real_sub_rn(current,sv);
      data[hb+i*nv+k]=real_mul_rn(updated,cinv);
      // Original two i-loops can interleave across i: no other element reads
      // this i's updated cell or vec. Within i the data write precedes vec.
      vec[vb+i]=real_sub_rn(real_mul_rn(c,vec[vb+i]),real_mul_rn(s,data[hb+i*nv+k]));
    }
    __syncthreads();
  }
  if (lane==0) rank_output[w]=rank;
}
