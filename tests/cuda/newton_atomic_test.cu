// Actual separately compiled device functions versus independent long-double
// products: all 13 shapes, two worlds, row tails, NaN masking and scratch reuse.
#include "newton_atomic.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

static void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
static double max_scaled_error = 0;
static void near(double actual, long double reference) {
  if (std::isnan(reference)) {
    if (!std::isnan(actual)) throw std::runtime_error("Live NaN was incorrectly masked");
    return;
  }
  const long double error = fabsl(actual-reference)/(1+fabsl(reference));
  max_scaled_error=std::max(max_scaled_error,double(error));
  if (!std::isfinite(actual) || error > 1e-11L)
    throw std::runtime_error("Atomic product differs from long-double reference");
}

__global__ void products(const double *jt, const double *jg, const double *force,
    const double *D, const int *state, double *out, double *scratch, int nv, int capacity, int mode) {
  extern __shared__ double block[];
  const int world=int(blockIdx.x), rows=world && capacity ? capacity/2+1 : capacity;
  const int cells=nv*(nv+1)/2;
  const double *Jt=jt+size_t(world)*capacity*nv, *Jg=jg+size_t(world)*capacity*nv;
  double *values=out+size_t(world)*(nv+nv*nv), *work=scratch+size_t(world)*(cells+2)+1;
  physics_newton::atomic_reduction::transpose(Jt,force+world*capacity,rows,nv,values,work);
  __syncthreads(); // caller closes vector-output and scratch-read lifetimes
  const int shared_cells=physics_newton::atomic_reduction::gram_block_cells(nv);
  if(threadIdx.x==0){block[0]=-34567.25;block[shared_cells+1]=-34567.25;}
  __syncthreads();
  if(mode==0)
    physics_newton::atomic_reduction::gram(Jg,D+world*capacity,state+world*capacity,rows,nv,work);
  else
    physics_newton::atomic_reduction::gram_block(Jg,D+world*capacity,state+world*capacity,rows,nv,work,
                                                block+1,mode==3?0:shared_cells);
  for (int z=threadIdx.x;z<nv*nv;z+=blockDim.x) {
    const int i=z/nv,j=z%nv;
    if (j>i) continue;
    const double mass=i==j ? 2.0+nv : .01*(1+i+j)/(1+nv);
    const double value=work[i*(i+1)/2+j]+mass;
    values[nv+z]=value;values[nv+j*nv+i]=value;
  }
  __syncthreads();
  if(threadIdx.x==0&&(block[0]!=-34567.25||block[shared_cells+1]!=-34567.25))values[0]=nan("");
}

static void run(int nv,int capacity,int threads,int mode) {
  const int cells=nv*(nv+1)/2;
  const size_t jsize=std::max(size_t(1),size_t(2)*capacity*nv);
  std::vector<double> jt(jsize),jg(jsize),f(std::max(1,2*capacity)),d(f.size());
  std::vector<int> state(f.size());
  for (int w=0;w<2;++w) {
    const int rows=w && capacity ? capacity/2+1 : capacity;
    for (int r=0;r<capacity;++r) {
      f[w*capacity+r]=r%3 && r<rows ? sin(.07*r+.2*w) : 0.0;
      state[w*capacity+r]=r%4 ? 1 : 2;
      d[w*capacity+r]=r%7 ? .2+.003*(r%131) : 0;
      for (int j=0;j<nv;++j) {
        const size_t z=(size_t(w)*capacity+r)*nv+j;
        const double value=mode==1&&(j%11>1||r%13==0) ? 0.0 :
            sin(.03*r+.19*j+.2*w)*(r%2 ? 1.0 : -1.0);
        jt[z]=f[w*capacity+r]!=0 ? value : nan("");
        jg[z]=r<rows && state[w*capacity+r]==1 && d[w*capacity+r]!=0 ? value : nan("");
        if(mode==2&&r==1&&j==nv/2)jg[z]=nan("");
      }
    }
  }
  double *dj[2],*df,*dd,*output,*scratch;int *ds;
  check(cudaMalloc(&dj[0],jsize*8));check(cudaMalloc(&dj[1],jsize*8));
  check(cudaMalloc(&df,f.size()*8));check(cudaMalloc(&dd,d.size()*8));
  check(cudaMalloc(&ds,state.size()*sizeof(int)));
  check(cudaMalloc(&output,size_t(2)*(nv+nv*nv)*8));
  check(cudaMalloc(&scratch,size_t(2)*(cells+2)*8));
  std::vector<double> guards(2*(cells+2),-98765.25),actual(2*(nv+nv*nv));
  check(cudaMemcpy(dj[0],jt.data(),jsize*8,cudaMemcpyHostToDevice));
  check(cudaMemcpy(dj[1],jg.data(),jsize*8,cudaMemcpyHostToDevice));
  check(cudaMemcpy(df,f.data(),f.size()*8,cudaMemcpyHostToDevice));
  check(cudaMemcpy(dd,d.data(),d.size()*8,cudaMemcpyHostToDevice));
  check(cudaMemcpy(ds,state.data(),state.size()*sizeof(int),cudaMemcpyHostToDevice));
  check(cudaMemcpy(scratch,guards.data(),guards.size()*8,cudaMemcpyHostToDevice));
  const int shared_cells=physics_newton::atomic_reduction::gram_block_cells(nv);
  products<<<2,threads,(shared_cells+2)*sizeof(double)>>>(dj[0],dj[1],df,dd,ds,output,scratch,nv,capacity,mode);
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  check(cudaMemcpy(actual.data(),output,actual.size()*8,cudaMemcpyDeviceToHost));
  check(cudaMemcpy(guards.data(),scratch,guards.size()*8,cudaMemcpyDeviceToHost));
  for (int w=0;w<2;++w) {
    const int rows=w && capacity ? capacity/2+1 : capacity;
    if (guards[w*(cells+2)]!=-98765.25 || guards[w*(cells+2)+cells+1]!=-98765.25)
      throw std::runtime_error("World scratch bounds overwritten");
    const double *a=actual.data()+size_t(w)*(nv+nv*nv);
    for (int j=0;j<nv;++j) {
      long double expected=0;
      for (int r=0;r<rows;++r)
        if (f[w*capacity+r]!=0) expected+=(long double)jt[(size_t(w)*capacity+r)*nv+j]*f[w*capacity+r];
      near(a[j],expected);
    }
    for (int i=0;i<nv;++i) for (int j=0;j<=i;++j) {
      long double expected=0;
      for (int r=0;r<rows;++r)
        if (state[w*capacity+r]==1 && d[w*capacity+r]!=0)
          expected+=(long double)jg[(size_t(w)*capacity+r)*nv+i]*d[w*capacity+r]*jg[(size_t(w)*capacity+r)*nv+j];
      expected+=i==j ? 2.0+nv : .01*(1+i+j)/(1+nv);
      near(a[nv+i*nv+j],expected);
      if (a[nv+i*nv+j]!=a[nv+j*nv+i] &&
          !(std::isnan(a[nv+i*nv+j])&&std::isnan(a[nv+j*nv+i])))
        throw std::runtime_error("Hessian symmetry lost");
    }
  }
  for (double *p : {dj[0],dj[1],df,dd,output,scratch}) check(cudaFree(p));
  check(cudaFree(ds));
}

int main() {
  try {
    for (int nv : {21,22,24,27,30,33,36,37,39,43,45,51,57,65}) {
      for (int rows : {0,1,17,165,256,513}) for (int threads : {64,128})
        for(int mode=0;mode<4;++mode)run(nv,rows,threads,mode);
      printf("NV=%d: dense/sparse/live-NaN/budget fallback, two worlds, 64/128 threads passed\n",nv);
    }
    printf("PASS: 13 catalogue shapes + dynamic fallback; max scaled error %.3e\n",max_scaled_error);
  } catch (const std::exception &e) {fprintf(stderr,"%s\n",e.what());return 1;}
}
