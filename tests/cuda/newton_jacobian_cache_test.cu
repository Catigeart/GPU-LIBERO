// Actual independently compiled reductions, fresh/hit/refreshed generations,
// changing force/state, exceptional masked rows, and bounded-storage fallback.
#include "newton_atomic.cuh"
#include "newton_transpose.cuh"
#include "newton_jacobian_cache.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
using namespace physics_newton;
extern "C" int flash_libero_newton_configure_reductions(int, int);
static void check(cudaError_t e) { if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e)); }
static double max_error = 0;
static void near(double actual, long double ref) {
  if(std::isnan(ref)) { if(!std::isnan(actual))throw std::runtime_error("lost live NaN"); return; }
  if(std::isinf(ref)) { if(actual!=double(ref))throw std::runtime_error("lost live infinity"); return; }
  const double e=double(fabsl(actual-ref)/(1+fabsl(ref)));
  max_error=std::max(max_error,e);
  if(!std::isfinite(actual)||e>1e-11)throw std::runtime_error("long-double product mismatch");
}
__global__ void products(double *J,double *D,double *force,int *state,int nv,int capacity,
    int phase,double *out,double *scratch,unsigned char *cache,size_t pitch,int *verdict) {
  extern __shared__ double raw[];
  const int tid=threadIdx.x,w=blockIdx.x,rows=w&&capacity?capacity/2+1:capacity;
  const int cells=nv*(nv+1)/2;
  J+=size_t(w)*capacity*nv;D+=w*capacity;force+=w*capacity;state+=w*capacity;
  out+=w*(8*nv+cells);scratch+=w*(cells+2)+1;
  jacobian_cache::View view{cache+w*pitch+128,capacity,nv};
  if(phase==2) {
    for(int z=tid;z<rows*nv;z+=blockDim.x)J[z]*=1.01;
    for(int r=tid;r<rows;r+=blockDim.x)D[r]*=1.03;
  }
  __syncthreads();
  const bool fresh=!view.prepared(rows,phase==2?12:11);
  __syncthreads(); // all lanes capture before a new generation is published
  if(fresh) { jacobian_cache::prepare(view,J,D,rows,phase==2?12:11,raw);__syncthreads(); }
  if(tid==0)verdict[w*4+phase]=fresh;
  if(tid==0)verdict[w*4+3]=view.valid();
  // These are fresh iteration outputs, intentionally changed AFTER building J.
  for(int r=tid;r<rows;r+=blockDim.x) {
    force[r]=-force[r];state[r]=(r+phase)%4?1:2;
  }
  __syncthreads();
  double *partial=raw+128;
  double *loan=partial+int(blockDim.x)/8*nv+1;
  const int loan_cells=atomic_reduction::gram_block_cells(nv);
  if(tid==0){loan[-1]=-34567.25;loan[loan_cells]=-34567.25;}
  __syncthreads();
  for(int method=0;method<8;++method) {
    if(method==0)atomic_reduction::transpose(J,force,rows,nv,raw,scratch);
    else if(method<=4)transpose_reduction::dense(J,force,rows,nv,raw,partial,
        static_cast<transpose_reduction::Method>(method-1));
    else if(view.valid())jacobian_cache::transpose(view,J,force,rows,raw,method==5?1:method==6?4:8);
    else transpose_reduction::dense(J,force,rows,nv,raw,partial,transpose_reduction::Method::Direct);
    __syncthreads();
    for(int j=tid;j<nv;j+=blockDim.x)out[method*nv+j]=raw[j];
    __syncthreads();
  }
  if(view.valid())jacobian_cache::gram(view,D,state,rows,scratch,loan);
  else atomic_reduction::gram_block(J,D,state,rows,nv,scratch,loan,loan_cells);
  for(int z=tid;z<cells;z+=blockDim.x)out[8*nv+z]=scratch[z];
  __syncthreads();
  if(tid==0&&(loan[-1]!=-34567.25||loan[loan_cells]!=-34567.25))verdict[w*4+3]=-2;
}
static void run(int nv,int capacity,int threads,int mode) {
  const int cells=nv*(nv+1)/2;
  const size_t n=std::max(size_t(1),size_t(2)*capacity*nv);
  std::vector<double> J(n),D(std::max(1,2*capacity)),f(D.size());std::vector<int>s(D.size());
  for(int w=0;w<2;++w)for(int r=0;r<capacity;++r) {
    f[w*capacity+r]=r%3?sin(.07*r+.2*w):0.0;
    D[w*capacity+r]=r%7?.2+.003*(r%131):0.0;
    s[w*capacity+r]=2;
    for(int j=0;j<nv;++j) {
      double v=sin(.03*r+.19*j+.2*w);
      if(mode==1||mode==3||mode==4)if((j+r)%11>1)v=0.0;
      if(mode==5&&j>1)v=0.0;
      if(mode==2&&r%3==0)v=nan(""); // force-masked, may become live in Gram
      if(mode==3&&r==1&&j==nv/2)v=nan("");
      J[(size_t(w)*capacity+r)*nv+j]=v;
    }
    if(mode==4&&r==1)f[w*capacity+r]=INFINITY;
  }
  double *dj,*dd,*df,*out,*scratch;int *ds,*verdict;unsigned char*cache;
  const size_t bytes=NewtonJacobianLayout::bytes(capacity,nv),pitch=bytes+256;
  std::vector<unsigned char>guards(2*pitch,0xa7);
  std::fill(guards.begin()+128,guards.begin()+128+bytes,0);
  std::fill(guards.begin()+pitch+128,guards.begin()+pitch+128+bytes,0);
  std::vector<double>scratch_guards(2*(cells+2),-98765.25),actual(2*(8*nv+cells));
  check(cudaMalloc(&dj,n*8));check(cudaMalloc(&dd,D.size()*8));check(cudaMalloc(&df,f.size()*8));check(cudaMalloc(&ds,s.size()*4));
  check(cudaMalloc(&out,actual.size()*8));check(cudaMalloc(&scratch,scratch_guards.size()*8));check(cudaMalloc(&cache,guards.size()));check(cudaMalloc(&verdict,8*4));
  check(cudaMemcpy(dj,J.data(),n*8,cudaMemcpyHostToDevice));check(cudaMemcpy(dd,D.data(),D.size()*8,cudaMemcpyHostToDevice));check(cudaMemcpy(df,f.data(),f.size()*8,cudaMemcpyHostToDevice));check(cudaMemcpy(ds,s.data(),s.size()*4,cudaMemcpyHostToDevice));
  check(cudaMemcpy(cache,guards.data(),guards.size(),cudaMemcpyHostToDevice));check(cudaMemcpy(scratch,scratch_guards.data(),scratch_guards.size()*8,cudaMemcpyHostToDevice));
  for(int phase=0;phase<3;++phase) {
    const int sh=128+threads/8*nv+2+atomic_reduction::gram_block_cells(nv);
    products<<<2,threads,sh*8>>>(dj,dd,df,ds,nv,capacity,phase,out,scratch,cache,pitch,verdict);
    check(cudaGetLastError());check(cudaDeviceSynchronize());
    check(cudaMemcpy(actual.data(),out,actual.size()*8,cudaMemcpyDeviceToHost));
    for(int w=0;w<2;++w) {
      const int rows=w&&capacity?capacity/2+1:capacity;
      for(int r=0;r<rows;++r) {
        if(phase==2) {D[w*capacity+r]*=1.03;for(int j=0;j<nv;++j)J[(size_t(w)*capacity+r)*nv+j]*=1.01;}
        f[w*capacity+r]=-f[w*capacity+r];s[w*capacity+r]=(r+phase)%4?1:2;
      }
      for(int j=0;j<nv;++j) {
        long double ref=0;
        for(int r=0;r<rows;++r)if(f[w*capacity+r]!=0)ref+=(long double)J[(size_t(w)*capacity+r)*nv+j]*f[w*capacity+r];
        for(int method=0;method<8;++method)near(actual[w*(8*nv+cells)+method*nv+j],ref);
      }
      for(int i=0;i<nv;++i)for(int j=0;j<=i;++j) {
        long double ref=0;
        for(int r=0;r<rows;++r)if(s[w*capacity+r]==1&&D[w*capacity+r]!=0)
          ref+=(long double)J[(size_t(w)*capacity+r)*nv+i]*D[w*capacity+r]*J[(size_t(w)*capacity+r)*nv+j];
        near(actual[w*(8*nv+cells)+8*nv+i*(i+1)/2+j],ref);
      }
    }
  }
  int flags[8];check(cudaMemcpy(flags,verdict,sizeof(flags),cudaMemcpyDeviceToHost));
  check(cudaMemcpy(guards.data(),cache,guards.size(),cudaMemcpyDeviceToHost));check(cudaMemcpy(scratch_guards.data(),scratch,scratch_guards.size()*8,cudaMemcpyDeviceToHost));
  for(int w=0;w<2;++w) {
    if(flags[w*4]!=1||flags[w*4+1]!=0||flags[w*4+2]!=1||flags[w*4+3]==-2)throw std::runtime_error("generation or shared guard failed");
    if(nv==57&&capacity==2301&&w==0&&flags[w*4+3]!=0)throw std::runtime_error("entry-budget fallback was not exercised");
    if(mode==5&&w==0&&guards[w*pitch+128+NewtonJacobianLayout::products_valid]!=0)throw std::runtime_error("product-slot fallback was not exercised");
    if(nv==65&&flags[w*4+3]!=0)throw std::runtime_error("NV fallback was not exercised");
    for(size_t i=0;i<128;++i)if(guards[w*pitch+i]!=0xa7||guards[w*pitch+128+bytes+i]!=0xa7)throw std::runtime_error("cache world guard failed");
    if(scratch_guards[w*(cells+2)]!=-98765.25||scratch_guards[w*(cells+2)+cells+1]!=-98765.25)throw std::runtime_error("scratch world guard failed");
  }
  for(void*p:{(void*)dj,(void*)dd,(void*)df,(void*)ds,(void*)out,(void*)scratch,(void*)cache,(void*)verdict})check(cudaFree(p));
}
int main() {try {
  check(cudaSetDevice(0));
  check(static_cast<cudaError_t>(flash_libero_newton_configure_reductions(1, 7)));
  for(int nv:{21,22,24,27,30,33,36,37,39,43,45,51,57,65}) {
    for(int rows:{0,1,17,165,256,513})for(int threads:{64,128})for(int mode=0;mode<5;++mode)run(nv,rows,threads,mode);
    printf("NV=%d: fresh/hit/refresh, changed force/state, dense/sparse/nonfinite, 64/128 threads PASS\n",nv);fflush(stdout);
  }
  run(57,2301,64,0);
  run(39,257,64,5);
  check(static_cast<cudaError_t>(flash_libero_newton_configure_reductions(1, 0)));
  run(39,165,64,1); // auto builds products but deliberately omits CSC
  printf("PASS: 13 catalogue NV + NV65 and entry-budget fallback; max scaled error %.3e\n",max_error);
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
