// Independent L_new*L_new^T = L_old*L_old^T +/- v*v^T residual checks.
#include "newton_rank_update.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

static void check(cudaError_t status) {
  if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template<int N, bool Shared> __global__ void update(real_t *matrix,real_t *vector,int *ranks,int plus) {
  const int w=blockIdx.x;
  __shared__ real_t shared_matrix[Shared ? N*N : 1], shared_vector[Shared ? N : 1];
  real_t *factor=matrix+w*N*N, *values=vector+w*N;
  if constexpr (Shared) {
    for(int i=threadIdx.x;i<N*N;i+=blockDim.x) shared_matrix[i]=factor[i];
    for(int i=threadIdx.x;i<N;i+=blockDim.x) shared_vector[i]=values[i];
    __syncthreads();
    factor=shared_matrix;values=shared_vector;
  }
  const int rank=physics_newton::rank_update::collective<N>(factor,values,plus);
  if(threadIdx.x==0) ranks[w]=rank;
  if constexpr (Shared) {
    for(int i=threadIdx.x;i<N*N;i+=blockDim.x) matrix[w*N*N+i]=factor[i];
    for(int i=threadIdx.x;i<N;i+=blockDim.x) vector[w*N+i]=values[i];
  }
}
template<int N, bool Shared> static void run(int plus,int pattern) {
  std::vector<real_t> matrix(2*N*N),vector(2*N);
  for(int w=0;w<2;++w) for(int i=0;i<N;++i) {
    for(int j=0;j<N;++j)
      matrix[w*N*N+i*N+j]=i<j ? REAL_C(-7.25) :
          (i==j ? real_t(1.+.01*i+.1*w) : real_t(.02*std::sin(.3*i+.17*j)/N));
    vector[w*N+i]=pattern==0 ? real_t(.025*std::sin(.3*i+.1*w)) :
        (pattern==1 ? REAL_C(0.) : (i==N-1 ? REAL_C(.1) : REAL_C(0.)));
  }
  const auto original=matrix,initial=vector;
  real_t *dm=nullptr,*dv=nullptr;int *dr=nullptr;
  check(cudaMalloc(&dm,matrix.size()*sizeof(real_t)));check(cudaMalloc(&dv,vector.size()*sizeof(real_t)));
  check(cudaMalloc(&dr,2*sizeof(int)));
  check(cudaMemcpy(dm,matrix.data(),matrix.size()*sizeof(real_t),cudaMemcpyHostToDevice));
  check(cudaMemcpy(dv,vector.data(),vector.size()*sizeof(real_t),cudaMemcpyHostToDevice));
  update<N,Shared><<<2,64>>>(dm,dv,dr,plus);check(cudaGetLastError());check(cudaDeviceSynchronize());
  check(cudaMemcpy(matrix.data(),dm,matrix.size()*sizeof(real_t),cudaMemcpyDeviceToHost));
  int ranks[2];check(cudaMemcpy(ranks,dr,sizeof(ranks),cudaMemcpyDeviceToHost));
  check(cudaFree(dr));check(cudaFree(dv));check(cudaFree(dm));
  const double gamma=N*double(REAL_EPSILON)/(1.-N*double(REAL_EPSILON));
  double worst=0;
  for(int w=0;w<2;++w) {
    if(ranks[w]!=N) throw std::runtime_error("unexpected rank loss");
    for(int i=0;i<N;++i) for(int j=0;j<N;++j) {
      if(j>i && matrix[w*N*N+i*N+j]!=original[w*N*N+i*N+j])
        throw std::runtime_error("upper owner modified");
      long double expected=(plus?1.L:-1.L)*initial[w*N+i]*initial[w*N+j],actual=0;
      for(int k=0;k<=i && k<=j;++k) {
        expected+=(long double)original[w*N*N+i*N+k]*original[w*N*N+j*N+k];
        actual+=(long double)matrix[w*N*N+i*N+k]*matrix[w*N*N+j*N+k];
      }
      const double error=std::abs(double(actual-expected));worst=std::max(worst,error);
      if(!std::isfinite(error) || error>16*gamma*(1+std::abs(double(expected))))
        throw std::runtime_error("rank-update residual");
    }
  }
  std::printf("bits=%d N=%d shared=%d plus=%d pattern=%d residual=%g\n",FLASH_REAL_BITS,N,int(Shared),plus,pattern,worst);
}
template<int N> static void cases() {
  for(int plus=0;plus<2;++plus) for(int p=0;p<3;++p) {
    run<N,false>(plus,p);run<N,true>(plus,p);
  }
}

template<int N, bool Shared, bool Changes>
__global__ void batch_update(real_t *matrix,const real_t *vectors,const real_t *scales,
                             const int *rows,const int *state,int count,real_t *last,int *outcome) {
  const int w=blockIdx.x;
  __shared__ real_t storage[Shared ? N*N : 1];
  real_t *factor=matrix+w*N*N;
  if constexpr (Shared) {
    for(int i=threadIdx.x;i<N*N;i+=blockDim.x) storage[i]=factor[i];
    __syncthreads();factor=storage;
  }
  if(threadIdx.x<32) {
    int completed=count;
    if constexpr (Changes)
      completed=physics_newton::rank_update::ordered_rows<N>(
          factor,vectors+w*3*N,scales,rows,state,count,last+w*N);
    else
      physics_newton::rank_update::ordered_plus<N>(factor,vectors+w*3*N,count,last+w*N);
    if(threadIdx.x==0) outcome[w]=completed;
  }
  __syncthreads();
  if constexpr (Shared)
    for(int i=threadIdx.x;i<N*N;i+=blockDim.x) matrix[w*N*N+i]=factor[i];
}

// Patterns: ordinary, first-row rank loss, empty, zero final vector.
template<int N, bool Shared, bool Changes> static void batch_case(int pattern) {
  std::vector<real_t> matrix(2*N*N),vectors(2*3*N),last(2*N,REAL_C(123.));
  const real_t scales[3]={REAL_C(.5),REAL_C(1.25),REAL_C(.75)};
  const int rows[3]={0,1,2},state[3]={pattern==1 ? 0 : 1,0,1};
  const int count=pattern==2 ? 0 : 3;
  for(int w=0;w<2;++w) {
    for(int i=0;i<N;++i) for(int j=0;j<N;++j)
      matrix[w*N*N+i*N+j]=i<j ? REAL_C(-7.25) : (pattern==1 ? real_t(i==j) :
          (i==j ? real_t(1.+.01*i+.1*w) : real_t(.02*std::sin(.3*i+.17*j)/N)));
    for(int r=0;r<3;++r) for(int i=0;i<N;++i) {
      real_t v=real_t(.025*std::sin(.3*i+.2*r+.1*w));
      if(pattern==1) v=r ? std::numeric_limits<real_t>::quiet_NaN() : (i ? REAL_C(0.) : REAL_C(4.));
      if(pattern==2) v=std::numeric_limits<real_t>::quiet_NaN();
      if(pattern==3 && r==2) v=REAL_C(0.);
      vectors[w*3*N+r*N+i]=v;
    }
  }
  const auto original=matrix;
  real_t *dm=nullptr,*dv=nullptr,*ds=nullptr,*dl=nullptr;int *dr=nullptr,*dt=nullptr,*done=nullptr;
  check(cudaMalloc(&dm,matrix.size()*sizeof(real_t)));check(cudaMalloc(&dv,vectors.size()*sizeof(real_t)));
  check(cudaMalloc(&ds,sizeof(scales)));check(cudaMalloc(&dl,last.size()*sizeof(real_t)));
  check(cudaMalloc(&dr,sizeof(rows)));check(cudaMalloc(&dt,sizeof(state)));check(cudaMalloc(&done,2*sizeof(int)));
  check(cudaMemcpy(dm,matrix.data(),matrix.size()*sizeof(real_t),cudaMemcpyHostToDevice));
  check(cudaMemcpy(dv,vectors.data(),vectors.size()*sizeof(real_t),cudaMemcpyHostToDevice));
  check(cudaMemcpy(ds,scales,sizeof(scales),cudaMemcpyHostToDevice));
  check(cudaMemcpy(dl,last.data(),last.size()*sizeof(real_t),cudaMemcpyHostToDevice));
  check(cudaMemcpy(dr,rows,sizeof(rows),cudaMemcpyHostToDevice));check(cudaMemcpy(dt,state,sizeof(state),cudaMemcpyHostToDevice));
  batch_update<N,Shared,Changes><<<2,64>>>(dm,dv,ds,dr,dt,count,dl,done);
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  check(cudaMemcpy(matrix.data(),dm,matrix.size()*sizeof(real_t),cudaMemcpyDeviceToHost));
  check(cudaMemcpy(last.data(),dl,last.size()*sizeof(real_t),cudaMemcpyDeviceToHost));
  int completed[2];check(cudaMemcpy(completed,done,sizeof(completed),cudaMemcpyDeviceToHost));
  check(cudaFree(done));check(cudaFree(dt));check(cudaFree(dr));check(cudaFree(dl));
  check(cudaFree(ds));check(cudaFree(dv));check(cudaFree(dm));
  const double gamma=N*double(REAL_EPSILON)/(1.-N*double(REAL_EPSILON));
  double worst=0;
  for(int w=0;w<2;++w) {
    if(completed[w]!=(pattern==1 ? -1 : count)) throw std::runtime_error("batch completion/rebuild boundary");
    for(int i=0;i<N;++i) {
      if(!std::isfinite(last[w*N+i])) throw std::runtime_error("nonfinite final vector");
      if(pattern==2 && last[w*N+i]!=REAL_C(123.)) throw std::runtime_error("empty batch vector hold");
      if(pattern==3 && last[w*N+i]!=REAL_C(0.)) throw std::runtime_error("last zero vector not committed");
      if(pattern==1 && last[w*N+i]!=(i ? REAL_C(0.) : REAL_C(2.))) throw std::runtime_error("lost-rank vector");
      for(int j=0;j<N;++j) {
        if(j>i && matrix[w*N*N+i*N+j]!=original[w*N*N+i*N+j]) throw std::runtime_error("batch upper owner");
        if(pattern==2 && matrix[w*N*N+i*N+j]!=original[w*N*N+i*N+j]) throw std::runtime_error("empty batch factor hold");
        long double expected=0,actual=0;
        if(pattern==1) expected=i==j ? (i ? 1.L : (long double)REAL_C(1e-15)) : 0.L;
        else {
          for(int k=0;k<=i && k<=j;++k)
            expected+=(long double)original[w*N*N+i*N+k]*original[w*N*N+j*N+k];
          for(int r=0;r<count;++r) {
            const real_t scale=Changes ? scales[r] : REAL_C(1.);
            const real_t vi=vectors[w*3*N+r*N+i]*scale,vj=vectors[w*3*N+r*N+j]*scale;
            expected+=(Changes && state[r]!=1 ? -1.L : 1.L)*(long double)vi*vj;
          }
        }
        for(int k=0;k<=i && k<=j;++k) actual+=(long double)matrix[w*N*N+i*N+k]*matrix[w*N*N+j*N+k];
        const double error=std::abs(double(actual-expected));worst=std::max(worst,error);
        if(!std::isfinite(error) || error>16*gamma*(1+std::abs(double(expected)))) throw std::runtime_error("batch residual");
      }
    }
  }
  std::printf("batch bits=%d N=%d shared=%d changes=%d pattern=%d residual=%g\n",FLASH_REAL_BITS,N,int(Shared),int(Changes),pattern,worst);
}
template<int N,bool Shared> static void batch_space() {
  for(int p : {0,2,3}) {batch_case<N,Shared,false>(p);batch_case<N,Shared,true>(p);}
  batch_case<N,Shared,true>(1);
}
template<int N> static void batch_cases() {batch_space<N,false>();batch_space<N,true>();}
int main() {
  cases<21>();cases<22>();cases<30>();cases<33>();cases<39>();cases<57>();cases<64>();cases<65>();
  batch_cases<21>();batch_cases<22>();batch_cases<33>();batch_cases<39>();batch_cases<57>();batch_cases<64>();
  return 0;
}
