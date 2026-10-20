// Independent J^T D J residual and redzones around the production shared loan.
#include "newton_atomic.cuh"
#include "newton_jacobian_cache.cuh"
#include "substep_shape.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
using namespace physics_newton;
constexpr int NV=39, ROWS=366, CELLS=NV*(NV+1)/2, GUARD=512;
constexpr int LOAN=flash_libero::SubstepShape::newton_mathdx_cells;
constexpr real_t CANARY=real_t(17.25);
__global__ void gram_case(const real_t *J,const real_t *D,const int *state,
                         unsigned char *cache,size_t pitch,real_t *output,int *guards,int method,int sparse) {
  extern __shared__ __align__(16) unsigned char memory[];
  real_t *arena=reinterpret_cast<real_t*>(memory),*loan=arena+GUARD;
  const int w=blockIdx.x,tid=threadIdx.x,rows=w?83:ROWS;
  for(int z=tid;z<LOAN+2*GUARD;z+=blockDim.x)arena[z]=CANARY;
  __syncthreads();
  J+=w*ROWS*NV;D+=w*ROWS;state+=w*ROWS;output+=w*CELLS;
  if(method==0) {
    jacobian_cache::View view{cache+w*pitch,ROWS,NV};
    jacobian_cache::prepare(view,J,D,rows,1,loan);
    __syncthreads();
    if(tid==0 && sparse>=2) {
      const int products=sparse==2 ? rows : (rows+1)/2;
      const bool expected=products<=NewtonJacobianLayout::product_slots;
      if(bool(*view.integers(NewtonJacobianLayout::products_valid))!=expected)guards[w]|=2;
    }
    jacobian_cache::gram(view,D,state,rows,output,loan);
  } else {
    atomic_reduction::gram_block(J,D,state,rows,NV,output,loan,LOAN);
  }
  __syncthreads();
  bool damaged=false;
  for(int z=tid;z<GUARD;z+=blockDim.x)
    damaged|=arena[z]!=CANARY || arena[GUARD+LOAN+z]!=CANARY;
  if(__syncthreads_or(damaged)&&tid==0)guards[w]=1;
}
static bool run(int sparse) {
  std::vector<real_t> J(2*ROWS*NV),D(2*ROWS);std::vector<int>state(2*ROWS);
  for(int w=0;w<2;++w)for(int r=0;r<ROWS;++r) {
    D[w*ROWS+r]=real_t(.5+.25*(r%3));state[w*ROWS+r]=r%3?1:2;
    for(int j=0;j<NV;++j) {
      real_t value=real_t(((r*3+j+w)%11-5)*.125);
      if(sparse==1 && (j+r)%13>1)value=0;
      if(sparse>=2 && (j>=4 || (sparse==3 && r%2)))value=0;
      J[(w*ROWS+r)*NV+j]=value;
    }
  }
  real_t *dj=nullptr,*dd=nullptr,*out=nullptr;int *ds=nullptr,*guard=nullptr;unsigned char *cache=nullptr;
  const size_t pitch=NewtonJacobianLayout::bytes(ROWS,NV);
  if(cudaMalloc(&dj,J.size()*sizeof(real_t))||cudaMalloc(&dd,D.size()*sizeof(real_t))||
     cudaMalloc(&out,2*CELLS*sizeof(real_t))||cudaMalloc(&ds,state.size()*sizeof(int))||
     cudaMalloc(&guard,2*sizeof(int))||cudaMalloc(&cache,2*pitch))return false;
  cudaMemcpy(dj,J.data(),J.size()*sizeof(real_t),cudaMemcpyHostToDevice);
  cudaMemcpy(dd,D.data(),D.size()*sizeof(real_t),cudaMemcpyHostToDevice);
  cudaMemcpy(ds,state.data(),state.size()*sizeof(int),cudaMemcpyHostToDevice);
  bool passed=true;double worst=0.;
  for(int method=0;method<2;++method) {
    cudaMemset(cache,0,2*pitch);cudaMemset(guard,0,2*sizeof(int));
    gram_case<<<2,64,(LOAN+2*GUARD)*sizeof(real_t)>>>(dj,dd,ds,cache,pitch,out,guard,method,sparse);
    if(cudaDeviceSynchronize()!=cudaSuccess){passed=false;break;}
    std::vector<real_t> actual(2*CELLS);int flags[2];
    cudaMemcpy(actual.data(),out,actual.size()*sizeof(real_t),cudaMemcpyDeviceToHost);
    cudaMemcpy(flags,guard,sizeof(flags),cudaMemcpyDeviceToHost);
    passed&=!flags[0]&&!flags[1];
    for(int w=0;w<2;++w)for(int i=0;i<NV;++i)for(int j=0;j<=i;++j) {
      long double ref=0.;
      for(int r=0;r<(w?83:ROWS);++r)if(state[w*ROWS+r]==1)
        ref+=static_cast<long double>(J[(w*ROWS+r)*NV+i])*D[w*ROWS+r]*J[(w*ROWS+r)*NV+j];
      const double value=actual[w*CELLS+i*(i+1)/2+j];
      const double error=double(fabsl(value-ref)/(1+fabsl(ref)));
      worst=std::max(worst,error);passed&=std::isfinite(value)&&error<=64*REAL_EPSILON;
    }
    std::printf("bits=%d sparse=%d method=%d guards=%d,%d worst_scaled_error=%g\n",
                FLASH_REAL_BITS,sparse,method,flags[0],flags[1],worst);
  }
  cudaFree(dj);cudaFree(dd);cudaFree(out);cudaFree(ds);cudaFree(guard);cudaFree(cache);
  return passed;
}
int main(){bool passed=true;for(int sparse=0;sparse<4;++sparse)passed=run(sparse)&&passed;return passed?0:2;}
