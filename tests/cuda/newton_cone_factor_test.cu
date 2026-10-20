// Exercise the maintained private operator; no duplicated factorization math.
#include "newton_solver.cu"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

constexpr int N=flash_libero::SubstepShape::nv, R=29, C=4;
constexpr int tile=flash_libero::NewtonLayout::hessian_tile_rows;
constexpr int pending_cells=std::max(N*(N+1)/2,tile*N);
constexpr int guard=16;

static void check(cudaError_t status) {
  if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template<class T> static T *upload(const std::vector<T>& values) {
  T *data=nullptr;check(cudaMalloc(&data,values.size()*sizeof(T)));
  check(cudaMemcpy(data,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice));return data;
}
__global__ void cone_test(int *count,int *state,int *id,int *dim,real_t *contact,real_t *J,
                          real_t *H,real_t *HC,real_t *pending,int *updates,int *diagnostic,int *status) {
  const int w=blockIdx.x;
  extern __shared__ __align__(16) unsigned char arena[];
  real_t *temporary=reinterpret_cast<real_t*>(arena);
  real_t *local=temporary+flash_libero::NewtonLayout::mathdx_cells;
  real_t *ltj=local+36;
  physics_newton::hessian_cone_batch_dx(count+w,state+w*R,id+w*R,dim+w*C,contact+w*C*36,
      J+w*R*N,H+w*N*N,HC+w*N*N,local,ltj,updates+w,temporary,diagnostic+w*24,
      diagnostic+w*24+4,pending+w*(pending_cells+2*guard)+guard,diagnostic+w*24+12,
      nullptr,nullptr,nullptr,nullptr,nullptr,status+w);
}

static void run(bool cones,bool small_diagonal) {
  std::vector<real_t> H(2*N*N),HC(2*N*N,REAL_C(-41.)),J(2*R*N),contact(2*C*36,REAL_C(-11.));
  std::vector<real_t> pending(2*(pending_cells+2*guard),REAL_C(-33.));
  std::vector<int> count(2,cones?R:0),state(2*R,4),id(2*R),dim(2*C),updates(2,0),diagnostic(2*24,0),status(2,0);
  const int dimensions[C]={3,4,6,6}, groups[7]={0,0,1,2,3,0,1};
  for(int w=0;w<2;++w) {
    for(int i=0;i<N;++i) for(int j=0;j<N;++j)
      H[w*N*N+i*N+j]=j>i ? std::numeric_limits<real_t>::quiet_NaN() :
          (i==j ? (small_diagonal && i==N-1 ? REAL_C(.001) : real_t(1.+.01*i+.1*w)) :
           real_t(.02*std::sin(.3*i+.17*j)/N));
    int row=0;
    for(int c : groups) for(int r=0;r<dimensions[c];++r) id[w*R+row++]=c;
    for(int c=0;c<C;++c) {
      const int d=dimensions[c];dim[w*C+c]=d;
      real_t lower[36]={};
      for(int i=0;i<d;++i) for(int j=0;j<=i;++j)
        lower[i*d+j]=i==j ? real_t(.6+.01*c+.03*w) : real_t(.015*std::sin(i+.2*j));
      for(int i=0;i<d;++i) for(int j=0;j<d;++j) {
        real_t value=0;
        for(int k=0;k<=std::min(i,j);++k) value+=lower[i*d+k]*lower[j*d+k];
        contact[w*C*36+c*36+i*d+j]=value;
      }
    }
    for(int r=0;r<R;++r) for(int i=0;i<N;++i)
      J[w*R*N+r*N+i]=real_t(.025*std::sin(.13*r+.21*i+.1*w));
  }
  const auto original=H;
  real_t *dH=upload(H),*dHC=upload(HC),*dJ=upload(J),*dcontact=upload(contact),*dpending=upload(pending);
  int *dcount=upload(count),*dstate=upload(state),*did=upload(id),*ddim=upload(dim),*dupdates=upload(updates),
      *ddiag=upload(diagnostic),*dstatus=upload(status);
  constexpr int shared_bytes=(flash_libero::NewtonLayout::mathdx_cells+36+6*N)*sizeof(real_t);
  check(cudaFuncSetAttribute(cone_test,cudaFuncAttributeMaxDynamicSharedMemorySize,shared_bytes));
  cone_test<<<2,flash_libero::SubstepShape::threads,shared_bytes>>>(dcount,dstate,did,ddim,dcontact,dJ,dH,dHC,dpending,dupdates,ddiag,dstatus);
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  check(cudaMemcpy(H.data(),dH,H.size()*sizeof(real_t),cudaMemcpyDeviceToHost));
  check(cudaMemcpy(HC.data(),dHC,HC.size()*sizeof(real_t),cudaMemcpyDeviceToHost));
  check(cudaMemcpy(pending.data(),dpending,pending.size()*sizeof(real_t),cudaMemcpyDeviceToHost));
  check(cudaMemcpy(updates.data(),dupdates,updates.size()*sizeof(int),cudaMemcpyDeviceToHost));
  check(cudaMemcpy(status.data(),dstatus,status.size()*sizeof(int),cudaMemcpyDeviceToHost));
  void *pointers[]={dH,dHC,dJ,dcontact,dpending,dcount,dstate,did,ddim,dupdates,ddiag,dstatus};
  for(void *pointer : pointers) check(cudaFree(pointer));
  if(std::memcmp(H.data(),original.data(),H.size()*sizeof(real_t))) throw std::runtime_error("base factor owner changed");
  const double gamma=(N+R+12)*double(REAL_EPSILON)/(1-(N+R+12)*double(REAL_EPSILON));
  double worst=0;
  for(int w=0;w<2;++w) {
    if(status[w] || updates[w]!=(cones?R:0)) throw std::runtime_error("cone status/update count");
    for(int k=0;k<guard;++k)
      if(pending[w*(pending_cells+2*guard)+k]!=REAL_C(-33.) ||
         pending[w*(pending_cells+2*guard)+guard+pending_cells+k]!=REAL_C(-33.))
        throw std::runtime_error("pending workspace redzone");
    for(int i=0;i<N;++i) for(int j=0;j<N;++j) {
      if(j>i && std::memcmp(&HC[w*N*N+i*N+j],&original[w*N*N+i*N+j],sizeof(real_t)))
        throw std::runtime_error("upper owner bits changed");
      long double expected=0,actual=0;
      for(int k=0;k<=std::min(i,j);++k) {
        expected+=(long double)original[w*N*N+i*N+k]*original[w*N*N+j*N+k];
        actual+=(long double)HC[w*N*N+i*N+k]*HC[w*N*N+j*N+k];
      }
      if(cones) for(int r=0;r<R;) {
        const int c=id[w*R+r],d=dim[w*C+c];
        for(int u=0;u<d;++u) for(int v=0;v<d;++v)
          expected+=(long double)J[w*R*N+(r+u)*N+i]*contact[w*C*36+c*36+u*d+v]*J[w*R*N+(r+v)*N+j];
        r+=d;
      }
      const double error=std::abs(double(actual-expected));worst=std::max(worst,error);
      if(!std::isfinite(error) || error>64*gamma*(1+std::abs(double(expected))))
        throw std::runtime_error("independent cone-factor residual");
    }
  }
  std::printf("bits=%d N=%d cones=%d small_diagonal=%d residual=%g\n",FLASH_REAL_BITS,N,int(cones),int(small_diagonal),worst);
}
int main() {for(bool cones:{false,true}) for(bool small:{false,true}) run(cones,small);}
