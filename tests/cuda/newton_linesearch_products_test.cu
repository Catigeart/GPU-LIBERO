// Frozen-region comparison of actual native Jv/quadratic preparation. The
// original tile path is the reference, including its exceptional write prefix.
#include "newton_solver.cu"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
using namespace physics_newton;
static void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct Buffers {
  std::vector<void*> owners;
  template<class T> T *put(const std::vector<T>& v) {
    T *p; check(cudaMalloc(&p, v.size() * sizeof(T)));
    owners.push_back(p);
    check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
  }
  ~Buffers() { for (void *p : owners) cudaFree(p); }
};
__global__ void products(const NewtonWorldData *arguments, int rows, const real_t *search,
                         bool candidate, int repeats, int *status) {
  const NewtonWorldData a = arguments[blockIdx.x];
  status += 2 * blockIdx.x;
  extern __shared__ __align__(16) unsigned char arena[];
  auto &c = newton_control(arena);
  for (int i = threadIdx.x; i < NewtonControl::integer_count; i += blockDim.x) c.integers[i] = 0;
  for (int i = threadIdx.x; i < NewtonControl::real_count; i += blockDim.x) c.reals[i] = 0;
  real_t *v = flash_libero::shared_at<real_t>(arena, flash_libero::NewtonLayout::search);
  for (int j = threadIdx.x; j < SubstepShape::nv; j += blockDim.x) v[j] = search[j];
  if (threadIdx.x == 0) { c.count() = rows; c.active() = 1; }
  __syncthreads();
  bool full = false;
  NewtonModelData model{};
  for (int k = 0; k < repeats; ++k) {
    if (candidate) full = s07_full_products_block(arena, model, a, nullptr);
    if (!full)
      s07_jv_blas_dx(arena, model, a, nullptr, 0, rows, SubstepShape::rows,
          flash_libero::shared_at<real_t>(arena, flash_libero::NewtonLayout::mathdx),
          a.blas_diagnostic, true, a.constraint_group_heads, a.constraint_head_rows,
          a.constraint_tile_head_offset);
  }
  if (threadIdx.x == 0) { status[0] = c.s07_status(); status[1] = full; }
}
static void run(int rows, int exceptional, bool timing, int contact_width=3, int threads=64) {
  constexpr int nv = SubstepShape::nv, capacity = 256;
  Buffers buffers;
  std::vector<real_t> J(capacity * nv), D(capacity, real_t(.75)), jar(capacity);
  std::vector<real_t> friction(capacity * 5, real_t(.8)), mu(capacity, real_t(.9));
  std::vector<int> type(capacity), dim(capacity, 1), heads, groups(capacity, -1);
  for (int r = 0; r < rows;) {
    const int width = r >= 9 && r + contact_width <= rows ? contact_width : 1;
    heads.push_back(r); groups[r] = r;
    for (int k = 0; k < width; ++k) { type[r+k] = width > 1 ? 7 : 1; dim[r+k] = width; }
    r += width;
  }
  for (int r = 0; r < rows; ++r) {
    jar[r] = real_t(std::sin(.13 * r));
    for (int j = 0; j < nv; ++j) J[r*nv+j] = real_t(std::sin(.03*r+.11*j));
  }
  if (exceptional == 1) J[19 * nv] = NAN;
  if (exceptional == 2) jar[0] = real_t(1.e200);
  std::vector<int> offsets((capacity + 15) / 16 + 1);
  for (int t = 0, k = 0; t < int(offsets.size()); ++t) {
    while (k < int(heads.size()) && (heads[k] + dim[heads[k]] - 1) / 16 < t) ++k;
    offsets[t] = k;
  }
  std::vector<real_t> search(nv), coefficients(NewtonCoefficientLayout::bytes / sizeof(real_t));
  for (int j = 0; j < nv; ++j) search[j] = real_t(std::cos(.17*j));
  for (int r : heads) coefficients[NewtonCoefficientLayout::cone_dm/sizeof(real_t)+r] =
      D[r] / (mu[r] * mu[r] * (1 + mu[r]*mu[r]));
  NewtonWorldData base{};
  base.J=buffers.put(J); base.D=buffers.put(D); base.jar=buffers.put(jar);
  base.efc_type=buffers.put(type); base.cone_dim=buffers.put(dim);
  base.cone_mu=buffers.put(mu); base.cone_friction=buffers.put(friction);
  base.constraint_head_rows=buffers.put(heads);
  base.constraint_head_count=buffers.put(std::vector<int>{int(heads.size())});
  base.constraint_group_heads=buffers.put(groups); base.constraint_tile_head_offset=buffers.put(offsets);
  base.precomputed=reinterpret_cast<unsigned char*>(buffers.put(coefficients));
  real_t *v=buffers.put(search);
  std::vector<real_t> jv[2], quad[2]; int result[2][4];
  float elapsed[2]={};
  for (int mode = 0; mode < 2; ++mode) {
    std::vector<NewtonWorldData> worlds(2,base);
    for (auto &a : worlds) {
      a.Jv=buffers.put(std::vector<real_t>(capacity,real_t(123.25)));
      a.quad=buffers.put(std::vector<real_t>(capacity*3,real_t(123.25)));
      a.efc_b=buffers.put(std::vector<real_t>(capacity));
      a.blas_diagnostic=buffers.put(std::vector<int>(7));
    }
    NewtonWorldData *arguments=buffers.put(worlds);
    int *status=buffers.put(std::vector<int>(4));
    auto launch=[&](int n) {
      products<<<2,threads,flash_libero::substep_shared_bytes>>>(arguments,rows,v,mode,n,status);
      check(cudaGetLastError());
    };
    check(cudaFuncSetAttribute(products,cudaFuncAttributeMaxDynamicSharedMemorySize,
                               flash_libero::substep_shared_bytes));
    launch(1); check(cudaDeviceSynchronize());
    jv[mode].resize(2*capacity); quad[mode].resize(2*capacity*3);
    for (int w=0;w<2;++w) {
      check(cudaMemcpy(jv[mode].data()+w*capacity,worlds[w].Jv,capacity*sizeof(real_t),cudaMemcpyDeviceToHost));
      check(cudaMemcpy(quad[mode].data()+w*capacity*3,worlds[w].quad,capacity*3*sizeof(real_t),cudaMemcpyDeviceToHost));
    }
    check(cudaMemcpy(result[mode],status,4*sizeof(int),cudaMemcpyDeviceToHost));
    if (timing) {
      cudaEvent_t start,end;check(cudaEventCreate(&start));check(cudaEventCreate(&end));
      launch(5);check(cudaEventRecord(start));launch(256);check(cudaEventRecord(end));
      check(cudaEventSynchronize(end));check(cudaEventElapsedTime(&elapsed[mode],start,end));
      check(cudaEventDestroy(start));check(cudaEventDestroy(end));
    }
  }
  if (result[0][0]!=result[1][0] || result[0][2]!=result[1][2]) throw std::runtime_error("line-search status changed");
  const bool expected=threads==64 && rows>=64 && heads.size()<=64 && exceptional==0;
  if (bool(result[1][1])!=expected || bool(result[1][3])!=expected) throw std::runtime_error("full-product selection differs");
  double worst=0;
  for (int field=0;field<2;++field) {
    const auto &x=field?quad[0]:jv[0], &y=field?quad[1]:jv[1];
    for (size_t i=0;i<x.size();++i) {
      if (std::isnan(x[i])) {if(!std::isnan(y[i]))throw std::runtime_error("NaN prefix differs");continue;}
      if (!std::isfinite(x[i])) {if(x[i]!=y[i])throw std::runtime_error("infinite prefix differs");continue;}
      if (!std::isfinite(y[i])) throw std::runtime_error("nonfinite candidate endpoint");
      const double e=double(std::abs(x[i]-y[i])/(1+std::abs(x[i])));
      worst=std::max(worst,e);
      if (e>128*REAL_EPSILON) throw std::runtime_error("Jv/quad endpoint differs");
    }
  }
  if (!exceptional) for (int r=0;r<rows;++r) {
    long double dot=0,scale=0;
    for (int j=0;j<nv;++j) {const long double term=(long double)J[r*nv+j]*search[j];dot+=term;scale+=std::abs(term);}
    if(std::abs(jv[1][r]-dot)>(nv+8)*REAL_EPSILON*(1+scale) ||
       std::abs(jv[1][capacity+r]-dot)>(nv+8)*REAL_EPSILON*(1+scale))throw std::runtime_error("independent Jv residual");
  }
  std::printf("rows=%d exceptional=%d selected=%d status=%d scaled_error=%.8g baseline_us=%.5f candidate_us=%.5f\n",
      rows,exceptional,result[1][1],result[1][0],worst,elapsed[0]*1000/256,elapsed[1]*1000/256);
}
int main(int argc,char **argv){try{
  const bool timing=argc==2 && std::strcmp(argv[1],"--timing")==0;
  if(argc>1 && !timing) throw std::runtime_error("expected optional --timing");
  for(int rows:{63,64,129,165,201})run(rows,0,timing && rows==165);
  run(165,1,false);run(165,2,false);
  run(165,0,false,4);run(165,0,false,6);
  run(165,0,false,3,128);
}catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
