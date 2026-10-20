// Both-precision independent residual and output-hold checks: all catalogue NV, mixed two-world row
// counts, empty/odd/tile tails, dispatch boundaries, and masked NaN J rows.
#include "newton_gemv.cuh"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

static void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <int NV> __global__ void products(const real_t *input, real_t *output, int capacity,
                                         bool masked) {
  const int world = blockIdx.x;
  const int rows = world && capacity ? capacity / 2 + 1 : capacity;
  const real_t *J = input + size_t(world) * (capacity * (NV + 2) + 2 * NV);
  const real_t *force = J + capacity * NV, *aref = force + capacity;
  const real_t *smooth = aref + capacity, *warm = smooth + NV;
  real_t *a = output + size_t(world) * (3 * capacity + NV);
  real_t *b = a + capacity, *v = b + capacity, *t = v + capacity;
  __shared__ real_t tile[16], partials[2 * NV];
  constexpr int lanes = FLASH_REAL_BITS == 32 || NV <= 32 ? 8 : 16;
  if (!masked) {
    physics_newton::gemv::candidates<NV, lanes>(J,smooth,warm,aref,rows,a,b);
    __syncthreads();
    for (int begin=0; begin<rows; begin+=16) {
      physics_newton::gemv::tile<NV, lanes>(J,smooth,begin,rows,tile);
      __syncthreads();
      for (int r=threadIdx.x; r<16 && begin+r<rows; r+=blockDim.x) v[begin+r]=tile[r];
      __syncthreads();
    }
  }
  physics_newton::gemv::transpose<NV>(J,force,rows,t,partials);
}

static void near(real_t value, long double expected, long double scale, int depth) {
  if (!std::isfinite(value) || fabsl(value-expected)>(depth+8)*REAL_EPSILON*(1+scale))
    throw std::runtime_error("GEMV differs from long-real_t reference");
}

template <int NV> static void run(int capacity, bool masked) {
  const size_t in_stride=size_t(capacity)*(NV+2)+2*NV, out_stride=3*capacity+NV;
  std::vector<real_t> input(2*in_stride),output(2*out_stride,real_t(123.25));
  for (int w=0;w<2;++w) {
    real_t *J=input.data()+w*in_stride,*f=J+capacity*NV,*a=f+capacity,*s=a+capacity,*v=s+NV;
    for (int j=0;j<NV;++j) {s[j]=sin(.17*j+.3*w);v[j]=cos(.11*j+.2*w);}
    for (int r=0;r<capacity;++r) {
      f[r]=r%3 ? sin(.07*r+.1*w) : 0; a[r]=cos(.13*r);
      for (int j=0;j<NV;++j) J[r*NV+j]=masked && f[r]==0 ? nan("") : sin(.03*r+.19*j+.2*w);
    }
  }
  real_t *device_input=nullptr,*device_output=nullptr;
  check(cudaMalloc(&device_input,input.size()*sizeof(real_t)));
  check(cudaMalloc(&device_output,output.size()*sizeof(real_t)));
  check(cudaMemcpy(device_input,input.data(),input.size()*sizeof(real_t),cudaMemcpyHostToDevice));
  check(cudaMemcpy(device_output,output.data(),output.size()*sizeof(real_t),cudaMemcpyHostToDevice));
  products<NV><<<2,64>>>(device_input,device_output,capacity,masked);
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  check(cudaMemcpy(output.data(),device_output,output.size()*sizeof(real_t),cudaMemcpyDeviceToHost));
  check(cudaFree(device_output));check(cudaFree(device_input));
  for (int w=0;w<2;++w) {
    const int rows=w && capacity ? capacity/2+1 : capacity;
    const real_t *J=input.data()+w*in_stride,*f=J+capacity*NV,*a=f+capacity,*s=a+capacity,*v=s+NV;
    const real_t *out=output.data()+w*out_stride;
    if (!masked) for (int r=0;r<rows;++r) {
      long double x=0,y=0,sx=0,sy=0;
      for (int j=0;j<NV;++j) {const long double a=(long double)J[r*NV+j]*s[j],b=(long double)J[r*NV+j]*v[j];x+=a;y+=b;sx+=fabsl(a);sy+=fabsl(b);}
      near(out[r],x-a[r],sx+fabsl(a[r]),NV);near(out[capacity+r],y-a[r],sy+fabsl(a[r]),NV);near(out[2*capacity+r],x,sx,NV);
    }
    for(int band=0;band<3;++band) for(int r=masked?0:rows;r<capacity;++r)
      if(out[band*capacity+r]!=real_t(123.25))throw std::runtime_error("GEMV output hold damaged");
    for (int j=0;j<NV;++j) {
      long double value=0,scale=0;
      for (int r=0;r<rows;++r) if (f[r]!=0) {const long double term=(long double)J[r*NV+j]*f[r];value+=term;scale+=fabsl(term);}
      near(out[3*capacity+j],value,scale,rows);
    }
  }
}

template <int NV> static void family() {
  for (int rows : {0,1,15,16,17,185,255,256,257,366,513}) {
    run<NV>(rows,false);run<NV>(rows,true);
  }
  if constexpr (NV==21 || NV==57) run<NV>(30075,true);
  printf("NV=%d: two-world GEMV checks passed\n",NV);
}

int main() {
  try {
    family<21>();family<22>();family<24>();family<27>();family<30>();family<33>();family<36>();
    family<37>();family<39>();family<43>();family<45>();family<51>();family<57>();
  } catch (const std::exception &error) {fprintf(stderr,"%s\n",error.what());return 1;}
}
