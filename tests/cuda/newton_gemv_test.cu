// Offline-built mathematical checks: all catalogue NV, mixed two-world row
// counts, empty/odd/tile tails, dispatch boundaries, and masked NaN J rows.
#include "newton_gemv.cuh"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

static void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <int NV> __global__ void products(const double *input, double *output, int capacity,
                                         bool masked) {
  const int world = blockIdx.x;
  const int rows = world && capacity ? capacity / 2 + 1 : capacity;
  const double *J = input + size_t(world) * (capacity * (NV + 2) + 2 * NV);
  const double *force = J + capacity * NV, *aref = force + capacity;
  const double *smooth = aref + capacity, *warm = smooth + NV;
  double *a = output + size_t(world) * (3 * capacity + NV);
  double *b = a + capacity, *v = b + capacity, *t = v + capacity;
  __shared__ double tile[16], partials[2 * NV];
  constexpr int lanes = NV <= 32 ? 8 : 16;
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

static void near(double value, long double expected) {
  if (!std::isfinite(value) || fabsl(value-expected)>1e-11L*(1+fabsl(expected)))
    throw std::runtime_error("GEMV differs from long-double reference");
}

template <int NV> static void run(int capacity, bool masked) {
  const size_t in_stride=size_t(capacity)*(NV+2)+2*NV, out_stride=3*capacity+NV;
  std::vector<double> input(2*in_stride),output(2*out_stride);
  for (int w=0;w<2;++w) {
    double *J=input.data()+w*in_stride,*f=J+capacity*NV,*a=f+capacity,*s=a+capacity,*v=s+NV;
    for (int j=0;j<NV;++j) {s[j]=sin(.17*j+.3*w);v[j]=cos(.11*j+.2*w);}
    for (int r=0;r<capacity;++r) {
      f[r]=r%3 ? sin(.07*r+.1*w) : 0; a[r]=cos(.13*r);
      for (int j=0;j<NV;++j) J[r*NV+j]=masked && f[r]==0 ? nan("") : sin(.03*r+.19*j+.2*w);
    }
  }
  double *device_input=nullptr,*device_output=nullptr;
  check(cudaMalloc(&device_input,input.size()*sizeof(double)));
  check(cudaMalloc(&device_output,output.size()*sizeof(double)));
  check(cudaMemcpy(device_input,input.data(),input.size()*sizeof(double),cudaMemcpyHostToDevice));
  products<NV><<<2,64>>>(device_input,device_output,capacity,masked);
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  check(cudaMemcpy(output.data(),device_output,output.size()*sizeof(double),cudaMemcpyDeviceToHost));
  check(cudaFree(device_output));check(cudaFree(device_input));
  for (int w=0;w<2;++w) {
    const int rows=w && capacity ? capacity/2+1 : capacity;
    const double *J=input.data()+w*in_stride,*f=J+capacity*NV,*a=f+capacity,*s=a+capacity,*v=s+NV;
    const double *out=output.data()+w*out_stride;
    if (!masked) for (int r=0;r<rows;++r) {
      long double x=0,y=0;
      for (int j=0;j<NV;++j) {x+=(long double)J[r*NV+j]*s[j];y+=(long double)J[r*NV+j]*v[j];}
      near(out[r],x-a[r]);near(out[capacity+r],y-a[r]);near(out[2*capacity+r],x);
    }
    for (int j=0;j<NV;++j) {
      long double value=0;
      for (int r=0;r<rows;++r) if (f[r]!=0) value+=(long double)J[r*NV+j]*f[r];
      near(out[3*capacity+j],value);
    }
  }
}

template <int NV> static void family() {
  for (int rows : {0,1,15,16,17,255,256,257,513}) {
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
