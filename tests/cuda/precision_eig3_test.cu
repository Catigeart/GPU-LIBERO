// Exercise the production covariance eigensolver with independent residuals.
#include "collision/collision_bounds.cuh"
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

__global__ void eigen_cases(const real_t *input, real_t *output, int *status, int count) {
  const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= count) return;
  real_t values[3], frame[9], quaternion[4], temporary[9], work[9];
  status[i] = collision_cuda::cf_eig3(input + 9 * i, values, frame, quaternion, temporary, work);
  for (int k = 0; k < 3; ++k) output[12 * i + k] = values[k];
  for (int k = 0; k < 9; ++k) output[12 * i + 3 + k] = frame[k];
}

int main() {
  std::vector<real_t> input = {
      real_t(0.0336080901324749), real_t(0.0006891652592457831), real_t(-0.006049806252121925),
      real_t(0.0006891652592457831), real_t(0.027324628084897995), real_t(-0.002083723433315754),
      real_t(-0.006049806252121925), real_t(-0.002083723433315754), real_t(0.04187101498246193)};
  std::mt19937 engine(1207);
  std::normal_distribution<double> distribution;
  for (double scale : {1e-6, 1., 1e6}) {
    for (int test = 0; test < 32; ++test) {
      double a[9]; for (double &v : a) v = distribution(engine);
      for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
        double value = 0.;
        for (int k = 0; k < 3; ++k) value += a[3*i+k] * a[3*j+k];
        input.push_back(real_t(scale * value));
      }
    }
  }
  for (const std::array<double, 3> diagonal : {
           std::array<double,3>{0.,0.,0.}, {1.,1.,1.}, {1.,1e-8,0.}})
    for (int i=0;i<3;++i) for(int j=0;j<3;++j) input.push_back(real_t(i==j ? diagonal[i] : 0.));
  const int count = int(input.size()/9);
  real_t *device_input = nullptr, *device_output = nullptr;
  int *device_status = nullptr;
  if (cudaMalloc(&device_input,input.size()*sizeof(real_t)) ||
      cudaMalloc(&device_output,count*12*sizeof(real_t)) || cudaMalloc(&device_status,count*sizeof(int))) return 3;
  cudaMemcpy(device_input,input.data(),input.size()*sizeof(real_t),cudaMemcpyHostToDevice);
  eigen_cases<<<(count+31)/32,32>>>(device_input,device_output,device_status,count);
  if (cudaDeviceSynchronize()!=cudaSuccess) return 4;
  std::vector<real_t> output(count*12);std::vector<int> status(count);
  cudaMemcpy(output.data(),device_output,output.size()*sizeof(real_t),cudaMemcpyDeviceToHost);
  cudaMemcpy(status.data(),device_status,count*sizeof(int),cudaMemcpyDeviceToHost);
  int failures=0;double worst_relative=0.,worst_orthogonality=0.;
  for(real_t value : output) if(!std::isfinite(double(value))) ++failures;
  for(int z=0;z<count;++z) {
    const real_t *matrix=input.data()+9*z,*values=output.data()+12*z,*q=values+3;
    double norm=0.,residual=0.,orthogonality=0.;
    for(int i=0;i<3;++i) {
      double row=0.;for(int j=0;j<3;++j)row+=std::abs(double(matrix[3*i+j]));
      norm=std::max(norm,row);
      for(int j=0;j<3;++j) {
        double r=-double(q[3*i+j])*values[j],o=-(i==j?1.:0.);
        for(int k=0;k<3;++k){r+=double(matrix[3*i+k])*q[3*k+j];o+=double(q[3*k+i])*q[3*k+j];}
        residual=std::max(residual,std::abs(r));orthogonality=std::max(orthogonality,std::abs(o));
      }
    }
    const double relative_limit=FLASH_REAL_BITS==32 ? 64.*REAL_EPSILON : 4e-6;
    const bool passed=status[z]>=0 && residual<=relative_limit*norm+4e-12 &&
        orthogonality<=32.*REAL_EPSILON && values[0]>=values[1] && values[1]>=values[2];
    if(!passed){++failures;std::printf("case=%d status=%d residual=%g norm=%g orthogonality=%g\n",z,status[z],residual,norm,orthogonality);}
    worst_relative=std::max(worst_relative,residual/std::max(norm,1e-12));
    worst_orthogonality=std::max(worst_orthogonality,orthogonality);
  }
  cudaFree(device_input);cudaFree(device_output);cudaFree(device_status);
  std::printf("real_bits=%d cases=%d failures=%d worst_relative_residual=%g worst_orthogonality=%g\n",
              FLASH_REAL_BITS,count,failures,worst_relative,worst_orthogonality);
  return failures?2:0;
}
