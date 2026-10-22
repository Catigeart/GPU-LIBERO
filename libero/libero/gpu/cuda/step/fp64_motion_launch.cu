// Copyright 2026 Flash-LIBERO contributors
// SPDX-License-Identifier: Apache-2.0
#include "fp64_motion_passive.cuh"
#include "newton_shared_layout.cuh"
using namespace flash_libero;
static_assert(FLASH_REAL_BITS==64 && SubstepShape::threads==64,
              "Peeled FP64 motion retains the native Newton/OSC SDK objects");
extern "C" size_t flash_libero_motion_arguments_size() { return sizeof(SubstepArguments); }
extern "C" int flash_libero_motion_nv() { return SubstepShape::nv; }
extern "C" int flash_libero_motion_configure(int *registers,int *shared,int *local) {
  auto e=cudaFuncSetAttribute(motion_residual_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,substep_shared_bytes);
  if(e!=cudaSuccess)return int(e);
  const void *kernels[]={reinterpret_cast<const void*>(fp64_motion_passive_kernel),
                          reinterpret_cast<const void*>(motion_residual_kernel)};
  for(int i=0;i<2;++i) {
    cudaFuncAttributes a; e=cudaFuncGetAttributes(&a,kernels[i]);
    if(e!=cudaSuccess)return int(e);
    registers[i]=a.numRegs;shared[i]=int(a.sharedSizeBytes);local[i]=int(a.localSizeBytes);
  }
  return 0;
}
extern "C" int flash_libero_launch_motion(const SubstepArguments *arguments,
    const int *plan,int *ready,int worlds,int entry_value,int threads,void *stream) {
  if(worlds<1 || entry_value<0 || entry_value>2 ||
     (threads!=128&&threads!=256&&threads!=512))return int(cudaErrorInvalidValue);
  fp64_motion_passive_kernel<<<dim3(1,worlds),threads,0,static_cast<cudaStream_t>(stream)>>>(
      arguments,plan,ready,static_cast<SubstepEntry>(entry_value));
  return int(cudaGetLastError());
}
extern "C" int flash_libero_launch_motion_substep(const SubstepArguments *arguments,
    const int *plan,int *ready,int worlds,int policy,int entry_value,int post,int threads,void *stream) {
  if(worlds<1 || entry_value<0 || entry_value>3 || (post!=0&&post!=1) ||
     (threads!=0&&threads!=128&&threads!=256&&threads!=512))return int(cudaErrorInvalidValue);
  auto s=static_cast<cudaStream_t>(stream);
  const auto entry=static_cast<SubstepEntry>(entry_value);
  if(threads && entry!=SubstepEntry::AccelerationReset) {
    const int status=flash_libero_launch_motion(arguments,plan,ready,worlds,entry_value,threads,stream);
    if(status)return status;
  }
  motion_residual_kernel<<<dim3(1,worlds),substep_threads,substep_shared_bytes,s>>>(
      arguments,policy!=0,entry,post!=0,threads&&entry!=SubstepEntry::AccelerationReset?ready:nullptr);
  return int(cudaGetLastError());
}
