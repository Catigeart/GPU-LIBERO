#include "staged_substep.cuh"
#include "newton_shared_layout.cuh"
using namespace flash_libero;
static_assert(FLASH_REAL_BITS == 64 && SubstepShape::threads == 64,
              "Initial staged baseline requires native FP64/64-thread SDK objects");
extern "C" size_t flash_libero_staged_arguments_size() { return sizeof(StagedSubstepArguments); }
extern "C" int flash_libero_staged_scalar_bytes() { return sizeof(real_t); }
extern "C" int flash_libero_staged_nv() { return SubstepShape::nv; }
extern "C" int flash_libero_staged_threads() { return SubstepShape::threads; }
extern "C" int flash_libero_staged_configure(int *registers, int *static_shared) {
  const void *kernels[] = {reinterpret_cast<const void*>(staged_prepare_kernel),
      reinterpret_cast<const void*>(staged_newton_kernel), reinterpret_cast<const void*>(staged_post_kernel),
      reinterpret_cast<const void*>(staged_reset_tail_kernel)};
  for (int i=0;i<4;++i) {
    auto e=cudaFuncSetAttribute(kernels[i],cudaFuncAttributeMaxDynamicSharedMemorySize,substep_shared_bytes);
    if(e!=cudaSuccess)return int(e);
    cudaFuncAttributes a; e=cudaFuncGetAttributes(&a,kernels[i]);if(e!=cudaSuccess)return int(e);
    registers[i]=a.numRegs;static_shared[i]=a.sharedSizeBytes;
  }
  return 0;
}
extern "C" int flash_libero_launch_staged_forward(const StagedSubstepArguments *args,
    int worlds, int outputs, void *stream) {
  auto s=static_cast<cudaStream_t>(stream);dim3 grid(1,worlds);
  staged_prepare_kernel<<<grid,substep_threads,substep_shared_bytes,s>>>(args,SubstepEntry::Start,0);
  staged_newton_kernel<<<grid,substep_threads,substep_shared_bytes,s>>>(args,0);
  staged_post_kernel<<<grid,substep_threads,substep_shared_bytes,s>>>(args,SubstepEntry::Start,0,outputs!=0,outputs!=0,false);
  return int(cudaGetLastError());
}
extern "C" int flash_libero_launch_staged_substep(const StagedSubstepArguments *args,
    int worlds, int policy, int entry_value, int post, void *stream) {
  if(worlds<=0 || entry_value<0 || entry_value>3 || (post!=0 && post!=1))return int(cudaErrorInvalidValue);
  auto s=static_cast<cudaStream_t>(stream);dim3 grid(1,worlds);
  auto entry=static_cast<SubstepEntry>(entry_value);
  if(entry==SubstepEntry::AccelerationReset) {
    staged_reset_tail_kernel<<<grid,substep_threads,substep_shared_bytes,s>>>(args);
  } else {
    const int first=entry==SubstepEntry::Start?0:entry==SubstepEntry::ControllerPosition?1:2;
    for(int phase=first;phase<3;++phase) {
      staged_prepare_kernel<<<grid,substep_threads,substep_shared_bytes,s>>>(args,entry,phase);
      staged_newton_kernel<<<grid,substep_threads,substep_shared_bytes,s>>>(args,phase);
      staged_post_kernel<<<grid,substep_threads,substep_shared_bytes,s>>>(args,entry,phase,policy!=0,post!=0,true);
    }
  }
  return int(cudaGetLastError());
}
