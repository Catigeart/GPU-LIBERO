#include "staged_substep.cuh"
#include "newton_shared_layout.cuh"
#include "staged_workspace.cuh"
using namespace flash_libero;
static_assert(FLASH_REAL_BITS == 64 && SubstepShape::threads == 64,
              "Native FP64 Newton/OSC SDK objects require 64 threads");
extern "C" size_t flash_libero_staged_arguments_size() { return sizeof(StagedSubstepArguments); }
extern "C" int flash_libero_staged_scalar_bytes() { return sizeof(real_t); }
extern "C" int flash_libero_staged_nv() { return SubstepShape::nv; }
extern "C" int flash_libero_staged_threads() { return SubstepShape::threads; }
extern "C" size_t flash_libero_staged_shared_bytes(int region, int bodies) {
  switch (region) {
    case 0: return SmoothLayout::bytes;
    case 1: return NewtonLayout::bytes;
    case 2: return bodies > 0 ? staged_post_bytes(bodies) : 0;
    case 3: return EulerLayout::bytes;
    case 4: return 0;
    default: return 0;
  }
}
extern "C" int flash_libero_staged_configure(int *registers, int *static_shared,
    int *local, int *dynamic, int bodies) {
  const void *kernels[] = {reinterpret_cast<const void*>(staged_prepare_kernel),
      reinterpret_cast<const void*>(staged_newton_kernel), reinterpret_cast<const void*>(staged_post_kernel),
      reinterpret_cast<const void*>(staged_reset_tail_kernel), reinterpret_cast<const void*>(staged_motion_kernel)};
  for (int i=0;i<5;++i) {
    dynamic[i]=int(flash_libero_staged_shared_bytes(i,bodies));
    if (dynamic[i]) {
      auto e=cudaFuncSetAttribute(kernels[i],cudaFuncAttributeMaxDynamicSharedMemorySize,dynamic[i]);
      if(e!=cudaSuccess)return int(e);
    }
    cudaFuncAttributes a; auto e=cudaFuncGetAttributes(&a,kernels[i]);if(e!=cudaSuccess)return int(e);
    registers[i]=a.numRegs;static_shared[i]=a.sharedSizeBytes;local[i]=a.localSizeBytes;
  }
  return 0;
}
namespace {
bool valid_motion_threads(int n) { return n==0 || n==64 || n==128 || n==256; }
void launch_prepare(const StagedSubstepArguments *args, dim3 grid, SubstepEntry entry,
                    int phase, int motion_threads, cudaStream_t stream) {
  const int first=entry==SubstepEntry::Start?0:entry==SubstepEntry::ControllerPosition?1:2;
  // Ordinary later forwards reuse velocity/bias. Keep their tiny publication
  // path inside A; a prefix continuation admits a new first motion producer.
  // An unexpected later miss still uses A's original collective fallback.
  const bool separate_motion=motion_threads && phase==first;
  if(separate_motion)
    staged_motion_kernel<<<grid,motion_threads,0,stream>>>(args,entry,phase);
  staged_prepare_kernel<<<grid,substep_threads,SmoothLayout::bytes,stream>>>(args,entry,phase,separate_motion);
}
}
extern "C" int flash_libero_launch_staged_prepare(const StagedSubstepArguments *args,
    int worlds, int entry, int phase, int motion_threads, void *stream) {
  if(worlds<=0 || entry<0 || entry>2 || phase<0 || phase>2 || !valid_motion_threads(motion_threads))
    return int(cudaErrorInvalidValue);
  launch_prepare(args,dim3(1,worlds),static_cast<SubstepEntry>(entry),phase,
                 motion_threads,static_cast<cudaStream_t>(stream));
  return int(cudaGetLastError());
}
extern "C" int flash_libero_launch_staged_forward(const StagedSubstepArguments *args,
    int worlds, int outputs, int motion_threads, int post_bytes, void *stream) {
  if(worlds<=0 || !valid_motion_threads(motion_threads) || post_bytes<=0) return int(cudaErrorInvalidValue);
  auto s=static_cast<cudaStream_t>(stream);dim3 grid(1,worlds);
  launch_prepare(args,grid,SubstepEntry::Start,0,motion_threads,s);
  staged_newton_kernel<<<grid,substep_threads,NewtonLayout::bytes,s>>>(args,0);
  staged_post_kernel<<<grid,substep_threads,post_bytes,s>>>(args,SubstepEntry::Start,0,outputs!=0,outputs!=0,false);
  return int(cudaGetLastError());
}
extern "C" int flash_libero_launch_staged_substep(const StagedSubstepArguments *args,
    int worlds, int policy, int entry_value, int post, int motion_threads, int post_bytes, void *stream) {
  if(worlds<=0 || entry_value<0 || entry_value>3 || (post!=0 && post!=1) ||
      !valid_motion_threads(motion_threads) || post_bytes<=0)return int(cudaErrorInvalidValue);
  auto s=static_cast<cudaStream_t>(stream);dim3 grid(1,worlds);
  auto entry=static_cast<SubstepEntry>(entry_value);
  if(entry==SubstepEntry::AccelerationReset) {
    staged_reset_tail_kernel<<<grid,substep_threads,EulerLayout::bytes,s>>>(args);
  } else {
    const int first=entry==SubstepEntry::Start?0:entry==SubstepEntry::ControllerPosition?1:2;
    for(int phase=first;phase<3;++phase) {
      launch_prepare(args,grid,entry,phase,motion_threads,s);
      staged_newton_kernel<<<grid,substep_threads,NewtonLayout::bytes,s>>>(args,phase);
      staged_post_kernel<<<grid,substep_threads,post_bytes,s>>>(args,entry,phase,policy!=0,post!=0,true);
    }
  }
  return int(cudaGetLastError());
}
