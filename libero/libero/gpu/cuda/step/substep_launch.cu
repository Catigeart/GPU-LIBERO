#include "substep_kernel.cuh"
#include "newton_types.cuh"
#include "newton_shared_layout.cuh"
#include "substep_profile.cuh"

using namespace flash_libero;

// Kept with the maintained ctypes mirror in host/runtime/step/substep_types.py.
static_assert(sizeof(SubstepArguments) == 6288, "Update both CUDA and ctypes argument layouts");

extern "C" size_t flash_libero_newton_arguments_size() { return sizeof(physics_newton::NewtonArguments); }
extern "C" size_t flash_libero_newton_world_size() { return sizeof(physics_newton::NewtonWorldData); }
extern "C" size_t flash_libero_newton_model_size() { return sizeof(physics_newton::NewtonModelData); }
extern "C" size_t flash_libero_newton_post_size() { return sizeof(physics_newton::NewtonPostData); }

extern "C" size_t flash_libero_substep_arguments_size() { return sizeof(SubstepArguments); }
extern "C" size_t flash_libero_substep_shared_bytes() { return substep_shared_bytes; }
extern "C" int flash_libero_substep_dimension(int index) {
  constexpr int values[] = {SubstepShape::nv, SubstepShape::nq, SubstepShape::packed_mass,
      SubstepShape::njnt, SubstepShape::nu, SubstepShape::rows, SubstepShape::contacts,
      SubstepShape::threads, SubstepShape::quaternion_joints, SubstepShape::sm};
  return index >= 0 && index < 10 ? values[index] : -1;
}

extern "C" int flash_libero_substep_configure(int *registers, int *static_shared) {
  cudaError_t status =
      cudaFuncSetAttribute(fused_substep_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, substep_shared_bytes);
  if (status != cudaSuccess)
    return int(status);
  cudaFuncAttributes attributes;
  status = cudaFuncGetAttributes(&attributes, fused_substep_kernel);
  if (status != cudaSuccess)
    return int(status);
  *registers = attributes.numRegs;
  *static_shared = attributes.sharedSizeBytes;
  return 0;
}

extern "C" int flash_libero_launch_substep(const SubstepArguments *arguments, int worlds, int policy_step, int entry,
                                           void *stream) {
  fused_substep_kernel<<<dim3(1, worlds), substep_threads, substep_shared_bytes, static_cast<cudaStream_t>(stream)>>>(
      arguments, policy_step != 0, static_cast<SubstepEntry>(entry), true);
  return int(cudaGetLastError());
}

// Explicit publication demand; the original launch ABI retains complete Post.
extern "C" int flash_libero_launch_substep_with_post(const SubstepArguments *arguments, int worlds,
                                                    int policy_step, int entry, int publish_post, void *stream) {
  if (publish_post != 0 && publish_post != 1)
    return int(cudaErrorInvalidValue);
  fused_substep_kernel<<<dim3(1, worlds), substep_threads, substep_shared_bytes, static_cast<cudaStream_t>(stream)>>>(
      arguments, policy_step != 0, static_cast<SubstepEntry>(entry), publish_post != 0);
  return int(cudaGetLastError());
}

extern "C" int flash_libero_select_substep_continuation(const SubstepArguments *arguments, int entry, int worlds,
                                                        int *predicate, void *stream) {
  select_substep_continuation<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(
      arguments, static_cast<SubstepEntry>(entry), worlds, predicate);
  return int(cudaGetLastError());
}

#if defined(FLASH_SUBSTEP_PROFILE)
namespace flash_libero::substep_profile {
__device__ Configuration configuration{};
}
extern "C" int flash_libero_substep_profile_configure(void *pointer, int capacity) {
  if (pointer && capacity <= 0) return int(cudaErrorInvalidValue);
  const substep_profile::Configuration config{static_cast<uint64_t *>(pointer), capacity};
  return int(cudaMemcpyToSymbol(substep_profile::configuration, &config, sizeof(config)));
}
extern "C" int flash_libero_substep_profile_header_cells() { return substep_profile::header_cells; }
extern "C" int flash_libero_substep_profile_record_cells() { return substep_profile::record_cells; }
extern "C" int flash_libero_substep_profile_phase_count() { return int(substep_profile::Phase::Count); }
extern "C" const char *flash_libero_substep_profile_phase_name(int phase) {
  static const char *names[] = {
    "forward", "velocity_cold", "velocity_reuse", "reference_cold", "reference_reuse",
    "bias_cold", "bias_reuse", "actuation", "smooth", "newton_bridge", "newton", "post",
    "forward_publish", "controller", "begin_forward", "checks", "integrate", "total"
  };
  static_assert(sizeof(names) / sizeof(names[0]) == int(substep_profile::Phase::Count));
  return phase >= 0 && phase < int(substep_profile::Phase::Count) ? names[phase] : nullptr;
}
#endif
