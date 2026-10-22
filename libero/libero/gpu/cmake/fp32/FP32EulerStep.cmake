# Optional FP32 mass backsubstitution; independently compiled and device-linked.
set(fp32_euler_step_sources
  "${FLASH_GPU}/cuda/fp32/fused_euler_step.cu"
  "${FLASH_GPU}/cuda/fp32/fused_tail.cuh"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_euler.cuh"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_publication.cuh"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_mass_factor.cuh"
  "${CMAKE_CURRENT_LIST_FILE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${fp32_euler_step_sources})
set(euler_step_hashes)
foreach(source IN LISTS fp32_euler_step_sources)
  file(SHA256 "${source}" source_hash)
  list(APPEND euler_step_hashes "${source_hash}")
endforeach()
list(JOIN euler_step_hashes ":" euler_step_hash_input)
string(SHA256 euler_step_source_id "${euler_step_hash_input}")
add_library(flash_fp32_fused_euler_step SHARED EXCLUDE_FROM_ALL
  "${FLASH_GPU}/cuda/fp32/fused_euler_step.cu")
target_include_directories(flash_fp32_fused_euler_step PRIVATE "${FLASH_MATHDX_ROOT}/include"
  "${FLASH_MATHDX_ROOT}/external/cutlass/include")
target_compile_definitions(flash_fp32_fused_euler_step PRIVATE FLASH_FP32_EULER_STEP_SOURCE_ID="${euler_step_source_id}" FLASH_CUDA_ARCH=${FLASH_CUDA_ARCH})
target_compile_options(flash_fp32_fused_euler_step PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-arch=lto_${FLASH_CUDA_ARCH};-dlto;-lineinfo;--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
target_link_options(flash_fp32_fused_euler_step PRIVATE "$<DEVICE_LINK:-dlto;-arch=sm_${FLASH_CUDA_ARCH}>")
target_link_libraries(flash_fp32_fused_euler_step PRIVATE "${FLASH_SOLVERDX}")
set_target_properties(flash_fp32_fused_euler_step PROPERTIES
  PREFIX "" OUTPUT_NAME "fp32_fused_euler_step" CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON CUDA_ARCHITECTURES OFF
  UNITY_BUILD OFF JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
