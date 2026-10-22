set(fp32_motion_passive_sources
  "${FLASH_GPU}/cuda/fp32/fused_motion_passive.cu"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_velocity.cuh"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_bias.cuh"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_fluid.cuh"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_passive.cuh"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_passive_math.cuh"
  "${CMAKE_CURRENT_LIST_FILE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${fp32_motion_passive_sources})
set(motion_passive_hashes)
foreach(source IN LISTS fp32_motion_passive_sources)
  file(SHA256 "${source}" source_hash)
  list(APPEND motion_passive_hashes "${source_hash}")
endforeach()
list(JOIN motion_passive_hashes ":" motion_passive_hash_input)
string(SHA256 motion_passive_source_id "${motion_passive_hash_input}")
add_library(flash_fp32_fused_motion_passive SHARED EXCLUDE_FROM_ALL "${FLASH_GPU}/cuda/fp32/fused_motion_passive.cu")
target_compile_definitions(flash_fp32_fused_motion_passive PRIVATE
  FLASH_FP32_MOTION_PASSIVE_SOURCE_ID="${motion_passive_source_id}" FLASH_CUDA_ARCH=${FLASH_CUDA_ARCH})
target_compile_options(flash_fp32_fused_motion_passive PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
set_target_properties(flash_fp32_fused_motion_passive PROPERTIES PREFIX "" OUTPUT_NAME "fp32_fused_motion_passive"
  CUDA_ARCHITECTURES ${FLASH_CUDA_ARCH} UNITY_BUILD OFF
  JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
