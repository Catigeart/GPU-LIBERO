set(fp32_observation_sources
  "${FLASH_GPU}/cuda/fp32/fused_observations.cu"
  "${FLASH_GPU}/cuda/fp32/fused_observations.cuh"
  "${CMAKE_CURRENT_LIST_FILE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${fp32_observation_sources})
set(observation_hashes)
foreach(source IN LISTS fp32_observation_sources)
  file(SHA256 "${source}" source_hash)
  list(APPEND observation_hashes "${source_hash}")
endforeach()
list(JOIN observation_hashes ":" observation_hash_input)
string(SHA256 observation_source_id "${observation_hash_input}")
add_library(flash_fp32_fused_observations SHARED EXCLUDE_FROM_ALL "${FLASH_GPU}/cuda/fp32/fused_observations.cu")
target_compile_definitions(flash_fp32_fused_observations PRIVATE
  FLASH_FP32_OBSERVATIONS_SOURCE_ID="${observation_source_id}" FLASH_CUDA_ARCH=${FLASH_CUDA_ARCH})
target_compile_options(flash_fp32_fused_observations PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
set_target_properties(flash_fp32_fused_observations PROPERTIES PREFIX "" OUTPUT_NAME "fp32_fused_observations"
  CUDA_ARCHITECTURES ${FLASH_CUDA_ARCH} UNITY_BUILD OFF
  JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
