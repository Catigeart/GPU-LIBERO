# Optional bias-only RNE; separate from com_vel and all native physics objects.
set(mjwarp_bias_sources "${FLASH_GPU}/cuda/dynamics/mjwarp_bias.cu"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_bias.cuh"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_spatial.cuh" "${CMAKE_CURRENT_LIST_FILE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${mjwarp_bias_sources})
set(bias_hashes)
foreach(source IN LISTS mjwarp_bias_sources)
  file(SHA256 "${source}" source_hash)
  list(APPEND bias_hashes "${source_hash}")
endforeach()
list(JOIN bias_hashes ":" bias_hash_input)
string(SHA256 bias_source_id "${bias_hash_input}")
add_library(flash_mjwarp_bias_fp32 SHARED EXCLUDE_FROM_ALL "${FLASH_GPU}/cuda/dynamics/mjwarp_bias.cu")
target_compile_definitions(flash_mjwarp_bias_fp32 PRIVATE
  FLASH_MJW_BIAS_SOURCE_ID="${bias_source_id}" FLASH_MJW_BIAS_ARCH=${FLASH_CUDA_ARCH})
target_compile_options(flash_mjwarp_bias_fp32 PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
set_target_properties(flash_mjwarp_bias_fp32 PROPERTIES PREFIX "" OUTPUT_NAME "mjwarp_bias_fp32"
  CUDA_ARCHITECTURES ${FLASH_CUDA_ARCH} UNITY_BUILD OFF
  JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
