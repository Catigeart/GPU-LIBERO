# Independent optional stateless actuation, CSR projection and joint limits.
set(mjwarp_actuation_sources "${FLASH_GPU}/cuda/dynamics/mjwarp_actuation.cu"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_actuation.cuh" "${FLASH_GPU}/cuda/dynamics/mjwarp_smooth_force.cuh" "${CMAKE_CURRENT_LIST_FILE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${mjwarp_actuation_sources})
set(actuation_hashes)
foreach(source IN LISTS mjwarp_actuation_sources)
  file(SHA256 "${source}" source_hash)
  list(APPEND actuation_hashes "${source_hash}")
endforeach()
list(JOIN actuation_hashes ":" actuation_hash_input)
string(SHA256 actuation_source_id "${actuation_hash_input}")
add_library(flash_mjwarp_actuation_fp32 SHARED EXCLUDE_FROM_ALL "${FLASH_GPU}/cuda/dynamics/mjwarp_actuation.cu")
target_compile_definitions(flash_mjwarp_actuation_fp32 PRIVATE
  FLASH_MJW_ACTUATION_SOURCE_ID="${actuation_source_id}" FLASH_MJW_ACTUATION_ARCH=${FLASH_CUDA_ARCH})
target_compile_options(flash_mjwarp_actuation_fp32 PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
set_target_properties(flash_mjwarp_actuation_fp32 PROPERTIES PREFIX "" OUTPUT_NAME "mjwarp_actuation_fp32"
  CUDA_ARCHITECTURES ${FLASH_CUDA_ARCH} UNITY_BUILD OFF
  JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
