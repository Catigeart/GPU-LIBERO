# Independent applied/contact external-wrench prefix of postconstraint RNE.
set(mjwarp_external_wrench_sources "${FLASH_GPU}/cuda/dynamics/mjwarp_external_wrench.cu"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_external_wrench.cuh" "${CMAKE_CURRENT_LIST_FILE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${mjwarp_external_wrench_sources})
set(external_wrench_hashes)
foreach(source IN LISTS mjwarp_external_wrench_sources)
  file(SHA256 "${source}" source_hash)
  list(APPEND external_wrench_hashes "${source_hash}")
endforeach()
list(JOIN external_wrench_hashes ":" external_wrench_hash_input)
string(SHA256 external_wrench_source_id "${external_wrench_hash_input}")
add_library(flash_mjwarp_external_wrench_fp32 SHARED EXCLUDE_FROM_ALL "${FLASH_GPU}/cuda/dynamics/mjwarp_external_wrench.cu")
target_compile_definitions(flash_mjwarp_external_wrench_fp32 PRIVATE
  FLASH_MJW_EXTERNAL_WRENCH_SOURCE_ID="${external_wrench_source_id}" FLASH_MJW_EXTERNAL_WRENCH_ARCH=${FLASH_CUDA_ARCH})
target_compile_options(flash_mjwarp_external_wrench_fp32 PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
set_target_properties(flash_mjwarp_external_wrench_fp32 PROPERTIES PREFIX "" OUTPUT_NAME "mjwarp_external_wrench_fp32"
  CUDA_ARCHITECTURES ${FLASH_CUDA_ARCH} UNITY_BUILD OFF
  JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
