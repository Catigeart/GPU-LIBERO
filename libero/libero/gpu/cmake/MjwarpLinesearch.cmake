# Optional dense FP32 Newton line search, independently compiled.
set(mjwarp_linesearch_sources
  "${FLASH_GPU}/cuda/newton/mjwarp_linesearch.cu"
  "${FLASH_GPU}/cuda/newton/mjwarp_linesearch.cuh"
  "${FLASH_GPU}/cuda/newton/mjwarp_products.cuh"
  "${CMAKE_CURRENT_LIST_FILE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${mjwarp_linesearch_sources})
set(linesearch_hashes)
foreach(source IN LISTS mjwarp_linesearch_sources)
  file(SHA256 "${source}" source_hash)
  list(APPEND linesearch_hashes "${source_hash}")
endforeach()
list(JOIN linesearch_hashes ":" linesearch_hash_input)
string(SHA256 linesearch_source_id "${linesearch_hash_input}")
add_library(flash_mjwarp_linesearch_fp32 SHARED EXCLUDE_FROM_ALL
  "${FLASH_GPU}/cuda/newton/mjwarp_linesearch.cu")
target_compile_definitions(flash_mjwarp_linesearch_fp32 PRIVATE
  FLASH_MJW_LINESEARCH_SOURCE_ID="${linesearch_source_id}" FLASH_MJW_LINESEARCH_ARCH=${FLASH_CUDA_ARCH})
target_compile_options(flash_mjwarp_linesearch_fp32 PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
set_target_properties(flash_mjwarp_linesearch_fp32 PROPERTIES
  PREFIX "" OUTPUT_NAME "mjwarp_linesearch_fp32" CUDA_SEPARABLE_COMPILATION OFF
  CUDA_ARCHITECTURES "${FLASH_CUDA_ARCH}-real" UNITY_BUILD OFF
  JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
