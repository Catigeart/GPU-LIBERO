# Optional postconstraint RNE suffix; bias and native physics remain independent.
set(mjwarp_postconstraint_sources "${FLASH_GPU}/cuda/dynamics/mjwarp_postconstraint.cu"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_postconstraint.cuh"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_queries.cuh"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_spatial.cuh" "${CMAKE_CURRENT_LIST_FILE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${mjwarp_postconstraint_sources})
set(postconstraint_hashes)
foreach(source IN LISTS mjwarp_postconstraint_sources)
  file(SHA256 "${source}" source_hash)
  list(APPEND postconstraint_hashes "${source_hash}")
endforeach()
list(JOIN postconstraint_hashes ":" postconstraint_hash_input)
string(SHA256 postconstraint_source_id "${postconstraint_hash_input}")
add_library(flash_mjwarp_postconstraint_fp32 SHARED EXCLUDE_FROM_ALL "${FLASH_GPU}/cuda/dynamics/mjwarp_postconstraint.cu")
target_compile_definitions(flash_mjwarp_postconstraint_fp32 PRIVATE
  FLASH_MJW_POSTCONSTRAINT_SOURCE_ID="${postconstraint_source_id}" FLASH_MJW_POSTCONSTRAINT_ARCH=${FLASH_CUDA_ARCH})
target_compile_options(flash_mjwarp_postconstraint_fp32 PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
set_target_properties(flash_mjwarp_postconstraint_fp32 PROPERTIES PREFIX "" OUTPUT_NAME "mjwarp_postconstraint_fp32"
  CUDA_ARCHITECTURES ${FLASH_CUDA_ARCH} UNITY_BUILD OFF
  JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
