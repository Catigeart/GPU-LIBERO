# Optional dense FP32 constraint update: no dependencies on native fusion.
set(mjwarp_constraint_sources
  "${FLASH_GPU}/cuda/newton/mjwarp_constraint.cu"
  "${FLASH_GPU}/cuda/newton/mjwarp_constraint.cuh"
  "${CMAKE_CURRENT_LIST_FILE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${mjwarp_constraint_sources})
set(constraint_hashes)
foreach(source IN LISTS mjwarp_constraint_sources)
  file(SHA256 "${source}" source_hash)
  list(APPEND constraint_hashes "${source_hash}")
endforeach()
list(JOIN constraint_hashes ":" constraint_hash_input)
string(SHA256 constraint_source_id "${constraint_hash_input}")
add_library(flash_mjwarp_constraint_fp32 SHARED EXCLUDE_FROM_ALL
  "${FLASH_GPU}/cuda/newton/mjwarp_constraint.cu")
target_compile_definitions(flash_mjwarp_constraint_fp32 PRIVATE
  FLASH_MJW_CONSTRAINT_SOURCE_ID="${constraint_source_id}" FLASH_MJW_CONSTRAINT_ARCH=${FLASH_CUDA_ARCH})
target_compile_options(flash_mjwarp_constraint_fp32 PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
set_target_properties(flash_mjwarp_constraint_fp32 PROPERTIES
  PREFIX "" OUTPUT_NAME "mjwarp_constraint_fp32" CUDA_SEPARABLE_COMPILATION OFF
  CUDA_ARCHITECTURES "${FLASH_CUDA_ARCH}-real" UNITY_BUILD OFF
  JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
