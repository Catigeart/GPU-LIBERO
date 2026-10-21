# Independent rigid dense constraint row construction.
set(mjwarp_constraint_build_sources "${FLASH_GPU}/cuda/dynamics/mjwarp_constraint_build.cu"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_constraint_build.cuh" "${CMAKE_CURRENT_LIST_FILE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${mjwarp_constraint_build_sources})
set(constraint_build_hashes)
foreach(source IN LISTS mjwarp_constraint_build_sources)
  file(SHA256 "${source}" source_hash)
  list(APPEND constraint_build_hashes "${source_hash}")
endforeach()
list(JOIN constraint_build_hashes ":" constraint_build_hash_input)
string(SHA256 constraint_build_source_id "${constraint_build_hash_input}")
add_library(flash_mjwarp_constraint_build_fp32 SHARED EXCLUDE_FROM_ALL "${FLASH_GPU}/cuda/dynamics/mjwarp_constraint_build.cu")
target_compile_definitions(flash_mjwarp_constraint_build_fp32 PRIVATE
  FLASH_MJW_CONSTRAINT_BUILD_SOURCE_ID="${constraint_build_source_id}" FLASH_MJW_CONSTRAINT_BUILD_ARCH=${FLASH_CUDA_ARCH})
target_compile_options(flash_mjwarp_constraint_build_fp32 PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
set_target_properties(flash_mjwarp_constraint_build_fp32 PROPERTIES PREFIX "" OUTPUT_NAME "mjwarp_constraint_build_fp32"
  CUDA_ARCHITECTURES ${FLASH_CUDA_ARCH} UNITY_BUILD OFF
  JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
