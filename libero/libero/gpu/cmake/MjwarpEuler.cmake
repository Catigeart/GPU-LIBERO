# Independent optional Euler damping preparation and state advancement.
set(mjwarp_euler_sources "${FLASH_GPU}/cuda/dynamics/mjwarp_euler.cu"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_euler.cuh" "${CMAKE_CURRENT_LIST_FILE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${mjwarp_euler_sources})
set(euler_hashes)
foreach(source IN LISTS mjwarp_euler_sources)
  file(SHA256 "${source}" source_hash)
  list(APPEND euler_hashes "${source_hash}")
endforeach()
list(JOIN euler_hashes ":" euler_hash_input)
string(SHA256 euler_source_id "${euler_hash_input}")
add_library(flash_mjwarp_euler_fp32 SHARED EXCLUDE_FROM_ALL "${FLASH_GPU}/cuda/dynamics/mjwarp_euler.cu")
target_compile_definitions(flash_mjwarp_euler_fp32 PRIVATE
  FLASH_MJW_EULER_SOURCE_ID="${euler_source_id}" FLASH_MJW_EULER_ARCH=${FLASH_CUDA_ARCH})
target_compile_options(flash_mjwarp_euler_fp32 PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
set_target_properties(flash_mjwarp_euler_fp32 PROPERTIES PREFIX "" OUTPUT_NAME "mjwarp_euler_fp32"
  CUDA_ARCHITECTURES ${FLASH_CUDA_ARCH} UNITY_BUILD OFF
  JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
