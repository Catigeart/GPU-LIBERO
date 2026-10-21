# Independent FIXED camera and light world-frame updates.
set(mjwarp_camlight_sources "${FLASH_GPU}/cuda/dynamics/mjwarp_camlight.cu"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_camlight.cuh" "${CMAKE_CURRENT_LIST_FILE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${mjwarp_camlight_sources})
set(camlight_hashes)
foreach(source IN LISTS mjwarp_camlight_sources)
  file(SHA256 "${source}" source_hash)
  list(APPEND camlight_hashes "${source_hash}")
endforeach()
list(JOIN camlight_hashes ":" camlight_hash_input)
string(SHA256 camlight_source_id "${camlight_hash_input}")
add_library(flash_mjwarp_camlight_fp32 SHARED EXCLUDE_FROM_ALL "${FLASH_GPU}/cuda/dynamics/mjwarp_camlight.cu")
target_compile_definitions(flash_mjwarp_camlight_fp32 PRIVATE
  FLASH_MJW_CAMLIGHT_SOURCE_ID="${camlight_source_id}" FLASH_MJW_CAMLIGHT_ARCH=${FLASH_CUDA_ARCH})
target_compile_options(flash_mjwarp_camlight_fp32 PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
set_target_properties(flash_mjwarp_camlight_fp32 PROPERTIES PREFIX "" OUTPUT_NAME "mjwarp_camlight_fp32"
  CUDA_ARCHITECTURES ${FLASH_CUDA_ARCH} UNITY_BUILD OFF
  JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
