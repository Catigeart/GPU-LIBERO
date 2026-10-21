# Independent optional scalar transmission and generic CSR actuator velocity.
set(mjwarp_transmission_sources "${FLASH_GPU}/cuda/dynamics/mjwarp_transmission.cu"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_transmission.cuh" "${CMAKE_CURRENT_LIST_FILE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${mjwarp_transmission_sources})
set(transmission_hashes)
foreach(source IN LISTS mjwarp_transmission_sources)
  file(SHA256 "${source}" source_hash)
  list(APPEND transmission_hashes "${source_hash}")
endforeach()
list(JOIN transmission_hashes ":" transmission_hash_input)
string(SHA256 transmission_source_id "${transmission_hash_input}")
add_library(flash_mjwarp_transmission_fp32 SHARED EXCLUDE_FROM_ALL "${FLASH_GPU}/cuda/dynamics/mjwarp_transmission.cu")
target_compile_definitions(flash_mjwarp_transmission_fp32 PRIVATE
  FLASH_MJW_TRANSMISSION_SOURCE_ID="${transmission_source_id}" FLASH_MJW_TRANSMISSION_ARCH=${FLASH_CUDA_ARCH})
target_compile_options(flash_mjwarp_transmission_fp32 PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
set_target_properties(flash_mjwarp_transmission_fp32 PROPERTIES PREFIX "" OUTPUT_NAME "mjwarp_transmission_fp32"
  CUDA_ARCHITECTURES ${FLASH_CUDA_ARCH} UNITY_BUILD OFF
  JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
