# Independent optional joint spring/damper and passive-force combination.
set(mjwarp_passive_sources "${FLASH_GPU}/cuda/dynamics/mjwarp_passive.cu"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_passive.cuh" "${FLASH_GPU}/cuda/dynamics/mjwarp_passive_math.cuh" "${CMAKE_CURRENT_LIST_FILE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${mjwarp_passive_sources})
set(passive_hashes)
foreach(source IN LISTS mjwarp_passive_sources)
  file(SHA256 "${source}" source_hash)
  list(APPEND passive_hashes "${source_hash}")
endforeach()
list(JOIN passive_hashes ":" passive_hash_input)
string(SHA256 passive_source_id "${passive_hash_input}")
add_library(flash_mjwarp_passive_fp32 SHARED EXCLUDE_FROM_ALL "${FLASH_GPU}/cuda/dynamics/mjwarp_passive.cu")
target_compile_definitions(flash_mjwarp_passive_fp32 PRIVATE
  FLASH_MJW_PASSIVE_SOURCE_ID="${passive_source_id}" FLASH_MJW_PASSIVE_ARCH=${FLASH_CUDA_ARCH})
target_compile_options(flash_mjwarp_passive_fp32 PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
set_target_properties(flash_mjwarp_passive_fp32 PROPERTIES PREFIX "" OUTPUT_NAME "mjwarp_passive_fp32"
  CUDA_ARCHITECTURES ${FLASH_CUDA_ARCH} UNITY_BUILD OFF
  JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
