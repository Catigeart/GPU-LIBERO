# Optional com_vel translation unit, independent of all native physics objects.
set(mjwarp_velocity_sources "${FLASH_GPU}/cuda/dynamics/mjwarp_velocity.cu"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_velocity.cuh"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_spatial.cuh" "${CMAKE_CURRENT_LIST_FILE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${mjwarp_velocity_sources})
set(velocity_hashes)
foreach(source IN LISTS mjwarp_velocity_sources)
  file(SHA256 "${source}" source_hash)
  list(APPEND velocity_hashes "${source_hash}")
endforeach()
list(JOIN velocity_hashes ":" velocity_hash_input)
string(SHA256 velocity_source_id "${velocity_hash_input}")
add_library(flash_mjwarp_velocity_fp32 SHARED EXCLUDE_FROM_ALL "${FLASH_GPU}/cuda/dynamics/mjwarp_velocity.cu")
target_compile_definitions(flash_mjwarp_velocity_fp32 PRIVATE
  FLASH_MJW_VELOCITY_SOURCE_ID="${velocity_source_id}" FLASH_MJW_VELOCITY_ARCH=${FLASH_CUDA_ARCH})
target_compile_options(flash_mjwarp_velocity_fp32 PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
set_target_properties(flash_mjwarp_velocity_fp32 PROPERTIES PREFIX "" OUTPUT_NAME "mjwarp_velocity_fp32"
  CUDA_ARCHITECTURES ${FLASH_CUDA_ARCH} UNITY_BUILD OFF
  JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
