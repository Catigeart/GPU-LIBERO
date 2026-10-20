# Independent FP32 leaf product: no fused-substep/Newton/vendor dependencies.
set(mjwarp_smooth_force_sources
  "${FLASH_GPU}/cuda/dynamics/mjwarp_smooth_force.cu"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_smooth_force.cuh"
  "${CMAKE_CURRENT_LIST_FILE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${mjwarp_smooth_force_sources})
set(mjwarp_smooth_force_hashes)
foreach(source IN LISTS mjwarp_smooth_force_sources)
  file(SHA256 "${source}" source_hash)
  list(APPEND mjwarp_smooth_force_hashes "${source_hash}")
endforeach()
list(JOIN mjwarp_smooth_force_hashes ":" mjwarp_smooth_force_hash_input)
string(SHA256 mjwarp_smooth_force_source_id "${mjwarp_smooth_force_hash_input}")
add_library(flash_mjwarp_smooth_force_fp32 SHARED EXCLUDE_FROM_ALL
  "${FLASH_GPU}/cuda/dynamics/mjwarp_smooth_force.cu")
target_compile_definitions(flash_mjwarp_smooth_force_fp32 PRIVATE
  FLASH_MJW_SMOOTH_FORCE_SOURCE_ID="${mjwarp_smooth_force_source_id}"
  FLASH_MJW_SMOOTH_FORCE_ARCH=${FLASH_CUDA_ARCH})
target_compile_options(flash_mjwarp_smooth_force_fp32 PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
set_target_properties(flash_mjwarp_smooth_force_fp32 PROPERTIES
  PREFIX "" OUTPUT_NAME "mjwarp_smooth_force_fp32" CUDA_SEPARABLE_COMPILATION OFF
  CUDA_ARCHITECTURES "${FLASH_CUDA_ARCH}-real" UNITY_BUILD OFF
  JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
