# Optional FP32 direction product; independently compiled and device-linked.
set(mjwarp_direction_sources
  "${FLASH_GPU}/cuda/newton/mjwarp_direction.cu"
  "${FLASH_GPU}/cuda/newton/mjwarp_direction.cuh"
  "${FLASH_STEP}/instantiations/libero_models.def"
  "${FLASH_GPU}/cuda/newton/mjwarp_solver_state.cuh"
  "${CMAKE_CURRENT_LIST_FILE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${mjwarp_direction_sources})
set(direction_hashes)
foreach(source IN LISTS mjwarp_direction_sources)
  file(SHA256 "${source}" source_hash)
  list(APPEND direction_hashes "${source_hash}")
endforeach()
list(JOIN direction_hashes ":" direction_hash_input)
string(SHA256 direction_source_id "${direction_hash_input}")
add_library(flash_mjwarp_direction_fp32 SHARED EXCLUDE_FROM_ALL
  "${FLASH_GPU}/cuda/newton/mjwarp_direction.cu")
target_include_directories(flash_mjwarp_direction_fp32 PRIVATE "${FLASH_MATHDX_ROOT}/include"
  "${FLASH_MATHDX_ROOT}/external/cutlass/include")
target_compile_definitions(flash_mjwarp_direction_fp32 PRIVATE FLASH_MJW_DIRECTION_SOURCE_ID="${direction_source_id}")
target_compile_options(flash_mjwarp_direction_fp32 PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-arch=lto_${FLASH_CUDA_ARCH};-dlto;-lineinfo;--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
target_link_options(flash_mjwarp_direction_fp32 PRIVATE "$<DEVICE_LINK:-dlto;-arch=sm_${FLASH_CUDA_ARCH}>")
target_link_libraries(flash_mjwarp_direction_fp32 PRIVATE "${FLASH_SOLVERDX}")
set_target_properties(flash_mjwarp_direction_fp32 PROPERTIES
  PREFIX "" OUTPUT_NAME "mjwarp_direction_fp32" CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON CUDA_ARCHITECTURES OFF
  UNITY_BUILD OFF JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
