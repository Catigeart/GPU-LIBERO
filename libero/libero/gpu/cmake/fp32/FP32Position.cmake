set(fp32_position_sources
  "${FLASH_GPU}/cuda/fp32/fused_position.cu"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_com_pos.cuh"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_mass_matrix.cuh"
  "${CMAKE_CURRENT_LIST_FILE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${fp32_position_sources})
set(position_hashes)
foreach(source IN LISTS fp32_position_sources)
  file(SHA256 "${source}" source_hash)
  list(APPEND position_hashes "${source_hash}")
endforeach()
list(JOIN position_hashes ":" position_hash_input)
string(SHA256 position_source_id "${position_hash_input}")
add_library(flash_fp32_fused_position SHARED EXCLUDE_FROM_ALL "${FLASH_GPU}/cuda/fp32/fused_position.cu")
target_compile_definitions(flash_fp32_fused_position PRIVATE
  FLASH_FP32_POSITION_SOURCE_ID="${position_source_id}" FLASH_CUDA_ARCH=${FLASH_CUDA_ARCH})
target_compile_options(flash_fp32_fused_position PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
set_target_properties(flash_fp32_fused_position PROPERTIES PREFIX "" OUTPUT_NAME "fp32_fused_position"
  CUDA_ARCHITECTURES ${FLASH_CUDA_ARCH} UNITY_BUILD OFF
  JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
