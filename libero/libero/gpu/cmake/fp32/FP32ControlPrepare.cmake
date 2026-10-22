set(fp32_control_prepare_sources
  "${FLASH_GPU}/cuda/fp32/fused_control_prepare.cu"
  "${FLASH_GPU}/cuda/fp32/fused_control_prepare.cuh"
  "${FLASH_GPU}/cuda/runtime/bridge_types.cuh"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_queries.cuh"
  "${CMAKE_CURRENT_LIST_FILE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${fp32_control_prepare_sources})
set(control_prepare_hashes)
foreach(source IN LISTS fp32_control_prepare_sources)
  file(SHA256 "${source}" source_hash)
  list(APPEND control_prepare_hashes "${source_hash}")
endforeach()
list(JOIN control_prepare_hashes ":" control_prepare_hash_input)
string(SHA256 control_prepare_source_id "${control_prepare_hash_input}")
add_library(flash_fp32_fused_control_prepare SHARED EXCLUDE_FROM_ALL "${FLASH_GPU}/cuda/fp32/fused_control_prepare.cu")
target_compile_definitions(flash_fp32_fused_control_prepare PRIVATE
  FLASH_FP32_CONTROL_PREPARE_SOURCE_ID="${control_prepare_source_id}" FLASH_CUDA_ARCH=${FLASH_CUDA_ARCH})
target_compile_options(flash_fp32_fused_control_prepare PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
set_target_properties(flash_fp32_fused_control_prepare PROPERTIES PREFIX "" OUTPUT_NAME "fp32_fused_control_prepare"
  CUDA_ARCHITECTURES ${FLASH_CUDA_ARCH} UNITY_BUILD OFF
  JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
