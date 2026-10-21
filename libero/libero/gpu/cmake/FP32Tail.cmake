# Independent, readable FP32 fusion translation units. No generated bodies.
set(fp32_tail_sources
  "${FLASH_GPU}/cuda/fp32/fused_tail.cu"
  "${FLASH_GPU}/cuda/fp32/fused_tail.cuh"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_euler.cuh"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_publication.cuh"
  "${CMAKE_CURRENT_LIST_FILE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${fp32_tail_sources})
set(tail_hashes)
foreach(source IN LISTS fp32_tail_sources)
  file(SHA256 "${source}" source_hash)
  list(APPEND tail_hashes "${source_hash}")
endforeach()
list(JOIN tail_hashes ":" tail_hash_input)
string(SHA256 tail_source_id "${tail_hash_input}")
add_library(flash_fp32_fused_tail SHARED EXCLUDE_FROM_ALL "${FLASH_GPU}/cuda/fp32/fused_tail.cu")
target_compile_definitions(flash_fp32_fused_tail PRIVATE
  FLASH_FP32_TAIL_SOURCE_ID="${tail_source_id}" FLASH_CUDA_ARCH=${FLASH_CUDA_ARCH})
target_compile_options(flash_fp32_fused_tail PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
set_target_properties(flash_fp32_fused_tail PROPERTIES PREFIX "" OUTPUT_NAME "fp32_fused_tail"
  CUDA_ARCHITECTURES ${FLASH_CUDA_ARCH} UNITY_BUILD OFF
  JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
