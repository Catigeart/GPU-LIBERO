# Independent FP32 rigid Jacobian, force/torque sensors and gravity projection.
set(mjwarp_queries_sources "${FLASH_GPU}/cuda/dynamics/mjwarp_queries.cu"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_queries.cuh" "${FLASH_GPU}/cuda/dynamics/mjwarp_spatial.cuh" "${CMAKE_CURRENT_LIST_FILE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${mjwarp_queries_sources})
set(queries_hashes)
foreach(source IN LISTS mjwarp_queries_sources)
  file(SHA256 "${source}" source_hash)
  list(APPEND queries_hashes "${source_hash}")
endforeach()
list(JOIN queries_hashes ":" queries_hash_input)
string(SHA256 queries_source_id "${queries_hash_input}")
add_library(flash_mjwarp_queries_fp32 SHARED EXCLUDE_FROM_ALL "${FLASH_GPU}/cuda/dynamics/mjwarp_queries.cu")
target_compile_definitions(flash_mjwarp_queries_fp32 PRIVATE
  FLASH_MJW_QUERIES_SOURCE_ID="${queries_source_id}" FLASH_MJW_QUERIES_ARCH=${FLASH_CUDA_ARCH})
target_compile_options(flash_mjwarp_queries_fp32 PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
set_target_properties(flash_mjwarp_queries_fp32 PROPERTIES PREFIX "" OUTPUT_NAME "mjwarp_queries_fp32"
  CUDA_ARCHITECTURES ${FLASH_CUDA_ARCH} UNITY_BUILD OFF
  JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
