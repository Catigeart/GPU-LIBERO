# Independent FP32 time and overflow publication.
set(mjwarp_publication_sources "${FLASH_GPU}/cuda/dynamics/mjwarp_publication.cu"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_publication.cuh" "${CMAKE_CURRENT_LIST_FILE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${mjwarp_publication_sources})
set(publication_hashes)
foreach(source IN LISTS mjwarp_publication_sources)
  file(SHA256 "${source}" source_hash)
  list(APPEND publication_hashes "${source_hash}")
endforeach()
list(JOIN publication_hashes ":" publication_hash_input)
string(SHA256 publication_source_id "${publication_hash_input}")
add_library(flash_mjwarp_publication_fp32 SHARED EXCLUDE_FROM_ALL "${FLASH_GPU}/cuda/dynamics/mjwarp_publication.cu")
target_compile_definitions(flash_mjwarp_publication_fp32 PRIVATE
  FLASH_MJW_PUBLICATION_SOURCE_ID="${publication_source_id}" FLASH_MJW_PUBLICATION_ARCH=${FLASH_CUDA_ARCH})
target_compile_options(flash_mjwarp_publication_fp32 PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
set_target_properties(flash_mjwarp_publication_fp32 PROPERTIES PREFIX "" OUTPUT_NAME "mjwarp_publication_fp32"
  CUDA_ARCHITECTURES ${FLASH_CUDA_ARCH} UNITY_BUILD OFF
  JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
