# Independent six plane collision families and contact material/publication.
set(mjwarp_plane_sources "${FLASH_GPU}/cuda/collision/mjwarp_plane.cu"
  "${FLASH_GPU}/cuda/collision/mjwarp_plane.cuh" "${FLASH_GPU}/cuda/collision/mjwarp_contact.cuh" "${CMAKE_CURRENT_LIST_FILE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${mjwarp_plane_sources})
set(plane_hashes)
foreach(source IN LISTS mjwarp_plane_sources)
  file(SHA256 "${source}" source_hash)
  list(APPEND plane_hashes "${source_hash}")
endforeach()
list(JOIN plane_hashes ":" plane_hash_input)
string(SHA256 plane_source_id "${plane_hash_input}")
add_library(flash_mjwarp_plane_fp32 SHARED EXCLUDE_FROM_ALL "${FLASH_GPU}/cuda/collision/mjwarp_plane.cu")
target_compile_definitions(flash_mjwarp_plane_fp32 PRIVATE
  FLASH_MJW_PLANE_SOURCE_ID="${plane_source_id}" FLASH_MJW_PLANE_ARCH=${FLASH_CUDA_ARCH})
target_compile_options(flash_mjwarp_plane_fp32 PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
set_target_properties(flash_mjwarp_plane_fp32 PROPERTIES PREFIX "" OUTPUT_NAME "mjwarp_plane_fp32"
  CUDA_ARCHITECTURES ${FLASH_CUDA_ARCH} UNITY_BUILD OFF
  JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
